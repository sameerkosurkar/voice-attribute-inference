"""Backend registry.

WHY A REGISTRY AND NOT AN IF/ELIF FACTORY.

Before this, adding a model meant editing three files: widen a
`Literal[...]` in config.py, add a branch to `build_backend()` in service.py,
and write the implementation. Two of those three are core files that have
nothing to do with the new model -- classic shotgun surgery, and exactly the
kind of friction that stops people swapping the model at all.

That matters here more than usual, because the default checkpoint is
**CC-BY-NC-SA licensed and cannot ship commercially**. A replacement is not a
hypothetical future nicety, it is on the critical path to using this in
production. So the swap has to be genuinely cheap.

With the registry, a new backend is ONE self-contained file:

    # app/inference/my_backend.py
    from app.inference.registry import register_backend
    from app.inference.types import RawPrediction

    @register_backend("my-model", description="ECAPA embeddings + trained head")
    class MyBackend:
        name = "my-model"

        def __init__(self, settings): ...
        def load(self): ...
        def warmup(self): ...
        @property
        def ready(self) -> bool: ...
        def predict(self, samples, sample_rate) -> RawPrediction: ...

then `VA_BACKEND=my-model`. No edit to config.py, service.py, or any sibling.

Registration is by name rather than by scanning the package, so importing a
module is what makes a backend available -- explicit, greppable, and it keeps
optional heavyweight imports (torch, onnxruntime) out of the process until the
backend is actually selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BackendSpec:
    name: str
    factory: Callable
    description: str
    # Optional predicate: "can this backend actually run here?" Used by `auto`
    # to skip a backend whose artefacts are missing (e.g. no ONNX export)
    # without importing or constructing it.
    is_available: Callable[..., bool] | None = None
    # Higher wins when VA_BACKEND=auto. Lets a backend express "prefer me here"
    # without the selector needing to know anything about it.
    auto_priority: Callable[..., int] | None = None


_REGISTRY: dict[str, BackendSpec] = {}

# Modules that self-register on import. Importing them is cheap -- the heavy
# dependencies live inside load(), not at module scope.
_BUILTIN_MODULES = (
    "app.inference.mock",
    "app.inference.audeering",
    "app.inference.onnx_backend",
)
_builtins_loaded = False


def register_backend(
    name: str,
    *,
    description: str = "",
    is_available: Callable[..., bool] | None = None,
    auto_priority: Callable[..., int] | None = None,
):
    """Class decorator that makes a backend selectable by `VA_BACKEND=<name>`."""

    def decorate(cls):
        key = name.strip().lower()
        existing = _REGISTRY.get(key)
        if existing is not None and existing.factory is not cls:
            # Re-registering the same class is fine (module reimport); a
            # different class silently shadowing another is not.
            raise ValueError(
                f"backend {key!r} is already registered by "
                f"{existing.factory!r}; pick a different name"
            )
        _REGISTRY[key] = BackendSpec(
            name=key,
            factory=cls,
            description=description or (cls.__doc__ or "").strip().splitlines()[0:1] and
                        (cls.__doc__ or "").strip().splitlines()[0] or "",
            is_available=is_available,
            auto_priority=auto_priority,
        )
        return cls

    return decorate


def load_builtins() -> None:
    """Import the shipped backends so they register themselves."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    import importlib

    for module in _BUILTIN_MODULES:
        try:
            importlib.import_module(module)
        except Exception:  # pragma: no cover - a broken optional backend
            log.warning("backend_module_import_failed", module=module, exc_info=True)
    _builtins_loaded = True


def available_backends() -> list[str]:
    load_builtins()
    return sorted(_REGISTRY)


def specs() -> Iterable[BackendSpec]:
    load_builtins()
    return tuple(_REGISTRY.values())


def get_spec(name: str) -> BackendSpec | None:
    load_builtins()
    return _REGISTRY.get(name.strip().lower())


def create(name: str, settings) -> object:
    """Instantiate a registered backend by name."""
    spec = get_spec(name)
    if spec is None:
        raise KeyError(
            f"unknown backend {name!r}. Registered: {', '.join(available_backends())}"
        )
    return spec.factory(settings)


def select_auto(settings) -> str:
    """Pick the best available backend name for this host.

    Each backend advertises its own availability and priority, so this function
    contains no knowledge of any specific implementation -- adding a backend that
    wants to win on some platform requires no change here.
    """
    load_builtins()
    candidates: list[tuple[int, str]] = []
    for spec in _REGISTRY.values():
        if spec.auto_priority is None:
            continue  # not an auto candidate (e.g. the mock backend)
        if spec.is_available is not None and not spec.is_available(settings):
            continue
        candidates.append((spec.auto_priority(settings), spec.name))

    if not candidates:
        raise KeyError(
            "no backend is available for auto-selection; "
            f"registered: {', '.join(available_backends())}"
        )
    candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    return candidates[0][1]
