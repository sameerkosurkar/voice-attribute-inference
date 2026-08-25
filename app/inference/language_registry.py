"""Registry for language-identification backends.

Deliberately separate from the attribute-backend registry rather than a shared
generic one. The two have different contracts (`predict` vs `identify`) and
different failure semantics -- an attribute backend failing is fatal, a language
backend failing is a null field -- and collapsing them behind one generic
registry would mean the type of the thing you get back depends on a string,
which is how you end up with runtime AttributeErrors instead of a clear seam.

Two small explicit registries beat one clever generic one here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    name: str
    factory: Callable
    description: str


_REGISTRY: dict[str, LanguageSpec] = {}
_BUILTIN_MODULES = ("app.inference.language",)
_builtins_loaded = False


def register_language_backend(name: str, *, description: str = ""):
    def decorate(cls):
        key = name.strip().lower()
        existing = _REGISTRY.get(key)
        if existing is not None and existing.factory is not cls:
            raise ValueError(f"language backend {key!r} is already registered")
        _REGISTRY[key] = LanguageSpec(key, cls, description)
        return cls

    return decorate


def load_builtins() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    import importlib

    for module in _BUILTIN_MODULES:
        try:
            importlib.import_module(module)
        except Exception:  # pragma: no cover
            log.warning("language_backend_import_failed", module=module, exc_info=True)
    _builtins_loaded = True


def available_language_backends() -> list[str]:
    load_builtins()
    return sorted(_REGISTRY)


def create(name: str, settings) -> object:
    load_builtins()
    spec = _REGISTRY.get(name.strip().lower())
    if spec is None:
        raise KeyError(
            f"unknown language backend {name!r}. Registered: "
            f"{', '.join(available_language_backends())}"
        )
    return spec.factory(settings)
