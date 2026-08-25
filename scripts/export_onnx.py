#!/usr/bin/env python3
"""Export the age/gender checkpoint to ONNX.

Run at image build time so the container never pays the export cost and needs
no torch-to-ONNX toolchain at runtime. See app/inference/onnx_backend.py for
the measurements that justify using it.

    python scripts/export_onnx.py --out /opt/models/onnx/age_gender.onnx
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("/opt/models/onnx/age_gender.onnx"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--check", action="store_true", default=True,
                        help="Verify ONNX outputs match PyTorch after export.")
    args = parser.parse_args()

    import numpy as np
    import torch

    from app.config import Settings
    from app.inference.audeering import AudeeringBackend

    settings = Settings(**({"age_gender_model": args.model} if args.model else {}))
    print(f"[onnx] loading {settings.age_gender_model}")
    backend = AudeeringBackend(settings)
    backend.load()
    model = backend.torch_module().eval()   # public accessor, not a private reach-in

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, settings.target_sample_rate * 5)

    print(f"[onnx] exporting -> {args.out}")
    torch.onnx.export(
        model,
        (dummy,),
        str(args.out),
        input_names=["input_values"],
        output_names=["hidden", "age", "gender"],
        # The time axis must be dynamic: callers send anything from 1 s to the
        # 10 s analysis cap, and a fixed-length graph would silently reject or
        # mis-shape them.
        dynamic_axes={
            "input_values": {0: "batch", 1: "samples"},
            "hidden": {0: "batch"},
            "age": {0: "batch"},
            "gender": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"[onnx]   {args.out.stat().st_size // (1024 * 1024)} MB")

    if args.check:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        worst_age = worst_gender = 0.0
        # Check several lengths -- a dynamic axis that only works at the export
        # length is the classic ONNX footgun.
        for seconds in (1.0, 2.5, 5.0, 10.0):
            n = int(seconds * settings.target_sample_rate)
            x = rng.standard_normal((1, n)).astype(np.float32) * 0.1
            with torch.inference_mode():
                _, age_t, gender_t = model(torch.from_numpy(x))
            _, age_o, gender_o = session.run(None, {"input_values": x})
            worst_age = max(worst_age, float(np.abs(age_t.numpy() - age_o).max()))
            worst_gender = max(worst_gender, float(np.abs(gender_t.numpy() - gender_o).max()))
            print(f"[onnx]   {seconds:4.1f}s ok")

        print(f"[onnx] max |age| diff   : {worst_age:.2e}")
        print(f"[onnx] max |gender| diff: {worst_gender:.2e}")
        tolerance = 1e-3
        if worst_age > tolerance or worst_gender > tolerance:
            print(f"[onnx] FAIL: outputs diverge by more than {tolerance}")
            return 1
        print("[onnx] parity ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
