"""Compare Pascal plan compilation, execution, gradients, and saved memory.

Use --source-root with a preserved package directory to capture a baseline.
All candidates return the same chronological coordinates. CPU allocation totals
are cumulative allocations, not peak memory; saved storage covers autograd only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        nargs="+",
        default=["float32", "float64"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only the smaller batch/window count and stride one.",
    )
    args = parser.parse_args()
    if args.source_root is not None:
        sys.path.insert(0, str(args.source_root))

    import torch
    import tensorsig
    from tensorsig.offset_coded_signatures.codes import OffsetCode
    from tensorsig.offset_coded_signatures.layers import LocalFeatures
    from tensorsig.offset_coded_signatures.plans import LocalRealization

    torch.set_num_threads(1)
    device = torch.device(args.device)
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if not available[args.device]:
        parser.error(f"{args.device} is unavailable")
    root = Path(tensorsig.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    report = {
        "torch": torch.__version__,
        "source_root": str(root),
        "source_sha256": digest.hexdigest(),
        "available_devices": available,
        "oracle": "CPU float64 dense evaluation of the runtime path and code values",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": [],
    }

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def elapsed(fn):
        sync()
        start = time.perf_counter()
        result = fn()
        sync()
        return result, (time.perf_counter() - start) * 1000

    def samples(fn):
        fn()
        return [elapsed(fn)[1] for _ in range(args.repeats)]

    def error(actual, expected):
        absolute = float((actual - expected).abs().max()) if actual.numel() else 0.0
        relative = (
            absolute / max(float(expected.abs().max()), 1.0)
            if expected.numel()
            else 0.0
        )
        if not torch.isfinite(actual).all():
            raise AssertionError("nonfinite output or gradient")
        return absolute, relative

    for R, P, C, M in ((3, 2, 2, 4), (3, 2, 2, 8), (16, 2, 2, 3), (64, 2, 3, 4)):
        sizes = (
            [(1, 8)]
            if args.quick
            else ([(1, 8), (2, 16)] if M == 8 else [(1, 32), (4, 128)])
        )
        for dtype_name in args.dtype:
            dtype = getattr(torch, dtype_name)
            for orthonormal in (False, True):
                for batch, count in sizes:
                    for stride in (1,) if args.quick else (1, 2):
                        config = dict(
                            R=R,
                            P=P,
                            C=C,
                            M=M,
                            dtype=dtype_name,
                            device=str(device),
                            orthonormal=orthonormal,
                            batch=batch,
                            windows=count,
                            stride=stride,
                        )
                        code = OffsetCode.pascal(
                            R, P, orthonormal=orthonormal, dtype=dtype, device=device
                        )
                        realization = LocalRealization(code, channels=C, max_depth=M)
                        plan, compile_ms = elapsed(realization.signature)
                        direct_state, dependency_setup_ms = elapsed(
                            lambda: plan.state_count
                        )
                        dense_plan = realization.signature(coordinates="dense")
                        offsets, base = [], 0
                        for depth, words in enumerate(plan.word_indices, start=1):
                            indices = torch.zeros(len(words), dtype=torch.long)
                            for letter in words.T.cpu():
                                indices = indices * (P * C) + letter
                            offsets.append(indices + base)
                            base += (P * C) ** depth
                        selected = torch.cat(offsets).to(device)
                        generator = torch.Generator().manual_seed(417)
                        increments = torch.randn(
                            batch,
                            R + (count - 1) * stride,
                            C,
                            generator=generator,
                            dtype=torch.float64,
                        ) * (0.1 / R)
                        path = torch.cat(
                            (
                                torch.zeros(batch, 1, C, dtype=torch.float64),
                                increments.cumsum(1),
                            ),
                            dim=1,
                        ).to(device=device, dtype=dtype)
                        reference_code = OffsetCode(code.matrix.detach().cpu().double())
                        reference_plan = LocalRealization(
                            reference_code, channels=C, max_depth=M
                        ).signature(coordinates="dense")
                        dense_layer = LocalFeatures(
                            reference_plan, stride=stride, backend="vectorized"
                        )
                        reference_path = path.detach().cpu().double().requires_grad_()
                        reference = dense_layer(reference_path).index_select(
                            -1, selected.cpu()
                        )
                        cotangent = torch.randn(
                            reference.shape, generator=generator, dtype=torch.float64
                        )
                        reference_vjp = (
                            torch.autograd.grad(reference, reference_path, cotangent)[0]
                            .detach()
                            .to(device=device, dtype=dtype)
                        )
                        reference = reference.detach().to(device=device, dtype=dtype)
                        cotangent = cotangent.to(device=device, dtype=dtype)
                        for name, candidate_plan, backend in (
                            ("direct", plan, "vectorized"),
                            ("rolling", plan, "rolling"),
                            ("dense", dense_plan, "vectorized"),
                            ("dense_rolling", dense_plan, "rolling"),
                        ):
                            row = dict(
                                config,
                                candidate=name,
                                compile_ms=compile_ms,
                                dependency_setup_ms=dependency_setup_ms,
                                features=plan.feature_count,
                                direct_state=direct_state,
                            )
                            try:
                                layer, row["layer_setup_ms"] = elapsed(
                                    lambda: LocalFeatures(
                                        candidate_plan, stride=stride, backend=backend
                                    )
                                )

                                def forward(p):
                                    value = layer(p)
                                    return (
                                        value.index_select(-1, selected)
                                        if candidate_plan is dense_plan
                                        else value
                                    )

                                def inference():
                                    with torch.no_grad():
                                        return forward(path)

                                def training():
                                    p = path.detach().requires_grad_()
                                    value = forward(p)
                                    return torch.autograd.grad(value, p, cotangent)[0]

                                actual, row["first_forward_ms"] = elapsed(inference)
                                row["value_abs_error"], row["value_scaled_error"] = (
                                    error(actual, reference)
                                )
                                vjp = training()
                                row["vjp_abs_error"], row["vjp_scaled_error"] = error(
                                    vjp, reference_vjp
                                )
                                tolerance = 1e-8 if dtype == torch.float64 else 2e-3
                                if (
                                    max(
                                        row["value_scaled_error"],
                                        row["vjp_scaled_error"],
                                    )
                                    > tolerance
                                ):
                                    raise AssertionError(
                                        "value/VJP verification failed"
                                    )
                                row["forward_samples_ms"] = samples(inference)
                                row["training_samples_ms"] = samples(training)
                                row["forward_ms"] = statistics.median(
                                    row["forward_samples_ms"]
                                )
                                row["training_ms"] = statistics.median(
                                    row["training_samples_ms"]
                                )
                                saved, storage = [], {}

                                def pack(tensor):
                                    saved.append(tensor.numel() * tensor.element_size())
                                    storage[
                                        (
                                            str(tensor.device),
                                            tensor.untyped_storage().data_ptr(),
                                        )
                                    ] = tensor.untyped_storage().nbytes()
                                    return tensor

                                with torch.autograd.graph.saved_tensors_hooks(
                                    pack, lambda tensor: tensor
                                ):
                                    training()
                                row["saved_logical_bytes"] = sum(saved)
                                row["saved_unique_storage_bytes"] = sum(
                                    storage.values()
                                )
                                if args.profile:
                                    with torch.profiler.profile(
                                        activities=[
                                            torch.profiler.ProfilerActivity.CPU
                                        ],
                                        profile_memory=True,
                                    ) as profiler:
                                        training()
                                    row["cpu_positive_allocation_bytes"] = sum(
                                        max(0, event.self_cpu_memory_usage)
                                        for event in profiler.events()
                                    )
                                if name == "rolling" and hasattr(
                                    plan, "_rolling_program"
                                ):
                                    program = plan._rolling_program()
                                    row["rolling_state_per_lane"] = program.state_count
                                    row["rolling_lanes"] = (
                                        count + max(1, R // stride) - 1
                                    ) // max(1, R // stride)
                                row["status"] = "passed"
                            except (
                                TypeError,
                                ValueError,
                                RuntimeError,
                                AssertionError,
                                NotImplementedError,
                            ) as exc:
                                row["status"] = (
                                    "unavailable"
                                    if name == "rolling"
                                    and "requires dense" in str(exc)
                                    else "failed"
                                )
                                row["error"] = str(exc)
                            report["results"].append(row)
                            print(
                                f"{config} {name}: {row['status']} {row.get('forward_ms', '-')}",
                                flush=True,
                            )
                            args.output.parent.mkdir(parents=True, exist_ok=True)
                            args.output.write_text(json.dumps(report, indent=2) + "\n")
                            gc.collect()
    if any(row["status"] == "failed" for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
