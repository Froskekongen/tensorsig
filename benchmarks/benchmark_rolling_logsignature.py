"""Verified runtime comparison: python benchmarks/benchmark_rolling_logsignature.py."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
import platform
import random
import resource
from pathlib import Path
import statistics
import sys
import time

import torch

from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    compile_lie_quotient,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import ProjectedAlphabet
from tensorsig.offset_coded_signatures.quotients import (
    DependenceGraph,
    LieQuotient,
    TemporalDegree,
)
from tensorsig.offset_coded_signatures.window_logsignature import WindowLogSignature
from tensorsig.offset_coded_signatures.rolling_logsignature import RollingLogSignature

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._rolling_profile import profile_call


BASELINES = {
    "direct",
    "pbw",
    "tensor",
    "direct_batched",
    "pbw_batched",
    "tensor_batched",
    "tensor_sized",
}


def _tensor_sized_chunk_windows(plan, window, batch, windows, derivative_order):
    """Benchmark-only intermediate chunks; execution uses the public tensor API."""
    target = 2**26 if derivative_order == 2 else 2**27
    per_window = max(1, batch * window * plan.alphabet.width**plan.max_depth)
    return min(windows, max(1, target // per_window))


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [
        *(root / "src/tensorsig").rglob("*.py"),
        Path(__file__).resolve(),
        root / "benchmarks/_rolling_profile.py",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def load_oracle_cache(directory, metadata, shapes):
    """Reuse only an exact, validated independent CPU oracle checkpoint."""
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = Path(directory) / f"{key}.pt" if directory else None
    tensors = {}
    record = {}
    if path is not None and path.exists():
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("metadata") != metadata:
            raise ValueError("oracle cache metadata mismatch")
        tensors = record.get("tensors", {})
        if set(tensors) not in (
            {"value", "vjp"},
            {"value", "vjp", "loss_gradient", "hvp"},
        ):
            raise ValueError("oracle cache fields are incomplete or unknown")
        for name, tensor in tensors.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != tuple(shapes[name])
                or tensor.dtype != torch.float64
                or tensor.requires_grad
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"oracle cache tensor {name!r} is invalid")
    provenance = {
        "key": key,
        "path": str(path) if path else None,
        "reused_fields": sorted(tensors),
        "written_fields": [],
    }
    if "completed_hvp_compute_seconds" in record:
        seconds = record["completed_hvp_compute_seconds"]
        if (
            "hvp" not in tensors
            or type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("oracle cache HVP timing provenance is invalid")
        provenance["hvp_progress"] = {
            "resumed_from_window": shapes["value"][1],
            "resumed_compute_seconds": seconds,
            "completed_windows": shapes["value"][1],
            "completed_chunk_compute_seconds": seconds,
            "new_completed_chunk_compute_seconds": 0.0,
            "timing_scope": "completed oracle chunks only; excludes interrupted work and checkpoint I/O; complete cache reused",
        }
    return path, tensors, provenance


def save_oracle_cache(path, metadata, tensors, provenance):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    record = {"metadata": metadata, "tensors": tensors}
    if "hvp" in tensors and "hvp_progress" in provenance:
        record["completed_hvp_compute_seconds"] = provenance["hvp_progress"][
            "completed_chunk_compute_seconds"
        ]
    torch.save(record, temporary)
    temporary.replace(path)
    provenance["written_fields"] = sorted(tensors)


def time_call(call, *, sync, repeats):
    """Use the same synchronized >=0.1s warm interval for every candidate."""
    warmup_started = time.monotonic()
    warmup_calls = 0
    while True:
        call()
        sync()
        warmup_calls += 1
        warmup_ms = (time.monotonic() - warmup_started) * 1000
        if warmup_ms >= 100:
            break
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "warmup_calls": warmup_calls,
        "warmup_ms": warmup_ms,
    }


def window_scaled_errors(actual, expected, *, atol, rtol):
    """Maximum componentwise scaled error per output window, across batches."""
    if not actual.shape[-1]:
        return [0.0] * actual.shape[1]
    scaled = ((actual - expected).abs() / (atol + rtol * expected.abs())).amax(
        dim=(0, 2)
    )
    return [value if math.isfinite(value) else None for value in scaled.tolist()]


def run_phase(entry, mode, phase, function, checkpoint):
    """Keep earlier mode evidence even when a later derivative/kernel fails."""
    mode = str(mode)
    entry["active_phase"] = {"mode": mode, "phase": phase}
    checkpoint()
    started = time.perf_counter()
    try:
        result = function()
    except Exception as exc:
        entry.setdefault("mode_failures", {}).setdefault(mode, {})[phase] = (
            f"{type(exc).__name__}: {exc}"
        )
        success, result = False, None
    else:
        success = True
    entry.setdefault("phase_seconds", {}).setdefault(mode, {})[phase] = (
        time.perf_counter() - started
    )
    entry.pop("active_phase", None)
    checkpoint()
    return success, result


def error_metrics(actual, expected, *, atol, rtol):
    """Componentwise errors; nonfinite results can never pass verification."""
    difference = (actual - expected).abs()
    scaled = difference / (atol + rtol * expected.abs())
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {
        "max_abs": difference.max().item() if difference.numel() else 0.0,
        "max_scaled": scaled.max().item() if scaled.numel() else 0.0,
        "rms_relative": (difference / (atol + expected.abs()))
        .square()
        .mean()
        .sqrt()
        .item()
        if difference.numel()
        else 0.0,
        "passed": finite and (not scaled.numel() or bool((scaled <= 1).all())),
    }


def make_increments(shape, *, seed, scale, family="gaussian"):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(shape, generator=generator, dtype=torch.float64)
    if family == "correlated":
        for t in range(1, shape[1]):
            values[:, t] = 0.8 * values[:, t - 1] + 0.6 * values[:, t]
    elif family == "constant":
        values = torch.ones(shape, dtype=torch.float64)
    elif family == "alternating":
        values = torch.ones(shape, dtype=torch.float64)
        values[:, 1::2] = -1
    elif family == "impulse":
        values = torch.zeros(shape, dtype=torch.float64)
        values[:, shape[1] // 2] = 1
    elif family == "zero":
        values = torch.zeros(shape, dtype=torch.float64)
    elif family != "gaussian":
        raise ValueError("unknown input family")
    return values * scale


def value_gradient_hvp(evaluate, inputs, weights, direction):
    """Nonlinear output loss exercises both Jacobian and second derivatives."""
    value = evaluate(inputs)
    loss = (value * weights + value.square() / 2).sum()
    # The zero anchor handles constant/empty quotients without weakening checks.
    gradient = torch.autograd.grad(
        loss + inputs.square().sum() * 0, inputs, create_graph=True
    )[0]
    hvp = torch.autograd.grad((gradient * direction).sum(), inputs)[0]
    return value, gradient, hvp


def reference_hvp_in_chunks(
    compiled,
    increments,
    weights,
    direction,
    *,
    stride,
    chunk_windows,
    partial_path=None,
    metadata=None,
    progress=None,
    report_checkpoint=None,
):
    """Accumulate complete window chunks, optionally resuming an exact CPU prefix."""
    count = weights.shape[1]
    layout = dict(
        window=compiled.window,
        stride=stride,
        chunk_windows=chunk_windows,
        windows=count,
    )
    gradient, hvp = torch.zeros_like(increments), torch.zeros_like(increments)
    next_window, compute_seconds = 0, 0.0
    if partial_path is not None:
        partial_path = Path(partial_path)
        if increments.device.type != "cpu" or increments.dtype != torch.float64:
            raise ValueError("partial HVP checkpoints require CPU float64 inputs")
        if partial_path.exists():
            record = torch.load(partial_path, map_location="cpu", weights_only=True)
            if not isinstance(record, dict):
                raise ValueError("partial HVP checkpoint record is invalid")
            if record.get("metadata") != metadata or record.get("layout") != layout:
                raise ValueError("partial HVP checkpoint metadata mismatch")
            next_window = record.get("next_window")
            compute_seconds = record.get("completed_chunk_compute_seconds")
            if (
                type(next_window) is not int
                or not 0 <= next_window <= count
                or (next_window != count and next_window % chunk_windows)
                or type(compute_seconds) not in (int, float)
                or not math.isfinite(compute_seconds)
                or compute_seconds < 0
            ):
                raise ValueError("partial HVP checkpoint progress is invalid")
            stop = (next_window - 1) * stride + compiled.window if next_window else 0
            for key in ("gradient", "hvp"):
                tensor = record.get(key)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != increments.shape
                    or tensor.dtype != torch.float64
                    or tensor.layout != torch.strided
                    or tensor.requires_grad
                    or not bool(torch.isfinite(tensor).all())
                    or bool(torch.count_nonzero(tensor[:, stop:]))
                ):
                    raise ValueError(f"partial HVP checkpoint {key} is invalid")
            gradient, hvp = record["gradient"], record["hvp"]
    resumed_window, resumed_seconds = next_window, compute_seconds
    progress = {} if progress is None else progress

    def record_progress():
        progress.update(
            partial_path=str(partial_path) if partial_path is not None else None,
            resumed_from_window=resumed_window,
            resumed_compute_seconds=resumed_seconds,
            completed_windows=next_window,
            completed_chunk_compute_seconds=compute_seconds,
            new_completed_chunk_compute_seconds=compute_seconds - resumed_seconds,
            timing_scope="completed oracle chunks only; excludes interrupted work and checkpoint I/O; process watchdog retains whole-attempt wall time",
        )
        if report_checkpoint is not None:
            report_checkpoint()

    record_progress()
    for first in range(next_window, count, chunk_windows):
        started = time.perf_counter()
        last = min(first + chunk_windows, count)
        start, stop = first * stride, (last - 1) * stride + compiled.window
        piece = increments[:, start:stop].detach().requires_grad_()
        _, grad, hess = value_gradient_hvp(
            lambda x: compiled.reference(x, stride=stride, method="tensor"),
            piece,
            weights[:, first:last],
            direction[:, start:stop],
        )
        gradient[:, start:stop] += grad.detach()
        hvp[:, start:stop] += hess.detach()
        compute_seconds += time.perf_counter() - started
        next_window = last
        if partial_path is not None:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = partial_path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(
                dict(
                    metadata=metadata,
                    layout=layout,
                    next_window=next_window,
                    completed_chunk_compute_seconds=compute_seconds,
                    gradient=gradient,
                    hvp=hvp,
                ),
                temporary,
            )
            temporary.replace(partial_path)
        record_progress()
    return gradient, hvp


def reference_value_and_vjp(compiled, increments, weights, *, stride, chunk_windows):
    """Bound the independent oracle tape; add VJPs where windows overlap."""
    count = weights.shape[1]
    values = []
    gradient = torch.zeros_like(increments)
    for first in range(0, count, chunk_windows):
        last = min(first + chunk_windows, count)
        start = first * stride
        stop = (last - 1) * stride + compiled.window
        piece = increments[:, start:stop].detach().requires_grad_()
        value = compiled.reference(piece, stride=stride, method="tensor")
        piece_gradient = torch.autograd.grad(value, piece, weights[:, first:last])[0]
        gradient[:, start:stop] += piece_gradient.detach()
        values.append(value.detach())
    return torch.cat(values, dim=1), gradient


def make_candidates(compiled, args):
    """The same named runtime calls for timing and isolated memory measurements."""
    public = WindowLogSignature(compiled.plan, compiled.window)
    derivative_order = 2 if args.hvp else 1
    candidates = {
        method: (
            lambda x, m=method: public(
                x, algorithm=m, stride=args.stride, derivative_order=derivative_order
            )
        )
        for method in ("direct", "pbw", "tensor", "active")
    }
    for name in ("auto", "tensor_sized"):
        candidates[name] = public
    if args.baseline_chunk_windows is not None:
        for method in ("direct", "pbw", "tensor", "active"):
            candidates[f"{method}_batched"] = lambda x, m=method: public(
                x,
                stride=args.stride,
                algorithm=m,
                derivative_order=derivative_order,
                chunk_windows=args.baseline_chunk_windows,
            )
    for shift in ("dense", "sparse"):
        candidates[f"typed_{shift}"] = lambda x, s=shift: compiled(
            x,
            stride=args.stride,
            backend="typed",
            shift_backend=s,
            refresh_every=args.refresh_every,
        )
        for chunk in args.chunks:
            candidates[f"typed_{shift}_chunk_{chunk}"] = lambda x, c=chunk, s=shift: (
                compiled(
                    x,
                    stride=args.stride,
                    chunk_windows=c,
                    backend="typed",
                    shift_backend=s,
                    refresh_every=args.refresh_every,
                )
            )
    if args.fused:
        for chunk in (None, *args.chunks):
            name = "fused" if chunk is None else f"fused_chunk_{chunk}"
            candidates[name] = lambda x, c=chunk: compiled(
                x,
                backend="fused",
                chunk_windows=c,
                stride=args.stride,
                refresh_every=args.refresh_every,
            )
    for chunk in args.chunks:
        for backend in ("joint", "joint_fused"):
            candidates[f"{backend}_chunk_{chunk}"] = lambda x, c=chunk, b=backend: (
                compiled(
                    x,
                    algorithm="chunks",
                    backend=b,
                    chunk_windows=c,
                    derivative_order=2 if args.hvp else 1,
                    stride=args.stride,
                    refresh_every=args.refresh_every,
                )
            )
            candidates[f"scan_{backend}_chunk_{chunk}"] = lambda x, c=chunk, b=backend: (
                compiled.scan(
                    x,
                    backend=b,
                    chunk_windows=c,
                    derivative_order=2 if args.hvp else 1,
                    stride=args.stride,
                    refresh_every=args.refresh_every,
                )
            )
    for name, algorithm, backend in (
        ("scan_joint", "scan", "joint"),
        ("scan_joint_fused", "scan", "joint_fused"),
    ):
        candidates[name] = lambda x, a=algorithm, b=backend: compiled(
            x,
            algorithm=a,
            backend=b,
            derivative_order=2 if args.hvp else 1,
            stride=args.stride,
            refresh_every=0
            if a == "scan" and args.refresh_every is None
            else args.refresh_every,
        )
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--depth", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--code-dim", type=int, default=2)
    parser.add_argument("--temporal-degree", type=int, default=1)
    parser.add_argument(
        "--graph", action="store_true", help="Use a path dependence graph."
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--chunks", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--family",
        choices=[
            "gaussian",
            "correlated",
            "constant",
            "alternating",
            "impulse",
            "zero",
        ],
        default="gaussian",
    )
    parser.add_argument(
        "--hvp",
        action="store_true",
        help="Verify and time nonlinear-loss second derivatives.",
    )
    parser.add_argument(
        "--randomize", action="store_true", help="Seeded candidate ordering."
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Gaussian increment standard deviation; default: 0.1/window**(code_dim-1).",
    )
    parser.add_argument(
        "--refresh-every",
        type=int,
        help="Override every recurrence's refresh interval; zero disables refresh. "
        "Default preserves each named candidate's policy.",
    )
    parser.add_argument(
        "--oracle-chunk-windows",
        type=int,
        default=32,
        help="CPU float64 tensor oracle windows per value/VJP chunk (default: 32).",
    )
    parser.add_argument(
        "--oracle-cache",
        help="Directory for exact source/runtime/input-keyed CPU oracle checkpoints.",
    )
    parser.add_argument(
        "--max-tensor-elements",
        type=int,
        help="Optional limit on batch*windows*window*(channels*code_dim)**depth. "
        "Skip over-budget tensor baselines; fail if an oracle chunk exceeds it. "
        "This guards one ambient tensor level, not peak memory or bridge setup.",
    )
    parser.add_argument(
        "--baseline-chunk-windows",
        type=int,
        help="Also compare direct_batched/pbw_batched/tensor_batched using this many "
        "independent windows per call. Bounds forward intermediates, not the full "
        "training tape retained by autograd.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--backward", action="store_true", help="Also time forward plus VJP."
    )
    parser.add_argument("--output", help="Write the complete JSON report to this file.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Measure allocator samples and saved tensors without operator tracing.",
    )
    parser.add_argument(
        "--fused",
        action="store_true",
        help="Include fullgraph torch.compile endpoint kernels.",
    )
    parser.add_argument(
        "--only", nargs="+", help="Restrict candidate names for focused runs."
    )
    args = parser.parse_args()
    for name in (
        "windows",
        "batch",
        "channels",
        "code_dim",
        "stride",
        "repeats",
        "threads",
        "oracle_chunk_windows",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if any(value < 1 for value in (*args.window, *args.depth, *args.chunks)):
        parser.error("--window, --depth, and --chunks must be positive")
    if args.scale is not None and (not math.isfinite(args.scale) or args.scale <= 0):
        parser.error("--scale must be finite and positive")
    if args.refresh_every is not None and args.refresh_every < 0:
        parser.error("--refresh-every must be nonnegative")
    if args.max_tensor_elements is not None and args.max_tensor_elements < 1:
        parser.error("--max-tensor-elements must be positive")
    if args.baseline_chunk_windows is not None and args.baseline_chunk_windows < 1:
        parser.error("--baseline-chunk-windows must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device("mps" if args.device == "metal" else args.device)
    dtype = getattr(torch, args.dtype)
    atol, rtol = (1e-9, 1e-8) if dtype == torch.float64 else (3e-5, 3e-4)
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Metal promotion measurements require CPU fallback disabled")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    report = {
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "source_sha256": source_hashes(),
        "tolerances": {"atol": atol, "rtol": rtol},
        "arguments": vars(args),
        "cases": [],
        "complete": False,
        "verification_contract": "verified means inference correctness; use verified_modes and timed_modes for each derivative order",
    }

    def checkpoint():
        if args.output:
            target = Path(args.output)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(target)

    checkpoint()
    for window in args.window:
        for depth in args.depth:
            elements_per_window = (
                args.batch * window * (args.channels * args.code_dim) ** depth
            )
            oracle_elements = elements_per_window * min(
                args.windows, args.oracle_chunk_windows
            )
            if (
                args.max_tensor_elements is not None
                and oracle_elements > args.max_tensor_elements
            ):
                parser.error(
                    f"tensor oracle chunk needs {oracle_elements} ambient-level elements; "
                    "reduce --oracle-chunk-windows or the workload, or increase "
                    "--max-tensor-elements"
                )
            rules = [TemporalDegree(args.temporal_degree)]
            if args.graph:
                rules.append(
                    DependenceGraph(tuple((i, i + 1) for i in range(args.channels - 1)))
                )
            started = time.perf_counter()
            plan = compile_lie_quotient(
                LieQuotient(*rules),
                ProjectedAlphabet.newton(args.code_dim, args.channels),
                depth,
            ).to(device=device, dtype=dtype)
            sync()
            quotient_compile_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            compiled = RollingLogSignature(plan, window)
            sync()
            rolling_compile_ms = (time.perf_counter() - started) * 1000
            generator = torch.Generator().manual_seed(args.seed)
            scale = (
                0.1 / window ** (args.code_dim - 1)
                if args.scale is None
                else args.scale
            )
            dx = (
                make_increments(
                    (
                        args.batch,
                        window + (args.windows - 1) * args.stride,
                        args.channels,
                    ),
                    seed=args.seed,
                    scale=scale,
                    family=args.family,
                )
                .to(device=device, dtype=dtype)
                .requires_grad_()
            )
            candidates = make_candidates(compiled, args)
            if args.only:
                unknown = set(args.only) - candidates.keys()
                if unknown:
                    parser.error(
                        f"--only contains unknown candidates: {', '.join(sorted(unknown))}"
                    )
                candidates = {k: v for k, v in candidates.items() if k in args.only}
            if args.randomize:
                names = list(candidates)
                random.Random(args.seed).shuffle(names)
                candidates = {name: candidates[name] for name in names}
            # A CPU float64 oracle also works when a production baseline cannot
            # run on the target device. It is never included in device timings.
            oracle_plan = compile_lie_quotient(
                LieQuotient(*rules),
                ProjectedAlphabet.newton(args.code_dim, args.channels),
                depth,
            )
            oracle = RollingLogSignature(oracle_plan, window)
            oracle_dx = dx.detach().cpu().double()
            weights = torch.randn(
                (args.batch, args.windows, oracle_plan.dimension),
                generator=generator,
                dtype=torch.float64,
            )
            device_weights = weights.to(device=device, dtype=dtype)
            weights = device_weights.cpu().double()
            direction = torch.randn(
                oracle_dx.shape, generator=generator, dtype=torch.float64
            )
            device_direction = direction.to(device=device, dtype=dtype)
            direction = device_direction.cpu().double()
            oracle_metadata = {
                "version": 1,
                "source_sha256": report["source_sha256"],
                "torch": report["torch"],
                "python": report["python"],
                "window": window,
                "depth": depth,
                **{
                    name: getattr(args, name)
                    for name in (
                        "windows",
                        "batch",
                        "channels",
                        "code_dim",
                        "temporal_degree",
                        "graph",
                        "stride",
                        "seed",
                        "family",
                        "dtype",
                        "oracle_chunk_windows",
                    )
                },
                "scale": scale,
                "random_protocol": "cpu-float64-randn-weights-then-direction-target-dtype-v1",
                "input_sha256": {
                    name: hashlib.sha256(
                        value.contiguous().numpy().tobytes()
                    ).hexdigest()
                    for name, value in (
                        ("increments", oracle_dx),
                        ("weights", weights),
                        ("direction", direction),
                    )
                },
            }
            cache_path, oracle_tensors, cache_provenance = load_oracle_cache(
                args.oracle_cache,
                oracle_metadata,
                {
                    "value": weights.shape,
                    "vjp": oracle_dx.shape,
                    "loss_gradient": oracle_dx.shape,
                    "hvp": oracle_dx.shape,
                },
            )
            if "value" not in oracle_tensors:
                expected, gradient = reference_value_and_vjp(
                    oracle,
                    oracle_dx,
                    weights,
                    stride=args.stride,
                    chunk_windows=args.oracle_chunk_windows,
                )
                oracle_tensors.update(value=expected, vjp=gradient)
                save_oracle_cache(
                    cache_path, oracle_metadata, oracle_tensors, cache_provenance
                )
            expected, gradient = oracle_tensors["value"], oracle_tensors["vjp"]
            expected_loss_gradient = oracle_tensors.get("loss_gradient")
            expected_hvp = oracle_tensors.get("hvp")

            def compute_oracle_hvp():
                partial_path = (
                    cache_path.with_suffix(".hvp-partial.pt")
                    if cache_path is not None
                    else None
                )
                loss_gradient, hvp = reference_hvp_in_chunks(
                    oracle,
                    oracle_dx,
                    weights,
                    direction,
                    stride=args.stride,
                    chunk_windows=args.oracle_chunk_windows,
                    partial_path=partial_path,
                    metadata=oracle_metadata,
                    progress=cache_provenance.setdefault("hvp_progress", {}),
                    report_checkpoint=checkpoint,
                )
                oracle_tensors.update(loss_gradient=loss_gradient, hvp=hvp)
                save_oracle_cache(
                    cache_path, oracle_metadata, oracle_tensors, cache_provenance
                )
                # Keep a complete partial checkpoint until the full cache is atomically promoted.
                if partial_path is not None:
                    partial_path.unlink(missing_ok=True)
                return loss_gradient, hvp

            case = {
                "window": window,
                "depth": depth,
                "dimension": plan.dimension,
                "bracket_rows": len(plan.bracket_indices),
                "typed_rows_per_append": compiled.append.row_count,
                "shift_nnz": len(compiled.transport.values),
                "increment_scale": scale,
                "increments": dx.shape[1],
                "oracle": "CPU float64 tensor independent windows",
                "oracle_cache": cache_provenance,
                "oracle_chunk_windows": args.oracle_chunk_windows,
                "oracle_ambient_level_elements": oracle_elements,
                "tensor_baseline_ambient_level_elements": elements_per_window
                * args.windows,
                "tensor_batched_ambient_level_elements": elements_per_window
                * min(args.windows, args.baseline_chunk_windows)
                if args.baseline_chunk_windows is not None
                else None,
                "quotient_compile_ms": quotient_compile_ms,
                "rolling_compile_ms": rolling_compile_ms,
                "candidates": {},
            }
            report["cases"].append(case)
            checkpoint()
            for name, evaluate in candidates.items():
                tensor_windows = (
                    min(args.windows, args.baseline_chunk_windows)
                    if name == "tensor_batched"
                    else _tensor_sized_chunk_windows(
                        compiled.plan, window, args.batch, args.windows, 0
                    )
                    if name == "tensor_sized"
                    else args.windows
                )
                if (
                    name in {"tensor", "tensor_batched", "tensor_sized"}
                    and args.max_tensor_elements is not None
                    and elements_per_window * tensor_windows > args.max_tensor_elements
                ):
                    case["candidates"][name] = {
                        "verified": False,
                        "skipped": "ambient tensor level exceeds --max-tensor-elements; "
                        "oracle remains verified in smaller window chunks",
                    }
                    continue
                entry = {
                    "verified": False,
                    "verified_modes": {},
                    "timed_modes": {},
                    "mode_failures": {},
                    "profiles": {},
                }
                case["candidates"][name] = entry

                def evaluate_mode(inputs, mode):
                    if name not in {"auto", "tensor_sized"}:
                        return evaluate(inputs)
                    options = dict(stride=args.stride, derivative_order=mode)
                    if name == "tensor_sized":
                        options.update(
                            algorithm="tensor",
                            chunk_windows=_tensor_sized_chunk_windows(
                                compiled.plan,
                                window,
                                inputs.shape[0],
                                args.windows,
                                mode,
                            ),
                        )
                    selections = entry.setdefault("selection_by_mode", {})
                    if str(mode) not in selections:
                        selections[str(mode)] = asdict(
                            evaluate.select(inputs, **options)
                        )
                    return evaluate(inputs, **options)

                def verify(mode):
                    sync()
                    started = time.perf_counter()
                    if mode == 0:
                        with torch.no_grad():
                            result = evaluate_mode(dx, mode)
                    else:
                        result = evaluate_mode(dx, mode)
                        vjp = torch.autograd.grad((result * device_weights).sum(), dx)[
                            0
                        ]
                    sync()
                    if mode == 0:
                        entry["verification_call_ms"] = (
                            time.perf_counter() - started
                        ) * 1000
                    actual = result.detach().cpu().double()
                    value_error = error_metrics(actual, expected, atol=atol, rtol=rtol)
                    entry["value_error" if mode == 0 else "training_value_error"] = (
                        value_error
                    )
                    if mode == 0:
                        entry["max_abs_error"] = value_error["max_abs"]
                        entry["value_max_scaled_error_by_window"] = (
                            window_scaled_errors(actual, expected, atol=atol, rtol=rtol)
                        )
                        entry["value_error_by_grade"] = {
                            str(grade): error_metrics(
                                actual[..., oracle_plan.basis_depths == grade],
                                expected[..., oracle_plan.basis_depths == grade],
                                atol=atol,
                                rtol=rtol,
                            )
                            for grade in range(1, depth + 1)
                        }
                    if mode == 1:
                        entry["vjp_error"] = error_metrics(
                            vjp.detach().cpu().double(), gradient, atol=atol, rtol=rtol
                        )
                        entry["max_abs_vjp_error"] = entry["vjp_error"]["max_abs"]
                    if not value_error["passed"] or (
                        mode == 1 and not entry["vjp_error"]["passed"]
                    ):
                        raise AssertionError(
                            "nonfinite result or componentwise value/VJP tolerance exceeded"
                        )

                def forward():
                    with torch.no_grad():
                        return evaluate_mode(dx, 0)

                def backward():
                    return torch.autograd.grad(
                        (evaluate_mode(dx, 1) * device_weights).sum(), dx
                    )

                def second_order():
                    return value_gradient_hvp(
                        lambda x: evaluate_mode(x, 2),
                        dx,
                        device_weights,
                        device_direction,
                    )

                def measure(mode, function, key):
                    passed, timing = run_phase(
                        entry,
                        mode,
                        "timing",
                        lambda: time_call(function, sync=sync, repeats=args.repeats),
                        checkpoint,
                    )
                    entry["timed_modes"][str(mode)] = passed
                    if passed:
                        entry[key] = timing
                    checkpoint()
                    if passed and args.profile:
                        profiled, metrics = run_phase(
                            entry,
                            mode,
                            "profile",
                            lambda: profile_call(
                                lambda x: evaluate_mode(x, mode),
                                dx,
                                device_weights,
                                direction=device_direction if mode == 2 else None,
                                derivative_order=mode,
                                trace_operators=False,
                            ),
                            checkpoint,
                        )
                        if profiled:
                            entry["profiles"][str(mode)] = metrics
                            if mode == 1:
                                entry["profile"] = metrics
                            elif mode == 2:
                                entry["profile_hvp"] = metrics
                        checkpoint()

                # Complete and checkpoint each mode before beginning the next.
                for mode in (0, 1) if args.backward or args.hvp else (0,):
                    passed, _ = run_phase(
                        entry,
                        mode,
                        "verification",
                        lambda m=mode: verify(m),
                        checkpoint,
                    )
                    entry["verified_modes"][str(mode)] = passed
                    entry["verified"] = entry["verified_modes"].get("0", False)
                    checkpoint()
                    if passed and (mode == 0 or args.backward):
                        measure(
                            mode,
                            forward if mode == 0 else backward,
                            "forward" if mode == 0 else "forward_vjp",
                        )
                if args.hvp and (name == "fused" or name.startswith("fused_chunk_")):
                    entry["verified_modes"]["2"] = False
                    entry["timed_modes"]["2"] = False
                    entry["mode_failures"]["2"] = {
                        "capability": "unsupported derivative_order=2: fused uses AOTAutograd; "
                        "its declared capability is derivative orders 0 and 1"
                    }
                    checkpoint()
                elif args.hvp:
                    oracle_ready = expected_hvp is not None
                    if not oracle_ready:
                        oracle_ready, oracle_derivatives = run_phase(
                            entry,
                            2,
                            "oracle",
                            compute_oracle_hvp,
                            checkpoint,
                        )
                        if oracle_ready:
                            expected_loss_gradient, expected_hvp = oracle_derivatives

                    def verify_second():
                        linear_value = evaluate_mode(dx, 2)
                        linear_gradient = torch.autograd.grad(
                            linear_value, dx, device_weights, create_graph=True
                        )[0]
                        entry["second_order_vjp_error"] = error_metrics(
                            linear_gradient.detach().cpu().double(),
                            gradient,
                            atol=atol,
                            rtol=rtol,
                        )
                        del linear_value, linear_gradient
                        if not entry["second_order_vjp_error"]["passed"]:
                            raise AssertionError(
                                "second-order linear-loss VJP tolerance exceeded"
                            )
                        value, loss_gradient, hvp = second_order()
                        for key, actual, reference in (
                            ("second_order_value_error", value, expected),
                            (
                                "loss_gradient_error",
                                loss_gradient,
                                expected_loss_gradient,
                            ),
                            ("hvp_error", hvp, expected_hvp),
                        ):
                            entry[key] = error_metrics(
                                actual.detach().cpu().double(),
                                reference,
                                atol=atol,
                                rtol=rtol,
                            )
                        if not all(
                            entry[key]["passed"]
                            for key in (
                                "second_order_value_error",
                                "loss_gradient_error",
                                "hvp_error",
                            )
                        ):
                            raise AssertionError(
                                "nonlinear-loss value/gradient/HVP tolerance exceeded"
                            )

                    passed = False
                    if oracle_ready:
                        passed, _ = run_phase(
                            entry, 2, "verification", verify_second, checkpoint
                        )
                    entry["verified_modes"]["2"] = passed
                    checkpoint()
                    if passed:
                        measure(2, second_order, "forward_hvp")

                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                entry["process_peak_rss_bytes"] = (
                    rss if platform.system() == "Darwin" else rss * 1024
                )
                entry["memory_scope"] = (
                    "whole process including compilation and independent oracle; total-process budget only, "
                    "not a runtime peak-memory regression measure"
                )
                if device.type == "mps":

                    def memory_sample():
                        sync()
                        return {
                            "current_allocated_bytes": torch.mps.current_allocated_memory(),
                            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                        }

                    measured, metrics = run_phase(
                        entry, 0, "memory-sample", memory_sample, checkpoint
                    )
                    if measured:
                        entry["mps_current_allocated_bytes"] = metrics[
                            "current_allocated_bytes"
                        ]
                        entry["mps_driver_allocated_bytes"] = metrics[
                            "driver_allocated_bytes"
                        ]
                    entry["mps_memory_scope"] = (
                        "sample after timing, not peak allocation"
                    )
                checkpoint()
            case["programs"] = {
                name: [
                    {
                        "rows": program.row_count,
                        "live_width_sum": program.activation_width,
                        "live_nodes": len(program.live),
                    }
                    for program in programs
                ]
                for name, programs in compiled._programs.items()
                if name != "fused"
            }
            case["powered_shift_nnz"] = [
                len(power.values) for power in compiled._powers
            ]
            case["polynomial_programs"] = {
                f"{kind}:{key}": {
                    "rows": program.row_count,
                    "live_width_sum": program.activation_width,
                    "live_nodes": len(program.nodes),
                    "packed_widths": program.widths,
                    "output_width": len(program.output_coordinates),
                }
                for kind, programs in (
                    ("joint", compiled._joint_programs),
                    ("bch", compiled._general_programs),
                )
                for key, program in programs.items()
            }
            for mode, key in (
                ("0", "forward"),
                ("1", "forward_vjp"),
                ("2", "forward_hvp"),
            ):
                valid = {
                    name: entry
                    for name, entry in case["candidates"].items()
                    if entry.get("verified_modes", {}).get(mode)
                    and entry.get("timed_modes", {}).get(mode)
                }
                case[f"fastest_{key}"] = min(
                    valid, key=lambda n: valid[n][key]["median_ms"], default=None
                )
            checkpoint()
            print(json.dumps(case), flush=True)
    report["complete"] = True
    checkpoint()
    # Baseline device limitations are recorded; a failed experimental candidate
    # or a numerical mismatch must also fail the process for automated sweeps.
    if any(
        result.get("mode_failures")
        and (
            name not in BASELINES
            or any(
                message.startswith("AssertionError:")
                for phases in result["mode_failures"].values()
                for message in phases.values()
            )
        )
        for case in report["cases"]
        for name, result in case["candidates"].items()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
