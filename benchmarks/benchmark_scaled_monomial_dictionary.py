"""Benchmark compiled local polynomial features and readouts."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import time

import torch

from tensorsig.offset_coded_signatures.layers import (
    LocalFeatures,
    LocalLinear,
)
from tensorsig.offset_coded_signatures.plans import compile_polynomial
from tensorsig.offset_coded_signatures.windows import increment_windows


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing result for one benchmark case."""

    name: str
    setup_ms: float
    forward_ms: float
    backward_ms: float | None
    peak_memory_mb: float | None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    name: str,
    setup: Callable[[], Callable[[], object]],
    device: torch.device,
    warmup: int,
    repeat: int,
    backward: bool,
) -> BenchmarkResult:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    setup_start = time.perf_counter()
    fn = setup()
    _synchronize(device)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    for _ in range(warmup):
        out = fn()
        if backward:
            if not isinstance(out, torch.Tensor):
                raise TypeError("Backward benchmarks must return a tensor.")
            out.sum().backward()
        _synchronize(device)

    forward_times = []
    backward_times = []
    for _ in range(repeat):
        forward_start = time.perf_counter()
        out = fn()
        _synchronize(device)
        forward_times.append((time.perf_counter() - forward_start) * 1000.0)
        if backward:
            if not isinstance(out, torch.Tensor):
                raise TypeError("Backward benchmarks must return a tensor.")
            backward_start = time.perf_counter()
            out.sum().backward()
            _synchronize(device)
            backward_times.append((time.perf_counter() - backward_start) * 1000.0)

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)

    return BenchmarkResult(
        name=name,
        setup_ms=setup_ms,
        forward_ms=sum(forward_times) / len(forward_times),
        backward_ms=(
            sum(backward_times) / len(backward_times) if backward_times else None
        ),
        peak_memory_mb=peak_memory_mb,
    )


def _make_path(
    batch_size: int,
    increments: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> torch.Tensor:
    dx = torch.randn(
        batch_size,
        increments,
        channels,
        device=device,
        dtype=dtype,
    )
    path = torch.cat(
        [
            torch.zeros(batch_size, 1, channels, device=device, dtype=dtype),
            dx.cumsum(dim=1),
        ],
        dim=1,
    )
    if requires_grad:
        path.requires_grad_()
    return path


def run_case(args: argparse.Namespace) -> list[BenchmarkResult]:
    """Run compilation, prepared-core, and end-to-end module benchmarks."""
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    path = _make_path(
        batch_size=args.batch_size,
        increments=args.increments,
        channels=args.channels,
        device=device,
        dtype=dtype,
        requires_grad=args.backward,
    )
    compile_kwargs = {
        "window": args.kernel_size,
        "channels": args.channels,
        "max_depth": args.max_depth,
        "max_offset_span": args.max_offset_span,
        "max_offset_multiplicity": args.max_offset_multiplicity,
    }
    plan = compile_polynomial(**compile_kwargs)
    windows = increment_windows(path.detach(), size=args.kernel_size).detach()
    windows.requires_grad_(args.backward)
    weight = torch.randn(
        args.out_channels,
        plan.feature_count,
        device=device,
        dtype=dtype,
        requires_grad=args.backward,
    )

    feature_module = LocalFeatures(plan, chunk_size=args.chunk_size)
    linear_module = LocalLinear(
        plan,
        args.out_channels,
        bias=False,
        chunk_size=args.chunk_size,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        linear_module.weight.copy_(weight)
        features = plan.evaluate(windows, chunk_size=args.chunk_size)
        contracted = plan.contract(windows, weight, chunk_size=args.chunk_size)
        expected = features @ weight.T
        feature_output = feature_module(path)
        linear_output = linear_module(path)
    # The fused contraction and materialized matrix product accumulate thousands
    # of float32 terms in different orders at the default shape.
    tolerance = {torch.float32: (2e-4, 2e-3), torch.float64: (1e-12, 1e-10)}
    atol, rtol = tolerance[dtype]
    torch.testing.assert_close(feature_output, features, atol=atol, rtol=rtol)
    torch.testing.assert_close(contracted, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(linear_output, contracted, atol=atol, rtol=rtol)
    del feature_module, linear_module
    del features, contracted, expected, feature_output, linear_output

    def compile_setup() -> Callable[[], object]:
        def run() -> object:
            return compile_polynomial(**compile_kwargs)

        return run

    def windows_setup() -> Callable[[], object]:
        def run() -> object:
            return increment_windows(path, size=args.kernel_size)

        return run

    def prepared_features_setup() -> Callable[[], object]:
        def run() -> object:
            return plan.evaluate(windows, chunk_size=args.chunk_size)

        return run

    def prepared_linear_setup() -> Callable[[], object]:
        def run() -> object:
            return plan.contract(windows, weight, chunk_size=args.chunk_size)

        return run

    def local_features_setup() -> Callable[[], object]:
        module = LocalFeatures(plan, chunk_size=args.chunk_size)

        def run() -> object:
            return module(path)

        return run

    def local_linear_setup() -> Callable[[], object]:
        module = LocalLinear(
            plan,
            args.out_channels,
            bias=False,
            chunk_size=args.chunk_size,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            module.weight.copy_(weight)

        def run() -> object:
            return module(path)

        return run

    cases = [
        ("compile-polynomial", compile_setup, False),
        ("increment-windows", windows_setup, args.backward),
        ("prepared-features", prepared_features_setup, args.backward),
        ("prepared-linear", prepared_linear_setup, args.backward),
        ("local-features", local_features_setup, args.backward),
        ("local-linear", local_linear_setup, args.backward),
    ]
    return [
        _measure(
            name=name,
            setup=setup,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
            backward=backward,
        )
        for name, setup, backward in cases
    ]


def main() -> None:
    """Run the benchmark from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--increments", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-offset-span", type=int, default=None)
    parser.add_argument("--max-offset-multiplicity", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    results = run_case(args)
    print(
        "name,setup_ms,forward_ms,backward_ms,peak_memory_mb",
        flush=True,
    )
    for result in results:
        backward = "" if result.backward_ms is None else f"{result.backward_ms:.3f}"
        memory = "" if result.peak_memory_mb is None else f"{result.peak_memory_mb:.3f}"
        print(
            f"{result.name},{result.setup_ms:.3f},{result.forward_ms:.3f},"
            f"{backward},{memory}",
            flush=True,
        )


if __name__ == "__main__":
    main()
