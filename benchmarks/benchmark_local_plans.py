#!/usr/bin/env python3
"""Benchmark compiled local feature plans and sliding backends."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.utils import generate_paths, time_call  # noqa: E402
from tensorsig.offset_coded_signatures.codes import OffsetCode  # noqa: E402
from tensorsig.offset_coded_signatures.layers import (  # noqa: E402
    LocalFeatures,
    LocalLinear,
)
from tensorsig.offset_coded_signatures.plans import (  # noqa: E402
    LocalRealization,
    LogSignaturePlan,
    PolynomialPlan,
    compile_polynomial,
)
from tensorsig.offset_coded_signatures.windows import increment_windows  # noqa: E402


@dataclass(frozen=True)
class BenchmarkConfig:
    """One local signature/logsignature benchmark configuration."""

    channels: int
    code_dim: int
    depth: int
    increments: int
    batch: int
    kernel_size: int
    hop_size: int
    out_channels: int

    @property
    def num_windows(self) -> int:
        return 1 + (self.increments - self.kernel_size) // self.hop_size


def _time_operation(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    backward: bool,
    gradient_tensors: tuple[torch.Tensor, ...] = (),
    module: torch.nn.Module | None = None,
) -> float:
    if backward:

        def run() -> torch.Tensor:
            if module is not None:
                module.zero_grad(set_to_none=True)
            for tensor in gradient_tensors:
                tensor.grad = None
            output = fn()
            loss = output.square().mean()
            loss.backward()
            return loss

    else:

        def run() -> torch.Tensor:
            with torch.no_grad():
                return fn()

    return time_call(run, device=device, warmup=warmup, repeats=repeats) * 1e3


def _compile_plans(
    config: BenchmarkConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[PolynomialPlan, LogSignaturePlan, float, float]:
    start = time.perf_counter()
    raw_plan = compile_polynomial(
        window=config.kernel_size,
        channels=config.channels,
        max_depth=config.depth,
    )
    raw_compile_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    code = OffsetCode.pascal(
        config.kernel_size,
        config.code_dim,
        device=device,
        dtype=dtype,
    )
    log_plan = LocalRealization(
        code,
        channels=config.channels,
        max_depth=config.depth,
    ).log_signature()
    log_compile_ms = (time.perf_counter() - start) * 1e3
    return raw_plan, log_plan, raw_compile_ms, log_compile_ms


def _assert_parity(
    core: Callable[[], torch.Tensor],
    end_to_end: Callable[[], torch.Tensor],
    dtype: torch.dtype,
) -> None:
    with torch.no_grad():
        expected = core()
        actual = end_to_end()
    if dtype == torch.float64:
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def _assert_sliding_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Check rolling drift relative to the largest computed coefficient."""
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}.")
    scale = expected.abs().max().clamp_min(torch.finfo(dtype).tiny)
    relative_error = (actual - expected).abs().max() / scale
    tolerance = 1e-9 if dtype == torch.float64 else 2e-3
    if relative_error > tolerance:
        raise AssertionError(
            f"Relative sliding error {float(relative_error):.3e} exceeds "
            f"{tolerance:.1e}."
        )


def benchmark(
    configs: Sequence[BenchmarkConfig],
    *,
    dtype: torch.dtype,
    device: torch.device,
    target: str,
    seed: int,
    warmup: int,
    repeats: int,
    max_basis_elements: int,
) -> list[dict[str, float | int | str]]:
    """Run the benchmark grid with compilation and execution phases separated."""
    backward = target.endswith("_backward")
    feature_target = target.startswith("features")
    records = []
    print(
        " C  P dep    L   R   B     N    raw_dim  primitive "
        "compile(raw/log) window  core(raw/log)   e2e(raw/log)  log/raw status"
    )

    for index, config in enumerate(configs):
        width = config.channels * config.code_dim
        dense_basis_elements = sum(width**depth for depth in range(1, config.depth + 1))
        if dense_basis_elements > max_basis_elements:
            print(
                f"{config.channels:>2} {config.code_dim:>2} {config.depth:>3} "
                f"{config.increments:>4} {config.kernel_size:>3} {config.batch:>3} "
                f"{config.num_windows:>5} {'-':>10} {'-':>10} skipped"
            )
            continue

        raw_plan, log_plan, raw_compile_ms, log_compile_ms = _compile_plans(
            config,
            dtype=dtype,
            device=device,
        )
        raw_dim = raw_plan.feature_count
        primitive_dim = log_plan.feature_count

        path = generate_paths(
            batch=config.batch,
            length=config.increments + 1,
            width=config.channels,
            dtype=dtype,
            device=device,
            seed=seed + index,
        )
        if backward:
            path = path.detach().requires_grad_(True)

        def extract_windows() -> torch.Tensor:
            return increment_windows(
                path,
                size=config.kernel_size,
                stride=config.hop_size,
            )

        window_ms = _time_operation(
            extract_windows,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward=False,
        )
        core_windows = extract_windows().detach()
        if backward:
            core_windows.requires_grad_(True)

        if feature_target:
            raw_module: torch.nn.Module = LocalFeatures(
                raw_plan,
                stride=config.hop_size,
            ).to(device=device, dtype=dtype)
            log_module: torch.nn.Module = LocalFeatures(
                log_plan,
                stride=config.hop_size,
            ).to(device=device, dtype=dtype)

            def raw_core() -> torch.Tensor:
                return raw_plan.evaluate(core_windows)

            def log_core() -> torch.Tensor:
                return log_plan.evaluate(core_windows)

        else:
            raw_module = LocalLinear(
                raw_plan,
                config.out_channels,
                stride=config.hop_size,
                device=device,
                dtype=dtype,
            )
            log_module = LocalLinear(
                log_plan,
                config.out_channels,
                stride=config.hop_size,
                device=device,
                dtype=dtype,
            )

            def raw_core() -> torch.Tensor:
                assert isinstance(raw_module, LocalLinear)
                output = raw_plan.contract(core_windows, raw_module.weight)
                if raw_module.bias is not None:
                    output = output + raw_module.bias
                return output

            def log_core() -> torch.Tensor:
                assert isinstance(log_module, LocalLinear)
                output = log_plan.contract(core_windows, log_module.weight)
                if log_module.bias is not None:
                    output = output + log_module.bias
                return output

        def raw_end_to_end() -> torch.Tensor:
            return raw_module(path)

        def log_end_to_end() -> torch.Tensor:
            return log_module(path)

        _assert_parity(raw_core, raw_end_to_end, dtype)
        _assert_parity(log_core, log_end_to_end, dtype)

        raw_core_ms = _time_operation(
            raw_core,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward=backward,
            gradient_tensors=(core_windows,),
            module=raw_module,
        )
        log_core_ms = _time_operation(
            log_core,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward=backward,
            gradient_tensors=(core_windows,),
            module=log_module,
        )
        raw_e2e_ms = _time_operation(
            raw_end_to_end,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward=backward,
            gradient_tensors=(path,),
            module=raw_module,
        )
        log_e2e_ms = _time_operation(
            log_end_to_end,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward=backward,
            gradient_tensors=(path,),
            module=log_module,
        )

        ratio = log_e2e_ms / raw_e2e_ms
        record: dict[str, float | int | str] = {
            "channels": config.channels,
            "code_dim": config.code_dim,
            "depth": config.depth,
            "increments": config.increments,
            "batch": config.batch,
            "kernel_size": config.kernel_size,
            "hop_size": config.hop_size,
            "num_windows": config.num_windows,
            "raw_feature_dim": raw_dim,
            "primitive_feature_dim": primitive_dim,
            "log_over_raw_time": ratio,
            "raw_compile_ms": raw_compile_ms,
            "log_compile_ms": log_compile_ms,
            "window_ms": window_ms,
            "raw_core_ms": raw_core_ms,
            "log_core_ms": log_core_ms,
            "raw_e2e_ms": raw_e2e_ms,
            "log_e2e_ms": log_e2e_ms,
            "target": target,
            "dtype": str(dtype).removeprefix("torch."),
            "device": device.type,
        }
        records.append(record)
        print(
            f"{config.channels:>2} {config.code_dim:>2} {config.depth:>3} "
            f"{config.increments:>4} {config.kernel_size:>3} {config.batch:>3} "
            f"{config.num_windows:>5} {raw_dim:>10} {primitive_dim:>10} "
            f"{raw_compile_ms:>7.3f}/{log_compile_ms:<7.3f} "
            f"{window_ms:>6.3f} {raw_core_ms:>7.3f}/{log_core_ms:<7.3f} "
            f"{raw_e2e_ms:>7.3f}/{log_e2e_ms:<7.3f} {ratio:>7.2f}x ok"
        )
    return records


def benchmark_sliding(
    configs: Sequence[BenchmarkConfig],
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    warmup: int,
    repeats: int,
    max_basis_elements: int,
) -> list[dict[str, float | int | str]]:
    """Compare vectorized and overlap-reusing dense signature evaluation."""
    records = []
    print(
        " C  P dep    L   R   B     N   features  "
        "vectorized  rolling  roll/vec     auto  auto/vec status"
    )

    for index, config in enumerate(configs):
        width = config.channels * config.code_dim
        feature_count = sum(width**depth for depth in range(1, config.depth + 1))
        if feature_count > max_basis_elements:
            print(
                f"{config.channels:>2} {config.code_dim:>2} {config.depth:>3} "
                f"{config.increments:>4} {config.kernel_size:>3} {config.batch:>3} "
                f"{config.num_windows:>5} {'-':>10} skipped"
            )
            continue

        code = (
            OffsetCode.constant(
                config.kernel_size,
                device=device,
                dtype=dtype,
            )
            if config.code_dim == 1
            else OffsetCode.pascal(
                config.kernel_size,
                config.code_dim,
                device=device,
                dtype=dtype,
            )
        )
        plan = LocalRealization(
            code,
            channels=config.channels,
            max_depth=config.depth,
        ).signature(coordinates="dense")
        vectorized_module = LocalFeatures(
            plan,
            stride=config.hop_size,
            backend="vectorized",
        ).to(device=device, dtype=dtype)
        rolling_module = LocalFeatures(
            plan,
            stride=config.hop_size,
            backend="rolling",
        ).to(device=device, dtype=dtype)
        auto_module = LocalFeatures(
            plan,
            stride=config.hop_size,
            backend="auto",
        ).to(device=device, dtype=dtype)
        path = generate_paths(
            batch=config.batch,
            length=config.increments + 1,
            width=config.channels,
            dtype=dtype,
            device=device,
            seed=seed + index,
        )

        def vectorized() -> torch.Tensor:
            return vectorized_module(path)

        def rolling() -> torch.Tensor:
            return rolling_module(path)

        with torch.no_grad():
            expected = vectorized()
            actual = rolling()
        _assert_sliding_parity(actual, expected, dtype)

        vectorized_ms = _time_operation(
            vectorized,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward=False,
        )
        rolling_ms = _time_operation(
            rolling,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward=False,
        )
        rolling_ratio = rolling_ms / vectorized_ms

        auto_ms = float("nan")
        auto_ratio = float("nan")
        auto_display = "       -"
        auto_ratio_display = "         -"
        if config.code_dim == 1:

            def automatic() -> torch.Tensor:
                return auto_module(path)

            with torch.no_grad():
                automatic_output = automatic()
            _assert_sliding_parity(automatic_output, expected, dtype)
            auto_ms = _time_operation(
                automatic,
                device=device,
                warmup=warmup,
                repeats=repeats,
                backward=False,
            )
            auto_ratio = auto_ms / vectorized_ms
            auto_display = f"{auto_ms:>8.3f}"
            auto_ratio_display = f"{auto_ratio:>9.2f}x"

        records.append(
            {
                "channels": config.channels,
                "code_dim": config.code_dim,
                "depth": config.depth,
                "increments": config.increments,
                "batch": config.batch,
                "kernel_size": config.kernel_size,
                "hop_size": config.hop_size,
                "num_windows": config.num_windows,
                "feature_dim": feature_count,
                "vectorized_ms": vectorized_ms,
                "rolling_ms": rolling_ms,
                "rolling_over_vectorized_time": rolling_ratio,
                "auto_ms": auto_ms,
                "auto_over_vectorized_time": auto_ratio,
                "target": "sliding",
                "dtype": str(dtype).removeprefix("torch."),
                "device": device.type,
            }
        )
        print(
            f"{config.channels:>2} {config.code_dim:>2} {config.depth:>3} "
            f"{config.increments:>4} {config.kernel_size:>3} {config.batch:>3} "
            f"{config.num_windows:>5} {feature_count:>10} "
            f"{vectorized_ms:>10.3f} {rolling_ms:>8.3f} "
            f"{rolling_ratio:>8.2f}x {auto_display} {auto_ratio_display} ok"
        )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--code-dims", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--lengths", type=int, nargs="+", default=[100])
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--batches", type=int, nargs="+", default=[1])
    parser.add_argument("--out-channels", type=int, default=16)
    parser.add_argument("--hop-size", type=int, default=1)
    parser.add_argument(
        "--target",
        choices=(
            "features",
            "features_backward",
            "linear",
            "linear_backward",
            "sliding",
        ),
        default="features",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-basis-elements", type=int, default=2_000_000)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    configs = [
        BenchmarkConfig(
            channels=channels,
            code_dim=code_dim,
            depth=depth,
            increments=increments,
            batch=batch,
            kernel_size=kernel_size,
            hop_size=args.hop_size,
            out_channels=args.out_channels,
        )
        for channels, code_dim, depth, increments, batch, kernel_size in itertools.product(
            args.channels,
            args.code_dims,
            args.depths,
            args.lengths,
            args.batches,
            args.kernel_sizes,
        )
        if kernel_size <= increments and code_dim <= kernel_size
    ]
    if args.target == "sliding":
        records = benchmark_sliding(
            configs,
            dtype=dtype,
            device=device,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            max_basis_elements=args.max_basis_elements,
        )
    else:
        records = benchmark(
            configs,
            dtype=dtype,
            device=device,
            target=args.target,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            max_basis_elements=args.max_basis_elements,
        )
    if args.output_csv is not None and records:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
