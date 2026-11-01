#!/usr/bin/env python3
"""Benchmark a strict free Volterra head against its gap factorization."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.utils import generate_paths, maybe_sync  # noqa: E402
from tensorsig.offset_coded_signatures.layers import (  # noqa: E402
    GapFactorizedVolterra,
    LocalLinear,
)
from tensorsig.offset_coded_signatures.plans import compile_polynomial  # noqa: E402


def _expanded_weight(layer: GapFactorizedVolterra) -> torch.Tensor:
    """Expand a factorized layer into its plan's free linear weight."""
    blocks = []
    for level, depth in enumerate(layer.depths):
        offsets = layer.plan.offset_indices[level].to(layer.channel_weights[0].device)
        channels = layer.plan.channel_indices[level].to(layer.channel_weights[0].device)
        temporal = torch.ones(
            layer.temporal_rank,
            offsets.shape[0],
            device=layer.channel_weights[0].device,
            dtype=layer.channel_weights[0].dtype,
        )
        internal = layer.internal_gap_factors[level]
        for position in range(depth - 1):
            assert internal is not None
            gaps = offsets[:, position + 1] - offsets[:, position]
            temporal = temporal * internal[:, position, gaps - 1]
        terminal_lags = layer.window - 1 - offsets[:, -1]
        temporal = temporal * layer.terminal_lag_factors[level][:, terminal_lags]

        channel_weight = layer.channel_weights[level]
        channel_indices = tuple(channels[:, position] for position in range(depth))
        selected = channel_weight[(slice(None), slice(None), *channel_indices)]
        blocks.append((selected * temporal.unsqueeze(0)).sum(dim=1))
    return torch.cat(blocks, dim=1)


def _median_ms(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        function()
    maybe_sync(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        maybe_sync(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def benchmark(
    window_sizes: Sequence[int],
    *,
    channels: int,
    depth: int,
    increments: int,
    batch: int,
    out_features: int,
    temporal_rank: int,
    stride: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
    seed: int,
) -> None:
    """Run exact-output and runtime comparisons over the requested windows."""
    print("  R  features  free_params  gap_params  free_ms   gap_ms  gap/free status")
    for index, window in enumerate(window_sizes):
        if depth > window or window > increments:
            print(f"{window:>3} skipped (need depth <= window <= increments)")
            continue
        plan = compile_polynomial(
            window=window,
            channels=channels,
            max_depth=depth,
            max_offset_multiplicity=1,
        )
        factorized = GapFactorizedVolterra(
            plan,
            out_features=out_features,
            temporal_rank=temporal_rank,
            stride=stride,
            backend="dynamic",
            gap_initialization="uniform",
            device=device,
            dtype=dtype,
        )
        free = LocalLinear(
            plan,
            out_features=out_features,
            stride=stride,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            free.weight.copy_(_expanded_weight(factorized))
            if free.bias is not None:
                assert factorized.bias is not None
                free.bias.copy_(factorized.bias)

        path = generate_paths(
            batch=batch,
            length=increments + 1,
            width=channels,
            dtype=dtype,
            device=device,
            seed=seed + index,
        )

        with torch.inference_mode():
            expected = free(path)
            actual = factorized(path)
            tolerance = 1e-10 if dtype == torch.float64 else 2e-5
            torch.testing.assert_close(
                actual,
                expected,
                atol=tolerance,
                rtol=tolerance,
            )

            free_ms = _median_ms(
                lambda: free(path),
                device=device,
                warmup=warmup,
                repeats=repeats,
            )
            factorized_ms = _median_ms(
                lambda: factorized(path),
                device=device,
                warmup=warmup,
                repeats=repeats,
            )

        ratio = factorized_ms / free_ms
        print(
            f"{window:>3} {plan.feature_count:>9} {_parameter_count(free):>12} "
            f"{_parameter_count(factorized):>10} {free_ms:>8.3f} "
            f"{factorized_ms:>8.3f} {ratio:>8.2f}x ok"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--increments", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--temporal-rank", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    if args.threads < 1:
        raise SystemExit("--threads must be positive.")
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    benchmark(
        args.window_sizes,
        channels=args.channels,
        depth=args.depth,
        increments=args.increments,
        batch=args.batch,
        out_features=args.out_features,
        temporal_rank=args.temporal_rank,
        stride=args.stride,
        dtype=dtype,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
