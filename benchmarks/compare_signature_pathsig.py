#!/usr/bin/env python3
"""Benchmark tensorsig signatures against pathsig.

``pathsig`` currently exposes a GPU-only signature implementation, so this
benchmark runs on CUDA and compares non-streaming tensor-coordinate signatures.
Both libraries return level-wise flattened signatures without the empty word.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.utils import generate_paths, maybe_sync, time_call_once
from tensorsig.classical.signatures import signature


def _format_ms(value: float) -> str:
    return f"{value * 1e3:8.2f} ms"


def _load_pathsig():
    try:
        import pathsig
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "The pathsig package is required for this benchmark. "
            "Install it in a CUDA-enabled environment, for example with "
            "`uv pip install --python .venv/bin/python pathsig`."
        ) from exc
    return pathsig


def _make_signature_fn(
    depth: int,
    compile_mode: str,
    compile_fullgraph: bool,
):
    def fn(path: torch.Tensor) -> torch.Tensor:
        return signature(path, depth=depth, stream=False)

    if compile_mode == "none":
        return fn
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")
    return torch.compile(fn, mode=compile_mode, fullgraph=compile_fullgraph)


def _default_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-8, 1e-7


def benchmark(
    lengths: Sequence[int],
    widths: Sequence[int],
    depths: Sequence[int],
    batches: Sequence[int],
    repeats: int,
    warmup: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    compile_mode: str,
    compile_fullgraph: bool,
    check_outputs: bool,
    atol: float,
    rtol: float,
) -> List[dict]:
    """Run a CUDA timing sweep against pathsig.

    Parameters
    ----------
    lengths, widths, depths, batches : Sequence[int]
        Shape grid for paths of shape ``(batch, length, width)``.
    repeats : int
        Number of timed repetitions per configuration.
    warmup : int
        Number of untimed warmup calls for each implementation.
    dtype : torch.dtype
        Floating point dtype for sampled paths.
    device : torch.device
        CUDA device used by both implementations.
    seed : int
        Base random seed for reproducible path generation.
    compile_mode : str
        ``torch.compile`` mode for this implementation, or ``"none"``.
    compile_fullgraph : bool
        Whether to pass ``fullgraph=True`` to ``torch.compile``.
    check_outputs : bool
        Whether to assert close numerical agreement before timing each shape.
    atol, rtol : float
        Numerical tolerances used when ``check_outputs`` is true.

    Returns
    -------
    list[dict]
        One record per benchmarked shape.
    """

    pathsig = _load_pathsig()
    header = (
        "width depth length batch dtype device compile_mode "
        "ours_time pathsig_time ours_speedup max_err"
    )
    print(header)
    records: List[dict] = []
    for width, depth, length, batch in itertools.product(
        widths, depths, lengths, batches
    ):
        paths = generate_paths(
            batch=batch,
            length=length,
            width=width,
            dtype=dtype,
            device=device,
            seed=seed + width + depth + length + batch,
        ).contiguous()

        ours_fn = _make_signature_fn(
            depth=depth,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
        )

        def pathsig_fn(path: torch.Tensor) -> torch.Tensor:
            return pathsig.signature(path, truncation_level=depth)

        max_err = float("nan")
        with torch.inference_mode():
            if check_outputs:
                ours = ours_fn(paths)
                ref = pathsig_fn(paths)
                maybe_sync(device)
                if ours.shape != ref.shape:
                    raise RuntimeError(
                        f"Shape mismatch for width={width}, depth={depth}, "
                        f"length={length}, batch={batch}: ours {ours.shape}, "
                        f"pathsig {ref.shape}"
                    )
                torch.testing.assert_close(
                    ours,
                    ref,
                    atol=atol,
                    rtol=rtol,
                    msg=(
                        "Outputs differ for "
                        f"width={width}, depth={depth}, length={length}, batch={batch}"
                    ),
                )
                max_err = (ours - ref).abs().max().item()

            for _ in range(warmup):
                ours_fn(paths)
                pathsig_fn(paths)
            maybe_sync(device)

            ours_times: list[float] = []
            pathsig_times: list[float] = []
            for _ in range(repeats):
                _, ours_time = time_call_once(ours_fn, paths, device=device)
                _, pathsig_time = time_call_once(pathsig_fn, paths, device=device)
                ours_times.append(ours_time)
                pathsig_times.append(pathsig_time)

        mean_ours = statistics.fmean(ours_times)
        mean_pathsig = statistics.fmean(pathsig_times)
        speedup = mean_pathsig / mean_ours if mean_ours > 0 else float("inf")
        record = {
            "width": width,
            "depth": depth,
            "length": length,
            "batch": batch,
            "dtype": str(dtype).replace("torch.", ""),
            "device": device.type,
            "compile_mode": compile_mode,
            "ours_time_ms": mean_ours * 1e3,
            "pathsig_time_ms": mean_pathsig * 1e3,
            "ours_speedup": speedup,
            "max_err": max_err,
        }
        records.append(record)
        print(
            f"{width:>5} {depth:>5} {length:>6} {batch:>5} "
            f"{record['dtype']:>7} {record['device']:>6} {compile_mode[:12]:>12} "
            f"{_format_ms(mean_ours)} {_format_ms(mean_pathsig)} "
            f"{speedup:7.2f}x {max_err:.2e}"
        )
    return records


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Benchmark tensorsig.classical.signatures against pathsig.")
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 3, 4, 8])
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument(
        "--repeats", type=int, default=10, help="Timing repeats per config."
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per config.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device for both implementations, for example 'cuda' or 'cuda:1'.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="none",
        help="torch.compile mode for this implementation; use 'none' to disable.",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Use fullgraph=True when compiling this implementation.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip output agreement checks before timing.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance for output checks; defaults by dtype.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance for output checks; defaults by dtype.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV output path for regression tracking.",
    )
    parser.add_argument(
        "--require-speedup",
        type=float,
        default=None,
        help=(
            "If set, fail when this implementation is slower than pathsig by "
            "more than this factor. For example, 1.5 allows up to 1.5x slower."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("pathsig is GPU-only, so this benchmark requires CUDA.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    atol_default, rtol_default = _default_tolerance(dtype)
    torch.set_grad_enabled(False)
    records = benchmark(
        lengths=args.lengths,
        widths=args.widths,
        depths=args.depths,
        batches=args.batches,
        repeats=args.repeats,
        warmup=args.warmup,
        dtype=dtype,
        device=device,
        seed=args.seed,
        compile_mode=args.compile_mode,
        compile_fullgraph=args.compile_fullgraph,
        check_outputs=not args.no_check,
        atol=args.atol if args.atol is not None else atol_default,
        rtol=args.rtol if args.rtol is not None else rtol_default,
    )
    if args.output_csv:
        import csv

        fieldnames = [
            "width",
            "depth",
            "length",
            "batch",
            "dtype",
            "device",
            "compile_mode",
            "ours_time_ms",
            "pathsig_time_ms",
            "ours_speedup",
            "max_err",
        ]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    if args.require_speedup is not None:
        factor = args.require_speedup
        slow_cases = [
            r for r in records if r["ours_time_ms"] > r["pathsig_time_ms"] * factor
        ]
        if slow_cases:
            details = ", ".join(
                f"(w={r['width']},d={r['depth']},L={r['length']},"
                f"B={r['batch']},speedup={r['ours_speedup']:.2f}x)"
                for r in slow_cases
            )
            raise SystemExit(
                f"Speedup requirement not met (factor={factor}). Slow cases: {details}"
            )


if __name__ == "__main__":
    main()
