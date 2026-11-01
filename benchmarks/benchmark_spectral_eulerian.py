"""Compare the subset-cut projector with factorial projection.

CPU float64 only. Correctness and one seeded VJP are checked before timing.
Memory limits cover estimated dense matrix storage, not measured process peak
memory. Any failed check is retained in JSON and exits nonzero; failed candidates
receive no timing.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

from tensorsig.experimental.offset_coded_signatures import spectral_eulerian
from tensorsig.offset_coded_signatures import primitive_chart


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--max-matrix-mib", type=float, default=32)
    parser.add_argument("--max-factorial-terms", type=int, default=40320)
    args = parser.parse_args()
    positive = args.widths + args.depths + [args.columns, args.repeats, args.threads]
    if min(positive) < 1:
        parser.error("sizes must be positive")
    if (
        not math.isfinite(args.max_matrix_mib)
        or args.max_matrix_mib <= 0
        or args.max_factorial_terms < 1
        or not math.isfinite(args.atol)
        or args.atol < 0
        or not math.isfinite(args.rtol)
        or args.rtol < 0
    ):
        parser.error("limits must be positive and tolerances finite and nonnegative")
    torch.set_num_threads(args.threads)
    package = Path(primitive_chart.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    projector = Path(spectral_eulerian.__file__)
    digest.update(b"research/spectral_eulerian.py")
    digest.update(projector.read_bytes())
    report = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "threads": torch.get_num_threads(),
            "device": "cpu",
            "dtype": "float64",
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_sha256": digest.hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "projector_sha256": hashlib.sha256(projector.read_bytes()).hexdigest(),
        "oracle": "maintained factorial Eulerian projector",
        "memory_scope": "dense input matrix byte estimate; not measured peak memory",
        "timing_scope": "fresh output per call, one warmup, repeated wall-clock samples; caches may be warm",
        "results": [],
    }
    generator = torch.Generator().manual_seed(args.seed)
    failures = 0

    def error(actual, expected):
        return (
            float((actual - expected).detach().abs().max()) if actual.numel() else 0.0
        )

    def check(actual, expected):
        torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)

    def measure(fn):
        fn()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value = fn()
            times.append((time.perf_counter() - start) * 1000)
            del value
        return {"samples_ms": times, "median_ms": statistics.median(times)}

    def feasible(row, rows, columns, depth):
        row["estimated_dense_matrix_bytes"] = rows * columns * 8
        reason = None
        if row["estimated_dense_matrix_bytes"] > args.max_matrix_mib * 2**20:
            reason = "dense matrix exceeds --max-matrix-mib"
        elif math.factorial(depth) > args.max_factorial_terms:
            reason = "factorial oracle exceeds --max-factorial-terms"
        if reason:
            row.update(status="skipped", reason=reason)
            report["results"].append(row)
        return reason is None

    for width, depth in itertools.product(args.widths, args.depths):
        row = dict(
            kind="projector",
            width=width,
            depth=depth,
            columns=args.columns,
            factorial_terms=math.factorial(depth),
            spectral_terms=(depth - 1) * 2**depth,
        )
        if not feasible(row, width**depth, args.columns, depth):
            continue
        matrix = torch.randn(
            width**depth, args.columns, generator=generator, dtype=torch.float64
        )
        cotangent = torch.randn(matrix.shape, generator=generator, dtype=matrix.dtype)

        def baseline(x):
            return primitive_chart._eulerian_project_columns(x, width, depth)

        def candidate(x):
            return spectral_eulerian.spectral_eulerian_project_columns(x, width, depth)

        try:
            expected = baseline(matrix)
            actual = candidate(matrix)
            reference_input = matrix.detach().requires_grad_()
            expected_vjp = torch.autograd.grad(
                baseline(reference_input), reference_input, cotangent
            )[0]
            actual_input = matrix.detach().requires_grad_()
            actual_vjp = torch.autograd.grad(
                candidate(actual_input), actual_input, cotangent
            )[0]
            row.update(
                value_abs_error=error(actual, expected),
                vjp_abs_error=error(actual_vjp, expected_vjp),
                idempotence_abs_error=error(candidate(actual), actual),
            )
            check(actual, expected)
            check(actual_vjp, expected_vjp)
            check(candidate(actual), actual)
            row["factorial"] = measure(lambda: baseline(matrix))
            row["spectral"] = measure(lambda: candidate(matrix))
            row["speedup"] = (
                row["factorial"]["median_ms"] / row["spectral"]["median_ms"]
            )
            row["status"] = "passed"
        except Exception as exc:
            failures += 1
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        report["results"].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    counts = {
        status: sum(row["status"] == status for row in report["results"])
        for status in ("passed", "failed", "skipped")
    }
    print(json.dumps(dict(output=str(args.output), **counts)))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
