#!/usr/bin/env python3
"""Benchmark three Lie-quotient runtimes and selected-word evaluation.

Examples
--------
Run the representative CPU grid::

    uv run python benchmarks/benchmark_lie_quotients.py

Run the small numerical self-check suitable for CI::

    uv run python benchmarks/benchmark_lie_quotients.py --quick

Plan timings are reported separately and excluded from runtime measurements.
All cases use already-computed increment windows to isolate plan evaluation. The
three quotient methods are direct SparseBCH, a quotient PBW signature scan with
Eulerian readout, and the ordinary ambient tensor signature→log implementation
followed by the plan's lazy Lyndon-word bridge.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import statistics
import time

import torch
from torch import Tensor

from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.coordinates import WordSelection
from tensorsig.offset_coded_signatures.plans import LocalRealization
from tensorsig.offset_coded_signatures.quotients import (
    DependenceGraph,
    LieQuotient,
    OffsetSpan,
    TemporalDegree,
)
from tensorsig.offset_coded_signatures.windows import increment_windows
from tensorsig.offset_coded_signatures._lie_quotients.enveloping import (
    compile_quotient_signature_plan,
)
from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    compile_lie_quotient,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import (
    OffsetAlphabet,
    ProjectedAlphabet,
)


@dataclass(frozen=True)
class QuotientCase:
    """One overlapping-window quotient comparison."""

    name: str
    code: OffsetCode
    channels: int
    quotient: LieQuotient
    depth: int
    batch: int
    windows: int


@dataclass(frozen=True)
class PathQuotientCase:
    """One whole-path quotient comparison on precomputed increments."""

    name: str
    code: OffsetCode
    channels: int
    quotient: LieQuotient
    depth: int
    batch: int


@dataclass(frozen=True)
class WordCase:
    """One sparse-word versus dense-tensor comparison."""

    name: str
    channels: int
    code_dim: int
    depth: int
    kernel_size: int
    batch: int
    windows: int
    selection: WordSelection


@dataclass(frozen=True)
class Timing:
    """Median call time and its raw samples."""

    median_ms: float
    samples_ms: tuple[float, ...]


def _pascal_code(window: int, dimension: int) -> OffsetCode:
    return OffsetCode.pascal(window, dimension, dtype=torch.float64)


def _constant_code(window: int) -> OffsetCode:
    return OffsetCode.constant(window, dtype=torch.float64)


def _ambient_width(code: OffsetCode, channels: int) -> int:
    letters = code.window if code.retains_physical_offsets else code.dimension
    return letters * channels


def _compiler_alphabet(
    code: OffsetCode,
    channels: int,
) -> ProjectedAlphabet | OffsetAlphabet:
    if code.retains_physical_offsets:
        return OffsetAlphabet(code.window, channels)
    return ProjectedAlphabet(code.dimension, channels, code.temporal_degrees)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_calls(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Timing:
    for _ in range(warmup):
        fn()
    _synchronize(device)

    samples = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter_ns()
        fn()
        _synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return Timing(statistics.median(samples), tuple(samples))


def _random_windows(
    batch: int,
    windows: int,
    kernel_size: int,
    channels: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        batch,
        windows,
        kernel_size,
        channels,
        generator=generator,
        dtype=dtype,
    ).to(device)


def _random_path(
    batch: int,
    increments: int,
    channels: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dx = torch.randn(
        batch,
        increments,
        channels,
        generator=generator,
        dtype=dtype,
    ).to(device)
    origin = torch.zeros(batch, 1, channels, device=device, dtype=dtype)
    return torch.cat((origin, dx.cumsum(dim=1)), dim=1)


def _compile_quotient_timing(
    case: QuotientCase | PathQuotientCase,
    *,
    device: torch.device,
):
    plans = []
    alphabet = _compiler_alphabet(case.code, case.channels)

    def compile_once() -> None:
        plans.append(compile_lie_quotient(case.quotient, alphabet, case.depth))

    timing = _time_calls(
        compile_once,
        device=device,
        warmup=0,
        repeats=1,
    )
    return timing, plans[-1]


def _compile_pbw_timing(
    quotient_plan,
    *,
    device: torch.device,
):
    plans = []

    def compile_once() -> None:
        plans.append(compile_quotient_signature_plan(quotient_plan))

    timing = _time_calls(
        compile_once,
        device=device,
        warmup=0,
        repeats=1,
    )
    return timing, plans[-1]


def _compile_bridge_timing(quotient_plan, *, device: torch.device) -> Timing:
    return _time_calls(
        lambda: quotient_plan._word_bridge,
        device=device,
        warmup=0,
        repeats=1,
    )


def _compile_local_plan_timing(
    case: QuotientCase | PathQuotientCase,
    code: OffsetCode,
    *,
    device: torch.device,
    repeats: int,
):
    plans = []
    realization = LocalRealization(code, case.channels, case.depth)

    def compile_once() -> None:
        plans.append(realization.log_signature(quotient=case.quotient))

    timing = _time_calls(
        compile_once,
        device=device,
        warmup=0,
        repeats=repeats,
    )
    return timing, plans[-1]


def _winner(timings: dict[str, Timing]) -> str:
    """Return the fastest method, allowing a five-percent tie."""
    best_name, best = min(timings.items(), key=lambda item: item[1].median_ms)
    tied = [
        name
        for name, timing in timings.items()
        if timing.median_ms <= 1.05 * best.median_ms
    ]
    return best_name if len(tied) == 1 else "similar"


def _quotient_tolerances(
    dtype: torch.dtype,
    depth: int,
    length: int,
) -> tuple[float, float]:
    if dtype == torch.float64:
        return 2e-9, 2e-8
    if depth >= 6 or length >= 512:
        return 5e-4, 1e-2
    return 5e-5, 5e-4


def _verdict(
    speedup: float,
    numerator_method: str,
    denominator_method: str,
    tolerance: float = 0.05,
) -> str:
    if speedup > 1.0 + tolerance:
        return denominator_method
    if speedup < 1.0 - tolerance:
        return numerator_method
    return "similar"


def _word_break_even(compile_ms: float, sparse_ms: float, dense_ms: float) -> str:
    """Describe when a sparse word plan repays its construction cost."""
    if _verdict(dense_ms / sparse_ms, "dense", "sparse") == "similar":
        return "shape-dependent"
    saving = dense_ms - sparse_ms
    if saving <= 0:
        return "dense dominates"
    calls = max(1, int(compile_ms / saving) + 1)
    return f"sparse after ~{calls} calls"


def run_quotient_case(
    case: QuotientCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
    compile_repeats: int,
    seed: int,
) -> dict[str, object]:
    """Benchmark and numerically verify one quotient case."""
    code = case.code.to(device=device, dtype=dtype)
    quotient_compile, quotient_plan = _compile_quotient_timing(
        case,
        device=device,
    )
    pbw_compile, signature_plan = _compile_pbw_timing(
        quotient_plan,
        device=device,
    )
    tensor_bridge = _compile_bridge_timing(quotient_plan, device=device)
    local_compile, plan = _compile_local_plan_timing(
        case,
        code,
        device=device,
        repeats=compile_repeats,
    )
    path = _random_path(
        case.batch,
        case.windows + code.window - 1,
        case.channels,
        dtype=dtype,
        device=device,
        seed=seed,
    )
    windows = increment_windows(path, size=code.window)

    def direct() -> Tensor:
        return plan.evaluate(windows, backend="direct")

    def tensor() -> Tensor:
        return plan.evaluate(windows, backend="tensor")

    def pbw() -> Tensor:
        return plan.evaluate(windows, backend="pbw")

    direct_output = direct()
    pbw_output = pbw()
    tensor_output = tensor()
    atol, rtol = _quotient_tolerances(dtype, case.depth, code.window)
    torch.testing.assert_close(direct_output, pbw_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(direct_output, tensor_output, atol=atol, rtol=rtol)

    direct_timing = _time_calls(
        direct,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    tensor_timing = _time_calls(
        tensor,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    pbw_timing = _time_calls(
        pbw,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    timings = {
        "direct": direct_timing,
        "pbw": pbw_timing,
        "tensor": tensor_timing,
    }
    width = _ambient_width(code, case.channels)
    return {
        "name": case.name,
        "shape": f"{width}/{code.window}/{case.batch * case.windows}",
        "ambient_dimension": quotient_plan.ambient_dimension,
        "ambient_signature_dimension": sum(
            width**level for level in range(1, case.depth + 1)
        ),
        "local_dimension": plan.feature_count,
        "quotient_dimension": quotient_plan.dimension,
        "brackets": len(quotient_plan.bracket_values),
        "pbw_dimension": signature_plan.dimension,
        "action_nnz": signature_plan.action_nnz,
        "local_compile_ms": local_compile.median_ms,
        "quotient_compile_ms": quotient_compile.median_ms,
        "pbw_compile_ms": pbw_compile.median_ms,
        "tensor_bridge_ms": tensor_bridge.median_ms,
        "direct_ms": direct_timing.median_ms,
        "pbw_ms": pbw_timing.median_ms,
        "tensor_ms": tensor_timing.median_ms,
        "pbw_vs_tensor": tensor_timing.median_ms / pbw_timing.median_ms,
        "winner": _winner(timings),
        "max_error": max(
            float((direct_output - pbw_output).abs().max()),
            float((direct_output - tensor_output).abs().max()),
        ),
    }


def run_path_quotient_case(
    case: PathQuotientCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
    compile_repeats: int,
    seed: int,
) -> dict[str, object]:
    """Benchmark one long-path case with compilation fully amortized."""
    code = case.code.to(device=device, dtype=dtype)
    quotient_compile, quotient_plan = _compile_quotient_timing(
        case,
        device=device,
    )
    pbw_compile, signature_plan = _compile_pbw_timing(
        quotient_plan,
        device=device,
    )
    tensor_bridge = _compile_bridge_timing(quotient_plan, device=device)
    local_compile, plan = _compile_local_plan_timing(
        case,
        code,
        device=device,
        repeats=compile_repeats,
    )
    increments = 0.05 * _random_windows(
        case.batch,
        1,
        code.window,
        case.channels,
        dtype=dtype,
        device=device,
        seed=seed,
    )

    def evaluate(method: str) -> Tensor:
        return plan.evaluate(increments, backend=method)

    outputs = {method: evaluate(method) for method in ("direct", "pbw", "tensor")}
    atol, rtol = _quotient_tolerances(dtype, case.depth, code.window)
    torch.testing.assert_close(outputs["pbw"], outputs["direct"], atol=atol, rtol=rtol)
    torch.testing.assert_close(outputs["pbw"], outputs["tensor"], atol=atol, rtol=rtol)
    timings = {
        method: _time_calls(
            lambda method=method: evaluate(method),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        for method in ("direct", "pbw", "tensor")
    }
    width = _ambient_width(code, case.channels)
    ambient_signature_dimension = sum(
        width**level for level in range(1, case.depth + 1)
    )
    return {
        "name": case.name,
        "shape": f"{width}/{code.window}/{case.batch}",
        "local_dimension": plan.feature_count,
        "quotient_dimension": quotient_plan.dimension,
        "pbw_dimension": signature_plan.dimension,
        "ambient_signature_dimension": ambient_signature_dimension,
        "action_nnz": signature_plan.action_nnz,
        "local_compile_ms": local_compile.median_ms,
        "quotient_compile_ms": quotient_compile.median_ms,
        "pbw_compile_ms": pbw_compile.median_ms,
        "tensor_bridge_ms": tensor_bridge.median_ms,
        "direct_ms": timings["direct"].median_ms,
        "pbw_ms": timings["pbw"].median_ms,
        "tensor_ms": timings["tensor"].median_ms,
        "pbw_vs_tensor": timings["tensor"].median_ms / timings["pbw"].median_ms,
        "winner": _winner(timings),
        "max_error": max(
            float((outputs["pbw"] - outputs["direct"]).abs().max()),
            float((outputs["pbw"] - outputs["tensor"]).abs().max()),
        ),
    }


def run_word_case(
    case: WordCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
    compile_repeats: int,
    seed: int,
) -> dict[str, object]:
    """Benchmark and numerically verify one selected-word case."""
    code = _pascal_code(case.kernel_size, case.code_dim).to(
        device=device,
        dtype=dtype,
    )
    realization = LocalRealization(code, case.channels, case.depth)
    sparse_plans = []
    dense_plans = []

    def compile_sparse() -> None:
        sparse_plans.append(
            realization.signature(
                selection=case.selection,
                depths=(case.depth,),
            )
        )

    def compile_dense() -> None:
        dense_plans.append(
            realization.signature(
                coordinates="dense",
                depths=(case.depth,),
            )
        )

    sparse_compile = _time_calls(
        compile_sparse,
        device=device,
        warmup=0,
        repeats=compile_repeats,
    )
    dense_compile = _time_calls(
        compile_dense,
        device=device,
        warmup=0,
        repeats=compile_repeats,
    )
    sparse_plan = sparse_plans[-1]
    dense_plan = dense_plans[-1]
    dxw = _random_windows(
        case.batch,
        case.windows,
        case.kernel_size,
        case.channels,
        dtype=dtype,
        device=device,
        seed=seed,
    )
    width = case.channels * case.code_dim
    powers = torch.tensor(
        [width**power for power in reversed(range(case.depth))],
        dtype=torch.long,
        device=device,
    )
    flat_indices = sparse_plan.word_indices[0].to(device=device) @ powers

    def sparse_words() -> Tensor:
        return sparse_plan.evaluate(dxw)

    def dense_then_gather() -> Tensor:
        return dense_plan.evaluate(dxw).index_select(-1, flat_indices)

    sparse_output = sparse_words()
    dense_output = dense_then_gather()
    atol = 2e-9 if dtype == torch.float64 else 5e-5
    rtol = 2e-8 if dtype == torch.float64 else 5e-4
    torch.testing.assert_close(sparse_output, dense_output, atol=atol, rtol=rtol)

    sparse_timing = _time_calls(
        sparse_words,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    dense_timing = _time_calls(
        dense_then_gather,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    speedup = dense_timing.median_ms / sparse_timing.median_ms
    return {
        "name": case.name,
        "dense_states": 1 + sum(width**depth for depth in range(1, case.depth + 1)),
        "sparse_states": sparse_plan.state_count,
        "features": sparse_plan.feature_count,
        "sparse_compile_ms": sparse_compile.median_ms,
        "dense_compile_ms": dense_compile.median_ms,
        "sparse_ms": sparse_timing.median_ms,
        "dense_ms": dense_timing.median_ms,
        "speedup": speedup,
        "winner": _verdict(speedup, "dense", "sparse"),
        "lifecycle": _word_break_even(
            sparse_compile.median_ms,
            sparse_timing.median_ms,
            dense_timing.median_ms,
        ),
        "max_error": float((sparse_output - dense_output).abs().max()),
    }


def _default_quotient_cases() -> tuple[QuotientCase, ...]:
    cases = []
    for depth in (3, 4, 5):
        cases.extend(
            (
                QuotientCase(
                    name=f"projected-weak-d{depth}",
                    code=_pascal_code(8, 2),
                    channels=2,
                    quotient=LieQuotient(TemporalDegree(depth - 1)),
                    depth=depth,
                    batch=8,
                    windows=16,
                ),
                QuotientCase(
                    name=f"projected-strong-d{depth}",
                    code=_pascal_code(8, 2),
                    channels=2,
                    quotient=LieQuotient(TemporalDegree(0)),
                    depth=depth,
                    batch=8,
                    windows=16,
                ),
                QuotientCase(
                    name=f"offset-weak-d{depth}",
                    code=OffsetCode.identity(2, dtype=torch.float64),
                    channels=2,
                    quotient=LieQuotient(OffsetSpan(1)),
                    depth=depth,
                    batch=8,
                    windows=16,
                ),
                QuotientCase(
                    name=f"offset-strong-d{depth}",
                    code=OffsetCode.identity(2, dtype=torch.float64),
                    channels=2,
                    quotient=LieQuotient(OffsetSpan(0)),
                    depth=depth,
                    batch=8,
                    windows=16,
                ),
                QuotientCase(
                    name=f"projected-wide-strong-d{depth}",
                    code=_pascal_code(2, 2),
                    channels=3,
                    quotient=LieQuotient(TemporalDegree(0)),
                    depth=depth,
                    batch=8,
                    windows=16,
                ),
                QuotientCase(
                    name=f"offset-wide-strong-d{depth}",
                    code=OffsetCode.identity(2, dtype=torch.float64),
                    channels=3,
                    quotient=LieQuotient(OffsetSpan(0)),
                    depth=depth,
                    batch=8,
                    windows=16,
                ),
                QuotientCase(
                    name=f"graph-chain-d{depth}",
                    code=_pascal_code(8, 1),
                    channels=3,
                    quotient=LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
                    depth=depth,
                    batch=8,
                    windows=16,
                ),
            )
        )
    return tuple(cases)


def _default_word_cases() -> tuple[WordCase, ...]:
    return (
        WordCase(
            "words-moderate",
            channels=3,
            code_dim=2,
            depth=3,
            kernel_size=12,
            batch=16,
            windows=32,
            selection=WordSelection(max_complexity=2),
        ),
        WordCase(
            "words-strong",
            channels=2,
            code_dim=4,
            depth=4,
            kernel_size=16,
            batch=8,
            windows=16,
            selection=WordSelection(max_complexity=1),
        ),
    )


def _default_path_quotient_cases() -> tuple[PathQuotientCase, ...]:
    strong_quotient = LieQuotient(TemporalDegree(0))
    cases = [
        PathQuotientCase(
            name=f"strong-d5-b{batch}-l{length}",
            code=_pascal_code(length, 2),
            channels=2,
            quotient=strong_quotient,
            depth=5,
            batch=batch,
        )
        for batch, length in ((1, 64), (8, 256), (8, 1024), (32, 128))
    ]
    cases.extend(
        (
            PathQuotientCase(
                name="strong-d6-b8-l256",
                code=_pascal_code(256, 2),
                channels=2,
                quotient=strong_quotient,
                depth=6,
                batch=8,
            ),
            PathQuotientCase(
                name="moderate-d5-b8-l256",
                code=_pascal_code(256, 2),
                channels=2,
                quotient=LieQuotient(TemporalDegree(1)),
                depth=5,
                batch=8,
            ),
            PathQuotientCase(
                name="graph-market-d5-b8-l256",
                code=_constant_code(256),
                channels=4,
                quotient=LieQuotient(DependenceGraph(((0, 1), (0, 2), (1, 2)))),
                depth=5,
                batch=8,
            ),
            PathQuotientCase(
                name="abelian-control-d5-b8-l256",
                code=_constant_code(256),
                channels=4,
                quotient=LieQuotient(DependenceGraph()),
                depth=5,
                batch=8,
            ),
            PathQuotientCase(
                name="weak-control-d4-b8-l256",
                code=_pascal_code(256, 2),
                channels=2,
                quotient=LieQuotient(TemporalDegree(3)),
                depth=4,
                batch=8,
            ),
        )
    )
    return tuple(cases)


def _quick_cases() -> tuple[
    tuple[QuotientCase, ...],
    tuple[PathQuotientCase, ...],
    tuple[WordCase, ...],
]:
    quotient_cases = (
        QuotientCase(
            name="quick-projected-weak-d3",
            code=_pascal_code(4, 2),
            channels=2,
            quotient=LieQuotient(TemporalDegree(2)),
            depth=3,
            batch=2,
            windows=3,
        ),
        QuotientCase(
            name="quick-projected-strong-d5",
            code=_pascal_code(4, 2),
            channels=2,
            quotient=LieQuotient(TemporalDegree(0)),
            depth=5,
            batch=2,
            windows=3,
        ),
        QuotientCase(
            name="quick-offset-weak-d3",
            code=OffsetCode.identity(2, dtype=torch.float64),
            channels=2,
            quotient=LieQuotient(OffsetSpan(1)),
            depth=3,
            batch=2,
            windows=3,
        ),
        QuotientCase(
            name="quick-offset-strong-d5",
            code=OffsetCode.identity(2, dtype=torch.float64),
            channels=2,
            quotient=LieQuotient(OffsetSpan(0)),
            depth=5,
            batch=2,
            windows=3,
        ),
        QuotientCase(
            name="quick-graph-chain-d3",
            code=_pascal_code(4, 1),
            channels=3,
            quotient=LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
            depth=3,
            batch=2,
            windows=3,
        ),
    )
    word_cases = (
        WordCase(
            "quick-words",
            channels=2,
            code_dim=2,
            depth=3,
            kernel_size=4,
            batch=2,
            windows=3,
            selection=WordSelection(max_complexity=1),
        ),
    )
    path_cases = (
        PathQuotientCase(
            name="quick-strong-d5",
            code=_pascal_code(16, 2),
            channels=2,
            quotient=LieQuotient(TemporalDegree(0)),
            depth=5,
            batch=2,
        ),
        PathQuotientCase(
            name="quick-graph-d3",
            code=_constant_code(16),
            channels=3,
            quotient=LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
            depth=3,
            batch=2,
        ),
        PathQuotientCase(
            name="quick-abelian-d3",
            code=_constant_code(16),
            channels=3,
            quotient=LieQuotient(DependenceGraph()),
            depth=3,
            batch=2,
        ),
    )
    return quotient_cases, path_cases, word_cases


def _print_quotient_results(records: list[dict[str, object]]) -> None:
    print("\nLocal quotients: direct BCH vs quotient PBW vs ambient tensor")
    print("shape is ambient-width/kernel-size/parallel paths")
    print(
        f"{'case':<27} {'shape':>11} {'l/q/u/t':>22} {'annz':>7} "
        f"{'local':>8} {'qbuild':>8} {'ubuild':>8} {'bridge':>8} "
        f"{'direct':>9} {'pbw':>9} {'tensor':>9} {'T/P':>7} {'winner':>8}"
    )
    for record in records:
        print(
            f"{record['name']:<27} "
            f"{record['shape']:>11} "
            f"{record['local_dimension']:>4}/{record['quotient_dimension']}/"
            f"{record['pbw_dimension']}/"
            f"{record['ambient_signature_dimension']:<8} "
            f"{record['action_nnz']:>7} "
            f"{record['local_compile_ms']:>7.2f} "
            f"{record['quotient_compile_ms']:>7.2f} "
            f"{record['pbw_compile_ms']:>7.2f} "
            f"{record['tensor_bridge_ms']:>7.2f} "
            f"{record['direct_ms']:>8.3f} "
            f"{record['pbw_ms']:>8.3f} "
            f"{record['tensor_ms']:>8.3f} "
            f"{record['pbw_vs_tensor']:>6.2f}x "
            f"{record['winner']:>8}"
        )
    winners = {
        name: sum(record["winner"] == name for record in records)
        for name in ("direct", "pbw", "tensor", "similar")
    }
    print(
        "Execution verdict: "
        f"direct {winners['direct']}, PBW {winners['pbw']}, "
        f"tensor {winners['tensor']}, "
        f"similar {winners['similar']}."
    )


def _print_path_quotient_results(records: list[dict[str, object]]) -> None:
    print("\nWhole paths: precomputed increments → quotient logsignature")
    print("shape is ambient-width/path-length/batch; compile times are excluded")
    print(
        f"{'case':<29} {'shape':>12} {'l/q/u/t':>23} {'annz':>8} "
        f"{'local':>9} {'qbuild':>9} {'ubuild':>9} {'bridge':>9} "
        f"{'direct':>10} {'pbw':>10} {'tensor':>10} {'T/P':>8} {'winner':>8}"
    )
    for record in records:
        print(
            f"{record['name']:<29} "
            f"{record['shape']:>12} "
            f"{record['local_dimension']:>4}/{record['quotient_dimension']}/"
            f"{record['pbw_dimension']}/"
            f"{record['ambient_signature_dimension']:<9} "
            f"{record['action_nnz']:>8} "
            f"{record['local_compile_ms']:>8.2f} "
            f"{record['quotient_compile_ms']:>8.2f} "
            f"{record['pbw_compile_ms']:>8.2f} "
            f"{record['tensor_bridge_ms']:>8.2f} "
            f"{record['direct_ms']:>9.3f} "
            f"{record['pbw_ms']:>9.3f} "
            f"{record['tensor_ms']:>9.3f} "
            f"{record['pbw_vs_tensor']:>7.2f}x "
            f"{record['winner']:>8}"
        )


def _print_word_results(records: list[dict[str, object]]) -> None:
    print("\nSelected words: sparse prefix plan vs dense coded tensor then gather")
    print(
        f"{'case':<20} {'states':>11} {'features':>9} {'sbuild':>9} {'dbuild':>9} "
        f"{'sparse':>9} {'dense':>9} {'speedup':>8} {'winner':>8}  lifecycle"
    )
    for record in records:
        print(
            f"{record['name']:<20} "
            f"{record['sparse_states']:>4}/{record['dense_states']:<6} "
            f"{record['features']:>8} "
            f"{record['sparse_compile_ms']:>8.2f} "
            f"{record['dense_compile_ms']:>8.2f} "
            f"{record['sparse_ms']:>8.3f} "
            f"{record['dense_ms']:>8.3f} "
            f"{record['speedup']:>7.2f}x "
            f"{record['winner']:>8}  {record['lifecycle']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compile-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small numerical and timing self-check suitable for CI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.compile_repeats < 1:
        raise SystemExit("warmup must be non-negative; repeat counts must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but is not available.")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.quick:
        quotient_cases, path_cases, word_cases = _quick_cases()
        warmup = min(args.warmup, 1)
        repeats = min(args.repeats, 3)
        compile_repeats = 1
    else:
        quotient_cases = _default_quotient_cases()
        path_cases = _default_path_quotient_cases()
        word_cases = _default_word_cases()
        warmup = args.warmup
        repeats = args.repeats
        compile_repeats = args.compile_repeats

    print(
        f"device={device}, dtype={dtype}, warmup={warmup}, "
        f"median of {repeats} executions; local/word build median={compile_repeats}"
    )
    print(f"torch={torch.__version__}, CPU threads={torch.get_num_threads()}")
    print(
        "Times are milliseconds; abstract quotient/PBW/bridge setup is one lazy "
        "call, so cache hits are visible."
    )
    torch.set_grad_enabled(False)
    quotient_records = [
        run_quotient_case(
            case,
            dtype=dtype,
            device=device,
            warmup=warmup,
            repeats=repeats,
            compile_repeats=compile_repeats,
            seed=args.seed + index,
        )
        for index, case in enumerate(quotient_cases)
    ]
    path_records = [
        run_path_quotient_case(
            case,
            dtype=dtype,
            device=device,
            warmup=warmup,
            repeats=repeats,
            compile_repeats=compile_repeats,
            seed=args.seed + len(quotient_cases) + index,
        )
        for index, case in enumerate(path_cases)
    ]
    word_records = [
        run_word_case(
            case,
            dtype=dtype,
            device=device,
            warmup=warmup,
            repeats=repeats,
            compile_repeats=compile_repeats,
            seed=args.seed + len(quotient_cases) + len(path_cases) + index,
        )
        for index, case in enumerate(word_cases)
    ]
    _print_quotient_results(quotient_records)
    _print_path_quotient_results(path_records)
    _print_word_results(word_records)
    print(
        "\nAll direct, quotient-PBW, and tensor-bridge outputs passed "
        "numerical equivalence checks."
    )


if __name__ == "__main__":
    main()
