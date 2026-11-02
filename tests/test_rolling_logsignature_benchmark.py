"""Checks for bounded oracle verification and reproducible benchmark controls."""

import json
import sys

import pytest
import torch

from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    compile_lie_quotient,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import ProjectedAlphabet
from tensorsig.offset_coded_signatures.rolling_logsignature import (
    RollingLogSignature,
)
from benchmarks import benchmark_rolling_logsignature as benchmark
from tensorsig.offset_coded_signatures.quotients import LieQuotient, TemporalDegree


def test_chunked_hvp_oracle_and_nonfinite_gate():
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 2), 3
    )
    rolling = RollingLogSignature(plan, window=4)
    dx = benchmark.make_increments((1, 9, 2), seed=91, scale=0.01).requires_grad_()
    weights = torch.ones(1, 6, plan.dimension, dtype=torch.float64)
    direction = torch.linspace(-1, 1, dx.numel(), dtype=dx.dtype).reshape_as(dx)
    _, gradient, hvp = benchmark.value_gradient_hvp(
        lambda x: rolling.reference(x, method="tensor"), dx, weights, direction
    )
    chunk_gradient, chunk_hvp = benchmark.reference_hvp_in_chunks(
        rolling, dx, weights, direction, stride=1, chunk_windows=2
    )
    torch.testing.assert_close(chunk_gradient, gradient)
    torch.testing.assert_close(chunk_hvp, hvp)
    assert not benchmark.error_metrics(
        torch.tensor([float("nan")]), torch.ones(1), atol=1e-9, rtol=1e-8
    )["passed"]


@pytest.mark.parametrize(
    "family", ["gaussian", "correlated", "constant", "alternating", "impulse", "zero"]
)
def test_input_families_are_deterministic(family):
    a = benchmark.make_increments((2, 5, 3), seed=17, scale=0.01, family=family)
    b = benchmark.make_increments((2, 5, 3), seed=17, scale=0.01, family=family)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("stride,chunk", [(1, 3), (2, 2), (5, 3)])
def test_chunked_tensor_oracle_accumulates_overlapping_vjps(stride, chunk):
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 2), 3
    )
    rolling = RollingLogSignature(plan, window=4)
    generator = torch.Generator().manual_seed(73)
    increments = (
        torch.randn(2, 4 + 6 * stride, 2, generator=generator, dtype=torch.float64)
        * 0.03
    ).requires_grad_()
    expected = rolling.reference(increments, stride=stride, method="tensor")
    weights = torch.randn(expected.shape, generator=generator, dtype=torch.float64)
    expected_gradient = torch.autograd.grad(expected, increments, weights)[0]
    batched = benchmark.WindowLogSignature(plan, rolling.window)(
        increments, stride=stride, algorithm="tensor", chunk_windows=chunk
    )
    batched_gradient = torch.autograd.grad(batched, increments, weights)[0]
    torch.testing.assert_close(batched, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        batched_gradient, expected_gradient, atol=1e-12, rtol=1e-12
    )
    value, gradient = benchmark.reference_value_and_vjp(
        rolling, increments, weights, stride=stride, chunk_windows=chunk
    )
    torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)
    assert not value.requires_grad
    assert not gradient.requires_grad


def test_benchmark_controls_and_tensor_budget(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    calls = []
    scans = []
    original_call = RollingLogSignature.__call__
    original_scan = RollingLogSignature.scan

    def call(self, increments, **kwargs):
        calls.append((increments.detach().clone(), kwargs["refresh_every"]))
        return original_call(self, increments, **kwargs)

    def scan(self, increments, **kwargs):
        scans.append(kwargs["refresh_every"])
        return original_scan(self, increments, **kwargs)

    monkeypatch.setattr(RollingLogSignature, "__call__", call)
    monkeypatch.setattr(RollingLogSignature, "scan", scan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--window",
            "4",
            "--depth",
            "2",
            "--windows",
            "7",
            "--batch",
            "1",
            "--channels",
            "2",
            "--repeats",
            "1",
            "--seed",
            "29",
            "--scale",
            "0.025",
            "--refresh-every",
            "2",
            "--oracle-chunk-windows",
            "2",
            "--max-tensor-elements",
            "128",
            "--baseline-chunk-windows",
            "2",
            "--backward",
            "--only",
            "tensor",
            "tensor_batched",
            "typed_dense",
            "scan_joint",
            "--output",
            str(output),
        ],
    )
    benchmark.main()
    report = json.loads(output.read_text())
    case = report["cases"][0]
    assert case["increment_scale"] == 0.025
    assert case["increments"] == 10
    assert case["oracle_ambient_level_elements"] == 128
    assert case["tensor_baseline_ambient_level_elements"] == 448
    assert "skipped" in case["candidates"]["tensor"]
    assert case["tensor_batched_ambient_level_elements"] == 128
    assert case["candidates"]["tensor_batched"]["verified"]
    assert case["candidates"]["typed_dense"]["verified"]
    assert case["candidates"]["scan_joint"]["verified"]
    assert calls and all(refresh == 2 for _, refresh in calls)
    assert scans and all(refresh == 2 for refresh in scans)
    expected = (
        torch.randn(
            1, 10, 2, generator=torch.Generator().manual_seed(29), dtype=torch.float64
        )
        * 0.025
    )
    torch.testing.assert_close(calls[0][0], expected, atol=0, rtol=0)


def test_tensor_oracle_budget_fails_before_compilation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--max-tensor-elements", "1"],
    )
    with pytest.raises(SystemExit, match="2"):
        benchmark.main()


def small_benchmark(monkeypatch, output, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--window",
            "3",
            "--depth",
            "2",
            "--windows",
            "3",
            "--batch",
            "1",
            "--channels",
            "2",
            "--repeats",
            "1",
            "--backward",
            "--only",
            "typed_dense",
            "--output",
            str(output),
            *extra,
        ],
    )


def test_oracle_cache_reuses_exact_values_vjp_and_lazy_hvp(tmp_path, monkeypatch):
    output, cache = tmp_path / "report.json", tmp_path / "cache"
    small_benchmark(monkeypatch, output, "--oracle-cache", str(cache), "--hvp")
    benchmark.main()
    assert not list(cache.glob("*.hvp-partial.pt"))
    record_path = next(cache.glob("*.pt"))
    fresh = torch.load(record_path, weights_only=True)
    assert set(fresh["tensors"]) == {"value", "vjp", "loss_gradient", "hvp"}

    def unexpected(*args, **kwargs):
        raise AssertionError("cached independent oracle was recomputed")

    monkeypatch.setattr(benchmark, "reference_value_and_vjp", unexpected)
    monkeypatch.setattr(benchmark, "reference_hvp_in_chunks", unexpected)
    benchmark.main()
    case = json.loads(output.read_text())["cases"][0]
    assert case["candidates"]["typed_dense"]["verified_modes"] == {
        "0": True,
        "1": True,
        "2": True,
    }
    assert case["oracle_cache"]["reused_fields"] == sorted(fresh["tensors"])
    assert not case["oracle_cache"]["written_fields"]
    assert case["oracle_cache"]["hvp_progress"]["resumed_from_window"] == 3
    assert (
        case["oracle_cache"]["hvp_progress"]["new_completed_chunk_compute_seconds"] == 0
    )
    assert fresh["completed_hvp_compute_seconds"] >= 0
    cached = torch.load(record_path, weights_only=True)
    for name, tensor in fresh["tensors"].items():
        torch.testing.assert_close(cached["tensors"][name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["metadata", "shape"])
def test_oracle_cache_rejects_invalid_record(tmp_path, corruption):
    metadata = {"source": "exact", "version": 1}
    shapes = {"value": (1, 2, 3), "vjp": (1, 4, 2)}
    path, _, provenance = benchmark.load_oracle_cache(tmp_path, metadata, shapes)
    tensors = {
        name: torch.zeros(shape, dtype=torch.float64) for name, shape in shapes.items()
    }
    benchmark.save_oracle_cache(path, metadata, tensors, provenance)
    record = torch.load(path, weights_only=True)
    if corruption == "record":
        record = []
    elif corruption == "metadata":
        record["metadata"]["version"] = 2
    else:
        record["tensors"]["vjp"] = torch.zeros(1, dtype=torch.float64)
    torch.save(record, path)
    with pytest.raises(ValueError, match="oracle cache"):
        benchmark.load_oracle_cache(tmp_path, metadata, shapes)


def test_oracle_cache_retains_first_order_after_hvp_failure(tmp_path, monkeypatch):
    output, cache = tmp_path / "report.json", tmp_path / "cache"

    def fail(*args, **kwargs):
        raise RuntimeError("oracle HVP interrupted")

    monkeypatch.setattr(benchmark, "reference_hvp_in_chunks", fail)
    small_benchmark(monkeypatch, output, "--oracle-cache", str(cache), "--hvp")
    with pytest.raises(SystemExit, match="1"):
        benchmark.main()
    record = torch.load(next(cache.glob("*.pt")), weights_only=True)
    assert set(record["tensors"]) == {"value", "vjp"}


def test_hvp_failure_keeps_completed_modes_and_checkpoint(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    original = RollingLogSignature.__call__
    checkpoints = []

    class FirstOrderOnly(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, seed):
            if torch.is_grad_enabled():
                checkpoints.append(json.loads(output.read_text()))
                raise RuntimeError("intentional higher-order limitation")
            return seed

    def evaluate(self, increments, **kwargs):
        return FirstOrderOnly.apply(original(self, increments, **kwargs))

    monkeypatch.setattr(RollingLogSignature, "__call__", evaluate)
    small_benchmark(monkeypatch, output, "--hvp")
    with pytest.raises(SystemExit, match="1"):
        benchmark.main()
    report = json.loads(output.read_text())
    case = report["cases"][0]
    entry = case["candidates"]["typed_dense"]
    assert report["complete"] and entry["verified"]
    assert entry["verified_modes"] == {"0": True, "1": True, "2": False}
    assert "forward" in entry and "forward_vjp" in entry and "forward_hvp" not in entry
    assert "intentional higher-order" in entry["mode_failures"]["2"]["verification"]
    assert case["fastest_forward"] == case["fastest_forward_vjp"] == "typed_dense"
    assert case["fastest_forward_hvp"] is None
    partial = checkpoints[0]
    assert not partial["complete"]
    retained = partial["cases"][0]["candidates"]["typed_dense"]
    assert retained["timed_modes"] == {"0": True, "1": True}
    assert retained["active_phase"] == {"mode": "2", "phase": "verification"}


def test_vjp_failure_does_not_hide_inference(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    original = RollingLogSignature.__call__

    class NoBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, seed):
            raise RuntimeError("intentional VJP limitation")

    monkeypatch.setattr(
        RollingLogSignature,
        "__call__",
        lambda self, x, **kwargs: NoBackward.apply(original(self, x, **kwargs)),
    )
    small_benchmark(monkeypatch, output)
    with pytest.raises(SystemExit, match="1"):
        benchmark.main()
    case = json.loads(output.read_text())["cases"][0]
    entry = case["candidates"]["typed_dense"]
    assert entry["verified_modes"] == {"0": True, "1": False}
    assert entry["timed_modes"] == {"0": True}
    assert case["fastest_forward"] == "typed_dense"
    assert case["fastest_forward_vjp"] is None


def test_inference_timing_failure_preserves_training(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    original = RollingLogSignature.__call__
    inference_calls = 0

    def evaluate(self, inputs, **kwargs):
        nonlocal inference_calls
        if not torch.is_grad_enabled():
            inference_calls += 1
            if inference_calls > 1:
                raise RuntimeError("inference warmup failed")
        return original(self, inputs, **kwargs)

    monkeypatch.setattr(RollingLogSignature, "__call__", evaluate)
    small_benchmark(monkeypatch, output)
    with pytest.raises(SystemExit, match="1"):
        benchmark.main()
    case = json.loads(output.read_text())["cases"][0]
    entry = case["candidates"]["typed_dense"]
    assert entry["verified_modes"] == {"0": True, "1": True}
    assert entry["timed_modes"] == {"0": False, "1": True}
    assert case["fastest_forward"] is None
    assert case["fastest_forward_vjp"] == "typed_dense"


def test_profile_failure_and_rounded_oracle_inputs(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    original = benchmark.reference_hvp_in_chunks
    seen = []

    def oracle(plan, increments, weights, direction, **kwargs):
        assert torch.equal(weights, weights.float().double())
        assert torch.equal(direction, direction.float().double())
        seen.append(True)
        return original(plan, increments, weights, direction, **kwargs)

    def unavailable(*args, **kwargs):
        raise RuntimeError("profiler unavailable")

    monkeypatch.setattr(benchmark, "reference_hvp_in_chunks", oracle)
    monkeypatch.setattr(benchmark, "profile_call", unavailable)
    small_benchmark(monkeypatch, output, "--profile", "--hvp", "--dtype", "float32")
    with pytest.raises(SystemExit, match="1"):
        benchmark.main()
    entry = json.loads(output.read_text())["cases"][0]["candidates"]["typed_dense"]
    assert seen and entry["verified_modes"] == {"0": True, "1": True, "2": True}
    assert entry["timed_modes"] == {"0": True, "1": True, "2": True}
    assert all("profile" in failures for failures in entry["mode_failures"].values())


def test_profile_records_higher_order_saved_storage():
    from benchmarks._rolling_profile import (
        profile_call,
    )

    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    def evaluate(value):
        return value.square() + value

    profile = profile_call(evaluate, x, torch.ones_like(x), torch.ones_like(x))
    assert profile["derivative_order"] == 2
    assert profile["saved_tensor_references"] > 0
    assert profile["saved_unique_storage_bytes"] >= x.numel() * x.element_size()
    inference = profile_call(evaluate, x, torch.ones_like(x), derivative_order=0)
    assert inference["saved_tensor_references"] == 0


def test_window_error_vector_for_drift_and_nonfinite_values():
    expected = torch.zeros(2, 3, 2)
    actual = expected.clone()
    actual[0, 1, 1] = 0.5
    actual[1, 1, 0] = 1
    assert benchmark.window_scaled_errors(actual, expected, atol=0.5, rtol=0) == [
        0.0,
        2.0,
        0.0,
    ]
    actual[0, 2, 0] = float("nan")
    assert benchmark.window_scaled_errors(actual, expected, atol=0.5, rtol=0)[2] is None
    empty = torch.zeros(2, 3, 0)
    assert benchmark.window_scaled_errors(empty, empty, atol=0.5, rtol=0) == [0.0] * 3


@pytest.mark.parametrize("order", [0, 1, 2])
def test_light_profile_preserves_saved_storage_without_operator_trace(
    order, monkeypatch
):
    from benchmarks._rolling_profile import (
        profile_call,
    )

    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    weights, direction = torch.ones_like(x), torch.ones_like(x)

    def evaluate(value):
        return value.square() + value

    traced = profile_call(
        evaluate,
        x,
        weights,
        direction if order == 2 else None,
        derivative_order=order,
        trace_operators=True,
    )

    def forbidden_trace(*args, **kwargs):
        raise AssertionError("light saved-storage measurement entered operator tracing")

    monkeypatch.setattr(torch.profiler, "profile", forbidden_trace)
    light = profile_call(
        evaluate,
        x,
        weights,
        direction if order == 2 else None,
        derivative_order=order,
    )
    assert not light["operator_trace_enabled"] and traced["operator_trace_enabled"]
    assert traced["cpu_operator_events"] > 0
    for name in (
        "saved_tensor_references",
        "saved_tensor_logical_bytes",
        "saved_unique_storage_bytes",
    ):
        assert light[name] == traced[name]
    for name in (
        "cpu_operator_events",
        "cpu_positive_self_allocation_bytes",
        "cuda_device_events",
        "cuda_kernel_events",
        "gpu_launch_count",
        "dram_bytes",
    ):
        assert light[name] is None


def test_fused_hvp_is_an_explicit_capability_refusal(tmp_path, monkeypatch):
    output = tmp_path / "report.json"

    def candidates(compiled, args):
        return {
            "fused_chunk_8": lambda x: compiled(x, backend="typed", chunk_windows=8)
        }

    def unexpected_oracle(*args, **kwargs):
        raise AssertionError("unsupported HVP must not compute the second-order oracle")

    monkeypatch.setattr(benchmark, "make_candidates", candidates)
    monkeypatch.setattr(benchmark, "reference_hvp_in_chunks", unexpected_oracle)
    small_benchmark(monkeypatch, output, "--hvp", "--fused", "--only", "fused_chunk_8")
    with pytest.raises(SystemExit, match="1"):
        benchmark.main()
    entry = json.loads(output.read_text())["cases"][0]["candidates"]["fused_chunk_8"]
    assert entry["verified_modes"] == {"0": True, "1": True, "2": False}
    assert entry["timed_modes"] == {"0": True, "1": True, "2": False}
    assert "unsupported derivative_order=2" in entry["mode_failures"]["2"]["capability"]
    assert "2" not in entry["phase_seconds"]


@pytest.mark.parametrize("inference_only", [False, True])
def test_auto_benchmark_passes_actual_derivative_order_and_records_selection(
    tmp_path, monkeypatch, inference_only
):
    from tensorsig.offset_coded_signatures.windows import WindowSelection

    seen = []

    class PublicWrapper:
        def __init__(self, plan, window):
            self.rolling = RollingLogSignature(plan, window)

        def select(self, inputs, *, stride, derivative_order):
            return WindowSelection("tensor", derivative_order, reason="test policy")

        def __call__(self, inputs, *, stride, derivative_order):
            seen.append(derivative_order)
            return self.rolling.reference(inputs, stride=stride, method="tensor")

    monkeypatch.setattr(benchmark, "WindowLogSignature", PublicWrapper)
    output = tmp_path / "auto.json"
    small_benchmark(
        monkeypatch,
        output,
        "--profile",
        "--only",
        "auto",
        *([] if inference_only else ["--hvp"]),
    )
    if inference_only:
        monkeypatch.setattr(
            sys, "argv", [arg for arg in sys.argv if arg != "--backward"]
        )
    benchmark.main()
    entry = json.loads(output.read_text())["cases"][0]["candidates"]["auto"]
    modes = (0,) if inference_only else (0, 1, 2)
    assert entry["verified_modes"] == {str(mode): True for mode in modes}
    assert entry["timed_modes"] == {str(mode): True for mode in modes}
    assert set(entry["selection_by_mode"]) == {str(mode) for mode in modes}
    assert seen == sorted(seen) and set(seen) == set(modes)
    for mode in modes:
        assert seen.count(mode) >= 4  # verification, warmup, timing and profile
        assert entry["selection_by_mode"][str(mode)]["derivative_order"] == mode
        assert entry["selection_by_mode"][str(mode)]["algorithm"] == "tensor"
        assert entry["profiles"][str(mode)]["derivative_order"] == mode


def test_auto_second_order_checks_its_own_linear_vjp(tmp_path, monkeypatch):
    from tensorsig.offset_coded_signatures.windows import WindowSelection

    class WrongVJP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, seed):
            return seed * 1.01

    class PublicWrapper:
        def __init__(self, plan, window):
            self.rolling = RollingLogSignature(plan, window)

        def select(self, inputs, *, stride, derivative_order):
            return WindowSelection("tensor", derivative_order, reason="test policy")

        def __call__(self, inputs, *, stride, derivative_order):
            value = self.rolling.reference(inputs, stride=stride, method="tensor")
            return WrongVJP.apply(value) if derivative_order == 2 else value

    monkeypatch.setattr(benchmark, "WindowLogSignature", PublicWrapper)
    output = tmp_path / "auto.json"
    small_benchmark(monkeypatch, output, "--hvp", "--only", "auto")
    with pytest.raises(SystemExit, match="1"):
        benchmark.main()
    entry = json.loads(output.read_text())["cases"][0]["candidates"]["auto"]
    assert entry["verified_modes"] == {"0": True, "1": True, "2": False}
    assert not entry["second_order_vjp_error"]["passed"]
    assert "second_order_value_error" not in entry
    assert "linear-loss VJP tolerance" in entry["mode_failures"]["2"]["verification"]


@pytest.mark.parametrize("call_seconds,expected_warmup_calls", [(0.0625, 2), (0.25, 1)])
def test_timing_has_minimum_warm_interval_separate_from_samples(
    monkeypatch, call_seconds, expected_warmup_calls
):
    clock = [0.0]
    calls, synchronizations = [], []

    def call():
        calls.append(clock[0])
        clock[0] += call_seconds

    def sync():
        synchronizations.append(clock[0])

    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    result = benchmark.time_call(call, sync=sync, repeats=3)
    assert result["warmup_calls"] == expected_warmup_calls
    assert result["warmup_ms"] == expected_warmup_calls * call_seconds * 1000
    assert result["warmup_ms"] >= 100
    assert result["samples_ms"] == [call_seconds * 1000] * 3
    assert result["median_ms"] == call_seconds * 1000
    assert len(calls) == len(synchronizations) == expected_warmup_calls + 3


def test_partial_hvp_resume_is_bitwise_and_never_returns_incomplete(
    tmp_path, monkeypatch
):
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 2), 3
    )
    rolling = RollingLogSignature(plan, 4)
    x = benchmark.make_increments((1, 12, 2), seed=71, scale=0.01)
    weights = torch.ones(1, 5, plan.dimension, dtype=torch.float64)
    direction = torch.linspace(-1, 1, x.numel(), dtype=x.dtype).reshape_as(x)
    expected = benchmark.reference_hvp_in_chunks(
        rolling, x, weights, direction, stride=2, chunk_windows=2
    )
    path, progress, metadata = (
        tmp_path / "oracle.hvp-partial.pt",
        {},
        {"source": "exact"},
    )

    def interrupt():
        if progress["completed_windows"] == 2:
            raise RuntimeError("stop after a atomically saved prefix")

    with pytest.raises(RuntimeError, match="atomically saved prefix"):
        benchmark.reference_hvp_in_chunks(
            rolling,
            x,
            weights,
            direction,
            stride=2,
            chunk_windows=2,
            partial_path=path,
            metadata=metadata,
            progress=progress,
            report_checkpoint=interrupt,
        )
    partial = torch.load(path, weights_only=True)
    assert partial["next_window"] == 2
    assert progress["resumed_from_window"] == 0
    assert progress["resumed_compute_seconds"] == 0
    # The prefix of two stride-2 windows reaches input index6, not index4.
    assert torch.count_nonzero(partial["gradient"][:, 4:6]) > 0
    assert torch.count_nonzero(partial["gradient"][:, 6:]) == 0
    resumed = {}
    actual = benchmark.reference_hvp_in_chunks(
        rolling,
        x,
        weights,
        direction,
        stride=2,
        chunk_windows=2,
        partial_path=path,
        metadata=metadata,
        progress=resumed,
    )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert resumed["resumed_from_window"] == 2
    assert (
        resumed["resumed_compute_seconds"] == partial["completed_chunk_compute_seconds"]
    )
    assert resumed["completed_windows"] == 5
    assert path.exists()  # Kept until the caller atomically promotes a full oracle.

    def unexpected(*args, **kwargs):
        raise AssertionError("a complete checkpoint must not repeat oracle work")

    monkeypatch.setattr(benchmark, "value_gradient_hvp", unexpected)
    complete = {}
    reused = benchmark.reference_hvp_in_chunks(
        rolling,
        x,
        weights,
        direction,
        stride=2,
        chunk_windows=2,
        partial_path=path,
        metadata=metadata,
        progress=complete,
    )
    assert complete["resumed_from_window"] == 5
    assert complete["new_completed_chunk_compute_seconds"] == 0
    for value, reference in zip(reused, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "corruption",
    [
        "record",
        "metadata",
        "layout",
        "prefix",
        "shape",
        "suffix",
        "nonfinite",
        "timing",
    ],
)
def test_partial_hvp_rejects_invalid_checkpoint(tmp_path, corruption):
    from types import SimpleNamespace

    x = torch.zeros(1, 12, 2, dtype=torch.float64)
    weights = torch.zeros(1, 5, 1, dtype=torch.float64)
    path, metadata = tmp_path / "partial.pt", {"source": "exact"}
    record = {
        "metadata": metadata,
        "layout": {"window": 4, "stride": 2, "chunk_windows": 2, "windows": 5},
        "next_window": 2,
        "completed_chunk_compute_seconds": 0.25,
        "gradient": torch.zeros_like(x),
        "hvp": torch.zeros_like(x),
    }
    if corruption == "record":
        record = []
    elif corruption == "metadata":
        record["metadata"] = {"source": "stale"}
    elif corruption == "layout":
        record["layout"]["chunk_windows"] = 1
    elif corruption == "prefix":
        record["next_window"] = 1
    elif corruption == "shape":
        record["gradient"] = torch.zeros(1, dtype=torch.float64)
    elif corruption == "suffix":
        record["gradient"][:, 6] = 1
    elif corruption == "nonfinite":
        record["hvp"][0, 0, 0] = float("nan")
    else:
        record["completed_chunk_compute_seconds"] = -1
    torch.save(record, path)
    with pytest.raises(ValueError, match="partial HVP checkpoint"):
        benchmark.reference_hvp_in_chunks(
            SimpleNamespace(window=4),
            x,
            weights,
            x,
            stride=2,
            chunk_windows=2,
            partial_path=path,
            metadata=metadata,
        )


def test_sized_tensor_benchmark_binds_each_mode_and_reports_actual_chunk(
    tmp_path, monkeypatch
):
    original = benchmark.WindowLogSignature
    seen = []

    class TrackedPublicWrapper(original):
        def __call__(self, inputs, **options):
            seen.append((options["derivative_order"], options["chunk_windows"]))
            return super().__call__(inputs, **options)

    monkeypatch.setattr(benchmark, "WindowLogSignature", TrackedPublicWrapper)
    monkeypatch.setattr(
        benchmark,
        "_tensor_sized_chunk_windows",
        lambda plan, window, batch, count, mode: 1 if mode == 2 else 2,
    )
    output = tmp_path / "sized.json"
    small_benchmark(monkeypatch, output, "--hvp", "--profile", "--only", "tensor_sized")
    benchmark.main()
    entry = json.loads(output.read_text())["cases"][0]["candidates"]["tensor_sized"]
    assert (
        entry["verified_modes"]
        == entry["timed_modes"]
        == {"0": True, "1": True, "2": True}
    )
    assert set(seen) == {(0, 2), (1, 2), (2, 1)}
    for mode in range(3):
        selected = entry["selection_by_mode"][str(mode)]
        assert selected["algorithm"] == "tensor"
        assert selected["derivative_order"] == mode
        assert selected["chunk_windows"] == (1 if mode == 2 else 2)
        assert entry["profiles"][str(mode)]["derivative_order"] == mode
    assert entry["second_order_vjp_error"]["passed"]


def test_independent_benchmark_candidates_share_the_public_evaluator(monkeypatch):
    from types import SimpleNamespace

    created, calls = [], []

    def construct(plan, window):
        created.append((plan, window))

        def evaluate(inputs, **options):
            calls.append(options)
            return inputs

        return evaluate

    monkeypatch.setattr(benchmark, "WindowLogSignature", construct)
    compiled = SimpleNamespace(plan=object(), window=3)
    args = SimpleNamespace(
        stride=2,
        hvp=True,
        baseline_chunk_windows=4,
        chunks=[],
        fused=False,
        refresh_every=None,
    )
    candidates = benchmark.make_candidates(compiled, args)
    token = object()
    for method in ("direct", "pbw", "tensor", "active"):
        assert candidates[method](token) is token
        assert candidates[f"{method}_batched"](token) is token
        assert calls[-2] == dict(algorithm=method, stride=2, derivative_order=2)
        assert calls[-1] == dict(
            algorithm=method, stride=2, derivative_order=2, chunk_windows=4
        )
    assert created == [(compiled.plan, 3)]
    assert candidates["auto"] is candidates["tensor_sized"]
    assert "active_rolling" not in candidates
    assert "chen" not in candidates


def test_tensor_control_keeps_its_explicit_experimental_sizing():
    from types import SimpleNamespace

    plan = SimpleNamespace(alphabet=ProjectedAlphabet.newton(2, 16), max_depth=3)
    assert benchmark._tensor_sized_chunk_windows(plan, 8, 1, 4096, 0) == 512
    assert benchmark._tensor_sized_chunk_windows(plan, 8, 1, 4096, 1) == 512
    assert benchmark._tensor_sized_chunk_windows(plan, 8, 1, 4096, 2) == 256
    plan = SimpleNamespace(alphabet=ProjectedAlphabet.newton(2, 8), max_depth=3)
    assert benchmark._tensor_sized_chunk_windows(plan, 8, 1, 8192, 0) == 4096
