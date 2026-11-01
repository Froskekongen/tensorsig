"""The public stream contract shares existing algebra and derivative kernels."""

import pytest
import torch

from tensorsig.offset_coded_signatures.quotients import (
    LieQuotient,
    OffsetAlphabet,
    ProjectedAlphabet,
    TemporalDegree,
    WindowLogSignature,
    compile_lie_quotient,
)


@pytest.mark.parametrize("algorithm", ["auto", "pbw", "active", "rolling", "scan"])
def test_stream_algorithms_and_second_derivatives(algorithm):
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 2), 3
    )
    runtime = WindowLogSignature(plan, 5)
    x = (torch.randn(2, 14, 2, dtype=torch.float64) * 0.01).requires_grad_()
    expected = runtime(x, algorithm="tensor", stride=2)
    actual = runtime(
        x,
        algorithm=algorithm,
        stride=2,
        derivative_order=2,
        **({} if algorithm == "auto" else {"chunk_windows": 3}),
    )
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
    assert actual.shape == (2, 5, plan.dimension)
    direction = torch.randn_like(x)

    def hvp(value):
        gradient = torch.autograd.grad(value.sin().sum(), x, create_graph=True)[0]
        return torch.autograd.grad((gradient * direction).sum(), x, retain_graph=True)[
            0
        ]

    torch.testing.assert_close(hvp(actual), hvp(expected), atol=1e-9, rtol=1e-8)
    selection = runtime.select(x, algorithm=algorithm, derivative_order=2)
    assert selection.derivative_order == 2
    assert selection == runtime.select(x * 0, algorithm=algorithm, derivative_order=2)


def test_arbitrary_code_gradients_and_explicit_capability_errors():
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 2), 3
    )
    code = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    runtime = WindowLogSignature(plan, 3, offset_code=code)
    x = (torch.randn(1, 6, 2, dtype=torch.float64) * 0.01).requires_grad_()
    actual, expected = runtime(x, algorithm="active"), runtime(x, algorithm="tensor")
    for left, right in zip(
        torch.autograd.grad(actual.square().sum(), (x, code)),
        torch.autograd.grad(expected.square().sum(), (x, code)),
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-8)
    with pytest.raises(ValueError, match="certified"):
        runtime(x, algorithm="scan")
    with pytest.raises(ValueError, match="derivative_order"):
        runtime(x, derivative_order=True)
    for algorithm in ("made-up", "scan_reference", "chen", "active_rolling"):
        with pytest.raises(ValueError, match="Unknown"):
            runtime(x, algorithm=algorithm)


def test_offset_alphabet_and_explicit_backend_validation():
    plan = compile_lie_quotient(LieQuotient(), OffsetAlphabet(2, 1), 3)
    runtime = WindowLogSignature(plan, 2)
    x = torch.randn(1, 5, 1, dtype=torch.float64)
    torch.testing.assert_close(
        runtime(x, algorithm="active"), runtime(x, algorithm="tensor")
    )
    with pytest.raises(ValueError, match="backend"):
        runtime(x, algorithm="tensor", backend="joint")


@pytest.mark.parametrize("algorithm", ["rolling", "scan"])
def test_selection_rejects_unknown_backend_before_constructing_runtime(algorithm):
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 1), 2
    )
    runtime = WindowLogSignature(plan, 3)
    with pytest.raises(ValueError, match="backend must be"):
        runtime.select(torch.zeros(1, 5, 1), algorithm=algorithm, backend="made-up")
    assert not runtime._rolling


def test_inspecting_generated_code_cannot_break_pascal_certification():
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 1), 2
    )
    runtime = WindowLogSignature(plan, 3)
    x = torch.randn(1, 7, 1, dtype=torch.float64)
    runtime.offset_code[:, 1].mul_(2)
    runtime.offset_code.requires_grad_()
    torch.testing.assert_close(
        runtime(x, algorithm="tensor"), runtime(x, algorithm="rolling")
    )


def test_generated_pascal_code_is_cached_by_device_and_dtype(monkeypatch):
    from tensorsig.offset_coded_signatures import window_logsignature as module

    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 1), 2
    )
    runtime = WindowLogSignature(plan, 3)
    seen = []

    def capture(windows, plan, code, *, method):
        seen.append(code)
        return windows[..., 0, :]

    monkeypatch.setattr(
        module, "quotient_log_signature_from_increment_windows", capture
    )
    for dtype in (torch.float32, torch.float32, torch.float64, torch.float32):
        runtime(torch.zeros(1, 5, 1, dtype=dtype), algorithm="tensor")
    assert seen[0] is seen[1] is seen[3]
    assert seen[2] is not seen[0]
    assert seen[0].dtype == torch.float32 and seen[2].dtype == torch.float64
    assert all(not code.requires_grad for code in seen)
    runtime.offset_code.mul_(0)
    torch.testing.assert_close(
        seen[0], torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    )


def test_custom_code_is_reconverted_after_updates_with_connected_gradients(monkeypatch):
    from tensorsig.offset_coded_signatures import window_logsignature as module

    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 1), 2
    )
    code = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=torch.float64, requires_grad=True
    )
    runtime = WindowLogSignature(plan, 3, offset_code=code)
    inputs = torch.full((1, 5, 1), 0.1, dtype=torch.float32, requires_grad=True)
    original, seen = module.quotient_log_signature_from_increment_windows, []

    def capture(windows, plan, converted, *, method):
        seen.append(converted)
        return original(windows, plan, converted, method=method)

    monkeypatch.setattr(
        module, "quotient_log_signature_from_increment_windows", capture
    )
    values = []
    for _ in range(2):
        value = runtime(inputs, algorithm="tensor")
        values.append(value.detach())
        gradient = torch.autograd.grad(value.square().sum(), code, create_graph=True)[0]
        second = torch.autograd.grad(gradient.sum(), code)[0]
        assert torch.isfinite(gradient).all() and torch.isfinite(second).all()
        with torch.no_grad():
            code.mul_(1.1)
    assert seen[0] is not seen[1]
    assert all(
        converted.requires_grad and converted.dtype == torch.float32
        for converted in seen
    )
    assert not torch.equal(values[0], values[1])
    assert not runtime._pascal_codes


def policy_metadata(window, windows, channels, graph, device="mps"):
    from types import SimpleNamespace
    from tensorsig.offset_coded_signatures.quotients import DependenceGraph

    rules = [TemporalDegree(1)]
    if graph:
        rules.append(DependenceGraph((i, i + 1) for i in range(channels - 1)))
    # Only metadata is required by select: no huge tensor or quotient compilation.
    runtime = object.__new__(WindowLogSignature)
    runtime.window = window
    runtime._pascal = True
    runtime._auto_graph = graph
    runtime.plan = SimpleNamespace(
        quotient=LieQuotient(*rules),
        alphabet=ProjectedAlphabet.newton(2, channels),
        max_depth=3,
    )
    inputs = SimpleNamespace(
        ndim=3,
        shape=(1, window + windows - 1, channels),
        dtype=torch.float32,
        device=torch.device(device),
    )
    return runtime, inputs


@pytest.mark.parametrize("device", ["mps", "cuda"])
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize(
    "channels,windows", [(8, 1024), (8, 4096), (16, 1024), (16, 4096), (12, 2048)]
)
def test_typed_policy_records_and_executes_the_measured_sparse_schedule(
    device, order, channels, windows
):
    runtime, inputs = policy_metadata(8, windows, channels, True, device)
    selected = runtime.select(inputs, derivative_order=order)
    assert (
        selected.algorithm,
        selected.backend,
        selected.shift_backend,
        selected.chunk_windows,
    ) == ("rolling", "typed", "sparse", 8)
    calls = []

    def record(value, **options):
        calls.append(options)
        return value

    runtime._rolling = {(inputs.device, inputs.dtype): record}
    assert runtime(inputs, derivative_order=order) is inputs
    assert calls == [
        dict(
            algorithm="chunks",
            backend="typed",
            shift_backend="sparse",
            chunk_windows=8,
            refresh_every=None,
            stride=1,
            derivative_order=order,
        )
    ]


@pytest.mark.parametrize("order", [0, 1, 2])
def test_gpu_policy_contains_only_the_declared_leaves(order):
    import itertools

    leaves = set()
    for window, windows, channels, graph in itertools.product(
        (7, 8, 9, 64),
        (1023, 1024, 4096, 4097, 16384, 16385),
        (7, 8, 12, 16, 17),
        (False, True),
    ):
        runtime, inputs = policy_metadata(window, windows, channels, graph)
        try:
            selected = runtime.select(inputs, derivative_order=order)
        except NotImplementedError:
            continue
        leaves.add(
            (
                selected.algorithm,
                selected.backend,
                selected.shift_backend,
                selected.chunk_windows,
            )
        )
    assert leaves == ({("rolling", "typed", "sparse", 8)} if order in {0, 1} else set())


@pytest.mark.parametrize(
    "window,windows,channels,graph,order",
    [
        (7, 2048, 8, True, 0),
        (9, 2048, 8, True, 0),
        (8, 1023, 8, True, 0),
        (8, 4097, 8, True, 1),
        (8, 2048, 7, True, 0),
        (8, 2048, 17, True, 0),
        (8, 2048, 12, False, 0),
        (8, 2048, 8, True, 2),
        (8, 4096, 8, False, 0),
        (8, 16385, 8, False, 0),
        (8, 8192, 9, False, 0),
        (16, 8192, 8, False, 0),
        (8, 8192, 8, False, 1),
        (8, 8192, 8, False, 2),
    ],
)
def test_gpu_auto_refuses_outside_the_typed_region(
    window, windows, channels, graph, order
):
    runtime, inputs = policy_metadata(window, windows, channels, graph)
    with pytest.raises(
        NotImplementedError, match="Unsupported GPU auto.*explicit algorithm"
    ):
        runtime(inputs, derivative_order=order)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_cpu_auto_stays_direct_and_explicit_gpu_choices_remain_available(order):
    runtime, inputs = policy_metadata(8, 4096, 8, True, "cpu")
    runtime._pascal = False
    runtime._auto_graph = None
    assert runtime.select(inputs, derivative_order=order).algorithm == "direct"
    runtime, inputs = policy_metadata(8, 4096, 8, True)
    assert (
        runtime.select(
            inputs, algorithm="tensor", derivative_order=order, chunk_windows=3
        ).chunk_windows
        == 3
    )
    assert (
        runtime.select(inputs, algorithm="direct", derivative_order=order).algorithm
        == "direct"
    )


def test_auto_schedule_overrides_and_explicit_transport_validation():
    runtime, inputs = policy_metadata(8, 4096, 8, True)
    for options in (
        {"chunk_windows": 3},
        {"shift_backend": "dense"},
        {"backend": "typed"},
    ):
        with pytest.raises(ValueError, match="explicit algorithm"):
            runtime.select(inputs, **options)
    with pytest.raises(ValueError, match="explicit algorithm"):
        runtime(inputs, refresh_every=1)
    selected = runtime.select(
        inputs, algorithm="rolling", backend="fused", derivative_order=0
    )
    assert selected.shift_backend == "dense"
    selected = runtime.select(
        inputs,
        algorithm="rolling",
        backend="typed",
        shift_backend="sparse",
        chunk_windows=8,
    )
    assert selected.shift_backend == "sparse"
    with pytest.raises(ValueError, match="shift_backend"):
        runtime.select(inputs, algorithm="rolling", shift_backend="unknown")


@pytest.mark.parametrize(
    "change",
    ["dtype64", "dtype16", "batch", "stride", "custom_code", "algebra", "device"],
)
def test_gpu_auto_refuses_uncalibrated_metadata_without_fallback(change):
    runtime, inputs = policy_metadata(8, 2048, 8, True)
    options = {}
    if change.startswith("dtype"):
        inputs.dtype = torch.float64 if change == "dtype64" else torch.float16
    elif change == "batch":
        inputs.shape = (2, inputs.shape[1], inputs.shape[2])
    elif change == "stride":
        options["stride"] = 2
    elif change in {"custom_code", "algebra"}:
        runtime._auto_graph = None
        runtime._pascal = change != "custom_code"
    else:
        inputs.device = torch.device("xpu")
    with pytest.raises(
        NotImplementedError, match="Unsupported GPU auto.*explicit algorithm"
    ):
        runtime(inputs, **options)


@pytest.mark.parametrize("order", [0, 1])
def test_public_typed_sparse_matches_the_named_benchmark_candidate(order):
    from types import SimpleNamespace
    from tensorsig.offset_coded_signatures.rolling_logsignature import (
        RollingLogSignature,
    )
    from benchmarks.benchmark_rolling_logsignature import make_candidates

    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 2), 3
    )
    public, rolling = WindowLogSignature(plan, 8), RollingLogSignature(plan, 8)
    args = SimpleNamespace(
        stride=1,
        hvp=False,
        baseline_chunk_windows=None,
        chunks=[8],
        fused=False,
        refresh_every=None,
    )
    candidate = make_candidates(rolling, args)["typed_sparse_chunk_8"]
    x = (
        torch.linspace(-0.03, 0.04, 46, dtype=torch.float64)
        .reshape(1, 23, 2)
        .requires_grad_()
    )
    actual = public(
        x,
        algorithm="rolling",
        backend="typed",
        shift_backend="sparse",
        chunk_windows=8,
        derivative_order=order,
    )
    expected = candidate(x)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
    if order:
        ga = torch.autograd.grad(actual.square().sum(), x)[0]
        ge = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(ga, ge, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("graph", [False, True])
def test_auto_quotient_eligibility_is_computed_once_at_construction(monkeypatch, graph):
    from types import SimpleNamespace
    from tensorsig.offset_coded_signatures import window_logsignature as module
    from tensorsig.offset_coded_signatures.quotients import DependenceGraph

    rules = [TemporalDegree(1)]
    if graph:
        rules.append(DependenceGraph([(0, 1)]))
    plan = compile_lie_quotient(LieQuotient(*rules), ProjectedAlphabet.newton(2, 2), 3)
    runtime = WindowLogSignature(plan, 8)
    assert runtime._auto_graph is graph
    custom = WindowLogSignature(plan, 8, offset_code=runtime.offset_code)
    assert custom._auto_graph is None
    inputs = SimpleNamespace(
        ndim=3, shape=(1, 39, 2), dtype=torch.float32, device=torch.device("mps")
    )

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "warm selection must not reconstruct immutable quotient metadata"
        )

    monkeypatch.setattr(module, "LieQuotient", unexpected)
    monkeypatch.setattr(module, "DependenceGraph", unexpected)
    for order in range(3):
        with pytest.raises(NotImplementedError, match="explicit algorithm"):
            runtime.select(inputs, derivative_order=order)
        with pytest.raises(NotImplementedError, match="explicit algorithm"):
            custom.select(inputs, derivative_order=order)


@pytest.mark.parametrize(
    "algorithm,backend,message",
    [
        ("scan", "typed", "Affine scan supports only"),
        ("scan", "joint", "Affine scan supports only"),
        ("scan", "joint_fused", "Affine scan supports only"),
        ("rolling", "joint", "Joint chunks support only"),
        ("rolling", "joint_fused", "Joint chunks support only"),
    ],
)
def test_selection_rejects_unimplemented_sparse_transport(algorithm, backend, message):
    runtime, inputs = policy_metadata(8, 32, 8, True)
    with pytest.raises(ValueError, match=message):
        runtime.select(
            inputs, algorithm=algorithm, backend=backend, shift_backend="sparse"
        )
    assert (
        runtime.select(inputs, algorithm=algorithm, backend=backend).shift_backend
        == "dense"
    )


@pytest.mark.parametrize("device", ["mps", "cuda"])
@pytest.mark.parametrize("windows", [4097, 6144, 8192, 12288, 16384])
def test_withdrawn_tensor_region_refuses_auto_but_preserves_explicit_tensor(
    device, windows
):
    runtime, inputs = policy_metadata(8, windows, 8, False, device)
    with pytest.raises(
        NotImplementedError, match="Unsupported GPU auto.*explicit algorithm"
    ):
        runtime(inputs, derivative_order=0)
    selected = runtime.select(
        inputs, algorithm="tensor", derivative_order=0, chunk_windows=4096
    )
    assert (
        selected.algorithm,
        selected.backend,
        selected.shift_backend,
        selected.chunk_windows,
    ) == ("tensor", None, None, 4096)


def test_explicit_tensor_dispatch_preserves_fixed_chunks(monkeypatch):
    from tensorsig.offset_coded_signatures import window_logsignature as module

    runtime, _ = policy_metadata(8, 4097, 8, False)
    # Use real CPU tensor views to verify explicit dispatch without a GPU allocation.
    runtime._offset_code = None
    calls = []

    def evaluate(windows, plan, code, *, method):
        calls.append((windows.shape[1], method, code))
        return windows[..., 0, :]

    monkeypatch.setattr(
        module, "quotient_log_signature_from_increment_windows", evaluate
    )
    values = torch.arange(4104 * 8, dtype=torch.float32).reshape(1, 4104, 8)
    actual = runtime(values, algorithm="tensor", derivative_order=0, chunk_windows=4096)
    assert calls == [(4096, "tensor", None), (1, "tensor", None)]
    torch.testing.assert_close(actual, values[:, :4097])
