"""Joint circuits and optimized scan adjoints."""

import pytest
import torch

from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    compile_lie_quotient,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import ProjectedAlphabet
from tensorsig.offset_coded_signatures.rolling_logsignature import (
    RollingLogSignature,
)
from tensorsig.offset_coded_signatures.rolling_logsignature._scan import (
    adjoint_affine_scan,
)
from tensorsig.offset_coded_signatures.quotients import (
    ChannelMultiplicity,
    DependenceGraph,
    LieQuotient,
    TemporalDegree,
)


def make_plan(depth=3, graph=False, width=2, channels=2, window=3):
    rules = [TemporalDegree(width - 1)]
    if graph:
        rules.append(DependenceGraph(tuple((i, i + 1) for i in range(channels - 1))))
    plan = compile_lie_quotient(
        LieQuotient(*rules), ProjectedAlphabet.newton(width, channels), depth
    )
    return RollingLogSignature(plan.to(dtype=torch.float64), window)


@pytest.mark.parametrize("algorithm", ["chunks", "scan"])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "stride,chunk,refresh", [(1, None, 0), (2, 4, 3), (4, 3, None)]
)
def test_joint_algorithms_values_vjp_hvp(algorithm, graph, stride, chunk, refresh):
    rolling = make_plan(depth=4, graph=graph, channels=3 if graph else 2, window=5)
    torch.manual_seed(125)
    x = (
        torch.randn(1, 19, rolling.plan.alphabet.in_channels, dtype=torch.float64)
        * 0.015
    ).requires_grad_()
    value = rolling(
        x,
        algorithm=algorithm,
        backend="joint",
        stride=stride,
        chunk_windows=chunk,
        refresh_every=refresh,
        derivative_order=2,
    )
    reference = rolling.reference(x, stride=stride, method="tensor")
    weights = torch.randn_like(value)
    direction = torch.randn_like(x)

    def derivatives(y):
        loss = (weights * y.sin() + 0.3 * y.square()).sum()
        grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        hvp = torch.autograd.grad((grad * direction).sum(), x)[0]
        return grad, hvp

    actual, expected = derivatives(value), derivatives(reference)
    torch.testing.assert_close(value, reference, atol=1e-11, rtol=1e-10)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("algorithm", ["chunks", "scan"])
def test_algorithm_gradgradcheck(algorithm):
    rolling = make_plan()
    torch.manual_seed(312)
    x = (torch.randn(1, 6, 2, dtype=torch.float64) * 0.02).requires_grad_()

    def evaluate(value):
        return rolling(
            value,
            algorithm=algorithm,
            backend="joint",
            derivative_order=2,
            refresh_every=0,
        )

    assert torch.autograd.gradcheck(evaluate, (x,), fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, (x,), fast_mode=True)


def test_linear_scan_transposed_adjoint_and_gradgrad():
    matrix = torch.tensor([[1.0, -0.2], [0.0, 1.0]], dtype=torch.float64)
    powers = tuple(torch.linalg.matrix_power(matrix, 2**i) for i in range(3))
    forcing = torch.randn(1, 5, 2, dtype=torch.float64, requires_grad=True)
    initial = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)

    def function(b, x):
        return adjoint_affine_scan(b, x, powers)

    assert torch.autograd.gradgradcheck(function, (forcing, initial), fast_mode=True)
    values, state = [], initial
    for b in forcing.unbind(1):
        state = state @ matrix + b
        values.append(state)
    torch.testing.assert_close(function(forcing, initial), torch.stack(values, 1))


def test_joint_forcing_prunes_inputs_and_matches_composition():
    rolling = make_plan(depth=4, width=3, window=4)
    torch.manual_seed(63)
    state = torch.randn(2, rolling.plan.dimension, dtype=torch.float64) * 0.02
    outgoing = torch.randn(2, 2, dtype=torch.float64) @ rolling.embeddings[0]
    incoming = torch.randn(2, 2, dtype=torch.float64) @ rolling.embeddings[-1]
    joint = rolling._joint_program()
    expected = rolling.append(
        rolling.transport(-rolling.append(-state, outgoing), "dense"), incoming
    )
    torch.testing.assert_close(joint(state, outgoing, incoming), expected)
    for grade in range(1, 5):
        program = rolling._joint_program(grade)
        lower = state * (rolling.plan.basis_depths < grade)
        assert torch.all(
            rolling.plan.basis_depths[program.index_buffer[program.input_slices[0]]]
            < grade
        )
        indices = rolling.plan.basis_depths == grade
        torch.testing.assert_close(
            program(state, outgoing, incoming),
            joint(lower, outgoing, incoming)[..., indices],
        )
        assert program.row_count <= joint.row_count


def test_compiled_polynomial_has_only_two_metadata_buffers(monkeypatch):
    original_compile = torch.compile
    captures = []

    def backend(graph, inputs):
        captures.append([value for value in inputs if isinstance(value, torch.Tensor)])
        return graph.forward

    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: original_compile(fn, backend=backend, **kwargs),
    )
    rolling = make_plan()
    program = rolling._joint_program(compiled=True)
    inputs = tuple(
        torch.randn(2, rolling.plan.dimension, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    )
    value = program(*inputs)
    gradient = torch.autograd.grad(value.square().sum(), inputs, create_graph=True)
    second = torch.autograd.grad(sum(g.sum() for g in gradient), inputs)
    assert all(torch.isfinite(g).all() for g in second)
    assert len(captures) == 3
    for tensors, dynamic_count in zip(captures, (3, 4, 7), strict=True):
        assert len(tensors) == dynamic_count + 2
        assert sum(t.dtype == torch.long for t in tensors) == 1
        assert any(t.data_ptr() == program.index_buffer.data_ptr() for t in tensors)
        assert any(
            t.data_ptr() == program.coefficient_buffer.data_ptr() for t in tensors
        )


@pytest.mark.parametrize("algorithm", ["chunks", "scan"])
def test_empty_one_window_and_one_increment(algorithm):
    plan = compile_lie_quotient(
        LieQuotient(ChannelMultiplicity(0)), ProjectedAlphabet.newton(2, 2), 3
    )
    rolling = RollingLogSignature(plan, 3)
    x = torch.randn(1, 8, 2, dtype=torch.float64, requires_grad=True)
    result = rolling(x, algorithm=algorithm, backend="joint", derivative_order=2)
    assert result.shape == (1, 6, 0)
    torch.testing.assert_close(
        torch.autograd.grad(result.sum(), x)[0], torch.zeros_like(x)
    )
    rolling = make_plan(width=1, channels=1, window=1)
    x = torch.randn(2, 7, 1, dtype=torch.float64)
    torch.testing.assert_close(rolling(x, algorithm=algorithm, backend="joint"), x)
    rolling = make_plan(window=3)
    x = torch.randn(1, 3, 2, dtype=torch.float64) * 0.02
    torch.testing.assert_close(
        rolling(x, algorithm=algorithm, backend="joint"), rolling.reference(x)
    )


@pytest.mark.parametrize("backend", ["generic", "packed", "endpoint", "bernoulli"])
def test_removed_backend_aliases_are_rejected(backend):
    rolling = make_plan()
    x = torch.zeros(1, 8, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="backend"):
        rolling(x, backend=backend)


def test_removed_reference_scan_api_is_rejected():
    rolling = make_plan()
    x = torch.zeros(1, 8, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="algorithm"):
        rolling(x, algorithm="scan_reference")
    with pytest.raises(TypeError, match="optimized"):
        rolling.scan(x, optimized=False)


def test_scan_saves_no_doubling_levels():
    rolling = make_plan()
    x = (torch.randn(1, 34, 2, dtype=torch.float64) * 0.01).requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(t.shape) or t, lambda t: t
    ):
        rolling(x, algorithm="scan", backend="joint", refresh_every=0)
    # One final trajectory, not one full trajectory per grade/scan level.
    assert saved.count(torch.Size((1, 32, rolling.plan.dimension))) == 1


@pytest.mark.parametrize("depth", [5, 6])
def test_exact_joint_schedule_above_closed_form(depth):
    rolling = make_plan(depth=depth, channels=1)
    x = (torch.randn(1, 7, 1, dtype=torch.float64) * 0.03).requires_grad_()
    torch.testing.assert_close(
        rolling(x, backend="joint"), rolling.reference(x), atol=1e-11, rtol=1e-10
    )


@pytest.mark.parametrize("algorithm", ["chunks", "scan"])
@pytest.mark.parametrize("backend", ["joint", "joint_fused"])
def test_hessian_symmetry_and_mixed_cotangent_derivative(
    algorithm, backend, monkeypatch
):
    from tensorsig.offset_coded_signatures.rolling_logsignature import (
        _scan,
    )

    captures = []
    if backend == "joint_fused":
        original_compile = torch.compile

        def capture_graph(graph, inputs):
            captures.append(graph)
            return graph.forward

        # Exercise all owned derivative graphs without invoking native compilation.
        monkeypatch.setattr(
            torch,
            "compile",
            lambda fn, **kwargs: original_compile(fn, backend=capture_graph, **kwargs),
        )

    scan_calls = {"forward": 0, "backward": 0}
    for name in scan_calls:
        original = getattr(_scan._GradeScan, name)

        def counted(*args, _name=name, _original=original):
            scan_calls[_name] += 1
            return _original(*args)

        monkeypatch.setattr(_scan._GradeScan, name, staticmethod(counted))

    rolling = make_plan(depth=3, graph=True, channels=3, window=3)
    generator = torch.Generator().manual_seed(721)
    x = (
        torch.randn(1, 9, 3, generator=generator, dtype=torch.float64) * 0.02
    ).requires_grad_()
    u = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    v = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    cotangent = torch.randn(
        1,
        7,
        rolling.plan.dimension,
        generator=generator,
        dtype=x.dtype,
        requires_grad=True,
    )

    def derivatives(evaluate):
        value = evaluate(x)
        loss = (value * cotangent + value.square() / 2).sum()
        gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
        hu, mixed = torch.autograd.grad(
            (gradient * u).sum(), (x, cotangent), retain_graph=True
        )
        hv = torch.autograd.grad((gradient * v).sum(), x)[0]
        return value, gradient, hu, hv, mixed

    actual = derivatives(
        lambda value: rolling(
            value,
            algorithm=algorithm,
            backend=backend,
            derivative_order=2,
            chunk_windows=4,
            refresh_every=0,
        )
    )
    expected = derivatives(lambda value: rolling.reference(value, method="tensor"))
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, atol=1e-9, rtol=1e-8)
    _, _, hu, hv, _ = actual
    torch.testing.assert_close((u * hv).sum(), (v * hu).sum(), atol=1e-9, rtol=1e-8)
    if algorithm == "scan":
        assert scan_calls["forward"] > 0
        assert scan_calls["backward"] > 0
    if backend == "joint_fused":
        assert len(captures) >= 3  # Forward, VJP and differentiated-VJP graphs.


@pytest.mark.parametrize(
    "algorithm,expected_backend",
    [("chunks", "typed"), ("scan", "joint")],
)
def test_algorithm_default_backend_is_explicitly_dispatched(
    monkeypatch, algorithm, expected_backend
):
    rolling = make_plan()
    x = torch.linspace(-0.02, 0.04, 16, dtype=torch.float64).reshape(1, 8, 2)
    seen = []
    if algorithm == "chunks":
        original = rolling._endpoints

        def record(backend):
            seen.append(backend)
            return original(backend)

        monkeypatch.setattr(rolling, "_endpoints", record)
    else:
        original = getattr(rolling, algorithm)

        def record(value, **options):
            seen.append(options["backend"])
            return original(value, **options)

        monkeypatch.setattr(rolling, algorithm, record)
    actual = rolling(x, algorithm=algorithm)
    torch.testing.assert_close(
        actual, rolling.reference(x, method="tensor"), atol=1e-9, rtol=1e-8
    )
    assert seen and set(seen) == {expected_backend}


@pytest.mark.parametrize("algorithm", ["active", "active_rolling", "chen"])
def test_retired_schedules_are_rejected(algorithm):
    rolling = make_plan()
    x = torch.zeros(1, 8, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="algorithm"):
        rolling(x, algorithm=algorithm)
    assert not hasattr(rolling, "chen")


@pytest.mark.parametrize("backend", ["typed", "joint", "joint_fused"])
@pytest.mark.parametrize("entry", ["method", "call"])
def test_scan_rejects_unimplemented_sparse_transport(backend, entry):
    rolling = make_plan()
    x = torch.zeros(1, 6, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="Affine scan supports only.*dense"):
        if entry == "method":
            rolling.scan(x, backend=backend, shift_backend="sparse")
        else:
            rolling(x, algorithm="scan", backend=backend, shift_backend="sparse")


@pytest.mark.parametrize("backend", ["joint", "joint_fused"])
def test_joint_chunks_reject_unimplemented_sparse_transport(backend):
    rolling = make_plan()
    x = torch.zeros(1, 6, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="Joint chunks support only.*dense"):
        rolling(x, backend=backend, shift_backend="sparse")
