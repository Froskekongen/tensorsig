"""Numerical regression checks, not certificates of general complexity or stability."""

from fractions import Fraction

import pytest
import torch

from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    compile_lie_quotient,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import ProjectedAlphabet
from tensorsig.offset_coded_signatures.rolling_logsignature import (
    RollingLogSignature,
)
from tensorsig.offset_coded_signatures.rolling_logsignature._compile import (
    pascal_transport,
)
from tensorsig.offset_coded_signatures.quotients import (
    ChannelMultiplicity,
    ChannelSupport,
    DependenceGraph,
    LieQuotient,
    TemporalDegree,
)


def experiment(depth=3, graph=False, window=4, dtype=torch.float64):
    rules = [TemporalDegree(1)]
    if graph:
        rules.append(DependenceGraph(((0, 1), (1, 2))))
    plan = compile_lie_quotient(
        LieQuotient(*rules),
        ProjectedAlphabet.newton(2, 3 if graph else 2),
        depth,
    ).to(dtype=dtype)
    return RollingLogSignature(plan, window)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_transport_exact_inverse_and_identity(graph, depth):
    plan = experiment(depth, graph).plan
    identity = pascal_transport(plan, 0)
    assert identity == [{i: Fraction(1)} for i in range(plan.dimension)]
    forward, backward = pascal_transport(plan, 1), pascal_transport(plan, -1)
    for i, row in enumerate(forward):
        result = {}
        for j, a in row.items():
            for k, b in backward[j].items():
                result[k] = result.get(k, Fraction()) + a * b
        assert {k: v for k, v in result.items() if v} == identity[i]


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6])
def test_typed_bch_matches_generic_on_arbitrary_state(depth):
    compiled = experiment(depth)
    generator = torch.Generator().manual_seed(37)
    x = (
        torch.randn(
            2, compiled.plan.dimension, generator=generator, dtype=torch.float64
        )
        * 0.1
    )
    y = torch.randn(x.shape, generator=generator, dtype=x.dtype) * 0.1
    y[:, compiled.plan.basis_depths != 1] = 0
    torch.testing.assert_close(
        compiled.append(x, y), compiled.plan._runtime.bch(x, y), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "backend,shift_backend,chunk,stride,refresh",
    [
        ("typed", "dense", None, 1, 0),
        ("typed", "dense", None, 1, None),
        ("typed", "sparse", 3, 1, 0),
        ("typed", "dense", 4, 2, None),
        ("typed", "sparse", 1, 3, 0),
        ("typed", "dense", 2, 4, 0),
        ("typed", "dense", None, 6, None),
    ],
)
def test_values_and_gradients_against_tensor_windows(
    graph, backend, shift_backend, chunk, stride, refresh
):
    compiled = experiment(4, graph)
    generator = torch.Generator().manual_seed(101)
    dx = (
        torch.randn(
            2,
            19,
            compiled.plan.alphabet.in_channels,
            generator=generator,
            dtype=torch.float64,
        )
        * 0.03
    ).requires_grad_()
    actual = compiled(
        dx,
        backend=backend,
        shift_backend=shift_backend,
        chunk_windows=chunk,
        stride=stride,
        refresh_every=refresh,
    )
    expected = compiled.reference(dx, stride=stride, method="tensor")
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
    weights = torch.randn(actual.shape, generator=generator, dtype=dx.dtype)
    actual_grad = torch.autograd.grad((actual * weights).sum(), dx)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), dx)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize(
    "quotient",
    [
        LieQuotient(),
        LieQuotient(ChannelMultiplicity(0)),
        LieQuotient(ChannelMultiplicity((0, 2)), ChannelSupport(1)),
        LieQuotient(DependenceGraph()),
    ],
)
def test_zero_abelian_and_unpruned_quotients(quotient):
    plan = compile_lie_quotient(quotient, ProjectedAlphabet.newton(2, 2), 3)
    compiled = RollingLogSignature(plan, 3)
    dx = torch.randn(1, 10, 2, dtype=torch.float64) * 0.01
    torch.testing.assert_close(
        compiled(dx, chunk_windows=4), compiled.reference(dx, method="pbw")
    )


def test_single_increment_windows_and_validation():
    plan = compile_lie_quotient(LieQuotient(), ProjectedAlphabet.newton(1, 1), 3)
    compiled = RollingLogSignature(plan, 1)
    dx = torch.randn(2, 5, 1, dtype=torch.float64)
    torch.testing.assert_close(compiled(dx, chunk_windows=2), dx)
    for kwargs in (
        {"stride": 0},
        {"chunk_windows": 0},
        {"refresh_every": -1},
        {"backend": "auto"},
        {"shift_backend": "auto"},
    ):
        with pytest.raises(ValueError):
            compiled(dx, **kwargs)
    with pytest.raises(ValueError, match="dtype"):
        compiled(dx.float())
    with pytest.raises(ValueError, match="at least"):
        compiled(dx[:, :0])
    nonnewton = compile_lie_quotient(LieQuotient(), ProjectedAlphabet(2, 1), 2)
    with pytest.raises(ValueError, match="newton"):
        RollingLogSignature(nonnewton, 4)


def test_float32_long_stream_refresh_and_noncontiguous_input():
    compiled = experiment(dtype=torch.float32)
    dx = torch.randn(1, 2, 200, dtype=torch.float32).transpose(1, 2) * 0.01
    torch.testing.assert_close(
        compiled(dx, chunk_windows=17), compiled.reference(dx), atol=2e-5, rtol=2e-4
    )


@pytest.mark.parametrize("graph", [False, True])
def test_three_temporal_letters_against_tensor_oracle(graph):
    rules = [TemporalDegree(2)]
    channels = 3 if graph else 2
    if graph:
        rules.append(DependenceGraph(((0, 1), (1, 2))))
    plan = compile_lie_quotient(
        LieQuotient(*rules), ProjectedAlphabet.newton(3, channels), 3
    )
    compiled = RollingLogSignature(plan, 5)
    dx = (torch.randn(2, 14, channels, dtype=torch.float64) * 0.01).requires_grad_()
    actual = compiled(dx, chunk_windows=4, refresh_every=0)
    expected = compiled.reference(dx, method="tensor")
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), dx)[0],
        torch.autograd.grad(expected.square().sum(), dx)[0],
        atol=1e-10,
        rtol=1e-9,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal GPU unavailable"
)
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("shift_backend", ["dense", "sparse"])
def test_metal_values_and_vjp_against_cpu_float64(graph, shift_backend):
    cpu = experiment(4, graph)
    metal = RollingLogSignature(cpu.plan.to(device="mps", dtype=torch.float32), 4)
    generator = torch.Generator().manual_seed(71)
    dx = (
        torch.randn(
            2,
            27,
            cpu.plan.alphabet.in_channels,
            dtype=torch.float64,
            generator=generator,
        )
        * 0.03
    ).requires_grad_()
    gpu_dx = dx.detach().to(device="mps", dtype=torch.float32).requires_grad_()
    expected = cpu.reference(dx, stride=2, method="tensor")
    actual = metal(gpu_dx, stride=2, chunk_windows=5, shift_backend=shift_backend)
    weights = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    expected_grad = torch.autograd.grad((expected * weights).sum(), dx)[0]
    actual_grad = torch.autograd.grad(
        (actual * weights.to(device="mps", dtype=torch.float32)).sum(), gpu_dx
    )[0]
    torch.testing.assert_close(actual.cpu().double(), expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(
        actual_grad.cpu().double(), expected_grad, atol=2e-4, rtol=2e-4
    )


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6])
def test_fused_endpoint_program_arbitrary_state_vjp(depth):
    from tensorsig.offset_coded_signatures.rolling_logsignature._program import (
        BCHProgram,
    )

    compiled = experiment(depth)
    program = BCHProgram(compiled.plan)
    torch.manual_seed(37)
    x = (
        torch.randn(2, compiled.plan.dimension, dtype=torch.float64) * 0.1
    ).requires_grad_()
    raw_y = torch.randn_like(x).requires_grad_()
    y = raw_y * (compiled.plan.basis_depths == 1) * 0.1
    actual = program(x, y)
    expected = compiled.plan._runtime.bch(x, y)
    weights = torch.randn_like(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad((actual * weights).sum(), (x, raw_y), retain_graph=True)
    b = torch.autograd.grad((expected * weights).sum(), (x, raw_y))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("backend", ["typed", "joint"])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "scan,refresh,chunk,stride",
    [
        (False, 0, 3, 1),
        (True, 0, None, 1),
        (True, 0, 5, 2),
        (True, 3, None, 1),
        (True, None, None, 6),
    ],
)
def test_retained_programs_and_scan_tensor_vjp(
    backend, graph, scan, refresh, chunk, stride
):
    compiled = experiment(4, graph)
    torch.manual_seed(101)
    dx = (
        torch.randn(2, 15, compiled.plan.alphabet.in_channels, dtype=torch.float64)
        * 0.03
    ).requires_grad_()
    evaluate = compiled.scan if scan else compiled
    actual = evaluate(
        dx, backend=backend, refresh_every=refresh, chunk_windows=chunk, stride=stride
    )
    expected = compiled.reference(dx, stride=stride, method="tensor")
    weights = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
    torch.testing.assert_close(
        torch.autograd.grad((actual * weights).sum(), dx)[0],
        torch.autograd.grad((expected * weights).sum(), dx)[0],
        atol=2e-10,
        rtol=2e-9,
    )


def test_scan_power_boundary_and_empty_quotient():
    from tensorsig.offset_coded_signatures.rolling_logsignature._scan import (
        affine_scan,
    )

    matrix = torch.tensor([[1.0, -1.0], [0.0, 1.0]], dtype=torch.float64)
    for count in [0, 1, 2, 3, 5, 8, 9]:
        b = torch.randn(2, count, 2, dtype=torch.float64)
        x = torch.randn(2, 2, dtype=torch.float64)
        powers = [
            torch.linalg.matrix_power(matrix, 2**k)
            for k in range(max(1, (max(1, count) - 1).bit_length()))
        ]
        actual = affine_scan(b, x, powers)
        states = []
        for forcing in b.unbind(1):
            x = x @ matrix + forcing
            states.append(x)
        expected = torch.stack(states, 1) if count else b
        torch.testing.assert_close(actual, expected)
    plan = compile_lie_quotient(
        LieQuotient(ChannelMultiplicity(0)), ProjectedAlphabet.newton(2, 2), 3
    )
    compiled = RollingLogSignature(plan, 3)
    dx = torch.randn(1, 7, 2, dtype=torch.float64)
    assert compiled.scan(dx).shape == (1, 5, 0)
    other = experiment()
    dx = torch.randn(1, 4, 2, dtype=torch.float64) * 0.01
    torch.testing.assert_close(other.scan(dx), other.reference(dx))


def test_scan_float32_long_stream_and_three_temporal_codes():
    torch.manual_seed(29)
    for code_dim, count in [(2, 257), (3, 17)]:
        plan = compile_lie_quotient(
            LieQuotient(TemporalDegree(code_dim - 1)),
            ProjectedAlphabet.newton(code_dim, 2),
            3,
        ).to(dtype=torch.float32)
        compiled = RollingLogSignature(plan, 5)
        dx = (torch.randn(1, count + 4, 2) * 0.001).requires_grad_()
        actual = compiled.scan(dx, refresh_every=0 if code_dim == 3 else 11)
        expected = compiled.reference(dx)
        weights = torch.randn_like(actual)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
        torch.testing.assert_close(
            torch.autograd.grad((actual * weights).sum(), dx)[0],
            torch.autograd.grad((expected * weights).sum(), dx)[0],
            atol=2e-4,
            rtol=2e-4,
        )
