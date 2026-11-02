"""Independent polynomial, quotient and higher-derivative gates."""

from fractions import Fraction
from itertools import product
from math import factorial

import pytest
import torch

from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    compile_lie_quotient,
    quotient_log_signature_from_increment_windows,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import (
    DependenceGraph,
    LieQuotient,
    OffsetAlphabet,
    OffsetSupport,
    ProjectedAlphabet,
    TemporalDegree,
)
from tensorsig.offset_coded_signatures.active_word_log import (
    active_word_log_signature,
    compile_active_word_log,
)
from tensorsig.offset_coded_signatures._word_program import (
    _compile_word_log_readout,
    _compile_word_program,
)


def test_factor_log_matches_independent_exact_tensor_polynomial():
    depth = 4

    def multiply(a, b):
        result = {}
        for u, x in a.items():
            for v, y in b.items():
                if len(u) + len(v) <= depth:
                    result[u + v] = result.get(u + v, Fraction()) + x * y
        return result

    signature = {(): Fraction(1)}
    for increment in ((1, 2), (-2, 1), (3, -1)):
        exponential = {(): Fraction(1)}
        for n in range(1, depth + 1):
            for word in product(range(2), repeat=n):
                coefficient = Fraction(1, factorial(n))
                for letter in word:
                    coefficient *= increment[letter]
                exponential[word] = coefficient
        signature = multiply(signature, exponential)
    augmentation = {w: x for w, x in signature.items() if w}
    power = {(): Fraction(1)}
    logarithm = {}
    for k in range(1, depth + 1):
        power = multiply(power, augmentation)
        for word, value in power.items():
            logarithm[word] = (
                logarithm.get(word, Fraction()) + Fraction((-1) ** (k - 1), k) * value
            )
    requested = tuple(w for w in signature if w) + ((0, 1, 0),)
    program = _compile_word_program(requested, 1, factors=True)
    readout = _compile_word_log_readout(program, requested)
    state = torch.tensor(
        [float(signature[word]) for word in program.words], dtype=torch.float64
    )
    expected = torch.tensor(
        [float(logarithm[w]) for w in requested], dtype=torch.float64
    )
    torch.testing.assert_close(readout(state), expected, atol=1e-12, rtol=1e-12)
    prefix_only = _compile_word_program(((0, 1, 2),), 1)
    with pytest.raises(ValueError, match="factor"):
        _compile_word_log_readout(prefix_only, ((0, 1, 2),))


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("full_offsets", [False, True])
def test_active_quotient_values_vjps_hvps_and_exact_support(graph, full_offsets):
    torch.manual_seed(41)
    alphabet = OffsetAlphabet(3, 2) if full_offsets else ProjectedAlphabet.newton(2, 2)
    rules = [OffsetSupport(2)] if full_offsets else [TemporalDegree(1)]
    if graph:
        rules.append(DependenceGraph(((0, 1),)))
    plan = compile_lie_quotient(LieQuotient(*rules), alphabet, 3)
    x = (torch.randn(1, 2, 3, 2, dtype=torch.float64) * 0.1).requires_grad_()
    code = (
        None
        if full_offsets
        else torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
    )
    inputs = (x,) if code is None else (x, code)
    active = quotient_log_signature_from_increment_windows(
        x, plan, code, method="active"
    )
    reference = quotient_log_signature_from_increment_windows(
        x, plan, code, method="tensor"
    )
    torch.testing.assert_close(active, reference, atol=1e-12, rtol=1e-10)
    directions = tuple(torch.randn_like(value) for value in inputs)
    first = torch.autograd.grad(active.sin().square().sum(), inputs, create_graph=True)
    expected_first = torch.autograd.grad(
        reference.sin().square().sum(), inputs, create_graph=True
    )
    second = torch.autograd.grad(
        sum((g * v).sum() for g, v in zip(first, directions)), inputs
    )
    expected_second = torch.autograd.grad(
        sum((g * v).sum() for g, v in zip(expected_first, directions)), inputs
    )
    for a, b in zip((*first, *second), (*expected_first, *expected_second)):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)
    bridge = plan._word_bridge
    dense = bridge.quotient_matrix.to_dense()
    assert bridge.active_columns == tuple(torch.where(dense.any(0))[0].tolist())
    assert (
        plan.to(dtype=torch.float32)._word_bridge.active_columns
        == bridge.active_columns
    )
    assert compile_active_word_log(plan) is compile_active_word_log(plan)


def test_active_independent_windows_third_derivatives():
    torch.manual_seed(22)
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 2), 3
    )
    code = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    x = (torch.randn(1, 4, 3, 2, dtype=torch.float64) * 0.04).requires_grad_()
    actual = active_word_log_signature(x, plan, code)
    reference = quotient_log_signature_from_increment_windows(
        x, plan, code, method="tensor"
    )
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-9)
    v = torch.randn_like(x)
    for output in (actual, reference):
        first = torch.autograd.grad(output.square().sum(), x, create_graph=True)[0]
        second = torch.autograd.grad((first * v).sum(), x, create_graph=True)[0]
        third = torch.autograd.grad((second * v).sum(), x)[0]
        if output is actual:
            derivatives = (first, second, third)
        else:
            for a, b in zip(derivatives, (first, second, third)):
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)


def test_empty_abelian_zero_and_gradgradcheck():
    alphabet = ProjectedAlphabet.newton(2, 1)
    plan = compile_lie_quotient(LieQuotient(TemporalDegree(1)), alphabet, 3)
    code = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    x = torch.zeros(1, 2, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda v: active_word_log_signature(v, plan, code), (x,)
    )
    empty_plan = compile_lie_quotient(
        LieQuotient(OffsetSupport(0)), OffsetAlphabet(2, 1), 2
    )
    windows = torch.randn(1, 2, 1, dtype=torch.float64, requires_grad=True)
    result = active_word_log_signature(windows, empty_plan)
    assert result.shape == (1, 0)
    torch.testing.assert_close(
        torch.autograd.grad(result.sum(), windows)[0], torch.zeros_like(windows)
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_active_and_tensor_bridge():
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(0)), ProjectedAlphabet.newton(2, 2), 3
    )
    # Compile CPU COO first: device binding must not try to move unsupported COO.
    assert plan._word_bridge.quotient_matrix.layout == torch.sparse_coo
    plan = plan.to(device="mps", dtype=torch.float32)
    from tensorsig.offset_coded_signatures.windows import WindowLogSignature

    runtime = WindowLogSignature(plan, 3)
    x = (torch.randn(1, 7, 2, device="mps") * 0.03).requires_grad_()
    a = runtime(x, algorithm="active", chunk_windows=3)
    b = runtime(x, algorithm="tensor")
    torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
    ga = torch.autograd.grad(a.square().sum(), x, create_graph=True)[0]
    gb = torch.autograd.grad(b.square().sum(), x, create_graph=True)[0]
    torch.testing.assert_close(ga, gb, atol=3e-5, rtol=3e-4)
    torch.autograd.grad(ga.square().sum(), x)


def test_compiled_active_plan_reuses_device_dtype_bindings():
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)), ProjectedAlphabet.newton(2, 2), 3
    )
    compiled = compile_active_word_log(plan)
    assert compile_active_word_log(plan) is compiled
    assert compiled.to("cpu", torch.float64) is compiled
    single = compiled.to("cpu", torch.float32)
    assert compiled.to(torch.device("cpu"), torch.float32) is single
    torch.testing.assert_close(
        single.quotient_matrix.double(), compiled.quotient_matrix
    )
    with pytest.raises(TypeError, match="rolling"):
        compile_active_word_log(plan, rolling=True)
