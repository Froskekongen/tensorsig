"""Mathematical and runtime tests for quotient PBW signatures."""

from __future__ import annotations

from math import factorial

import pytest
import torch

from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.quotients import (
    ChannelSupport,
    DependenceGraph,
    LieQuotient,
    OffsetSpan,
    TemporalDegree,
)
from tensorsig.offset_coded_signatures._lie_quotients.enveloping import (
    compile_quotient_signature_plan,
    quotient_signature_from_increment_windows,
)
from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    compile_lie_quotient,
    quotient_log_signature_from_increment_windows,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import (
    OffsetAlphabet,
    ProjectedAlphabet,
)


@pytest.mark.parametrize(
    ("quotient", "alphabet", "depth", "expected"),
    [
        (
            LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
            ProjectedAlphabet(1, 3),
            5,
            (3, 8, 21, 55, 144),
        ),
        (
            LieQuotient(TemporalDegree(1)),
            ProjectedAlphabet.newton(2, 1),
            6,
            (2, 4, 7, 12, 19, 30),
        ),
    ],
)
def test_pbw_dimensions_match_theoretical_hilbert_series(
    quotient: LieQuotient,
    alphabet: ProjectedAlphabet,
    depth: int,
    expected: tuple[int, ...],
) -> None:
    lie_plan = compile_lie_quotient(quotient, alphabet, depth)
    signature_plan = compile_quotient_signature_plan(lie_plan)

    assert signature_plan.level_dimensions == expected
    assert signature_plan.dimension == sum(expected)
    assert signature_plan.action_nnz > 0


@pytest.mark.parametrize("depth", range(1, 6))
@pytest.mark.parametrize(
    ("quotient", "alphabet"),
    [
        (LieQuotient(), ProjectedAlphabet(1, 2)),
        (
            LieQuotient(TemporalDegree(1)),
            ProjectedAlphabet.newton(2, 1),
        ),
        (
            LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
            ProjectedAlphabet(1, 3),
        ),
        (
            LieQuotient(OffsetSpan(1)),
            OffsetAlphabet(3, 1),
        ),
    ],
)
def test_pbw_log_matches_bch_and_ambient_tensor_across_quotient_families(
    quotient: LieQuotient,
    alphabet: ProjectedAlphabet | OffsetAlphabet,
    depth: int,
) -> None:
    plan = compile_lie_quotient(quotient, alphabet, depth)
    kernel_size = alphabet.kernel_size if isinstance(alphabet, OffsetAlphabet) else 3
    generator = torch.Generator().manual_seed(4100 + 10 * alphabet.width + depth)
    windows = 0.1 * torch.randn(
        2,
        1,
        kernel_size,
        alphabet.in_channels,
        generator=generator,
        dtype=torch.float64,
    )
    phase = (
        None
        if isinstance(alphabet, OffsetAlphabet)
        else (
            OffsetCode.pascal(
                kernel_size,
                alphabet.code_dim,
                dtype=torch.float64,
            ).matrix
            if alphabet.temporal_degrees is not None
            else torch.randn(
                kernel_size,
                alphabet.code_dim,
                generator=generator,
                dtype=torch.float64,
            )
        )
    )

    direct = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        phase,
        method="direct",
    )
    pbw = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        phase,
        method="pbw",
    )
    tensor = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        phase,
        method="tensor",
    )

    torch.testing.assert_close(pbw, direct, atol=2e-10, rtol=2e-9)
    torch.testing.assert_close(pbw, tensor, atol=2e-10, rtol=2e-9)


def test_depth_six_one_generator_signature_is_the_truncated_exponential() -> None:
    lie_plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(0)),
        ProjectedAlphabet.newton(2, 1),
        max_depth=6,
    )
    signature_plan = compile_quotient_signature_plan(lie_plan)
    windows = torch.tensor(
        [[[[0.2], [-0.1], [0.3], [0.4]]]],
        dtype=torch.float64,
    )
    phase = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=torch.float64,
    )

    signature = quotient_signature_from_increment_windows(
        windows,
        signature_plan,
        phase,
    )
    total = windows.sum()
    expected = torch.stack(
        [total**level / factorial(level) for level in range(1, 7)]
    ).reshape(1, 1, -1)

    assert signature_plan.level_dimensions == (1, 1, 1, 1, 1, 1)
    torch.testing.assert_close(signature, expected)
    log_signature = quotient_log_signature_from_increment_windows(
        windows,
        lie_plan,
        phase,
        method="pbw",
    )
    torch.testing.assert_close(log_signature[..., 0], total.reshape(1, 1))


def test_pbw_signature_is_invariant_under_collinear_segment_subdivision() -> None:
    lie_plan = compile_lie_quotient(
        LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
        ProjectedAlphabet(1, 3),
        max_depth=4,
    )
    signature_plan = compile_quotient_signature_plan(lie_plan)
    generator = torch.Generator().manual_seed(20260803)
    windows = torch.randn(1, 2, 3, 3, generator=generator, dtype=torch.float64)
    subdivided = (
        torch.stack((windows / 2, windows / 2), dim=3).reshape(1, 2, 6, 3).contiguous()
    )

    original = quotient_signature_from_increment_windows(
        windows,
        signature_plan,
        torch.ones(3, 1, dtype=torch.float64),
    )
    refined = quotient_signature_from_increment_windows(
        subdivided,
        signature_plan,
        torch.ones(6, 1, dtype=torch.float64),
    )

    torch.testing.assert_close(original, refined, atol=2e-12, rtol=2e-12)


def test_associative_commutator_recovers_the_compiled_lie_bracket() -> None:
    lie_plan = compile_lie_quotient(
        LieQuotient(),
        ProjectedAlphabet(1, 2),
        max_depth=2,
    )
    signature_plan = compile_quotient_signature_plan(lie_plan)
    first = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    second = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

    associative_bracket = signature_plan._right_multiply(1, first, second)
    associative_bracket -= signature_plan._right_multiply(1, second, first)
    primitive = signature_plan._apply_log_readout(2, associative_bracket)

    first_index = signature_plan.basis[1][0][0]
    second_index = signature_plan.basis[1][1][0]
    first_lie = torch.zeros(1, lie_plan.dimension, dtype=torch.float64)
    second_lie = torch.zeros_like(first_lie)
    first_lie[0, first_index] = 1
    second_lie[0, second_index] = 1
    expected = lie_plan._runtime.bracket(first_lie, second_lie)
    level_two = lie_plan.basis_depths == 2

    torch.testing.assert_close(primitive, expected[:, level_two])


def test_pbw_runtime_is_differentiable_in_path_and_offset_code() -> None:
    lie_plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)),
        ProjectedAlphabet.newton(2, 2),
        max_depth=4,
    )
    windows = torch.randn(
        2,
        2,
        4,
        2,
        dtype=torch.float64,
        requires_grad=True,
    )
    phase = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)

    output = quotient_log_signature_from_increment_windows(
        windows,
        lie_plan,
        phase,
        method="pbw",
    )
    output.square().sum().backward()

    assert windows.grad is not None and torch.isfinite(windows.grad).all()
    assert phase.grad is not None and torch.isfinite(phase.grad).all()


def test_pbw_runtime_never_falls_back_to_ambient_or_bch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorsig.algebra.hall_bch as hall_bch
    import tensorsig.offset_coded_signatures._lie_quotients.compiler as quotient_module

    lie_plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)),
        ProjectedAlphabet.newton(2, 1),
        max_depth=4,
    )
    compile_quotient_signature_plan(lie_plan)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("PBW execution must not use an ambient or BCH runtime")

    monkeypatch.setattr(quotient_module, "_batch_log_signature_from_increments", fail)
    monkeypatch.setattr(hall_bch.SparseBCH, "bch", fail)

    output = quotient_log_signature_from_increment_windows(
        torch.randn(1, 1, 3, 1, dtype=torch.float64),
        lie_plan,
        OffsetCode.pascal(3, 2, dtype=torch.float64).matrix,
        method="pbw",
    )

    assert output.shape == (1, 1, lie_plan.dimension)


@pytest.mark.parametrize("method", ["enveloping", "PBW", ""])
def test_quotient_runtime_rejects_backend_aliases(method: str) -> None:
    lie_plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)),
        ProjectedAlphabet.newton(2, 1),
        max_depth=2,
    )

    with pytest.raises(ValueError, match="direct.*pbw.*tensor"):
        quotient_log_signature_from_increment_windows(
            torch.randn(1, 1, 3, 1, dtype=torch.float64),
            lie_plan,
            OffsetCode.pascal(3, 2, dtype=torch.float64).matrix,
            method=method,
        )


def test_zero_quotient_has_only_the_scalar_pbw_state() -> None:
    lie_plan = compile_lie_quotient(
        LieQuotient(ChannelSupport(0)),
        ProjectedAlphabet(1, 2),
        max_depth=3,
    )
    signature_plan = compile_quotient_signature_plan(lie_plan)
    windows = torch.randn(2, 3, 4, 2)
    phase = torch.ones(4, 1)

    signature = quotient_signature_from_increment_windows(
        windows,
        signature_plan,
        phase,
    )
    log_signature = quotient_log_signature_from_increment_windows(
        windows,
        lie_plan,
        phase,
        method="pbw",
    )

    assert signature_plan.level_dimensions == (0, 0, 0)
    assert signature.shape == (2, 3, 0)
    assert log_signature.shape == (2, 3, 0)


def test_quotient_signature_plan_binding_and_compilation_are_memoized() -> None:
    lie_plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(0)),
        ProjectedAlphabet.newton(2, 2),
        max_depth=4,
    )
    first = compile_quotient_signature_plan(lie_plan)
    second = compile_quotient_signature_plan(lie_plan)
    moved = first.to(dtype=torch.float32)

    assert first is second
    assert moved is first.to(dtype=torch.float32)
    assert moved.generator_map.dtype == torch.float32
    assert all(values.dtype == torch.float32 for values in moved.action_values)
    assert all(indices.dtype == torch.long for indices in moved.action_indices)
