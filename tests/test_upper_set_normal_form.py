from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch

import tensorsig.algebra.hall_bch as hall_bch_module
import tensorsig.algebra.hall_projection as hall_projection_module
from tensorsig.algebra.hall_bch import _sparse_constants
from tensorsig.algebra.hall import hall_basis
from tensorsig.algebra.hall_projection import HallBasisElement
from tensorsig.offset_coded_signatures.quotients import (
    ChannelMultiplicity,
    ChannelSupport,
    DependenceGraph,
    LieQuotient,
    OffsetMultiplicity,
    OffsetSpan,
    OffsetSupport,
    TemporalDegree,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import (
    OffsetAlphabet,
    ProjectedAlphabet,
    QuotientAlphabet,
)
from tensorsig.offset_coded_signatures._lie_quotients.upper_set_normal_form import (
    UpperSetHallNormalForm,
    compile_upper_set_hall_normal_form,
)


def _letters(element: HallBasisElement) -> tuple[int, ...]:
    if isinstance(element, int):
        return (element,)
    return _letters(element[0]) + _letters(element[1])


def _reference_retained(
    element: HallBasisElement,
    quotient: LieQuotient,
    alphabet: QuotientAlphabet,
) -> bool:
    leaves = _letters(element)
    channel_counts = [0] * alphabet.in_channels
    temporal_degree = 0
    if isinstance(alphabet, OffsetAlphabet):
        offset_counts = [0] * alphabet.kernel_size
    else:
        offset_counts = []

    for letter in leaves:
        zero_based = letter - 1
        if isinstance(alphabet, ProjectedAlphabet):
            channel = zero_based // alphabet.code_dim
            if alphabet.temporal_degrees is not None:
                temporal_degree += alphabet.temporal_degrees[
                    zero_based % alphabet.code_dim
                ]
        else:
            channel = zero_based % alphabet.in_channels
            offset_counts[zero_based // alphabet.in_channels] += 1
        channel_counts[channel] += 1

    for rule in quotient.rules:
        if isinstance(rule, TemporalDegree):
            if temporal_degree > rule.maximum:
                return False
        elif isinstance(rule, ChannelMultiplicity):
            caps = (
                (rule.maximum,) * alphabet.in_channels
                if isinstance(rule.maximum, int)
                else rule.maximum
            )
            if any(
                count > cap for count, cap in zip(channel_counts, caps, strict=True)
            ):
                return False
        elif isinstance(rule, ChannelSupport):
            if sum(count > 0 for count in channel_counts) > rule.maximum:
                return False
        elif isinstance(rule, OffsetMultiplicity):
            if max(offset_counts) > rule.maximum:
                return False
        elif isinstance(rule, OffsetSupport):
            if sum(count > 0 for count in offset_counts) > rule.maximum:
                return False
        elif isinstance(rule, OffsetSpan):
            support = [index for index, count in enumerate(offset_counts) if count]
            if support[-1] - support[0] > rule.maximum:
                return False
    return True


@pytest.mark.parametrize(
    ("alphabet", "quotient", "max_depth"),
    [
        pytest.param(
            ProjectedAlphabet.newton(code_dim=3, in_channels=2),
            LieQuotient(
                TemporalDegree(2),
                ChannelMultiplicity((2, 1)),
                ChannelSupport(1),
            ),
            4,
            id="projected-additive-and-support",
        ),
        pytest.param(
            OffsetAlphabet(kernel_size=3, in_channels=2),
            LieQuotient(
                ChannelMultiplicity(2),
                ChannelSupport(2),
                OffsetMultiplicity(2),
                OffsetSupport(2),
                OffsetSpan(1),
            ),
            3,
            id="full-offset-additive-support-and-span",
        ),
    ],
)
def test_direct_basis_is_exactly_the_filtered_ambient_hall_basis(
    alphabet: QuotientAlphabet,
    quotient: LieQuotient,
    max_depth: int,
) -> None:
    direct = compile_upper_set_hall_normal_form(
        quotient,
        alphabet,
        max_depth,
    )
    expected = tuple(
        element
        for element in hall_basis(alphabet.width, max_depth)
        if _reference_retained(element, quotient, alphabet)
    )

    assert direct.basis == expected


def _coo_entries(
    indices: torch.Tensor,
    values: torch.Tensor,
    *,
    tolerance: float = 0.0,
) -> dict[tuple[int, int, int], float]:
    result = {}
    for row, value in zip(indices.tolist(), values.tolist(), strict=True):
        if abs(value) > tolerance:
            result[tuple(row)] = value
    return result


def test_direct_constants_match_filtering_the_ambient_hall_constants() -> None:
    alphabet = ProjectedAlphabet.newton(code_dim=2, in_channels=2)
    quotient = LieQuotient(TemporalDegree(1), ChannelMultiplicity(2))
    max_depth = 4
    direct = compile_upper_set_hall_normal_form(
        quotient,
        alphabet,
        max_depth,
    )

    ambient_basis = tuple(hall_basis(alphabet.width, max_depth))
    ambient_indices, ambient_values = _sparse_constants(
        alphabet.width,
        max_depth,
    )
    direct_indices = {element: index for index, element in enumerate(direct.basis)}
    ambient_to_direct = {
        index: direct_indices[element]
        for index, element in enumerate(ambient_basis)
        if element in direct_indices
    }
    expected_indices = []
    expected_values = []
    for row, value in zip(
        ambient_indices.tolist(),
        ambient_values.tolist(),
        strict=True,
    ):
        if all(index in ambient_to_direct for index in row):
            expected_indices.append([ambient_to_direct[index] for index in row])
            expected_values.append(value)

    expected = _coo_entries(
        torch.tensor(expected_indices, dtype=torch.long).reshape(-1, 3),
        torch.tensor(expected_values, dtype=torch.float64),
        tolerance=1e-10,
    )
    actual = _coo_entries(direct.bracket_indices, direct.bracket_values)

    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        assert value == pytest.approx(expected[key], abs=1e-10)


def _forbidden_ambient_call(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise AssertionError("upper-set compilation touched an ambient constructor")


def test_strong_pruning_never_constructs_the_ambient_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hall_projection_module, "hall_basis", _forbidden_ambient_call)
    monkeypatch.setattr(
        hall_projection_module,
        "get_hall_projector",
        _forbidden_ambient_call,
    )
    monkeypatch.setattr(hall_bch_module, "_sparse_constants", _forbidden_ambient_call)

    normal_form = compile_upper_set_hall_normal_form(
        LieQuotient(TemporalDegree(0)),
        ProjectedAlphabet.newton(code_dim=5, in_channels=2),
        max_depth=4,
    )

    assert normal_form.dimension == 8
    assert normal_form.level_dimensions == (2, 1, 2, 3)
    assert len(normal_form.bracket_values) == 14


def _integer_bracket(
    normal_form: UpperSetHallNormalForm,
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    result = [0] * normal_form.dimension
    for (left_index, right_index, output_index), value in zip(
        normal_form.bracket_indices.tolist(),
        normal_form.bracket_values.tolist(),
        strict=True,
    ):
        result[output_index] += left[left_index] * right[right_index] * int(value)
    return tuple(result)


def _add_vectors(vectors: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
    vectors = tuple(vectors)
    return tuple(map(sum, zip(*vectors, strict=True)))


def test_depth_five_constants_are_integer_exact_and_satisfy_jacobi() -> None:
    normal_form = compile_upper_set_hall_normal_form(
        LieQuotient(),
        ProjectedAlphabet(code_dim=1, in_channels=2),
        max_depth=5,
    )
    assert normal_form.basis == tuple(hall_basis(2, 5))
    assert normal_form.level_dimensions == (2, 1, 2, 3, 6)
    torch.testing.assert_close(
        normal_form.bracket_values,
        normal_form.bracket_values.round(),
        rtol=0,
        atol=0,
    )

    generator = torch.Generator().manual_seed(718)
    for _ in range(10):
        x, y, z = (
            tuple(
                torch.randint(
                    -2,
                    3,
                    (normal_form.dimension,),
                    generator=generator,
                ).tolist()
            )
            for _ in range(3)
        )
        jacobi = _add_vectors(
            (
                _integer_bracket(normal_form, x, _integer_bracket(normal_form, y, z)),
                _integer_bracket(normal_form, y, _integer_bracket(normal_form, z, x)),
                _integer_bracket(normal_form, z, _integer_bracket(normal_form, x, y)),
            )
        )
        assert jacobi == (0,) * normal_form.dimension


def test_generator_embedding_handles_killed_generators_and_zero_quotients() -> None:
    partially_killed = compile_upper_set_hall_normal_form(
        LieQuotient(TemporalDegree(0)),
        ProjectedAlphabet.newton(code_dim=3, in_channels=1),
        max_depth=3,
    )
    torch.testing.assert_close(
        partially_killed.generator_map,
        torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64),
    )

    zero = compile_upper_set_hall_normal_form(
        LieQuotient(ChannelSupport(0)),
        ProjectedAlphabet(code_dim=2, in_channels=2),
        max_depth=5,
    )
    assert zero.dimension == 0
    assert zero.level_dimensions == (0, 0, 0, 0, 0)
    assert zero.generator_map.shape == (4, 0)
    assert zero.bracket_indices.shape == (0, 3)


def test_relation_rules_are_not_accepted_by_the_upper_set_compiler() -> None:
    with pytest.raises(ValueError, match="relation ideal"):
        compile_upper_set_hall_normal_form(
            LieQuotient(DependenceGraph(((0, 1),))),
            ProjectedAlphabet(code_dim=1, in_channels=2),
            max_depth=3,
        )
