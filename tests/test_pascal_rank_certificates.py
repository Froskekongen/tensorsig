"""Exact closed profiles and independently replayable PBW--forest witnesses."""

from dataclasses import replace
from fractions import Fraction
from itertools import combinations_with_replacement, permutations, product
from math import comb

import pytest
import torch

from tensorsig.offset_coded_signatures.pascal_certificate import (
    _eliminate,
    binary_pascal_rank_certificate,
)
from tensorsig.offset_coded_signatures.pascal_rank import (
    binary_pascal_profile,
    binary_pascal_rank,
)
from tensorsig.experimental.offset_coded_signatures.spectral_eulerian import (
    spectral_eulerian_project_columns,
)
from tensorsig.offset_coded_signatures.primitive_chart import (
    _build_signature_source_basis,
    _eulerian_project_columns,
)


@pytest.mark.parametrize(
    "degree, increments",
    [
        (2, (1,)),
        (3, (1, 1)),
        (4, (1,)),
        (5, (1, 2, 1)),
        (6, (1, 1, 1)),
        (8, (1, 1, 2, 1)),
        (12, (1, 2, 4, 2)),
    ],
)
def test_two_offset_exact_profile(degree, increments):
    profile = binary_pascal_profile(degree, 2)
    assert profile == tuple(sum(increments[:d]) for d in range(degree + 1))


def test_scope_and_unknown_regime():
    assert binary_pascal_rank(8, 3, 3) is None
    assert binary_pascal_rank(8, 3, 2) == 4
    assert binary_pascal_rank(5, 3, 3) == 5
    with pytest.raises(ValueError):
        binary_pascal_rank(3, 4, 1)
    with pytest.raises(TypeError):
        binary_pascal_rank(True, 2, 1)
    with pytest.raises(ValueError, match="budget"):
        binary_pascal_rank_certificate(11, 3, 3)


@pytest.mark.parametrize("degree", range(2, 7))
def test_projectors_against_value_side_profile_not_coordinate_filtration(degree):
    code = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    source = _build_signature_source_basis(
        code, in_channels=1, depth=degree, rank_tolerance=1e-10
    )
    weights = torch.tensor([sum(word) for word in product((0, 1), repeat=degree)])
    for projector in (_eulerian_project_columns, spectral_eulerian_project_columns):
        values = projector(source.reconstruction_matrix, 2, degree)
        for weight, expected in enumerate(binary_pascal_profile(degree, 2)):
            # These are ambient VALUE rows. max_degree cuts signature-source
            # columns and must not be used as an interchangeable rank oracle.
            assert (
                int(
                    torch.linalg.matrix_rank(
                        values[weights <= weight], atol=1e-10, rtol=0
                    )
                )
                == expected
            )


@pytest.mark.parametrize(
    "degree,window,weight",
    [(m, 2, m) for m in range(2, 8)] + [(5, 3, 3), (6, 4, 4), (7, 3, 2)],
)
def test_certificates_match_closed_ranks_and_verify(degree, window, weight):
    result = binary_pascal_rank_certificate(degree, window, weight)
    assert result.primitive_rank == binary_pascal_rank(degree, window, weight)
    assert result.verify()
    if result.kernel:
        assert not replace(result, kernel=result.kernel[:-1]).verify()


def test_certificate_rejects_vacuous_minors_wrong_input_and_malformed_witnesses():
    result = binary_pascal_rank_certificate(5, 2, 5)
    assert not replace(
        result,
        pivot_rows=(),
        pivot_columns=tuple(range(result.column_count)),
        kernel=(),
    ).verify()
    assert not replace(result, window=3).verify()
    assert not replace(result, pivot_rows=(0,) * len(result.pivot_rows)).verify()
    assert not replace(
        result, pivot_columns=(result.column_count,) + result.pivot_columns[1:]
    ).verify()
    assert not replace(
        result, rows=(((0, Fraction(1)), (0, Fraction(1))),) + result.rows[1:]
    ).verify()
    index = next(i for i, row in enumerate(result.rows) if row)
    assert not replace(
        result, rows=result.rows[:index] + ((),) + result.rows[index + 1 :]
    ).verify()


@pytest.mark.parametrize("degree,window", [(3, 2), (4, 2), (4, 3), (5, 3)])
def test_certificate_against_independent_factorial_value_projection(degree, window):
    # Direct right-action Eulerian projection on all weakly ordered sources.
    # This deliberately shares neither PBW nor forest formulas with the candidate.
    words = tuple(product((0, 1), repeat=degree))
    schedule = []
    for permutation in permutations(range(degree)):
        descents = sum(a > b for a, b in zip(permutation, permutation[1:]))
        schedule.append(
            (
                permutation,
                Fraction((-1) ** descents, degree * comb(degree - 1, descents)),
            )
        )
    sources = tuple(combinations_with_replacement(range(window), degree))
    rows = []
    for word in words:
        row = []
        for source in sources:
            value = Fraction()
            for permutation, coefficient in schedule:
                term = 1
                for position, letter in enumerate(word):
                    if letter:
                        term *= source[permutation[position]]
                value += coefficient * term
            row.append(value)
        rows.append(row)
    for weight in range(degree + 1):
        restricted = tuple(
            tuple((j, a) for j, a in enumerate(row) if a)
            for word, row in zip(words, rows)
            if sum(word) <= weight
        )
        _, pivots, _ = _eliminate(restricted, len(sources))
        assert binary_pascal_rank_certificate(
            degree, window, weight
        ).primitive_rank == len(pivots)


def test_integer_entries_cannot_turn_a_singular_exact_minor_into_a_rank_witness():
    result = binary_pascal_rank_certificate(4, 2, 4)
    assert result.matrix_rank == 9
    # Integer entries are legitimate exact rationals, but int/int must not
    # introduce float arithmetic during minor elimination.
    rows = tuple(
        tuple(
            (index, int(value) if value.denominator == 1 else value)
            for index, value in row
        )
        for row in result.rows
    )
    integer_entries = replace(result, rows=rows)
    assert integer_entries.verify()
    forged = replace(
        integer_entries,
        pivot_rows=(4, 6, 7, 2, 5, 1, 3, 9, 8, 0),
        pivot_columns=(6, 9, 3, 1, 2, 5, 7, 0, 10, 4),
        kernel=result.kernel[1:],
    )
    assert forged.matrix_rank == 10
    assert not forged.verify()
