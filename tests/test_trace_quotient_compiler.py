from __future__ import annotations

from collections import deque
from fractions import Fraction
from itertools import combinations

import pytest
import torch

from tensorsig.algebra.hall_bch import SparseBCH
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
from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    compile_lie_quotient,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import (
    OffsetAlphabet,
    ProjectedAlphabet,
    QuotientRule,
)
from tensorsig.offset_coded_signatures._lie_quotients.trace_compiler import (
    _TraceMonoid,
    compile_trace_lie_quotient,
)


def _flat_word(index: int, width: int, depth: int) -> tuple[int, ...]:
    letters = [0] * depth
    for position in range(depth - 1, -1, -1):
        letters[position] = index % width
        index //= width
    return tuple(letters)


def _independent_trace_normal_form(
    word: tuple[int, ...],
    alphabet: ProjectedAlphabet,
    edges: frozenset[tuple[int, int]],
) -> tuple[int, ...]:
    """Canonicalize by exhaustive adjacent commutations, independently."""
    pending = deque((word,))
    seen = {word}
    while pending:
        current = pending.popleft()
        for position in range(len(current) - 1):
            left = current[position] // alphabet.code_dim
            right = current[position + 1] // alphabet.code_dim
            edge = (min(left, right), max(left, right))
            if left == right or edge in edges:
                continue
            swapped = (
                current[:position]
                + (current[position + 1], current[position])
                + current[position + 2 :]
            )
            if swapped not in seen:
                seen.add(swapped)
                pending.append(swapped)
    return min(seen)


def _independent_trace_product(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
    alphabet: ProjectedAlphabet,
    edges: frozenset[tuple[int, int]],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            word = _independent_trace_normal_form(
                left_word + right_word,
                alphabet,
                edges,
            )
            result[word] = (
                result.get(word, Fraction()) + left_coefficient * right_coefficient
            )
    return {word: value for word, value in result.items() if value}


@pytest.mark.parametrize("max_depth", [1, 2, 3, 4])
def test_trace_compiler_matches_all_three_channel_graph_dimensions(
    max_depth: int,
) -> None:
    pairs = tuple(combinations(range(3), 2))
    alphabet = ProjectedAlphabet(code_dim=1, in_channels=3)
    for mask in range(1 << len(pairs)):
        graph = DependenceGraph(
            edge for position, edge in enumerate(pairs) if mask & (1 << position)
        )
        quotient = LieQuotient(graph)
        direct = compile_trace_lie_quotient(quotient, alphabet, max_depth)
        ambient = compile_lie_quotient(quotient, alphabet, max_depth)

        assert direct.level_dimensions == ambient.level_dimensions


@pytest.mark.parametrize(
    ("quotient", "alphabet", "expected_edges"),
    [
        (
            LieQuotient(
                DependenceGraph(((0, 1), (1, 2))),
                TemporalDegree(1),
                ChannelSupport(2),
            ),
            ProjectedAlphabet.newton(code_dim=2, in_channels=3),
            frozenset({(0, 1), (1, 2)}),
        ),
        (
            LieQuotient(
                DependenceGraph(((0, 1),)),
                OffsetSpan(1),
                OffsetMultiplicity(2),
            ),
            OffsetAlphabet(kernel_size=3, in_channels=3),
            frozenset({(0, 1)}),
        ),
        (
            LieQuotient(
                DependenceGraph(((0, 1), (1, 2))),
                DependenceGraph(((0, 1),)),
                ChannelMultiplicity(1),
            ),
            ProjectedAlphabet(code_dim=2, in_channels=3),
            frozenset({(0, 1)}),
        ),
    ],
)
def test_trace_compiler_matches_combined_graph_upper_set_quotients(
    quotient: LieQuotient,
    alphabet: ProjectedAlphabet | OffsetAlphabet,
    expected_edges: frozenset[tuple[int, int]],
) -> None:
    direct = compile_trace_lie_quotient(quotient, alphabet, max_depth=3)
    ambient = compile_lie_quotient(quotient, alphabet, max_depth=3)

    assert direct.level_dimensions == ambient.level_dimensions
    assert direct.dependence_edges == expected_edges


@pytest.mark.parametrize(
    ("rule", "alphabet"),
    [
        (TemporalDegree(1), ProjectedAlphabet.newton(2, 3)),
        (ChannelMultiplicity((1, 2, 1)), ProjectedAlphabet(2, 3)),
        (ChannelSupport(2), ProjectedAlphabet(2, 3)),
        (OffsetMultiplicity(1), OffsetAlphabet(3, 3)),
        (OffsetSupport(2), OffsetAlphabet(3, 3)),
        (OffsetSpan(1), OffsetAlphabet(3, 3)),
    ],
)
def test_every_upper_set_rule_can_be_conjoined_with_a_graph(
    rule: QuotientRule,
    alphabet: ProjectedAlphabet | OffsetAlphabet,
) -> None:
    quotient = LieQuotient(
        DependenceGraph(((0, 1), (1, 2))),
        rule,
    )

    direct = compile_trace_lie_quotient(quotient, alphabet, max_depth=3)
    ambient = compile_lie_quotient(quotient, alphabet, max_depth=3)

    assert direct.level_dimensions == ambient.level_dimensions


def test_unified_compiler_preserves_the_exact_trace_bracket() -> None:
    quotient = LieQuotient(
        DependenceGraph(((0, 1), (1, 2))),
        TemporalDegree(1),
        ChannelSupport(2),
    )
    alphabet = ProjectedAlphabet.newton(code_dim=2, in_channels=3)
    direct = compile_trace_lie_quotient(quotient, alphabet, max_depth=3)
    ambient = compile_lie_quotient(quotient, alphabet, max_depth=3)
    direct_runtime = SparseBCH(
        dim=direct.dimension,
        depth=3,
        sparse_indices=direct.bracket_indices,
        sparse_values=direct.bracket_values,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(142)
    left = torch.randn(5, direct.dimension, generator=generator, dtype=torch.float64)
    right = torch.randn(5, direct.dimension, generator=generator, dtype=torch.float64)

    expected = ambient._runtime.bracket(left, right)
    actual = direct_runtime.bracket(left, right)

    assert direct.basis == ambient.basis
    torch.testing.assert_close(direct.generator_map, ambient.generator_map)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


def test_trace_structure_constants_match_independent_trace_word_brackets() -> None:
    edges = frozenset({(0, 1), (1, 2)})
    alphabet = ProjectedAlphabet(code_dim=1, in_channels=3)
    plan = compile_trace_lie_quotient(
        LieQuotient(DependenceGraph(edges)),
        alphabet,
        max_depth=4,
    )
    actual = torch.zeros(
        plan.dimension,
        plan.dimension,
        plan.dimension,
        dtype=torch.float64,
    )
    if plan.bracket_indices.numel():
        actual.index_put_(
            tuple(plan.bracket_indices.T),
            plan.bracket_values,
            accumulate=True,
        )

    for left_index, left_element in enumerate(plan.basis):
        for right_index, right_element in enumerate(plan.basis):
            target_depth = left_element.depth + right_element.depth
            if target_depth > plan.max_depth:
                continue
            left = dict(left_element.terms)
            right = dict(right_element.terms)
            bracket = _independent_trace_product(left, right, alphabet, edges)
            reverse = _independent_trace_product(right, left, alphabet, edges)
            for word, value in reverse.items():
                updated = bracket.get(word, Fraction()) - value
                if updated:
                    bracket[word] = updated
                else:
                    bracket.pop(word, None)

            expected = torch.zeros(plan.dimension, dtype=torch.float64)
            for output_index, element in enumerate(plan.basis):
                if element.depth != target_depth:
                    continue
                coefficient = bracket.get(element.pivot, Fraction())
                if not coefficient:
                    continue
                expected[output_index] = float(coefficient)
                for word, value in element.terms:
                    updated = bracket.get(word, Fraction()) - coefficient * value
                    if updated:
                        bracket[word] = updated
                    else:
                        bracket.pop(word, None)

            assert not bracket
            torch.testing.assert_close(
                actual[left_index, right_index],
                expected,
                atol=0,
                rtol=0,
            )


def test_trace_compiler_does_not_touch_ambient_hall_or_dense_svd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorsig.algebra.hall_bch as hall_bch
    import tensorsig.algebra.hall_projection as hall_projection

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("ambient compiler machinery was called")

    monkeypatch.setattr(hall_bch, "_sparse_constants", fail)
    monkeypatch.setattr(hall_projection, "hall_basis", fail)
    monkeypatch.setattr(torch.linalg, "svd", fail)

    plan = compile_trace_lie_quotient(
        LieQuotient(DependenceGraph(((0, 1),))),
        ProjectedAlphabet(code_dim=1, in_channels=3),
        max_depth=4,
    )

    assert plan.level_dimensions == (3, 1, 2, 3)


def test_trace_normal_form_uses_the_dependence_poset() -> None:
    monoid = _TraceMonoid(
        ProjectedAlphabet(code_dim=1, in_channels=3),
        frozenset({(0, 2)}),
    )

    assert monoid.normal_form((2, 1, 0)) == (1, 2, 0)
    assert monoid.normal_form((1, 2, 0)) == (1, 2, 0)
    assert monoid.normal_form((2, 0, 1)) == (1, 2, 0)


def test_trace_compiler_is_not_limited_to_bch_runtime_depth_four() -> None:
    plan = compile_trace_lie_quotient(
        LieQuotient(DependenceGraph(((0, 1),))),
        ProjectedAlphabet(code_dim=1, in_channels=2),
        max_depth=5,
    )

    assert plan.level_dimensions == (2, 1, 2, 3, 6)


def test_empty_dependence_graph_is_abelian_for_one_letter_per_channel() -> None:
    plan = compile_trace_lie_quotient(
        LieQuotient(DependenceGraph()),
        ProjectedAlphabet(code_dim=1, in_channels=5),
        max_depth=6,
    )

    assert plan.level_dimensions == (5, 0, 0, 0, 0, 0)
    assert plan.bracket_indices.shape == (0, 3)


def test_upper_set_may_remove_every_generator() -> None:
    plan = compile_trace_lie_quotient(
        LieQuotient(DependenceGraph(), ChannelSupport(0)),
        ProjectedAlphabet(code_dim=2, in_channels=3),
        max_depth=4,
    )

    assert plan.dimension == 0
    assert plan.level_dimensions == (0, 0, 0, 0)
    assert plan.generator_map.shape == (6, 0)
    assert plan.bracket_indices.shape == (0, 3)


def test_trace_basis_and_constants_are_exactly_deterministic() -> None:
    quotient = LieQuotient(
        DependenceGraph(((0, 1), (1, 2))),
        ChannelSupport(2),
    )
    alphabet = ProjectedAlphabet(code_dim=1, in_channels=3)

    first = compile_trace_lie_quotient(quotient, alphabet, max_depth=4)
    second = compile_trace_lie_quotient(quotient, alphabet, max_depth=4)

    assert first.basis == second.basis
    torch.testing.assert_close(first.generator_map, second.generator_map)
    torch.testing.assert_close(first.bracket_indices, second.bracket_indices)
    torch.testing.assert_close(first.bracket_values, second.bracket_values)
