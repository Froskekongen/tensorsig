import heapq
import itertools
import math
from fractions import Fraction

import pytest

import tensorsig.offset_coded_signatures._graph.tree_moments as intermediate_rank
from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    cell_survives,
    good_tree_catalogue,
    projected_witness_antichain,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    labelled_tree_order_derivative,
    labelled_tree_order_polynomial_value,
    tree_moment,
    young_averaged_tree_derivative,
)
from tensorsig.offset_coded_signatures.quotients import DependenceGraph


def _compositions(total: int) -> tuple[tuple[int, ...], ...]:
    if total == 0:
        return ((),)
    return tuple(
        (first, *rest)
        for first in range(1, total + 1)
        for rest in _compositions(total - first)
    )


def _labelled_trees(degree: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    trees = []
    for code in itertools.product(range(degree), repeat=max(0, degree - 2)):
        degrees = [1] * degree
        for vertex in code:
            degrees[vertex] += 1
        leaves = [vertex for vertex, value in enumerate(degrees) if value == 1]
        heapq.heapify(leaves)
        edges = []
        for vertex in code:
            leaf = heapq.heappop(leaves)
            edges.append((min(leaf, vertex), max(leaf, vertex)))
            degrees[leaf] -= 1
            degrees[vertex] -= 1
            if degrees[vertex] == 1:
                heapq.heappush(leaves, vertex)
        left = heapq.heappop(leaves)
        right = heapq.heappop(leaves)
        edges.append((min(left, right), max(left, right)))
        trees.append(tuple(sorted(edges)))
    return tuple(trees)


def _young_labelings(composition: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    blocks = []
    start = 0
    for part in composition:
        blocks.append(tuple(range(start, start + part)))
        start += part
    return tuple(
        tuple(label for block in block_labels for label in block)
        for block_labels in itertools.product(
            *(tuple(itertools.permutations(block)) for block in blocks)
        )
    )


def _eulerian_coefficient(permutation: tuple[int, ...]) -> Fraction:
    descents = sum(left > right for left, right in zip(permutation, permutation[1:]))
    return Fraction(
        -1 if descents % 2 else 1,
        len(permutation) * math.comb(len(permutation) - 1, descents),
    )


def _walsh_tree_moment(
    composition: tuple[int, ...],
    arrows: tuple[tuple[int, int], ...],
) -> Fraction:
    degree = sum(composition)
    labelings = _young_labelings(composition)
    total = Fraction()
    for word in itertools.permutations(range(degree)):
        positions = {vertex: index for index, vertex in enumerate(word)}
        amplitude = sum(
            (
                _eulerian_coefficient(tuple(labeling[vertex] for vertex in word))
                for labeling in labelings
            ),
            start=Fraction(),
        ) / len(labelings)
        character = math.prod(
            1 if positions[lower] < positions[upper] else -1 for lower, upper in arrows
        )
        total += amplitude * character
    return total


def _trace_survives(
    composition: tuple[int, ...],
    occurrence_channels: tuple[int, ...],
    graph_edges: frozenset[tuple[int, int]],
) -> bool:
    degree = sum(composition)
    labelings = _young_labelings(composition)
    dependence_edges = tuple(
        (left, right)
        for left in range(degree)
        for right in range(left + 1, degree)
        if occurrence_channels[left] == occurrence_channels[right]
        or (
            min(occurrence_channels[left], occurrence_channels[right]),
            max(occurrence_channels[left], occurrence_channels[right]),
        )
        in graph_edges
    )
    fibres: dict[tuple[bool, ...], Fraction] = {}
    for word in itertools.permutations(range(degree)):
        positions = {vertex: index for index, vertex in enumerate(word)}
        trace = tuple(
            positions[left] < positions[right] for left, right in dependence_edges
        )
        amplitude = sum(
            (
                _eulerian_coefficient(tuple(labeling[vertex] for vertex in word))
                for labeling in labelings
            ),
            start=Fraction(),
        ) / len(labelings)
        fibres[trace] = fibres.get(trace, Fraction()) + amplitude
    return any(fibres.values())


def test_bidirectional_tree_order_polynomial_dp() -> None:
    covers = ((1, 0), (1, 2))
    labeling = (0, 1, 2)

    assert tuple(
        labelled_tree_order_polynomial_value(covers, labeling, n) for n in range(4)
    ) == (0, 0, 2, 8)
    assert labelled_tree_order_derivative(covers, labeling) == Fraction(-1, 3)


def test_young_labelings_yield_before_exhausting_a_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def guarded_permutations(block: tuple[int, ...]):
        yield block
        raise AssertionError("permutations were eagerly pooled")

    monkeypatch.setattr(intermediate_rank, "permutations", guarded_permutations)

    assert next(intermediate_rank._young_labelings((9,))) == tuple(range(9))


def test_tree_moment_matches_independent_walsh_oracle_through_degree_four() -> None:
    for degree in range(2, 5):
        for composition in _compositions(degree):
            for tree in _labelled_trees(degree):
                assert tree_moment(composition, tree) == _walsh_tree_moment(
                    composition,
                    tree,
                )


@pytest.mark.parametrize(
    ("composition", "tree", "expected"),
    [
        ((1, 1), ((0, 1),), Fraction(1)),
        ((2,), ((0, 1),), Fraction()),
        ((2, 2), ((0, 1), (0, 2), (0, 3)), Fraction(-2, 3)),
        ((1, 2, 1), ((0, 1), (0, 2), (0, 3)), Fraction()),
        ((1, 2, 1), ((0, 1), (0, 2), (1, 3)), Fraction(2, 3)),
        ((1, 1, 1, 2), ((0, 3), (1, 2), (2, 3), (3, 4)), Fraction(14, 15)),
        (
            (1,) * 6,
            ((0, 1), (2, 1), (1, 4), (1, 5), (3, 2)),
            Fraction(),
        ),
        (
            (1,) * 6,
            ((0, 1), (4, 1), (5, 1), (1, 3), (3, 2)),
            Fraction(-16, 15),
        ),
        (
            (1,) * 9,
            ((0, 5), (0, 8), (1, 2), (1, 4), (3, 4), (3, 6), (4, 5), (4, 7)),
            Fraction(),
        ),
    ],
)
def test_tree_moment_reference_fixtures(
    composition: tuple[int, ...],
    tree: tuple[tuple[int, int], ...],
    expected: Fraction,
) -> None:
    actual = tree_moment(composition, tree)

    assert isinstance(actual, Fraction)
    assert actual == expected


def test_tree_moment_respects_reference_coordinate_changes() -> None:
    composition = (2, 2)
    tree = ((0, 1), (0, 2), (0, 3))
    expected = tree_moment(composition, tree)

    assert tree_moment(composition, tuple(reversed(tree))) == expected
    assert tree_moment(composition, ((1, 0), *tree[1:])) == -expected

    relabeling = (1, 0, 2, 3)
    relabelled_tree = tuple(
        (relabeling[lower], relabeling[upper]) for lower, upper in tree
    )
    assert tree_moment(composition, relabelled_tree) == expected


def test_degree_nine_zero_is_reference_orientation_independent() -> None:
    tree = (
        (0, 5),
        (0, 8),
        (1, 2),
        (1, 4),
        (3, 4),
        (3, 6),
        (4, 5),
        (4, 7),
    )
    reoriented = tuple(
        (right, left) if index % 2 else (left, right)
        for index, (left, right) in enumerate(reversed(tree))
    )

    assert tree_moment((1,) * 9, tree) == 0
    assert tree_moment((1,) * 9, reoriented) == 0


def test_good_tree_catalogue_censuses() -> None:
    expected_degree_four = {
        (4,): 0,
        (3, 1): 0,
        (1, 3): 0,
        (2, 2): 12,
        (2, 1, 1): 11,
        (1, 2, 1): 8,
        (1, 1, 2): 11,
        (1, 1, 1, 1): 14,
    }
    assert {
        composition: len(good_tree_catalogue(composition))
        for composition in _compositions(4)
    } == expected_degree_four

    assert tuple(len(good_tree_catalogue((1,) * degree)) for degree in (4, 5, 6)) == (
        14,
        125,
        994,
    )


def test_strict_good_trees_fail_basis_exchange() -> None:
    catalogue = set(good_tree_catalogue((1, 1, 1, 1)))
    tree_a = frozenset({(0, 1), (0, 3), (1, 2)})
    tree_b = frozenset({(0, 1), (0, 2), (1, 3)})

    assert tree_a in catalogue
    assert tree_b in catalogue
    assert tree_a - {(1, 2)} | {(0, 2)} not in catalogue
    assert tree_a - {(1, 2)} | {(1, 3)} not in catalogue


def test_projected_witnesses_and_cell_survival() -> None:
    cross_channel = projected_witness_antichain((1, 1), (0, 1))
    same_channel = projected_witness_antichain((1, 1), (0, 0))

    assert cross_channel == (frozenset({(0, 1)}),)
    assert same_channel == (frozenset(),)
    assert not cell_survives((1, 1), (0, 1), DependenceGraph())
    assert cell_survives((1, 1), (0, 1), DependenceGraph({(0, 1)}))
    assert cell_survives((1, 1), (0, 0), DependenceGraph())
    assert not cell_survives((2,), (0, 0), DependenceGraph())


def test_cell_survival_rejects_coefficient_one_deletion_contraction() -> None:
    composition = (1, 2, 1)
    channels = (0, 1, 2, 3)
    edges = {(0, 1), (0, 2), (0, 3), (1, 3)}

    original = cell_survives(composition, channels, DependenceGraph(edges))
    deletion = cell_survives(
        composition,
        channels,
        DependenceGraph(edges - {(0, 1)}),
    )
    contraction = cell_survives(
        composition,
        (0, 0, 1, 2),
        DependenceGraph({(0, 1), (0, 2)}),
    )

    assert (original, deletion, contraction) == (True, True, True)
    assert int(original) != int(deletion) + int(contraction)


def test_cell_survival_distinguishes_coarse_invariant_pair() -> None:
    composition = (1, 4, 1)
    channels = (4, 1, 2, 3, 3, 0)
    edges_zero = {(0, 2), (1, 2), (1, 3), (1, 4)}
    edges_one = {(0, 1), (1, 2), (1, 4), (2, 3)}

    assert cell_survives(composition, channels, DependenceGraph(edges_zero))
    assert not cell_survives(composition, channels, DependenceGraph(edges_one))


def test_cell_survival_matches_independent_trace_oracle() -> None:
    composition = (1, 2, 1)
    channel_assignments = (
        (0, 0, 1, 2),
        (0, 1, 1, 2),
        (0, 1, 2, 0),
        (0, 1, 2, 2),
    )
    channel_edges = ((0, 1), (0, 2), (1, 2))
    for edge_count in range(4):
        for edges in itertools.combinations(channel_edges, edge_count):
            graph = DependenceGraph(edges)
            for channels in channel_assignments:
                assert cell_survives(composition, channels, graph) == _trace_survives(
                    composition,
                    channels,
                    graph.edges,
                )


def test_young_average_has_the_c6_normalization() -> None:
    tree = ((0, 1), (0, 2), (0, 3))
    derivative = young_averaged_tree_derivative((2, 2), tree)

    assert derivative == Fraction(-1, 12)
    assert tree_moment((2, 2), tree) == 8 * derivative


@pytest.mark.parametrize(
    ("function", "args", "message"),
    [
        (tree_moment, ((1, 0), ((0, 1),)), "positive"),
        (tree_moment, ((1, 1, 1), ((0, 1), (1, 2), (2, 0))), "exactly"),
        (tree_moment, ((1, 1, 1), ((0, 1), (1, 0))), "duplicates"),
        (tree_moment, ((1, 1), ((0, 2),)), "0, ..., m - 1"),
        (
            labelled_tree_order_polynomial_value,
            (((0, 1),), (0, 0), 2),
            "permutation",
        ),
        (
            labelled_tree_order_polynomial_value,
            (((0, 1),), (0, 1), -1),
            "non-negative",
        ),
        (cell_survives, ((1, 1), (0, 1), object()), "DependenceGraph"),
    ],
)
def test_experimental_evaluator_rejects_invalid_inputs(
    function: object,
    args: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        function(*args)  # type: ignore[operator]
