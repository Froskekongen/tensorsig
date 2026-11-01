import heapq
import itertools
from fractions import Fraction

import pytest

import tensorsig.offset_coded_signatures._graph.tree_moments as explicit
import tensorsig.offset_coded_signatures._graph.young_dp as young_dp
from tensorsig.offset_coded_signatures._graph.young_dp import (
    YoungDpMetrics,
    young_dp_order_polynomial_value,
    young_dp_tree_derivative,
    young_dp_tree_moment,
)


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
    for code in itertools.product(range(degree), repeat=degree - 2):
        degrees = [1] * degree
        for vertex in code:
            degrees[vertex] += 1
        leaves = [vertex for vertex, value in enumerate(degrees) if value == 1]
        heapq.heapify(leaves)
        edges = []
        for vertex in code:
            leaf = heapq.heappop(leaves)
            edges.append((leaf, vertex))
            degrees[leaf] -= 1
            degrees[vertex] -= 1
            if degrees[vertex] == 1:
                heapq.heappush(leaves, vertex)
        edges.append((heapq.heappop(leaves), heapq.heappop(leaves)))
        trees.append(tuple(edges))
    return tuple(trees)


def _orient(
    tree: tuple[tuple[int, int], ...],
    mask: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (upper, lower) if mask >> index & 1 else (lower, upper)
        for index, (lower, upper) in enumerate(tree)
    )


def _explicit_young_order_value(
    composition: tuple[int, ...],
    covers: tuple[tuple[int, int], ...],
    n: int,
) -> Fraction:
    tree = explicit._prepare_tree(covers, sum(composition))
    labelings = tuple(explicit._young_labelings(composition))
    return Fraction(
        sum(
            explicit._order_polynomial_value(tree, labeling, n)
            for labeling in labelings
        ),
        len(labelings),
    )


@pytest.mark.parametrize(
    ("composition", "covers", "values", "derivative"),
    [
        ((2,), ((0, 1),), (Fraction(1, 2), Fraction(2)), Fraction()),
        ((1, 1), ((0, 1),), (Fraction(1), Fraction(3)), Fraction(1, 2)),
        ((1, 1), ((1, 0),), (Fraction(), Fraction(1)), Fraction(-1, 2)),
        (
            (2, 2),
            ((0, 1), (0, 2), (0, 3)),
            (Fraction(1, 2), Fraction(13, 2), Fraction(29), Fraction(85)),
            Fraction(-1, 12),
        ),
        (
            (1, 2, 1),
            ((0, 1), (0, 2), (0, 3)),
            (Fraction(1), Fraction(9), Fraction(36), Fraction(100)),
            Fraction(),
        ),
        (
            (1, 2, 1),
            ((0, 1), (0, 2), (1, 3)),
            (Fraction(1), Fraction(7), Fraction(25), Fraction(65)),
            Fraction(1, 12),
        ),
    ],
)
def test_young_dp_polynomial_value_fixtures(
    composition: tuple[int, ...],
    covers: tuple[tuple[int, int], ...],
    values: tuple[Fraction, ...],
    derivative: Fraction,
) -> None:
    assert (
        tuple(
            young_dp_order_polynomial_value(composition, covers, n)
            for n in range(1, sum(composition) + 1)
        )
        == values
    )
    assert young_dp_tree_derivative(composition, covers) == derivative


def test_young_dp_matches_explicit_polynomials_through_degree_four() -> None:
    checks = 0
    for degree in range(2, 5):
        for composition in _compositions(degree):
            for tree in _labelled_trees(degree):
                for mask in range(1 << (degree - 1)):
                    covers = _orient(tree, mask)
                    for n in range(1, degree + 1):
                        assert young_dp_order_polynomial_value(
                            composition,
                            covers,
                            n,
                        ) == _explicit_young_order_value(composition, covers, n)
                        checks += 1
    assert checks == 4_248


@pytest.mark.parametrize(
    ("composition", "covers"),
    [
        ((1, 1, 1, 2), ((0, 3), (1, 2), (2, 3), (3, 4))),
        ((3, 2), ((0, 1), (2, 1), (1, 3), (3, 4))),
        ((2, 2, 2), ((0, 4), (1, 4), (2, 1), (3, 1), (4, 5))),
        ((3, 1, 2), ((0, 2), (1, 2), (2, 4), (3, 4), (5, 4))),
    ],
)
def test_young_dp_selected_degree_five_and_six_oracles(
    composition: tuple[int, ...],
    covers: tuple[tuple[int, int], ...],
) -> None:
    assert young_dp_tree_derivative(
        composition,
        covers,
    ) == explicit.young_averaged_tree_derivative(composition, covers)


@pytest.mark.parametrize(
    ("tree", "expected"),
    [
        (((0, 1), (2, 1), (1, 4), (1, 5), (3, 2)), Fraction()),
        (((0, 1), (4, 1), (5, 1), (1, 3), (3, 2)), Fraction(-16, 15)),
    ],
)
def test_young_dp_degree_six_strict_controls(
    tree: tuple[tuple[int, int], ...],
    expected: Fraction,
) -> None:
    assert young_dp_tree_moment((1,) * 6, tree) == expected


def test_young_dp_degree_nine_zero() -> None:
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

    assert young_dp_tree_moment((1,) * 9, tree) == 0
    assert young_dp_tree_moment((1,) * 9, reoriented) == 0


def test_root_rank_graft_is_child_order_independent() -> None:
    composition = (4,)
    covers = ((1, 0), (0, 2), (3, 0))
    tree = explicit._prepare_tree(covers, 4)
    directed, children, traversal = tree
    reversed_children = tuple(tuple(reversed(row)) for row in children)
    reordered_tree = (directed, reversed_children, traversal)

    for n in range(1, 5):
        assert young_dp._young_order_polynomial_value_dp(
            composition,
            tree,
            n,
        ) == young_dp._young_order_polynomial_value_dp(
            composition,
            reordered_tree,
            n,
        )


def test_root_rank_singleton_grafts_record_the_boundary_rank() -> None:
    singleton = {(1, 1): Fraction(1)}

    assert young_dp._graft_profiles(singleton, singleton, True) == {(2, 2): Fraction(1)}
    assert young_dp._graft_profiles(singleton, singleton, False) == {
        (2, 1): Fraction(1)
    }


def test_young_dp_metrics_report_exact_state_sizes() -> None:
    metrics = YoungDpMetrics()
    covers = ((0, 1), (1, 2), (2, 3))

    assert young_dp_tree_derivative((2, 2), covers, metrics=metrics) == Fraction(
        1,
        24,
    )
    assert metrics == YoungDpMetrics(
        polynomial_evaluations=4,
        state_cells=52,
        peak_state_cells=7,
        max_numerator_bits=4,
        max_denominator_bits=2,
    )


@pytest.mark.parametrize(
    ("function", "args", "message"),
    [
        (young_dp_tree_moment, ((1, 0), ((0, 1),)), "positive"),
        (
            young_dp_tree_derivative,
            ((1, 1, 1), ((0, 1), (1, 2), (2, 0))),
            "exactly",
        ),
        (
            young_dp_order_polynomial_value,
            ((1, 1), ((0, 1),), -1),
            "non-negative",
        ),
        (
            young_dp_order_polynomial_value,
            ((1, 1), ((0, 1),), 1.5),
            "integer",
        ),
    ],
)
def test_young_dp_rejects_invalid_inputs(
    function: object,
    args: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        function(*args)  # type: ignore[operator]


def test_young_dp_rejects_invalid_metrics() -> None:
    with pytest.raises(TypeError, match="YoungDpMetrics"):
        young_dp_tree_derivative((1, 1), ((0, 1),), metrics=object())  # type: ignore[arg-type]
