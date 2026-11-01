"""Exact tree-poset evaluation and explicit small-Young scalar backend."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from fractions import Fraction
from itertools import permutations
from math import comb, factorial, prod

from ._graph_common import (
    validate_composition as _validate_composition,
)

type DirectedCover = tuple[int, int]
type OccurrenceTree = frozenset[tuple[int, int]]
type _PreparedTree = tuple[
    frozenset[DirectedCover],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]


def _validate_labeling(labeling: Sequence[int]) -> tuple[int, ...]:
    labels = tuple(labeling)
    if not labels:
        raise ValueError("labeling must not be empty.")
    if any(not isinstance(label, int) or isinstance(label, bool) for label in labels):
        raise TypeError("labeling entries must be integers.")
    if set(labels) != set(range(len(labels))):
        raise ValueError("labeling must be a permutation of 0, ..., m - 1.")
    return labels


def _prepare_tree(
    covers: Iterable[tuple[int, int]],
    degree: int,
) -> _PreparedTree:
    directed: list[DirectedCover] = []
    undirected: set[tuple[int, int]] = set()
    adjacency = [set() for _ in range(degree)]

    for edge in covers:
        try:
            pair = tuple(edge)
        except TypeError as error:
            raise TypeError(
                "each cover must be an iterable pair of vertices."
            ) from error
        if len(pair) != 2:
            raise ValueError("each cover must contain two vertices.")
        lower, upper = pair
        if (
            not isinstance(lower, int)
            or isinstance(lower, bool)
            or not isinstance(upper, int)
            or isinstance(upper, bool)
        ):
            raise TypeError("cover vertices must be integers.")
        if not 0 <= lower < degree or not 0 <= upper < degree:
            raise ValueError("cover vertices must lie in 0, ..., m - 1.")
        if lower == upper:
            raise ValueError("a cover cannot be a loop.")

        edge_key = (min(lower, upper), max(lower, upper))
        if edge_key in undirected:
            raise ValueError("the undirected cover edges must not contain duplicates.")
        undirected.add(edge_key)
        directed.append((lower, upper))
        adjacency[lower].add(upper)
        adjacency[upper].add(lower)

    if len(directed) != degree - 1:
        raise ValueError("the undirected cover graph must have exactly m - 1 edges.")

    parents = [-2] * degree
    parents[0] = -1
    traversal = [0]
    for vertex in traversal:
        for neighbor in sorted(adjacency[vertex]):
            if neighbor == parents[vertex]:
                continue
            if parents[neighbor] != -2:
                raise ValueError("the undirected cover graph must be a tree.")
            parents[neighbor] = vertex
            traversal.append(neighbor)
    if len(traversal) != degree:
        raise ValueError("the undirected cover graph must be connected.")

    children = tuple(
        tuple(
            neighbor
            for neighbor in sorted(adjacency[vertex])
            if parents[neighbor] == vertex
        )
        for vertex in range(degree)
    )
    return frozenset(directed), children, tuple(traversal)


def _order_polynomial_value(
    tree: _PreparedTree,
    labeling: tuple[int, ...],
    n: int,
) -> int:
    if n == 0:
        return 0

    covers, children, traversal = tree
    branch_values: list[list[int] | None] = [None] * len(labeling)
    for vertex in reversed(traversal):
        values = [1] * n
        for child in children[vertex]:
            child_values = branch_values[child]
            assert child_values is not None

            prefix = [0] * (n + 1)
            for value in range(1, n + 1):
                prefix[value] = prefix[value - 1] + child_values[value - 1]
            suffix = [0] * (n + 2)
            for value in range(n, 0, -1):
                suffix[value] = suffix[value + 1] + child_values[value - 1]

            if (child, vertex) in covers:
                strict = int(labeling[child] > labeling[vertex])
                for value in range(1, n + 1):
                    values[value - 1] *= prefix[value - strict]
            else:
                strict = int(labeling[vertex] > labeling[child])
                for value in range(1, n + 1):
                    values[value - 1] *= suffix[value + strict]
        branch_values[vertex] = values

    root_values = branch_values[0]
    assert root_values is not None
    return sum(root_values)


def labelled_tree_order_polynomial_value(
    covers: Iterable[tuple[int, int]],
    labeling: Sequence[int],
    n: int,
) -> int:
    """Return the exact labelled order-polynomial value ``Omega(P, omega)(n)``.

    Vertices are ``0, ..., m - 1``.  A cover ``(u, v)`` means ``u <_P v``;
    ``labeling[v]`` is ``omega(v)`` and must be a permutation.  The
    undirected cover graph must be a tree.
    """
    labels = _validate_labeling(labeling)
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer.")
    if n < 0:
        raise ValueError("n must be non-negative.")
    tree = _prepare_tree(covers, len(labels))
    return _order_polynomial_value(tree, labels, n)


def _order_derivative(
    tree: _PreparedTree,
    labeling: tuple[int, ...],
) -> Fraction:
    degree = len(labeling)
    return sum(
        (
            Fraction((-1) ** (n - 1) * comb(degree, n), n)
            * _order_polynomial_value(tree, labeling, n)
        )
        for n in range(1, degree + 1)
    )


def labelled_tree_order_derivative(
    covers: Iterable[tuple[int, int]],
    labeling: Sequence[int],
) -> Fraction:
    """Return ``Omega'_(P, omega)(0)`` exactly for a labelled tree poset."""
    labels = _validate_labeling(labeling)
    tree = _prepare_tree(covers, len(labels))
    return _order_derivative(tree, labels)


def _young_labelings(composition: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    blocks = []
    start = 0
    for part in composition:
        blocks.append(tuple(range(start, start + part)))
        start += part

    labeling = [0] * start

    def visit(block_index: int) -> Iterator[tuple[int, ...]]:
        if block_index == len(blocks):
            yield tuple(labeling)
            return

        block = blocks[block_index]
        for ascending_vertices in permutations(block):
            for rank, vertex in zip(block, ascending_vertices, strict=True):
                labeling[vertex] = rank
            yield from visit(block_index + 1)

    yield from visit(0)


def _young_averaged_derivative(
    composition: tuple[int, ...],
    tree: _PreparedTree,
) -> Fraction:
    # Small Young groups are cheaper to enumerate than to populate the DP.
    # The adaptive dispatcher selects the DP for larger groups.
    total = sum(
        (
            _order_derivative(tree, labeling)
            for labeling in _young_labelings(composition)
        ),
        start=Fraction(),
    )
    return total / prod(factorial(part) for part in composition)


def young_averaged_tree_derivative(
    composition: Sequence[int],
    reference_covers: Iterable[tuple[int, int]],
) -> Fraction:
    """Return the exact reference tree coefficient ``f_T(+)``.

    Composition blocks are consecutive vertex intervals.  Reversing one
    reference cover changes the corresponding top-tree character sign.
    Runtime is ``O(|H_q| m^3)`` arithmetic operations.
    """
    parts = _validate_composition(composition)
    tree = _prepare_tree(reference_covers, sum(parts))
    return _young_averaged_derivative(parts, tree)


def tree_moment(
    composition: Sequence[int],
    reference_covers: Iterable[tuple[int, int]],
) -> Fraction:
    """Return the exact characteristic-zero IR1 moment ``mu_q(T)``.

    The directed covers define the reference-arrow tree poset.  The result is
    ``2**(m - 1)`` times :func:`young_averaged_tree_derivative`.
    """
    parts = _validate_composition(composition)
    degree = sum(parts)
    tree = _prepare_tree(reference_covers, degree)
    return (1 << (degree - 1)) * _young_averaged_derivative(parts, tree)
