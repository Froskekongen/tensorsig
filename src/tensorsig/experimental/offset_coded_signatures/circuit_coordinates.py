"""Finite exact A.1 circuit-coordinate compression experiment.

The compiler enumerates every labelled tree and every spanning unicyclic
graph, then row-reduces their signed circuit relations over the rationals.
It is a reconstruction experiment, not a compact all-degree catalogue
compiler: its input and compilation state are still Cayley-sized.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations

from tensorsig.offset_coded_signatures._graph._graph_common import (
    signed_unicyclic_relation as _signed_unicyclic_relation,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import OccurrenceTree

from .intermediate_rank import _labelled_trees

type SignedTreeRelation = tuple[tuple[OccurrenceTree, int], ...]
type _SparseRow = dict[int, Fraction]
type _Reducer = tuple[int, tuple[tuple[int, Fraction], ...]]


def _validate_degree(degree: int) -> int:
    if not isinstance(degree, int) or isinstance(degree, bool):
        raise TypeError("degree must be an integer.")
    if degree < 2:
        raise ValueError("degree must be at least two.")
    return degree


def _relations_from_trees(
    degree: int,
    trees: Iterable[OccurrenceTree],
) -> Iterator[SignedTreeRelation]:
    all_edges = tuple(combinations(range(degree), 2))
    seen: set[OccurrenceTree] = set()
    for tree in trees:
        for edge in all_edges:
            if edge in tree:
                continue
            unicyclic = tree | {edge}
            if unicyclic in seen:
                continue
            seen.add(unicyclic)
            yield _signed_unicyclic_relation(unicyclic, degree)


def signed_unicyclic_relations(degree: int) -> Iterator[SignedTreeRelation]:
    """Yield every signed spanning-unicyclic A.1 relation exactly once.

    Edges use the ascending global reference arrow.  The unique cycle starts
    at its least vertex and first follows its lesser cycle neighbour.  The
    coefficient of ``U - e`` is ``+1`` when that traversal agrees with the
    global arrow on ``e`` and ``-1`` otherwise.  Reversing the traversal
    would only negate the whole relation.
    """
    degree = _validate_degree(degree)
    yield from _relations_from_trees(degree, _labelled_trees(degree))


def _subtract_multiple(
    target: _SparseRow,
    source: _SparseRow,
    multiple: Fraction,
) -> None:
    for column, coefficient in source.items():
        updated = target.get(column, Fraction()) - multiple * coefficient
        if updated:
            target[column] = updated
        else:
            target.pop(column, None)


def _rref(
    rows: Iterable[_SparseRow],
) -> tuple[
    dict[int, _SparseRow],
    int,
    int,
    int,
    int,
    int,
]:
    pivots: dict[int, _SparseRow] = {}
    relation_count = 0
    raw_nonzeros = 0
    max_width = 0
    max_numerator_bits = 0
    max_denominator_bits = 0

    def observe(row: _SparseRow) -> None:
        nonlocal max_width, max_numerator_bits, max_denominator_bits
        max_width = max(max_width, len(row))
        for coefficient in row.values():
            max_numerator_bits = max(
                max_numerator_bits,
                abs(coefficient.numerator).bit_length(),
            )
            max_denominator_bits = max(
                max_denominator_bits,
                coefficient.denominator.bit_length(),
            )

    for row in rows:
        relation_count += 1
        raw_nonzeros += len(row)
        observe(row)
        for pivot, reducer in sorted(pivots.items()):
            coefficient = row.get(pivot)
            if coefficient is not None:
                _subtract_multiple(row, reducer, coefficient)
                observe(row)
        if not row:
            continue

        pivot = min(row)
        leading = row[pivot]
        row = {column: value / leading for column, value in row.items()}
        observe(row)
        for reducer in pivots.values():
            coefficient = reducer.get(pivot)
            if coefficient is not None:
                _subtract_multiple(reducer, row, coefficient)
                observe(reducer)
        pivots[pivot] = row

    return (
        pivots,
        relation_count,
        raw_nonzeros,
        max_width,
        max_numerator_bits,
        max_denominator_bits,
    )


@dataclass(frozen=True, slots=True)
class CircuitCoordinatePlan:
    """Exact finite-degree reconstruction from free tree coordinates."""

    degree: int
    trees: tuple[OccurrenceTree, ...] = field(repr=False)
    basis_indices: tuple[int, ...]
    reducers: tuple[_Reducer, ...] = field(repr=False)
    raw_relation_count: int
    relation_rank: int
    raw_relation_nonzeros: int
    max_row_width: int
    max_numerator_bits: int
    max_denominator_bits: int

    @property
    def basis_trees(self) -> tuple[OccurrenceTree, ...]:
        """Return the free coordinates in reconstruction input order."""
        return tuple(self.trees[index] for index in self.basis_indices)

    def reconstruct(
        self,
        basis_values: Sequence[Fraction | int],
    ) -> tuple[Fraction, ...]:
        """Reconstruct all tree coordinates from exact basis values."""
        exact_values = tuple(Fraction(value) for value in basis_values)
        if len(exact_values) != len(self.basis_indices):
            raise ValueError(
                f"expected {len(self.basis_indices)} basis values, "
                f"got {len(exact_values)}."
            )

        coordinates = [Fraction()] * len(self.trees)
        for index, value in zip(self.basis_indices, exact_values, strict=True):
            coordinates[index] = value
        for pivot, terms in self.reducers:
            coordinates[pivot] = -sum(
                (coefficient * coordinates[column] for column, coefficient in terms),
                start=Fraction(),
            )
        return tuple(coordinates)


def compile_tree_circuit_coordinates(degree: int) -> CircuitCoordinatePlan:
    """Compile an exact rational RREF plan for all degree-``degree`` trees.

    The compilation deliberately enumerates all ``degree**(degree - 2)``
    labelled trees.  Sparse elimination can also incur fill-in and rational
    coefficient growth; callers should retain the returned plan if reusing it.
    """
    degree = _validate_degree(degree)
    # ponytail: this Cayley-sized scan is the finite experiment; replace it
    # only after an all-degree circuit compiler is proved.
    trees = tuple(_labelled_trees(degree))
    tree_indices = {tree: index for index, tree in enumerate(trees)}
    rows = (
        {tree_indices[tree]: Fraction(coefficient) for tree, coefficient in relation}
        for relation in _relations_from_trees(degree, trees)
    )
    (
        pivots,
        relation_count,
        raw_nonzeros,
        max_width,
        max_numerator_bits,
        max_denominator_bits,
    ) = _rref(rows)
    pivot_indices = frozenset(pivots)
    basis_indices = tuple(
        index for index in range(len(trees)) if index not in pivot_indices
    )
    reducers = tuple(
        (
            pivot,
            tuple(
                (column, coefficient)
                for column, coefficient in sorted(row.items())
                if column != pivot
            ),
        )
        for pivot, row in sorted(pivots.items())
    )
    assert all(column in basis_indices for _, terms in reducers for column, _ in terms)

    return CircuitCoordinatePlan(
        degree=degree,
        trees=trees,
        basis_indices=basis_indices,
        reducers=reducers,
        raw_relation_count=relation_count,
        relation_rank=len(pivots),
        raw_relation_nonzeros=raw_nonzeros,
        max_row_width=max_width,
        max_numerator_bits=max_numerator_bits,
        max_denominator_bits=max_denominator_bits,
    )


__all__ = [
    "CircuitCoordinatePlan",
    "compile_tree_circuit_coordinates",
    "signed_unicyclic_relations",
]
