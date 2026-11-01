"""Compiled chordal PEO-parent/NBC tree families for exact graph queries."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod

from ..quotients import DependenceGraph
from ._graph_common import (
    is_connected as _is_connected,
)
from ._graph_common import (
    occurrence_adjacency as _occurrence_adjacency,
)
from ._graph_common import (
    perfect_elimination_order as _perfect_elimination_order,
)
from ._graph_common import validate_channels as _validate_occurrence_channels

type DirectedOccurrenceTree = tuple[tuple[int, int], ...]
type _Adjacency = tuple[frozenset[int], ...]


@dataclass(frozen=True, slots=True)
class ChordalNbcPlan:
    """Direct PEO-parent tree family for one connected chordal occurrence graph."""

    peo: tuple[int, ...]
    parent_sets: tuple[tuple[int, ...], ...]
    occurrence_edges: tuple[tuple[int, int], ...]

    @property
    def tree_count(self) -> int:
        """Return ``prod_i |N_+(v_i)|`` for the selected PEO."""
        return prod(len(parents) for parents in self.parent_sets)

    @property
    def max_bag_load(self) -> int:
        """Return ``1 + max_i |N_+(v_i)|`` for the selected PEO."""
        return 1 + max((len(parents) for parents in self.parent_sets), default=0)

    def trees(self) -> Iterator[DirectedOccurrenceTree]:
        """Yield the proved chordal NBC parent trees without a Cayley scan."""
        for parents in product(*self.parent_sets):
            yield tuple(zip(self.peo[:-1], parents, strict=True))


def _compile_chordal_nbc(
    channels: tuple[int, ...],
    graph: DependenceGraph,
) -> ChordalNbcPlan | None:
    adjacency = _occurrence_adjacency(channels, graph)
    if not _is_connected(adjacency):
        return None
    peo = _perfect_elimination_order(adjacency)
    if peo is None:
        raise ValueError("the occurrence dependence graph must be chordal.")
    return _parent_plan(adjacency, peo)


def _parent_plan(adjacency: _Adjacency, peo: tuple[int, ...]) -> ChordalNbcPlan:
    """Compile parent choices for a connected graph with an already checked PEO."""
    positions = {vertex: index for index, vertex in enumerate(peo)}
    parent_sets = tuple(
        tuple(
            sorted(
                neighbor
                for neighbor in adjacency[vertex]
                if positions[neighbor] > index
            )
        )
        for index, vertex in enumerate(peo[:-1])
    )
    assert all(parent_sets)
    occurrence_edges = tuple(
        (left, right)
        for left in range(len(adjacency))
        for right in sorted(adjacency[left])
        if left < right
    )
    return ChordalNbcPlan(peo, parent_sets, occurrence_edges)


def compile_chordal_nbc(
    occurrence_channels: Sequence[int],
    graph: DependenceGraph,
) -> ChordalNbcPlan:
    """Compile the direct parent-tree family for a chordal occurrence graph.

    The occurrence graph joins equal-channel occurrences and lifts every edge
    of ``graph`` to a complete join.  It must be connected and chordal.
    """
    if not isinstance(graph, DependenceGraph):
        raise TypeError("graph must be a DependenceGraph.")
    channels = tuple(occurrence_channels)
    if len(channels) < 2:
        raise ValueError("occurrence_channels must contain at least two vertices.")
    channels = _validate_occurrence_channels(channels, len(channels))
    plan = _compile_chordal_nbc(channels, graph)
    if plan is None:
        raise ValueError("the occurrence dependence graph must be connected.")
    return plan


__all__ = [
    "ChordalNbcPlan",
    "compile_chordal_nbc",
]
