"""Exhaustive occurrence-tree catalogues and support reference queries.

These computations enumerate Cayley trees. Scalar and support
queries live in tensorsig.offset_coded_signatures.graph_queries.
"""

import heapq
from collections.abc import Iterator, Sequence
from itertools import product

from tensorsig.offset_coded_signatures._graph._graph_common import (
    validate_channels as _validate_occurrence_channels,
)
from tensorsig.offset_coded_signatures._graph._graph_common import (
    validate_composition as _validate_composition,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    OccurrenceTree,
    _prepare_tree,
    _young_averaged_derivative,
)
from tensorsig.offset_coded_signatures.quotients import DependenceGraph

type ChannelWitness = frozenset[tuple[int, int]]


def _labelled_trees(degree: int) -> Iterator[OccurrenceTree]:
    """Yield labelled trees in Pruefer-code order without retaining them."""
    for code in product(range(degree), repeat=degree - 2):
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
        yield frozenset(edges)


def _tree_is_good(
    composition: tuple[int, ...],
    tree: OccurrenceTree,
) -> bool:
    prepared = _prepare_tree(tree, sum(composition))
    return bool(_young_averaged_derivative(composition, prepared))


def _project_tree(
    tree: OccurrenceTree,
    channels: tuple[int, ...],
) -> ChannelWitness:
    return frozenset(
        (min(channels[left], channels[right]), max(channels[left], channels[right]))
        for left, right in tree
        if channels[left] != channels[right]
    )


def good_tree_catalogue(
    composition: Sequence[int],
) -> tuple[OccurrenceTree, ...]:
    """Return all good labelled occurrence trees for ``composition``.

    Every returned edge is stored as ``(min_vertex, max_vertex)``; that
    ascending direction is the reference orientation used by
    :func:`tree_moment`.  This exact baseline enumerates all
    ``m**(m - 2)`` labelled trees and deliberately materializes the good
    output.  Its worst-case time is ``O(m**(m - 2) * |H_q| * m**3)``
    arithmetic operations and its output memory is ``Theta(g * m)`` for
    ``g`` good trees; enumeration itself is streaming and uncached.
    """
    parts = _validate_composition(composition)
    # ponytail: Cayley enumeration is the exact baseline; replace this scan
    # when a circuit, species, or Young-averaged catalogue compiler closes.
    return tuple(
        tree for tree in _labelled_trees(sum(parts)) if _tree_is_good(parts, tree)
    )


def projected_witness_antichain(
    composition: Sequence[int],
    occurrence_channels: Sequence[int],
) -> tuple[ChannelWitness, ...]:
    """Return the exact minimal projected channel witnesses from IR1 C.7.

    Occurrence vertices follow the consecutive blocks of ``composition``.
    Same-channel tree edges are mandatory and therefore omitted from each
    returned channel-edge set.
    """
    parts = _validate_composition(composition)
    channels = _validate_occurrence_channels(occurrence_channels, sum(parts))
    minimal: set[ChannelWitness] = set()
    for tree in _labelled_trees(sum(parts)):
        if not _tree_is_good(parts, tree):
            continue
        support = _project_tree(tree, channels)
        if not support:
            return (support,)
        if any(previous <= support for previous in minimal):
            continue
        minimal = {previous for previous in minimal if not support < previous}
        minimal.add(support)
    return tuple(sorted(minimal, key=lambda row: (len(row), sorted(row))))


def cell_survives(
    composition: Sequence[int],
    occurrence_channels: Sequence[int],
    graph: DependenceGraph,
) -> bool:
    """Return whether one fine occupancy cell survives on ``graph``.

    This is the exact IR1 C.1/C.7 decision using exhaustive streaming tree
    enumeration.  Candidate supports are filtered against ``graph`` before
    their moments are evaluated, and the scan stops at its first witness.
    Worst-case time remains ``O(m**(m - 2) * |H_q| * m**3)`` arithmetic
    operations, but no catalogue or antichain is materialized or cached.
    """
    if not isinstance(graph, DependenceGraph):
        raise TypeError("graph must be a DependenceGraph.")
    parts = _validate_composition(composition)
    channels = _validate_occurrence_channels(occurrence_channels, sum(parts))
    for tree in _labelled_trees(sum(parts)):
        if _project_tree(tree, channels) <= graph.edges and _tree_is_good(parts, tree):
            return True
    return False
