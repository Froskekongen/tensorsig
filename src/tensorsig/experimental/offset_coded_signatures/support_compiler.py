"""Enumeration-fed exact support compiler for the IR1 polynomial (B.7).

This module aggregates projected occurrence-tree supports with their exact
``mu(T) ** 2`` weights, minimizes the resulting support family, and stores the
minimal family in a reduced zero-suppressed decision diagram.  It deliberately
retains the Cayley-tree scan; it is a bounded exact baseline, not a compact
catalogue recurrence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from tensorsig.offset_coded_signatures._graph._graph_common import (
    validate_channels as _validate_occurrence_channels,
)
from tensorsig.offset_coded_signatures._graph._graph_common import (
    validate_composition as _validate_composition,
)
from tensorsig.offset_coded_signatures._graph.adaptive import (
    adaptive_tree_moment,
    composition_is_universally_zero,
    tree_moment_backend,
)
from tensorsig.offset_coded_signatures.quotients import DependenceGraph

from .intermediate_rank import (
    ChannelWitness,
    _labelled_trees,
    _project_tree,
)

type ChannelEdge = tuple[int, int]
type _ZDDNode = tuple[int, int, int]


def _support_key(support: ChannelWitness) -> tuple[int, tuple[ChannelEdge, ...]]:
    return len(support), tuple(sorted(support))


def _normalize_support(
    support: Iterable[Sequence[int]],
) -> ChannelWitness:
    pairs = []
    for edge in support:
        try:
            pair = tuple(edge)
        except TypeError as error:
            raise TypeError("support edges must be iterable pairs.") from error
        if len(pair) != 2:
            raise ValueError("support edges must contain two channels.")
        if pair[0] == pair[1]:
            raise ValueError("support edges must join distinct channels.")
        pairs.append(pair)
    return DependenceGraph(pairs).edges


def _minimal_antichain(
    supports: Iterable[ChannelWitness],
) -> tuple[ChannelWitness, ...]:
    minimal: list[ChannelWitness] = []
    for support in sorted(set(supports), key=_support_key):
        if not any(previous <= support for previous in minimal):
            minimal.append(support)
    return tuple(minimal)


@dataclass(frozen=True)
class LexZDD:
    """Immutable lexicographic reduced ZDD for an exact support family.

    Node identifiers ``0`` and ``1`` are the false and true terminals.  Each
    stored node is ``(variable_index, low, high)``; equal nodes are merged and
    nodes with a false high child are suppressed.
    """

    variables: tuple[ChannelEdge, ...]
    nodes: tuple[_ZDDNode, ...]
    root: int

    @classmethod
    def build(
        cls,
        supports: Iterable[Iterable[Sequence[int]]],
    ) -> LexZDD:
        """Build the reduced ZDD from an explicitly supplied support family."""
        family = frozenset(_normalize_support(support) for support in supports)
        variables = tuple(sorted(set().union(*family))) if family else ()
        nodes: list[_ZDDNode] = []
        unique: dict[_ZDDNode, int] = {}
        memo: dict[tuple[int, frozenset[ChannelWitness]], int] = {}

        def build_level(
            level: int,
            remaining: frozenset[ChannelWitness],
        ) -> int:
            if not remaining:
                return 0
            if level == len(variables):
                return int(frozenset() in remaining)

            key = level, remaining
            cached = memo.get(key)
            if cached is not None:
                return cached

            variable = variables[level]
            low_family = frozenset(
                support for support in remaining if variable not in support
            )
            high_family = frozenset(
                support - {variable} for support in remaining if variable in support
            )
            low = build_level(level + 1, low_family)
            high = build_level(level + 1, high_family)
            if high == 0:
                result = low
            else:
                node = level, low, high
                result = unique.get(node, 0)
                if not result:
                    result = len(nodes) + 2
                    nodes.append(node)
                    unique[node] = result
            memo[key] = result
            return result

        root = build_level(0, family)
        return cls(variables=variables, nodes=tuple(nodes), root=root)

    @property
    def node_count(self) -> int:
        """Return the number of stored nodes including both terminals."""
        return len(self.nodes) + 2

    def enumerate_supports(self) -> tuple[ChannelWitness, ...]:
        """Return exactly the support family represented by this ZDD."""
        found: list[ChannelWitness] = []

        def visit(node_id: int, prefix: ChannelWitness) -> None:
            if node_id == 0:
                return
            if node_id == 1:
                found.append(prefix)
                return
            variable_index, low, high = self.nodes[node_id - 2]
            visit(low, prefix)
            visit(high, prefix | {self.variables[variable_index]})

        visit(self.root, frozenset())
        return tuple(sorted(found, key=_support_key))

    def contains_subset(self, graph: DependenceGraph) -> bool:
        """Return whether one represented support is contained in ``graph``."""
        if not isinstance(graph, DependenceGraph):
            raise TypeError("graph must be a DependenceGraph.")
        memo: dict[int, bool] = {}

        def visit(node_id: int) -> bool:
            if node_id < 2:
                return bool(node_id)
            cached = memo.get(node_id)
            if cached is not None:
                return cached
            variable_index, low, high = self.nodes[node_id - 2]
            result = visit(low) or (
                self.variables[variable_index] in graph.edges and visit(high)
            )
            memo[node_id] = result
            return result

        return visit(self.root)

    def one_witness(self, graph: DependenceGraph) -> ChannelWitness | None:
        """Return one represented support contained in ``graph``, if any."""
        if not isinstance(graph, DependenceGraph):
            raise TypeError("graph must be a DependenceGraph.")
        memo: dict[int, ChannelWitness | None] = {}

        def visit(node_id: int) -> ChannelWitness | None:
            if node_id == 0:
                return None
            if node_id == 1:
                return frozenset()
            if node_id in memo:
                return memo[node_id]

            variable_index, low, high = self.nodes[node_id - 2]
            variable = self.variables[variable_index]
            result = None
            if variable in graph.edges:
                suffix = visit(high)
                if suffix is not None:
                    result = suffix | {variable}
            if result is None:
                result = visit(low)
            memo[node_id] = result
            return result

        return visit(self.root)


@dataclass(frozen=True)
class SupportCompilation:
    """Exact Cayley-fed output for one composition and fine occupancy cell."""

    composition: tuple[int, ...]
    occurrence_channels: tuple[int, ...]
    support_weights: tuple[tuple[ChannelWitness, Fraction], ...]
    minimal_supports: tuple[ChannelWitness, ...]
    zdd: LexZDD
    trees_scanned: int
    good_tree_count: int
    scalar_backend: str


def compile_projected_support(
    composition: Sequence[int],
    occurrence_channels: Sequence[int],
) -> SupportCompilation:
    """Compile the exact characteristic-zero B.7 support polynomial.

    Trees with the same projected channel support are aggregated with weight
    ``sum_T mu_q(T) ** 2``.  The returned ZDD represents the divisibility-
    minimal supports.  This reference compiler still scans all
    ``m ** (m - 2)`` labelled occurrence trees.  Tree scalars use the adaptive
    exact evaluator, including the proved universal-zero prefilter.
    """
    parts = _validate_composition(composition)
    degree = sum(parts)
    channels = _validate_occurrence_channels(occurrence_channels, degree)
    scalar_backend = tree_moment_backend(parts)
    if composition_is_universally_zero(parts):
        return SupportCompilation(
            composition=parts,
            occurrence_channels=channels,
            support_weights=(),
            minimal_supports=(),
            zdd=LexZDD.build(()),
            trees_scanned=0,
            good_tree_count=0,
            scalar_backend=scalar_backend,
        )
    weights: dict[ChannelWitness, Fraction] = {}
    trees_scanned = 0
    good_tree_count = 0
    for tree in _labelled_trees(degree):
        trees_scanned += 1
        moment = adaptive_tree_moment(parts, tree)
        if not moment:
            continue
        good_tree_count += 1
        support = _project_tree(tree, channels)
        weights[support] = weights.get(support, Fraction()) + moment * moment

    support_weights = tuple(
        sorted(weights.items(), key=lambda item: _support_key(item[0]))
    )
    minimal_supports = _minimal_antichain(weights)
    return SupportCompilation(
        composition=parts,
        occurrence_channels=channels,
        support_weights=support_weights,
        minimal_supports=minimal_supports,
        zdd=LexZDD.build(minimal_supports),
        trees_scanned=trees_scanned,
        good_tree_count=good_tree_count,
        scalar_backend=scalar_backend,
    )
