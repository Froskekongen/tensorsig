"""Direct ordered NBC generation and signed target-tree replay."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from ..quotients import DependenceGraph
from ._graph_common import (
    is_connected as _is_connected,
)
from ._graph_common import (
    occurrence_adjacency as _occurrence_adjacency,
)
from ._graph_common import signed_unicyclic_relation as _signed_unicyclic_relation
from ._graph_common import validate_channels as _validate_occurrence_channels
from .tree_moments import (
    OccurrenceTree,
    _prepare_tree,
)

type OccurrenceEdge = tuple[int, int]
type DirectedOccurrenceEdge = tuple[int, int]
type SignedTreeExpansion = tuple[tuple[OccurrenceTree, int], ...]
type _Partition = tuple[int, ...]


def _normalize_edge(
    edge: Sequence[int],
    degree: int,
    *,
    name: str,
) -> OccurrenceEdge:
    try:
        pair = tuple(edge)
    except TypeError as error:
        raise TypeError(f"{name} entries must be iterable vertex pairs.") from error
    if len(pair) != 2:
        raise ValueError(f"{name} entries must contain two vertices.")
    left, right = pair
    if (
        not isinstance(left, int)
        or isinstance(left, bool)
        or not isinstance(right, int)
        or isinstance(right, bool)
    ):
        raise TypeError(f"{name} vertices must be integers.")
    if not 0 <= left < degree or not 0 <= right < degree:
        raise ValueError(f"{name} vertices must lie in 0, ..., m - 1.")
    if left == right:
        raise ValueError(f"{name} entries cannot be loops.")
    return min(left, right), max(left, right)


def _actual_edges(adjacency: tuple[frozenset[int], ...]) -> frozenset[OccurrenceEdge]:
    return frozenset(
        (left, right)
        for left, neighbors in enumerate(adjacency)
        for right in neighbors
        if left < right
    )


def _validate_edge_order(
    edge_order: Iterable[Sequence[int]] | None,
    actual_edges: frozenset[OccurrenceEdge],
    degree: int,
) -> tuple[OccurrenceEdge, ...]:
    if edge_order is None:
        return tuple(sorted(actual_edges))
    ordered = tuple(
        _normalize_edge(edge, degree, name="edge_order") for edge in edge_order
    )
    if len(ordered) != len(actual_edges) or frozenset(ordered) != actual_edges:
        raise ValueError(
            "edge_order must contain every actual occurrence edge exactly once."
        )
    return ordered


def _validate_reference_arrows(
    reference_arrows: Iterable[Sequence[int]] | None,
    edge_order: tuple[OccurrenceEdge, ...],
    degree: int,
) -> tuple[DirectedOccurrenceEdge, ...]:
    if reference_arrows is None:
        return edge_order
    by_edge: dict[OccurrenceEdge, DirectedOccurrenceEdge] = {}
    for arrow in reference_arrows:
        try:
            directed = tuple(arrow)
        except TypeError as error:
            raise TypeError(
                "reference_arrows entries must be iterable vertex pairs."
            ) from error
        edge = _normalize_edge(directed, degree, name="reference_arrows")
        if edge in by_edge:
            raise ValueError(
                "reference_arrows must orient every actual occurrence edge "
                "exactly once."
            )
        by_edge[edge] = directed
    if frozenset(by_edge) != frozenset(edge_order):
        raise ValueError(
            "reference_arrows must orient every actual occurrence edge exactly once."
        )
    return tuple(by_edge[edge] for edge in edge_order)


def _suffix_is_connected(
    partition: _Partition,
    suffix: tuple[OccurrenceEdge, ...],
) -> bool:
    blocks = set(partition)
    if len(blocks) == 1:
        return True
    adjacency = {block: set() for block in blocks}
    for left, right in suffix:
        left_block = partition[left]
        right_block = partition[right]
        if left_block != right_block:
            adjacency[left_block].add(right_block)
            adjacency[right_block].add(left_block)
    seen = {partition[0]}
    todo = list(seen)
    while todo:
        for neighbor in adjacency[todo.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                todo.append(neighbor)
    return seen == blocks


def _merge_blocks(partition: _Partition, left: int, right: int) -> _Partition:
    left_block = partition[left]
    right_block = partition[right]
    representative = min(left_block, right_block)
    return tuple(
        representative if block in (left_block, right_block) else block
        for block in partition
    )


def _tree_path(
    tree: OccurrenceTree,
    start: int,
    stop: int,
    degree: int,
) -> tuple[OccurrenceEdge, ...]:
    adjacency = [set() for _ in range(degree)]
    for left, right in tree:
        adjacency[left].add(right)
        adjacency[right].add(left)
    parent = {start: -1}
    todo = [start]
    while stop not in parent:
        vertex = todo.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in parent:
                parent[neighbor] = vertex
                todo.append(neighbor)
    path = []
    vertex = stop
    while parent[vertex] != -1:
        previous = parent[vertex]
        path.append((min(vertex, previous), max(vertex, previous)))
        vertex = previous
    return tuple(path)


@dataclass(frozen=True, slots=True)
class DirectNbcPlan:
    """One exact ordered actual-graph NBC family and its reference arrows."""

    degree: int
    edge_order: tuple[OccurrenceEdge, ...]
    reference_arrows: tuple[DirectedOccurrenceEdge, ...]

    def trees(self) -> Iterator[OccurrenceTree]:
        """Yield every NBC spanning tree once, with LOW before HIGH."""
        edges = self.edge_order
        stack: list[tuple[int, tuple[OccurrenceEdge, ...], _Partition]] = [
            (0, (), tuple(range(self.degree)))
        ]
        while stack:
            index, selected, partition = stack.pop()
            if index == len(edges):
                yield frozenset(selected)
                continue

            left, right = edges[index]
            suffix = edges[index + 1 :]
            left_block = partition[left]
            right_block = partition[right]
            if not any(
                (
                    partition[higher_left] == left_block
                    and partition[higher_right] == right_block
                )
                or (
                    partition[higher_left] == right_block
                    and partition[higher_right] == left_block
                )
                for higher_left, higher_right in suffix
            ):
                stack.append(
                    (
                        index + 1,
                        (*selected, (left, right)),
                        _merge_blocks(partition, left, right),
                    )
                )
            if _suffix_is_connected(partition, suffix):
                stack.append((index + 1, selected, partition))

    def replay(self, target: Iterable[Sequence[int]]) -> SignedTreeExpansion:
        """Express one actual spanning-tree moment in the NBC family exactly."""
        supplied = tuple(
            _normalize_edge(edge, self.degree, name="target") for edge in target
        )
        tree = frozenset(supplied)
        if len(supplied) != len(tree):
            raise ValueError("target edges must not contain duplicates.")
        if not tree <= frozenset(self.edge_order):
            raise ValueError("target must use only actual occurrence edges.")
        _prepare_tree(tree, self.degree)

        ranks = {edge: rank for rank, edge in enumerate(self.edge_order)}
        arrow_signs = {
            edge: 1 if arrow == edge else -1
            for edge, arrow in zip(
                self.edge_order,
                self.reference_arrows,
                strict=True,
            )
        }

        def tree_key(value: OccurrenceTree) -> tuple[int, ...]:
            return tuple(sorted(ranks[edge] for edge in value))

        rewrites: dict[
            OccurrenceTree,
            tuple[tuple[OccurrenceTree, int], ...],
        ] = {}
        pending = [tree]
        while pending:
            value = pending.pop()
            if value in rewrites:
                continue
            reduction: tuple[OccurrenceEdge, tuple[OccurrenceEdge, ...]] | None = None
            for higher in reversed(self.edge_order):
                if higher in value:
                    continue
                path = _tree_path(value, *higher, self.degree)
                if all(ranks[edge] < ranks[higher] for edge in path):
                    reduction = higher, path
                    break
            if reduction is None:
                rewrites[value] = ()
                continue

            higher, path = reduction
            unicyclic = value | {higher}
            deltas = {}
            for relation_tree, coefficient in _signed_unicyclic_relation(
                unicyclic,
                self.degree,
            ):
                omitted = next(iter(unicyclic - relation_tree))
                deltas[omitted] = coefficient * arrow_signs[omitted]

            children = tuple(
                (
                    frozenset((value - {removed}) | {higher}),
                    -deltas[higher] * deltas[removed],
                )
                for removed in path
            )
            rewrites[value] = children
            pending.extend(child for child, _ in children)

        expansions: dict[OccurrenceTree, SignedTreeExpansion] = {}
        for value in sorted(
            rewrites,
            key=lambda item: sum(ranks[edge] for edge in item),
            reverse=True,
        ):
            children = rewrites[value]
            if not children:
                expansions[value] = ((value, 1),)
                continue
            totals: dict[OccurrenceTree, int] = {}
            for child, local_coefficient in children:
                for terminal, coefficient in expansions[child]:
                    totals[terminal] = (
                        totals.get(terminal, 0) + local_coefficient * coefficient
                    )
            expansions[value] = tuple(
                (terminal, coefficient)
                for terminal, coefficient in sorted(
                    totals.items(), key=lambda item: tree_key(item[0])
                )
                if coefficient
            )
        return expansions[tree]

    def reference_covers(
        self, tree: OccurrenceTree
    ) -> tuple[DirectedOccurrenceEdge, ...]:
        """Orient a generated or replayed tree by the stored reference arrows."""
        prepared = _prepare_tree(tree, self.degree)
        tree = frozenset(tuple(sorted(edge)) for edge in prepared[0])
        if not tree <= frozenset(self.edge_order):
            raise ValueError("tree must use only actual occurrence edges.")
        return self._reference_covers(tree)

    def _reference_covers(
        self, tree: OccurrenceTree
    ) -> tuple[DirectedOccurrenceEdge, ...]:
        """Orient an internally generated, already valid tree."""
        return tuple(
            arrow
            for edge, arrow in zip(
                self.edge_order,
                self.reference_arrows,
                strict=True,
            )
            if edge in tree
        )


def _compile_direct_nbc(
    occurrence_channels: Sequence[int],
    graph: DependenceGraph,
    *,
    edge_order: Iterable[Sequence[int]] | None = None,
    reference_arrows: Iterable[Sequence[int]] | None = None,
) -> DirectNbcPlan | None:
    """Compile Section 10.1 for one connected actual occurrence graph."""
    if not isinstance(graph, DependenceGraph):
        raise TypeError("graph must be a DependenceGraph.")
    channels = tuple(occurrence_channels)
    if len(channels) < 2:
        raise ValueError("occurrence_channels must contain at least two vertices.")
    channels = _validate_occurrence_channels(channels, len(channels))
    adjacency = _occurrence_adjacency(channels, graph)
    ordered = _validate_edge_order(
        edge_order,
        _actual_edges(adjacency),
        len(channels),
    )
    arrows = _validate_reference_arrows(reference_arrows, ordered, len(channels))
    if not _is_connected(adjacency):
        return None
    return DirectNbcPlan(len(channels), ordered, arrows)


def compile_direct_nbc(
    occurrence_channels: Sequence[int],
    graph: DependenceGraph,
    *,
    edge_order: Iterable[Sequence[int]] | None = None,
    reference_arrows: Iterable[Sequence[int]] | None = None,
) -> DirectNbcPlan:
    """Compile the exact NBC family of a connected actual occurrence graph."""
    plan = _compile_direct_nbc(
        occurrence_channels,
        graph,
        edge_order=edge_order,
        reference_arrows=reference_arrows,
    )
    if plan is None:
        raise ValueError("the occurrence dependence graph must be connected.")
    return plan


__all__ = [
    "DirectNbcPlan",
    "SignedTreeExpansion",
    "compile_direct_nbc",
]
