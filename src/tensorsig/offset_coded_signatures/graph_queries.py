"""Exact CPU queries for graph-block moments and occurrence-cell support.

These are offline rational computations, independent of tensor device/dtype.
The backend policy is deliberately small; richer Gram/energy/coordinate
experiments have separate APIs and do not change the meaning of a witness.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ._graph._graph_common import (
    ResourceLimit as ResourceLimit,
)
from ._graph._graph_common import (
    is_connected,
    occurrence_adjacency,
    perfect_elimination_order,
    validate_cell,
)
from ._graph.adaptive import (
    adaptive_tree_moment as tree_moment,
)
from ._graph.adaptive import (
    tree_moment_backend,
)
from ._graph.chordal_nbc import (
    ChordalNbcPlan as ChordalNbcPlan,
)
from ._graph.chordal_nbc import (
    _parent_plan,
)
from ._graph.chordal_nbc import (
    compile_chordal_nbc as compile_chordal_nbc,
)
from ._graph.direct_nbc import (
    DirectNbcPlan as DirectNbcPlan,
)
from ._graph.direct_nbc import (
    _compile_direct_nbc,
)
from ._graph.direct_nbc import (
    compile_direct_nbc as compile_direct_nbc,
)
from ._graph.literal_energy import literal_tree_energy as literal_tree_energy
from ._graph.one_centre_gram import (
    one_centre_gram_energy as one_centre_gram_energy,
)
from ._graph.rational_coordinates import (
    compile_graph_block_coordinates as compile_graph_block_coordinates,
)
from ._graph.rational_coordinates import (
    decode_coordinates as decode_coordinates,
)
from ._graph.rational_coordinates import (
    rational as rational,
)
from ._graph.rational_coordinates import (
    replay as replay,
)
from ._graph.star_gram import star_trace_gram_energy as star_trace_gram_energy
from .quotients import DependenceGraph


@dataclass(slots=True)
class CellQueryMetrics:
    """Work counters for a query; candidate count is known for parent-tree plans."""

    backend: str = ""
    candidate_trees: int | None = None
    trees_examined: int = 0
    scalar_backend: str = ""


def _trees(adjacency, parts, metrics):
    """Yield a complete deterministic witness family for validated data."""
    if (len(parts) == 2 and parts[-1] == 1) or sum(map(len, adjacency)) == 2 * (
        len(adjacency) - 1
    ):
        # A tree has only one candidate. Every spanning tree also has a nonzero
        # terminal hook after the zero guard; neither case needs an NBC plan.
        seen = {0}
        todo = [0]
        edges = []
        for vertex in todo:
            for neighbor in sorted(adjacency[vertex]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    todo.append(neighbor)
                    edges.append((vertex, neighbor))
        if metrics is not None:
            metrics.backend = "spanning_tree"
            metrics.candidate_trees = 1
        yield tuple(edges)
        return
    # The measured complete-graph lane favours the lexicographic NBC family:
    # its first scalar witness can be cheaper than the first PEO-parent tree.
    complete = all(len(row) == len(adjacency) - 1 for row in adjacency)
    peo = None if complete else perfect_elimination_order(adjacency)
    if peo is not None:
        plan = _parent_plan(adjacency, peo)
        if metrics is not None:
            metrics.backend = "chordal_nbc"
            metrics.candidate_trees = plan.tree_count
        yield from plan.trees()
        return
    edges = tuple(
        (u, v) for u, row in enumerate(adjacency) for v in sorted(row) if u < v
    )
    plan = DirectNbcPlan(len(adjacency), edges, edges)
    if metrics is not None:
        metrics.backend = "direct_nbc"
    yield from plan.trees()


def cell_witness(
    composition: Sequence[int],
    occurrence_channels: Sequence[int],
    graph: DependenceGraph,
    *,
    max_trees: int | None = None,
    backend: str = "auto",
    metrics: CellQueryMetrics | None = None,
    edge_order: Iterable[Sequence[int]] | None = None,
    reference_arrows: Iterable[Sequence[int]] | None = None,
) -> frozenset[tuple[int, int]] | None:
    """Return an actual nonzero occurrence tree, or None after exact rejection.

    ``max_trees`` limits evaluated candidates; exhaustion raises ResourceLimit.
    The returned edges are unordered. Their ascending orientation has a
    nonzero moment; changing edge directions only changes its sign.

    ``backend`` selects ``auto``, ``direct_nbc`` or ``chordal_nbc``. The latter
    rejects connected nonchordal graphs unless the universal-zero guard already
    decides the cell. Custom edge order/reference arrows require ``direct_nbc``.
    Compiled plans are also exposed for independent reuse.
    Metrics are reset for each query; resource exhaustion never means zero.
    """
    parts, channels = validate_cell(composition, occurrence_channels, graph)
    if max_trees is not None and (type(max_trees) is not int or max_trees < 1):
        raise ValueError("max_trees must be a positive integer or None.")
    if not isinstance(backend, str) or backend not in {
        "auto",
        "direct_nbc",
        "chordal_nbc",
    }:
        raise ValueError("backend must be auto, direct_nbc or chordal_nbc.")
    if metrics is not None and not isinstance(metrics, CellQueryMetrics):
        raise TypeError("metrics must be a CellQueryMetrics instance or None.")
    if backend != "direct_nbc" and (
        edge_order is not None or reference_arrows is not None
    ):
        raise ValueError(
            "edge_order and reference_arrows require backend='direct_nbc'."
        )
    scalar_backend = tree_moment_backend(parts)
    if metrics is not None:
        metrics.backend = backend
        metrics.candidate_trees = None
        metrics.trees_examined = 0
        metrics.scalar_backend = scalar_backend
    # Validate custom edge data even when a zero/disconnected guard can decide.
    direct = (
        _compile_direct_nbc(
            channels, graph, edge_order=edge_order, reference_arrows=reference_arrows
        )
        if backend == "direct_nbc"
        else None
    )
    if scalar_backend == "zero":
        return None
    if backend == "direct_nbc":
        if direct is None:
            return None
        trees = (direct._reference_covers(tree) for tree in direct.trees())
    else:
        adjacency = occurrence_adjacency(channels, graph)
        if not is_connected(adjacency):
            return None
        if backend == "chordal_nbc":
            peo = perfect_elimination_order(adjacency)
            if peo is None:
                raise ValueError("the occurrence dependence graph must be chordal.")
            plan = _parent_plan(adjacency, peo)
            if metrics is not None:
                metrics.candidate_trees = plan.tree_count
            trees = plan.trees()
        else:
            trees = _trees(adjacency, parts, metrics)
    for index, covers in enumerate(trees):
        if max_trees is not None and index >= max_trees:
            raise ResourceLimit(f"evaluated trees exceed {max_trees}.")
        if metrics is not None:
            metrics.trees_examined += 1
        if tree_moment(parts, covers):
            return frozenset(tuple(sorted(edge)) for edge in covers)
    return None


def cell_survives(
    composition: Sequence[int],
    occurrence_channels: Sequence[int],
    graph: DependenceGraph,
    *,
    max_trees: int | None = None,
    backend: str = "auto",
    metrics: CellQueryMetrics | None = None,
    edge_order: Iterable[Sequence[int]] | None = None,
    reference_arrows: Iterable[Sequence[int]] | None = None,
) -> bool:
    """Decide support exactly, using a small deterministic adaptive policy."""
    return (
        cell_witness(
            composition,
            occurrence_channels,
            graph,
            max_trees=max_trees,
            backend=backend,
            metrics=metrics,
            edge_order=edge_order,
            reference_arrows=reference_arrows,
        )
        is not None
    )


__all__ = [
    "tree_moment",
    "CellQueryMetrics",
    "DirectNbcPlan",
    "ChordalNbcPlan",
    "compile_direct_nbc",
    "compile_chordal_nbc",
    "cell_survives",
    "cell_witness",
    "ResourceLimit",
    "literal_tree_energy",
    "star_trace_gram_energy",
    "one_centre_gram_energy",
    "compile_graph_block_coordinates",
    "decode_coordinates",
    "rational",
    "replay",
]
