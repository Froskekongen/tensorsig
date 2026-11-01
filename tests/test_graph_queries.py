from fractions import Fraction
from itertools import combinations, product

import pytest

from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    _labelled_trees,
)
from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    cell_survives as reference_cell_survives,
)
from tensorsig.offset_coded_signatures._graph.adaptive import (
    tree_moment_backend,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    tree_moment as reference_tree_moment,
)
from tensorsig.offset_coded_signatures.graph_queries import (
    ResourceLimit,
    cell_survives,
    cell_witness,
    tree_moment,
)
from tensorsig.offset_coded_signatures.quotients import DependenceGraph


def _compositions(total):
    if not total:
        yield ()
    for first in range(1, total + 1):
        for tail in _compositions(total - first):
            yield (first, *tail)


def test_terminal_hook_all_small_trees_and_orientations():
    checks = 0
    for degree in range(2, 5):
        for tree in _labelled_trees(degree):
            edges = sorted(tree)
            for directions in product((False, True), repeat=degree - 1):
                arrows = tuple(
                    (b, a) if reverse else (a, b)
                    for (a, b), reverse in zip(edges, directions)
                )
                assert tree_moment((degree - 1, 1), arrows) == reference_tree_moment(
                    (degree - 1, 1), arrows
                )
                checks += 1
    assert checks == 142
    assert tree_moment_backend((4, 1)) == "terminal_hook"
    assert tree_moment((1, 1), ((0, 1),)) == 1
    assert tree_moment((4, 1), tuple((v, 4) for v in range(4))) == Fraction(-8, 15)


def test_hook_large_shapes_against_independent_young_dp():
    from tensorsig.offset_coded_signatures._graph.young_dp import (
        young_dp_tree_moment,
    )

    for degree in (5, 7):
        for edges in (
            tuple((i, i + 1) for i in range(degree - 1)),
            tuple((0, i) for i in range(1, degree)),
            tuple((i, degree - 1) for i in range(degree - 1)),
            tuple(((i - 1) // 2, i) for i in range(1, degree)),
        ):
            for pattern in range(4):
                arrows = tuple(
                    (b, a) if (i + pattern) % 4 < 2 else (a, b)
                    for i, (a, b) in enumerate(edges)
                )
                assert tree_moment((degree - 1, 1), arrows) == young_dp_tree_moment(
                    (degree - 1, 1), arrows
                )


def test_adaptive_cells_all_graphs_through_degree_four():
    checks = 0
    for degree in range(2, 5):
        edges = tuple(combinations(range(degree), 2))
        for mask in range(1 << len(edges)):
            graph = DependenceGraph(
                edge for i, edge in enumerate(edges) if mask >> i & 1
            )
            for composition in _compositions(degree):
                expected = reference_cell_survives(
                    composition, tuple(range(degree)), graph
                )
                assert (
                    cell_survives(composition, tuple(range(degree)), graph) == expected
                )
                witness = cell_witness(composition, tuple(range(degree)), graph)
                assert (witness is not None) == expected
                if witness:
                    assert witness <= graph.edges
                    assert reference_tree_moment(composition, witness)
                checks += 1
    assert checks == 548


def test_occupancy_to_occurrence_and_refusal():
    # Rows are offsets, columns are channels; repeat each channel by occupancy.
    occupancy = ((1, 0, 0), (0, 2, 0), (0, 0, 1))
    composition = tuple(sum(row) for row in occupancy)
    channels = tuple(
        c for row in occupancy for c, count in enumerate(row) for _ in range(count)
    )
    graph = DependenceGraph(((0, 1), (1, 2)))
    assert cell_survives(composition, channels, graph) == reference_cell_survives(
        composition, channels, graph
    )
    with pytest.raises(ValueError, match="positive integer"):
        cell_survives((1, 1), (0, 1), graph, max_trees=0)
    with pytest.raises(ValueError, match="one channel"):
        cell_survives((2,), (0,), graph)
    # This connected zero cell requires checking several NBC trees.
    zero = DependenceGraph(((0, 1), (0, 2), (0, 3)))
    assert not cell_survives((1, 1, 1, 1), (0, 1, 2, 3), zero)
    with pytest.raises(ResourceLimit, match="evaluated trees"):
        cell_survives(
            (1, 1, 1, 1),
            (0, 1, 2, 3),
            DependenceGraph(combinations(range(4), 2)),
            max_trees=1,
        )
    assert issubclass(ResourceLimit, RuntimeError)
