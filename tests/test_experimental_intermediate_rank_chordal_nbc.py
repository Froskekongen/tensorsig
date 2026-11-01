from fractions import Fraction
from itertools import combinations

import pytest

from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    cell_survives,
)
from tensorsig.offset_coded_signatures import graph_queries
from tensorsig.offset_coded_signatures._graph.adaptive import (
    adaptive_tree_moment,
    composition_is_universally_zero,
    tree_moment_backend,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    tree_moment,
)
from tensorsig.offset_coded_signatures.graph_queries import (
    CellQueryMetrics,
    compile_chordal_nbc,
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


def _graphs(degree: int) -> tuple[DependenceGraph, ...]:
    edges = tuple(combinations(range(degree), 2))
    return tuple(
        DependenceGraph(edge for index, edge in enumerate(edges) if mask >> index & 1)
        for mask in range(1 << len(edges))
    )


@pytest.mark.parametrize(
    ("composition", "expected"),
    [
        ((2,), True),
        ((1, 1), False),
        ((1, 3), True),
        ((3, 1), True),
        ((1, 4), False),
        ((2, 3), False),
    ],
)
def test_universal_zero_classification(
    composition: tuple[int, ...],
    expected: bool,
) -> None:
    assert composition_is_universally_zero(composition) is expected


@pytest.mark.parametrize(
    ("composition", "backend"),
    [
        ((6,), "zero"),
        ((1, 5), "zero"),
        ((1,) * 6, "explicit"),
        ((2, 2, 2), "explicit"),
        ((3, 3), "young_dp"),
        ((4, 2), "young_dp"),
    ],
)
def test_adaptive_backend_selection(
    composition: tuple[int, ...],
    backend: str,
) -> None:
    assert tree_moment_backend(composition) == backend


@pytest.mark.parametrize(
    ("composition", "covers"),
    [
        ((1,) * 6, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))),
        ((2, 2, 2), ((0, 4), (1, 4), (2, 1), (3, 1), (4, 5))),
        ((3, 3), ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))),
        ((4, 2), ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))),
        ((1, 3), ((0, 1), (1, 2), (2, 3))),
    ],
)
def test_adaptive_tree_moment_matches_explicit_oracle(
    composition: tuple[int, ...],
    covers: tuple[tuple[int, int], ...],
) -> None:
    assert adaptive_tree_moment(composition, covers) == tree_moment(
        composition,
        covers,
    )


def test_complete_graph_plan_is_the_factorial_parent_family() -> None:
    graph = DependenceGraph(combinations(range(6), 2))
    plan = compile_chordal_nbc(tuple(range(6)), graph)

    assert plan.tree_count == 120
    assert plan.max_bag_load == 6
    assert len(tuple(plan.trees())) == 120
    assert all(len(tree) == 5 for tree in plan.trees())


def test_ch4_tree_controls_return_exact_witnesses_and_work_counts() -> None:
    zero_graph = DependenceGraph(((0, 1), (2, 3), (1, 2), (1, 4), (1, 5)))
    live_graph = DependenceGraph(((0, 1), (2, 3), (1, 3), (1, 4), (1, 5)))
    channels = tuple(range(6))
    composition = (1,) * 6
    metrics = CellQueryMetrics()

    assert not graph_queries.cell_survives(
        composition, channels, zero_graph, metrics=metrics, backend="chordal_nbc"
    )
    assert metrics == CellQueryMetrics(
        backend="chordal_nbc",
        candidate_trees=1,
        trees_examined=1,
        scalar_backend="explicit",
    )
    witness = graph_queries.cell_witness(
        composition, channels, live_graph, backend="chordal_nbc"
    )
    assert witness == live_graph.edges
    assert adaptive_tree_moment(composition, witness) == Fraction(16, 15)


def test_chordal_nbc_matches_streaming_on_all_small_chordal_graphs() -> None:
    checks = 0
    for degree in range(2, 5):
        channels = tuple(range(degree))
        for graph in _graphs(degree):
            try:
                compile_chordal_nbc(channels, graph)
            except ValueError:
                continue
            for composition in _compositions(degree):
                assert graph_queries.cell_survives(
                    composition, channels, graph, backend="chordal_nbc"
                ) == cell_survives(composition, channels, graph)
                checks += 1
    assert checks == 298


def test_chordal_nbc_matches_repeated_channel_cells() -> None:
    composition = (1, 2, 1)
    channels = (0, 1, 1, 2)
    for graph in _graphs(3):
        assert graph_queries.cell_survives(
            composition, channels, graph, backend="chordal_nbc"
        ) == cell_survives(composition, channels, graph)


def test_chordal_compiler_rejects_disconnected_and_nonchordal_graphs() -> None:
    with pytest.raises(ValueError, match="connected"):
        compile_chordal_nbc((0, 1, 2), DependenceGraph(((0, 1),)))
    with pytest.raises(ValueError, match="chordal"):
        compile_chordal_nbc(
            (0, 1, 2, 3),
            DependenceGraph(((0, 1), (1, 2), (2, 3), (0, 3))),
        )


def test_chordal_nbc_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="DependenceGraph"):
        compile_chordal_nbc((0, 1), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CellQueryMetrics"):
        graph_queries.cell_survives(
            (1, 1),
            (0, 1),
            DependenceGraph(((0, 1),)),
            metrics=object(),  # type: ignore[arg-type]
            backend="chordal_nbc",
        )
