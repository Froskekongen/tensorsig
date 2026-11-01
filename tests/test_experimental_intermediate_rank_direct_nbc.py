from fractions import Fraction
from itertools import combinations, permutations, product

import pytest

from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    _labelled_trees,
    cell_survives,
)
from tensorsig.offset_coded_signatures import graph_queries
from tensorsig.offset_coded_signatures._graph.adaptive import (
    adaptive_tree_moment,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    OccurrenceTree,
)
from tensorsig.offset_coded_signatures.graph_queries import (
    compile_direct_nbc,
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


def _path(
    tree: OccurrenceTree,
    start: int,
    stop: int,
    degree: int,
) -> frozenset[tuple[int, int]]:
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
    result = set()
    vertex = stop
    while parent[vertex] != -1:
        previous = parent[vertex]
        result.add((min(vertex, previous), max(vertex, previous)))
        vertex = previous
    return frozenset(result)


def _oracle_nbc_trees(
    degree: int,
    edge_order: tuple[tuple[int, int], ...],
) -> frozenset[OccurrenceTree]:
    actual_edges = frozenset(edge_order)
    ranks = {edge: rank for rank, edge in enumerate(edge_order)}
    return frozenset(
        tree
        for tree in _labelled_trees(degree)
        if tree <= actual_edges
        and not any(
            all(
                ranks[path_edge] < ranks[higher]
                for path_edge in _path(
                    tree,
                    *higher,
                    degree,
                )
            )
            for higher in actual_edges - tree
        )
    )


def test_k3_low_precedes_high_and_parallel_edges_are_not_coalesced() -> None:
    graph = DependenceGraph(((0, 1), (0, 2), (1, 2)))
    plan = compile_direct_nbc((0, 1, 2), graph)

    assert tuple(plan.trees()) == (
        frozenset(((0, 2), (1, 2))),
        frozenset(((0, 1), (1, 2))),
    )


def test_generation_depth_is_not_limited_by_python_recursion() -> None:
    degree = 46
    plan = compile_direct_nbc((0,) * degree, DependenceGraph())

    tree = next(plan.trees())

    assert len(plan.edge_order) == 1035
    assert len(tree) == degree - 1


def test_direct_generator_matches_nbc_definition_on_all_small_graphs() -> None:
    checks = 0
    for degree in range(2, 5):
        channels = tuple(range(degree))
        for graph in _graphs(degree):
            try:
                plan = compile_direct_nbc(channels, graph)
            except ValueError:
                continue
            generated = tuple(plan.trees())
            assert len(generated) == len(set(generated))
            assert frozenset(generated) == _oracle_nbc_trees(
                degree,
                plan.edge_order,
            )
            checks += 1
    assert checks == 43


@pytest.mark.parametrize(
    "edge_order",
    tuple(permutations(((0, 1), (0, 2), (1, 2)))),
)
def test_direct_generator_respects_every_k3_edge_order(
    edge_order: tuple[tuple[int, int], ...],
) -> None:
    graph = DependenceGraph(edge_order)
    plan = compile_direct_nbc((0, 1, 2), graph, edge_order=edge_order)

    assert frozenset(plan.trees()) == _oracle_nbc_trees(3, edge_order)


def test_direct_nbc_matches_streaming_on_all_small_connected_graphs() -> None:
    checks = 0
    for degree in range(2, 5):
        channels = tuple(range(degree))
        for graph in _graphs(degree):
            try:
                compile_direct_nbc(channels, graph)
            except ValueError:
                continue
            for composition in _compositions(degree):
                assert graph_queries.cell_survives(
                    composition, channels, graph, backend="direct_nbc"
                ) == cell_survives(composition, channels, graph)
                checks += 1
    assert checks == 322


def test_direct_nbc_matches_repeated_channel_cells() -> None:
    composition = (1, 2, 1)
    channels = (0, 1, 1, 2)
    checks = 0
    for graph in _graphs(3):
        try:
            compile_direct_nbc(channels, graph)
        except ValueError:
            continue
        assert graph_queries.cell_survives(
            composition, channels, graph, backend="direct_nbc"
        ) == cell_survives(composition, channels, graph)
        checks += 1
    assert checks == 4


def test_target_specific_replay_matches_exact_moments_with_mixed_arrows() -> None:
    edges = tuple(combinations(range(4), 2))
    edge_order = (edges[5], edges[1], edges[3], edges[0], edges[4], edges[2])
    reference_arrows = tuple(
        (right, left) if index % 2 else (left, right)
        for index, (left, right) in enumerate(reversed(edges))
    )
    plan = compile_direct_nbc(
        (0, 1, 2, 3),
        DependenceGraph(edges),
        edge_order=edge_order,
        reference_arrows=reference_arrows,
    )
    nbc_trees = frozenset(plan.trees())
    nontrivial_replays = 0

    for target, composition in product(_labelled_trees(4), _compositions(4)):
        expansion = plan.replay(target)
        assert all(terminal in nbc_trees for terminal, _ in expansion)
        if expansion != ((target, 1),):
            nontrivial_replays += 1
        assert adaptive_tree_moment(
            composition,
            plan.reference_covers(target),
        ) == sum(
            (
                Fraction(coefficient)
                * adaptive_tree_moment(
                    composition,
                    plan.reference_covers(terminal),
                )
                for terminal, coefficient in expansion
            ),
            start=Fraction(),
        )
    assert nontrivial_replays


def test_witness_uses_the_declared_reference_arrows() -> None:
    edges = ((0, 1), (0, 2), (1, 2))
    arrows = ((1, 0), (0, 2), (2, 1))
    witness = graph_queries.cell_witness(
        (1, 1, 1),
        (0, 1, 2),
        DependenceGraph(edges),
        reference_arrows=arrows,
        backend="direct_nbc",
    )

    assert witness is not None
    plan = compile_direct_nbc(
        (0, 1, 2),
        DependenceGraph(edges),
        reference_arrows=arrows,
    )
    assert adaptive_tree_moment((1, 1, 1), plan.reference_covers(witness))


def test_direct_compiler_validates_connectivity_order_and_arrows() -> None:
    triangle = DependenceGraph(((0, 1), (0, 2), (1, 2)))
    with pytest.raises(ValueError, match="connected"):
        compile_direct_nbc((0, 1, 2), DependenceGraph(((0, 1),)))
    with pytest.raises(ValueError, match="edge_order"):
        compile_direct_nbc(
            (0, 1, 2),
            triangle,
            edge_order=((0, 1), (0, 2)),
        )
    with pytest.raises(ValueError, match="edge_order"):
        compile_direct_nbc(
            (0, 1, 2),
            triangle,
            edge_order=((0, 1), (1, 0), (1, 2)),
        )
    with pytest.raises(ValueError, match="reference_arrows"):
        compile_direct_nbc(
            (0, 1, 2),
            triangle,
            reference_arrows=((0, 1), (0, 2)),
        )
    with pytest.raises(ValueError, match="reference_arrows"):
        compile_direct_nbc(
            (0, 1, 2),
            triangle,
            reference_arrows=((0, 1), (1, 0), (1, 2)),
        )
    with pytest.raises(TypeError, match="DependenceGraph"):
        compile_direct_nbc((0, 1), object())  # type: ignore[arg-type]


def test_disconnected_cell_dies_without_compilation_error() -> None:
    graph = DependenceGraph(((0, 1),))

    assert not graph_queries.cell_survives(
        (1, 1, 1), (0, 1, 2), graph, backend="direct_nbc"
    )
    assert (
        graph_queries.cell_witness((1, 1, 1), (0, 1, 2), graph, backend="direct_nbc")
        is None
    )
    with pytest.raises(ValueError, match="edge_order"):
        graph_queries.cell_survives(
            (1, 1, 1), (0, 1, 2), graph, edge_order=((99, 100),), backend="direct_nbc"
        )


def test_replay_rejects_non_tree_or_non_actual_targets() -> None:
    plan = compile_direct_nbc(
        (0, 1, 2, 3),
        DependenceGraph(((0, 1), (1, 2), (2, 3), (0, 3))),
    )
    with pytest.raises(ValueError, match="exactly m - 1"):
        plan.replay(((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="actual occurrence"):
        plan.replay(((0, 1), (1, 2), (0, 2)))
    with pytest.raises(ValueError, match="duplicates"):
        plan.replay(((0, 1), (1, 0), (2, 3)))
