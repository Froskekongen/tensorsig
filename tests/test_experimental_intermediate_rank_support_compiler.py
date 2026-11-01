import itertools
from fractions import Fraction

import pytest

from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    cell_survives,
    good_tree_catalogue,
)
from tensorsig.experimental.offset_coded_signatures.support_compiler import (
    LexZDD,
    compile_projected_support,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    tree_moment,
)
from tensorsig.offset_coded_signatures.quotients import DependenceGraph


def _project(
    tree: frozenset[tuple[int, int]],
    channels: tuple[int, ...],
) -> frozenset[tuple[int, int]]:
    return frozenset(
        (min(channels[left], channels[right]), max(channels[left], channels[right]))
        for left, right in tree
        if channels[left] != channels[right]
    )


def _graphs(
    edges: tuple[tuple[int, int], ...],
) -> tuple[DependenceGraph, ...]:
    return tuple(
        DependenceGraph(edge for index, edge in enumerate(edges) if mask >> index & 1)
        for mask in range(1 << len(edges))
    )


def test_support_weights_are_the_exact_projected_sums_of_squares() -> None:
    composition = (1, 2, 1)
    channels = (0, 1, 1, 2)
    compiled = compile_projected_support(composition, channels)
    expected: dict[frozenset[tuple[int, int]], Fraction] = {}
    catalogue = good_tree_catalogue(composition)
    for tree in catalogue:
        support = _project(tree, channels)
        moment = tree_moment(composition, sorted(tree))
        expected[support] = expected.get(support, Fraction()) + moment * moment

    assert dict(compiled.support_weights) == expected
    assert all(weight > 0 for _, weight in compiled.support_weights)
    assert compiled.trees_scanned == 16
    assert compiled.good_tree_count == len(catalogue) == 8
    assert dict(compiled.support_weights) == {
        frozenset({(0, 1), (1, 2)}): Fraction(8, 3),
        frozenset({(0, 1), (0, 2), (1, 2)}): Fraction(8, 9),
    }


def test_minimal_supports_have_the_same_upward_closure() -> None:
    compiled = compile_projected_support((1, 4, 1), (4, 1, 2, 3, 3, 0))
    raw = tuple(support for support, _ in compiled.support_weights)
    minimal = compiled.minimal_supports
    edge_universe = tuple(itertools.combinations(range(5), 2))

    assert all(
        not left <= right and not right <= left
        for index, left in enumerate(minimal)
        for right in minimal[index + 1 :]
    )
    assert all(any(witness <= support for witness in minimal) for support in raw)
    for graph in _graphs(edge_universe):
        assert any(support <= graph.edges for support in raw) == any(
            witness <= graph.edges for witness in minimal
        )


@pytest.mark.parametrize(
    ("composition", "channels", "edge_universe"),
    [
        ((1, 2, 1), (0, 1, 1, 2), tuple(itertools.combinations(range(3), 2))),
        ((1, 1, 1, 1), (0, 1, 2, 3), tuple(itertools.combinations(range(4), 2))),
    ],
)
def test_zdd_roundtrip_queries_and_witnesses_match_direct_streaming(
    composition: tuple[int, ...],
    channels: tuple[int, ...],
    edge_universe: tuple[tuple[int, int], ...],
) -> None:
    compiled = compile_projected_support(composition, channels)

    assert compiled.zdd.enumerate_supports() == compiled.minimal_supports
    assert compiled.zdd.node_count == len(compiled.zdd.nodes) + 2
    for graph in _graphs(edge_universe):
        expected = cell_survives(composition, channels, graph)
        witness = compiled.zdd.one_witness(graph)
        assert compiled.zdd.contains_subset(graph) == expected
        if expected:
            assert witness in compiled.minimal_supports
            assert witness is not None and witness <= graph.edges
        else:
            assert witness is None


def test_projected_supports_retain_a_cyclic_nonminimal_witness() -> None:
    compiled = compile_projected_support((2, 2, 2), (0, 0, 1, 1, 2, 2))
    triangle = frozenset({(0, 1), (0, 2), (1, 2)})

    assert compiled.trees_scanned == 1296
    assert compiled.good_tree_count == 918
    assert len(compiled.support_weights) == 4
    assert triangle in dict(compiled.support_weights)
    assert triangle not in compiled.minimal_supports
    assert compiled.minimal_supports == (
        frozenset({(0, 1), (0, 2)}),
        frozenset({(0, 1), (1, 2)}),
        frozenset({(0, 2), (1, 2)}),
    )


def test_retained_q141_pair_is_separated_by_the_compiled_zdd() -> None:
    composition = (1, 4, 1)
    channels = (4, 1, 2, 3, 3, 0)
    compiled = compile_projected_support(composition, channels)
    survives = DependenceGraph({(0, 2), (1, 2), (1, 3), (1, 4)})
    dies = DependenceGraph({(0, 1), (1, 2), (1, 4), (2, 3)})

    assert compiled.zdd.contains_subset(survives)
    assert compiled.zdd.one_witness(survives) is not None
    assert not compiled.zdd.contains_subset(dies)
    assert compiled.zdd.one_witness(dies) is None


def test_zdd_handles_empty_families_and_the_empty_support() -> None:
    false = LexZDD.build(())
    true = LexZDD.build((frozenset(),))
    graph = DependenceGraph()

    assert false.enumerate_supports() == ()
    assert not false.contains_subset(graph)
    assert false.one_witness(graph) is None
    assert true.enumerate_supports() == (frozenset(),)
    assert true.contains_subset(graph)
    assert true.one_witness(graph) == frozenset()


def test_support_compiler_short_circuits_universal_zero_compositions() -> None:
    compiled = compile_projected_support((1, 3), (0, 1, 2, 3))

    assert compiled.scalar_backend == "zero"
    assert compiled.trees_scanned == 0
    assert compiled.good_tree_count == 0
    assert compiled.support_weights == compiled.minimal_supports == ()
    assert not compiled.zdd.contains_subset(DependenceGraph(((0, 1),)))


@pytest.mark.parametrize(
    ("function", "args", "message"),
    [
        (compile_projected_support, ((1, 0), (0,)), "positive"),
        (compile_projected_support, ((1, 1), (0,)), "one channel per vertex"),
        (compile_projected_support, ((1, 1), (0, -1)), "non-negative"),
        (LexZDD.build, (((((0, 0),),)),), "distinct channels"),
    ],
)
def test_support_compiler_rejects_invalid_inputs(
    function: object,
    args: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        function(*args)  # type: ignore[operator]


def test_zdd_queries_require_a_dependence_graph() -> None:
    zdd = LexZDD.build(({(0, 1)},))

    with pytest.raises(TypeError, match="DependenceGraph"):
        zdd.contains_subset(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DependenceGraph"):
        zdd.one_witness(object())  # type: ignore[arg-type]
