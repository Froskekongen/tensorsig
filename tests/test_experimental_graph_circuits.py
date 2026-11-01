from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from math import factorial

import pytest

from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    _labelled_trees,
)
from tensorsig.offset_coded_signatures._graph._graph_common import (
    ResourceLimit,
)
from tensorsig.offset_coded_signatures._graph.literal_energy import (
    literal_tree_energy,
)
from tensorsig.offset_coded_signatures._graph.star_gram import (
    star_trace_gram_energy,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    tree_moment,
)
from tensorsig.offset_coded_signatures.graph_queries import cell_survives
from tensorsig.offset_coded_signatures.quotients import DependenceGraph


def _compositions(total):
    if total == 0:
        yield ()
    for first in range(1, total + 1):
        for tail in _compositions(total - first):
            yield (first, *tail)


def _trace_log_norm(composition, channels, centre=0):
    """Independent direct word expansion of log(prod exp), then trace merging."""
    blocks = tuple(i for i, n in enumerate(composition) for _ in range(n))
    target = Counter(zip(blocks, channels))
    degree = len(channels)

    def multiply(left, right):
        result = {}
        for a, x in left.items():
            for b, y in right.items():
                word = a + b
                if len(word) > degree or any(
                    n > target[t] for t, n in Counter(word).items()
                ):
                    continue
                result[word] = result.get(word, Fraction()) + x * y
        return result

    group = {(): Fraction(1)}
    for offset, length in enumerate(composition):
        letters = tuple(t for t in target if t[0] == offset)
        exponential = {(): Fraction(1)}
        for n in range(1, length + 1):
            for word in product(letters, repeat=n):
                if all(count <= target[t] for t, count in Counter(word).items()):
                    exponential[word] = Fraction(1, factorial(n))
        group = multiply(group, exponential)
    del group[()]
    power = {(): Fraction(1)}
    logarithm = {}
    for n in range(1, degree + 1):
        power = multiply(power, group)
        for word, value in power.items():
            if Counter(word) == target:
                logarithm[word] = (
                    logarithm.get(word, Fraction())
                    + Fraction((-1) ** (n + 1), n) * value
                )
    leaves = sorted(set(channels) - {centre})
    traces = {}
    for word, value in logarithm.items():
        centres = tuple(offset for offset, channel in word if channel == centre)
        gap_words = {c: [[] for _ in range(len(centres) + 1)] for c in leaves}
        gap = 0
        for offset, channel in word:
            if channel == centre:
                gap += 1
            else:
                gap_words[channel][gap].append(offset)
        key = centres, tuple(tuple(tuple(w) for w in gap_words[c]) for c in leaves)
        traces[key] = traces.get(key, Fraction()) + value
    return sum((x * x for x in traces.values()), Fraction())


def test_fixed_centre_gram_against_literal_log():
    checks = 0
    for degree in (2, 3):
        for channels in product((0, 1), repeat=degree):
            if 0 not in channels:
                continue
            for composition in _compositions(degree):
                graph = DependenceGraph(((0, 1),))
                expected = _trace_log_norm(composition, channels)
                actual = star_trace_gram_energy(
                    composition, channels, graph, centre_channel=0
                )
                assert actual == expected
                assert bool(actual) == cell_survives(composition, channels, graph)
                checks += 1
    assert checks == 34
    # Repeated centre types, distinct centre types, two commuting leaf channels,
    # and three-centre skeletons with cuts through equal-type runs.
    for composition, channels in (
        ((2, 2), (0, 0, 1, 2)),
        ((1, 2, 1), (0, 1, 2, 0)),
        ((1, 1, 1, 1), (0, 1, 0, 2)),
        ((1, 1, 1, 1), (0, 0, 0, 1)),
        ((3, 1), (0, 0, 0, 1)),
        ((1, 3), (1, 0, 0, 0)),
        ((1, 2, 1), (0, 1, 1, 0)),
        ((4, 1), (0, 0, 0, 0, 1)),
        ((1, 1, 1, 1), (0, 0, 0, 0)),
    ):
        graph = DependenceGraph(((0, 1), (0, 2)))
        actual = star_trace_gram_energy(composition, channels, graph, centre_channel=0)
        assert actual == _trace_log_norm(composition, channels)
        assert bool(actual) == cell_survives(composition, channels, graph)


def _literal_tree_sum(composition, graph):
    return sum(
        (
            tree_moment(composition, tree) ** 2
            for tree in _labelled_trees(sum(composition))
            if tree <= graph.edges
        ),
        Fraction(),
    )


def test_le6_against_all_tree_energy_and_peo_choices():
    checks = 0
    for degree in (2, 3):
        edges = tuple(combinations(range(degree), 2))
        for mask in range(1 << len(edges)):
            graph = DependenceGraph(
                edge for i, edge in enumerate(edges) if mask >> i & 1
            )
            for composition in _compositions(degree):
                expected = _literal_tree_sum(composition, graph)
                assert (
                    literal_tree_energy(composition, tuple(range(degree)), graph)
                    == expected
                )
                checks += 1
    assert checks == 36
    graph = DependenceGraph(combinations(range(3), 2))
    assert literal_tree_energy((1, 1, 1), (0, 1, 2), graph) == Fraction(8, 3)
    assert literal_tree_energy((2, 1), (0, 1, 2), graph, peo=(2, 1, 0)) == Fraction(
        2, 3
    )
    assert literal_tree_energy((2, 1), (0, 1, 2), graph, peo=(0, 1, 2)) == Fraction(
        2, 3
    )
    for edges, composition in (
        (((0, 1), (0, 2), (1, 2), (1, 3)), (2, 2)),
        (((0, 1), (0, 2), (1, 2), (1, 3), (2, 3)), (1, 1, 1, 1)),
    ):
        graph = DependenceGraph(edges)
        assert literal_tree_energy(composition, range(4), graph) == _literal_tree_sum(
            composition, graph
        )


def test_le6_repeated_join_and_coloured_hook_counterexample():
    graph = DependenceGraph(((0, 1), (0, 2), (0, 3)))
    for composition in _compositions(4):
        expected = tree_moment(composition, tuple(graph.edges)) ** 2
        assert (
            literal_tree_energy(composition, range(4), graph, peo=(1, 2, 3, 0))
            == expected
        )
    common = ((0, 7), (1, 7), (2, 1), (5, 3), (4, 6))
    first = DependenceGraph((*common, (3, 0), (6, 2)))
    second = DependenceGraph((*common, (6, 0), (3, 2)))
    assert literal_tree_energy((5, 2, 1), range(8), first) == Fraction(64, 11025)
    assert literal_tree_energy((5, 2, 1), range(8), second) == Fraction(1024, 99225)


def test_graph_circuit_validation_and_refusal():
    graph = DependenceGraph(((0, 1),))
    for call in (
        lambda: star_trace_gram_energy(
            (1, 1, 1), (0, 1, 0), graph, centre_channel=0, max_terms=1
        ),
        lambda: literal_tree_energy((1, 1), (0, 1), graph, max_terms=1),
        lambda: literal_tree_energy((1, 1), (0, 1), graph, max_entries=1),
    ):
        with pytest.raises(ResourceLimit):
            call()
    with pytest.raises(ValueError, match="commute"):
        star_trace_gram_energy(
            (1, 1, 1),
            (0, 1, 2),
            DependenceGraph(combinations(range(3), 2)),
            centre_channel=0,
        )
    with pytest.raises(ValueError, match="chordal"):
        literal_tree_energy(
            (1, 1, 1, 1), range(4), DependenceGraph(((0, 1), (1, 2), (2, 3), (0, 3)))
        )
    with pytest.raises(ValueError, match="perfect elimination"):
        literal_tree_energy((1, 1), (0, 1), graph, peo=(0, 0))
