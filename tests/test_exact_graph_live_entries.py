"""Dynamic state budgets include retained inputs and simultaneous outputs."""

from fractions import Fraction

import pytest

from tensorsig.offset_coded_signatures._graph import (
    literal_energy as le,
)
from tensorsig.offset_coded_signatures._graph import (
    star_gram as st,
)
from tensorsig.offset_coded_signatures._graph._graph_common import (
    ExactWork,
    ResourceLimit,
)
from tensorsig.offset_coded_signatures.quotients import DependenceGraph


def test_public_global_budget_boundaries_and_exact_values(monkeypatch):
    edge = DependenceGraph(((0, 1),))
    for module, function, options, old_cap, sufficient, expected in (
        (le, le.literal_tree_energy, {}, 16, 20, Fraction(1)),
        (st, st.star_trace_gram_energy, {"centre_channel": 0}, 7, 10, Fraction(1, 2)),
    ):
        works = []

        def work_factory(*args, **kwargs):
            work = ExactWork(*args, **kwargs)
            works.append(work)
            return work

        monkeypatch.setattr(module, "ExactWork", work_factory)
        with pytest.raises(ResourceLimit, match="sparse entries"):
            function((1, 1), (0, 1), edge, max_entries=old_cap, **options)
        assert works[-1].peak_entries <= old_cap
        assert (
            function((1, 1), (0, 1), edge, max_entries=sufficient, **options)
            == expected
        )
        assert works[-1].peak_entries == sufficient
        assert works[-1].live_entries == 0
        with pytest.raises(ResourceLimit, match="generated terms"):
            function(
                (1, 1), (0, 1), edge, max_entries=sufficient, max_terms=1, **options
            )


def test_extension_counts_the_input_and_output_and_preserves_aliases():
    key = (0, (0, 0, 0, 0), 0, 0)
    work = ExactWork(max_entries=1)
    original = {}
    le._add(original, key, Fraction(1), work)
    alias = original
    with pytest.raises(ResourceLimit):
        le._extend(original, (0,), (0, 1), work)
    assert alias == {key: 1}
    assert work.live_entries == 1
    work.max_entries = 2
    extended = le._extend(original, (0,), (0, 1), work)
    assert work.live_entries == 2
    work.discard(original)
    assert alias == {} and len(extended) == 1
    work.discard(extended)
    assert work.live_entries == 0


def test_shared_join_input_is_charged_once_and_cancellation_releases():
    work = ExactWork(max_entries=2)
    key = (0, (0, 0, 0, 0), 0, 0)
    original = {}
    le._add(original, key, Fraction(1), work)
    joined = le._join(original, original, 1, work)
    assert work.live_entries == 2  # One owned input, one output; aliases borrow.
    le._add(joined, key, Fraction(-1), work)
    assert not joined and work.live_entries == 1
    work.discard(original)
    assert work.live_entries == 0


def test_leaf_construction_includes_already_retained_entries():
    work = ExactWork(max_entries=5)
    work.acquire(4)  # Already retained leaf maps/skeleton weights.
    skeleton = st._Skeleton(((0, 0, 0),), ((0,),), 1, Fraction(1))
    with pytest.raises(ResourceLimit):
        st._leaf_map(skeleton, (1,), 1, work)
    assert work.peak_entries == 5  # Retained four + input one; output must refuse.


def test_joined_messages_and_repeated_centres_release_all_owned_entries(monkeypatch):
    graph = DependenceGraph(((0, 1), (0, 2)))
    for module, function, q, channels, options, expected in (
        (le, le.literal_tree_energy, (1, 1, 1), (1, 0, 2), {}, Fraction(16, 9)),
        (
            st,
            st.star_trace_gram_energy,
            (2, 1),
            (0, 0, 1),
            {"centre_channel": 0},
            Fraction(1, 24),
        ),
    ):
        works = []

        def work_factory(*args, **kwargs):
            work = ExactWork(*args, **kwargs)
            works.append(work)
            return work

        monkeypatch.setattr(module, "ExactWork", work_factory)
        assert function(q, channels, graph, **options) == expected
        assert works[-1].live_entries == 0
