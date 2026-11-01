"""One query boundary with independent compiled families."""

import subprocess
import sys
from itertools import combinations

import pytest

from tensorsig.offset_coded_signatures import graph_queries as queries
from tensorsig.offset_coded_signatures.quotients import DependenceGraph


def test_maintained_import_does_not_load_research():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tensorsig.offset_coded_signatures import graph_queries; "
            "assert not any(name == 'tensorsig.experimental' or name.startswith('tensorsig.experimental.') "
            "for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_explicit_backends_validate_options_and_reset_metrics():
    graph = DependenceGraph(combinations(range(3), 2))
    order = ((1, 2), (0, 2), (0, 1))
    arrows = ((2, 1), (0, 2), (1, 0))
    plan = queries.compile_direct_nbc(
        (0, 1, 2), graph, edge_order=order, reference_arrows=arrows
    )
    metrics = queries.CellQueryMetrics()
    tree = queries.cell_witness(
        (1, 1, 1),
        (0, 1, 2),
        graph,
        backend="direct_nbc",
        metrics=metrics,
        edge_order=order,
        reference_arrows=arrows,
    )
    assert tree in tuple(plan.trees())
    assert queries.tree_moment((1, 1, 1), plan.reference_covers(tree))
    assert metrics.backend == "direct_nbc"
    assert metrics.trees_examined > 0
    assert metrics.candidate_trees is None
    assert not queries.cell_survives((3,), (0, 1, 2), graph, metrics=metrics)
    assert metrics == queries.CellQueryMetrics(backend="auto", scalar_backend="zero")
    for backend in ("auto", "chordal_nbc"):
        with pytest.raises(ValueError, match="require backend"):
            queries.cell_survives(
                (3,), (0, 1, 2), graph, backend=backend, edge_order=order
            )
    with pytest.raises(ValueError, match="edge_order"):
        queries.cell_survives(
            (3,), (0, 1, 2), graph, backend="direct_nbc", edge_order=()
        )
    for backend in ("unknown", [], None):
        with pytest.raises(ValueError, match="backend"):
            queries.cell_survives((1, 1, 1), (0, 1, 2), graph, backend=backend)


@pytest.mark.parametrize("backend", ["auto", "direct_nbc", "chordal_nbc"])
def test_all_query_backends_preserve_resource_refusal(backend):
    graph = DependenceGraph(combinations(range(4), 2))
    metrics = queries.CellQueryMetrics()
    with pytest.raises(queries.ResourceLimit, match="evaluated trees"):
        queries.cell_survives(
            (1, 1, 1, 1),
            (0, 1, 2, 3),
            graph,
            backend=backend,
            max_trees=1,
            metrics=metrics,
        )
    assert metrics.trees_examined == 1
    assert queries.cell_survives((1, 1, 1, 1), (0, 1, 2, 3), graph, backend=backend)
