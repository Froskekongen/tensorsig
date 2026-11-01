"""Memory gates distinguish query-boundary RSS from final process peaks."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/confirm_exact_graph_memory.py"
spec = importlib.util.spec_from_file_location("graph_memory_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def _row(method, peak, status="passed"):
    return {
        "case": {"id": "example"},
        "method": method,
        "status": status,
        "query_peak_rss_bytes": peak,
        "process_peak_rss_bytes": peak,
    }


def test_peak_gate_does_not_hide_one_high_process_behind_the_median():
    rows = [_row("baseline", peak) for peak in (100, 100, 100)]
    rows += [_row("candidate", peak) for peak in (100, 100, 112)]
    result = benchmark.summarize(rows)[0]
    assert result["median_query_peak_candidate_over_baseline"] == 1
    assert result["maximum_query_peak_candidate_over_baseline"] == 1.12
    assert not result["within_ten_percent_query_peak_rss"]


def test_refused_process_cannot_be_reported_as_a_passed_memory_gate():
    rows = [_row("baseline", 100) for _ in range(3)]
    rows += [
        _row("candidate", 100),
        _row("candidate", 100),
        _row("candidate", 1, "memory_refused"),
    ]
    assert benchmark.summarize(rows) == [{"case": "example", "status": "incomplete"}]


def test_declared_pairs_match_the_retained_exact_portfolio():
    reference = json.loads(
        (ROOT / "benchmarks/fixtures/exact_graph_reference.json").read_text()
    )
    cases = benchmark.cases(reference)
    assert len(cases) == 7
    for case in cases:
        degree = sum(case["composition"])
        assert len(case["edges"]) == degree - 1
    assert cases[-1]["composition"] == [6, 1]
    assert cases[-1]["channels"] == list(range(7))


def test_parent_watchdog_retains_explicit_timeout():
    case = {
        "id": "timeout",
        "kind": "scalar",
        "composition": [2, 1],
        "edges": [[0, 1], [1, 2]],
        "expected": [1, 1],
    }
    result = benchmark.run_child(
        ROOT, {"case": case, "method": "baseline"}, 0.001, 4096
    )
    assert result["status"] == "timeout"
    assert "query_peak_rss_bytes" not in result
    assert result["watchdog"]["wall_seconds"] < 10


def test_child_final_and_sampled_memory_caps_retain_refusal():
    case = {
        "id": "memory",
        "kind": "scalar",
        "composition": [2, 1],
        "edges": [[0, 1], [1, 2]],
        "expected": [1, 1],
    }
    result = benchmark.run_child(ROOT, {"case": case, "method": "baseline"}, 20, 1)
    assert result["status"] == "memory_refused"
    assert result["process_peak_rss_bytes"] > 2**20


def test_final_diagnostic_process_peak_is_retained_separately_from_query_scope():
    rows = [_row("baseline", 100) for _ in range(3)]
    rows += [_row("candidate", 100) for _ in range(3)]
    rows[-1]["process_peak_rss_bytes"] = 150
    result = benchmark.summarize(rows)[0]
    assert result["query_peak_samples_bytes"]["candidate"] == [100, 100, 100]
    assert result["final_process_peak_samples_bytes"]["candidate"] == [100, 100, 150]
    assert result["within_ten_percent_query_peak_rss"]
    assert "before any diagnostic call" in result["rss_gate_scope"]
