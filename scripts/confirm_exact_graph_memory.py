"""Fresh-process RSS evidence for the seven promoted exact graph query pairs.

Imports and three calls of only the declared method count toward query peak RSS.
This complements, and does not replace, the separate Python-allocation report.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from functools import partial
from math import isfinite
from pathlib import Path
from statistics import median
from time import monotonic


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cases(reference):
    """Match the existing confirmation inputs and independent expected values."""
    rows = {row["case"]: row for row in reference["rows"]}
    result = []
    for degree in (3, 5, 7):
        for shape in ("path", "star"):
            name = f"terminal_hook_{degree}_{shape}"
            edges = (
                [(i, i + 1) for i in range(degree - 1)]
                if shape == "path"
                else [(0, i) for i in range(1, degree)]
            )
            result.append(
                dict(
                    id=name,
                    kind="scalar",
                    composition=[degree - 1, 1],
                    edges=edges,
                    expected=rows[name]["value"],
                )
            )
    result.append(
        dict(
            id="terminal_cell",
            kind="cell",
            composition=[6, 1],
            channels=list(range(7)),
            edges=[(i, i + 1) for i in range(6)],
            expected=rows["terminal_cell"]["survives"],
        )
    )
    return result


def source_hashes(root):
    confirmation = load_module(
        root / "scripts/confirm_exact_graph_fastpaths.py", "_graph_confirmation"
    )
    hashes = confirmation.source_hashes()
    helper = root / "benchmarks/benchmark_exact_compilers.py"
    hashes[str(helper.relative_to(root))] = hashlib.sha256(
        helper.read_bytes()
    ).hexdigest()
    return hashes


def worker(root, item, memory_mib):
    exact = load_module(
        root / "benchmarks/benchmark_exact_compilers.py", "_exact_compiler_benchmark"
    )
    stopped = threading.Event()
    row = dict(case=item["case"], method=item["method"], status="running")

    def refuse_if_over_limit():
        peak = exact.peak_rss_bytes()
        if peak > memory_mib * 2**20:
            row.update(
                status="memory_refused",
                reason=f"child peak RSS exceeds {memory_mib} MiB",
                process_peak_rss_bytes=peak,
            )
            print(json.dumps(row), flush=True)
            os._exit(86)

    def guard():
        while not stopped.wait(0.1):
            refuse_if_over_limit()

    threading.Thread(target=guard, daemon=True).start()
    try:
        sys.path.insert(0, str(root / "src"))
        bench = load_module(
            root / "scripts/benchmark_intermediate_rank_candidates.py",
            "_graph_benchmark",
        )
        import torch

        row.update(
            source_sha256=source_hashes(root),
            environment=dict(python=sys.version, torch=torch.__version__, device="cpu"),
            peak_rss_after_import_bytes=exact.peak_rss_bytes(),
        )
        case, method = item["case"], item["method"]
        composition = tuple(case["composition"])
        edges = tuple(map(tuple, case["edges"]))
        if case["kind"] == "scalar":
            baseline = (
                bench.young_dp_tree_moment
                if sum(composition) >= 7
                else bench.tree_moment
            )
            call = partial(
                baseline if method == "baseline" else bench.adaptive_tree_moment,
                composition,
                edges,
            )
            expected = Fraction(*case["expected"])
        else:
            graph = bench.DependenceGraph(edges)
            fn = (
                partial(bench.graph_queries.cell_survives, backend="direct_nbc")
                if method == "baseline"
                else bench.adaptive_cell_survives
            )
            call = partial(fn, composition, tuple(case["channels"]), graph)
            expected = case["expected"]
        for _ in range(3):
            value = call()
            if value != expected:
                raise AssertionError(
                    "Declared method disagrees with retained independent exact value"
                )
        row["query_peak_rss_bytes"] = exact.peak_rss_bytes()
        row["queries"] = 3
        row["output"] = (
            [value.numerator, value.denominator]
            if isinstance(value, Fraction)
            else value
        )
        row["output_rational_bits"] = (
            dict(
                numerator=abs(value.numerator).bit_length(),
                denominator=value.denominator.bit_length(),
            )
            if isinstance(value, Fraction)
            else None
        )
        row["work_metrics"] = None
        # Diagnostic collection happens after the RSS query boundary. Existing
        # Young-DP counters measure partial-message sizes, not global live state.
        if method == "baseline" and case["kind"] == "scalar" and sum(composition) == 7:
            metrics = bench.YoungDpMetrics()
            if (
                bench.young_dp_tree_moment(composition, edges, metrics=metrics)
                != expected
            ):
                raise AssertionError("Instrumented Young-DP result changed")
            row["work_metrics"] = dict(
                scope="separate diagnostic call; peak_state_cells is largest one message; bit sizes cover observed partial messages",
                **asdict(metrics),
            )
        if source_hashes(root) != row["source_sha256"]:
            row.update(
                status="source_changed",
                reason="Source changed during memory measurement",
            )
        else:
            row["status"] = "passed"
    except MemoryError as exc:
        row.update(status="memory_refused", reason=str(exc) or "allocation failed")
    except Exception as exc:
        row.update(
            status="resource_refused"
            if type(exc).__name__ == "ResourceLimit"
            else "failed",
            error_type=type(exc).__name__,
            reason=str(exc),
        )
    row["process_peak_rss_bytes"] = exact.peak_rss_bytes()
    refuse_if_over_limit()
    stopped.set()
    print(json.dumps(row), flush=True)


def run_child(root, item, time_limit, memory_mib):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--repo-root",
        str(root),
        "--worker",
        json.dumps(item),
        "--memory-mib",
        str(memory_mib),
    ]
    started = monotonic()
    child = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=time_limit)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGKILL)
        stdout, stderr = child.communicate()
    snapshots = []
    for line in stdout.splitlines():
        try:
            snapshots.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    row = snapshots[-1] if snapshots else dict(case=item["case"], method=item["method"])
    row["watchdog"] = dict(
        wall_seconds=monotonic() - started, exit_code=child.returncode
    )
    if timed_out:
        row.update(status="timeout", reason=f"parent wall cap {time_limit} seconds")
    elif (
        child.returncode
        and row.get("status") != "memory_refused"
        or row.get("status") in (None, "running")
    ):
        row.update(
            status="failed",
            reason="worker exited without successful final result",
            stderr=stderr[-4000:],
        )
    return row


def summarize(rows):
    summaries = []
    for name in dict.fromkeys(row["case"]["id"] for row in rows):
        pair = [row for row in rows if row["case"]["id"] == name]
        methods = {
            method: [
                row["query_peak_rss_bytes"]
                for row in pair
                if row["method"] == method and row["status"] == "passed"
            ]
            for method in ("baseline", "candidate")
        }
        if any(len(values) != 3 for values in methods.values()):
            summaries.append(dict(case=name, status="incomplete"))
            continue
        ratio = max(methods["candidate"]) / max(methods["baseline"])
        summaries.append(
            dict(
                case=name,
                status="passed",
                query_peak_samples_bytes=methods,
                final_process_peak_samples_bytes={
                    method: [
                        row["process_peak_rss_bytes"]
                        for row in pair
                        if row["method"] == method and row["status"] == "passed"
                    ]
                    for method in ("baseline", "candidate")
                },
                rss_gate_scope="process ru_maxrss through imports and three uninstrumented queries, before any diagnostic call",
                maximum_query_peak_candidate_over_baseline=ratio,
                median_query_peak_candidate_over_baseline=median(methods["candidate"])
                / median(methods["baseline"]),
                within_ten_percent_query_peak_rss=ratio <= 1.10,
                saved_tensor_storage="not applicable: pure exact CPU arithmetic",
            )
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("benchmarks/fixtures/exact_graph_reference.json"),
    )
    parser.add_argument("--time-limit", type=float, default=600)
    parser.add_argument("--memory-mib", type=float, default=4096)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(
        not isfinite(value) or value <= 0
        for value in (args.time_limit, args.memory_mib)
    ):
        parser.error("resource caps must be finite and positive")
    root = args.repo_root.resolve()
    if args.worker:
        worker(root, json.loads(args.worker), args.memory_mib)
        return
    if args.output is None:
        parser.error("--output is required")
    reference = (
        args.reference if args.reference.is_absolute() else root / args.reference
    )
    declared = cases(json.loads(reference.read_text()))
    report = dict(
        schema=1,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_hashes(root),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference=dict(
            path=str(reference),
            sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        ),
        limits=dict(wall_seconds=args.time_limit, memory_mib=args.memory_mib),
        scope="Three fresh processes per method/case; imports and three declared-method queries count toward query peak RSS. ru_maxrss is process high-water RSS, not incremental allocation. query_peak_rss_bytes stops at the pre-diagnostic query boundary; process_peak_rss_bytes is the actual final process high-water RSS after any diagnostic call and source check. The final cap check uses that full-process peak. Parent wall cap includes startup; 0.1-second child RSS sampling can overshoot between polls, with a final cap check. No confidence interval or GPU-memory claim; tiny query allocations may be obscured by import RSS. Saved tensors are not applicable.",
        results=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repetition in range(3):
        for case in declared:
            # Alternate method order across repeats to reduce fixed-order drift.
            for method in (
                ("baseline", "candidate")
                if repetition % 2 == 0
                else ("candidate", "baseline")
            ):
                row = run_child(
                    root,
                    dict(case=case, method=method),
                    args.time_limit,
                    args.memory_mib,
                )
                row["repetition"] = repetition + 1
                if (
                    row.get("status") == "passed"
                    and row["source_sha256"] != report["source_sha256"]
                ):
                    row.update(
                        status="source_changed",
                        reason="Sources differ from parent snapshot",
                    )
                report["results"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(case["id"], method, repetition + 1, row["status"], flush=True)
    report["comparisons"] = summarize(report["results"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(row["status"] != "passed" for row in report["results"]) or any(
        not row.get("within_ten_percent_query_peak_rss")
        for row in report["comparisons"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
