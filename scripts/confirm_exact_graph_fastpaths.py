"""Repeat the existing graph portfolio in three fresh CPU processes.

Keep all ten calibrated raw samples per method/process. Confidence intervals
use the three process-level log speed ratios, never thirty pseudoreplicates.
Run without competing benchmark workloads.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from math import exp, log, sqrt
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts/benchmark_intermediate_rank_candidates.py"
SOURCE = ROOT / "src/tensorsig/offset_coded_signatures"


def source_hashes():
    """Bind the exact graph implementations and the measurement scripts."""
    paths = [Path(__file__).resolve(), BENCHMARK, SOURCE / "graph_queries.py"]
    paths += [SOURCE / "_lie_quotients/rules.py"]
    paths += [
        SOURCE / "_graph" / f"{name}.py"
        for name in (
            "_graph_common",
            "adaptive",
            "tree_moments",
            "young_dp",
            "direct_nbc",
            "chordal_nbc",
            "one_centre_gram",
            "star_gram",
            "literal_energy",
        )
    ]
    paths += [
        ROOT / "src/tensorsig/experimental/offset_coded_signatures" / f"{name}.py"
        for name in ("intermediate_rank", "circuit_coordinates", "support_compiler")
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def summarize(processes):
    """Use a two-sided Student-t interval on three process log ratios (df=2)."""
    assert len(processes) == 3
    summaries = []
    for rows in zip(*(process["report"]["rows"] for process in processes), strict=True):
        identities = [
            {
                key: value
                for key, value in row.items()
                if key not in {"baseline", "candidate", "refusal_ms"}
            }
            for row in rows
        ]
        assert identities[0] == identities[1] == identities[2]
        if "candidate" not in rows[0]:
            continue
        ratios = [
            row["baseline"]["median_ms"] / row["candidate"]["median_ms"] for row in rows
        ]
        logs = list(map(log, ratios))
        center = fmean(logs)
        margin = 4.302652729911275 * stdev(logs) / sqrt(3)
        summaries.append(
            {
                **identities[0],
                "process_baseline_over_candidate": ratios,
                "geometric_mean_speed_ratio": exp(center),
                "process_log_t95_interval": [
                    exp(center - margin),
                    exp(center + margin),
                ],
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-B",
        str(BENCHMARK),
        "--new-graph-only",
        "--repeats",
        "10",
    ]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "torch": version("torch"),
        "command": command,
        "source_sha256": source_hashes(),
        "statistical_scope": (
            "Three fresh processes; ten calibrated repeats per method per process. "
            "Baseline and candidate batches run sequentially, not paired per repeat. "
            "A 95% Student-t interval uses process-median log ratios with df=2, "
            "assuming approximately normal process log ratios. Only three independent "
            "processes limit uncertainty estimation; no cross-machine guarantee. "
            "Memory tracing is separate. Refusals are not zero results."
        ),
        "processes": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        started = perf_counter()
        child = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=180
        )
        result = json.loads(child.stdout)
        assert result["repeats"] == 10
        for row in result["rows"]:
            for method in ("baseline", "candidate"):
                if method in row:
                    assert len(row[method]["samples_ms"]) == 10
        assert source_hashes() == report["source_sha256"], (
            "Source changed during confirmation"
        )
        report["processes"].append(
            {
                "index": index + 1,
                "wall_seconds": perf_counter() - started,
                "report": result,
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Completed process {index + 1}/3", file=sys.stderr, flush=True)
    report["comparisons"] = summarize(report["processes"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
