"""The benchmark must retain resource refusals and enforce its watchdog."""

import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_exact_compilers.py"


def test_prescribed_dimension_refusal_is_not_a_zero_or_success(tmp_path):
    output = tmp_path / "refusal.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            "pit2_dimension_refusal",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    row = json.loads(output.read_text())["results"][0]
    assert row["status"] == "resource_refused"
    assert "coordinate dimension" in row["reason"]
    assert "structure" not in row
    assert row["memory"]["process_peak_rss_bytes"] > 0


def test_watchdog_retains_timeout_without_timings(tmp_path):
    output = tmp_path / "timeout.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            "profile_m2",
            "--time-limit",
            "0.001",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 1
    row = json.loads(output.read_text())["results"][0]
    assert row["status"] == "timeout"
    assert "cold_compile" not in row
    assert row["watchdog"]["wall_seconds"] < 10


def test_watchdog_retains_memory_refusal(tmp_path):
    output = tmp_path / "memory.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            "profile_m2",
            "--memory-mib",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 1
    row = json.loads(output.read_text())["results"][0]
    assert row["status"] == "memory_refused"
    assert row["memory"]["process_peak_rss_bytes"] > 2**20
    assert "structure" not in row
