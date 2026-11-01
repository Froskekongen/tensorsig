"""Bounded CPU benchmarks for Pascal rank certificates and PIT2 coordinates.

Each case runs in a fresh child process. Independent watchdogs enforce wall-time
and sampled RSS caps; numerical/resource failures remain explicit report
rows. The declared default scope is deliberately small: binary two-offset
degrees 2--7, PIT2 (1,1)/(1,2) at one channel, one common two-channel degree-two
source, and a prescribed dimension refusal. This is not a scaling claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]


def declared_cases():
    cases = []
    for degree in range(2, 8):
        cases.extend(
            (
                {
                    "id": f"profile_m{degree}",
                    "family": "pascal_profile",
                    "degree": degree,
                    "window": 2,
                },
                {
                    "id": f"certificate_m{degree}",
                    "family": "pascal_certificate",
                    "degree": degree,
                    "window": 2,
                    "weight": degree,
                },
            )
        )
    for name, composition, channels, edges, dimension in (
        ("pit2_q11_c1", (1, 1), 1, (), 20000),
        ("pit2_q12_c1", (1, 2), 1, (), 20000),
        ("pit2_q11_c2_common", (1, 1), 2, ((0, 1),), 20000),
        ("pit2_dimension_refusal", (1, 1), 1, (), 1),
    ):
        cases.append(
            {
                "id": name,
                "family": "pit2",
                "composition": composition,
                "channels": channels,
                "edges": edges,
                "limits": {"vertices": 5000, "dimension": dimension, "outputs": 128},
                "expected_refusal": dimension == 1,
            }
        )
    return cases


def source_hashes():
    files = (
        "src/tensorsig/offset_coded_signatures/pascal_rank.py",
        "src/tensorsig/offset_coded_signatures/pascal_certificate.py",
        "src/tensorsig/offset_coded_signatures/_graph/rational_coordinates.py",
        "src/tensorsig/offset_coded_signatures/_lie_quotients/compiler.py",
        "src/tensorsig/offset_coded_signatures/_lie_quotients/enveloping.py",
    )
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files
    }


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024 * value)


def operand_sizes(values, parse=Fraction):
    count = numerator = denominator = 0
    for value in values:
        value = parse(value)
        count += 1
        numerator = max(numerator, abs(value.numerator).bit_length())
        denominator = max(denominator, value.denominator.bit_length())
    return {
        "stored_rational_count": count,
        "max_numerator_bits": numerator,
        "max_denominator_bits": denominator,
    }


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, 1000 * (time.perf_counter() - start)


def summary(samples):
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def worker(case, repeats, memory_mib):
    stopped = threading.Event()

    def guard_memory():
        while not stopped.wait(0.1):
            peak = peak_rss_bytes()
            if peak > memory_mib * 2**20:
                print(
                    json.dumps(
                        {
                            "case": case,
                            "status": "memory_refused",
                            "reason": f"child RSS exceeds {memory_mib} MiB",
                            "memory": {"process_peak_rss_bytes": peak},
                        }
                    ),
                    flush=True,
                )
                os._exit(86)

    threading.Thread(target=guard_memory, daemon=True).start()
    # Imports are outside compilation timing but inside process memory/time caps.
    from tensorsig.offset_coded_signatures.pascal_rank import (
        binary_pascal_profile,
        binary_pascal_rank,
    )
    from tensorsig.offset_coded_signatures.pascal_certificate import (
        binary_pascal_rank_certificate,
    )
    from tensorsig.offset_coded_signatures._graph import (
        rational_coordinates as pit,
    )
    from tensorsig.offset_coded_signatures.quotients import DependenceGraph
    import torch

    baseline = peak_rss_bytes()
    row = {
        "case": case,
        "status": "running",
        "source_sha256": source_hashes(),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "memory": {"peak_rss_after_import_bytes": baseline},
    }

    def emit():
        row["memory"]["process_peak_rss_bytes"] = peak_rss_bytes()
        print(json.dumps(row, sort_keys=True), flush=True)

    emit()
    start = time.perf_counter()
    try:
        if case["family"] == "pascal_profile":

            def compile_case():
                return binary_pascal_profile(case["degree"], case["window"])
        elif case["family"] == "pascal_certificate":

            def compile_case():
                return binary_pascal_rank_certificate(
                    case["degree"],
                    case["window"],
                    case["weight"],
                    max_degree=10,
                    max_columns=4096,
                )
        else:
            spec = {
                "kind": "graph-block",
                "q": list(case["composition"]),
                "channels": case["channels"],
                "edges": [list(edge) for edge in case["edges"]],
            }
            graph = DependenceGraph(case["edges"])
            limits = case["limits"]

            def compile_case():
                return pit.compile_graph_block_coordinates(
                    case["composition"],
                    graph,
                    channels=case["channels"],
                    max_vertices=limits["vertices"],
                    max_dimension=limits["dimension"],
                    max_outputs=limits["outputs"],
                )

        reference, cold_ms = timed(compile_case)
        row["cold_compile"] = summary([cold_ms])
        if case.get("expected_refusal"):
            raise AssertionError(
                "The declared dimension-refusal case unexpectedly compiled."
            )
        if case["family"] == "pascal_profile":
            row["structure"] = {
                "profile": reference,
                "rank": reference[-1],
                "known_entries": sum(x is not None for x in reference),
            }
            row["operands"] = operand_sizes(x for x in reference if x is not None)
        elif case["family"] == "pascal_certificate":
            row["structure"] = {
                "rows": len(reference.rows),
                "columns": reference.column_count,
                "nonzeros": reference.nonzeros,
                "matrix_rank": reference.matrix_rank,
                "primitive_rank": reference.primitive_rank,
                "kernel_dimension": len(reference.kernel),
            }
            row["operands"] = operand_sizes(
                value
                for rows in (reference.rows, reference.kernel)
                for entries in rows
                for _, value in entries
            )
            expected = binary_pascal_rank(
                case["degree"], case["window"], case["weight"]
            )
            if expected is not None and reference.primitive_rank != expected:
                raise AssertionError(
                    "PBW certificate disagrees with proved Pascal rank."
                )
            row["exact_oracle_rank"] = expected
        else:
            row["structure"] = {
                "vertices": reference["vertices"],
                "dimension": reference["dimension"],
                "factor_dimensions": reference["factor_dimensions"],
                "source_columns": len(reference["columns"]),
                "image_rank": reference["image_rank"],
                "nonzeros": sum(
                    len(column["coordinates"]) for column in reference["columns"]
                ),
                "decoder_nonzeros": sum(
                    bool(pit.rational(value))
                    for values in reference["decoder"]["inverse"]
                    for value in values
                ),
            }
            values = [
                value
                for column in reference["columns"]
                for _, value in column["coordinates"]
            ]
            values.extend(column["squared_norm"] for column in reference["columns"])
            values.extend(
                value for values in reference["decoder"]["inverse"] for value in values
            )
            row["operands"] = operand_sizes(values, pit.rational)
            row["certificate_sha256"] = pit.digest(reference)
        emit()
        samples = []
        for _ in range(repeats):
            result, elapsed = timed(compile_case)
            if result != reference:
                if isinstance(result, dict) and isinstance(reference, dict):
                    changed = [
                        key
                        for key in result.keys() | reference.keys()
                        if result.get(key) != reference.get(key)
                    ]
                    row["changed_certificate_fields"] = changed
                    if changed == ["evaluator_sha256"]:
                        row.update(
                            status="source_changed",
                            reason="Evaluator source changed during the case; the certificate provenance changed.",
                            new_evaluator_sha256=result["evaluator_sha256"],
                        )
                        emit()
                        stopped.set()
                        return
                raise AssertionError("Repeated exact compilation changed its result.")
            del result
            samples.append(elapsed)
            row["warm_compile"] = summary(samples)
            emit()
        if case["family"] == "pascal_certificate":
            samples = []
            for _ in range(repeats):
                accepted, elapsed = timed(reference.verify)
                if not accepted:
                    raise AssertionError("Certificate minor/kernel replay failed.")
                samples.append(elapsed)
            row["witness_replay"] = summary(samples)
        elif case["family"] == "pit2":
            vector = [Fraction()] * reference["dimension"]
            expected = tuple(
                (-1) ** i * (i + 2) for i in range(reference["image_rank"])
            )
            for coefficient, column in zip(
                expected, reference["decoder"]["basis_columns"], strict=True
            ):
                for index, value in reference["columns"][column]["coordinates"]:
                    vector[index] += coefficient * pit.rational(value)
            samples = []
            for _ in range(repeats):
                decoded, elapsed = timed(
                    lambda: pit.decode_coordinates(reference, vector)
                )
                if decoded != expected:
                    raise AssertionError("PIT2 source-image decoding failed.")
                samples.append(elapsed)
            row["decode"] = summary(samples)
            row["decoded_basis_coefficients"] = expected
            emit()
            samples = []
            for _ in range(repeats):
                _, elapsed = timed(lambda: pit.replay(spec, reference, limits))
                samples.append(elapsed)
                row["full_replay"] = summary(samples)
                emit()
        row["status"] = "passed"
    except pit.ResourceLimit as exc:
        row.update(
            status="resource_refused",
            reason=str(exc),
            refusal_ms=1000 * (time.perf_counter() - start),
        )
    except MemoryError as exc:
        row.update(
            status="memory_refused", reason=str(exc) or "Python allocation failed"
        )
    except Exception as exc:
        row.update(status="failed", error_type=type(exc).__name__, reason=str(exc))
    row["worker_elapsed_seconds"] = time.perf_counter() - start
    emit()
    stopped.set()


def run_case(case, args):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        json.dumps(case),
        "--repeats",
        str(args.repeats),
        "--memory-mib",
        str(args.memory_mib),
    ]
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    started = time.monotonic()
    refusal = None
    while True:
        try:
            stdout, stderr = child.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if time.monotonic() - started >= args.time_limit:
                refusal = "timeout"
            if refusal:
                os.killpg(child.pid, signal.SIGKILL)
                stdout, stderr = child.communicate()
                break
    snapshots = []
    for line in stdout.splitlines():
        try:
            snapshots.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    row = snapshots[-1] if snapshots else {"case": case}
    row["watchdog"] = {
        "wall_seconds": time.monotonic() - started,
        "reported_peak_rss_bytes": row.get("memory", {}).get("process_peak_rss_bytes"),
        "exit_code": child.returncode,
    }
    if refusal:
        row.update(
            status=refusal,
            reason=f"parent watchdog: {args.time_limit}s wall / {args.memory_mib} MiB RSS cap",
        )
    elif (child.returncode and row.get("status") != "memory_refused") or row.get(
        "status"
    ) in (None, "running"):
        row.update(
            status="failed",
            reason="worker exited without a final result",
            stderr=stderr[-4000:],
        )
    elif stderr:
        row["stderr"] = stderr[-4000:]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=600)
    parser.add_argument("--memory-mib", type=float, default=4096)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only this declared case ID; repeat for several IDs",
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or not isfinite(args.time_limit)
        or not isfinite(args.memory_mib)
        or args.time_limit <= 0
        or args.memory_mib <= 0
    ):
        parser.error("repetitions and resource caps must be positive")
    if args.worker:
        worker(json.loads(args.worker), args.repeats, args.memory_mib)
        return
    if args.output is None:
        parser.error("--output is required")
    cases = declared_cases()
    if args.cases:
        unknown = set(args.cases) - {case["id"] for case in cases}
        if unknown:
            parser.error(f"unknown case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in args.cases]
    report = {
        "schema": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
            "arithmetic": "exact rational",
        },
        "source_sha256": source_hashes(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "worker"
        },
        "timing_scope": "First compile after imports in one fresh child per case; 3 warm repetitions by default. Imports, structural statistics and input-vector construction are excluded. PIT2 full replay recomputes the compiler; PBW witness replay rebuilds input-bound F.60 rows and checks its minor and nullspace. A reference result stays live for all repeated checks.",
        "memory_scope": "Child ru_maxrss including imports and reference result; a child guard samples ru_maxrss every 0.1s and exits on cap. Sampling may overshoot the cap between polls. Parent independently enforces wall time. Process setup/imports count toward the wall cap. No GPU memory or tensor-runtime claims.",
        "cold_sample_scope": "One process-cold sample per case, not a confidence interval; warm and replay stages use the requested repetitions.",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        row = run_case(case, args)
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{case['id']}: {row['status']} ({row['watchdog']['wall_seconds']:.2f}s)",
            flush=True,
        )
    failed = [
        row
        for row in report["results"]
        if row["status"] != "passed"
        and not (
            row["status"] == "resource_refused" and row["case"].get("expected_refusal")
        )
    ]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
