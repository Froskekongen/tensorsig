#!/usr/bin/env python3
"""Evaluate or replay bounded exact graph-block/tensor-program coordinates."""

import argparse
import json
import sys
from pathlib import Path

from tensorsig.offset_coded_signatures._graph.rational_coordinates import (
    ResourceLimit,
    _canonical_json,
    evaluate,
    load_json,
    replay,
    require,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, help="graph-block or tensor-abp JSON"
    )
    parser.add_argument(
        "--replay", type=Path, help="recompute and compare a saved result JSON"
    )
    parser.add_argument("--max-dimension", type=int, default=20000)
    parser.add_argument("--max-vertices", type=int, default=5000)
    parser.add_argument("--max-outputs", type=int, default=128)
    args = parser.parse_args()
    limits = {
        "dimension": args.max_dimension,
        "vertices": args.max_vertices,
        "outputs": args.max_outputs,
    }
    try:
        require(all(x > 0 for x in limits.values()), "limits must be positive")
        require(args.input is not None, "input JSON is required")
        spec = load_json(args.input)
        result = (
            replay(spec, load_json(args.replay), limits)
            if args.replay
            else evaluate(spec, limits)
        )
        print(_canonical_json(result))
    except (ValueError, KeyError, TypeError, OSError, ResourceLimit) as exc:
        print(
            json.dumps(
                {
                    "status": "resource-limit"
                    if isinstance(exc, ResourceLimit)
                    else "error",
                    "message": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 3 if isinstance(exc, ResourceLimit) else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
