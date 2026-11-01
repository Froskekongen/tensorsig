"""PIT2's retained controls plus the maintained library boundary."""

import json
import subprocess
import sys
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from tensorsig.offset_coded_signatures._graph import (
    rational_coordinates as pit,
)
from tensorsig.offset_coded_signatures.quotients import DependenceGraph


def test_library_common_source_rank_decoder_and_zero_image():
    result = pit.compile_graph_block_coordinates(
        (1, 1), DependenceGraph([(0, 1)]), channels=2
    )
    assert result["image_rank"] == 4
    vector = [Fraction(0)] * result["dimension"]
    coefficients = (2, -3, 5, -7)
    for coefficient, column_index in zip(
        coefficients, result["decoder"]["basis_columns"]
    ):
        for row, value in result["columns"][column_index]["coordinates"]:
            vector[row] += coefficient * pit.rational(value)
    assert pit.decode_coordinates(result, vector) == coefficients
    vector[0] += 1
    with pytest.raises(ValueError, match="outside"):
        pit.decode_coordinates(result, vector)
    zero = pit.compile_graph_block_coordinates((2,), DependenceGraph(), channels=1)
    assert zero["image_rank"] == 0
    assert pit.decode_coordinates(zero, [0] * zero["dimension"]) == ()


def test_resource_refusal_is_not_a_zero_certificate():
    with pytest.raises(pit.ResourceLimit, match="coordinate dimension"):
        pit.compile_graph_block_coordinates(
            (1, 1), DependenceGraph(), channels=1, max_dimension=1
        )
    with pytest.raises(ValueError, match="positive integers"):
        pit.compile_graph_block_coordinates(
            (1, 1), DependenceGraph(), channels=1, max_dimension=True
        )
    with pytest.raises(ValueError, match="row sum"):
        pit.compile_graph_block_coordinates(
            (2,), DependenceGraph(), channels=1, occupancy=[[1]]
        )


def test_cell_scope_and_replay_binding():
    spec = {
        "kind": "graph-block",
        "q": [1, 1],
        "channels": 1,
        "edges": [],
        "occupancy": [[1], [1]],
    }
    result = pit.evaluate(spec)
    assert pit.replay(spec, result) == result
    with pytest.raises(ValueError, match="mismatch"):
        pit.replay(spec, dict(result, image_rank=99))


def _constant_program(value=2):
    return {
        "kind": "tensor-abp",
        "vertices": 1,
        "alphabet_sizes": [1],
        "degree_bound": 0,
        "edges": [],
        "start": [value],
        "outputs": [{"name": "constant", "weights": [1]}],
    }


def test_import_preserves_process_integer_conversion_limit():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; before=sys.get_int_max_str_digits(); "
            "import tensorsig.offset_coded_signatures._graph.rational_coordinates; "
            "assert sys.get_int_max_str_digits()==before; print(before)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(result.stdout.strip()) > 0


def test_large_exact_coefficient_local_serialization_and_cli_replay(tmp_path):
    before = sys.get_int_max_str_digits()
    large = 10**6000 + 7
    spec = _constant_program(large)
    result = pit.evaluate(spec)
    assert pit.rational(result["columns"][0]["coordinates"][0][1]) == large
    assert pit.rational(result["columns"][0]["squared_norm"]) == large * large
    assert pit.decode_coordinates(result, [large]) == (1,)
    assert pit.replay(spec, result) == result
    assert sys.get_int_max_str_digits() == before
    source, saved = tmp_path / "input.json", tmp_path / "result.json"
    source.write_text(pit._canonical_json(spec))
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts/evaluate_graph_block_coordinates.py"),
        str(source),
    ]
    execution = subprocess.run(command, check=True, capture_output=True, text=True)
    saved.write_text(execution.stdout)
    assert pit.load_json(source) == spec
    assert pit.load_json(saved) == result
    replay = subprocess.run(
        [*command, "--replay", str(saved)], check=True, capture_output=True, text=True
    )
    assert json.loads(replay.stdout) == result


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["decoder"]["inverse"].append([]),
        lambda r: r["decoder"].update(inverse=[["0"]]),
        lambda r: r["decoder"].update(pivot_rows=[-1]),
        lambda r: r["decoder"].update(basis_columns=[True]),
        lambda r: r["columns"][0]["coordinates"].append([0, "2"]),
        lambda r: r["columns"][0].update(coordinates=[[-1, "2"]]),
        lambda r: r.update(image_rank=2),
        lambda r: r.update(columns=[]),
    ],
)
def test_decoder_rejects_malformed_sparse_columns_and_matrices(change):
    result = deepcopy(pit.evaluate(_constant_program()))
    change(result)
    with pytest.raises(ValueError):
        pit.decode_coordinates(result, [0])


def test_decoder_limits_and_local_decimal_ceiling(monkeypatch):
    result = pit.evaluate(_constant_program())
    with pytest.raises(pit.ResourceLimit, match="decoder coordinate dimension"):
        pit.decode_coordinates(dict(result, dimension=10**12), [])
    monkeypatch.setattr(pit, "DECIMAL_DIGIT_LIMIT", 10)
    for value in ("1e20", "1/100000000000000000000"):
        with pytest.raises(pit.ResourceLimit):
            pit.rational(value)
    with pytest.raises(pit.ResourceLimit):
        pit._rational_text(Fraction(1, 10**20))
    for value in ("NaN", "Infinity", "1__2", "1.0/2"):
        with pytest.raises(ValueError):
            pit.rational(value)
    assert pit.rational("-1_234.5e-2") == Fraction(-2469, 200)


def test_canonical_json_matches_stdlib_for_normal_inputs():
    data = {"null": None, "nested": [True, False, 3, -5, 1.25, {"text": 'å\\"'}]}
    assert pit._canonical_json(data) == json.dumps(
        data, sort_keys=True, separators=(",", ":")
    )
