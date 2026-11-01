"""Exact tensor-log identities in physical offset coordinates."""

import itertools

import torch

from tensorsig.offset_coded_signatures.offset_patterns import (
    log_signature_offset_coefficient,
    log_signature_offset_coefficient_fraction,
)
from tensorsig.classical.signatures import signature
from tensorsig.tensor_ops.series import (
    _signature_to_logsignature_tensor,
    unflatten_signature_levels,
)


def test_log_signature_offset_coefficients_match_exact_examples() -> None:
    examples = {
        (0, 0, 0, 0): 0,
        (0, 0, 0, 1): 0,
        (0, 1, 1, 1): 0,
        (0, 0, 1, 1): 1 / 24,
        (0, 0, 1, 2): 1 / 12,
        (0, 1, 1, 2): 1 / 12,
        (0, 1, 2, 2): 1 / 12,
        (0, 1, 2, 3): 1 / 4,
    }
    for offsets, expected in examples.items():
        assert float(log_signature_offset_coefficient_fraction(offsets)) == expected


def test_ordinary_tensor_log_quotient_needs_nonsorted_offset_words() -> None:
    z0 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    z1 = torch.tensor([3.0, 5.0], dtype=torch.float64)
    word = (0, 1)

    full_hidden_offset_sum = 0.0
    sorted_only_sum = 0.0
    for offsets in itertools.product(range(2), repeat=2):
        gamma = log_signature_offset_coefficient(offsets)
        values = [z0, z1]
        contribution = gamma * values[offsets[0]][word[0]] * values[offsets[1]][word[1]]
        full_hidden_offset_sum += contribution
        if offsets[0] <= offsets[1]:
            sorted_only_sum += contribution

    assert float(full_hidden_offset_sum) == -0.5
    assert float(sorted_only_sum) == 2.5
    assert float(full_hidden_offset_sum) != float(sorted_only_sum)


def test_full_hidden_offset_sum_matches_ordinary_tensor_log_coordinate() -> None:
    z = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64)
    path = torch.cat([torch.zeros(1, 2, dtype=z.dtype), z.cumsum(dim=0)], dim=0)
    sig = signature(path.unsqueeze(0), depth=2)
    levels = unflatten_signature_levels(sig, width=2, depth=2)
    log_levels = _signature_to_logsignature_tensor(levels, width=2, depth=2)

    expected = 0.0
    for offsets in itertools.product(range(2), repeat=2):
        gamma = log_signature_offset_coefficient(offsets)
        expected += gamma * z[offsets[0], 0] * z[offsets[1], 1]

    torch.testing.assert_close(log_levels[1][0, 0, 1], expected)
