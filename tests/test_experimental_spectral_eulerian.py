"""Numerical equivalence checks for the subset-cut projector."""

import pytest
import torch

from tensorsig.experimental.offset_coded_signatures.spectral_eulerian import (
    spectral_eulerian_project_columns,
)
from tensorsig.algebra.eulerian import eulerian_project_columns


@pytest.mark.parametrize(
    ("depth", "dtype"),
    [(depth, torch.float64) for depth in range(1, 7)]
    + [(depth, torch.float32) for depth in range(1, 5)],
)
def test_projection_matches_factorial_values_vjps_and_idempotence(depth, dtype):
    generator = torch.Generator().manual_seed(20261013 + depth)
    matrix = torch.randn(
        2**depth, 3, dtype=dtype, generator=generator, requires_grad=True
    )
    cotangent = torch.randn(matrix.shape, dtype=dtype, generator=generator)
    actual = spectral_eulerian_project_columns(matrix, 2, depth)
    expected = eulerian_project_columns(matrix, 2, depth)
    tolerance = 2e-5 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        spectral_eulerian_project_columns(actual, 2, depth),
        actual,
        atol=tolerance,
        rtol=tolerance,
    )
    actual_vjp = torch.autograd.grad(actual, matrix, cotangent)[0]
    expected_vjp = torch.autograd.grad(expected, matrix, cotangent)[0]
    torch.testing.assert_close(actual_vjp, expected_vjp, atol=tolerance, rtol=tolerance)


def test_projection_preserves_cut_multiplicity_and_accepts_noncontiguous_columns():
    matrix = torch.arange(48, dtype=torch.float64).reshape(3, 16).T[:, ::2]
    assert not matrix.is_contiguous()
    torch.testing.assert_close(
        spectral_eulerian_project_columns(matrix, 2, 4),
        eulerian_project_columns(matrix, 2, 4),
        atol=1e-13,
        rtol=1e-13,
    )
    assert spectral_eulerian_project_columns(matrix[:, :0], 2, 4).shape == (16, 0)


@pytest.mark.parametrize(
    ("matrix", "width", "depth", "message"),
    [
        (torch.ones(4, 1), 0, 2, "width"),
        (torch.ones(4, 1), True, 2, "width"),
        (torch.ones(4, 1), 2, 0, "depth"),
        (torch.ones(4, 1), 2, 1.5, "depth"),
        (torch.ones(4), 2, 2, "shape"),
        (torch.ones(3, 1), 2, 2, "shape"),
        (torch.ones(4, 1, dtype=torch.long), 2, 2, "floating-point"),
    ],
)
def test_projection_rejects_invalid_inputs(matrix, width, depth, message):
    with pytest.raises(ValueError, match=message):
        spectral_eulerian_project_columns(matrix, width, depth)
