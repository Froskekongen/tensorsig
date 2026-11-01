"""Separation properties of local feature families."""

import torch

from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.layers import LocalFeatures
from tensorsig.offset_coded_signatures.plans import (
    LocalRealization,
    compile_polynomial,
)
from tensorsig.offset_coded_signatures.windows import increment_windows
from tensorsig.offset_coded_signatures.diagnostics import (
    offset_contrast_energy,
    offset_tie_decomposition,
)


def _path_from_increments(dx: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros(dx.shape[0], 1, dx.shape[-1], dtype=dx.dtype, device=dx.device)
    return torch.cat([zero, dx.cumsum(dim=1)], dim=1)


def test_one_ray_summed_signature_collapses_to_total_increment() -> None:
    dtype = torch.float64
    ray = torch.tensor([2.0, -1.0], dtype=dtype)
    coeffs_a = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=dtype)
    coeffs_b = torch.tensor([0.0, 1.5, -3.5, 4.5], dtype=dtype)
    assert torch.sum(coeffs_a) == torch.sum(coeffs_b)

    x_a = _path_from_increments((coeffs_a[:, None] * ray[None, :]).unsqueeze(0))
    x_b = _path_from_increments((coeffs_b[:, None] * ray[None, :]).unsqueeze(0))
    plan = LocalRealization(
        OffsetCode.constant(4, dtype=dtype),
        channels=2,
        max_depth=4,
    ).signature(coordinates="dense")
    out_a = LocalFeatures(plan)(x_a)
    out_b = LocalFeatures(plan)(x_b)

    torch.testing.assert_close(out_a, out_b, atol=1e-9, rtol=1e-7)


def test_offset_forgetting_cannot_separate_single_spike_offsets_but_polynomial_can() -> (
    None
):
    dtype = torch.float64
    dx_left = torch.zeros(1, 4, 1, dtype=dtype)
    dx_right = torch.zeros(1, 4, 1, dtype=dtype)
    dx_left[0, 1, 0] = 2.0
    dx_right[0, 3, 0] = 2.0
    x_left = _path_from_increments(dx_left)
    x_right = _path_from_increments(dx_right)

    summed_plan = LocalRealization(
        OffsetCode.constant(4, dtype=dtype),
        channels=1,
        max_depth=3,
    ).signature(coordinates="dense")
    summed_left = LocalFeatures(summed_plan)(x_left)
    summed_right = LocalFeatures(summed_plan)(x_right)
    torch.testing.assert_close(summed_left, summed_right, atol=1e-9, rtol=1e-7)

    polynomial = compile_polynomial(
        window=4,
        channels=1,
        max_depth=1,
        scaling="support",
    )
    left_features = polynomial.evaluate(increment_windows(x_left, size=4))
    right_features = polynomial.evaluate(increment_windows(x_right, size=4))
    offset_one = int(
        torch.nonzero((polynomial.offset_indices[0] == 1).all(dim=1))[0].item()
    )
    torch.testing.assert_close(
        left_features[..., offset_one],
        torch.tensor([[2.0]], dtype=dtype),
    )
    torch.testing.assert_close(
        right_features[..., offset_one],
        torch.tensor([[0.0]], dtype=dtype),
    )


def test_degree_one_offset_tying_matches_constant_code() -> None:
    torch.manual_seed(0)
    dtype = torch.float64
    x = torch.randn(2, 8, 3, dtype=dtype)
    projection_kernel = torch.randn(4, 3, dtype=dtype)
    polynomial_weight = projection_kernel[:, None, :].expand(4, 5, 3).reshape(4, -1)
    windows = increment_windows(x, size=5)
    summed_plan = LocalRealization(
        OffsetCode.constant(5, dtype=dtype),
        channels=3,
        max_depth=1,
    ).signature(coordinates="dense")
    polynomial_plan = compile_polynomial(
        window=5,
        channels=3,
        max_depth=1,
        scaling="support",
    )
    summed = summed_plan.contract(windows, projection_kernel)
    direct = polynomial_plan.contract(windows, polynomial_weight)
    torch.testing.assert_close(direct, summed, atol=1e-9, rtol=1e-7)


def test_offset_tie_diagnostic_detects_first_order_contrasts() -> None:
    tied = torch.tensor(
        [[[1.0, -2.0], [1.0, -2.0], [1.0, -2.0]]],
        dtype=torch.float64,
    )
    contrast = tied.clone()
    contrast[0, 1, 0] += 3.0

    tied_parts = offset_tie_decomposition(tied)
    contrast_parts = offset_tie_decomposition(contrast)
    torch.testing.assert_close(tied_parts.contrast, torch.zeros_like(tied))
    assert contrast_parts.contrast.abs().sum() > 0
    torch.testing.assert_close(
        offset_contrast_energy(tied), torch.zeros(1, dtype=tied.dtype)
    )
    assert torch.all(offset_contrast_energy(contrast) > 0)
