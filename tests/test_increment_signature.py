import pytest
import torch

from tensorsig.classical.signatures import signature
from tensorsig.tensor_ops.series import (
    flatten_signature_levels,
    signature_levels_from_increments,
    unflatten_signature_levels,
)


def test_increment_window_recurrence_matches_explicit_noncommutative_paths() -> None:
    windows = torch.tensor(
        [
            [
                [[0.4, -0.2], [0.1, 0.7], [-0.3, 0.5]],
                [[-0.6, 0.2], [0.8, -0.1], [0.2, 0.9]],
            ]
        ],
        dtype=torch.float64,
    )
    actual = signature_levels_from_increments(windows, max_depth=3)

    zero = windows.new_zeros(*windows.shape[:2], 1, windows.shape[-1])
    paths = torch.cat((zero, windows.cumsum(dim=2)), dim=2)
    flattened = signature(paths.flatten(0, 1), depth=3).reshape(
        *windows.shape[:2],
        -1,
    )
    expected = unflatten_signature_levels(flattened, width=2, depth=3)

    for actual_level, expected_level in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_level, expected_level, atol=1e-12, rtol=1e-10)


def test_increment_window_recurrence_preserves_empty_batch_and_window_axes() -> None:
    for windows in (
        torch.empty(0, 2, 3, 2),
        torch.empty(2, 0, 3, 2),
    ):
        levels = signature_levels_from_increments(windows, max_depth=2)
        assert levels[0].shape == (*windows.shape[:2], 2)
        assert levels[1].shape == (*windows.shape[:2], 2, 2)


@pytest.mark.parametrize("leading_shape", [(), (2,), (2, 1, 3)])
def test_increment_levels_preserve_leading_axes_and_gradients(leading_shape) -> None:
    increments = (
        torch.randn(*leading_shape, 2, 3, dtype=torch.float64)
        .transpose(-2, -1)
        .requires_grad_()
    )
    actual = flatten_signature_levels(
        signature_levels_from_increments(increments, max_depth=3)
    )
    batched = increments.reshape(-1, 3, 2)
    path = torch.cat((batched.new_zeros(len(batched), 1, 2), batched.cumsum(1)), 1)
    expected = signature(path, depth=3).reshape(*leading_shape, -1)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), increments)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), increments)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("shape", [(2,), (3, 0, 2), (3, 2, 0)])
def test_increment_levels_reject_missing_step_or_width_axes(shape) -> None:
    with pytest.raises(ValueError):
        signature_levels_from_increments(torch.empty(shape), max_depth=2)
