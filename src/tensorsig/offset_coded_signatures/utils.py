"""Validation and tensor-shape utilities for local signature layers."""

from __future__ import annotations

from torch import Tensor


def validate_local_signature_args(kernel_size: int, hop_size: int, depth: int) -> None:
    """Validate common local-signature integer arguments."""
    if kernel_size < 1:
        raise ValueError("kernel_size must be at least 1 increment.")
    if hop_size < 1:
        raise ValueError("hop_size must be positive.")
    if depth < 1:
        raise ValueError("depth must be at least 1.")


def validate_path_points(
    x: Tensor,
    in_channels: int | None = None,
) -> tuple[int, int, int]:
    """Validate path points with shape ``(batch, increments + 1, channels)``.

    Parameters
    ----------
    x : Tensor
        Path tensor with shape ``(batch, increments + 1, channels)``.
    in_channels : int, optional
        Expected channel count.

    Returns
    -------
    tuple[int, int, int]
        ``(batch, path_length, channels)``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape (B, L + 1, C), got {tuple(x.shape)}.")
    if not x.dtype.is_floating_point:
        raise ValueError("x must have a floating-point dtype.")
    batch_size, path_length, channels = x.shape
    if path_length < 2:
        raise ValueError("x must contain at least two path points.")
    if in_channels is not None and channels != in_channels:
        raise ValueError(f"Expected {in_channels} input channels, got {channels}.")
    return batch_size, path_length, channels


def unfold_increments(dx: Tensor, kernel_size: int, hop_size: int) -> Tensor:
    """Return local increment windows as a strided view.

    Parameters
    ----------
    dx : Tensor
        Increment tensor of shape ``(batch, increments, channels)``.
    kernel_size : int
        Number of increments in each local window.
    hop_size : int
        Step between consecutive window starts, measured in increments.

    Returns
    -------
    Tensor
        Strided view of shape ``(batch, num_windows, kernel_size, channels)``.
    """
    validate_local_signature_args(kernel_size, hop_size, depth=1)
    if dx.ndim != 3:
        raise ValueError(f"Expected dx of shape (B, L, C), got {tuple(dx.shape)}.")

    batch_size, increments, channels = dx.shape
    if kernel_size > increments:
        raise ValueError(
            f"kernel_size={kernel_size} exceeds number of increments {increments}."
        )

    num_windows = 1 + (increments - kernel_size) // hop_size
    batch_stride, increment_stride, channel_stride = dx.stride()
    return dx.as_strided(
        size=(batch_size, num_windows, kernel_size, channels),
        stride=(
            batch_stride,
            hop_size * increment_stride,
            increment_stride,
            channel_stride,
        ),
    )


def local_increment_windows_from_path(
    x: Tensor,
    kernel_size: int,
    hop_size: int,
    in_channels: int | None = None,
) -> Tensor:
    """Return local increment windows for path points ``x``.

    Parameters
    ----------
    x : Tensor
        Path tensor with shape ``(batch, increments + 1, channels)``.
    kernel_size : int
        Number of increments in each local window.
    hop_size : int
        Step between consecutive window starts, measured in increments.
    in_channels : int, optional
        Expected channel count.

    Returns
    -------
    Tensor
        Local increments with shape ``(batch, num_windows, kernel_size,
        channels)``.
    """
    validate_path_points(x, in_channels=in_channels)
    dx = x[:, 1:, :] - x[:, :-1, :]
    return unfold_increments(dx, kernel_size=kernel_size, hop_size=hop_size)
