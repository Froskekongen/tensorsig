"""Window extraction for local increment transforms.

Local plans consume increments with shape
``(batch, windows, size, channels)``. Window ``size`` and ``stride`` both count
increments, and extraction uses only valid windows without padding.
"""

from __future__ import annotations

from torch import Tensor

from .utils import local_increment_windows_from_path


def increment_windows(path: Tensor, size: int, stride: int = 1) -> Tensor:
    """Return valid windows of path increments.

    ``size`` counts increments, not path points. The result has shape
    ``(batch, windows, size, channels)`` and uses no padding.
    """
    for name, value in (("size", size), ("stride", stride)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    return local_increment_windows_from_path(
        path,
        kernel_size=size,
        hop_size=stride,
    )


def validate_increment_windows(
    windows: Tensor,
    *,
    size: int | None = None,
    channels: int | None = None,
) -> tuple[int, int, int, int]:
    """Validate ``(batch, windows, increments, channels)`` input."""
    if windows.ndim != 4:
        raise ValueError(
            "Expected windows with shape (batch, windows, increments, channels)."
        )
    if not windows.dtype.is_floating_point:
        raise ValueError("windows must have a floating-point dtype.")
    batch, count, actual_size, actual_channels = map(int, windows.shape)
    if actual_size < 1 or actual_channels < 1:
        raise ValueError("windows must have positive increment and channel axes.")
    if size is not None and actual_size != size:
        raise ValueError(f"Expected windows.shape[2] == {size}, got {actual_size}.")
    if channels is not None and actual_channels != channels:
        raise ValueError(
            f"Expected windows.shape[3] == {channels}, got {actual_channels}."
        )
    return batch, count, actual_size, actual_channels


__all__ = ["increment_windows"]
