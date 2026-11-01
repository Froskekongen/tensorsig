"""Dense tensor signatures of batched increment windows."""

from __future__ import annotations

from torch import Tensor

from tensorsig.tensor_ops.series import (
    _batch_mult_fused_restricted_exp,
    batch_restricted_exp,
)

from .windows import validate_increment_windows


def signature_levels_from_increment_windows(
    windows: Tensor,
    max_depth: int,
) -> tuple[Tensor, ...]:
    """Return dense signature levels for ``(batch, windows, steps, width)``."""
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer.")
    batch, count, steps, width = validate_increment_windows(windows)
    increments = windows.reshape(batch * count, steps, width)
    levels = batch_restricted_exp(increments[:, 0], depth=max_depth)
    for increment in increments[:, 1:].unbind(dim=1):
        levels = _batch_mult_fused_restricted_exp(increment, levels)
    return tuple(level.reshape(batch, count, *level.shape[1:]) for level in levels)


__all__ = ["signature_levels_from_increment_windows"]
