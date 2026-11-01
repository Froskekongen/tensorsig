"""Exact rolling updates for fixed, shift-closed offset codes."""

from __future__ import annotations

import torch
from torch import Tensor

from tensorsig.tensor_ops.linear import apply_linear_map_to_signature_levels
from tensorsig.tensor_ops.series import (
    _batch_mult_fused_restricted_exp,
    batch_restricted_exp,
    signature_inverse,
    signature_multiply,
)

from .alphabet import _exact_shift_operator, _floating_matrix
from .increment_signature import signature_levels_from_increment_windows
from .utils import (
    unfold_increments,
    validate_local_signature_args,
    validate_path_points,
)


def _lift(increments: Tensor, code_rows: Tensor) -> Tensor:
    """Lift ``(batch, steps, channels)`` increments by matching code rows."""
    lifted = torch.einsum("brc,re->brce", increments, code_rows)
    return lifted.reshape(*increments.shape[:2], -1)


def _segment_signature_levels(
    increments: Tensor,
    start: int,
    code_rows: Tensor,
    depth: int,
) -> list[Tensor]:
    """Return the signature of consecutive increments with prescribed rows."""
    lifted = _lift(
        increments[:, start : start + code_rows.shape[0]],
        code_rows,
    )
    current = batch_restricted_exp(lifted[:, 0], depth=depth)
    for offset in range(1, lifted.shape[1]):
        current = _batch_mult_fused_restricted_exp(lifted[:, offset], current)
    return current


def _direct_coded_signature_levels(
    increments: Tensor,
    matrix: Tensor,
    depth: int,
    stride: int,
) -> tuple[Tensor, ...]:
    """Evaluate all coded windows independently and in parallel."""
    windows = unfold_increments(increments, matrix.shape[0], stride)
    lifted = torch.einsum("bnrc,re->bnrce", windows, matrix)
    lifted = lifted.reshape(*windows.shape[:3], -1)
    return signature_levels_from_increment_windows(lifted, max_depth=depth)


def _rolling_coded_signature_levels(
    path: Tensor,
    matrix: Tensor,
    channels: int,
    depth: int,
    stride: int = 1,
) -> tuple[Tensor, ...]:
    """Return dense coded signature levels for every valid increment window.

    ``matrix`` has shape ``(window, code_dimension)`` and acts independently on
    each signal channel. Overlapping windows use exact Chen removal, alphabet
    rebasing, and insertion. The state is recomputed directly after at most one
    window length of accumulated shift to bound floating-point drift.

    The code must be fixed and exactly shift-closed for ``stride``. Nonoverlapping
    windows use the parallel direct evaluator because there is no state to reuse.
    Level ``m`` has shape ``(batch, windows, channels * code_dimension, ..., *)``.
    """
    _floating_matrix(matrix, "matrix", 2)
    if matrix.requires_grad:
        raise ValueError("Rolling evaluation requires a fixed offset code.")
    validate_path_points(path, in_channels=channels)
    validate_local_signature_args(matrix.shape[0], stride, depth)
    if matrix.shape[0] > path.shape[1] - 1:
        raise ValueError("The code window cannot exceed the path increments.")

    increments = path[:, 1:] - path[:, :-1]
    code = matrix.to(device=path.device, dtype=path.dtype)
    window = int(matrix.shape[0])
    if stride >= window:
        return _direct_coded_signature_levels(increments, code, depth, stride)

    shift = _exact_shift_operator(matrix, stride).to(
        device=path.device,
        dtype=path.dtype,
    )
    alphabet_shift = torch.kron(
        torch.eye(channels, device=path.device, dtype=path.dtype),
        shift.contiguous(),
    )
    count = 1 + (increments.shape[1] - window) // stride
    current = _segment_signature_levels(increments, 0, code, depth)
    outputs = [[level] for level in current]
    refresh_every = max(1, window // stride)

    for index in range(1, count):
        start = index * stride
        if index % refresh_every == 0:
            current = _segment_signature_levels(increments, start, code, depth)
        else:
            old_start = start - stride
            prefix = _segment_signature_levels(
                increments,
                old_start,
                code[:stride],
                depth,
            )
            current = signature_multiply(signature_inverse(prefix), current)
            current = apply_linear_map_to_signature_levels(
                current,
                matrix=alphabet_shift,
                shared_dims=1,
            )
            suffix = _segment_signature_levels(
                increments,
                old_start + window,
                code[window - stride :],
                depth,
            )
            current = signature_multiply(current, suffix)
        for level_outputs, level in zip(outputs, current, strict=True):
            level_outputs.append(level)

    return tuple(torch.stack(level_outputs, dim=1) for level_outputs in outputs)


__all__: list[str] = []
