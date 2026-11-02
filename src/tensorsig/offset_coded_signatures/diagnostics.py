"""Offset visibility, phase-kernel ranks, and local readout diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch

from torch import Tensor

from .offset_patterns import raw_offset_weight_fraction


@dataclass(frozen=True)
class OffsetTieDecomposition:
    """Offset-tied and offset-contrast parts of a degree-one readout."""

    tied: Tensor
    contrast: Tensor


def offset_tie_decomposition(weight: Tensor) -> OffsetTieDecomposition:
    """Split degree-one weights into offset-tied and contrast components.

    Parameters
    ----------
    weight : Tensor
        Degree-one readout weights with shape ``(out_channels, kernel_size,
        channels)`` or ``(kernel_size, channels)``.

    Returns
    -------
    OffsetTieDecomposition
        ``tied`` is constant across the offset axis and ``contrast`` has zero
        mean across offsets. A zero contrast means the first-order readout
        factors through the ordinary no-time local signature.
    """
    if weight.ndim == 2:
        mean = weight.mean(dim=0, keepdim=True)
    elif weight.ndim == 3:
        mean = weight.mean(dim=1, keepdim=True)
    else:
        raise ValueError(
            "weight must have shape (kernel_size, channels) or "
            "(out_channels, kernel_size, channels)."
        )
    tied = mean.expand_as(weight)
    return OffsetTieDecomposition(tied=tied, contrast=weight - tied)


def offset_contrast_energy(
    weight: Tensor,
    relative: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Return the size of the offset-contrast component of degree-one weights.

    Parameters
    ----------
    weight : Tensor
        Degree-one readout weights with shape ``(out_channels, kernel_size,
        channels)`` or ``(kernel_size, channels)``.
    relative : bool, optional
        When true, divide contrast norm by total norm.
    eps : float, optional
        Denominator floor for relative energy.

    Returns
    -------
    Tensor
        Scalar tensor for 2D input, or one value per output channel for 3D
        input.
    """
    decomposition = offset_tie_decomposition(weight)
    if weight.ndim == 2:
        contrast_norm = decomposition.contrast.norm()
        if not relative:
            return contrast_norm
        return contrast_norm / weight.norm().clamp_min(eps)

    contrast_norm = decomposition.contrast.flatten(start_dim=1).norm(dim=1)
    if not relative:
        return contrast_norm
    total_norm = weight.flatten(start_dim=1).norm(dim=1).clamp_min(eps)
    return contrast_norm / total_norm


@dataclass(frozen=True)
class PhaseKernelRankSummary:
    """Numerical rank summary for an offset-filter matrix."""

    rank: int
    tolerance: float
    singular_values: Tensor
    condition_number: float


def _as_grid_tensor(
    grid: Tensor | list[float] | tuple[float, ...],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Convert and validate a local interval grid."""
    grid_tensor = torch.as_tensor(grid, device=device, dtype=dtype)
    if grid_tensor.ndim != 1:
        raise ValueError("grid must be one-dimensional.")
    if grid_tensor.numel() < 2:
        raise ValueError("grid must contain at least two points.")
    if torch.any(grid_tensor[1:] <= grid_tensor[:-1]):
        raise ValueError("grid points must be strictly increasing.")
    return grid_tensor


def no_phase_kernel_matrix(
    kernel_size: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the no-phase first-order offset matrix.

    This is the single all-ones column. Its rank is one, so only offset-tied
    first-order filters are visible.
    """
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive.")
    return torch.ones(kernel_size, 1, device=device, dtype=dtype)


def no_phase_higher_order_kernel_matrix(
    kernel_size: int,
    order: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the no-phase order-``k`` phase-kernel matrix.

    The returned matrix has one column, corresponding to the empty deterministic
    phase in every gap. Rows are indexed by all ordered offset tuples in
    lexicographic order. Entries are the raw ordered-offset signature weights
    for nondecreasing offset tuples and zero for decreasing tuples.
    """
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive.")
    if order < 1:
        raise ValueError("order must be positive.")

    values = []
    for offsets in itertools.product(range(kernel_size), repeat=order):
        values.append(float(raw_offset_weight_fraction(offsets)))
    return torch.tensor(values, device=device, dtype=dtype).reshape(-1, 1)


def scalar_time_moment_matrix(
    grid: Tensor | list[float] | tuple[float, ...],
    max_depth: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return interval-averaged scalar-time moment filters.

    Column ``d`` is the interval average of ``s**d`` for ``d=0..max_depth-1``.
    These columns span exactly the first-order offset filters visible to a
    depth-``max_depth`` ordinary signature with one deterministic scalar time
    channel.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be positive.")
    grid_tensor = _as_grid_tensor(grid, device=device, dtype=dtype)
    left = grid_tensor[:-1]
    right = grid_tensor[1:]
    delta = right - left

    columns = []
    for degree in range(max_depth):
        columns.append((right ** (degree + 1) - left ** (degree + 1)) / delta)
        columns[-1] = columns[-1] / (degree + 1)
    return torch.stack(columns, dim=1)


def cumulative_offset_clock_matrix(
    grid: Tensor | list[float] | tuple[float, ...] | int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the depth-two cumulative offset-clock visibility matrix.

    The first column is the empty phase kernel. The remaining ``R - 1`` columns
    are interval averages of cumulative offset clocks. The matrix has full rank
    ``R`` for every valid grid.
    """
    if isinstance(grid, int):
        if grid < 1:
            raise ValueError("kernel_size must be positive.")
        kernel_size = grid
    else:
        grid_tensor = _as_grid_tensor(grid, device=device, dtype=dtype)
        kernel_size = int(grid_tensor.numel() - 1)

    matrix = torch.ones(kernel_size, kernel_size, device=device, dtype=dtype)
    if kernel_size == 1:
        return matrix

    for row in range(kernel_size):
        for clock in range(kernel_size - 1):
            if row < clock:
                value = 0.0
            elif row == clock:
                value = 0.5
            else:
                value = 1.0
            matrix[row, clock + 1] = value
    return matrix


def offset_retention_identity_matrix(
    kernel_size: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the depth-one first-order matrix for full offset retention."""
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive.")
    return torch.eye(kernel_size, device=device, dtype=dtype)


def default_rank_tolerance(matrix: Tensor, singular_values: Tensor) -> float:
    """Return a dtype-aware default numerical rank tolerance."""
    if singular_values.numel() == 0:
        return 0.0
    dtype = matrix.dtype if matrix.dtype.is_floating_point else torch.float64
    eps = torch.finfo(dtype).eps
    return float(max(matrix.shape) * eps * singular_values.max().item())


def phase_kernel_rank_summary(
    matrix: Tensor,
    tol: float | None = None,
) -> PhaseKernelRankSummary:
    """Return rank, singular values, and conditioning for an offset matrix."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    singular_values = torch.linalg.svdvals(matrix)
    tolerance = default_rank_tolerance(matrix, singular_values) if tol is None else tol
    rank = int((singular_values > tolerance).sum().item())
    if singular_values.numel() == 0 or rank == 0:
        condition_number = math.inf
    else:
        smallest = singular_values[rank - 1].item()
        condition_number = (
            math.inf if smallest == 0.0 else singular_values[0].item() / smallest
        )
    return PhaseKernelRankSummary(
        rank=rank,
        tolerance=float(tolerance),
        singular_values=singular_values,
        condition_number=float(condition_number),
    )


def invisible_offset_contrasts(matrix: Tensor, tol: float | None = None) -> Tensor:
    """Return an orthonormal basis for first-order invisible offset contrasts.

    Columns of the returned tensor span ``ker(matrix.T)``. If the matrix has
    full row rank, the returned tensor has shape ``(kernel_size, 0)``.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    transposed = matrix.T
    _, singular_values, vh = torch.linalg.svd(transposed, full_matrices=True)
    tolerance = default_rank_tolerance(matrix, singular_values) if tol is None else tol
    rank = int((singular_values > tolerance).sum().item())
    kernel_size = matrix.shape[0]
    if rank >= kernel_size:
        return torch.empty(kernel_size, 0, device=matrix.device, dtype=matrix.dtype)
    return vh[rank:].T.contiguous()


def finite_difference_contrast(
    order: int,
    kernel_size: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return an equal-grid finite-difference contrast of the given order."""
    if order < 0:
        raise ValueError("order must be non-negative.")
    if kernel_size <= order:
        raise ValueError("kernel_size must be at least order + 1.")
    contrast = torch.zeros(kernel_size, device=device, dtype=dtype)
    for offset in range(order + 1):
        contrast[offset] = ((-1.0) ** offset) * math.comb(order, offset)
    return contrast
