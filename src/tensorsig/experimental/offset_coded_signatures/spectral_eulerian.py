"""Subset-cut alternative to factorial Eulerian projection.

The first Eulerian idempotent can be expressed as a polynomial in the convolution
operator Psi. On degree m, Psi sums the 2**m ordered subset/complement cuts,
and E = product((Psi - 2**q I) / (2 - 2**q), q=2,...,m).

This identity is exact over the reals. Floating-point evaluation has cancellation
and is not an exact rank certificate.
"""

from __future__ import annotations

import torch
from torch import Tensor


def spectral_eulerian_project_columns(
    matrix: Tensor,
    width: int,
    depth: int,
) -> Tensor:
    """Project tensor-word columns using (depth - 1) * 2**depth cuts.

    Rows use the same lexicographic tensor-word order as the standard factorial
    projector. The final axis indexes independent columns and is never permuted.
    Autograd, dtype, and device are preserved. Prefer float64 for chart discovery;
    verify numerical rank and residuals before using higher degrees or float32.
    """
    for name, value in (("width", width), ("depth", depth)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    expected_rows = width**depth
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(
            f"Expected matrix shape ({expected_rows}, columns), got {tuple(matrix.shape)}."
        )
    if not matrix.dtype.is_floating_point:
        raise ValueError("matrix must have a floating-point dtype.")
    tensor = matrix.reshape(*([width] * depth), matrix.shape[1])
    cuts = tuple(
        tuple(position for position in range(depth) if mask & (1 << position))
        + tuple(position for position in range(depth) if not mask & (1 << position))
        + (depth,)
        for mask in range(1 << depth)
    )
    # Each cut contributes separately: several cuts produce identical permutations.
    # Descending eigenvalues reduce intermediate growth, but not all cancellation.
    for order in range(depth, 1, -1):
        convolution = torch.zeros_like(tensor)
        for cut in cuts:
            convolution = convolution + tensor.permute(cut)
        eigenvalue = 2**order
        tensor = (convolution - eigenvalue * tensor) / (2 - eigenvalue)
    return tensor.reshape(expected_rows, matrix.shape[1])
