"""First Eulerian idempotent on homogeneous tensor words.

The exact permutation coefficients are shared by dense tensor projection and
quotient enveloping-algebra readouts. Projection preserves tensor dtype,
device, and autograd; the permutation formula has factorial cost in depth.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from math import comb

import torch
from torch import Tensor


__all__ = ["eulerian_permutations", "eulerian_project_columns"]


@lru_cache(maxsize=None)
def eulerian_permutations(
    length: int,
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    """Return permutations and exact first-Eulerian-idempotent coefficients.

    ``length`` is a non-negative integer. Degree zero has an empty schedule,
    since the first Eulerian idempotent annihilates the scalar unit.
    """
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError("length must be a non-negative integer.")
    if length == 0:
        return ()
    schedule = []
    for permutation in permutations(range(length)):
        descents = sum(
            left > right for left, right in zip(permutation, permutation[1:])
        )
        schedule.append(
            (
                permutation,
                Fraction((-1) ** descents, length * comb(length - 1, descents)),
            )
        )
    return tuple(schedule)


def eulerian_project_columns(matrix: Tensor, width: int, depth: int) -> Tensor:
    """Apply the first Eulerian idempotent to tensor-word columns.

    ``matrix`` has shape ``(width**depth, columns)`` in lexicographic word
    order. The columns are independent inputs, and the result has the same
    shape. Width and depth must be positive integers.
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
    projected = torch.zeros_like(tensor)
    for permutation, coefficient in eulerian_permutations(depth):
        projected = projected + float(coefficient) * tensor.permute(*permutation, depth)
    return projected.reshape(expected_rows, matrix.shape[1])
