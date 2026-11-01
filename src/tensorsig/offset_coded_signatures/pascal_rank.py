"""Proved characteristic-zero ranks for raw binary scalar Pascal codes.

The parameters mean P=2, C=1, a_r=x+r*y. The filtered rank projects primitive
VALUES onto words of y-weight at most d. It is not the ``max_degree`` filtered
feature count of a signature-coordinate chart. None means outside a proved
closed regime, never a conjectural extrapolation.
"""

from math import comb, gcd

from ._lie_quotients.compiler import _mobius

__all__ = [
    "binary_pascal_rank",
    "binary_pascal_profile",
    "binary_lie_weight_dimension",
]


def _validate_parameters(degree: int, window: int, weight: int) -> None:
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (degree, window, weight)
    ):
        raise TypeError("degree, window and weight must be integers.")
    if degree < 2 or not 2 <= window <= degree or not 0 <= weight <= degree:
        raise ValueError(
            "Require degree >= 2, 2 <= window <= degree, 0 <= weight <= degree."
        )


def binary_lie_weight_dimension(degree: int, weight: int) -> int:
    """Multigraded binary Witt dimension in degree >= 2."""
    if degree < 2 or not 0 <= weight <= degree:
        raise ValueError("Require degree >= 2 and 0 <= weight <= degree.")
    if weight in (0, degree):
        return 0
    divisor = gcd(degree, weight)
    return (
        sum(
            _mobius(e) * comb(degree // e, weight // e)
            for e in range(1, divisor + 1)
            if divisor % e == 0
        )
        // degree
    )


def binary_pascal_rank(degree: int, window: int, weight: int) -> int | None:
    """Return the exact cumulative rank where the solved-regime theorem applies."""
    _validate_parameters(degree, window, weight)
    if window == 2:
        ell = (degree - 1) // 2
        if degree in (2, 4):
            increments = (0, 1)
        elif degree == 3:
            increments = (0, 1, 1)
        elif degree % 2:
            increments = (0, 1, ell, ell - 1)
        else:
            gamma = degree // 6
            increments = (0, 1, gamma, ell - 1, ell - 1 - gamma)
        return sum(increments[: weight + 1])
    if (
        weight < window
        or window >= degree - 1
        or (
            weight <= window
            and (2 * window >= degree + 2 or (degree, window) == (5, 3))
        )
    ):
        return sum(binary_lie_weight_dimension(degree, k) for k in range(weight + 1))
    return None


def binary_pascal_profile(degree: int, window: int) -> tuple[int | None, ...]:
    """Cumulative value-side ranks at weights 0,...,degree; unknowns stay None."""
    _validate_parameters(degree, window, 0)
    return tuple(binary_pascal_rank(degree, window, d) for d in range(degree + 1))
