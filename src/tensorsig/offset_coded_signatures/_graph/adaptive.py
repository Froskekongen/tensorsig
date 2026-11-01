"""Adaptive exact evaluation of one intermediate-rank tree moment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from math import factorial, prod

from ._graph_common import bernoulli_at_one
from ._graph_common import validate_composition as _validate_composition
from .tree_moments import (
    _prepare_tree,
    _young_averaged_derivative,
)
from .young_dp import YoungDpMetrics, _validate_metrics, _young_tree_derivative_dp


def _composition_is_universally_zero(composition: tuple[int, ...]) -> bool:
    """Return the proved all-graph whole-block zero classification."""
    if len(composition) == 1:
        return True
    return (
        len(composition) == 2
        and min(composition) == 1
        and max(composition) >= 3
        and max(composition) % 2 == 1
    )


def composition_is_universally_zero(composition: Sequence[int]) -> bool:
    """Return whether every tree moment for ``composition`` is zero.

    In degree at least two this holds exactly for one block, or for two blocks
    of sizes ``{1, r}`` with odd ``r >= 3``.
    """
    return _composition_is_universally_zero(_validate_composition(composition))


def _tree_moment_backend(composition: tuple[int, ...]) -> str:
    if _composition_is_universally_zero(composition):
        return "zero"
    if len(composition) == 2 and composition[-1] == 1:
        return "terminal_hook"
    degree = sum(composition)
    young_group_size = prod(factorial(part) for part in composition)
    # ponytail: this conservative measured crossover is deliberately one
    # number; add a topology-aware cost model only if misdispatch is material.
    return "young_dp" if young_group_size >= max(32, 4 * degree) else "explicit"


def tree_moment_backend(composition: Sequence[int]) -> str:
    """Return the backend selected by :func:`adaptive_tree_moment`."""
    return _tree_moment_backend(_validate_composition(composition))


def _terminal_hook_moment(tree, degree: int) -> Fraction:
    """LE.38, root at the last chronological singleton, with input signs."""
    arrows, _, _ = tree
    adjacency = [[] for _ in range(degree)]
    for left, right in arrows:
        adjacency[left].append(right)
        adjacency[right].append(left)
    root = degree - 1
    parent = [-1] * degree
    traversal = [root]
    parent[root] = root
    for vertex in traversal:
        for neighbor in adjacency[vertex]:
            if parent[neighbor] == -1:
                parent[neighbor] = vertex
                traversal.append(neighbor)
    sizes = [1] * degree
    for vertex in reversed(traversal[1:]):
        sizes[parent[vertex]] += sizes[vertex]
    reversals = sum((v, parent[v]) not in arrows for v in traversal[1:])
    return (
        (-1) ** reversals
        * Fraction(1 << (degree - 1), prod(sizes[:-1]))
        * bernoulli_at_one(degree - 1)[-1]
    )


def adaptive_tree_moment(
    composition: Sequence[int],
    reference_covers: Iterable[tuple[int, int]],
    *,
    metrics: YoungDpMetrics | None = None,
) -> Fraction:
    """Return one exact tree moment using the cheapest supported backend.

    Universal-zero compositions return immediately. Terminal-singleton blocks
    use the exact Bernoulli hook product. Other small Young groups use the
    explicit factorial oracle; larger groups use the proved root-rank DP.
    ``metrics`` is populated only when the DP is selected.
    """
    parts = _validate_composition(composition)
    _validate_metrics(metrics)
    degree = sum(parts)
    tree = _prepare_tree(reference_covers, degree)
    backend = _tree_moment_backend(parts)
    if backend == "zero":
        return Fraction()
    if backend == "terminal_hook":
        return _terminal_hook_moment(tree, degree)
    if backend == "young_dp":
        derivative = _young_tree_derivative_dp(parts, tree, metrics)
    else:
        derivative = _young_averaged_derivative(parts, tree)
    return (1 << (degree - 1)) * derivative


__all__ = [
    "adaptive_tree_moment",
    "composition_is_universally_zero",
    "tree_moment_backend",
]
