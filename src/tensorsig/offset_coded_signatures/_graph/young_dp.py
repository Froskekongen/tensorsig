"""Implicit Young average for IR1 tree moments.

The post-IR1 strategy account proves this root-rank recurrence. The maintained
scalar dispatcher uses it for larger Young groups. It uses exact
:class:`fractions.Fraction` arithmetic and never enumerates the Young subgroup
or linear extensions.

The proved optimized algorithm uses ``O(m**7)`` rational operations for a
derivative and ``O(m**4)`` stored state cells.  This implementation calls
``math.comb`` on demand rather than materializing the proof's Pascal cache.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial

from ._graph_common import validate_composition as _validate_composition
from .tree_moments import (
    _prepare_tree,
    _PreparedTree,
)

type _RankProfile = dict[tuple[int, int], Fraction]
type _ValueMessage = list[_RankProfile]


@dataclass(slots=True)
class YoungDpMetrics:
    """Accumulated size measurements for exact Young-DP calls.

    ``state_cells`` counts populated ``(value, size, rank)`` cells in every
    finalized vertex message.  ``peak_state_cells`` is the largest single
    partial or finalized vertex message.  Bit counts include every partial
    message observed.  Reusing an instance accumulates measurements across
    calls, including all polynomial evaluations used by a derivative.
    """

    polynomial_evaluations: int = 0
    state_cells: int = 0
    peak_state_cells: int = 0
    max_numerator_bits: int = 0
    max_denominator_bits: int = 0


def _validate_metrics(metrics: YoungDpMetrics | None) -> None:
    if metrics is not None and not isinstance(metrics, YoungDpMetrics):
        raise TypeError("metrics must be a YoungDpMetrics instance or None.")


def _observe_message(metrics: YoungDpMetrics, message: _ValueMessage) -> int:
    cells = sum(len(profile) for profile in message)
    metrics.peak_state_cells = max(metrics.peak_state_cells, cells)
    for profile in message:
        for value in profile.values():
            metrics.max_numerator_bits = max(
                metrics.max_numerator_bits,
                value.numerator.bit_length(),
            )
            metrics.max_denominator_bits = max(
                metrics.max_denominator_bits,
                value.denominator.bit_length(),
            )
    return cells


def _block_indices(composition: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        block_index for block_index, size in enumerate(composition) for _ in range(size)
    )


def _add_profile(
    target: _RankProfile,
    key: tuple[int, int],
    value: Fraction,
) -> None:
    if value:
        target[key] = target.get(key, Fraction()) + value


def _rank_prefix_or_suffix(
    profile: _RankProfile,
    child_below_parent: bool,
) -> _RankProfile:
    """Sum child profiles by how many child elements precede the parent."""
    result: _RankProfile = {}
    for size in sorted({size for size, _ in profile}):
        if child_below_parent:
            running = Fraction()
            for before_parent in range(size + 1):
                if before_parent:
                    running += profile.get((size, before_parent), Fraction())
                if running:
                    result[(size, before_parent)] = running
        else:
            running = sum(
                (profile.get((size, rank), Fraction()) for rank in range(1, size + 1)),
                start=Fraction(),
            )
            for before_parent in range(size + 1):
                if before_parent:
                    running -= profile.get((size, before_parent), Fraction())
                if running:
                    result[(size, before_parent)] = running
    return result


def _graft_profiles(
    current: _RankProfile,
    child: _RankProfile,
    child_below_parent: bool,
) -> _RankProfile:
    """Attach one equal-valued child component at the current boundary."""
    result: _RankProfile = {}
    child_sums = _rank_prefix_or_suffix(child, child_below_parent)
    for (parent_size, parent_rank), parent_count in current.items():
        for (child_size, before_parent), child_count in child_sums.items():
            coefficient = comb(
                parent_rank - 1 + before_parent,
                before_parent,
            ) * comb(
                parent_size - parent_rank + child_size - before_parent,
                child_size - before_parent,
            )
            _add_profile(
                result,
                (parent_size + child_size, parent_rank + before_parent),
                parent_count * child_count * coefficient,
            )
    return result


def _young_order_polynomial_value_dp(
    composition: tuple[int, ...],
    tree: _PreparedTree,
    n: int,
    metrics: YoungDpMetrics | None = None,
) -> Fraction:
    if metrics is not None:
        metrics.polynomial_evaluations += 1
    if n == 0:
        return Fraction()

    covers, children, traversal = tree
    degree = sum(composition)
    block = _block_indices(composition)
    factorials = tuple(factorial(size) for size in range(degree + 1))
    messages: list[_ValueMessage | None] = [None] * degree

    for vertex in reversed(traversal):
        by_value: _ValueMessage = [{(1, 1): Fraction(1)} for _ in range(n)]
        if metrics is not None:
            _observe_message(metrics, by_value)

        for child in children[vertex]:
            child_message = messages[child]
            assert child_message is not None
            closed = [
                sum(
                    (count / factorials[size] for (size, _), count in profile.items()),
                    start=Fraction(),
                )
                for profile in child_message
            ]
            prefix = [Fraction()] * (n + 1)
            for value in range(1, n + 1):
                prefix[value] = prefix[value - 1] + closed[value - 1]
            suffix = [Fraction()] * (n + 2)
            for value in range(n, 0, -1):
                suffix[value] = suffix[value + 1] + closed[value - 1]

            child_below_parent = (child, vertex) in covers
            same_block = block[child] == block[vertex]
            strict = (
                block[child] > block[vertex]
                if child_below_parent
                else block[vertex] > block[child]
            )
            next_by_value: _ValueMessage = []
            for parent_value, current in enumerate(by_value, start=1):
                if child_below_parent:
                    nonmerge = prefix[parent_value - int(strict)]
                else:
                    nonmerge = suffix[parent_value + int(strict)]
                if same_block:
                    nonmerge -= closed[parent_value - 1]

                updated = (
                    {key: count * nonmerge for key, count in current.items()}
                    if nonmerge
                    else {}
                )
                if same_block:
                    merged = _graft_profiles(
                        current,
                        child_message[parent_value - 1],
                        child_below_parent,
                    )
                    for key, count in merged.items():
                        _add_profile(updated, key, count)
                next_by_value.append(updated)
            by_value = next_by_value
            if metrics is not None:
                _observe_message(metrics, by_value)

        messages[vertex] = by_value
        if metrics is not None:
            metrics.state_cells += _observe_message(metrics, by_value)

    root_message = messages[0]
    assert root_message is not None
    return sum(
        (
            count / factorials[size]
            for profile in root_message
            for (size, _), count in profile.items()
        ),
        start=Fraction(),
    )


def young_dp_order_polynomial_value(
    composition: Sequence[int],
    reference_covers: Iterable[tuple[int, int]],
    n: int,
    *,
    metrics: YoungDpMetrics | None = None,
) -> Fraction:
    """Return the proved root-rank average ``Omega_bar(n)`` exactly.

    Composition blocks are consecutive vertex intervals and arithmetic is
    exact over :class:`fractions.Fraction`.
    """
    parts = _validate_composition(composition)
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer.")
    if n < 0:
        raise ValueError("n must be non-negative.")
    _validate_metrics(metrics)
    tree = _prepare_tree(reference_covers, sum(parts))
    return _young_order_polynomial_value_dp(parts, tree, n, metrics)


def _young_tree_derivative_dp(
    composition: tuple[int, ...],
    tree: _PreparedTree,
    metrics: YoungDpMetrics | None,
) -> Fraction:
    degree = sum(composition)
    return sum(
        (
            Fraction((-1) ** (n - 1) * comb(degree, n), n)
            * _young_order_polynomial_value_dp(composition, tree, n, metrics)
        )
        for n in range(1, degree + 1)
    )


def young_dp_tree_derivative(
    composition: Sequence[int],
    reference_covers: Iterable[tuple[int, int]],
    *,
    metrics: YoungDpMetrics | None = None,
) -> Fraction:
    """Return the implicit-Young value ``Omega_bar'(0)`` exactly."""
    parts = _validate_composition(composition)
    _validate_metrics(metrics)
    tree = _prepare_tree(reference_covers, sum(parts))
    return _young_tree_derivative_dp(parts, tree, metrics)


def young_dp_tree_moment(
    composition: Sequence[int],
    reference_covers: Iterable[tuple[int, int]],
    *,
    metrics: YoungDpMetrics | None = None,
) -> Fraction:
    """Return the exact IR1 moment ``2**(m - 1) Omega_bar'(0)``."""
    parts = _validate_composition(composition)
    _validate_metrics(metrics)
    degree = sum(parts)
    tree = _prepare_tree(reference_covers, degree)
    return (1 << (degree - 1)) * _young_tree_derivative_dp(parts, tree, metrics)
