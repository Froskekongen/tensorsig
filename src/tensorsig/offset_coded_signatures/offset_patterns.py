"""Offset-pattern combinatorics for local signature layers."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
import itertools
import math


@lru_cache(maxsize=None)
def positive_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    """Return positive integer compositions of ``total`` into ``parts`` parts."""
    if total < 1:
        raise ValueError("total must be positive.")
    if parts < 1:
        raise ValueError("parts must be positive.")
    if parts > total:
        return ()
    if parts == 1:
        return ((total,),)

    result = []
    for first in range(1, total - parts + 2):
        for rest in positive_compositions(total - first, parts - 1):
            result.append((first, *rest))
    return tuple(result)


def is_nondecreasing(values: Sequence[int]) -> bool:
    """Return true when ``values`` are nondecreasing."""
    return all(left <= right for left, right in zip(values, values[1:]))


def offset_run_lengths(offsets: Sequence[int]) -> tuple[int, ...]:
    """Return run lengths for a nondecreasing offset tuple."""
    if not offsets:
        return ()
    if not is_nondecreasing(offsets):
        raise ValueError("offsets must be nondecreasing.")

    runs = []
    current = int(offsets[0])
    count = 0
    for offset in offsets:
        offset = int(offset)
        if offset == current:
            count += 1
            continue
        runs.append(count)
        current = offset
        count = 1
    runs.append(count)
    return tuple(runs)


def offset_support_size(offsets: Sequence[int]) -> int:
    """Return the number of distinct offsets."""
    return len(set(int(offset) for offset in offsets))


def offset_span(offsets: Sequence[int]) -> int:
    """Return ``max(offsets) - min(offsets)``."""
    if not offsets:
        raise ValueError("offsets must be non-empty.")
    return max(offsets) - min(offsets)


def offset_max_multiplicity(offsets: Sequence[int]) -> int:
    """Return the largest repeated-offset multiplicity."""
    if not offsets:
        raise ValueError("offsets must be non-empty.")
    return max(offsets.count(offset) for offset in set(offsets))


def raw_offset_weight_fraction(offsets: Sequence[int]) -> Fraction:
    """Return the raw signature multiplicity-factorial offset weight."""
    if not is_nondecreasing(offsets):
        return Fraction(0)

    denominator = 1
    for offset in set(offsets):
        denominator *= math.factorial(offsets.count(offset))
    return Fraction(1, denominator)


@lru_cache(maxsize=None)
def log_signature_offset_coefficient_fraction(offsets: tuple[int, ...]) -> Fraction:
    """Return the exact tensor-log coefficient for one offset word.

    The coefficient is the Goldberg/BCH tensor-word coefficient induced by
    ``log(exp(X_0) ... exp(X_{R-1}))`` for the specified offset pattern. For a
    sorted offset tuple, it is the ordered-offset tensor-log projection weight.
    For a non-sorted tuple, contiguous blocks that are not nondecreasing
    contribute zero, matching the full offset-labelled tensor-log coordinate.
    """
    depth = len(offsets)
    if depth < 1:
        raise ValueError("offsets must be non-empty.")

    total = Fraction(0)
    for parts in range(1, depth + 1):
        coeff = Fraction(1 if parts % 2 == 1 else -1, parts)
        for composition in positive_compositions(depth, parts):
            start = 0
            product = Fraction(1)
            for block_len in composition:
                stop = start + block_len
                product *= raw_offset_weight_fraction(offsets[start:stop])
                start = stop
                if product == 0:
                    break
            total += coeff * product
    return total


def log_signature_offset_coefficient(offsets: tuple[int, ...]) -> float:
    """Return the tensor-log coefficient for one offset word as ``float``."""
    return float(log_signature_offset_coefficient_fraction(offsets))


def validate_offset_filter_args(
    max_offset_span: int | None = None,
    max_offset_support: int | None = None,
    max_offset_multiplicity: int | None = None,
    min_abs_gamma: float = 0.0,
) -> None:
    """Validate common offset-pattern filter arguments."""
    if max_offset_span is not None and max_offset_span < 0:
        raise ValueError("max_offset_span must be non-negative.")
    if max_offset_support is not None and max_offset_support < 1:
        raise ValueError("max_offset_support must be positive.")
    if max_offset_multiplicity is not None and max_offset_multiplicity < 1:
        raise ValueError("max_offset_multiplicity must be positive.")
    if min_abs_gamma < 0.0:
        raise ValueError("min_abs_gamma must be non-negative.")


def symmetric_run_channel_tuples(
    in_channels: int,
    run_lengths: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Return channel tuples sorted inside each repeated-offset run.

    Parameters
    ----------
    in_channels : int
        Number of increment channels.
    run_lengths : tuple[int, ...]
        Lengths of consecutive equal-offset runs.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Channel tuples with nondecreasing channel labels inside each run.
        Keeping one such representative removes tensor-coordinate duplicates
        without changing the scalar monomial function class.
    """
    if in_channels < 1:
        raise ValueError("in_channels must be positive.")
    if any(run_length < 1 for run_length in run_lengths):
        raise ValueError("run lengths must be positive.")

    per_run = [
        tuple(itertools.combinations_with_replacement(range(in_channels), run_length))
        for run_length in run_lengths
    ]
    return tuple(
        tuple(channel for run_tuple in run_product for channel in run_tuple)
        for run_product in itertools.product(*per_run)
    )
