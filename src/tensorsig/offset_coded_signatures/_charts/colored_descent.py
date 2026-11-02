"""Exact feature counts for generic offset-coded local signatures."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import itertools
import math

import torch
from torch import Tensor


def _validate_dimensions(
    kernel_size: int,
    code_dim: int,
    depth: int,
) -> None:
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive.")
    if code_dim < 1 or code_dim > kernel_size:
        raise ValueError("code_dim must satisfy 1 <= code_dim <= kernel_size.")
    if depth < 1:
        raise ValueError("depth must be positive.")


def _normalize_cuts(depth: int, cuts: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(int(cut) for cut in cuts))
    if len(set(normalized)) != len(normalized):
        raise ValueError("cuts must not contain duplicates.")
    if any(cut < 1 or cut >= depth for cut in normalized):
        raise ValueError(f"cuts must be positions in [1, {depth - 1}].")
    return normalized


@lru_cache(maxsize=None)
def bounded_gap_count(cutoff: int, code_dim: int, depth: int) -> int:
    """Count bounded gap words with total degree below ``cutoff``.

    This is

    ``#{alpha in {0, ..., code_dim - 1}**depth : sum(alpha) < cutoff}``.

    Parameters
    ----------
    cutoff : int
        Strict upper bound on the total gap degree.
    code_dim : int
        Effective dimension of the offset code.
    depth : int
        Homogeneous signature degree.

    Returns
    -------
    int
        The exact bounded-gap count.
    """
    if code_dim < 1:
        raise ValueError("code_dim must be positive.")
    if depth < 1:
        raise ValueError("depth must be positive.")
    if cutoff <= 0:
        return 0

    return sum(
        (-1) ** excluded
        * math.comb(depth, excluded)
        * math.comb(cutoff - 1 - excluded * code_dim + depth, depth)
        for excluded in range(min(depth, (cutoff - 1) // code_dim) + 1)
    )


def cut_rank_target(
    kernel_size: int,
    code_dim: int,
    depth: int,
    cuts: Iterable[int],
) -> int:
    """Return the generic/Pascal rank of a required-cut offset space.

    Parameters
    ----------
    kernel_size : int
        Number of offsets in the local window.
    code_dim : int
        Effective dimension of the offset code.
    depth : int
        Homogeneous signature degree.
    cuts : iterable of int
        One-based tensor positions that must be strict chronological cuts.

    Returns
    -------
    int
        The target rank ``h_(kernel_size - len(cuts), code_dim)(depth)``.
    """
    _validate_dimensions(kernel_size, code_dim, depth)
    normalized = _normalize_cuts(depth, cuts)
    return bounded_gap_count(kernel_size - len(normalized), code_dim, depth)


def temporal_standard_indices(
    kernel_size: int,
    code_dim: int,
    depth: int,
    cuts: Iterable[int],
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Enumerate bounded-gap temporal standard indices for one cut set.

    Parameters
    ----------
    kernel_size : int
        Number of offsets in the local window.
    code_dim : int
        Effective dimension of the offset code.
    depth : int
        Homogeneous signature degree.
    cuts : iterable of int
        One-based positions that require a strict chronological cut.
    device : torch.device or str, optional
        Device of the returned index tensors.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``alpha`` and its canonical nondecreasing offset words, both with
        shape ``(num_indices, depth)`` and dtype ``torch.long``.
    """
    _validate_dimensions(kernel_size, code_dim, depth)
    normalized = _normalize_cuts(depth, cuts)
    delta = (0, *(int(index in normalized) for index in range(1, depth)))
    alpha_rows = [
        alpha
        for alpha in itertools.product(range(code_dim), repeat=depth)
        if sum(alpha) + len(normalized) < kernel_size
    ]
    if not alpha_rows:
        empty = torch.empty((0, depth), device=device, dtype=torch.long)
        return empty, empty.clone()

    offset_rows = [
        tuple(itertools.accumulate(gap + shift for gap, shift in zip(alpha, delta)))
        for alpha in alpha_rows
    ]
    return (
        torch.tensor(alpha_rows, device=device, dtype=torch.long),
        torch.tensor(offset_rows, device=device, dtype=torch.long),
    )


def _composition_from_cuts(depth: int, cuts: tuple[int, ...]) -> tuple[int, ...]:
    boundaries = (0, *cuts, depth)
    return tuple(
        boundaries[index + 1] - boundaries[index]
        for index in range(len(boundaries) - 1)
    )


@lru_cache(maxsize=None)
def _ribbon_dimension_cached(
    in_channels: int,
    depth: int,
    cuts: tuple[int, ...],
) -> int:
    total = 0
    for subset_size in range(len(cuts) + 1):
        for subset in itertools.combinations(cuts, subset_size):
            symmetric_block_dimension = math.prod(
                math.comb(in_channels + block_size - 1, block_size)
                for block_size in _composition_from_cuts(depth, subset)
            )
            total += (-1) ** (len(cuts) - subset_size) * symmetric_block_dimension
    return total


def ribbon_dimension(
    in_channels: int,
    depth: int,
    cuts: Iterable[int],
) -> int:
    """Return the exact channel ribbon dimension for a cut set.

    Parameters
    ----------
    in_channels : int
        Signal-channel dimension.
    depth : int
        Homogeneous signature degree.
    cuts : iterable of int
        One-based cut positions defining the ribbon composition.

    Returns
    -------
    int
        ``r_beta(cuts)(1**in_channels)`` computed by inclusion-exclusion.
    """
    if in_channels < 1:
        raise ValueError("in_channels must be positive.")
    if depth < 1:
        raise ValueError("depth must be positive.")
    normalized = _normalize_cuts(depth, cuts)
    return _ribbon_dimension_cached(in_channels, depth, normalized)


@lru_cache(maxsize=None)
def generic_scalar_dimensions(
    kernel_size: int,
    code_dim: int,
    in_channels: int,
    depth: int,
) -> tuple[int, ...]:
    """Count all Pascal/colored-descent levels through ``depth`` exactly.

    The counters are Python integers. Channel prefix sums and temporal
    sliding sums take ``O(depth * kernel_size * in_channels)`` integer
    operations and ``O(kernel_size * in_channels)`` working counters.
    """
    _validate_dimensions(kernel_size, code_dim, depth)
    if in_channels < 1:
        raise ValueError("in_channels must be positive.")

    current = [[int(cost < code_dim)] * in_channels for cost in range(kernel_size)]
    counts = []
    for level in range(1, depth + 1):
        counts.append(sum(map(sum, current)))
        if level == depth:
            break
        channel_sums = []
        for cost, row in enumerate(current):
            previous = current[cost - 1] if cost else [0] * in_channels
            suffix = sum(previous)
            prefix = 0
            combined = []
            for channel in range(in_channels):
                prefix += row[channel]
                suffix -= previous[channel]
                combined.append(prefix + suffix)
            channel_sums.append(combined)
        next_counts = [[0] * in_channels for _ in range(kernel_size)]
        for channel in range(in_channels):
            total = 0
            for cost in range(kernel_size):
                total += channel_sums[cost][channel]
                if cost >= code_dim:
                    total -= channel_sums[cost - code_dim][channel]
                next_counts[cost][channel] = total
        current = next_counts
    return tuple(counts)


@lru_cache(maxsize=None)
def generic_scalar_dimension(
    kernel_size: int,
    code_dim: int,
    in_channels: int,
    depth: int,
) -> int:
    """Return the exact colored-descent feature count at one degree."""
    return generic_scalar_dimensions(kernel_size, code_dim, in_channels, depth)[-1]
