"""Diagnostics for local signature readouts."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


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
