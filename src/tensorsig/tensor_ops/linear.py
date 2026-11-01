"""Linear transformations and contractions for tensor-algebra levels.

A linear alphabet map acts on every tensor-coordinate axis of a homogeneous
level while preserving the declared shared leading axes. Signature-level maps
apply the same convention degree by degree. Readout contraction consumes
compiled feature blocks directly, avoiding materialization of a concatenated
feature tensor when a plan supports it.
"""

from math import prod

import torch
from torch import Tensor


__all__ = [
    "apply_linear_map_to_tensor_level",
    "apply_linear_map_to_signature_levels",
    "contract_features_with_readout",
]


def apply_linear_map_to_tensor_level(
    level: Tensor,
    matrix: Tensor,
    shared_dims: int,
) -> Tensor:
    """Apply a linear alphabet map to every tensor axis of one level.

    Parameters
    ----------
    level : Tensor
        Tensor with shape ``(*shared, in_width, ..., in_width)``.
    matrix : Tensor
        Linear map with shape ``(in_width, out_width)``. If ``v`` is a row
        vector of coordinates, the mapped vector is ``v @ matrix``.
    shared_dims : int
        Number of leading axes in ``level`` that are not tensor-coordinate axes.

    Returns
    -------
    Tensor
        Tensor with shape ``(*shared, out_width, ..., out_width)``.
    """
    if isinstance(shared_dims, bool) or not isinstance(shared_dims, int):
        raise TypeError("shared_dims must be an integer.")
    if shared_dims < 0:
        raise ValueError("shared_dims must be non-negative.")
    if level.ndim <= shared_dims:
        raise ValueError("level must contain at least one tensor-coordinate axis.")
    if matrix.ndim != 2:
        raise ValueError("matrix must have shape (in_width, out_width).")
    if matrix.dtype != level.dtype:
        raise ValueError("matrix and level must have the same dtype.")
    if matrix.device != level.device:
        raise ValueError("matrix and level must be on the same device.")

    degree = level.ndim - shared_dims
    in_width = matrix.shape[0]
    for axis_size in level.shape[shared_dims:]:
        if axis_size != in_width:
            raise ValueError(
                f"Expected tensor-coordinate axis size {in_width}, got {axis_size}."
            )

    out = level
    for axis in range(shared_dims, shared_dims + degree):
        out = torch.tensordot(out, matrix, dims=([axis], [0]))
        out = out.movedim(-1, axis)
    return out


def apply_linear_map_to_signature_levels(
    levels: list[Tensor],
    matrix: Tensor,
    shared_dims: int,
) -> list[Tensor]:
    """Apply a linear alphabet map to all homogeneous signature levels."""
    if not levels:
        raise ValueError("levels must be non-empty.")
    return [
        apply_linear_map_to_tensor_level(
            level,
            matrix=matrix,
            shared_dims=shared_dims,
        )
        for level in levels
    ]


def contract_features_with_readout(
    features: Tensor,
    readout: Tensor,
    shared_dims: int = 2,
) -> Tensor:
    """Contract trailing feature axes with a linear readout.

    Parameters
    ----------
    features : Tensor
        Feature tensor with shape ``(*shared, *feature_axes)``.
    readout : Tensor
        Readout tensor with shape ``(out_channels, *feature_axes)``.
    shared_dims : int, optional
        Number of leading axes in ``features`` to preserve.

    Returns
    -------
    Tensor
        Contracted tensor with shape ``(*shared, out_channels)``.
    """
    if isinstance(shared_dims, bool) or not isinstance(shared_dims, int):
        raise TypeError("shared_dims must be an integer.")
    if shared_dims < 0:
        raise ValueError("shared_dims must be non-negative.")
    if features.ndim < shared_dims:
        raise ValueError(
            f"features must have at least {shared_dims} shared dimensions, "
            f"got {features.ndim}."
        )
    expected_readout_ndim = features.ndim - shared_dims + 1
    if readout.ndim != expected_readout_ndim:
        raise ValueError(
            f"readout must have {expected_readout_ndim} dimensions, got {readout.ndim}."
        )
    if features.shape[shared_dims:] != readout.shape[1:]:
        raise ValueError(
            f"Feature axes {features.shape[shared_dims:]} do not match readout axes "
            f"{readout.shape[1:]}."
        )
    if readout.dtype != features.dtype:
        raise ValueError("readout and features must have the same dtype.")
    if readout.device != features.device:
        raise ValueError("readout and features must be on the same device.")

    shared_shape = features.shape[:shared_dims]
    out_channels = readout.shape[0]
    feature_size = prod(features.shape[shared_dims:])
    features_flat = features.reshape(*shared_shape, feature_size)
    readout_flat = readout.reshape(out_channels, feature_size)
    return torch.matmul(features_flat, readout_flat.T)
