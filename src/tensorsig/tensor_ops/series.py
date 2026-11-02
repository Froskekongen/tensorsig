"""Validated operations on homogeneous truncated tensor-series levels.

Path-signature functions return flattened tensors. Convert them to homogeneous
levels before applying series multiplication or inversion, then flatten the
result again when a feature vector is needed.

Examples
--------
>>> import torch
>>> from tensorsig.classical.signatures import signature
>>> from tensorsig.tensor_ops.series import (
...     flatten_signature_levels,
...     signature_inverse,
...     unflatten_signature_levels,
... )
>>> path = torch.tensor([[[0.0, 0.0], [1.0, -1.0]]])
>>> flattened = signature(path, depth=2)
>>> levels = unflatten_signature_levels(flattened, width=2, depth=2)
>>> inverse = flatten_signature_levels(signature_inverse(levels))
>>> inverse.shape == flattened.shape
True
"""

from functools import lru_cache

import torch
from torch import Tensor

from .products import tensor_product


__all__ = [
    "flatten_signature_levels",
    "unflatten_signature_levels",
    "signature_inverse",
    "signature_multiply",
    "batch_restricted_exp",
    "signature_levels_from_increments",
]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def _series_layout(
    levels: list[Tensor],
    name: str,
) -> tuple[int, torch.Size, int]:
    """Validate homogeneous levels and return shared axes, shape, and width."""
    if not isinstance(levels, list):
        raise TypeError(f"{name} must be a list of tensors.")
    if not levels:
        raise ValueError(f"{name} must be non-empty.")
    if not isinstance(levels[0], Tensor):
        raise TypeError(f"{name}[0] must be a tensor.")
    if levels[0].ndim < 1:
        raise ValueError(f"{name}[0] must contain a tensor-coordinate axis.")

    shared_dims = levels[0].ndim - 1
    shared_shape = levels[0].shape[:shared_dims]
    width = levels[0].shape[-1]
    if width < 1:
        raise ValueError(f"{name} must have positive tensor-algebra width.")

    first = levels[0]
    for index, level in enumerate(levels):
        if not isinstance(level, Tensor):
            raise TypeError(f"{name}[{index}] must be a tensor.")
        degree = index + 1
        expected_shape = (*shared_shape, *([width] * degree))
        if tuple(level.shape) != expected_shape:
            raise ValueError(
                f"{name}[{index}] must have shape {expected_shape}; "
                f"got {tuple(level.shape)}."
            )
        if level.dtype != first.dtype:
            raise ValueError(f"All {name} tensors must have the same dtype.")
        if level.device != first.device:
            raise ValueError(f"All {name} tensors must be on the same device.")
    return shared_dims, shared_shape, width


def _signature_level_sizes(width: int, depth: int) -> list[int]:
    return [width**degree for degree in range(1, depth + 1)]


def unflatten_signature_levels(sig: Tensor, width: int, depth: int) -> list[Tensor]:
    """Reshape flattened tensor-series blocks into homogeneous levels.

    Parameters
    ----------
    sig : Tensor
        Tensor with arbitrary leading dimensions and trailing dimension
        ``width + width**2 + ... + width**depth``.
    width : int
        Tensor-algebra width.
    depth : int
        Maximum series degree.

    Returns
    -------
    list[Tensor]
        Level ``m - 1`` has shape
        ``(*sig.shape[:-1], *([width] * m))``.
    """
    width = _positive_integer(width, "width")
    depth = _positive_integer(depth, "depth")
    if sig.ndim < 1:
        raise ValueError("Signature tensor must have a trailing feature dimension.")

    expected = sum(_signature_level_sizes(width, depth))
    if sig.shape[-1] != expected:
        raise ValueError(
            f"Expected trailing dimension {expected}, got {sig.shape[-1]}."
        )

    leading_shape = sig.shape[:-1]
    levels: list[Tensor] = []
    offset = 0
    for current_depth in range(1, depth + 1):
        size = width**current_depth
        chunk = sig[..., offset : offset + size]
        levels.append(chunk.reshape(*leading_shape, *([width] * current_depth)))
        offset += size
    return levels


def flatten_signature_levels(levels: list[Tensor]) -> Tensor:
    """Flatten homogeneous tensor-series levels into level-wise blocks.

    Parameters
    ----------
    levels : list[Tensor]
        Non-empty levels sharing the same leading dimensions.

    Returns
    -------
    Tensor
        Series levels concatenated along one trailing feature dimension.
    """
    shared_dims, shared_shape, width = _series_layout(levels, "levels")
    flattened: list[Tensor] = []
    for degree, level in enumerate(levels, start=1):
        flattened.append(level.reshape(*shared_shape, width**degree))
    return torch.cat(flattened, dim=-1)


def _unflatten_signature(sig: Tensor, width: int, depth: int) -> list[Tensor]:
    """Unflatten a batched tensor series."""
    if sig.ndim != 2:
        raise ValueError(
            f"Signature must be a 2D tensor of shape (batch, sigdim); got {sig.shape}."
        )
    return unflatten_signature_levels(sig, width=width, depth=depth)


def _unflatten_stream_signature(
    sign_tensor: Tensor, width: int, depth: int
) -> list[Tensor]:
    """Unflatten a batched streaming tensor series."""
    if sign_tensor.ndim != 3:
        raise ValueError(
            "Stream signature must be a 3D tensor of shape (batch, steps, sigdim); "
            f"got {sign_tensor.shape}."
        )
    return unflatten_signature_levels(sign_tensor, width=width, depth=depth)


def signature_inverse(levels: list[Tensor]) -> list[Tensor]:
    """Compute the inverse of a truncated tensor series via Chen recursion.

    Parameters
    ----------
    levels : list[Tensor]
        Batched homogeneous series levels.

    Returns
    -------
    list[Tensor]
        Inverse levels with the same shapes as ``levels``.
    """
    shared_dims, _, _ = _series_layout(levels, "levels")
    inverse: list[Tensor] = []
    for depth_index, level in enumerate(levels):
        current = -level
        for index in range(depth_index):
            current = current - tensor_product(
                levels[index],
                inverse[depth_index - index - 1],
                shared_dims=shared_dims,
            )
        inverse.append(current)
    return inverse


def signature_multiply(left: list[Tensor], right: list[Tensor]) -> list[Tensor]:
    """Multiply two truncated tensor series using Chen's identity.

    Parameters
    ----------
    left, right : list[Tensor]
        Batched homogeneous levels with equal truncation depth.

    Returns
    -------
    list[Tensor]
        Product levels with the same shapes as the operands.
    """
    if not isinstance(left, list) or not isinstance(right, list):
        raise TypeError("left and right must be lists of tensors.")
    if len(left) != len(right):
        raise ValueError("Signatures must have the same depth for multiplication.")
    shared_dims, left_shared, left_width = _series_layout(left, "left")
    right_shared_dims, right_shared, right_width = _series_layout(right, "right")
    if (
        right_shared_dims != shared_dims
        or right_shared != left_shared
        or right_width != left_width
    ):
        raise ValueError("Signatures must have the same leading shape and width.")
    if left[0].dtype != right[0].dtype:
        raise ValueError("Signatures must have the same dtype.")
    if left[0].device != right[0].device:
        raise ValueError("Signatures must be on the same device.")

    product: list[Tensor] = []
    for depth_index in range(len(left)):
        current = left[depth_index] + right[depth_index]
        for index in range(depth_index):
            current = current + tensor_product(
                left[index],
                right[depth_index - index - 1],
                shared_dims=shared_dims,
            )
        product.append(current)
    return product


def _infer_width_from_signature_dim(sigdim: int, depth: int) -> int:
    """Infer tensor-algebra width from a flattened series dimension."""
    sigdim = _positive_integer(sigdim, "sigdim")
    depth = _positive_integer(depth, "depth")

    width = 1
    while True:
        total = 0
        power = width
        for _ in range(depth):
            total += power
            power *= width

        if total == sigdim:
            return width
        if total > sigdim or width > sigdim:
            raise ValueError(
                f"Signature dimension {sigdim} is incompatible with depth {depth}."
            )
        width += 1


@lru_cache(maxsize=None)
def _compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts == 1:
        return ((total,),)
    result = []
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            result.append((first, *rest))
    return tuple(result)


def _signature_to_logsignature_tensor(
    sig_tensors: list[Tensor], width: int, depth: int
) -> list[Tensor]:
    """Apply the formal logarithm to homogeneous tensor-series levels.

    Parameters
    ----------
    sig_tensors : list[Tensor]
        Batched homogeneous signature levels.
    width : int
        Tensor-algebra width.
    depth : int
        Maximum series degree.

    Returns
    -------
    list[Tensor]
        Tensor-log levels with the same homogeneous shapes.
    """
    if depth == 0 or not sig_tensors:
        return []

    device = sig_tensors[0].device
    dtype = sig_tensors[0].dtype
    batch_size = sig_tensors[0].shape[0]
    log_sig: list[Tensor] = []

    for current_depth in range(1, depth + 1):
        if current_depth > len(sig_tensors):
            shape = [batch_size] + [width] * current_depth
            log_sig.append(torch.zeros(shape, device=device, dtype=dtype))
            continue

        accumulator = sig_tensors[current_depth - 1].clone()
        for order in range(2, current_depth + 1):
            coefficient = (-1) ** (order + 1) / order
            for composition in _compositions(current_depth, order):
                term = sig_tensors[composition[0] - 1]
                for index in composition[1:]:
                    term = tensor_product(
                        term,
                        sig_tensors[index - 1],
                        shared_dims=1,
                    )
                accumulator = accumulator + coefficient * term

        log_sig.append(accumulator)

    return log_sig


def batch_restricted_exp(input: Tensor, depth: int) -> list[Tensor]:
    """Batched truncated tensor exponential with a shared batch axis.

    Computes the truncated tensor exponential exp(input) - 1 for batched inputs,
    returning homogeneous components at each depth level. Each batch element
    receives the homogeneous components, enabling efficient signature scans
    over a batch of paths.

    Parameters
    ----------
    input : Tensor
        Tensor of shape ``(batch, width)`` representing degree-1 elements.
    depth : int
        Truncation depth (>=1).

    Returns
    -------
    list[Tensor]
        List of length ``depth`` where entry ``k`` has shape
        ``(batch, width, ..., width)`` with ``k+1`` trailing width dimensions
        (equivalently ``width**(k+1)`` elements when flattened).

    Examples
    --------
    >>> import torch
    >>> from tensorsig.tensor_ops.series import batch_restricted_exp
    >>>
    >>> input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (batch=2, width=2)
    >>> result = batch_restricted_exp(input_tensor, depth=2)
    >>> len(result)
    2
    >>> result[0].shape  # depth 1
    torch.Size([2, 2])
    >>> result[1].shape  # depth 2
    torch.Size([2, 2, 2])
    """
    depth = _positive_integer(depth, "depth")
    if input.ndim != 2:
        raise ValueError(
            f"input must have shape (batch, width); got {tuple(input.shape)}."
        )
    if input.shape[1] < 1:
        raise ValueError("input must have positive tensor-algebra width.")
    if not input.is_floating_point() and not input.is_complex():
        raise ValueError("input must have a floating-point or complex dtype.")

    ret = [input]
    for i in range(2, depth + 1):
        ret.append(tensor_product(ret[-1], input / i, shared_dims=1))
    return ret


def signature_levels_from_increments(
    increments: Tensor,
    max_depth: int,
) -> list[Tensor]:
    """Multiply segment exponentials into homogeneous signature levels.

    Parameters
    ----------
    increments : Tensor
        Floating-point increments shaped ``(..., steps, width)``. Any leading
        batch or window axes are preserved; steps and width must be positive.
    max_depth : int
        Maximum homogeneous degree, a positive integer.

    Returns
    -------
    list[Tensor]
        Non-scalar levels, with level ``k - 1`` shaped
        ``(..., width, ..., width)`` using ``k`` trailing width axes.
    """
    max_depth = _positive_integer(max_depth, "max_depth")
    if increments.ndim < 2:
        raise ValueError("increments must have shape (..., steps, width).")
    if not increments.dtype.is_floating_point:
        raise ValueError("increments must have a floating-point dtype.")
    steps, width = increments.shape[-2:]
    if steps < 1 or width < 1:
        raise ValueError("increments must have positive step and width axes.")
    leading_shape = increments.shape[:-2]
    flattened = increments.reshape(-1, steps, width)
    levels = batch_restricted_exp(flattened[:, 0], depth=max_depth)
    for increment in flattened[:, 1:].unbind(dim=1):
        levels = _batch_mult_fused_restricted_exp(increment, levels)
    return [level.reshape(*leading_shape, *level.shape[1:]) for level in levels]


def _batch_mult_fused_restricted_exp(z: Tensor, A: list[Tensor]) -> list[Tensor]:
    """Batched fused update of truncated tensor exponentials.

    Updates a list of truncated exponential terms by multiplying with a new
    degree-1 element. This is used in the signature scan to incrementally
    update signatures as we process path increments.

    Parameters
    ----------
    z : Tensor
        Tensor of shape ``(batch, n_features)`` living in degree 1,
        representing a path increment.
    A : list[Tensor]
        List of current exponential terms; ``A[k]`` has shape
        ``(batch, n_features, ..., n_features)`` with ``k + 1`` trailing
        ``n_features`` axes.

    Returns
    -------
    list[Tensor]
        Updated list of tensors with the same shapes as ``A``, representing
        the exponential terms after multiplication by exp(z).

    """
    depth = len(A)
    dtype = z.dtype
    device = z.device
    divisors = torch.arange(1, depth + 1, dtype=dtype, device=device).flip(0)
    divisor_view = divisors.reciprocal().view(-1, *([1] * z.ndim))
    z_divided_full = z.unsqueeze(0) * divisor_view
    unit = torch.ones(*z.shape[:1], dtype=dtype, device=device)
    ret: list[Tensor] = []
    for depth_index in range(depth):
        current = unit
        z_divided = z_divided_full[depth - depth_index - 1 :]
        for i in range(depth_index + 1):
            current = A[i] + tensor_product(
                current,
                z_divided[i],
                shared_dims=1,
            )
        ret.append(current)
    return ret
