"""Tensor products and Lie brackets with explicit shared leading axes.

``shared_dims`` is the number of leading batch, time, or window axes preserved
elementwise. Every remaining axis participates in the outer product. This one
operation therefore covers scalar, ordinary, batched, and batched-sequence
products without shape-specific aliases.
"""

from torch import Tensor


__all__ = ["tensor_product", "lie_bracket"]


def tensor_product(
    x: Tensor,
    y: Tensor,
    *,
    shared_dims: int = 0,
) -> Tensor:
    """Compute ``x ⊗ y`` while preserving shared leading axes.

    Parameters
    ----------
    x, y : Tensor
        Tensor operands. Their first ``shared_dims`` axes must have exactly the
        same shape; all remaining axes participate in the tensor product.
        Scalars are supported when ``shared_dims=0``.
    shared_dims : int, optional
        Number of leading axes to preserve. The default computes an ordinary
        outer product.

    Returns
    -------
    Tensor
        A tensor with shape
        ``(*shared, *x.shape[shared_dims:], *y.shape[shared_dims:])``.
    """
    if isinstance(shared_dims, bool) or not isinstance(shared_dims, int):
        raise TypeError("shared_dims must be an integer.")
    if shared_dims < 0:
        raise ValueError("shared_dims must be non-negative.")
    if x.ndim < shared_dims or y.ndim < shared_dims:
        raise ValueError(
            f"Expected both operands to have at least {shared_dims} dimensions; "
            f"got x.ndim={x.ndim} and y.ndim={y.ndim}."
        )

    x_shared = x.shape[:shared_dims]
    y_shared = y.shape[:shared_dims]
    if x_shared != y_shared:
        raise ValueError(
            "Shared leading dimensions must match exactly; "
            f"got {tuple(x_shared)} and {tuple(y_shared)}."
        )

    x_axes = x.ndim - shared_dims
    y_axes = y.ndim - shared_dims
    x_view = x.reshape((*x.shape, *([1] * y_axes)))
    y_view = y.reshape((*y_shared, *([1] * x_axes), *y.shape[shared_dims:]))
    return x_view * y_view


def lie_bracket(
    x: Tensor,
    y: Tensor,
    *,
    shared_dims: int = 0,
) -> Tensor:
    """Compute the commutator ``[x, y] = x ⊗ y - y ⊗ x``.

    ``shared_dims`` has the same meaning and validation as in
    [`tensor_product`][tensorsig.tensor_ops.products.tensor_product].
    """
    xy = tensor_product(x, y, shared_dims=shared_dims)
    yx = tensor_product(y, x, shared_dims=shared_dims)
    if xy.shape != yx.shape:
        raise ValueError(
            "Lie-bracket operands must produce matching product shapes; "
            f"got {tuple(xy.shape)} and {tuple(yx.shape)}."
        )
    return xy - yx
