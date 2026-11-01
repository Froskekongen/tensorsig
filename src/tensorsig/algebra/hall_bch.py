"""Hall-basis BCH utilities for incremental log-signature computation.

This module provides utilities for computing log-signatures using the
Baker-Campbell-Hausdorff formula directly in Hall basis coordinates,
avoiding the need to compute the full signature first.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from .bch_coefficients import bch_coeffs
from .hall import hall_basis, logsigdim
from .hall_projection import (
    _hall_basis_tensors,
    _hall_element_depth,
    get_hall_projector,
)
from ..tensor_ops.products import lie_bracket


@lru_cache(maxsize=None)
def _dynkin_bch_schedule(depth: int) -> tuple[tuple[int, int, float], ...]:
    """Return a bracket schedule for BCH terms above the closed-form fast path.

    The Dynkin--Specht--Wever projection sends a word ``a_1 ... a_n`` to the
    left-nested bracket ``[..., [[a_1, a_2], a_3], ..., a_n]`` and acts as
    multiplication by ``n`` on homogeneous Lie elements.  Applying it to the
    exact tensor-word coefficients of ``log(exp(x) exp(y))`` therefore gives a
    basis-independent BCH evaluator for any nilpotent Lie algebra.
    """
    coefficients = {}
    for word, coefficient in bch_coeffs(width=2, depth=depth).items():
        if len(word) < 2 or word[0] == word[1]:
            continue
        canonical = (1, 2, *word[2:])
        sign = 1 if word[:2] == (1, 2) else -1
        coefficients[canonical] = coefficients.get(canonical, 0) + (
            sign * coefficient / len(word)
        )

    coefficients = {word: value for word, value in coefficients.items() if value}
    prefixes = {
        word[:length] for word in coefficients for length in range(2, len(word) + 1)
    }
    ordered = sorted(prefixes, key=lambda word: (len(word), word))
    positions = {word: index for index, word in enumerate(ordered)}
    return tuple(
        (
            -1 if len(word) == 2 else positions[word[:-1]],
            word[-1],
            float(coefficients.get(word, 0)),
        )
        for word in ordered
    )


@lru_cache(maxsize=None)
def _sparse_constants(width: int, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return COO structure constants without materializing ``dim**3``."""
    basis = hall_basis(width, depth)
    dim = len(basis)
    projector = get_hall_projector(
        width=width,
        depth=depth,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    basis_tensors = _hall_basis_tensors(width, depth)

    zeros_template = [
        torch.zeros((1, *([width] * current_depth)), dtype=torch.float64)
        for current_depth in range(1, depth + 1)
    ]

    index_blocks = []
    value_blocks = []
    for i, elem_i in enumerate(basis):
        depth_i = _hall_element_depth(elem_i)
        tensor_i = basis_tensors[elem_i]
        for j in range(i + 1, dim):
            elem_j = basis[j]
            total_depth = depth_i + _hall_element_depth(elem_j)
            if total_depth > depth:
                continue
            bracket_tensor = lie_bracket(tensor_i, basis_tensors[elem_j])
            log_sig_tensors = [t.clone() for t in zeros_template]
            log_sig_tensors[total_depth - 1] = bracket_tensor.unsqueeze(0)
            coeffs = projector.project(log_sig_tensors).squeeze(0)
            output_indices = torch.nonzero(coeffs, as_tuple=False).flatten()
            if not output_indices.numel():
                continue
            forward = torch.column_stack(
                (
                    torch.full_like(output_indices, i),
                    torch.full_like(output_indices, j),
                    output_indices,
                )
            )
            reverse = forward[:, [1, 0, 2]]
            index_blocks.extend((forward, reverse))
            selected = coeffs.index_select(0, output_indices)
            value_blocks.extend((selected, -selected))

    if not index_blocks:
        return (
            torch.empty((0, 3), dtype=torch.long),
            torch.empty(0, dtype=torch.float64),
        )
    indices = torch.cat(index_blocks)
    values = torch.cat(value_blocks)
    order = torch.argsort((indices[:, 0] * dim + indices[:, 1]) * dim + indices[:, 2])
    return indices.index_select(0, order), values.index_select(0, order)


class SparseBCH:
    """Truncated BCH for an explicitly supplied sparse Lie bracket.

    Parameters
    ----------
    dim : int
        Dimension of the Lie algebra state.
    depth : int
        Positive nilpotency step used to truncate BCH.
    sparse_indices : torch.Tensor
        Integer tensor of shape ``(nnz, 3)``. A row ``(i, j, k)`` identifies
        a nonzero contribution to the coefficient of basis vector ``k`` in
        ``[e_i, e_j]``.
    sparse_values : torch.Tensor
        Coefficients of shape ``(nnz,)`` corresponding to ``sparse_indices``.
        Multiple rows may target the same ``(i, j, k)`` entry.
    device : torch.device
        Device on which to store the bracket table.
    dtype : torch.dtype
        Data type for runtime computations.

    Notes
    -----
    The caller is responsible for supplying structure constants for a valid
    step-filtered Lie algebra. In particular, this runtime does not check
    antisymmetry or the Jacobi identity.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        sparse_indices: torch.Tensor,
        sparse_values: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if dim < 0:
            raise ValueError(f"dim must be non-negative, got {dim}")
        if not sparse_bch_supports_depth(depth):
            raise ValueError(f"depth must be positive, got {depth}")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating-point.")
        if sparse_indices.ndim != 2 or sparse_indices.shape[1] != 3:
            raise ValueError(
                "sparse_indices must have shape (nnz, 3), got "
                f"{tuple(sparse_indices.shape)}"
            )
        if sparse_indices.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("sparse_indices must have an integer dtype.")
        if sparse_values.ndim != 1 or sparse_values.shape[0] != len(sparse_indices):
            raise ValueError(
                "sparse_values must have shape (nnz,), got "
                f"{tuple(sparse_values.shape)}"
            )
        if not torch.isfinite(sparse_values).all():
            raise ValueError("sparse_values must be finite.")
        if sparse_indices.numel() and (
            torch.any(sparse_indices < 0) or torch.any(sparse_indices >= dim)
        ):
            raise ValueError("sparse_indices entries must lie in [0, dim)")

        self.dim = dim
        self.depth = depth
        self.device = torch.device(device)
        self.dtype = dtype
        sparse_indices = sparse_indices.to(device=self.device, dtype=torch.long)
        self._sparse_vals = sparse_values.to(device=self.device, dtype=dtype)
        self._left_idx = sparse_indices[:, 0].contiguous()
        self._right_idx = sparse_indices[:, 1].contiguous()
        self._output_idx = sparse_indices[:, 2].contiguous()
        self._bch_schedule = (
            _dynkin_bch_schedule(depth)
            if depth > 4 and self._output_idx.numel()
            else ()
        )

    def _validate_states(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"x and y must have trailing dimension {self.dim}.")
        if x.device != self.device or y.device != self.device:
            raise ValueError(f"x and y must be on {self.device}.")
        if x.dtype != self.dtype or y.dtype != self.dtype:
            raise ValueError(f"x and y must have dtype {self.dtype}.")

    def _bracket(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._output_idx.numel() == 0:
            return (x + y) * 0
        contrib = x[..., self._left_idx] * y[..., self._right_idx] * self._sparse_vals
        return torch.zeros_like(x).scatter_add(
            -1,
            self._output_idx.expand_as(contrib),
            contrib,
        )

    def bracket(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute ``[x, y]`` using the sparse structure constants.

        Both inputs have shape ``(..., dim)`` and the returned tensor has the
        same shape.
        """
        self._validate_states(x, y)
        return self._bracket(x, y)

    def bracket_sparse(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for :meth:`bracket`."""
        return self.bracket(x, y)

    def bch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute ``log(exp(x) exp(y))`` through the configured step."""
        self._validate_states(x, y)
        depth = self.depth
        z = x + y
        if depth == 1 or self._output_idx.numel() == 0:
            return z
        xy = self._bracket(x, y)
        z = z + 0.5 * xy
        if depth == 2:
            return z
        x_xy = self._bracket(x, xy)
        y_yx = self._bracket(y, self._bracket(y, x))
        z = z + (1.0 / 12.0) * (x_xy + y_yx)
        if depth == 3:
            return z
        z = z - (1.0 / 24.0) * self._bracket(y, x_xy)
        if depth == 4:
            return z

        # The explicit formula above is the fastest route through step four.
        # Higher steps use the universal Dynkin projection schedule instead.
        z = x + y
        brackets = []
        for parent, letter, coefficient in self._bch_schedule:
            value = (
                self._bracket(x, y)
                if parent < 0
                else self._bracket(brackets[parent], x if letter == 1 else y)
            )
            brackets.append(value)
            if coefficient:
                z = z + coefficient * value
        return z


class HallBCH(SparseBCH):
    """Truncated BCH on Hall-basis coordinates.

    This class implements the Baker-Campbell-Hausdorff formula directly in
    Hall basis coordinates, allowing incremental log-signature computation
    without materializing the full tensor-algebra signature.

    Parameters
    ----------
    width : int
        Path dimension (size of the alphabet).
    depth : int
        Positive nilpotency step.
    device : torch.device
        Device on which to store structure constants and perform computations.
    dtype : torch.dtype
        Data type for computations.

    Attributes
    ----------
    width : int
        Path dimension.
    depth : int
        Truncation depth.
    dim : int
        Dimension of the log-signature (logsigdim(width, depth)).
    device : torch.device
        Device for computations.
    dtype : torch.dtype
        Data type for computations.

    Examples
    --------
    >>> import torch
    >>> from tensorsig.algebra.hall_bch import HallBCH
    >>>
    >>> bch = HallBCH(width=2, depth=2, device=torch.device("cpu"), dtype=torch.float32)
    >>> x = torch.tensor([[1.0, 2.0, 0.0]])  # Hall coordinates (batch=1, dim=3)
    >>> y = torch.tensor([[3.0, 4.0, 0.0]])  # Hall coordinates (batch=1, dim=3)
    >>> result = bch.bch(x, y)
    >>> result.shape
    torch.Size([1, 3])
    """

    def __init__(
        self,
        width: int,
        depth: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.width = width
        sparse_idx, sparse_vals = _sparse_constants(width, depth)
        super().__init__(
            dim=logsigdim(width, depth),
            depth=depth,
            sparse_indices=sparse_idx,
            sparse_values=sparse_vals,
            device=device,
            dtype=dtype,
        )

    def increment_to_hall(self, delta: torch.Tensor) -> torch.Tensor:
        """Embed path increment (batch, width) into Hall coordinates.

        Converts a path increment (which lives in degree-1 of the free Lie algebra)
        into Hall basis coordinates by placing it in the first ``width`` components
        and zeroing the rest.

        Parameters
        ----------
        delta : torch.Tensor
            Tensor of shape ``(batch, width)`` representing path increments.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, self.dim)`` with the increment embedded
            in Hall coordinates.

        Examples
        --------
        >>> import torch
        >>> from tensorsig.algebra.hall_bch import HallBCH
        >>>
        >>> bch = HallBCH(width=2, depth=2, device=torch.device("cpu"), dtype=torch.float32)
        >>> delta = torch.tensor([[1.0, 2.0]])  # (batch=1, width=2)
        >>> result = bch.increment_to_hall(delta)
        >>> result.shape
        torch.Size([1, 3])
        >>> result[:, :2]  # First two components match delta
        tensor([[1., 2.]])
        """
        batch = delta.shape[0]
        out = torch.zeros(batch, self.dim, device=self.device, dtype=self.dtype)
        out[:, : self.width] = delta
        return out


def sparse_bch_supports_depth(depth: int) -> bool:
    """Return True if the BCH truncation is implemented for this depth.

    The runtime uses compact closed forms through step four and a universal
    Dynkin projection above it.  The latter is mathematically valid at every
    positive step, although its number of nested brackets grows exponentially.

    Parameters
    ----------
    depth : int
        Truncation depth to check.

    Returns
    -------
    bool
        True for every positive depth, False otherwise.
    """
    return depth >= 1
