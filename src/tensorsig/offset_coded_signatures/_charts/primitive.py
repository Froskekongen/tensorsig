"""Compile minimal primitive-coordinate charts for fixed offset codes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

from tensorsig.algebra.eulerian import eulerian_project_columns
from tensorsig.algebra.lyndon import lyndon_words
from tensorsig.algebra.lyndon_words import WordsBasisElement

from .chronological import (
    _ChronologicalLevel,
    _build_chronological_level,
    _canonical_value_basis,
)


def _resolve_rank_tolerance(matrix: Tensor, tolerance: float | None) -> float:
    """Resolve the absolute singular-value cutoff used for numerical rank."""
    if tolerance is not None:
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("rank_tolerance must be finite and non-negative.")
        return float(tolerance)
    if matrix.numel() == 0 or min(matrix.shape) == 0:
        return 0.0
    largest = float(torch.linalg.svdvals(matrix)[0])
    return torch.finfo(matrix.dtype).eps * max(matrix.shape) * largest


def _rank_revealing_subspaces(
    matrix: Tensor,
    tolerance: float | None,
) -> tuple[Tensor, Tensor, float]:
    """Return SVD-truncated row/column subspaces and the resolved tolerance."""
    resolved = (
        _resolve_rank_tolerance(matrix, tolerance) if tolerance is not None else 0.0
    )
    if matrix.numel() == 0 or min(matrix.shape) == 0:
        return (
            matrix.new_empty(matrix.shape[0], 0),
            matrix.new_empty(matrix.shape[1], 0),
            resolved,
        )
    left, singular_values, right_transpose = torch.linalg.svd(
        matrix,
        full_matrices=False,
    )
    if tolerance is None:
        resolved = (
            torch.finfo(matrix.dtype).eps
            * max(matrix.shape)
            * float(singular_values[0])
        )
    rank = int((singular_values > resolved).sum().item())
    return left[:, :rank], right_transpose[:rank].T, resolved


def _flat_word_index(word: Sequence[int], width: int) -> int:
    index = 0
    for letter in word:
        index = index * width + int(letter) - 1
    return index


def _word_temporal_degree(word: Sequence[int], code_dim: int) -> int:
    """Return temporal weight for the channel-major projected alphabet."""
    return sum((int(letter) - 1) % code_dim for letter in word)


def _ordered_independent_rows(
    row_subspace: Tensor,
    ordered_indices: Sequence[int],
) -> tuple[int, ...]:
    """Select ordered coordinate rows from an orthonormal numerical subspace."""
    rank = row_subspace.shape[1]
    if rank == 0:
        return ()
    orthogonal = torch.empty(
        rank,
        0,
        dtype=row_subspace.dtype,
        device=row_subspace.device,
    )
    tolerance = math.sqrt(torch.finfo(row_subspace.dtype).eps) * max(row_subspace.shape)
    selected = []
    for index in ordered_indices:
        residual = row_subspace[index]
        if orthogonal.shape[1]:
            residual = residual - orthogonal @ (orthogonal.T @ residual)
            residual = residual - orthogonal @ (orthogonal.T @ residual)
        norm = float(residual.norm())
        if norm <= tolerance:
            continue
        selected.append(int(index))
        orthogonal = torch.cat(
            (orthogonal, (residual / norm).unsqueeze(1)),
            dim=1,
        )
        if len(selected) == rank:
            break
    if len(selected) != rank:
        raise RuntimeError("Could not select the numerical subspace coordinates.")
    return tuple(selected)


def _null_space(matrix: Tensor, tolerance: float) -> Tensor:
    """Return a column basis for the numerical null space."""
    if matrix.shape[1] == 0:
        return torch.empty(0, 0, dtype=matrix.dtype, device=matrix.device)
    if matrix.shape[0] == 0:
        return torch.eye(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    _, singular_values, right = torch.linalg.svd(matrix, full_matrices=True)
    rank = int((singular_values > tolerance).sum().item())
    return right[rank:].T


def _decode_flat_words(
    flat_indices: Tensor,
    width: int,
    code_dim: int,
    depth: int,
) -> tuple[Tensor, Tensor]:
    temporal_rows = []
    channel_rows = []
    for flat_index in flat_indices.tolist():
        letters = []
        remainder = int(flat_index)
        for power in reversed(range(depth)):
            divisor = width**power
            letters.append(remainder // divisor)
            remainder %= divisor
        temporal_rows.append(tuple(letter % code_dim for letter in letters))
        channel_rows.append(tuple(letter // code_dim for letter in letters))
    return (
        torch.tensor(temporal_rows, dtype=torch.long).reshape(-1, depth),
        torch.tensor(channel_rows, dtype=torch.long).reshape(-1, depth),
    )


def _dynamic_signature_basis(
    code: Tensor,
    in_channels: int,
    depth: int,
    rank_tolerance: float | None,
) -> _ChronologicalLevel:
    """Build a deterministic tensor-word chart outside the Pascal chart."""
    kernel_size, code_dim = map(int, code.shape)
    width = code_dim * in_channels
    value_basis = _canonical_value_basis(code, in_channels, depth)
    row_subspace, column_subspace, tolerance = _rank_revealing_subspaces(
        value_basis,
        rank_tolerance,
    )
    source_rank = row_subspace.shape[1]
    column_indices = _ordered_independent_rows(
        column_subspace,
        range(value_basis.shape[1]),
    )

    row_weights = []
    for flat_index in range(width**depth):
        remainder = flat_index
        weight = 0
        for power in reversed(range(depth)):
            divisor = width**power
            letter = remainder // divisor
            remainder %= divisor
            weight += letter % code_dim
        row_weights.append(weight)
    ordered_rows = sorted(
        range(width**depth),
        key=lambda index: (row_weights[index], index),
    )
    pivot_rows = _ordered_independent_rows(
        row_subspace,
        ordered_rows,
    )
    value_basis = value_basis[:, column_indices]
    flat_indices = torch.tensor(pivot_rows, dtype=torch.long)
    coordinate_matrix = value_basis.index_select(0, flat_indices)
    if source_rank:
        reconstruction = value_basis @ torch.linalg.solve(
            coordinate_matrix,
            torch.eye(source_rank, dtype=torch.float64),
        )
    else:
        reconstruction = torch.empty(width**depth, 0, dtype=torch.float64)
    temporal, channels = _decode_flat_words(
        flat_indices,
        width=width,
        code_dim=code_dim,
        depth=depth,
    )
    return _ChronologicalLevel(
        reconstruction_matrix=reconstruction,
        flat_indices=flat_indices,
        temporal_indices=temporal,
        channel_indices=channels,
        descent_mask=channels[:, :-1] > channels[:, 1:],
        window=kernel_size,
        code_dimension=code_dim,
        channels=in_channels,
        depth=depth,
        full_feature_count=source_rank,
    )


def _build_signature_source_basis(
    code: Tensor,
    in_channels: int,
    depth: int,
    rank_tolerance: float | None,
) -> _ChronologicalLevel:
    if rank_tolerance is None:
        try:
            return _build_chronological_level(
                code,
                channels=in_channels,
                depth=depth,
            )
        except ValueError as error:
            if "outside the admissible projected-word chart" not in str(error):
                raise
    return _dynamic_signature_basis(
        code,
        in_channels=in_channels,
        depth=depth,
        rank_tolerance=rank_tolerance,
    )


@dataclass(frozen=True)
class _PrimitiveLevel:
    """Compiled primitive coordinates for one homogeneous degree."""

    signature_basis: _ChronologicalLevel
    ambient_words: tuple[WordsBasisElement, ...]
    feature_matrix: Tensor
    ambient_decoder: Tensor | None
    primitive_dimension: int
    rank_tolerance: float
    chart_margin: float | None
    _runtime_cache: dict[
        tuple[torch.device, torch.dtype],
        tuple[Tensor, Tensor, Tensor | None],
    ] = field(init=False, default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        source_dimension = self.signature_basis.feature_count
        if (
            self.feature_matrix.ndim != 2
            or self.feature_matrix.shape[1] != source_dimension
        ):
            raise ValueError("feature_matrix has an invalid source dimension.")
        if self.primitive_dimension < self.feature_count:
            raise ValueError(
                "primitive_dimension cannot be smaller than feature_count."
            )
        if self.ambient_decoder is not None and self.ambient_decoder.shape != (
            len(self.ambient_words),
            self.feature_count,
        ):
            raise ValueError("ambient_decoder has an invalid shape.")
        if self.is_lossless and self.ambient_decoder is None:
            raise ValueError("A lossless primitive level requires an ambient decoder.")

    @property
    def depth(self) -> int:
        return self.signature_basis.depth

    @property
    def feature_count(self) -> int:
        return int(self.feature_matrix.shape[0])

    @property
    def is_lossless(self) -> bool:
        return self.feature_count == self.primitive_dimension

    def runtime_tensors(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        key = (device, dtype)
        cached = self._runtime_cache.get(key)
        if cached is not None:
            return cached
        bound = (
            self.signature_basis.flat_indices.to(device=device),
            self.feature_matrix.to(device=device, dtype=dtype),
            (
                None
                if self.ambient_decoder is None
                else self.ambient_decoder.to(device=device, dtype=dtype)
            ),
        )
        self._runtime_cache[key] = bound
        return bound


def compile_primitive_chart(
    offset_code: Tensor,
    in_channels: int,
    depth: int,
    *,
    max_degree: int | None = None,
    rank_tolerance: float | None = None,
) -> _PrimitiveLevel:
    """Compile a minimal local primitive chart for one degree."""
    if offset_code.ndim != 2:
        raise ValueError("offset_code must have shape (kernel_size, code_dim).")
    if not offset_code.dtype.is_floating_point:
        raise ValueError("offset_code must have a floating-point dtype.")
    if in_channels < 1:
        raise ValueError("in_channels must be positive.")
    if depth < 1:
        raise ValueError("depth must be positive.")
    if max_degree is not None and (
        not isinstance(max_degree, int)
        or isinstance(max_degree, bool)
        or max_degree < 0
    ):
        raise ValueError("max_degree must be a non-negative integer or None.")

    code = offset_code.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(code).all():
        raise ValueError("offset_code must contain only finite values.")
    kernel_size, code_dim = map(int, code.shape)
    if code_dim < 1 or code_dim > kernel_size:
        raise ValueError("code_dim must satisfy 1 <= code_dim <= kernel_size.")

    signature_basis = _build_signature_source_basis(
        code,
        in_channels=in_channels,
        depth=depth,
        rank_tolerance=rank_tolerance,
    )
    width = code_dim * in_channels
    primitive_values = eulerian_project_columns(
        signature_basis.reconstruction_matrix,
        width,
        depth,
    )
    ambient_words = tuple(
        word for word in lyndon_words(width, depth) if len(word) == depth
    )
    ambient_indices = torch.tensor(
        [_flat_word_index(word, width) for word in ambient_words],
        dtype=torch.long,
    )
    ambient_coordinates = primitive_values.index_select(0, ambient_indices)
    primitive_subspace, _, tolerance = _rank_revealing_subspaces(
        ambient_coordinates,
        rank_tolerance,
    )
    primitive_dimension = primitive_subspace.shape[1]
    ordered_indices = sorted(
        range(len(ambient_words)),
        key=lambda index: (
            _word_temporal_degree(ambient_words[index], code_dim),
            ambient_words[index],
        ),
    )
    selected = torch.tensor(
        _ordered_independent_rows(
            primitive_subspace,
            ordered_indices,
        ),
        dtype=torch.long,
    )
    base_coordinates = ambient_coordinates.index_select(0, selected)

    if primitive_dimension:
        reconstruction = torch.linalg.lstsq(
            base_coordinates.T,
            ambient_coordinates.T,
        ).solution.T
    else:
        reconstruction = torch.empty(
            len(ambient_words),
            0,
            dtype=torch.float64,
        )

    source_degrees = signature_basis.temporal_degree
    if max_degree is None or not torch.any(source_degrees > max_degree):
        feature_matrix = base_coordinates
        decoder: Tensor | None = reconstruction
    else:
        high_matrix = base_coordinates[:, source_degrees > max_degree].T
        _, _, high_tolerance = _rank_revealing_subspaces(
            high_matrix,
            rank_tolerance,
        )
        transform = _null_space(
            high_matrix,
            high_tolerance,
        )
        feature_matrix = transform.T @ base_coordinates
        decoder = (
            reconstruction @ transform
            if transform.shape[1] == primitive_dimension
            else None
        )

    return _PrimitiveLevel(
        signature_basis=signature_basis,
        ambient_words=ambient_words,
        feature_matrix=feature_matrix,
        ambient_decoder=decoder,
        primitive_dimension=primitive_dimension,
        rank_tolerance=tolerance,
        chart_margin=(
            float(torch.linalg.svdvals(base_coordinates)[-1])
            if base_coordinates.numel()
            else None
        ),
    )
