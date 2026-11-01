"""Canonical chronological word coordinates for coded local signatures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import itertools

import torch
from torch import Tensor

from .colored_descent import (
    generic_scalar_dimension,
    generic_scalar_dimensions,
    ribbon_dimension,
    temporal_standard_indices,
)
from .coordinates import WordSelection
from .offset_patterns import symmetric_run_channel_tuples
from .word_program import _compile_word_program, evaluate_word_program


@dataclass(frozen=True, eq=False, init=False)
class _ChronologicalLevel:
    """Per-level coordinate metadata, independent of its execution program."""

    reconstruction_matrix: Tensor | None
    temporal_indices: Tensor
    channel_indices: Tensor
    descent_mask: Tensor
    window: int
    code_dimension: int
    channels: int
    depth: int
    full_feature_count: int
    selection: WordSelection | None
    words: tuple[tuple[int, ...], ...]
    _flat_indices: Tensor | None

    def __init__(
        self,
        reconstruction_matrix: Tensor | None,
        flat_indices: Tensor | None,
        temporal_indices: Tensor,
        channel_indices: Tensor,
        descent_mask: Tensor,
        window: int,
        code_dimension: int,
        channels: int,
        depth: int,
        full_feature_count: int,
        selection: WordSelection | None = None,
    ) -> None:
        for name, value in (
            ("reconstruction_matrix", reconstruction_matrix),
            ("_flat_indices", flat_indices),
            ("temporal_indices", temporal_indices),
            ("channel_indices", channel_indices),
            ("descent_mask", descent_mask),
            ("window", window),
            ("code_dimension", code_dimension),
            ("channels", channels),
            ("depth", depth),
            ("full_feature_count", full_feature_count),
            ("selection", selection),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()
        object.__setattr__(
            self, "words", tuple(tuple(row) for row in self.letters.cpu().tolist())
        )

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be positive.")
        if self.code_dimension < 1 or self.code_dimension > self.window:
            raise ValueError("code_dimension must satisfy 1 <= P <= window.")
        if self.channels < 1:
            raise ValueError("channels must be positive.")
        if self.depth < 1:
            raise ValueError("depth must be positive.")
        if self.selection is not None and not isinstance(self.selection, WordSelection):
            raise TypeError("selection must be a WordSelection.")
        count = len(self.temporal_indices)
        if self.temporal_indices.shape != (count, self.depth):
            raise ValueError("temporal_indices has an invalid shape.")
        if self.channel_indices.shape != (count, self.depth):
            raise ValueError("channel_indices has an invalid shape.")
        if self.descent_mask.shape != (count, self.depth - 1):
            raise ValueError("descent_mask has an invalid shape.")
        indices = (self.temporal_indices, self.channel_indices)
        if self._flat_indices is not None:
            if self._flat_indices.shape != (count,):
                raise ValueError("flat_indices has an invalid shape.")
            indices += (self._flat_indices,)
        if any(value.dtype != torch.long for value in indices):
            raise ValueError("compiled word indices must have dtype torch.long.")
        if self.descent_mask.dtype != torch.bool:
            raise ValueError("descent_mask must have dtype torch.bool.")
        if len({value.device for value in (*indices, self.descent_mask)}) != 1:
            raise ValueError("compiled word metadata must share one device.")
        if (
            self.reconstruction_matrix is not None
            and self.reconstruction_matrix.shape != (self.ambient_dimension, count)
        ):
            raise ValueError("reconstruction_matrix has an invalid shape.")

    @property
    def alphabet_width(self) -> int:
        return self.channels * self.code_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.alphabet_width**self.depth

    @property
    def feature_count(self) -> int:
        return len(self.temporal_indices)

    @property
    def is_lossless(self) -> bool:
        return self.feature_count == self.full_feature_count

    @property
    def letters(self) -> Tensor:
        """Channel-major letters, without encoding an entire word as an int64."""
        return self.channel_indices * self.code_dimension + self.temporal_indices

    @property
    def flat_indices(self) -> Tensor:
        """Dense tensor indices, checked only when a dense consumer needs them."""
        if self._flat_indices is None:
            object.__setattr__(
                self,
                "_flat_indices",
                _flat_word_indices(self.words, self.alphabet_width, self.device),
            )
        return self._flat_indices

    @property
    def temporal_degree(self) -> Tensor:
        return self.temporal_indices.sum(dim=1)

    @property
    def descent_count(self) -> Tensor:
        return self.descent_mask.sum(dim=1)

    @property
    def complexity(self) -> Tensor:
        return self.temporal_degree + self.descent_count

    @property
    def device(self) -> torch.device:
        return self.temporal_indices.device


def _flat_word_indices(
    words: tuple[tuple[int, ...], ...],
    width: int,
    device: torch.device | str | None = None,
) -> Tensor:
    rows = []
    for word in words:
        index = 0
        for letter in word:
            index = index * width + letter
        if index > torch.iinfo(torch.long).max:
            raise OverflowError(
                "Dense tensor word indices exceed int64; use compact word coordinates."
            )
        rows.append(index)
    return torch.tensor(rows, device=device, dtype=torch.long)


def _admissible_words(
    window: int,
    code_dimension: int,
    channels: int,
    depth: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate all retained levels by extending only admissible prefixes."""
    counts = generic_scalar_dimensions(window, code_dimension, channels, depth)
    frontier = [((), 0, -1)]
    levels = []
    for expected in counts:
        next_frontier = []
        for prefix, cost, previous_channel in frontier:
            for channel in range(channels):
                new_cost = cost + (previous_channel > channel)
                for temporal in range(min(code_dimension, window - new_cost)):
                    word = (*prefix, channel * code_dimension + temporal)
                    next_frontier.append((word, new_cost + temporal, channel))
        words = tuple(row[0] for row in next_frontier)
        if len(words) != expected:
            raise RuntimeError("Admissible word count disagrees with the exact count.")
        levels.append(words)
        frontier = next_frontier
    return tuple(levels)


def _admissible_word_indices(
    window: int,
    code_dimension: int,
    channels: int,
    depth: int,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return one retained level in the legacy dense-index representation."""
    words = _admissible_words(window, code_dimension, channels, depth)[-1]
    letters = torch.tensor(words, device=device, dtype=torch.long).reshape(-1, depth)
    return (
        letters % code_dimension,
        letters // code_dimension,
        _flat_word_indices(words, code_dimension * channels, device),
    )


def _composition_from_cuts(depth: int, cuts: tuple[int, ...]) -> tuple[int, ...]:
    boundaries = (0, *cuts, depth)
    return tuple(
        boundaries[index + 1] - boundaries[index]
        for index in range(len(boundaries) - 1)
    )


def _descent_orbit_section(
    in_channels: int,
    depth: int,
    cuts: tuple[int, ...],
) -> Tensor:
    """Return the canonical exact-descent orbit sums for one cut set."""
    composition = _composition_from_cuts(depth, cuts)
    representatives = tuple(
        word
        for word in symmetric_run_channel_tuples(in_channels, composition)
        if tuple(
            position
            for position in range(1, depth)
            if word[position - 1] > word[position]
        )
        == cuts
    )
    expected_count = ribbon_dimension(in_channels, depth, cuts)
    if len(representatives) != expected_count:
        raise RuntimeError("Exact-descent word count disagrees with ribbon dimension.")
    representative_indices = {
        representative: index for index, representative in enumerate(representatives)
    }
    matrix = torch.zeros(
        in_channels**depth,
        len(representatives),
        dtype=torch.float64,
    )
    boundaries = tuple(itertools.accumulate((0, *composition)))
    for row, word in enumerate(itertools.product(range(in_channels), repeat=depth)):
        representative = tuple(
            channel
            for start, stop in zip(boundaries, boundaries[1:])
            for channel in sorted(word[start:stop])
        )
        column = representative_indices.get(representative)
        if column is not None:
            matrix[row, column] = 1.0
    return matrix


def _tensor_product_rows(code: Tensor, offsets: Tensor) -> Tensor:
    result = code[int(offsets[0])]
    for offset in offsets[1:]:
        result = torch.kron(result, code[int(offset)])
    return result


def _combine_temporal_channel(
    temporal: Tensor,
    channel: Tensor,
    code_dim: int,
    in_channels: int,
    depth: int,
) -> Tensor:
    outer = channel.reshape(*([in_channels] * depth), *([1] * depth))
    outer = outer * temporal.reshape(*([1] * depth), *([code_dim] * depth))
    interleaved_axes = tuple(
        axis for position in range(depth) for axis in (position, depth + position)
    )
    return outer.permute(interleaved_axes).reshape(-1)


def _canonical_value_basis(
    code: Tensor,
    in_channels: int,
    depth: int,
) -> Tensor:
    """Build the canonical chronological standard basis of the value span."""
    kernel_size, code_dim = map(int, code.shape)
    columns = []
    for cut_count in range(depth):
        for cuts in itertools.combinations(range(1, depth), cut_count):
            channel_section = _descent_orbit_section(
                in_channels,
                depth,
                cuts,
            )
            _, offset_rows = temporal_standard_indices(
                kernel_size,
                code_dim,
                depth,
                cuts,
            )
            for offsets in offset_rows:
                temporal = _tensor_product_rows(code, offsets)
                for channel in channel_section.T:
                    columns.append(
                        _combine_temporal_channel(
                            temporal,
                            channel,
                            code_dim,
                            in_channels,
                            depth,
                        )
                    )

    expected_count = generic_scalar_dimension(
        kernel_size,
        code_dim,
        in_channels,
        depth,
    )
    if len(columns) != expected_count:
        raise RuntimeError("Reference basis size disagrees with the exact count.")
    return torch.stack(columns, dim=1)


def _normalize_descent_sets(
    descent_sets: Iterable[Iterable[int]] | None,
    depth: int,
) -> set[tuple[int, ...]] | None:
    if descent_sets is None:
        return None
    normalized = set()
    for descents in descent_sets:
        descent_tuple = tuple(sorted(int(position) for position in descents))
        if len(set(descent_tuple)) != len(descent_tuple):
            raise ValueError("descent sets must not contain duplicates.")
        if any(position < 1 or position >= depth for position in descent_tuple):
            raise ValueError(f"descent positions must lie in [1, {depth - 1}].")
        normalized.add(descent_tuple)
    return normalized


def _retained_word_mask(
    temporal_indices: Tensor,
    descent_mask: Tensor,
    descent_sets: set[tuple[int, ...]] | None,
    max_temporal_degree: int | None,
    max_channel_descents: int | None,
    max_complexity: int | None,
) -> Tensor:
    temporal_degree = temporal_indices.sum(dim=1)
    descent_count = descent_mask.sum(dim=1)
    keep = torch.ones(len(temporal_indices), dtype=torch.bool)
    if max_temporal_degree is not None:
        keep &= temporal_degree <= max_temporal_degree
    if max_channel_descents is not None:
        keep &= descent_count <= max_channel_descents
    if max_complexity is not None:
        keep &= temporal_degree + descent_count <= max_complexity
    if descent_sets is not None:
        keep &= torch.tensor(
            [
                tuple(position + 1 for position, present in enumerate(row) if present)
                in descent_sets
                for row in descent_mask
            ],
            dtype=torch.bool,
        )
    return keep


def _build_chronological_level(
    code: Tensor,
    channels: int,
    depth: int,
    *,
    selection: WordSelection | None = None,
) -> _ChronologicalLevel:
    """Build canonical chronological coordinates and their dense decoder.

    The unpruned basis is canonical for raw Pascal/Newton coordinates. The same
    word indices also define a basis for codes in the generic admissible-word
    chart, including the repository's ordered QR-Pascal code. QR coordinates
    retain the degree filtration but are not numerically equal to raw Pascal
    coordinates.

    Parameters
    ----------
    code : Tensor
        Fixed offset code with shape ``(window, code_dimension)``.
    channels : int
        Signal-channel dimension.
    depth : int
        Homogeneous signature degree.
    selection : WordSelection, optional
        Explicit coordinate-selection specification.

    Returns
    -------
    _ChronologicalLevel
        Direct word indices, retained labels, and a dense reconstruction map.
    """
    if code.ndim != 2:
        raise ValueError("code must have shape (window, code_dimension).")
    if not code.dtype.is_floating_point:
        raise ValueError("code must have a floating-point dtype.")
    if channels < 1:
        raise ValueError("channels must be positive.")
    if depth < 1:
        raise ValueError("depth must be positive.")
    if selection is not None and not isinstance(selection, WordSelection):
        raise TypeError("selection must be a WordSelection.")
    effective_selection = selection or WordSelection()
    window, code_dimension = map(int, code.shape)
    code = code.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(code).all():
        raise ValueError("code must contain only finite values.")

    temporal_indices, channel_indices, flat_indices = _admissible_word_indices(
        window,
        code_dimension,
        channels,
        depth,
    )
    descent_mask = channel_indices[:, :-1] > channel_indices[:, 1:]
    allowed = _normalize_descent_sets(effective_selection.descent_sets, depth)
    retained_mask = _retained_word_mask(
        temporal_indices,
        descent_mask,
        allowed,
        effective_selection.max_temporal_degree,
        effective_selection.max_channel_descents,
        effective_selection.max_complexity,
    )
    retained = retained_mask.nonzero(as_tuple=False).flatten()

    value_basis = _canonical_value_basis(code, channels, depth)
    coordinate_matrix = value_basis.index_select(0, flat_indices)
    full_feature_count = len(flat_indices)
    if int(torch.linalg.matrix_rank(coordinate_matrix)) != full_feature_count:
        raise ValueError(
            "The offset code is outside the admissible projected-word chart."
        )
    reconstruction = value_basis @ torch.linalg.solve(
        coordinate_matrix,
        torch.eye(full_feature_count, dtype=torch.float64),
    )

    return _ChronologicalLevel(
        reconstruction_matrix=reconstruction.index_select(1, retained),
        flat_indices=flat_indices.index_select(0, retained),
        temporal_indices=temporal_indices.index_select(0, retained),
        channel_indices=channel_indices.index_select(0, retained),
        descent_mask=descent_mask.index_select(0, retained),
        window=window,
        code_dimension=code_dimension,
        channels=channels,
        depth=depth,
        full_feature_count=full_feature_count,
        selection=effective_selection,
    )


def _compile_chronological_levels(
    window: int,
    code_dimension: int,
    channels: int,
    depths: tuple[int, ...],
    *,
    selection: WordSelection | None = None,
) -> tuple[_ChronologicalLevel, ...]:
    """Compile requested levels using one admissible walk and no dense indices."""
    if selection is not None and not isinstance(selection, WordSelection):
        raise TypeError("selection must be a WordSelection.")
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("depths must be nonempty and positive.")
    all_words = _admissible_words(window, code_dimension, channels, max(depths))
    levels = []
    for depth in depths:
        effective_selection = (selection or WordSelection())._for_depth(depth)
        words = all_words[depth - 1]
        letters = torch.tensor(words, dtype=torch.long).reshape(-1, depth)
        temporal, channel = letters % code_dimension, letters // code_dimension
        descents = channel[:, :-1] > channel[:, 1:]
        retained = (
            _retained_word_mask(
                temporal,
                descents,
                _normalize_descent_sets(effective_selection.descent_sets, depth),
                effective_selection.max_temporal_degree,
                effective_selection.max_channel_descents,
                effective_selection.max_complexity,
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        levels.append(
            _ChronologicalLevel(
                reconstruction_matrix=None,
                flat_indices=None,
                temporal_indices=temporal.index_select(0, retained),
                channel_indices=channel.index_select(0, retained),
                descent_mask=descents.index_select(0, retained),
                window=window,
                code_dimension=code_dimension,
                channels=channels,
                depth=depth,
                full_feature_count=len(words),
                selection=effective_selection,
            )
        )
    return tuple(levels)


def _compile_chronological_level(
    window: int,
    code_dimension: int,
    channels: int,
    depth: int,
    *,
    selection: WordSelection | None = None,
) -> _ChronologicalLevel:
    """Compile one retained level without constructing a dense decoder."""
    return _compile_chronological_levels(
        window, code_dimension, channels, (depth,), selection=selection
    )[0]


def _evaluate_chronological_words(
    dxw: Tensor,
    code: Tensor,
    level: _ChronologicalLevel,
) -> Tensor:
    """Compatibility entry point using the shared word-program evaluator."""
    if dxw.ndim < 2 or tuple(dxw.shape[-2:]) != (level.window, level.channels):
        raise ValueError("Increment windows disagree with the compiled level.")
    program = _compile_word_program(level.words, level.code_dimension)
    return evaluate_word_program(dxw, code, program)


def _reconstruct_chronological_level(
    coordinates: Tensor,
    basis: _ChronologicalLevel,
) -> Tensor:
    """Decode retained words into the dense tensor-level layout.

    For a pruned basis this reconstructs the component obtained by setting
    omitted admissible coordinates to zero.
    """
    if coordinates.ndim < 1 or coordinates.shape[-1] != basis.feature_count:
        raise ValueError(f"Expected coordinates.shape[-1] == {basis.feature_count}.")
    if basis.reconstruction_matrix is None:
        raise ValueError("This chronological level has no dense decoder.")
    matrix = basis.reconstruction_matrix.to(
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    flat = coordinates @ matrix.T
    return flat.reshape(
        *coordinates.shape[:-1],
        *((basis.alphabet_width,) * basis.depth),
    )
