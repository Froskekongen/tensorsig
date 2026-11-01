"""Compiled coordinate plans for local signatures and logsignatures."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from numbers import Real
from typing import Literal

import torch
from torch import Tensor

from tensorsig.algebra.lyndon import lyndon_words

from .codes import AlphabetMap, OffsetCode
from .chronological_words import (
    _ChronologicalLevel,
    _build_chronological_level,
    _canonical_value_basis,
    _compile_chronological_levels,
)
from .pascal_decoder import _PascalDecoder
from .word_program import _WordProgram, _compile_word_program, evaluate_word_program
from .coordinates import ChronologicalFiltration, WordSelection
from .increment_signature import signature_levels_from_increment_windows
from .primitive_chart import (
    _PrimitiveLevel,
    _build_signature_source_basis,
    _eulerian_project_columns,
    _flat_word_index,
    _ordered_independent_rows,
    _rank_revealing_subspaces,
    _resolve_rank_tolerance,
    compile_primitive_chart,
)
from .quotients import LieQuotient, QuotientBasisElement
from ._lie_quotients.compiler import (
    LieQuotientPlan,
    compile_lie_quotient,
    quotient_log_signature_from_increment_windows,
)
from ._lie_quotients.rules import OffsetAlphabet, ProjectedAlphabet
from ._lie_quotients.trace_compiler import TraceBasisElement
from .scaled_monomials import RawSignatureScale, ScaledMonomialDictionary
from .windows import validate_increment_windows


def _depths(max_depth: int, include: tuple[int, ...] | None) -> tuple[int, ...]:
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer.")
    depths = tuple(range(1, max_depth + 1)) if include is None else tuple(include)
    if any(not isinstance(depth, int) or isinstance(depth, bool) for depth in depths):
        raise ValueError("depths must contain only integers.")
    if not depths or tuple(sorted(set(depths))) != depths:
        raise ValueError(
            "depths must be a non-empty increasing tuple without duplicates."
        )
    if depths[0] < 1 or depths[-1] > max_depth:
        raise ValueError("depths must lie in 1, ..., max_depth.")
    return depths


def _slices(dimensions: tuple[int, ...]) -> tuple[slice, ...]:
    result = []
    start = 0
    for dimension in dimensions:
        result.append(slice(start, start + dimension))
        start += dimension
    return tuple(result)


def _validate_chunk_size(chunk_size: int | None) -> int | None:
    if chunk_size is not None and (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer or None.")
    return chunk_size


def _validate_rank_tolerance(rank_tolerance: float | None) -> float | None:
    if rank_tolerance is None:
        return None
    if (
        not isinstance(rank_tolerance, Real)
        or isinstance(rank_tolerance, bool)
        or not math.isfinite(rank_tolerance)
        or rank_tolerance < 0
    ):
        raise ValueError("rank_tolerance must be a finite non-negative real or None.")
    return float(rank_tolerance)


class _CompiledPlan:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            f"{type(self).__name__} objects are compiled; use LocalRealization "
            "or compile_polynomial."
        )


def _compiled_plan(plan_type: type[_CompiledPlan], **attributes: object):
    plan = object.__new__(plan_type)
    for name, value in attributes.items():
        object.__setattr__(plan, name, value)
    return plan


@dataclass(frozen=True, eq=False, init=False)
class SignaturePlan(_CompiledPlan):
    """A code-bound local signature coordinate plan."""

    mapping: OffsetCode | AlphabetMap
    channels: int
    max_depth: int
    depths: tuple[int, ...]
    coordinates: Literal["chronological", "dense"]
    selection: WordSelection | None
    certification: Literal["structural", "numerical", "dense"]
    rank_tolerance: float | None
    chart_margin: float | None
    _levels: tuple[_ChronologicalLevel, ...] = field(repr=False)
    _decoder_cache: dict[int, _ChronologicalLevel | _PascalDecoder] = field(
        init=False,
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _program_cache: dict[bool, _WordProgram] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )

    @property
    def window(self) -> int:
        return self.mapping.window

    @property
    def output_dimension(self) -> int:
        if isinstance(self.mapping, OffsetCode):
            return self.channels * self.mapping.dimension
        return self.mapping.output_dimension

    @property
    def level_dimensions(self) -> tuple[int, ...]:
        if self.coordinates == "dense":
            return tuple(self.output_dimension**depth for depth in self.depths)
        return tuple(level.feature_count for level in self._levels)

    @property
    def level_slices(self) -> tuple[slice, ...]:
        return _slices(self.level_dimensions)

    @property
    def feature_count(self) -> int:
        return sum(self.level_dimensions)

    @property
    def is_lossless(self) -> bool:
        return self.coordinates == "dense" or all(
            level.is_lossless for level in self._levels
        )

    @property
    def matrix(self) -> Tensor:
        return self.mapping.matrix

    def _matrix_tensor(self) -> Tensor:
        return self.mapping._runtime_matrix()

    @property
    def word_indices(self) -> tuple[Tensor, ...]:
        """Return zero-based channel-major words for chronological levels."""
        self._require_chronological()
        return tuple(level.letters.clone() for level in self._levels)

    @property
    def temporal_indices(self) -> tuple[Tensor, ...]:
        """Return zero-based temporal letters for chronological levels."""
        self._require_chronological()
        return tuple(level.temporal_indices.clone() for level in self._levels)

    @property
    def channel_indices(self) -> tuple[Tensor, ...]:
        """Return zero-based channel letters for chronological levels."""
        self._require_chronological()
        return tuple(level.channel_indices.clone() for level in self._levels)

    @property
    def state_count(self) -> int | None:
        """Unique prefix states, including the empty word, used by evaluation."""
        if self.coordinates != "chronological":
            return None
        return self._word_program.state_count

    def _program(self, *, rolling: bool) -> _WordProgram:
        self._require_chronological()
        cached = self._program_cache.get(rolling)
        if cached is None:
            assert isinstance(self.mapping, OffsetCode)
            cached = _compile_word_program(
                tuple(word for level in self._levels for word in level.words),
                self.mapping.dimension,
                rolling=rolling,
            )
            self._program_cache[rolling] = cached
        return cached

    @property
    def _word_program(self) -> _WordProgram:
        return self._program(rolling=False)

    def _rolling_program(self) -> _WordProgram:
        return self._program(rolling=True)

    def _require_chronological(self) -> None:
        if self.coordinates != "chronological":
            raise ValueError("Dense plans do not have chronological word labels.")

    def _coordinates_from_levels(self, levels: tuple[Tensor, ...]) -> Tensor:
        if self.coordinates == "dense":
            return torch.cat(
                [levels[depth - 1].flatten(start_dim=2) for depth in self.depths],
                dim=-1,
            )
        return torch.cat(
            [
                levels[level.depth - 1]
                .flatten(start_dim=2)
                .index_select(-1, level.flat_indices.to(levels[0].device))
                for level in self._levels
            ],
            dim=-1,
        )

    def _evaluate(self, windows: Tensor, matrix: Tensor) -> Tensor:
        validate_increment_windows(windows, size=self.window, channels=self.channels)
        matrix = matrix.to(device=windows.device, dtype=windows.dtype)
        if self.coordinates == "dense":
            if isinstance(self.mapping, OffsetCode):
                lifted = torch.einsum("bnrc,re->bnrce", windows, matrix).flatten(-2)
            else:
                lifted = torch.einsum("bnrc,rcd->bnrd", windows, matrix)
            return self._coordinates_from_levels(
                signature_levels_from_increment_windows(
                    lifted,
                    max_depth=max(self.depths),
                )
            )
        return evaluate_word_program(windows, matrix, self._word_program)

    def evaluate(self, windows: Tensor) -> Tensor:
        """Evaluate the compiled coordinates on increment windows."""
        return self._evaluate(windows, self._matrix_tensor())

    def contract(self, windows: Tensor, weight: Tensor) -> Tensor:
        """Contract features with ``(outputs, features)`` weights."""
        if weight.ndim != 2 or weight.shape[1] != self.feature_count:
            raise ValueError(f"weight must have shape (outputs, {self.feature_count}).")
        features = self.evaluate(windows)
        return features @ weight.to(device=features.device, dtype=features.dtype).T

    def reconstruct(self, features: Tensor) -> tuple[Tensor, ...]:
        """Reconstruct dense coded tensor levels from a lossless plan."""
        self._validate_features(features)
        if self.coordinates == "dense":
            return tuple(
                features[..., feature_slice].reshape(
                    *features.shape[:-1],
                    *([self.output_dimension] * depth),
                )
                for depth, feature_slice in zip(
                    self.depths,
                    self.level_slices,
                    strict=True,
                )
            )
        if not self.is_lossless:
            raise ValueError("Only lossless chronological plans have a decoder.")
        levels = []
        for depth, feature_slice, level in zip(
            self.depths,
            self.level_slices,
            self._levels,
            strict=True,
        ):
            if (
                self.certification == "structural"
                and level.feature_count == level.ambient_dimension
            ):
                values = features[..., feature_slice]
            else:
                if level.ambient_dimension > torch.iinfo(torch.long).max:
                    raise OverflowError(
                        "Dense reconstruction exceeds int64 indexing; use reconstruct_words instead."
                    )
                indices = torch.arange(level.ambient_dimension, dtype=torch.long)
                words = torch.stack(
                    [
                        (indices // self.output_dimension**power)
                        % self.output_dimension
                        for power in reversed(range(depth))
                    ],
                    dim=1,
                )
                values = self.reconstruct_words(features, words)
            levels.append(
                values.reshape(*features.shape[:-1], *([self.output_dimension] * depth))
            )
        return tuple(levels)

    def _validate_features(self, features: Tensor) -> None:
        if not isinstance(features, Tensor) or not features.dtype.is_floating_point:
            raise ValueError("features must have a floating-point dtype.")
        if features.ndim == 0 or features.shape[-1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} trailing features.")

    def reconstruct_words(self, features: Tensor, words: Tensor) -> Tensor:
        """Recover requested ordinary signature coordinates at one depth.

        ``words`` has shape ``(Q, m)`` and uses the zero-based channel-major
        letters of ``word_indices``. Results have shape ``(..., Q)`` in request
        order, including duplicates. Retained words can be looked up on lossy
        plans; decoding omitted words requires a complete chart at depth ``m``.
        """
        self._validate_features(features)
        if (
            not isinstance(words, Tensor)
            or words.ndim != 2
            or words.dtype
            not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
        ):
            raise ValueError("words must be a two-dimensional integer tensor.")
        depth = words.shape[1]
        if depth not in self.depths:
            raise ValueError("The requested word depth must be present in the plan.")
        words = words.detach().to(device="cpu", dtype=torch.long)
        if torch.any(words < 0) or torch.any(words >= self.output_dimension):
            raise ValueError("Word letters must lie in the plan's alphabet.")
        if not len(words):
            return features[..., :0]
        position = self.depths.index(depth)
        source = features[..., self.level_slices[position]]
        if self.coordinates == "chronological":
            level = self._levels[position]
            lookup = {word: index for index, word in enumerate(level.words)}
            targets = tuple(map(tuple, words.tolist()))
            if all(word in lookup for word in targets):
                indices = torch.tensor(
                    [lookup[word] for word in targets], device=features.device
                )
                return source.index_select(-1, indices)
            if not level.is_lossless:
                raise ValueError(
                    "Decoding omitted words requires a complete chart at the requested depth."
                )
            if self.certification == "structural":
                decoder = self._decoder_cache.get(depth)
                if decoder is None:
                    assert isinstance(self.mapping, OffsetCode)
                    decoder = _PascalDecoder(self.mapping, level.letters)
                    self._decoder_cache[depth] = decoder
                assert isinstance(decoder, _PascalDecoder)
                return decoder.decode(source, words)
        if self.output_dimension**depth > torch.iinfo(torch.long).max:
            raise OverflowError("Ambient word indices exceed int64 storage.")
        indices = torch.zeros(len(words), dtype=torch.long)
        for letter in words.T:
            indices = indices * self.output_dimension + letter
        if self.coordinates == "dense":
            return source.index_select(-1, indices.to(device=features.device))
        basis = self._levels[position]
        if basis.reconstruction_matrix is None:
            cached = self._decoder_cache.get(depth)
            if cached is None:
                assert isinstance(self.mapping, OffsetCode)
                cached = _build_chronological_level(
                    self.mapping._runtime_matrix(), channels=self.channels, depth=depth
                )
                self._decoder_cache[depth] = cached
            assert isinstance(cached, _ChronologicalLevel)
            basis = cached
        assert basis.reconstruction_matrix is not None
        decoder_rows = basis.reconstruction_matrix.index_select(
            0, indices.to(basis.reconstruction_matrix.device)
        )
        return source @ decoder_rows.to(device=features.device, dtype=features.dtype).T

    def summary(self) -> str:
        result = (
            f"signature[{self.coordinates}/{self.certification}]: "
            f"levels={self.level_dimensions}, features={self.feature_count}, "
            f"code={getattr(self.mapping, 'family', 'projection')}"
        )
        if self.chart_margin is not None:
            result += f", margin={self.chart_margin:.2e}"
        return result


def _compile_signature(
    realization: "LocalRealization",
    *,
    coordinates: Literal["chronological", "dense"] | None,
    selection: WordSelection | None,
    depths: tuple[int, ...] | None,
    rank_tolerance: float | None,
) -> SignaturePlan:
    mapping = realization.mapping
    selected_depths = _depths(realization.max_depth, depths)
    rank_tolerance = _validate_rank_tolerance(rank_tolerance)
    if coordinates is None:
        coordinates = "chronological" if isinstance(mapping, OffsetCode) else "dense"
    if coordinates not in {"chronological", "dense"}:
        raise ValueError("coordinates must be 'chronological' or 'dense'.")
    if coordinates == "chronological" and not isinstance(mapping, OffsetCode):
        raise ValueError("Chronological coordinates require an OffsetCode.")
    if selection is not None and coordinates != "chronological":
        raise ValueError("WordSelection applies only to chronological coordinates.")
    if selection is not None and not isinstance(selection, WordSelection):
        raise TypeError("selection must be a WordSelection or None.")
    if selection is not None and (
        not isinstance(mapping, OffsetCode) or not mapping.is_graded
    ):
        raise ValueError("WordSelection requires a certified graded OffsetCode.")

    if coordinates == "dense":
        if rank_tolerance is not None:
            raise ValueError("rank_tolerance applies only to chronological charts.")
        return _compiled_plan(
            SignaturePlan,
            mapping=mapping,
            channels=realization.channels,
            max_depth=max(selected_depths),
            depths=selected_depths,
            coordinates="dense",
            selection=None,
            certification="dense",
            rank_tolerance=None,
            chart_margin=None,
            _levels=(),
            _decoder_cache={},
            _program_cache={},
        )

    assert isinstance(mapping, OffsetCode)
    if mapping._runtime_matrix().requires_grad:
        raise ValueError(
            "Trainable codes require dense coordinates; a chronological chart "
            "cannot silently change during optimization."
        )
    tolerances = []
    margins = []
    structural = mapping.family == "pascal" and rank_tolerance is None
    levels = (
        list(
            _compile_chronological_levels(
                mapping.window,
                mapping.dimension,
                realization.channels,
                selected_depths,
                selection=selection,
            )
        )
        if structural
        else []
    )
    for depth in () if structural else selected_depths:
        if selection is not None:
            raise ValueError("Selected chronological charts require a Pascal code.")
        code = mapping._runtime_matrix().detach().cpu().double()
        try:
            basis = _build_signature_source_basis(
                code,
                in_channels=realization.channels,
                depth=depth,
                rank_tolerance=rank_tolerance,
            )
        except RuntimeError as error:
            if "expected local primitive rank" not in str(error):
                raise
            raise ValueError(
                "The numerical chronological chart is ambiguous; provide an "
                "explicit rank_tolerance or use coordinates='dense'."
            ) from error
        levels.append(basis)
        value_basis = _canonical_value_basis(code, realization.channels, depth)
        coordinate_matrix = value_basis.index_select(0, basis.flat_indices)
        tolerances.append(_resolve_rank_tolerance(value_basis, rank_tolerance))
        if coordinate_matrix.numel():
            margins.append(float(torch.linalg.svdvals(coordinate_matrix)[-1]))

    return _compiled_plan(
        SignaturePlan,
        mapping=mapping,
        channels=realization.channels,
        max_depth=max(selected_depths),
        depths=selected_depths,
        coordinates="chronological",
        selection=selection,
        certification="structural" if structural else "numerical",
        rank_tolerance=max(tolerances) if tolerances else None,
        chart_margin=min(margins) if margins else None,
        _levels=tuple(levels),
        _decoder_cache={},
        _program_cache={},
    )


@lru_cache(maxsize=128)
def _word_function_matrix_cached(
    matrix_values: tuple[tuple[float, ...], ...],
    channels: int,
    depth: int,
    physical_order: bool,
    rank_tolerance: float | None,
) -> tuple[Tensor, int]:
    """Return ambient Lyndon coordinates as functions of signature coordinates."""
    matrix = torch.tensor(matrix_values, dtype=torch.float64)
    source = _build_signature_source_basis(
        matrix,
        in_channels=channels,
        depth=depth,
        rank_tolerance=rank_tolerance,
    )
    window, dimension = matrix.shape
    width = dimension * channels
    primitive = _eulerian_project_columns(
        source.reconstruction_matrix,
        width=width,
        depth=depth,
    )
    words = tuple(word for word in lyndon_words(width, depth) if len(word) == depth)
    if physical_order:
        indices = []
        for word in words:
            channel_major = []
            for letter in word:
                offset, channel = divmod(letter - 1, channels)
                channel_major.append(channel * window + offset + 1)
            indices.append(_flat_word_index(tuple(channel_major), width))
    else:
        indices = [_flat_word_index(word, width) for word in words]
    rows = torch.tensor(indices, dtype=torch.long)
    return primitive.index_select(0, rows), source.feature_count


def _word_function_matrix(
    code: OffsetCode,
    channels: int,
    depth: int,
    *,
    physical_order: bool,
    rank_tolerance: float | None,
) -> tuple[Tensor, int]:
    matrix_values = tuple(
        tuple(row) for row in code._runtime_matrix().detach().cpu().double().tolist()
    )
    return _word_function_matrix_cached(
        matrix_values,
        channels,
        depth,
        physical_order,
        rank_tolerance,
    )


@dataclass(frozen=True, eq=False, init=False)
class LogSignaturePlan(_CompiledPlan):
    """A minimal code-bound local primitive or quotient-primitive plan."""

    code: OffsetCode
    channels: int
    max_depth: int
    depths: tuple[int, ...]
    filtration: ChronologicalFiltration | None
    quotient: LieQuotient | None
    primitive_dimension: int
    chronological_dimension: int
    rank_tolerance: float
    chart_margin: float | None
    _bases: tuple[_PrimitiveLevel, ...] = field(repr=False)
    _source_levels: tuple[_ChronologicalLevel, ...] = field(repr=False)
    _quotient_plan: LieQuotientPlan | None = field(repr=False)
    _quotient_indices: Tensor = field(repr=False)
    _quotient_decoder: Tensor | None = field(repr=False)
    _represented_quotient_indices: Tensor = field(repr=False)
    _feature_depths: Tensor = field(repr=False)
    _local_quotient_residual: float = field(repr=False)
    _runtime_quotient_cache: dict[
        tuple[torch.device, torch.dtype],
        tuple[Tensor, Tensor | None],
    ] = field(init=False, default_factory=dict, repr=False, compare=False)
    _source_program_cache: dict[str, _WordProgram] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )

    @property
    def window(self) -> int:
        return self.code.window

    @property
    def output_dimension(self) -> int:
        """Width of the coded alphabet before primitive projection."""
        return self.channels * self.code.dimension

    @property
    def matrix(self) -> Tensor:
        return self.code.matrix

    def _matrix_tensor(self) -> Tensor:
        return self.code._runtime_matrix()

    @property
    def feature_count(self) -> int:
        if self._quotient_plan is None:
            return sum(basis.feature_count for basis in self._bases)
        return int(self._quotient_indices.numel())

    @property
    def level_dimensions(self) -> tuple[int, ...]:
        if self._quotient_plan is None:
            return tuple(basis.feature_count for basis in self._bases)
        return tuple(
            int((self._feature_depths == depth).sum()) for depth in self.depths
        )

    @property
    def level_slices(self) -> tuple[slice, ...]:
        return _slices(self.level_dimensions)

    @property
    def quotient_lie_dimension(self) -> int | None:
        if self._quotient_plan is None:
            return None
        return int(self._represented_quotient_indices.numel())

    @property
    def local_quotient_dimension(self) -> int | None:
        return None if self._quotient_plan is None else self.feature_count

    @property
    def local_quotient_residual(self) -> float | None:
        """Return the reconstruction residual of the represented quotient map."""
        return None if self._quotient_plan is None else self._local_quotient_residual

    @property
    def quotient_coordinate_indices(self) -> Tensor:
        return self._quotient_indices.clone()

    @property
    def represented_quotient_indices(self) -> Tensor:
        """Coordinates reconstructed in the selected homogeneous levels."""
        return self._represented_quotient_indices.clone()

    @property
    def quotient_basis(self) -> tuple[QuotientBasisElement, ...] | None:
        """Stable labels aligned with reconstructed quotient coordinates."""
        if self._quotient_plan is None:
            return None
        labels = []
        for index in self._represented_quotient_indices.tolist():
            element = self._quotient_plan.basis[index]
            depth = int(self._quotient_plan.basis_depths[index])
            if self._quotient_plan.coordinate_aligned:
                assert isinstance(element, (int, tuple))
                labels.append(
                    QuotientBasisElement(kind="hall", depth=depth, tree=element)
                )
            else:
                assert isinstance(element, TraceBasisElement)
                labels.append(
                    QuotientBasisElement(
                        kind="trace",
                        depth=depth,
                        word=tuple(letter + 1 for letter in element.pivot),
                    )
                )
        return tuple(labels)

    @property
    def ambient_words(self) -> tuple[tuple[int, ...], ...] | None:
        """Lyndon labels for unquotiented coordinates returned by reconstruction."""
        if self._quotient_plan is not None:
            return None
        return tuple(word for level in self._bases for word in level.ambient_words)

    @property
    def is_lossless(self) -> bool:
        if self._quotient_plan is not None:
            return True
        return all(basis.is_lossless for basis in self._bases)

    def _runtime_quotient_tensors(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor | None]:
        key = (device, dtype)
        cached = self._runtime_quotient_cache.get(key)
        if cached is not None:
            return cached
        bound = (
            self._quotient_indices.to(device=device),
            (
                None
                if self._quotient_decoder is None
                else self._quotient_decoder.to(device=device, dtype=dtype)
            ),
        )
        self._runtime_quotient_cache[key] = bound
        return bound

    def _coordinates_from_signature_levels(
        self,
        levels: tuple[Tensor, ...],
    ) -> Tensor:
        features = []
        for basis in self._bases:
            level = levels[basis.depth - 1].flatten(start_dim=2)
            indices, transform, _ = basis.runtime_tensors(
                level.device,
                level.dtype,
            )
            features.append(level.index_select(-1, indices) @ transform.T)
        return torch.cat(features, dim=-1)

    def _evaluate(
        self,
        windows: Tensor,
        matrix: Tensor,
        *,
        backend: str = "auto",
    ) -> Tensor:
        validate_increment_windows(windows, size=self.window, channels=self.channels)
        matrix = matrix.to(device=windows.device, dtype=windows.dtype)
        if self._quotient_plan is not None:
            if backend not in {"auto", "direct", "pbw", "tensor", "active"}:
                raise ValueError(
                    "Quotient logsignatures use backend='direct', 'pbw', 'tensor', or 'active'."
                )
            method = "direct" if backend == "auto" else backend
            offset_code = None if self.code.retains_physical_offsets else matrix
            full = quotient_log_signature_from_increment_windows(
                windows,
                self._quotient_plan,
                offset_code,
                method=method,
            )
            indices, _ = self._runtime_quotient_tensors(
                windows.device,
                windows.dtype,
            )
            return full.index_select(
                -1,
                indices,
            )
        if backend not in {"auto", "dense", "sparse"}:
            raise ValueError("Local logsignatures use backend='dense' or 'sparse'.")
        if backend in {"auto", "dense"}:
            lifted = torch.einsum("bnrc,re->bnrce", windows, matrix).flatten(-2)
            levels = signature_levels_from_increment_windows(
                lifted,
                max_depth=max(self.depths),
            )
            return self._coordinates_from_signature_levels(levels)

        program = self._source_program_cache.get("source")
        if program is None:
            program = _compile_word_program(
                tuple(word for level in self._source_levels for word in level.words),
                self.code.dimension,
            )
            self._source_program_cache["source"] = program
        sources = evaluate_word_program(windows, matrix, program)
        source_start = 0
        features = []
        for basis, source_level in zip(
            self._bases,
            self._source_levels,
            strict=True,
        ):
            source = sources[
                ..., source_start : source_start + source_level.feature_count
            ]
            source_start += source_level.feature_count
            _, transform, _ = basis.runtime_tensors(
                windows.device,
                windows.dtype,
            )
            features.append(source @ transform.T)
        return torch.cat(features, dim=-1)

    def evaluate(self, windows: Tensor, *, backend: str = "auto") -> Tensor:
        """Evaluate minimal local logsignature coordinates."""
        return self._evaluate(windows, self._matrix_tensor(), backend=backend)

    def contract(
        self,
        windows: Tensor,
        weight: Tensor,
        *,
        backend: str = "auto",
    ) -> Tensor:
        if weight.ndim != 2 or weight.shape[1] != self.feature_count:
            raise ValueError(f"weight must have shape (outputs, {self.feature_count}).")
        features = self.evaluate(windows, backend=backend)
        return features @ weight.to(device=features.device, dtype=features.dtype).T

    def reconstruct(self, features: Tensor) -> Tensor:
        """Reconstruct represented ambient or quotient coordinates."""
        if not features.dtype.is_floating_point:
            raise ValueError("features must have a floating-point dtype.")
        if features.shape[-1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} trailing features.")
        if self._quotient_plan is not None:
            assert self._quotient_decoder is not None
            _, decoder = self._runtime_quotient_tensors(
                features.device,
                features.dtype,
            )
            assert decoder is not None
            return features @ decoder.T
        if not self.is_lossless:
            raise ValueError("A filtered primitive plan is not reconstructive.")
        blocks = []
        for basis, feature_slice in zip(
            self._bases,
            self.level_slices,
            strict=True,
        ):
            assert basis.ambient_decoder is not None
            _, _, decoder = basis.runtime_tensors(
                features.device,
                features.dtype,
            )
            assert decoder is not None
            blocks.append(features[..., feature_slice] @ decoder.T)
        return torch.cat(blocks, dim=-1)

    def summary(self) -> str:
        if self._quotient_plan is None:
            return (
                f"logsignature[numerical]: chronological={self.chronological_dimension}, "
                f"primitive={self.primitive_dimension}, features={self.feature_count}, "
                f"levels={self.level_dimensions}"
            )
        return (
            f"quotient logsignature[{self._quotient_plan.compiler}]: "
            f"lie={self.quotient_lie_dimension}, local={self.feature_count}, "
            f"levels={self.level_dimensions}, residual={self._local_quotient_residual:.2e}"
        )


def _compile_unquotiented_log(
    realization: "LocalRealization",
    depths: tuple[int, ...],
    filtration: ChronologicalFiltration | None,
    rank_tolerance: float | None,
) -> LogSignaturePlan:
    assert isinstance(realization.mapping, OffsetCode)
    code = realization.mapping
    if code._runtime_matrix().requires_grad:
        raise ValueError("Minimal logsignature charts require a fixed OffsetCode.")
    if filtration is not None and not code.is_graded:
        raise ValueError("ChronologicalFiltration requires a graded OffsetCode.")
    bases = tuple(
        compile_primitive_chart(
            code._runtime_matrix(),
            realization.channels,
            depth,
            max_degree=(None if filtration is None else filtration.max_degree),
            rank_tolerance=rank_tolerance,
        )
        for depth in depths
    )
    margins = [basis.chart_margin for basis in bases if basis.chart_margin is not None]
    return _compiled_plan(
        LogSignaturePlan,
        code=code,
        channels=realization.channels,
        max_depth=max(depths),
        depths=depths,
        filtration=filtration,
        quotient=None,
        primitive_dimension=sum(basis.primitive_dimension for basis in bases),
        chronological_dimension=sum(
            basis.signature_basis.full_feature_count for basis in bases
        ),
        rank_tolerance=max((basis.rank_tolerance for basis in bases), default=0.0),
        chart_margin=min(margins) if margins else None,
        _bases=bases,
        _source_levels=tuple(basis.signature_basis for basis in bases),
        _quotient_plan=None,
        _quotient_indices=torch.empty(0, dtype=torch.long),
        _quotient_decoder=None,
        _represented_quotient_indices=torch.empty(0, dtype=torch.long),
        _feature_depths=torch.empty(0, dtype=torch.long),
        _local_quotient_residual=0.0,
        _runtime_quotient_cache={},
        _source_program_cache={},
    )


def _compile_quotiented_log(
    realization: "LocalRealization",
    depths: tuple[int, ...],
    quotient: LieQuotient,
    rank_tolerance: float | None,
) -> LogSignaturePlan:
    assert isinstance(realization.mapping, OffsetCode)
    code = realization.mapping
    if code._runtime_matrix().requires_grad:
        raise ValueError("Minimal quotient charts require a fixed OffsetCode.")
    if code.retains_physical_offsets:
        alphabet = OffsetAlphabet(code.window, realization.channels)
    else:
        alphabet = ProjectedAlphabet(
            code.dimension,
            realization.channels,
            code.temporal_degrees,
        )
    lie_plan = compile_lie_quotient(quotient, alphabet, max(depths))
    ambient_words = lie_plan._word_bridge.ambient_words
    ambient_positions = {word: index for index, word in enumerate(ambient_words)}
    represented_rows = torch.tensor(
        [
            index
            for index, depth in enumerate(lie_plan.basis_depths.tolist())
            if depth in depths
        ],
        dtype=torch.long,
    )

    blocks = []
    chronological_dimension = 0
    for depth in depths:
        block, source_dimension = _word_function_matrix(
            code,
            realization.channels,
            depth,
            physical_order=code.retains_physical_offsets,
            rank_tolerance=rank_tolerance,
        )
        words = tuple(
            word for word in lyndon_words(alphabet.width, depth) if len(word) == depth
        )
        rows = torch.tensor(
            [ambient_positions[word] for word in words], dtype=torch.long
        )
        blocks.append((rows, block))
        chronological_dimension += source_dimension

    source_columns = sum(block.shape[1] for _, block in blocks)
    ambient_functions = torch.zeros(
        len(ambient_words),
        source_columns,
        dtype=torch.float64,
    )
    column = 0
    for rows, block in blocks:
        ambient_functions[rows, column : column + block.shape[1]] = block
        column += block.shape[1]

    quotient_matrix = lie_plan._word_bridge.quotient_matrix
    if quotient_matrix.layout != torch.strided:
        quotient_matrix = quotient_matrix.to_dense()
    represented_map = (
        quotient_matrix.index_select(0, represented_rows) @ ambient_functions
    )
    primitive_subspace, _, _ = _rank_revealing_subspaces(
        ambient_functions,
        rank_tolerance,
    )
    local_subspace, _, tolerance = _rank_revealing_subspaces(
        represented_map,
        rank_tolerance,
    )
    primitive_dimension = primitive_subspace.shape[1]
    local_rank = local_subspace.shape[1]
    selected_local = _ordered_independent_rows(
        local_subspace,
        range(represented_map.shape[0]),
    )
    selected_local_tensor = torch.tensor(selected_local, dtype=torch.long)
    selected_global = represented_rows.index_select(0, selected_local_tensor)
    selected_functions = represented_map.index_select(0, selected_local_tensor)
    if local_rank:
        decoder = torch.linalg.lstsq(
            selected_functions.T,
            represented_map.T,
        ).solution.T
    else:
        decoder = torch.empty(represented_map.shape[0], 0, dtype=torch.float64)
    residual = float((decoder @ selected_functions - represented_map).norm().item())

    return _compiled_plan(
        LogSignaturePlan,
        code=code,
        channels=realization.channels,
        max_depth=max(depths),
        depths=depths,
        filtration=None,
        quotient=quotient,
        primitive_dimension=primitive_dimension,
        chronological_dimension=chronological_dimension,
        rank_tolerance=tolerance,
        chart_margin=(
            float(torch.linalg.svdvals(selected_functions)[-1].item())
            if selected_functions.numel()
            else None
        ),
        _bases=(),
        _source_levels=(),
        _quotient_plan=lie_plan,
        _quotient_indices=selected_global,
        _quotient_decoder=decoder,
        _represented_quotient_indices=represented_rows,
        _feature_depths=lie_plan.basis_depths.index_select(0, selected_global),
        _local_quotient_residual=residual,
        _runtime_quotient_cache={},
        _source_program_cache={},
    )


@dataclass(frozen=True, eq=False, init=False)
class PolynomialPlan(_CompiledPlan):
    """A prepared scalar-monomial plan for raw local polynomial functions."""

    window: int
    channels: int
    max_depth: int
    depths: tuple[int, ...]
    max_offset_span: int | None
    max_offset_support: int | None
    max_offset_multiplicity: int | None
    scaling: RawSignatureScale
    _dictionaries: tuple[ScaledMonomialDictionary, ...] = field(repr=False)

    @property
    def level_dimensions(self) -> tuple[int, ...]:
        return tuple(dictionary.feature_count for dictionary in self._dictionaries)

    @property
    def level_slices(self) -> tuple[slice, ...]:
        return _slices(self.level_dimensions)

    @property
    def feature_count(self) -> int:
        return sum(self.level_dimensions)

    @property
    def offset_indices(self) -> tuple[Tensor, ...]:
        """Return zero-based physical offsets for every monomial level."""
        return tuple(dictionary.offsets.clone() for dictionary in self._dictionaries)

    @property
    def channel_indices(self) -> tuple[Tensor, ...]:
        """Return zero-based channel indices for every monomial level."""
        return tuple(dictionary.channels.clone() for dictionary in self._dictionaries)

    @property
    def coefficients(self) -> tuple[Tensor, ...]:
        """Return the signature scaling of every scalar monomial."""
        return tuple(
            dictionary.coefficients.clone() for dictionary in self._dictionaries
        )

    def evaluate(self, windows: Tensor, *, chunk_size: int | None = None) -> Tensor:
        validate_increment_windows(windows, size=self.window, channels=self.channels)
        chunk_size = _validate_chunk_size(chunk_size)
        return torch.cat(
            [
                dictionary.features(windows, chunk_size=chunk_size)
                for dictionary in self._dictionaries
            ],
            dim=-1,
        )

    def contract(
        self,
        windows: Tensor,
        weight: Tensor,
        *,
        chunk_size: int | None = None,
    ) -> Tensor:
        validate_increment_windows(windows, size=self.window, channels=self.channels)
        chunk_size = _validate_chunk_size(chunk_size)
        if weight.ndim != 2 or weight.shape[1] != self.feature_count:
            raise ValueError(f"weight must have shape (outputs, {self.feature_count}).")
        output = windows.new_zeros(*windows.shape[:2], weight.shape[0])
        for dictionary, feature_slice in zip(
            self._dictionaries,
            self.level_slices,
            strict=True,
        ):
            output = output + dictionary.contract(
                windows,
                weight[:, feature_slice],
                chunk_size=chunk_size,
            )
        return output

    def summary(self) -> str:
        return (
            f"polynomial: levels={self.level_dimensions}, features={self.feature_count}"
        )


def compile_polynomial(
    *,
    window: int,
    channels: int,
    max_depth: int,
    depths: tuple[int, ...] | None = None,
    max_offset_span: int | None = None,
    max_offset_support: int | None = None,
    max_offset_multiplicity: int | None = None,
    scaling: RawSignatureScale = "signature",
) -> PolynomialPlan:
    """Compile a raw scalar-polynomial coordinate dictionary once."""
    for name, value in (("window", window), ("channels", channels)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    restrictions = (
        ("max_offset_span", max_offset_span, 0),
        ("max_offset_support", max_offset_support, 1),
        ("max_offset_multiplicity", max_offset_multiplicity, 1),
    )
    for name, value, minimum in restrictions:
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < minimum
        ):
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer or None.")
    if not isinstance(scaling, str) or scaling not in {"signature", "support"}:
        raise ValueError("scaling must be 'signature' or 'support'.")
    selected_depths = _depths(max_depth, depths)
    dictionaries = tuple(
        ScaledMonomialDictionary.raw_signature(
            in_channels=channels,
            kernel_size=window,
            depth=depth,
            max_offset_span=max_offset_span,
            max_offset_support=max_offset_support,
            max_offset_multiplicity=max_offset_multiplicity,
            scale=scaling,
            dtype=torch.float64,
        )
        for depth in selected_depths
    )
    return _compiled_plan(
        PolynomialPlan,
        window=window,
        channels=channels,
        max_depth=max(selected_depths),
        depths=selected_depths,
        max_offset_span=max_offset_span,
        max_offset_support=max_offset_support,
        max_offset_multiplicity=max_offset_multiplicity,
        scaling=scaling,
        _dictionaries=dictionaries,
    )


@dataclass(frozen=True, eq=False)
class LocalRealization:
    """A local alphabet map together with window, channel, and depth semantics."""

    mapping: OffsetCode | AlphabetMap
    channels: int
    max_depth: int

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, (OffsetCode, AlphabetMap)):
            raise TypeError("mapping must be an OffsetCode or AlphabetMap.")
        if (
            not isinstance(self.channels, int)
            or isinstance(self.channels, bool)
            or self.channels < 1
        ):
            raise ValueError("channels must be positive.")
        if (
            not isinstance(self.max_depth, int)
            or isinstance(self.max_depth, bool)
            or self.max_depth < 1
        ):
            raise ValueError("max_depth must be positive.")
        if isinstance(self.mapping, AlphabetMap) and (
            self.mapping.channels != self.channels
        ):
            raise ValueError("AlphabetMap channels disagree with channels.")

    @property
    def window(self) -> int:
        return self.mapping.window

    def signature(
        self,
        *,
        coordinates: Literal["chronological", "dense"] | None = None,
        selection: WordSelection | None = None,
        depths: tuple[int, ...] | None = None,
        rank_tolerance: float | None = None,
    ) -> SignaturePlan:
        """Compile local signature coordinates."""
        return _compile_signature(
            self,
            coordinates=coordinates,
            selection=selection,
            depths=depths,
            rank_tolerance=rank_tolerance,
        )

    def log_signature(
        self,
        *,
        quotient: LieQuotient | None = None,
        filtration: ChronologicalFiltration | None = None,
        depths: tuple[int, ...] | None = None,
        rank_tolerance: float | None = None,
    ) -> LogSignaturePlan:
        """Compile minimal local primitive coordinates."""
        if not isinstance(self.mapping, OffsetCode):
            raise ValueError("Local logsignature plans require an OffsetCode.")
        rank_tolerance = _validate_rank_tolerance(rank_tolerance)
        if filtration is not None and not isinstance(
            filtration,
            ChronologicalFiltration,
        ):
            raise TypeError("filtration must be a ChronologicalFiltration or None.")
        selected_depths = _depths(self.max_depth, depths)
        if quotient is not None and filtration is not None:
            raise ValueError(
                "A coordinate filtration is not a Lie quotient; apply one or the other."
            )
        if quotient is None:
            return _compile_unquotiented_log(
                self,
                selected_depths,
                filtration,
                rank_tolerance,
            )
        if not isinstance(quotient, LieQuotient):
            raise TypeError("quotient must be a LieQuotient or None.")
        return _compile_quotiented_log(
            self,
            selected_depths,
            quotient,
            rank_tolerance,
        )


__all__ = [
    "LocalRealization",
    "LogSignaturePlan",
    "PolynomialPlan",
    "SignaturePlan",
    "compile_polynomial",
]
