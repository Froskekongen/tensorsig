"""Compile and evaluate structured Lie quotients without ambient construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

import torch
from torch import Tensor

from tensorsig.algebra.hall_bch import SparseBCH
from tensorsig.algebra.hall_projection import HallBasisElement
from tensorsig.algebra.lyndon import lyndon_words
from tensorsig.algebra.lyndon_words import WordsBasisElement
from tensorsig.classical.log_signature import (
    _batch_log_signature_from_increments,
)

from .rules import (
    ChannelMultiplicity,
    DependenceGraph,
    LieQuotient,
    OffsetAlphabet,
    OffsetMultiplicity,
    OffsetSpan,
    OffsetSupport,
    ProjectedAlphabet,
    QuotientAlphabet,
    TemporalDegree,
)
from .trace_compiler import TraceBasisElement, compile_trace_lie_quotient
from .upper_set_normal_form import compile_upper_set_hall_normal_form


@dataclass(frozen=True)
class _WordsBridge:
    """Dense or COO map from ambient Lyndon words to a quotient."""

    ambient_words: tuple[WordsBasisElement, ...]
    quotient_matrix: Tensor
    active_columns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.quotient_matrix.shape[1] != len(self.ambient_words):
            raise ValueError("quotient_matrix has an invalid shape.")

    def project(self, coordinates: Tensor) -> Tensor:
        """Project trailing ambient word coordinates into the quotient."""
        if coordinates.shape[-1] != len(self.ambient_words):
            raise ValueError(
                f"Expected coordinates.shape[-1] == {len(self.ambient_words)}."
            )
        matrix = self.quotient_matrix
        if coordinates.device.type == "mps" and matrix.layout != torch.strided:
            matrix = matrix.to_dense()
        matrix = matrix.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        if matrix.layout == torch.strided:
            return coordinates @ matrix.T
        leading_shape = coordinates.shape[:-1]
        flattened = coordinates.reshape(-1, coordinates.shape[-1])
        projected = torch.sparse.mm(matrix, flattened.T).T
        return projected.reshape(*leading_shape, matrix.shape[0])


@dataclass(frozen=True)
class LieQuotientPlan:
    """Intrinsic normal-form basis and sparse bracket table for one quotient.

    Unlike the former plan representation, this object contains no required
    ambient Hall basis or dense projection matrices. ``basis`` contains the
    retained Hall trees for upper-set quotients and exact trace normal forms for
    graph quotients. Tensor evaluation constructs its word projection lazily.
    """

    quotient: LieQuotient
    alphabet: QuotientAlphabet
    max_depth: int
    basis: tuple[HallBasisElement | TraceBasisElement, ...]
    generator_map: Tensor
    bracket_indices: Tensor
    bracket_values: Tensor
    bracket_coefficients: tuple[Fraction, ...]
    basis_depths: Tensor
    coordinate_aligned: bool
    compiler: str
    _runtime: SparseBCH = field(init=False, repr=False, compare=False)
    _generator_coordinates: Tensor | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    _word_bridge_cache: _WordsBridge | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _signature_plan: object | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _active_word_cache: object | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _bound_cache: dict[
        tuple[torch.device, torch.dtype],
        LieQuotientPlan,
    ] = field(
        init=False,
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        dimension = len(self.basis)
        if self.generator_map.shape != (self.alphabet.width, dimension):
            raise ValueError("generator_map has an invalid shape.")
        if self.bracket_indices.ndim != 2 or self.bracket_indices.shape[1] != 3:
            raise ValueError("bracket_indices must have shape (nnz, 3).")
        if self.bracket_values.shape != (self.bracket_indices.shape[0],):
            raise ValueError("bracket_values must have shape (nnz,).")
        if len(self.bracket_coefficients) != len(self.bracket_indices):
            raise ValueError("bracket_coefficients must have length nnz.")
        if self.basis_depths.shape != (dimension,):
            raise ValueError("basis_depths has an invalid shape.")
        generator_coordinates = torch.full(
            (self.alphabet.width,),
            -1,
            device=self.generator_map.device,
            dtype=torch.long,
        )
        nonzero = torch.nonzero(self.generator_map, as_tuple=False)
        simple_generator_map = not nonzero.numel() or (
            torch.bincount(nonzero[:, 0], minlength=self.alphabet.width).max() <= 1
            and len(torch.unique(nonzero[:, 1])) == len(nonzero)
            and torch.all(self.generator_map[tuple(nonzero.T)] == 1)
        )
        if simple_generator_map and nonzero.numel():
            generator_coordinates[nonzero[:, 0]] = nonzero[:, 1]
        object.__setattr__(
            self,
            "_generator_coordinates",
            generator_coordinates if simple_generator_map else None,
        )
        object.__setattr__(
            self,
            "_runtime",
            SparseBCH(
                dim=dimension,
                depth=self.max_depth,
                sparse_indices=self.bracket_indices,
                sparse_values=self.bracket_values,
                device=self.generator_map.device,
                dtype=self.generator_map.dtype,
            ),
        )

    @property
    def dimension(self) -> int:
        """Dimension of the quotient Lie algebra."""
        return len(self.basis)

    @property
    def ambient_dimension(self) -> int:
        """Dimension of the ambient free step-nilpotent Lie algebra."""
        return _free_lie_dimension(self.alphabet.width, self.max_depth)

    @property
    def level_dimensions(self) -> tuple[int, ...]:
        """Quotient dimensions in homogeneous levels ``1..max_depth``."""
        return tuple(
            int((self.basis_depths == level).sum())
            for level in range(1, self.max_depth + 1)
        )

    @property
    def _word_bridge(self) -> _WordsBridge:
        """Build the signature-to-quotient projection on first use."""
        return _compile_words_bridge(self)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LieQuotientPlan":
        """Return an intrinsic plan with runtime tensors on ``device``/``dtype``."""
        target_device = (
            self.generator_map.device if device is None else torch.device(device)
        )
        target_dtype = self.generator_map.dtype if dtype is None else dtype
        if not target_dtype.is_floating_point:
            raise ValueError("dtype must be floating-point.")
        if (
            target_device == self.generator_map.device
            and target_dtype == self.generator_map.dtype
        ):
            return self
        cache_key = (target_device, target_dtype)
        cached = self._bound_cache.get(cache_key)
        if cached is not None:
            return cached
        moved = LieQuotientPlan(
            quotient=self.quotient,
            alphabet=self.alphabet,
            max_depth=self.max_depth,
            basis=self.basis,
            generator_map=self.generator_map.to(
                device=target_device,
                dtype=target_dtype,
            ),
            bracket_indices=self.bracket_indices.to(device=target_device),
            bracket_values=self.bracket_values.to(
                device=target_device,
                dtype=target_dtype,
            ),
            bracket_coefficients=self.bracket_coefficients,
            basis_depths=self.basis_depths.to(device=target_device),
            coordinate_aligned=self.coordinate_aligned,
            compiler=self.compiler,
        )
        if self._word_bridge_cache is not None:
            object.__setattr__(
                moved,
                "_word_bridge_cache",
                _WordsBridge(
                    ambient_words=self._word_bridge_cache.ambient_words,
                    quotient_matrix=(
                        self._word_bridge_cache.quotient_matrix.to_dense()
                        if target_device.type == "mps"
                        and self._word_bridge_cache.quotient_matrix.layout
                        != torch.strided
                        else self._word_bridge_cache.quotient_matrix
                    ).to(
                        device=target_device,
                        dtype=target_dtype,
                    ),
                    active_columns=self._word_bridge_cache.active_columns,
                ),
            )
        source_key = (self.generator_map.device, self.generator_map.dtype)
        self._bound_cache.setdefault(source_key, self)
        self._bound_cache[cache_key] = moved
        object.__setattr__(moved, "_bound_cache", self._bound_cache)
        return moved

    def _embed_generators(self, increment: Tensor, start: int = 0) -> Tensor:
        """Embed consecutive alphabet coordinates into the quotient state."""
        rows = self.generator_map[start : start + increment.shape[-1]]
        if self._generator_coordinates is None or self.dimension == 0:
            return increment @ rows
        coordinates = self._generator_coordinates[start : start + increment.shape[-1]]
        retained = coordinates >= 0
        result = increment.new_zeros(*increment.shape[:-1], self.dimension)
        result[..., coordinates[retained]] = increment[..., retained]
        return result

    def summary(self) -> str:
        """Return a compact structural and computational summary."""
        return (
            f"{self.compiler}: {self.ambient_dimension} -> {self.dimension} "
            f"primitives; levels={self.level_dimensions}; "
            f"brackets={len(self.bracket_values)}"
        )


def _mobius(value: int) -> int:
    """Return the integer Möbius function."""
    remaining = value
    prime_factors = 0
    divisor = 2
    while divisor * divisor <= remaining:
        if remaining % divisor:
            divisor += 1
            continue
        remaining //= divisor
        if remaining % divisor == 0:
            return 0
        prime_factors += 1
        while remaining % divisor == 0:
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        prime_factors += 1
    return -1 if prime_factors % 2 else 1


@lru_cache(maxsize=None)
def _free_lie_dimension(width: int, depth: int) -> int:
    """Return the cumulative Witt dimension without enumerating a basis."""
    return sum(
        sum(
            _mobius(divisor) * width ** (level // divisor)
            for divisor in range(1, level + 1)
            if level % divisor == 0
        )
        // level
        for level in range(1, depth + 1)
    )


def _validate_rules(quotient: LieQuotient, alphabet: QuotientAlphabet) -> None:
    """Validate rule metadata before selecting a normal-form backend."""
    for rule in quotient.rules:
        if isinstance(rule, TemporalDegree):
            if not isinstance(alphabet, ProjectedAlphabet):
                raise ValueError("TemporalDegree requires a ProjectedAlphabet.")
            if alphabet.temporal_degrees is None:
                raise ValueError(
                    "TemporalDegree requires explicitly graded projected letters; "
                    "use ProjectedAlphabet.newton(...) for a Newton/Pascal code."
                )
        if isinstance(rule, (OffsetMultiplicity, OffsetSupport, OffsetSpan)):
            if not isinstance(alphabet, OffsetAlphabet):
                raise ValueError(f"{type(rule).__name__} requires an OffsetAlphabet.")
        if isinstance(rule, ChannelMultiplicity) and not isinstance(rule.maximum, int):
            if len(rule.maximum) != alphabet.in_channels:
                raise ValueError(
                    "ChannelMultiplicity requires one cap per alphabet channel."
                )
        if isinstance(rule, DependenceGraph):
            if any(right >= alphabet.in_channels for _, right in rule.edges):
                raise ValueError("DependenceGraph contains an out-of-range channel.")


def _graph_has_relations(
    quotient: LieQuotient,
    alphabet: QuotientAlphabet,
) -> bool:
    graphs = [rule for rule in quotient.rules if isinstance(rule, DependenceGraph)]
    if not graphs:
        return False
    edges = set(graphs[0].edges)
    for graph in graphs[1:]:
        edges.intersection_update(graph.edges)
    return any(
        (left, right) not in edges
        for left in range(alphabet.in_channels)
        for right in range(left + 1, alphabet.in_channels)
    )


@lru_cache(maxsize=128)
def _compile_lie_quotient_cached(
    quotient: LieQuotient,
    alphabet: QuotientAlphabet,
    max_depth: int,
) -> LieQuotientPlan:
    _validate_rules(quotient, alphabet)
    if max_depth < 1:
        raise ValueError("max_depth must be positive.")

    if _graph_has_relations(quotient, alphabet):
        compiled = compile_trace_lie_quotient(quotient, alphabet, max_depth)
        return LieQuotientPlan(
            quotient=quotient,
            alphabet=alphabet,
            max_depth=max_depth,
            basis=compiled.basis,
            generator_map=compiled.generator_map,
            bracket_indices=compiled.bracket_indices,
            bracket_values=compiled.bracket_values,
            bracket_coefficients=compiled.bracket_coefficients,
            basis_depths=compiled.basis_depths,
            coordinate_aligned=False,
            compiler="trace normal form",
        )

    upper_rules = tuple(
        rule for rule in quotient.rules if not isinstance(rule, DependenceGraph)
    )
    compiled = compile_upper_set_hall_normal_form(
        LieQuotient(*upper_rules),
        alphabet,
        max_depth,
    )
    return LieQuotientPlan(
        quotient=quotient,
        alphabet=alphabet,
        max_depth=max_depth,
        basis=compiled.basis,
        generator_map=compiled.generator_map,
        bracket_indices=compiled.bracket_indices,
        bracket_values=compiled.bracket_values,
        bracket_coefficients=compiled.bracket_coefficients,
        basis_depths=compiled.basis_depths,
        coordinate_aligned=True,
        compiler="Hall normal form",
    )


def compile_lie_quotient(
    quotient: LieQuotient,
    alphabet: QuotientAlphabet,
    max_depth: int,
) -> LieQuotientPlan:
    """Compile a quotient-native normal form and sparse runtime bracket.

    Compilation is exact for every current rule and has no numerical rank
    tolerance.
    """
    if not isinstance(quotient, LieQuotient):
        raise TypeError("quotient must be a LieQuotient.")
    if not isinstance(alphabet, (ProjectedAlphabet, OffsetAlphabet)):
        raise TypeError("alphabet must be ProjectedAlphabet or OffsetAlphabet.")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool):
        raise TypeError("max_depth must be an integer.")
    return _compile_lie_quotient_cached(
        quotient,
        alphabet,
        max_depth,
    )


@lru_cache(maxsize=None)
def _standard_lyndon_factorization(
    word: WordsBasisElement,
) -> tuple[WordsBasisElement, WordsBasisElement]:
    """Return the standard factorization by the longest Lyndon suffix."""
    for split in range(1, len(word)):
        suffix = word[split:]
        if all(
            suffix < suffix[index:] + suffix[:index] for index in range(1, len(suffix))
        ):
            return word[:split], suffix
    raise RuntimeError(f"Could not factor Lyndon word {word}.")


@lru_cache(maxsize=None)
def _standard_lyndon_expansion(
    word: WordsBasisElement,
) -> tuple[tuple[WordsBasisElement, int], ...]:
    """Expand a standard Lyndon bracket into sparse tensor words."""
    if len(word) == 1:
        return ((word, 1),)
    left, right = _standard_lyndon_factorization(word)
    left_terms = _standard_lyndon_expansion(left)
    right_terms = _standard_lyndon_expansion(right)
    result: dict[WordsBasisElement, int] = {}
    for left_word, left_value in left_terms:
        for right_word, right_value in right_terms:
            value = left_value * right_value
            forward = left_word + right_word
            reverse = right_word + left_word
            result[forward] = result.get(forward, 0) + value
            result[reverse] = result.get(reverse, 0) - value
    return tuple(sorted((term, value) for term, value in result.items() if value))


def _compile_words_bridge(plan: LieQuotientPlan) -> _WordsBridge:
    """Compile a projection from fast Lyndon-word logs to ``plan`` coordinates.

    The ordinary log-signature backend returns tensor-log coefficients at
    Lyndon words by indexed gathers. Standard Lyndon brackets form a triangular
    Lie basis; sparse back-substitution converts those gathered coordinates to
    the quotient without constructing an ambient Hall projector.
    """
    if not isinstance(plan, LieQuotientPlan):
        raise TypeError("plan must be a LieQuotientPlan.")
    cached = plan._word_bridge_cache
    if cached is not None:
        return cached

    ambient_words = lyndon_words(plan.alphabet.width, plan.max_depth)
    word_indices = {word: index for index, word in enumerate(ambient_words)}
    matrix = torch.zeros(
        plan.dimension,
        len(ambient_words),
        dtype=torch.float64,
    )
    active_columns = []
    if plan.dimension:
        # Exact structure constants certify zero support before any dtype binding.
        brackets = {}
        for (left, right, output), coefficient in zip(
            plan.bracket_indices.cpu().tolist(), plan.bracket_coefficients, strict=True
        ):
            brackets.setdefault((left, right), []).append((output, coefficient))

        def add(target, source, scale=Fraction(1)):
            for index, value in source.items():
                updated = target.get(index, Fraction()) + scale * value
                if updated:
                    target[index] = updated
                else:
                    target.pop(index, None)

        def bracket(left, right):
            result = {}
            for i, a in left.items():
                for j, b in right.items():
                    for output, coefficient in brackets.get((i, j), ()):
                        add(result, {output: coefficient}, a * b)
            return result

        generators = []
        for row in plan.generator_map.detach().cpu().tolist():
            if any(value != int(value) for value in row):
                raise ValueError(
                    "Exact word bridges require integral generator embeddings."
                )
            generators.append(
                {i: Fraction(int(value)) for i, value in enumerate(row) if value}
            )
        standard_coordinates = {}
        for word in ambient_words:
            if len(word) == 1:
                standard_coordinates[word] = generators[word[0] - 1]
            else:
                left, right = _standard_lyndon_factorization(word)
                standard_coordinates[word] = bracket(
                    standard_coordinates[left],
                    standard_coordinates[right],
                )

        columns = {}
        for column in reversed(range(len(ambient_words))):
            word = ambient_words[column]
            coordinate = standard_coordinates[word].copy()
            for tensor_word, coefficient in _standard_lyndon_expansion(word):
                row = word_indices.get(tensor_word)
                if row is None or row == column:
                    continue
                if row < column:  # pragma: no cover - Lyndon triangularity guard
                    raise RuntimeError("Standard Lyndon expansion is not triangular.")
                add(coordinate, columns[row], -coefficient)
            columns[column] = coordinate
            if coordinate:
                active_columns.append(column)
                for row, coefficient in coordinate.items():
                    matrix[row, column] = float(coefficient)

    matrix = matrix.to(
        device=plan.generator_map.device,
        dtype=plan.generator_map.dtype,
    )
    # Strong pruning makes COO both smaller and faster in the isolated bridge
    # kernel. Near-square weak quotients retain dense BLAS, which wins there.
    if (
        plan.dimension
        and 4 * plan.dimension <= len(ambient_words)
        and matrix.device.type != "mps"
    ):
        matrix = matrix.to_sparse_coo().coalesce()
    bridge = _WordsBridge(
        ambient_words=ambient_words,
        quotient_matrix=matrix,
        active_columns=tuple(sorted(active_columns)),
    )
    object.__setattr__(plan, "_word_bridge_cache", bridge)
    return bridge


def quotient_log_signature_from_increment_windows(
    dxw: Tensor,
    plan: LieQuotientPlan,
    offset_code: Tensor | None = None,
    *,
    method: str = "direct",
) -> Tensor:
    """Evaluate a compiled quotient logsignature from increment windows.

    Parameters
    ----------
    dxw : Tensor
        Increment windows with shape ``(batch, windows, kernel_size, channels)``.
    plan : LieQuotientPlan
        Intrinsic quotient plan for a projected or full-offset alphabet.
    offset_code : Tensor, optional
        Required for ``ProjectedAlphabet``; shape ``(kernel_size, code_dim)``.
    method : {"direct", "pbw", "tensor", "active"}, optional
        ``"direct"`` carries only quotient coordinates through sparse BCH.
        ``"pbw"`` computes the signature in the quotient universal enveloping
        algebra and applies a compiled Eulerian primitive readout.
        ``"tensor"`` uses the ordinary full signature-to-log kernel and then a
        compiled Lyndon-word bridge. The faster method depends on width, depth,
        window length, parallel path count, device, and pruning strength.
        ``"active"`` evaluates only the exact bridge support and its factors.

    Returns
    -------
    Tensor
        Quotient coordinates with shape ``(batch, windows, plan.dimension)``.
    """
    if dxw.ndim != 4:
        raise ValueError(f"Expected dxw shape (B, N, R, C), got {tuple(dxw.shape)}.")
    if not dxw.dtype.is_floating_point:
        raise ValueError("dxw must have a floating-point dtype.")
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if method not in {"direct", "pbw", "tensor", "active"}:
        raise ValueError("method must be 'direct', 'pbw', 'tensor', or 'active'.")
    batch_size, num_windows, kernel_size, in_channels = dxw.shape
    if in_channels != plan.alphabet.in_channels:
        raise ValueError("Input channels disagree with the quotient alphabet.")

    if method == "active":
        from ..active_word_log import active_word_log_signature

        return active_word_log_signature(dxw, plan, offset_code)

    if method == "pbw":
        from .enveloping import (
            _quotient_log_signature_from_increment_windows,
            compile_quotient_signature_plan,
        )

        runtime_plan = plan.to(device=dxw.device, dtype=dxw.dtype)
        signature_plan = compile_quotient_signature_plan(runtime_plan)
        return _quotient_log_signature_from_increment_windows(
            dxw,
            signature_plan,
            offset_code,
        )

    if isinstance(plan.alphabet, ProjectedAlphabet):
        if offset_code is None:
            raise ValueError("offset_code is required for a ProjectedAlphabet.")
        if not offset_code.dtype.is_floating_point:
            raise ValueError("offset_code must have a floating-point dtype.")
        if tuple(offset_code.shape) != (kernel_size, plan.alphabet.code_dim):
            raise ValueError(
                "offset_code shape disagrees with the windows and alphabet."
            )
        segments = torch.einsum("bnrc,re->bnrce", dxw, offset_code).flatten(-2)
    else:
        if offset_code is not None:
            raise ValueError("offset_code is not used with an OffsetAlphabet.")
        if kernel_size != plan.alphabet.kernel_size:
            raise ValueError("Window size disagrees with the OffsetAlphabet.")
        segments = dxw

    runtime_plan = plan.to(device=dxw.device, dtype=dxw.dtype)

    if method == "tensor":
        if isinstance(plan.alphabet, OffsetAlphabet):
            ambient_segments = dxw.new_zeros(
                batch_size,
                num_windows,
                kernel_size,
                kernel_size,
                in_channels,
            )
            offsets = torch.arange(kernel_size, device=dxw.device)
            ambient_segments[:, :, offsets, offsets] = dxw
            segments = ambient_segments.flatten(-2)
        flat_segments = segments.reshape(
            batch_size * num_windows,
            kernel_size,
            plan.alphabet.width,
        )
        ambient_log = _batch_log_signature_from_increments(
            flat_segments,
            depth=plan.max_depth,
            mode="words",
        )
        projected = runtime_plan._word_bridge.project(ambient_log)
        return projected.reshape(batch_size, num_windows, runtime_plan.dimension)

    state = torch.zeros(
        batch_size * num_windows,
        runtime_plan.dimension,
        device=dxw.device,
        dtype=dxw.dtype,
    )
    for offset in range(kernel_size):
        increment = segments[:, :, offset].reshape(batch_size * num_windows, -1)
        if isinstance(plan.alphabet, ProjectedAlphabet):
            quotient_increment = runtime_plan._embed_generators(increment)
        else:
            start = offset * in_channels
            quotient_increment = runtime_plan._embed_generators(increment, start)
        state = runtime_plan._runtime.bch(state, quotient_increment)
    return state.reshape(batch_size, num_windows, runtime_plan.dimension)


__all__ = [
    "LieQuotientPlan",
    "compile_lie_quotient",
    "quotient_log_signature_from_increment_windows",
]
