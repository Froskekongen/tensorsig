"""Direct signatures in a compiled quotient universal enveloping algebra."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Callable, TypeAlias

import torch
from torch import Tensor

from tensorsig.algebra.eulerian import eulerian_permutations

from .compiler import LieQuotientPlan
from .rules import ProjectedAlphabet


PBWMonomial: TypeAlias = tuple[int, ...]
_NormalForm: TypeAlias = tuple[tuple[PBWMonomial, Fraction], ...]


def _add_scaled(
    target: dict[PBWMonomial, Fraction],
    source: _NormalForm,
    scale: Fraction,
) -> None:
    for monomial, coefficient in source:
        value = target.get(monomial, Fraction()) + scale * coefficient
        if value:
            target[monomial] = value
        else:
            target.pop(monomial, None)


def _pbw_levels(
    basis_depths: tuple[int, ...],
    max_depth: int,
) -> tuple[tuple[PBWMonomial, ...], ...]:
    """Enumerate descending weighted PBW monomials by homogeneous level."""

    @lru_cache(maxsize=None)
    def monomials(
        remaining: int,
        maximum_index: int,
    ) -> tuple[PBWMonomial, ...]:
        if remaining == 0:
            return ((),)
        result = []
        for index in range(maximum_index, -1, -1):
            weight = basis_depths[index]
            if weight > remaining:
                continue
            result.extend(
                (index, *suffix) for suffix in monomials(remaining - weight, index)
            )
        return tuple(result)

    return (
        ((),),
        *(monomials(level, len(basis_depths) - 1) for level in range(1, max_depth + 1)),
    )


def _normalizer(
    plan: LieQuotientPlan,
) -> Callable[[PBWMonomial], _NormalForm]:
    brackets: dict[tuple[int, int], list[tuple[int, Fraction]]] = {}
    for indices, coefficient in zip(
        plan.bracket_indices.detach().cpu().tolist(),
        plan.bracket_coefficients,
    ):
        left, right, output = indices
        if coefficient:
            brackets.setdefault((left, right), []).append((output, coefficient))

    @lru_cache(maxsize=None)
    def normalize(word: PBWMonomial) -> _NormalForm:
        for position, (left, right) in enumerate(zip(word, word[1:])):
            if left >= right:
                continue
            prefix = word[:position]
            suffix = word[position + 2 :]
            result: dict[PBWMonomial, Fraction] = {}
            _add_scaled(
                result,
                normalize(prefix + (right, left) + suffix),
                Fraction(1),
            )
            for output, coefficient in brackets.get((left, right), ()):
                _add_scaled(
                    result,
                    normalize(prefix + (output,) + suffix),
                    coefficient,
                )
            return tuple(sorted(result.items()))
        return ((word, Fraction(1)),)

    return normalize


def _runtime_matrix(matrix: Tensor) -> Tensor:
    """Keep small/dense maps strided and strongly sparse maps in COO form."""
    if not matrix.numel():
        return matrix
    nonzero = int(torch.count_nonzero(matrix))
    if 4 * nonzero <= matrix.numel():
        return matrix.to_sparse_coo().coalesce()
    return matrix


@dataclass(frozen=True)
class QuotientSignaturePlan:
    """Compiled PBW signature runtime derived from a Lie quotient.

    Parameters
    ----------
    lie_plan : LieQuotientPlan
        Primitive quotient basis and exact bracket constants.
    basis : tuple[tuple[PBWMonomial, ...], ...]
        Weighted PBW monomials for levels zero through ``max_depth``.
    generator_map : Tensor
        Original alphabet generators in the degree-one primitive basis, with
        shape ``(alphabet_width, level_dimensions[0])``.
    action_indices : tuple[Tensor, ...]
        Per-level COO rows ``(source, degree_one_generator, target)`` for right
        multiplication from level ``n`` to ``n + 1``.
    action_values : tuple[Tensor, ...]
        Coefficients corresponding to ``action_indices``.
    log_readouts : tuple[Tensor, ...]
        Per-level first-Eulerian-idempotent maps from PBW signature coordinates
        to quotient primitive coordinates.
    """

    lie_plan: LieQuotientPlan
    basis: tuple[tuple[PBWMonomial, ...], ...]
    generator_map: Tensor
    action_indices: tuple[Tensor, ...]
    action_values: tuple[Tensor, ...]
    log_readouts: tuple[Tensor, ...]
    _bound_cache: dict[
        tuple[torch.device, torch.dtype],
        "QuotientSignaturePlan",
    ] = field(init=False, default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.basis) != self.max_depth + 1 or self.basis[0] != ((),):
            raise ValueError("basis must contain PBW levels zero through max_depth.")
        if self.generator_map.shape != (
            self.lie_plan.alphabet.width,
            self.lie_plan.level_dimensions[0],
        ):
            raise ValueError("generator_map has an invalid shape.")
        if len(self.action_indices) != max(0, self.max_depth - 1):
            raise ValueError("action_indices has an invalid number of levels.")
        if len(self.action_values) != len(self.action_indices):
            raise ValueError("action_values has an invalid number of levels.")
        if len(self.log_readouts) != self.max_depth:
            raise ValueError("log_readouts has an invalid number of levels.")
        for level, (indices, values) in enumerate(
            zip(self.action_indices, self.action_values),
            start=1,
        ):
            if indices.ndim != 2 or indices.shape[1] != 3:
                raise ValueError("action indices must have shape (nnz, 3).")
            if values.shape != (len(indices),):
                raise ValueError("action values must have shape (nnz,).")
            if indices.numel() and (
                torch.any(indices[:, 0] >= len(self.basis[level]))
                or torch.any(indices[:, 1] >= self.generator_map.shape[1])
                or torch.any(indices[:, 2] >= len(self.basis[level + 1]))
            ):
                raise ValueError("action indices contain an out-of-range entry.")
        for level, matrix in enumerate(self.log_readouts, start=1):
            expected = (
                len(self.basis[level]),
                self.lie_plan.level_dimensions[level - 1],
            )
            if matrix.shape != expected:
                raise ValueError("log_readout has an invalid shape.")

    @property
    def max_depth(self) -> int:
        """Largest represented homogeneous level."""
        return self.lie_plan.max_depth

    @property
    def level_dimensions(self) -> tuple[int, ...]:
        """Dimensions of PBW signature levels one through ``max_depth``."""
        return tuple(len(level) for level in self.basis[1:])

    @property
    def dimension(self) -> int:
        """Total non-scalar quotient signature dimension."""
        return sum(self.level_dimensions)

    @property
    def action_nnz(self) -> int:
        """Number of compiled right-action coefficients."""
        return sum(len(values) for values in self.action_values)

    def summary(self) -> str:
        """Return a compact structural summary."""
        return (
            f"PBW signature levels={self.level_dimensions}; "
            f"dimension={self.dimension}; actions={self.action_nnz}"
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "QuotientSignaturePlan":
        """Return the compiled runtime on ``device`` and ``dtype``."""
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
        key = (target_device, target_dtype)
        cached = self._bound_cache.get(key)
        if cached is not None:
            return cached
        moved = QuotientSignaturePlan(
            lie_plan=self.lie_plan.to(device=target_device, dtype=target_dtype),
            basis=self.basis,
            generator_map=self.generator_map.to(
                device=target_device,
                dtype=target_dtype,
            ),
            action_indices=tuple(
                indices.to(device=target_device) for indices in self.action_indices
            ),
            action_values=tuple(
                values.to(device=target_device, dtype=target_dtype)
                for values in self.action_values
            ),
            log_readouts=tuple(
                matrix.to(device=target_device, dtype=target_dtype)
                for matrix in self.log_readouts
            ),
        )
        source_key = (self.generator_map.device, self.generator_map.dtype)
        self._bound_cache.setdefault(source_key, self)
        self._bound_cache[key] = moved
        object.__setattr__(moved, "_bound_cache", self._bound_cache)
        object.__setattr__(moved.lie_plan, "_signature_plan", moved)
        return moved

    def _right_multiply(
        self,
        level: int,
        state: Tensor,
        increment: Tensor,
    ) -> Tensor:
        """Multiply a level-``level`` PBW state by a degree-one increment."""
        indices = self.action_indices[level - 1]
        values = self.action_values[level - 1]
        output_dimension = len(self.basis[level + 1])
        if not len(indices):
            zero = (
                state.sum(dim=-1, keepdim=True) + increment.sum(dim=-1, keepdim=True)
            ) * 0
            return zero.expand(*state.shape[:-1], output_dimension)
        contribution = (
            state[..., indices[:, 0]] * increment[..., indices[:, 1]] * values
        )
        return state.new_zeros(*state.shape[:-1], output_dimension).scatter_add(
            -1,
            indices[:, 2].expand_as(contribution),
            contribution,
        )

    def _apply_log_readout(self, level: int, state: Tensor) -> Tensor:
        matrix = self.log_readouts[level - 1]
        if matrix.layout == torch.strided:
            return state @ matrix
        leading_shape = state.shape[:-1]
        flattened = state.reshape(-1, state.shape[-1])
        projected = torch.sparse.mm(matrix.T, flattened.T).T
        return projected.reshape(*leading_shape, matrix.shape[1])


def compile_quotient_signature_plan(
    plan: LieQuotientPlan,
) -> QuotientSignaturePlan:
    """Compile weighted PBW normal forms and a direct signature runtime.

    Compilation is exact and may be substantially more expensive than one
    evaluation. The resulting plan is cached on ``plan`` so that this cost is
    amortized across time-series batches.
    """
    if not isinstance(plan, LieQuotientPlan):
        raise TypeError("plan must be a LieQuotientPlan.")
    cached = plan._signature_plan
    if isinstance(cached, QuotientSignaturePlan):
        return cached

    depths = tuple(int(value) for value in plan.basis_depths.detach().cpu())
    basis = _pbw_levels(depths, plan.max_depth)
    normalize = _normalizer(plan)
    degree_one = tuple(monomial[0] for monomial in basis[1])

    action_indices = []
    action_values = []
    for level in range(1, plan.max_depth):
        targets = {monomial: index for index, monomial in enumerate(basis[level + 1])}
        entries: dict[tuple[int, int, int], Fraction] = {}
        for source, monomial in enumerate(basis[level]):
            for generator, basis_index in enumerate(degree_one):
                for output, coefficient in normalize(monomial + (basis_index,)):
                    target = targets.get(output)
                    if target is None:  # pragma: no cover - guards PBW enumeration
                        raise RuntimeError("PBW action produced an unknown monomial.")
                    key = (source, generator, target)
                    entries[key] = entries.get(key, Fraction()) + coefficient
        ordered = tuple((key, value) for key, value in sorted(entries.items()) if value)
        action_indices.append(
            torch.tensor(
                [key for key, _ in ordered],
                dtype=torch.long,
            ).reshape(-1, 3)
        )
        action_values.append(
            torch.tensor(
                [float(value) for _, value in ordered],
                dtype=torch.float64,
            )
        )

    primitive_levels = tuple(
        tuple(index for index, depth in enumerate(depths) if depth == level)
        for level in range(1, plan.max_depth + 1)
    )
    if tuple(index for level in primitive_levels for index in level) != tuple(
        range(plan.dimension)
    ):
        raise RuntimeError("The quotient primitive basis must be grouped by level.")

    log_readouts = []
    for level, (monomials, primitives) in enumerate(
        zip(basis[1:], primitive_levels),
        start=1,
    ):
        primitive_positions = {
            basis_index: position for position, basis_index in enumerate(primitives)
        }
        readout = torch.zeros(
            len(monomials),
            len(primitives),
            dtype=torch.float64,
        )
        for source, monomial in enumerate(monomials):
            eulerian: dict[PBWMonomial, Fraction] = {}
            for permutation, coefficient in eulerian_permutations(len(monomial)):
                word = tuple(monomial[position] for position in permutation)
                _add_scaled(eulerian, normalize(word), coefficient)
            nonprimitive = [
                (term, value)
                for term, value in eulerian.items()
                if len(term) != 1 and value
            ]
            if nonprimitive:  # pragma: no cover - guards Eulerian/PBW exactness
                raise RuntimeError(
                    "The Eulerian readout did not reduce to primitive coordinates."
                )
            for (basis_index,), coefficient in eulerian.items():
                position = primitive_positions.get(basis_index)
                if position is None:  # pragma: no cover - guards the grading
                    raise RuntimeError("Eulerian readout changed homogeneous level.")
                readout[source, position] = float(coefficient)
        log_readouts.append(_runtime_matrix(readout))

    signature_plan = QuotientSignaturePlan(
        lie_plan=plan,
        basis=basis,
        generator_map=plan.generator_map[:, degree_one].contiguous(),
        action_indices=tuple(
            indices.to(device=plan.generator_map.device) for indices in action_indices
        ),
        action_values=tuple(
            values.to(
                device=plan.generator_map.device,
                dtype=plan.generator_map.dtype,
            )
            for values in action_values
        ),
        log_readouts=tuple(
            matrix.to(
                device=plan.generator_map.device,
                dtype=plan.generator_map.dtype,
            )
            for matrix in log_readouts
        ),
    )
    object.__setattr__(plan, "_signature_plan", signature_plan)
    return signature_plan


def _degree_one_segments(
    dxw: Tensor,
    plan: QuotientSignaturePlan,
    offset_code: Tensor | None,
) -> Tensor:
    if dxw.ndim != 4:
        raise ValueError(f"Expected dxw shape (B, N, R, C), got {tuple(dxw.shape)}.")
    if not dxw.dtype.is_floating_point:
        raise ValueError("dxw must have a floating-point dtype.")
    batch_size, num_windows, kernel_size, in_channels = dxw.shape
    alphabet = plan.lie_plan.alphabet
    if in_channels != alphabet.in_channels:
        raise ValueError("Input channels disagree with the quotient alphabet.")

    if isinstance(alphabet, ProjectedAlphabet):
        if offset_code is None:
            raise ValueError("offset_code is required for a ProjectedAlphabet.")
        if not offset_code.dtype.is_floating_point:
            raise ValueError("offset_code must have a floating-point dtype.")
        if tuple(offset_code.shape) != (kernel_size, alphabet.code_dim):
            raise ValueError(
                "offset_code shape disagrees with the windows and alphabet."
            )
        segments = torch.einsum("bnrc,re->bnrce", dxw, offset_code).flatten(-2)
        embedded = segments @ plan.generator_map
    else:
        if offset_code is not None:
            raise ValueError("offset_code is not used with an OffsetAlphabet.")
        if kernel_size != alphabet.kernel_size:
            raise ValueError("Window size disagrees with the OffsetAlphabet.")
        embedded = torch.stack(
            [
                dxw[:, :, offset]
                @ plan.generator_map[offset * in_channels : (offset + 1) * in_channels]
                for offset in range(kernel_size)
            ],
            dim=2,
        )
    return embedded.reshape(
        batch_size * num_windows,
        kernel_size,
        plan.generator_map.shape[1],
    )


def _signature_levels(
    degree_one_segments: Tensor,
    plan: QuotientSignaturePlan,
) -> list[Tensor]:
    """Return streaming non-scalar PBW levels for embedded increments."""
    if degree_one_segments.ndim != 3 or degree_one_segments.shape[1] < 1:
        raise ValueError("degree_one_segments must have shape (batch, steps, width).")
    stacked = [torch.cumsum(degree_one_segments, dim=1)]
    exponential = [degree_one_segments[:, 0]]
    for output_level in range(2, plan.max_depth + 1):
        current = stacked[0][:, :-1] + degree_one_segments[:, 1:] / output_level
        for current_level in range(1, output_level - 1):
            current = stacked[current_level][:, :-1] + plan._right_multiply(
                current_level,
                current,
                degree_one_segments[:, 1:] / (output_level - current_level),
            )
        current = plan._right_multiply(
            output_level - 1,
            current,
            degree_one_segments[:, 1:],
        )
        first = plan._right_multiply(
            output_level - 1,
            exponential[-1],
            degree_one_segments[:, 0] / output_level,
        )
        exponential.append(first)
        updates = torch.cat((first.unsqueeze(1), current), dim=1)
        stacked.append(torch.cumsum(updates, dim=1))
    return stacked


def quotient_signature_from_increment_windows(
    dxw: Tensor,
    plan: QuotientSignaturePlan,
    offset_code: Tensor | None = None,
) -> Tensor:
    """Evaluate quotient signatures directly in PBW coordinates.

    Parameters
    ----------
    dxw : Tensor
        Increment windows shaped ``(batch, windows, kernel_size, channels)``.
    plan : QuotientSignaturePlan
        Compiled quotient enveloping-algebra runtime.
    offset_code : Tensor, optional
        Required for a projected alphabet, with shape
        ``(kernel_size, code_dim)``.

    Returns
    -------
    Tensor
        Flattened non-scalar PBW levels with shape
        ``(batch, windows, plan.dimension)``.
    """
    if not isinstance(plan, QuotientSignaturePlan):
        raise TypeError("plan must be a QuotientSignaturePlan.")
    runtime_plan = plan.to(device=dxw.device, dtype=dxw.dtype)
    degree_one = _degree_one_segments(dxw, runtime_plan, offset_code)
    levels = _signature_levels(degree_one, runtime_plan)
    output = torch.cat([level[:, -1] for level in levels], dim=-1)
    return output.reshape(dxw.shape[0], dxw.shape[1], runtime_plan.dimension)


def _quotient_log_signature_from_increment_windows(
    dxw: Tensor,
    plan: QuotientSignaturePlan,
    offset_code: Tensor | None = None,
) -> Tensor:
    runtime_plan = plan.to(device=dxw.device, dtype=dxw.dtype)
    degree_one = _degree_one_segments(dxw, runtime_plan, offset_code)
    levels = _signature_levels(degree_one, runtime_plan)
    output = torch.cat(
        [
            runtime_plan._apply_log_readout(level, state[:, -1])
            for level, state in enumerate(levels, start=1)
        ],
        dim=-1,
    )
    return output.reshape(
        dxw.shape[0],
        dxw.shape[1],
        runtime_plan.lie_plan.dimension,
    )


__all__ = [
    "PBWMonomial",
    "QuotientSignaturePlan",
    "compile_quotient_signature_plan",
    "quotient_signature_from_increment_windows",
]
