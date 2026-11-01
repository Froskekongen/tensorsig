"""Quotient-native Hall normal forms for upper-set pruning ideals.

Upper-set rules kill whole homogeneous components of the free Lie algebra.
Their quotient basis is therefore the ordinary Hall basis restricted to the
retained phenomenon labels.  This module constructs that restriction directly
and reduces brackets with antisymmetry and Jacobi, without constructing an
ambient Hall basis, tensor-word projector, or ambient structure table.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import torch
from torch import Tensor

from tensorsig.algebra.hall_projection import (
    HallBasisElement,
    _hall_basis_key,
)

from .rules import (
    ChannelMultiplicity,
    ChannelSupport,
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


@dataclass(frozen=True)
class _PhenomenonLabel:
    """Additive/support label carried by one homogeneous Hall element."""

    level: int
    temporal_degree: int
    channel_counts: tuple[int, ...]
    offset_counts: tuple[int, ...]


@dataclass(frozen=True)
class UpperSetHallNormalForm:
    """A retained Hall basis and its exact sparse quotient bracket.

    Parameters
    ----------
    basis : tuple[HallBasisElement, ...]
        Retained Hall trees in the same order as filtering the ambient Hall
        basis degree by degree.
    generator_map : Tensor
        Embedding of alphabet generators into quotient coordinates, with shape
        ``(alphabet_width, dimension)``.
    bracket_indices : Tensor
        COO indices ``(left, right, output)`` with shape ``(nnz, 3)``.
    bracket_values : Tensor
        Exact integer Hall coefficients represented as ``float64`` values.
    bracket_coefficients : tuple[Fraction, ...]
        Exact coefficients matching ``bracket_indices``. These are retained for
        downstream symbolic compilation into enveloping-algebra normal forms.
    basis_depths : Tensor
        Lie level of each retained basis element.
    max_depth : int
        Highest Lie level considered during compilation.
    """

    basis: tuple[HallBasisElement, ...]
    generator_map: Tensor
    bracket_indices: Tensor
    bracket_values: Tensor
    bracket_coefficients: tuple[Fraction, ...]
    basis_depths: Tensor
    max_depth: int

    def __post_init__(self) -> None:
        dimension = len(self.basis)
        if self.generator_map.ndim != 2 or self.generator_map.shape[1] != dimension:
            raise ValueError("generator_map has an invalid shape.")
        if self.bracket_indices.ndim != 2 or self.bracket_indices.shape[1] != 3:
            raise ValueError("bracket_indices must have shape (nnz, 3).")
        if self.bracket_values.shape != (len(self.bracket_indices),):
            raise ValueError("bracket_values must have shape (nnz,).")
        if len(self.bracket_coefficients) != len(self.bracket_indices):
            raise ValueError("bracket_coefficients must have length nnz.")
        if self.basis_depths.shape != (dimension,):
            raise ValueError("basis_depths has an invalid shape.")

    @property
    def dimension(self) -> int:
        """Number of retained primitive coordinates."""
        return len(self.basis)

    @property
    def level_dimensions(self) -> tuple[int, ...]:
        """Retained primitive dimensions for all represented levels."""
        return tuple(
            int((self.basis_depths == level).sum())
            for level in range(1, self.max_depth + 1)
        )


def _validate_upper_set_rules(
    quotient: LieQuotient,
    alphabet: QuotientAlphabet,
) -> None:
    for rule in quotient.rules:
        if isinstance(rule, DependenceGraph):
            raise ValueError(
                "DependenceGraph is a relation ideal, not an upper-set rule."
            )
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


def _letter_label(letter: int, alphabet: QuotientAlphabet) -> _PhenomenonLabel:
    zero_based = letter - 1
    channel_counts = [0] * alphabet.in_channels
    if isinstance(alphabet, ProjectedAlphabet):
        channel = zero_based // alphabet.code_dim
        code_letter = zero_based % alphabet.code_dim
        temporal_degree = (
            alphabet.temporal_degrees[code_letter]
            if alphabet.temporal_degrees is not None
            else 0
        )
        offset_counts: tuple[int, ...] = ()
    else:
        channel = zero_based % alphabet.in_channels
        temporal_degree = 0
        offsets = [0] * alphabet.kernel_size
        offsets[zero_based // alphabet.in_channels] = 1
        offset_counts = tuple(offsets)
    channel_counts[channel] = 1
    return _PhenomenonLabel(
        level=1,
        temporal_degree=temporal_degree,
        channel_counts=tuple(channel_counts),
        offset_counts=offset_counts,
    )


def _combine_labels(
    left: _PhenomenonLabel,
    right: _PhenomenonLabel,
) -> _PhenomenonLabel:
    return _PhenomenonLabel(
        level=left.level + right.level,
        temporal_degree=left.temporal_degree + right.temporal_degree,
        channel_counts=tuple(
            first + second
            for first, second in zip(
                left.channel_counts,
                right.channel_counts,
                strict=True,
            )
        ),
        offset_counts=tuple(
            first + second
            for first, second in zip(
                left.offset_counts,
                right.offset_counts,
                strict=True,
            )
        ),
    )


def _is_retained(label: _PhenomenonLabel, quotient: LieQuotient) -> bool:
    for rule in quotient.rules:
        if isinstance(rule, TemporalDegree):
            if label.temporal_degree > rule.maximum:
                return False
        elif isinstance(rule, ChannelMultiplicity):
            caps = (
                (rule.maximum,) * len(label.channel_counts)
                if isinstance(rule.maximum, int)
                else rule.maximum
            )
            if any(
                count > cap
                for count, cap in zip(label.channel_counts, caps, strict=True)
            ):
                return False
        elif isinstance(rule, ChannelSupport):
            if sum(count > 0 for count in label.channel_counts) > rule.maximum:
                return False
        elif isinstance(rule, OffsetMultiplicity):
            if max(label.offset_counts) > rule.maximum:
                return False
        elif isinstance(rule, OffsetSupport):
            if sum(count > 0 for count in label.offset_counts) > rule.maximum:
                return False
        elif isinstance(rule, OffsetSpan):
            support = [
                offset for offset, count in enumerate(label.offset_counts) if count
            ]
            if support[-1] - support[0] > rule.maximum:
                return False
    return True


@lru_cache(maxsize=None)
def _hall_key(element: HallBasisElement) -> tuple[object, ...]:
    return _hall_basis_key(element)


def _is_valid_hall_pair(
    left: HallBasisElement,
    right: HallBasisElement,
) -> bool:
    if _hall_key(left) >= _hall_key(right):
        return False
    return not isinstance(right, tuple) or _hall_key(right[0]) <= _hall_key(left)


def _retained_hall_basis(
    quotient: LieQuotient,
    alphabet: QuotientAlphabet,
    max_depth: int,
) -> tuple[
    tuple[HallBasisElement, ...],
    dict[HallBasisElement, _PhenomenonLabel],
]:
    labels: dict[HallBasisElement, _PhenomenonLabel] = {}
    generators = []
    for letter in range(1, alphabet.width + 1):
        label = _letter_label(letter, alphabet)
        if _is_retained(label, quotient):
            labels[letter] = label
            generators.append(letter)

    depth_groups: dict[int, tuple[HallBasisElement, ...]] = {1: tuple(generators)}
    basis = list(generators)
    for depth in range(2, max_depth + 1):
        candidates = []
        for left_depth in range(1, depth):
            right_depth = depth - left_depth
            for left in depth_groups[left_depth]:
                for right in depth_groups[right_depth]:
                    if not _is_valid_hall_pair(left, right):
                        continue
                    label = _combine_labels(labels[left], labels[right])
                    if not _is_retained(label, quotient):
                        continue
                    element = (left, right)
                    labels[element] = label
                    candidates.append(element)
        candidates.sort(key=_hall_key)
        depth_groups[depth] = tuple(candidates)
        basis.extend(candidates)
    return tuple(basis), labels


def _normal_form_constants(
    basis: tuple[HallBasisElement, ...],
    labels: dict[HallBasisElement, _PhenomenonLabel],
    quotient: LieQuotient,
    max_depth: int,
) -> tuple[Tensor, Tensor, tuple[Fraction, ...]]:
    basis_indices = {element: index for index, element in enumerate(basis)}

    def add_scaled(
        target: dict[int, int],
        source: tuple[tuple[int, int], ...],
        scale: int,
    ) -> None:
        for index, value in source:
            coefficient = target.get(index, 0) + scale * value
            if coefficient:
                target[index] = coefficient
            else:
                target.pop(index, None)

    @lru_cache(maxsize=None)
    def reduce_bracket(
        left: HallBasisElement,
        right: HallBasisElement,
    ) -> tuple[tuple[int, int], ...]:
        combined_label = _combine_labels(labels[left], labels[right])
        if (
            combined_label.level > max_depth
            or not _is_retained(combined_label, quotient)
            or left == right
        ):
            return ()
        if _hall_key(left) > _hall_key(right):
            return tuple(
                (index, -coefficient)
                for index, coefficient in reduce_bracket(right, left)
            )
        if _is_valid_hall_pair(left, right):
            output = (left, right)
            try:
                return ((basis_indices[output], 1),)
            except KeyError as error:  # pragma: no cover - guards the basis invariant
                raise RuntimeError(
                    "A retained Hall pair is absent from the basis."
                ) from error

        right_left, right_right = right
        result: dict[int, int] = {}
        for index, coefficient in reduce_bracket(left, right_left):
            add_scaled(
                result,
                reduce_bracket(basis[index], right_right),
                coefficient,
            )
        for index, coefficient in reduce_bracket(left, right_right):
            add_scaled(
                result,
                reduce_bracket(right_left, basis[index]),
                coefficient,
            )
        return tuple(sorted(result.items()))

    entries: list[tuple[int, int, int, int]] = []
    for left_index, left in enumerate(basis):
        left_depth = labels[left].level
        for right_index in range(left_index + 1, len(basis)):
            right = basis[right_index]
            if left_depth + labels[right].level > max_depth:
                break
            for output_index, coefficient in reduce_bracket(left, right):
                entries.append((left_index, right_index, output_index, coefficient))
                entries.append((right_index, left_index, output_index, -coefficient))
    entries.sort(key=lambda entry: entry[:3])
    if not entries:
        return (
            torch.empty((0, 3), dtype=torch.long),
            torch.empty(0, dtype=torch.float64),
            (),
        )
    return (
        torch.tensor([entry[:3] for entry in entries], dtype=torch.long),
        torch.tensor([entry[3] for entry in entries], dtype=torch.float64),
        tuple(Fraction(entry[3]) for entry in entries),
    )


@lru_cache(maxsize=128)
def _compile_upper_set_hall_normal_form_cached(
    quotient: LieQuotient,
    alphabet: QuotientAlphabet,
    max_depth: int,
) -> UpperSetHallNormalForm:
    basis, labels = _retained_hall_basis(quotient, alphabet, max_depth)
    indices, values, coefficients = _normal_form_constants(
        basis,
        labels,
        quotient,
        max_depth,
    )
    generator_map = torch.zeros(alphabet.width, len(basis), dtype=torch.float64)
    basis_indices = {element: index for index, element in enumerate(basis)}
    for letter in range(1, alphabet.width + 1):
        index = basis_indices.get(letter)
        if index is not None:
            generator_map[letter - 1, index] = 1.0
    return UpperSetHallNormalForm(
        basis=basis,
        generator_map=generator_map,
        bracket_indices=indices,
        bracket_values=values,
        bracket_coefficients=coefficients,
        basis_depths=torch.tensor(
            [labels[element].level for element in basis],
            dtype=torch.long,
        ),
        max_depth=max_depth,
    )


def compile_upper_set_hall_normal_form(
    quotient: LieQuotient,
    alphabet: QuotientAlphabet,
    max_depth: int,
) -> UpperSetHallNormalForm:
    """Compile upper-set rules without constructing the ambient Lie algebra.

    Parameters
    ----------
    quotient : LieQuotient
        Conjunction of additive, support, or span rules. Relation rules are not
        accepted by this compiler.
    alphabet : ProjectedAlphabet or OffsetAlphabet
        Labelled generator alphabet on which the rules are defined.
    max_depth : int
        Maximum retained Lie level.

    Returns
    -------
    UpperSetHallNormalForm
        Retained Hall trees, generator embedding, and exact sparse bracket.

    Notes
    -----
    ``max_depth`` is limited only by the resources required by the retained
    basis and the higher-depth BCH expansion.
    """
    if not isinstance(quotient, LieQuotient):
        raise TypeError("quotient must be a LieQuotient.")
    if not isinstance(alphabet, (ProjectedAlphabet, OffsetAlphabet)):
        raise TypeError("alphabet must be ProjectedAlphabet or OffsetAlphabet.")
    max_depth = int(max_depth)
    if max_depth < 1:
        raise ValueError("max_depth must be positive.")
    _validate_upper_set_rules(quotient, alphabet)
    return _compile_upper_set_hall_normal_form_cached(
        quotient,
        alphabet,
        max_depth,
    )


__all__ = [
    "UpperSetHallNormalForm",
    "compile_upper_set_hall_normal_form",
]
