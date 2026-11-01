"""Declarative rules for algebraic local-signature quotients."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias


def _validate_non_negative(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True)
class TemporalDegree:
    """Kill primitive directions above a total Newton-degree budget."""

    maximum: int

    def __post_init__(self) -> None:
        _validate_non_negative("TemporalDegree.maximum", self.maximum)


@dataclass(frozen=True)
class ChannelMultiplicity:
    """Kill primitives exceeding per-channel occurrence caps.

    ``maximum`` may be one common non-negative cap or one cap per channel.
    """

    maximum: int | tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.maximum, int) and not isinstance(self.maximum, bool):
            caps = (self.maximum,)
        elif isinstance(self.maximum, Sequence) and not isinstance(
            self.maximum, (str, bytes)
        ):
            caps = tuple(self.maximum)
            object.__setattr__(self, "maximum", caps)
        else:
            raise TypeError(
                "ChannelMultiplicity.maximum must be an integer or sequence."
            )
        if not caps:
            raise ValueError("ChannelMultiplicity.maximum must not be empty.")
        for cap in caps:
            _validate_non_negative("ChannelMultiplicity.maximum", cap)


@dataclass(frozen=True)
class ChannelSupport:
    """Kill primitives involving more than ``maximum`` channels."""

    maximum: int

    def __post_init__(self) -> None:
        _validate_non_negative("ChannelSupport.maximum", self.maximum)


@dataclass(frozen=True)
class OffsetMultiplicity:
    """Kill primitives exceeding a per-offset occurrence cap."""

    maximum: int

    def __post_init__(self) -> None:
        _validate_non_negative("OffsetMultiplicity.maximum", self.maximum)


@dataclass(frozen=True)
class OffsetSupport:
    """Kill primitives involving more than ``maximum`` relative offsets."""

    maximum: int

    def __post_init__(self) -> None:
        _validate_non_negative("OffsetSupport.maximum", self.maximum)


@dataclass(frozen=True)
class OffsetSpan:
    """Kill primitives whose physical offset-support diameter is too large."""

    maximum: int

    def __post_init__(self) -> None:
        _validate_non_negative("OffsetSpan.maximum", self.maximum)


@dataclass(frozen=True, init=False)
class DependenceGraph:
    """Make distinct channels commute unless their pair is an allowed edge.

    Channel indices are zero-based.  Self-interactions are always retained and
    therefore need not be listed.
    """

    edges: frozenset[tuple[int, int]]

    def __init__(self, edges: Iterable[Sequence[int]] = ()) -> None:
        normalized: set[tuple[int, int]] = set()
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("DependenceGraph edges must contain two channels.")
            left, right = edge
            if (
                not isinstance(left, int)
                or isinstance(left, bool)
                or not isinstance(right, int)
                or isinstance(right, bool)
            ):
                raise TypeError("DependenceGraph channel indices must be integers.")
            if left < 0 or right < 0:
                raise ValueError(
                    "DependenceGraph channel indices must be non-negative."
                )
            if left == right:
                continue
            normalized.add((min(left, right), max(left, right)))
        object.__setattr__(self, "edges", frozenset(normalized))


QuotientRule: TypeAlias = (
    TemporalDegree
    | ChannelMultiplicity
    | ChannelSupport
    | OffsetMultiplicity
    | OffsetSupport
    | OffsetSpan
    | DependenceGraph
)


@dataclass(frozen=True, init=False)
class LieQuotient:
    """A conjunction of Lie-ideal rules defining one exact quotient.

    Passing several rules imposes all constraints, i.e. takes the join (sum)
    of their discarded ideals.
    """

    rules: tuple[QuotientRule, ...]

    def __init__(self, *rules: QuotientRule) -> None:
        if not all(
            isinstance(
                rule,
                (
                    TemporalDegree,
                    ChannelMultiplicity,
                    ChannelSupport,
                    OffsetMultiplicity,
                    OffsetSupport,
                    OffsetSpan,
                    DependenceGraph,
                ),
            )
            for rule in rules
        ):
            raise TypeError("LieQuotient accepts only structured quotient rules.")
        object.__setattr__(self, "rules", tuple(rules))

    def __and__(self, other: "LieQuotient") -> "LieQuotient":
        """Return the quotient imposing both sets of constraints."""
        if not isinstance(other, LieQuotient):
            return NotImplemented
        return LieQuotient(*self.rules, *other.rules)


@dataclass(frozen=True)
class ProjectedAlphabet:
    """Channel-major alphabet ``(channel, projected_code_letter)``.

    ``temporal_degrees`` is deliberately optional: arbitrary learned or dense
    projections do not automatically carry Newton-degree semantics. Use
    :meth:`newton` for an ordered Newton/Pascal code, or declare other additive
    degrees explicitly when the projection is homogeneous for that grading.
    """

    code_dim: int
    in_channels: int
    temporal_degrees: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _validate_positive("ProjectedAlphabet.code_dim", self.code_dim)
        _validate_positive("ProjectedAlphabet.in_channels", self.in_channels)
        if self.temporal_degrees is not None:
            normalized = tuple(self.temporal_degrees)
            if len(normalized) != self.code_dim:
                raise ValueError(
                    "temporal_degrees must contain one degree per code letter."
                )
            for degree in normalized:
                _validate_non_negative("ProjectedAlphabet.temporal_degrees", degree)
            object.__setattr__(self, "temporal_degrees", normalized)

    @classmethod
    def newton(cls, code_dim: int, in_channels: int) -> "ProjectedAlphabet":
        """Return the alphabet grading for ordered Newton/Pascal columns."""
        return cls(
            code_dim=code_dim,
            in_channels=in_channels,
            temporal_degrees=tuple(range(code_dim)),
        )

    @property
    def width(self) -> int:
        """Number of projected letters."""
        return self.code_dim * self.in_channels


@dataclass(frozen=True)
class OffsetAlphabet:
    """Offset-major full alphabet ``(offset, channel)``."""

    kernel_size: int
    in_channels: int

    def __post_init__(self) -> None:
        _validate_positive("OffsetAlphabet.kernel_size", self.kernel_size)
        _validate_positive("OffsetAlphabet.in_channels", self.in_channels)

    @property
    def width(self) -> int:
        """Number of full offset-labelled letters."""
        return self.kernel_size * self.in_channels


QuotientAlphabet: TypeAlias = ProjectedAlphabet | OffsetAlphabet
