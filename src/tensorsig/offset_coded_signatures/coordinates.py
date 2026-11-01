"""Coordinate restrictions for compiled local feature plans.

Selections restrict which coordinate functionals a plan returns; they do not
change the represented signature group and are not Lie quotients.
``WordSelection`` filters certified chronological tensor words, while
``ChronologicalFiltration`` intersects a local primitive image with a
bounded chronological degree. A restriction may be lossy, so compiled plans
report their realized rank and reconstruction status explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordSelection:
    """Select canonical chronological signature words."""

    max_temporal_degree: int | None = None
    max_channel_descents: int | None = None
    max_complexity: int | None = None
    descent_sets: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_temporal_degree",
            "max_channel_descents",
            "max_complexity",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        if self.descent_sets is None:
            return
        normalized = tuple(tuple(row) for row in self.descent_sets)
        for row in normalized:
            if any(
                not isinstance(position, int)
                or isinstance(position, bool)
                or position < 1
                for position in row
            ):
                raise ValueError("descent-set positions must be positive integers.")
            if tuple(sorted(set(row))) != row:
                raise ValueError(
                    "Each descent set must be increasing without duplicates."
                )
        if len(set(normalized)) != len(normalized):
            raise ValueError("descent_sets must not contain duplicates.")
        object.__setattr__(self, "descent_sets", normalized)

    def _for_depth(self, depth: int) -> "WordSelection":
        """Discard descent sets that cannot occur at one homogeneous depth."""
        if self.descent_sets is None:
            return self
        return WordSelection(
            max_temporal_degree=self.max_temporal_degree,
            max_channel_descents=self.max_channel_descents,
            max_complexity=self.max_complexity,
            descent_sets=tuple(
                row for row in self.descent_sets if not row or row[-1] < depth
            ),
        )


@dataclass(frozen=True)
class ChronologicalFiltration:
    """Keep primitive functionals through one chronological degree."""

    max_degree: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_degree, int)
            or isinstance(self.max_degree, bool)
            or self.max_degree < 0
        ):
            raise ValueError("max_degree must be a non-negative integer.")


__all__ = ["ChronologicalFiltration", "WordSelection"]
