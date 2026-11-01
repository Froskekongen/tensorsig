"""Finite exact Hall--PBW--forest certificates for binary Pascal rank.

Implements F.52--F.61 of pascal-rank-foundations-and-solved-regimes.md.
This enumerates an exponential PBW basis and may suffer elimination fill-in.
It is an offline certificate, not a fast primitive chart or runtime backend.
No factorial Eulerian/permutation sum or floating rank appears in this route.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import comb, prod

from tensorsig.algebra.lyndon import lyndon_words

from ._lie_quotients.compiler import _standard_lyndon_expansion
from ._lie_quotients.enveloping import _pbw_levels
from .pascal_rank import _validate_parameters, binary_lie_weight_dimension

__all__ = ["PascalRankCertificate", "binary_pascal_rank_certificate"]


type SparseRow = tuple[tuple[int, Fraction], ...]


@dataclass(frozen=True)
class PascalRankCertificate:
    """Input-bound PBW--forest rows, a nonsingular minor, and a nullspace basis.

    These exact witnesses certify a homogeneous primitive value rank for a raw
    binary scalar Pascal code. They do not supply a runtime encoder or decoder.
    """

    degree: int
    window: int
    weight: int
    column_count: int
    rows: tuple[SparseRow, ...]
    pivot_rows: tuple[int, ...]
    pivot_columns: tuple[int, ...]
    kernel: tuple[SparseRow, ...]

    @property
    def matrix_rank(self) -> int:
        """Rank of the certificate's PBW--forest constraint matrix."""
        return len(self.pivot_columns)

    @property
    def defect(self) -> int:
        """Nullity of the constraint matrix, subtracted from the ambient Lie rank."""
        return self.column_count - self.matrix_rank

    @property
    def primitive_rank(self) -> int:
        """Primitive value rank after keeping ambient weights up to ``weight``."""
        return (
            sum(
                binary_lie_weight_dimension(self.degree, k)
                for k in range(self.weight + 1)
            )
            - self.defect
        )

    @property
    def nonzeros(self) -> int:
        """Number of stored nonzero coefficients in the defining matrix."""
        return sum(map(len, self.rows))

    def verify(self, *, max_degree: int = 10, max_columns: int = 4096) -> bool:
        """Bind F.60 rows to the input, then replay the minor/nullspace witnesses."""
        try:
            _validate_parameters(self.degree, self.window, self.weight)
        except TypeError, ValueError:
            return False
        if (
            type(max_degree) is not int
            or type(max_columns) is not int
            or self.degree > max_degree
            or max_degree < 1
            or max_columns < 1
            or type(self.column_count) is not int
            or self.column_count
            != sum(comb(self.degree, s) for s in range(self.window, self.weight + 1))
            or self.column_count > max_columns
        ):
            return False
        if any(
            type(value) is not tuple
            for value in (self.rows, self.pivot_rows, self.pivot_columns, self.kernel)
        ):
            return False
        if (
            len(self.pivot_rows) != len(self.pivot_columns)
            or any(
                type(row) is not int or not 0 <= row < len(self.rows)
                for row in self.pivot_rows
            )
            or any(
                type(col) is not int or not 0 <= col < self.column_count
                for col in self.pivot_columns
            )
            or len(set(self.pivot_rows)) != len(self.pivot_rows)
            or len(set(self.pivot_columns)) != len(self.pivot_columns)
        ):
            return False
        for entries in (*self.rows, *self.kernel):
            if type(entries) is not tuple or any(
                type(pair) is not tuple or len(pair) != 2 for pair in entries
            ):
                return False
            if any(
                type(col) is not int
                or not 0 <= col < self.column_count
                or type(value) not in (int, Fraction)
                or not value
                for col, value in entries
            ):
                return False
            if len({col for col, _ in entries}) != len(entries):
                return False
        expected_rows = (
            _certificate_rows(self.degree, self.window, self.weight)[0]
            if self.column_count
            else ()
        )
        if self.rows != expected_rows:
            return False
        if len(self.kernel) != self.column_count - self.matrix_rank:
            return False
        free = sorted(set(range(self.column_count)) - set(self.pivot_columns))
        vectors = [dict(vector) for vector in self.kernel]
        if any(
            vector.get(column, 0) != int(i == j)
            for i, vector in enumerate(vectors)
            for j, column in enumerate(free)
        ):
            return False
        if any(
            sum(value * vector.get(column, 0) for column, value in row)
            for row in self.rows
            for vector in vectors
        ):
            return False
        minor = [
            [
                Fraction(dict(self.rows[row]).get(column, 0))
                for column in self.pivot_columns
            ]
            for row in self.pivot_rows
        ]
        # Independent dense elimination of the retained square minor proves >=rank.
        for pivot in range(len(minor)):
            chosen = next(
                (row for row in range(pivot, len(minor)) if minor[row][pivot]), None
            )
            if chosen is None:
                return False
            minor[pivot], minor[chosen] = minor[chosen], minor[pivot]
            for row in range(pivot + 1, len(minor)):
                scale = minor[row][pivot] / minor[pivot][pivot]
                if not scale:
                    continue
                for col in range(pivot, len(minor)):
                    minor[row][col] -= scale * minor[pivot][col]
        return True


def _eliminate(rows: tuple[SparseRow, ...], columns: int):
    reducers = {}
    chosen = []
    pivot_order = []
    for index, entries in enumerate(rows):
        row = dict(entries)
        for pivot, reducer in sorted(reducers.items()):
            scale = row.get(pivot, 0)
            if not scale:
                continue
            for column, coefficient in reducer.items():
                updated = row.get(column, Fraction()) - scale * coefficient
                if updated:
                    row[column] = updated
                else:
                    row.pop(column, None)
        if row:
            pivot = min(row)
            scale = row[pivot]
            reducers[pivot] = {column: value / scale for column, value in row.items()}
            chosen.append(index)
            pivot_order.append(pivot)
    kernel = []
    for free in sorted(set(range(columns)) - reducers.keys()):
        vector = {free: Fraction(1)}
        for pivot, reducer in sorted(reducers.items(), reverse=True):
            value = -sum(
                coefficient * vector.get(column, 0)
                for column, coefficient in reducer.items()
                if column != pivot
            )
            if value:
                vector[pivot] = value
        kernel.append(tuple(sorted(vector.items())))
    return tuple(chosen), tuple(pivot_order), tuple(kernel)


@lru_cache(maxsize=4096)
def _forest_column(
    high: tuple[int, ...], low_weight: int
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Column of D_{s,k}: a signed forest sum with its first root retained."""
    size = len(high)
    if not 0 <= low_weight <= size:
        return ()
    if not high:
        return (((), 1),) if low_weight == 0 else ()
    if low_weight == 0:
        return ()
    result = []
    for rest in combinations(range(1, size), low_weight - 1):
        indices = (0, *rest)
        coefficient = (-1) ** (size - low_weight) * prod(
            i for i in range(1, size) if i not in rest
        )
        result.append((tuple(high[i] for i in indices), coefficient))
    return tuple(result)


def _certificate_rows(degree: int, window: int, weight: int):
    hall = tuple(lyndon_words(2, degree))
    depths = tuple(map(len, hall))
    weights = tuple(word.count(2) for word in hall)
    expansions = tuple(_standard_lyndon_expansion(word) for word in hall)

    @lru_cache(maxsize=None)
    def symmetric(factors: tuple[int, ...]):
        if not factors:
            return (((), Fraction(1)),)
        result = {}
        # F.52 groups by final factor, retaining multiplicities of repeated factors.
        for index, multiplicity in Counter(factors).items():
            remaining = list(factors)
            remaining.remove(index)
            scale = Fraction(multiplicity, len(factors))
            for left, a in symmetric(tuple(remaining)):
                for right, b in expansions[index]:
                    word = left + right
                    result[word] = result.get(word, Fraction()) + scale * a * b
        return tuple(
            (word, coefficient)
            for word, coefficient in sorted(result.items())
            if coefficient
        )

    columns = tuple(
        (size, channel, subset)
        for size in range(window, weight + 1)
        for channel, count in (("x", size), ("y", size - 1))
        for subset in combinations(range(degree - 1), count)
    )
    rows = []
    for factors in _pbw_levels(depths, degree)[degree]:
        k = sum(weights[index] for index in factors)
        if len(factors) < 2 or k > weight:
            continue
        residuals = {1: {}, 2: {}}
        for word, coefficient in symmetric(factors):
            subset = tuple(i for i, letter in enumerate(word[:-1]) if letter == 2)
            residuals[word[-1]][subset] = coefficient

        def pair(high, low_weight, letter):
            return sum(
                coefficient * residuals[letter].get(low, 0)
                for low, coefficient in _forest_column(high, low_weight)
            )

        entries = []
        for column, (size, channel, subset) in enumerate(columns):
            value = (
                pair(subset, k, 1)
                if channel == "x"
                else pair(subset, k - 1, 2) - (size - 1) * pair(subset, k, 1)
            )
            if value:
                entries.append((column, Fraction(value)))
        rows.append(tuple(entries))
    return tuple(rows), len(columns)


def binary_pascal_rank_certificate(
    degree: int,
    window: int,
    weight: int,
    *,
    max_degree: int = 10,
    max_columns: int = 4096,
) -> PascalRankCertificate:
    """Compute a finite F.60 certificate with explicit exponential-work limits."""
    _validate_parameters(degree, window, weight)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in (max_degree, max_columns)
    ):
        raise ValueError("max_degree and max_columns must be positive integers.")
    columns = sum(comb(degree, s) for s in range(window, weight + 1))
    if degree > max_degree or columns > max_columns:
        raise ValueError(
            "Finite PBW certificate exceeds the declared degree/column budget."
        )
    rows, actual_columns = (
        _certificate_rows(degree, window, weight) if columns else ((), 0)
    )
    if actual_columns != columns:
        raise RuntimeError("PBW--forest column count disagrees with F.61.")
    pivot_rows, pivot_columns, kernel = _eliminate(rows, columns)
    return PascalRankCertificate(
        degree, window, weight, columns, rows, pivot_rows, pivot_columns, kernel
    )
