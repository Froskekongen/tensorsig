"""Exact one-centre star survival via contracted leaf-channel Gram tables."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations, product
from math import comb, factorial, prod

from ..quotients import DependenceGraph
from ._graph_common import validate_channels as _validate_occurrence_channels
from ._graph_common import validate_composition as _validate_composition

type _Occupancy = tuple[int, ...]
type _Word = tuple[int, ...]
type _WordMap = dict[_Word, Fraction]
type _GramTotals = tuple[int, int, int, int]
type _LeafRow = tuple[int, int, _WordMap]


def _prepare_star_cell(
    composition: Sequence[int],
    occurrence_channels: Sequence[int],
    graph: DependenceGraph,
    centre_channel: int,
) -> tuple[tuple[_Occupancy, ...], int, int]:
    if not isinstance(graph, DependenceGraph):
        raise TypeError("graph must be a DependenceGraph.")
    if not isinstance(centre_channel, int) or isinstance(centre_channel, bool):
        raise TypeError("centre_channel must be an integer.")
    if centre_channel < 0:
        raise ValueError("centre_channel must be non-negative.")

    parts = _validate_composition(composition)
    channels = _validate_occurrence_channels(occurrence_channels, sum(parts))
    centre_vertices = tuple(
        vertex for vertex, channel in enumerate(channels) if channel == centre_channel
    )
    if len(centre_vertices) != 1:
        raise ValueError("the cell must contain exactly one centre occurrence.")

    leaf_channels = tuple(sorted(set(channels) - {centre_channel}))
    for leaf in leaf_channels:
        edge = (min(centre_channel, leaf), max(centre_channel, leaf))
        if edge not in graph.edges:
            raise ValueError("the centre must depend on every active leaf channel.")
    if any(edge in graph.edges for edge in combinations(leaf_channels, 2)):
        raise ValueError("distinct active leaf channels must commute.")

    block_by_vertex = tuple(
        block for block, part in enumerate(parts) for _ in range(part)
    )
    centre_block = block_by_vertex[centre_vertices[0]]
    occupancies = []
    for leaf in leaf_channels:
        counts = [0] * len(parts)
        for block, channel in zip(block_by_vertex, channels, strict=True):
            if channel == leaf:
                counts[block] += 1
        occupancies.append(tuple(counts))
    return tuple(occupancies), centre_block, len(channels) - 1


def _ordered_word(counts: _Occupancy) -> _Word:
    return tuple(block for block, count in enumerate(counts) for _ in range(count))


def _power_word_maps(
    occupancy: _Occupancy,
    maximum_power: int,
) -> tuple[dict[_Occupancy, _WordMap], ...]:
    """Build the sparse maps ``P_j(nu)`` from recurrence (10.4)."""
    zero = (0,) * len(occupancy)
    tables: list[dict[_Occupancy, _WordMap]] = [{zero: {(): Fraction(1)}}]
    for _ in range(maximum_power):
        next_table: dict[_Occupancy, _WordMap] = {}
        for used, words in tables[-1].items():
            choices = product(
                *(range(limit - value + 1) for limit, value in zip(occupancy, used))
            )
            for increment in choices:
                target = tuple(
                    value + delta for value, delta in zip(used, increment, strict=True)
                )
                suffix = _ordered_word(increment)
                scale = Fraction(1, prod(factorial(value) for value in increment))
                output = next_table.setdefault(target, {})
                for word, coefficient in words.items():
                    key = word + suffix
                    output[key] = output.get(key, Fraction()) + coefficient * scale
        tables.append(next_table)
    return tuple(tables)


def _leaf_rows(
    occupancy: _Occupancy,
    centre_block: int,
    maximum_power: int,
) -> tuple[tuple[_LeafRow, ...], ...]:
    powers = _power_word_maps(occupancy, maximum_power)
    gammas = tuple(
        (*prefix, *((0,) * (len(occupancy) - centre_block - 1)))
        for prefix in product(
            *(range(occupancy[block] + 1) for block in range(centre_block + 1))
        )
    )
    rows = []
    for power in range(maximum_power + 1):
        power_rows = []
        for gamma in gammas:
            target = tuple(
                value - removed for value, removed in zip(occupancy, gamma, strict=True)
            )
            words = powers[power].get(target)
            if words is None:
                continue
            suffix = _ordered_word(gamma)
            scale = Fraction(1, prod(factorial(value) for value in gamma))
            power_rows.append(
                (
                    sum(gamma),
                    gamma[centre_block],
                    {
                        word + suffix: coefficient * scale
                        for word, coefficient in words.items()
                    },
                )
            )
        rows.append(tuple(power_rows))
    return tuple(rows)


def _word_inner_product(left: _WordMap, right: _WordMap) -> Fraction:
    if len(left) > len(right):
        left, right = right, left
    return sum(
        (
            coefficient * right.get(word, Fraction())
            for word, coefficient in left.items()
        ),
        start=Fraction(),
    )


def _local_gram(
    left: tuple[_LeafRow, ...], right: tuple[_LeafRow, ...]
) -> dict[_GramTotals, Fraction]:
    table: dict[_GramTotals, Fraction] = {}
    for left_degree, left_centre_degree, left_words in left:
        for right_degree, right_centre_degree, right_words in right:
            coefficient = _word_inner_product(left_words, right_words)
            if coefficient:
                key = (
                    left_degree,
                    left_centre_degree,
                    right_degree,
                    right_centre_degree,
                )
                table[key] = table.get(key, Fraction()) + coefficient
    return table


def _contract(
    accumulated: dict[_GramTotals, Fraction],
    local: dict[_GramTotals, Fraction],
) -> dict[_GramTotals, Fraction]:
    output: dict[_GramTotals, Fraction] = {}
    for totals, coefficient in accumulated.items():
        for increments, local_coefficient in local.items():
            key = tuple(
                value + increment
                for value, increment in zip(totals, increments, strict=True)
            )
            output[key] = output.get(key, Fraction()) + coefficient * local_coefficient
    return output


def _theta_table(maximum_degree: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            Fraction((-1) ** power)
            * sum(
                (
                    Fraction(comb(index, power), index + 1)
                    for index in range(power, degree + 1)
                ),
                start=Fraction(),
            )
            if power <= degree
            else Fraction()
            for power in range(maximum_degree + 1)
        )
        for degree in range(maximum_degree + 1)
    )


def one_centre_gram_energy(
    composition: Sequence[int],
    occurrence_channels: Sequence[int],
    graph: DependenceGraph,
    *,
    centre_channel: int,
) -> Fraction:
    """Return the exact rational endpoint-coordinate Gram energy.

    The active occurrence-channel graph must be a star with exactly one
    occurrence of ``centre_channel``. Distinct leaf channels must commute.
    The returned energy is zero exactly when the graph-block cell vanishes.
    """
    occupancies, centre_block, leaf_degree = _prepare_star_cell(
        composition,
        occurrence_channels,
        graph,
        centre_channel,
    )
    channel_rows = tuple(
        _leaf_rows(occupancy, centre_block, leaf_degree) for occupancy in occupancies
    )
    theta = _theta_table(leaf_degree)
    energy = Fraction()
    for left_power in range(leaf_degree + 1):
        for right_power in range(leaf_degree + 1):
            contracted: dict[_GramTotals, Fraction] = {(0, 0, 0, 0): Fraction(1)}
            for rows in channel_rows:
                local = _local_gram(rows[left_power], rows[right_power])
                contracted = _contract(contracted, local)
                if not contracted:
                    break
            for (
                left_degree,
                left_centre_degree,
                right_degree,
                right_centre_degree,
            ), coefficient in contracted.items():
                left_theta = theta[leaf_degree - left_degree][left_power]
                right_theta = theta[leaf_degree - right_degree][right_power]
                energy += (
                    coefficient
                    * left_theta
                    * right_theta
                    / ((left_centre_degree + 1) * (right_centre_degree + 1))
                )
    return energy


def one_centre_gram_cell_survives(
    composition: Sequence[int],
    occurrence_channels: Sequence[int],
    graph: DependenceGraph,
    *,
    centre_channel: int,
) -> bool:
    """Return exact survival for a validated one-centre active-star cell."""
    return bool(
        one_centre_gram_energy(
            composition,
            occurrence_channels,
            graph,
            centre_channel=centre_channel,
        )
    )


__all__ = ["one_centre_gram_cell_survives", "one_centre_gram_energy"]
