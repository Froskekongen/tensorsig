from fractions import Fraction
from itertools import product
from math import comb, factorial

import pytest

from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    cell_survives,
)
from tensorsig.offset_coded_signatures._graph.one_centre_gram import (
    one_centre_gram_cell_survives,
    one_centre_gram_energy,
)
from tensorsig.offset_coded_signatures.quotients import DependenceGraph

type _Counts = tuple[int, ...]
type _Word = tuple[int, ...]
type _Series = dict[tuple[_Counts, _Word], Fraction]


def _compositions(total: int) -> tuple[tuple[int, ...], ...]:
    if total == 0:
        return ((),)
    return tuple(
        (first, *rest)
        for first in range(1, total + 1)
        for rest in _compositions(total - first)
    )


def _multiply_series(left: _Series, right: _Series, target: _Counts) -> _Series:
    output: _Series = {}
    for (left_counts, left_word), left_coefficient in left.items():
        for (right_counts, right_word), right_coefficient in right.items():
            counts = tuple(
                a + b for a, b in zip(left_counts, right_counts, strict=True)
            )
            if any(value > limit for value, limit in zip(counts, target, strict=True)):
                continue
            key = (counts, left_word + right_word)
            output[key] = (
                output.get(key, Fraction()) + left_coefficient * right_coefficient
            )
    return output


def _literal_endpoint_vector(
    composition: tuple[int, ...],
    channels: tuple[int, ...],
    centre_channel: int,
) -> dict[tuple[tuple[int, ...], ...], Fraction]:
    """Expand ``log(g)`` literally, then apply the terminal-centre map."""
    block_by_vertex = tuple(
        block for block, part in enumerate(composition) for _ in range(part)
    )
    centre_vertex = channels.index(centre_channel)
    leaf_types = tuple(
        sorted(
            {
                (block, channel)
                for block, channel in zip(block_by_vertex, channels, strict=True)
                if channel != centre_channel
            }
        )
    )
    letter_types = (*leaf_types, (block_by_vertex[centre_vertex], centre_channel))
    centre_letter = len(letter_types) - 1
    target = tuple(
        sum(
            block == vertex_block and channel == vertex_channel
            for vertex_block, vertex_channel in zip(
                block_by_vertex, channels, strict=True
            )
        )
        for block, channel in letter_types
    )
    zero = (0,) * len(target)

    group: _Series = {(zero, ()): Fraction(1)}
    for block, part in enumerate(composition):
        letters = tuple(
            index
            for index, (letter_block, _) in enumerate(letter_types)
            if letter_block == block
        )
        exponential: _Series = {(zero, ()): Fraction(1)}
        for length in range(1, part + 1):
            for word in product(letters, repeat=length):
                counts = tuple(word.count(letter) for letter in range(len(target)))
                if any(
                    value > limit for value, limit in zip(counts, target, strict=True)
                ):
                    continue
                exponential[(counts, word)] = Fraction(1, factorial(length))
        group = _multiply_series(group, exponential, target)

    augmentation = dict(group)
    del augmentation[(zero, ())]
    logarithm: _Series = {}
    power: _Series = {(zero, ()): Fraction(1)}
    for exponent in range(1, sum(composition) + 1):
        power = _multiply_series(power, augmentation, target)
        scale = Fraction((-1) ** (exponent + 1), exponent)
        for key, coefficient in power.items():
            logarithm[key] = logarithm.get(key, Fraction()) + scale * coefficient

    leaf_channels = tuple(sorted(set(channels) - {centre_channel}))
    endpoint: dict[tuple[tuple[int, ...], ...], Fraction] = {}
    for (counts, word), coefficient in logarithm.items():
        if counts != target or not word or word[-1] != centre_letter:
            continue
        coordinate = tuple(
            tuple(
                letter_types[letter][0]
                for letter in word[:-1]
                if letter_types[letter][1] == leaf_channel
            )
            for leaf_channel in leaf_channels
        )
        endpoint[coordinate] = endpoint.get(coordinate, Fraction()) + coefficient
    return {coordinate: value for coordinate, value in endpoint.items() if value}


def _bernoulli_numbers(maximum_degree: int) -> tuple[Fraction, ...]:
    values = [Fraction(1)]
    for degree in range(1, maximum_degree + 1):
        values.append(
            -sum(
                (
                    Fraction(comb(degree + 1, index)) * values[index]
                    for index in range(degree)
                ),
                start=Fraction(),
            )
            / (degree + 1)
        )
    return tuple(values)


def _strict_leaf_bernoulli_energy(
    leaf_degree: int,
    leaves_before_centre_block: int,
    leaves_in_centre_block: int,
) -> Fraction:
    bernoulli = _bernoulli_numbers(leaf_degree)
    coefficient = sum(
        (
            Fraction(
                comb(leaves_before_centre_block, before)
                * comb(leaves_in_centre_block, same),
                same + 1,
            )
            * bernoulli[leaf_degree - before - same]
            for before in range(leaves_before_centre_block + 1)
            for same in range(leaves_in_centre_block + 1)
        ),
        start=Fraction(),
    )
    return coefficient * coefficient


def test_one_channel_augmentation_counterexample_has_positive_energy() -> None:
    graph = DependenceGraph(((0, 1),))

    assert one_centre_gram_energy(
        (1, 1, 2),
        (0, 1, 1, 1),
        graph,
        centre_channel=0,
    ) == Fraction(1, 72)
    assert one_centre_gram_cell_survives(
        (1, 1, 2),
        (0, 1, 1, 1),
        graph,
        centre_channel=0,
    )


def test_gram_energy_matches_literal_endpoint_oracle_through_degree_four() -> None:
    graph = DependenceGraph(((0, 1), (0, 2)))
    checks = 0
    for degree in range(2, 5):
        for composition in _compositions(degree):
            for centre_vertex in range(degree):
                for leaves in product((1, 2), repeat=degree - 1):
                    channels = list(leaves)
                    channels.insert(centre_vertex, 0)
                    channels = tuple(channels)
                    energy = one_centre_gram_energy(
                        composition,
                        channels,
                        graph,
                        centre_channel=0,
                    )
                    endpoint = _literal_endpoint_vector(composition, channels, 0)
                    assert energy == sum(
                        (
                            coefficient * coefficient
                            for coefficient in endpoint.values()
                        ),
                        start=Fraction(),
                    )
                    assert bool(energy) == cell_survives(composition, channels, graph)
                    checks += 1
    assert checks == 312


@pytest.mark.parametrize(
    ("composition", "centre_vertex"),
    [
        ((1, 1), 0),
        ((1, 1, 1), 1),
        ((2, 2), 1),
        ((1, 2, 2), 2),
        ((2, 1, 3), 4),
    ],
)
def test_strict_leaf_energy_matches_bernoulli_specialization(
    composition: tuple[int, ...],
    centre_vertex: int,
) -> None:
    degree = sum(composition)
    centre_channel = centre_vertex
    channels = tuple(range(degree))
    graph = DependenceGraph(
        (centre_channel, leaf) for leaf in channels if leaf != centre_channel
    )
    boundary = 0
    for part in composition:
        if centre_vertex < boundary + part:
            leaves_before = boundary
            leaves_same = part - 1
            break
        boundary += part
    else:
        raise AssertionError("centre vertex must belong to a composition block")

    assert one_centre_gram_energy(
        composition,
        channels,
        graph,
        centre_channel=centre_channel,
    ) == _strict_leaf_bernoulli_energy(
        degree - 1,
        leaves_before,
        leaves_same,
    )


@pytest.mark.parametrize(
    ("channels", "graph", "message"),
    [
        ((1, 1), DependenceGraph(((0, 1),)), "exactly one centre"),
        ((0, 0), DependenceGraph(((0, 1),)), "exactly one centre"),
        ((0, 1), DependenceGraph(), "depend on every active leaf"),
        (
            (0, 1, 2),
            DependenceGraph(((0, 1), (0, 2), (1, 2))),
            "leaf channels must commute",
        ),
    ],
)
def test_invalid_one_centre_star_cells_are_rejected(
    channels: tuple[int, ...],
    graph: DependenceGraph,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        one_centre_gram_energy(
            (1,) * len(channels),
            channels,
            graph,
            centre_channel=0,
        )


def test_public_input_types_are_validated() -> None:
    with pytest.raises(TypeError, match="DependenceGraph"):
        one_centre_gram_energy((1, 1), (0, 1), object(), centre_channel=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="centre_channel"):
        one_centre_gram_energy(
            (1, 1),
            (0, 1),
            DependenceGraph(((0, 1),)),
            centre_channel=True,  # type: ignore[arg-type]
        )
