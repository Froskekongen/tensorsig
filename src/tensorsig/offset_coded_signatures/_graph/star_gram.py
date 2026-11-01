"""ST3/ST4 exact fixed-centre leaf-gap Gram contraction.

Skeletons retain shared simplex variables until leaf channels are contracted.
No Cartesian product of channel word spaces or occurrence trees is formed.
The result is a normalized trace-coordinate norm, not literal tree energy or
the one-centre endpoint-coordinate norm.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb, factorial, prod

from ._graph_common import ExactWork, validate_cell


@dataclass(frozen=True, slots=True)
class _Skeleton:
    slots: tuple[tuple[int, int, int], ...]  # offset, gap, simplex symbol (-1=1)
    simplexes: tuple[tuple[int, ...], ...]
    symbols: int
    weight: Fraction


def _multiset_words(counts):
    counts = list(counts)
    word = []

    def visit():
        if not any(counts):
            yield tuple(word)
        else:
            for i, count in enumerate(counts):
                if count:
                    counts[i] -= 1
                    word.append(i)
                    yield from visit()
                    word.pop()
                    counts[i] += 1

    yield from visit()


def _bounded_tuples(width, maximum):
    if width == 0:
        yield ()
    else:
        for first in range(maximum + 1):
            for rest in _bounded_tuples(width - 1, maximum - first):
                yield (first, *rest)


def _skeletons(word, offsets, leaf_degree, work):
    p = len(word)
    for mask in range(1 << (p - 1)):
        cuts = [0, *(i + 1 for i in range(p - 1) if mask >> i & 1), p]
        chunks = tuple(word[a:b] for a, b in zip(cuts, cuts[1:]))
        if any(tuple(sorted(chunk)) != chunk for chunk in chunks):
            continue
        count = len(chunks)
        for powers in _bounded_tuples(count + 1, leaf_degree):
            work.step()
            total = sum(powers)
            weight = Fraction(
                (-1) ** (count + total + 1) * comb(leaf_degree + count, total + count),
                total + count,
            )
            slots = []
            simplexes = []
            gap = symbols = 0
            for index in range(count + 1):
                for _ in range(powers[index]):
                    slots.extend((offset, gap, -1) for offset in range(offsets))
                if index == count:
                    break
                centres = Counter(chunks[index])
                for offset in range(offsets):
                    r = centres[offset]
                    if not r:
                        slots.append((offset, gap, -1))
                        continue
                    group = tuple(range(symbols, symbols + r + 1))
                    symbols += r + 1
                    simplexes.append(group)
                    for j, symbol in enumerate(group):
                        slots.append((offset, gap, symbol))
                        if j < r:
                            gap += 1
            yield _Skeleton(tuple(slots), tuple(simplexes), symbols, weight)


def _add(table, key, value, work):
    if value:
        updated = table.get(key, Fraction()) + value
        if updated:
            if key not in table:
                work.acquire()
            table[key] = updated
        else:
            del table[key]
            work.release(1)


def _leaf_map(skeleton, occupancy, centres, work):
    empty = ((),) * (centres + 1)
    # Counts are redundant but avoid rescanning partial words at every slot.
    zero_counts = (0,) * len(occupancy)
    zero_exponents = (0,) * skeleton.symbols
    states = {}
    _add(states, (zero_counts, empty, zero_exponents), Fraction(1), work)
    for offset, gap, symbol in skeleton.slots:
        if not occupancy[offset]:
            continue
        updated = {}
        for (counts, words, powers), coefficient in states.items():
            for amount in range(occupancy[offset] - counts[offset] + 1):
                work.step()
                new_counts = list(counts)
                new_counts[offset] += amount
                new_words = list(words)
                new_words[gap] += (offset,) * amount
                new_powers = list(powers)
                if symbol >= 0:
                    new_powers[symbol] += amount
                key = tuple(new_counts), tuple(new_words), tuple(new_powers)
                _add(updated, key, coefficient / factorial(amount), work)
        work.discard(states)
        states = updated
    result = {}
    for (counts, words, powers), coefficient in states.items():
        if counts == occupancy:
            _add(result.setdefault(words, {}), powers, coefficient, work)
    work.discard(states)
    return result


def _match(left, right, work):
    result = {}
    for word in left.keys() & right.keys():
        for a, x in left[word].items():
            for b, y in right[word].items():
                work.step()
                _add(result, a + b, x * y, work)
    return result


def _multiply(left, right, work):
    result = {}
    for a, x in left.items():
        for b, y in right.items():
            work.step()
            key = tuple(i + j for i, j in zip(a, b, strict=True))
            _add(result, key, x * y, work)
    return result


def _simplex_moment(skeleton, exponents):
    return prod(
        (
            Fraction(
                prod(factorial(exponents[i]) for i in group),
                factorial(len(group) - 1 + sum(exponents[i] for i in group)),
            )
            for group in skeleton.simplexes
        ),
        start=Fraction(1),
    )


def star_trace_gram_energy(
    composition,
    occurrence_channels,
    graph,
    *,
    centre_channel,
    max_entries=100_000,
    max_terms=1_000_000,
):
    """Return ST4's exact normalized chronological-log trace squared norm.

    There may be any positive number of occurrences of the centre channel.
    Distinct active leaf channels commute and every leaf depends on the
    centre. Temporal letters within a channel remain free. Multiply by
    ``prod(factorial(n_offset_channel))**2`` for the source-image trace norm.
    ResourceLimit signals refusal, never zero. Leafless boundary controls
    are also accepted by the same finite logarithm identity.
    ``max_entries`` counts query-live sparse states, retained leaf maps and
    skeleton weights, and simultaneous contraction input/output polynomials.
    It is not a bound on Python object bytes or immutable scalar caches.
    """
    parts, channels = validate_cell(composition, occurrence_channels, graph)
    if type(centre_channel) is not int or centre_channel < 0:
        raise ValueError("centre_channel must be a non-negative integer.")
    work = ExactWork(max_entries, max_terms)
    active = set(channels)
    if centre_channel not in active:
        raise ValueError("the cell must contain at least one centre occurrence.")
    leaves = sorted(active - {centre_channel})
    if any(tuple(sorted((centre_channel, leaf))) not in graph.edges for leaf in leaves):
        raise ValueError("the centre must depend on every active leaf channel.")
    if any(edge in graph.edges for edge in combinations(leaves, 2)):
        raise ValueError("distinct active leaf channels must commute.")
    blocks = tuple(i for i, count in enumerate(parts) for _ in range(count))
    counts = {c: [0] * len(parts) for c in active}
    for block, channel in zip(blocks, channels, strict=True):
        counts[channel][block] += 1
    centre_counts = tuple(counts[centre_channel])
    p = sum(centre_counts)
    leaf_degree = len(channels) - p
    occupancies = tuple(tuple(counts[c]) for c in leaves)
    result = Fraction()
    for word in _multiset_words(centre_counts):
        # Only equal centre words can produce equal traces. Discard each word's
        # cache after contraction; the channel word spaces are never multiplied.
        rows = []
        for skeleton in _skeletons(word, len(parts), leaf_degree, work):
            work.acquire()  # Retained skeleton weight; maps own their entries.
            local = tuple(_leaf_map(skeleton, occ, p, work) for occ in occupancies)
            rows.append((skeleton, local))
        for left_index, (left, left_maps) in enumerate(rows):
            for right_index in range(left_index, len(rows)):
                work.step()
                right, right_maps = rows[right_index]
                polynomial = {}
                _add(
                    polynomial, (0,) * (left.symbols + right.symbols), Fraction(1), work
                )
                for a, b in zip(left_maps, right_maps, strict=True):
                    matched = _match(a, b, work)
                    multiplied = _multiply(polynomial, matched, work)
                    work.discard(polynomial)
                    work.discard(matched)
                    polynomial = multiplied
                    if not polynomial:
                        break
                paired = sum(
                    (
                        coefficient
                        * _simplex_moment(left, powers[: left.symbols])
                        * _simplex_moment(right, powers[left.symbols :])
                        for powers, coefficient in polynomial.items()
                    ),
                    Fraction(),
                )
                multiplicity = 1 if left_index == right_index else 2
                result += multiplicity * left.weight * right.weight * paired
                work.discard(polynomial)
        for _, local in rows:
            work.release(1)  # Skeleton weight.
            for row in local:
                for polynomial in row.values():
                    work.discard(polynomial)
        rows.clear()
        del left, right, skeleton, _
    if result < 0:
        raise ArithmeticError("an exact trace Gram norm cannot be negative.")
    return result


__all__ = ["star_trace_gram_energy"]
