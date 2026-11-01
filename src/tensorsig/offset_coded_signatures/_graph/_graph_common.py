"""Shared exact graph validation and bounded work accounting."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import comb

from ..quotients import DependenceGraph


class ResourceLimit(RuntimeError):
    """An exact computation was refused; this is never a zero certificate."""


@dataclass(slots=True)
class ExactWork:
    """Bound query-live dynamic sparse entries and generated arithmetic terms.

    Acquire before storing an entry and release when its owning table is
    cleared. Aliases borrow the same entry. Immutable operator/Bernoulli caches,
    container metadata and transient scalars are outside this state budget.
    """

    max_entries: int = 100_000
    max_terms: int = 1_000_000
    terms: int = 0
    peak_entries: int = 0
    live_entries: int = 0

    def __post_init__(self) -> None:
        for value in (self.max_entries, self.max_terms):
            if type(value) is not int or value < 1:
                raise ValueError("exact-work limits must be positive integers.")

    def step(self, count: int = 1) -> None:
        self.terms += count
        if self.terms > self.max_terms:
            raise ResourceLimit(f"generated terms exceed {self.max_terms}.")

    def acquire(self, entries: int = 1) -> None:
        live = self.live_entries + entries
        if live > self.max_entries:
            raise ResourceLimit(f"sparse entries exceed {self.max_entries}.")
        self.live_entries = live
        self.peak_entries = max(self.peak_entries, live)

    def release(self, entries: int) -> None:
        self.live_entries -= entries
        if self.live_entries < 0:
            raise AssertionError("released more sparse entries than were acquired")

    def discard(self, table) -> None:
        """Release an owned coefficient table; aliases then see an empty table."""
        self.release(len(table))
        table.clear()


def validate_composition(composition: Sequence[int]) -> tuple[int, ...]:
    parts = tuple(composition)
    if not parts:
        raise ValueError("composition must not be empty.")
    if any(not isinstance(part, int) or isinstance(part, bool) for part in parts):
        raise TypeError("composition entries must be integers.")
    if any(part < 1 for part in parts):
        raise ValueError("composition entries must be positive.")
    if sum(parts) < 2:
        raise ValueError("intermediate-rank tree moments require degree at least two.")
    return parts


def validate_channels(channels: Sequence[int], degree: int) -> tuple[int, ...]:
    channels = tuple(channels)
    if len(channels) != degree:
        raise ValueError("occurrence_channels must contain one channel per vertex.")
    if any(not isinstance(c, int) or isinstance(c, bool) for c in channels):
        raise TypeError("occurrence channels must be integers.")
    if any(c < 0 for c in channels):
        raise ValueError("occurrence channels must be non-negative.")
    return channels


def validate_cell(composition, channels, graph):
    if not isinstance(graph, DependenceGraph):
        raise TypeError("graph must be a DependenceGraph.")
    parts = validate_composition(composition)
    return parts, validate_channels(channels, sum(parts))


def occurrence_adjacency(channels, graph):
    adjacency = [set() for _ in channels]
    for left, right in combinations(range(len(channels)), 2):
        edge = tuple(sorted((channels[left], channels[right])))
        if channels[left] == channels[right] or edge in graph.edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return tuple(frozenset(neighbors) for neighbors in adjacency)


def is_connected(adjacency) -> bool:
    seen = {0}
    todo = [0]
    while todo:
        for neighbor in adjacency[todo.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                todo.append(neighbor)
    return len(seen) == len(adjacency)


def perfect_elimination_order(adjacency) -> tuple[int, ...] | None:
    weights = [0] * len(adjacency)
    remaining = set(range(len(adjacency)))
    selected = []
    while remaining:
        vertex = max(remaining, key=lambda item: (weights[item], -item))
        remaining.remove(vertex)
        selected.append(vertex)
        for neighbor in adjacency[vertex] & remaining:
            weights[neighbor] += 1
    peo = tuple(reversed(selected))
    return peo if valid_peo(adjacency, peo) else None


def valid_peo(adjacency, peo) -> bool:
    if any(type(v) is not int for v in peo):
        return False
    if len(peo) != len(adjacency) or set(peo) != set(range(len(adjacency))):
        return False
    positions = {vertex: index for index, vertex in enumerate(peo)}
    for index, vertex in enumerate(peo):
        later = sorted(n for n in adjacency[vertex] if positions[n] > index)
        if any(b not in adjacency[a] for a, b in combinations(later, 2)):
            return False
    return True


@lru_cache(maxsize=128)
def bernoulli_at_one(maximum: int) -> tuple[Fraction, ...]:
    """LE.39: B_0(1), ..., B_maximum(1), including B_1(1)=+1/2."""
    values = [Fraction(1)]
    for n in range(1, maximum + 1):
        values.append(
            1
            - sum((comb(n + 1, j) * values[j] for j in range(n)), Fraction()) / (n + 1)
        )
    return tuple(values)


def signed_unicyclic_relation(unicyclic, degree):
    """Return A.1 coefficients in one canonical cyclic orientation."""
    adjacency = [set() for _ in range(degree)]
    for left, right in unicyclic:
        adjacency[left].add(right)
        adjacency[right].add(left)

    remaining_degrees = [len(neighbors) for neighbors in adjacency]
    leaves = [vertex for vertex, value in enumerate(remaining_degrees) if value == 1]
    while leaves:
        vertex = leaves.pop()
        if remaining_degrees[vertex] != 1:
            continue
        remaining_degrees[vertex] = 0
        for neighbor in adjacency[vertex]:
            if remaining_degrees[neighbor] == 0:
                continue
            remaining_degrees[neighbor] -= 1
            if remaining_degrees[neighbor] == 1:
                leaves.append(neighbor)

    cycle_vertices = {vertex for vertex, value in enumerate(remaining_degrees) if value}
    start = min(cycle_vertices)
    previous = start
    current = min(adjacency[start] & cycle_vertices)
    traversal = [(start, current)]
    while current != start:
        following = next(
            neighbor
            for neighbor in adjacency[current] & cycle_vertices
            if neighbor != previous
        )
        if following != start:
            traversal.append((current, following))
        else:
            traversal.append((current, start))
        previous, current = current, following

    return tuple(
        (
            frozenset(unicyclic - {(min(tail, head), max(tail, head))}),
            1 if tail < head else -1,
        )
        for tail, head in traversal
    )
