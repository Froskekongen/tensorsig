"""LE6: exact chordal all-tree energy by coefficient balances and wedges.

Messages use the PEO clique tree. Exterior elements are stored in ambient
vertex-wedge coordinates, a constant-factor enlargement of the anchored
LE3 interface; extension and contraction then need no change of basis.
Neither spanning-tree enumeration nor a connectivity-partition table is used.
"""

from fractions import Fraction
from functools import lru_cache
from math import comb

from ._graph_common import (
    ExactWork,
    bernoulli_at_one,
    is_connected,
    occurrence_adjacency,
    perfect_elimination_order,
    valid_peo,
    validate_cell,
)


@lru_cache(maxsize=128)
def _attachment_entries(degree, comparison):
    """Nonzero K[output,input] from LE.45, with the proved finite bounds."""
    bernoulli = bernoulli_at_one(degree - 2)
    rows = []
    for a in range(degree - 1):
        power_sum = {
            a + 1 - k: Fraction(comb(a + 1, k), a + 1) * bernoulli[k]
            for k in range(a + 1)
            if bernoulli[k]
        }
        if comparison >= 0:
            shifted = {}
            for power, value in power_sum.items():
                for lower in range(power + 1):
                    shifted[lower] = shifted.get(lower, Fraction()) + (
                        value * comb(power, lower) * (-1) ** (power - lower)
                    )
            power_sum = shifted
        for b in range(degree - 1 - a):
            output = {(power, 0): value / (b + 1) for power, value in power_sum.items()}
            if comparison == 0:
                output[(a, b + 1)] = output.get((a, b + 1), Fraction()) + Fraction(
                    1, b + 1
                )
            rows.extend(
                ((a, b), exponent, value) for exponent, value in output.items() if value
            )
    return tuple(rows)


def _wedge(left, right):
    if left & right:
        return 0
    inversions = 0
    bits = right
    while bits:
        low = bits & -bits
        inversions += (left // (low << 1)).bit_count()
        bits ^= low
    return -1 if inversions % 2 else 1


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


def _extend(table, source, target, work):
    indices = {v: i for i, v in enumerate(target)}
    result = {}
    for (bits, balances, left, right), value in table.items():
        extended = [0] * (4 * len(target))
        for i, v in enumerate(source):
            j = indices[v]
            extended[4 * j : 4 * j + 4] = balances[4 * i : 4 * i + 4]
        _add(result, (bits, tuple(extended), left, right), value, work)
    return result


def _join(left_table, right_table, bound, work):
    output = {}
    for (p, a, x, y), first in left_table.items():
        for (q, b, z, w), second in right_table.items():
            work.step()
            if p & q:
                continue
            sign = _wedge(x, z) * _wedge(y, w)
            if not sign:
                continue
            balances = tuple(i + j for i, j in zip(a, b, strict=True))
            if any(abs(value) > bound for value in balances):
                continue
            _add(output, (p | q, balances, x | z, y | w), sign * first * second, work)
    return output


def _edge(table, bag, u, v, colours, root, degree, work, final_vertex=None):
    index = {vertex: 4 * i for i, vertex in enumerate(bag)}
    bound = degree - 1
    output = {}

    def accepted(bits, balances):
        if final_vertex is None:
            return True
        offset = index[final_vertex]
        return bits & (1 << final_vertex) and not any(balances[offset : offset + 4])

    operators = {}
    for child, parent in ((u, v), (v, u)):
        comparison = (colours[child] > colours[parent]) - (
            colours[child] < colours[parent]
        )
        entries = _attachment_entries(degree, comparison)
        by_input, by_output = {}, {}
        for alpha, gamma, coefficient in entries:
            by_input.setdefault(alpha, []).append((alpha, gamma, coefficient))
            by_output.setdefault(gamma, []).append((alpha, gamma, coefficient))
        operators[child] = entries, by_input, by_output

    for (bits, balances, left, right), coefficient in table.items():
        work.step()
        if accepted(bits, balances):
            _add(output, (bits, balances, left, right), coefficient, work)
        # Multiplication by (e_u-e_v) in each exterior copy.
        exterior = []
        for a, sa in ((u, 1), (v, -1)):
            for b, sb in ((u, 1), (v, -1)):
                sign = sa * sb * _wedge(left, 1 << a) * _wedge(right, 1 << b)
                if sign:
                    exterior.append((left | (1 << a), right | (1 << b), sign))
        if not exterior:
            continue
        for child, parent in ((u, v), (v, u)):
            if child == root or bits & (1 << child):
                continue
            entries, by_input, by_output = operators[child]
            first = second = entries
            if final_vertex is not None:
                at = index[final_vertex]
                if child == final_vertex:
                    first = by_input.get(balances[at : at + 2], ())
                    second = by_input.get(balances[at + 2 : at + 4], ())
                elif not bits & (1 << final_vertex):
                    continue
                else:
                    first = by_output.get(tuple(-x for x in balances[at : at + 2]), ())
                    second = by_output.get(
                        tuple(-x for x in balances[at + 2 : at + 4]), ()
                    )
            for alpha, gamma, a in first:
                for beta, delta, b in second:
                    work.step()
                    updated = list(balances)
                    for replica, (source, target) in enumerate(
                        ((alpha, gamma), (beta, delta))
                    ):
                        for axis in range(2):
                            updated[index[child] + 2 * replica + axis] -= source[axis]
                            updated[index[parent] + 2 * replica + axis] += target[axis]
                    if any(abs(x) > bound for x in updated):
                        continue
                    parent_bits = bits | (1 << child)
                    if not accepted(parent_bits, updated):
                        continue
                    for x, y, sign in exterior:
                        _add(
                            output,
                            (parent_bits, tuple(updated), x, y),
                            coefficient * a * b * sign,
                            work,
                        )
    return output


def _forget(table, bag, vertex, work):
    position = bag.index(vertex)
    bit = 1 << vertex
    output = {}
    for (bits, balances, left, right), value in table.items():
        work.step()
        if not bits & bit or any(balances[4 * position : 4 * position + 4]):
            continue
        if not left & bit or not right & bit:
            continue
        sign = (-1) ** (
            (left & (bit - 1)).bit_count() + (right & (bit - 1)).bit_count()
        )
        reduced = balances[: 4 * position] + balances[4 * position + 4 :]
        _add(output, (bits ^ bit, reduced, left ^ bit, right ^ bit), sign * value, work)
    return output


def literal_tree_energy(
    composition,
    occurrence_channels,
    graph,
    *,
    peo=None,
    max_entries=100_000,
    max_terms=1_000_000,
):
    """Return sum_T mu_q(T)^2 exactly, for a chordal occurrence graph.

    This is LE6's coefficient circuit, with two independent coefficient
    replicas on one common forest. ``peo`` may supply any valid occurrence
    PEO; otherwise deterministic MCS finds one. Disconnected graphs give zero.
    Nonchordal inputs are rejected. ResourceLimit is an inconclusive refusal.
    ``max_entries`` bounds all query-live dynamic sparse state entries together,
    including pending messages and simultaneous input/output tables. Immutable
    attachment/Bernoulli caches and Python object bytes are outside that budget.
    """
    parts, channels = validate_cell(composition, occurrence_channels, graph)
    work = ExactWork(max_entries, max_terms)
    adjacency = occurrence_adjacency(channels, graph)
    if peo is not None:
        peo = tuple(peo)
        if not valid_peo(adjacency, peo):
            raise ValueError(
                "peo must be a valid occurrence perfect elimination order."
            )
    else:
        peo = perfect_elimination_order(adjacency)
        if peo is None:
            raise ValueError("the occurrence dependence graph must be chordal.")
    if not is_connected(adjacency):
        return Fraction()
    degree = len(channels)
    root = peo[-1]
    colours = tuple(i for i, count in enumerate(parts) for _ in range(count))
    positions = {v: i for i, v in enumerate(peo)}
    children = {v: [] for v in peo}
    later = {
        v: tuple(
            sorted(
                (w for w in adjacency[v] if positions[w] > positions[v]),
                key=positions.get,
            )
        )
        for v in peo
    }
    for v in peo[:-1]:
        children[later[v][0]].append(v)
    messages = {}
    for vertex in peo:
        bag = tuple(sorted((vertex, *later[vertex])))
        table = {}
        _add(table, (0, (0,) * (4 * len(bag)), 0, 0), Fraction(1), work)
        for child in children[vertex]:
            child_bag, child_table = messages.pop(child)
            extended = _extend(child_table, child_bag, bag, work)
            work.discard(child_table)
            joined = _join(table, extended, degree - 1, work)
            work.discard(table)
            work.discard(extended)
            table = joined
        for i, neighbor in enumerate(later[vertex]):
            updated = _edge(
                table,
                bag,
                vertex,
                neighbor,
                colours,
                root,
                degree,
                work,
                final_vertex=vertex if i + 1 == len(later[vertex]) else None,
            )
            work.discard(table)
            table = updated
        if vertex != root:
            forgotten = _forget(table, bag, vertex, work)
            work.discard(table)
            messages[vertex] = (
                tuple(v for v in bag if v != vertex),
                forgotten,
            )
    bernoulli = bernoulli_at_one(degree - 1)
    energy = Fraction()
    for (bits, balances, left, right), coefficient in table.items():
        a, b, c, d = balances
        if bits or left or right or min(balances) < 0 or max(a + b, c + d) >= degree:
            continue
        energy += coefficient * bernoulli[a] * bernoulli[c] / ((b + 1) * (d + 1))
    work.discard(table)
    return (4 ** (degree - 1)) * energy


__all__ = ["literal_tree_energy"]
