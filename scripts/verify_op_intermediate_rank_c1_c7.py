#!/usr/bin/env python3
"""Exact bounded verification of OP-INTERMEDIATE-RANK (C.1)--(C.7).

This is evidence, not the all-degree proof.  It deliberately uses tensorsig's
production Eulerian schedule and trace normal form, while independently
building the source cells, polarization fibres, Walsh moments, occurrence
trees, and channel-witness antichains with exact ``Fraction`` arithmetic.

Run from the tensorsig repository root:

    uv run python scripts/verify_op_intermediate_rank_c1_c7.py
"""

from collections import defaultdict
from fractions import Fraction
from functools import cache
from itertools import combinations, permutations, product
from math import comb, factorial, prod

from tensorsig.offset_coded_signatures._lie_quotients.enveloping import (
    _eulerian_schedule,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import OffsetAlphabet
from tensorsig.offset_coded_signatures._lie_quotients.trace_compiler import (
    _TraceMonoid,
)
from tensorsig.offset_coded_signatures.offset_patterns import (
    log_signature_offset_coefficient_fraction,
)


MAX_DEGREE = 4
AMPLITUDE_MAX_DEGREE = 6


def add(target, key, value):
    value += target.get(key, Fraction())
    if value:
        target[key] = value
    else:
        target.pop(key, None)


@cache
def edges(vertices):
    return tuple(combinations(range(vertices), 2))


def edge_mask(vertices, selected):
    positions = {edge: bit for bit, edge in enumerate(edges(vertices))}
    return sum(1 << positions[tuple(sorted(edge))] for edge in selected)


def connected(vertices, mask):
    if vertices == 1:
        return True
    adjacency = [[] for _ in range(vertices)]
    for bit, (left, right) in enumerate(edges(vertices)):
        if mask >> bit & 1:
            adjacency[left].append(right)
            adjacency[right].append(left)
    seen, todo = {0}, [0]
    while todo:
        for neighbour in adjacency[todo.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                todo.append(neighbour)
    return len(seen) == vertices


@cache
def spanning_trees(vertices):
    return tuple(
        mask
        for mask in range(1 << len(edges(vertices)))
        if mask.bit_count() == vertices - 1 and connected(vertices, mask)
    )


def compositions(total):
    if total == 0:
        yield ()
        return
    for first in range(1, total + 1):
        for tail in compositions(total - first):
            yield (first,) + tail


def weak_compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in weak_compositions(total - first, parts - 1):
            yield (first,) + tail


def surjective_cells(composition, channels):
    for cell in product(*(tuple(weak_compositions(q, channels)) for q in composition)):
        if all(sum(row[channel] for row in cell) for channel in range(channels)):
            yield cell


@cache
def young_subgroup(composition):
    blocks, start = [], 0
    for size in composition:
        blocks.append(tuple(range(start, start + size)))
        start += size
    return tuple(
        sum(blocks_in_order, ())
        for blocks_in_order in product(*(permutations(block) for block in blocks))
    )


@cache
def amplitudes(composition):
    """Coefficient a_q(rho) of each polarized permutation word."""
    result = {}
    subgroup = young_subgroup(composition)
    for block_permutation in subgroup:
        for place_permutation, coefficient in _eulerian_schedule(sum(composition)):
            rho = tuple(block_permutation[position] for position in place_permutation)
            add(result, rho, coefficient / len(subgroup))

    # Independently replay a_q(rho)=|H|^-1 sum_h gamma(h^-1 rho).
    for rho in permutations(range(sum(composition))):
        explicit = Fraction()
        for h in subgroup:
            inverse = [0] * len(h)
            for position, value in enumerate(h):
                inverse[value] = position
            sigma = tuple(inverse[value] for value in rho)
            descents = sum(left > right for left, right in zip(sigma, sigma[1:]))
            explicit += Fraction(
                -1 if descents % 2 else 1,
                len(sigma) * comb(len(sigma) - 1, descents),
            )
        assert result.get(rho, Fraction()) == explicit / len(subgroup)

        offsets = tuple(
            offset
            for offset, multiplicity in enumerate(composition)
            for _ in range(multiplicity)
        )
        assert result.get(rho, Fraction()) == (
            log_signature_offset_coefficient_fraction(
                tuple(offsets[vertex] for vertex in rho)
            )
        )
    return result


def walsh(values):
    transformed = list(values)
    width = len(values).bit_length() - 1
    for bit in range(width):
        step = 1 << bit
        for start in range(0, len(values), 2 * step):
            for offset in range(step):
                left = transformed[start + offset]
                right = transformed[start + offset + step]
                transformed[start + offset] = left + right
                transformed[start + offset + step] = left - right
    return tuple(transformed)


@cache
def moments(composition):
    degree = sum(composition)
    fibres = [Fraction() for _ in range(1 << len(edges(degree)))]
    for word, coefficient in amplitudes(composition).items():
        positions = [0] * degree
        for position, letter in enumerate(word):
            positions[letter] = position
        orientation = sum(
            1 << bit
            for bit, (left, right) in enumerate(edges(degree))
            if positions[left] > positions[right]
        )
        fibres[orientation] += coefficient
    return walsh(fibres)


def occurrence_data(cell, channels):
    colours, original_letters, blocks = [], [], []
    for offset, row in enumerate(cell):
        block = []
        for channel, multiplicity in enumerate(row):
            for _ in range(multiplicity):
                block.append(len(colours))
                colours.append(channel)
                original_letters.append(offset * channels + channel)
        blocks.append(tuple(block))
    return tuple(colours), tuple(original_letters), tuple(blocks)


def occurrence_graph(colours, graph_mask):
    graph_edges = {
        edge
        for bit, edge in enumerate(edges(max(colours) + 1))
        if graph_mask >> bit & 1
    }
    return sum(
        1 << bit
        for bit, (left, right) in enumerate(edges(len(colours)))
        if colours[left] == colours[right]
        or tuple(sorted((colours[left], colours[right]))) in graph_edges
    )


def trace_monoid(offsets, channels, graph_mask):
    dependence = frozenset(
        edge for bit, edge in enumerate(edges(channels)) if graph_mask >> bit & 1
    )
    return _TraceMonoid(OffsetAlphabet(offsets, channels), dependence)


@cache
def unpolarized_image(cell, channels, graph_mask):
    composition = tuple(map(sum, cell))
    monoid = trace_monoid(len(cell), channels, graph_mask)
    row_words = []
    clone_factor = 1
    for offset, row in enumerate(cell):
        letters = tuple(
            offset * channels + channel
            for channel, multiplicity in enumerate(row)
            for _ in range(multiplicity)
        )
        row_words.append(tuple(sorted(set(permutations(letters)))))
        clone_factor *= prod(factorial(multiplicity) for multiplicity in row)
    source_weight = Fraction(clone_factor, prod(factorial(q) for q in composition))
    image = {}
    for rows in product(*row_words):
        word = sum(rows, ())
        for permutation, coefficient in _eulerian_schedule(sum(composition)):
            output = monoid.normal_form(
                tuple(word[position] for position in permutation)
            )
            add(image, output, source_weight * coefficient)
    return image


@cache
def polarized_image(composition, dependence_mask):
    degree = sum(composition)
    monoid = trace_monoid(1, degree, dependence_mask)
    image = {}
    for word, coefficient in amplitudes(composition).items():
        add(image, monoid.normal_form(word), coefficient)
    return image


def check_polarization(cell, channels, graph_mask, dependence_mask):
    composition = tuple(map(sum, cell))
    colours, original_letters, _ = occurrence_data(cell, channels)
    degree = len(colours)
    clone_factor = prod(factorial(value) for row in cell for value in row)
    clone_monoid = trace_monoid(1, degree, dependence_mask)
    original_monoid = trace_monoid(len(cell), channels, graph_mask)
    fibres = defaultdict(set)
    for clone_word in permutations(range(degree)):
        clone_trace = clone_monoid.normal_form(clone_word)
        original_trace = original_monoid.normal_form(
            tuple(original_letters[letter] for letter in clone_word)
        )
        fibres[original_trace].add(clone_trace)
    assert all(len(fibre) == clone_factor for fibre in fibres.values())

    expected = {}
    original = unpolarized_image(cell, channels, graph_mask)
    for original_trace, fibre in fibres.items():
        coefficient = original.get(original_trace, Fraction()) / clone_factor
        if coefficient:
            expected.update({clone_trace: coefficient for clone_trace in fibre})
    assert polarized_image(composition, dependence_mask) == expected


def orientation(word, selected_edges):
    positions = [0] * len(word)
    for position, letter in enumerate(word):
        positions[letter] = position
    return sum(
        1 << bit
        for bit, (left, right) in enumerate(selected_edges)
        if positions[left] > positions[right]
    )


def check_parseval_and_trees(composition, dependence_mask):
    degree = sum(composition)
    selected = tuple(
        edge for bit, edge in enumerate(edges(degree)) if dependence_mask >> bit & 1
    )
    f = [Fraction() for _ in range(1 << len(selected))]
    for word, coefficient in amplitudes(composition).items():
        f[orientation(word, selected)] += coefficient

    direct = {}
    for word, coefficient in polarized_image(composition, dependence_mask).items():
        word_orientation = orientation(word, selected)
        assert word_orientation not in direct
        direct[word_orientation] = coefficient
    assert direct == {index: value for index, value in enumerate(f) if value}

    local_moments = walsh(f)
    global_moments = moments(composition)
    for local_mask, value in enumerate(local_moments):
        global_mask = edge_mask(
            degree,
            (selected[bit] for bit in range(len(selected)) if local_mask >> bit & 1),
        )
        assert value == global_moments[global_mask]
    assert sum(value * value for value in f) == sum(
        value * value for value in local_moments
    ) / (1 << len(selected))

    good_trees = tuple(tree for tree in spanning_trees(degree) if global_moments[tree])
    assert bool(direct) == any(tree & ~dependence_mask == 0 for tree in good_trees)


def check_tree_coordinate(composition, tree):
    degree = sum(composition)
    selected = tuple(edge for bit, edge in enumerate(edges(degree)) if tree >> bit & 1)
    linear_extensions = tuple(
        word for word in permutations(range(degree)) if not orientation(word, selected)
    )
    reference_coefficient = sum(
        (amplitudes(composition).get(word, Fraction()) for word in linear_extensions),
        Fraction(),
    )
    assert moments(composition)[tree] == (1 << (degree - 1)) * reference_coefficient


def projected_witness(tree, colours, channels):
    return edge_mask(
        channels,
        {
            tuple(sorted((colours[left], colours[right])))
            for bit, (left, right) in enumerate(edges(len(colours)))
            if tree >> bit & 1 and colours[left] != colours[right]
        },
    )


def minimal_masks(masks):
    masks = set(masks)
    return frozenset(
        mask
        for mask in masks
        if not any(other != mask and other & ~mask == 0 for other in masks)
    )


def check_witness_antichain(cell, channels):
    composition = tuple(map(sum, cell))
    colours, _, _ = occurrence_data(cell, channels)
    good_trees = tuple(
        tree for tree in spanning_trees(len(colours)) if moments(composition)[tree]
    )
    projected = []
    for tree in good_trees:
        witness = projected_witness(tree, colours, channels)
        assert channels - 1 <= witness.bit_count() <= len(colours) - 1
        assert connected(channels, witness)
        projected.append(witness)
    predicted = minimal_masks(projected)

    surviving = {
        graph_mask
        for graph_mask in range(1 << len(edges(channels)))
        if unpolarized_image(cell, channels, graph_mask)
    }
    assert predicted == minimal_masks(surviving)
    assert all(
        bool(graph_mask in surviving)
        == any(witness & ~graph_mask == 0 for witness in predicted)
        for graph_mask in range(1 << len(edges(channels)))
    )


def cycle_orders(vertices):
    result = []
    for mask in range(1 << len(edges(vertices))):
        chosen = [edge for bit, edge in enumerate(edges(vertices)) if mask >> bit & 1]
        used = sorted({vertex for edge in chosen for vertex in edge})
        if len(used) < 3 or len(chosen) != len(used):
            continue
        adjacency = {vertex: [] for vertex in used}
        for left, right in chosen:
            adjacency[left].append(right)
            adjacency[right].append(left)
        if any(len(neighbours) != 2 for neighbours in adjacency.values()):
            continue
        start = used[0]
        order, previous, current = [start], None, start
        while True:
            following = next(v for v in adjacency[current] if v != previous)
            if following == start:
                break
            order.append(following)
            previous, current = current, following
        if len(order) == len(used):
            result.append(tuple(order))
    return tuple(result)


def check_cycle_identity(degree):
    checks = 0
    for cycle in cycle_orders(degree):
        oriented_edges = tuple(zip(cycle, cycle[1:] + cycle[:1]))
        width = len(cycle)
        for word in permutations(range(degree)):
            positions = {letter: position for position, letter in enumerate(word)}
            signs = tuple(
                1 if positions[left] < positions[right] else -1
                for left, right in oriented_edges
            )
            left = prod(signs)
            right = -sum(
                prod(signs[index] for index in range(width) if mask >> index & 1)
                for mask in range(1 << width)
                if mask != (1 << width) - 1 and mask.bit_count() % 2 == width % 2
            )
            assert left == right
            checks += 1
    return checks


def main():
    if not __debug__:
        raise RuntimeError("run without -O: this verifier is assertion-driven")

    amplitude_checks = 0
    for degree in range(2, AMPLITUDE_MAX_DEGREE + 1):
        for composition in compositions(degree):
            amplitudes(composition)
            amplitude_checks += factorial(degree)
    triples = polarization_checks = parseval_checks = witness_cells = 0
    tree_checks = cycle_checks = 0
    for degree in range(2, MAX_DEGREE + 1):
        cycle_checks += check_cycle_identity(degree)
        for composition in compositions(degree):
            for tree in spanning_trees(degree):
                check_tree_coordinate(composition, tree)
                tree_checks += 1
            for channels in range(1, degree + 1):
                for cell in surjective_cells(composition, channels):
                    check_witness_antichain(cell, channels)
                    witness_cells += 1
                    colours, _, _ = occurrence_data(cell, channels)
                    for graph_mask in range(1 << len(edges(channels))):
                        dependence_mask = occurrence_graph(colours, graph_mask)
                        check_polarization(cell, channels, graph_mask, dependence_mask)
                        check_parseval_and_trees(composition, dependence_mask)
                        predicted = any(
                            tree & ~dependence_mask == 0
                            for tree in spanning_trees(degree)
                            if moments(composition)[tree]
                        )
                        assert predicted == bool(
                            unpolarized_image(cell, channels, graph_mask)
                        )
                        triples += 1
                        polarization_checks += 1
                        parseval_checks += 1

    assert triples == 8 + 140 + 5_996
    print(
        "C.1--C.7 exact bounded replay passed: "
        f"{amplitude_checks} independent tensor-log amplitude checks "
        f"through degree {AMPLITUDE_MAX_DEGREE}; "
        f"{triples} labelled graph/cell triples; "
        f"{polarization_checks} polarization-fibre checks; "
        f"{parseval_checks} trace/Walsh/tree checks; "
        f"{tree_checks} tree-coordinate checks; "
        f"{witness_cells} witness-antichain checks; "
        f"{cycle_checks} pointwise circuit checks; degrees 2--{MAX_DEGREE}."
    )


if __name__ == "__main__":
    main()
