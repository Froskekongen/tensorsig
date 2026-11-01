#!/usr/bin/env python3
"""Independent exact checks for intermediate-rank identities.

This is bounded evidence, not an all-degree proof. It uses only standard-library
integer and Fraction arithmetic and deliberately derives the tensor-log word
coefficients without importing tensorsig's Eulerian implementation.

Run from the repository root with assertions enabled:

    python -B scripts/redteam_op_intermediate_rank_c1_c7.py
"""

from collections import defaultdict
from fractions import Fraction
from functools import cache
import heapq
from itertools import combinations, permutations, product
from math import comb, factorial, prod
from random import Random
from time import perf_counter


SEED = 20260827

Word = tuple[int, ...]
Polynomial = dict[Word, int]


@cache
def edge_universe(vertices: int) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(vertices), 2))


def connected(vertices: int, mask: int) -> bool:
    if vertices == 1:
        return True
    adjacency = [[] for _ in range(vertices)]
    for bit, (left, right) in enumerate(edge_universe(vertices)):
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
def spanning_trees(vertices: int) -> tuple[int, ...]:
    return tuple(
        mask
        for mask in range(1 << len(edge_universe(vertices)))
        if mask.bit_count() == vertices - 1 and connected(vertices, mask)
    )


def compositions(total: int):
    """Yield every positive ordered composition of total."""
    for cut_mask in range(1 << (total - 1)):
        result, previous = [], 0
        for position in range(total - 1):
            if cut_mask >> position & 1:
                result.append(position + 1 - previous)
                previous = position + 1
        yield (*result, total - previous)


@cache
def young_subgroup(composition: tuple[int, ...]) -> tuple[Word, ...]:
    blocks, start = [], 0
    for size in composition:
        blocks.append(tuple(range(start, start + size)))
        start += size
    return tuple(
        sum(ordered_blocks, ())
        for ordered_blocks in product(*(permutations(block) for block in blocks))
    )


def gamma(permutation: Word) -> Fraction:
    degree = len(permutation)
    descents = sum(left > right for left, right in zip(permutation, permutation[1:]))
    return Fraction((-1) ** descents, degree * comb(degree - 1, descents))


@cache
def eulerian_schedule(degree: int) -> tuple[tuple[Word, Fraction], ...]:
    return tuple(
        (permutation, gamma(permutation)) for permutation in permutations(range(degree))
    )


@cache
def direct_tensor_log_coefficient(offsets: tuple[int, ...]) -> Fraction:
    """Coefficient from log(1 + u), independently of the descent formula."""
    degree = len(offsets)
    total = Fraction()
    for cut_mask in range(1 << (degree - 1)):
        points = (
            (0,)
            + tuple(
                position + 1
                for position in range(degree - 1)
                if cut_mask >> position & 1
            )
            + (degree,)
        )
        weight = Fraction(1)
        for start, stop in zip(points, points[1:]):
            block = offsets[start:stop]
            if any(left > right for left, right in zip(block, block[1:])):
                weight = Fraction()
                break
            weight /= prod(factorial(block.count(offset)) for offset in set(block))
        if weight:
            factors = len(points) - 1
            total += Fraction((-1) ** (factors + 1), factors) * weight
    return total


def parabolic_coefficients(composition: tuple[int, ...]) -> dict[Word, Fraction]:
    degree = sum(composition)
    coefficients = {
        permutation: Fraction() for permutation in permutations(range(degree))
    }
    subgroup = young_subgroup(composition)
    denominator = len(subgroup)
    for block_permutation in subgroup:
        for place_permutation, coefficient in eulerian_schedule(degree):
            output = tuple(
                block_permutation[position] for position in place_permutation
            )
            coefficients[output] += coefficient / denominator
    return coefficients


@cache
def permutation_data(degree: int) -> tuple[tuple[Word, ...], tuple[int, ...]]:
    all_permutations = tuple(permutations(range(degree)))
    inversion_masks = []
    for permutation in all_permutations:
        positions = [0] * degree
        for position, vertex in enumerate(permutation):
            positions[vertex] = position
        inversion_masks.append(
            sum(
                1 << bit
                for bit, (left, right) in enumerate(edge_universe(degree))
                if positions[left] > positions[right]
            )
        )
    return all_permutations, tuple(inversion_masks)


def walsh(values):
    transformed = list(values)
    step = 1
    while step < len(transformed):
        for start in range(0, len(transformed), 2 * step):
            for offset in range(step):
                left = transformed[start + offset]
                right = transformed[start + offset + step]
                transformed[start + offset] = left + right
                transformed[start + offset + step] = left - right
        step *= 2
    return transformed


def subset_zeta(values):
    result = list(values)
    for bit in range(len(values).bit_length() - 1):
        for mask in range(len(result)):
            if mask >> bit & 1:
                result[mask] += result[mask ^ (1 << bit)]
    return result


def check_parabolic_c5_c6() -> dict[str, int]:
    coefficient_checks = moment_entries = tree_checks = 0
    for degree in range(2, 7):
        all_permutations, inversion_masks = permutation_data(degree)
        edge_count = len(edge_universe(degree))
        full_mask = (1 << edge_count) - 1
        for composition in compositions(degree):
            offsets = tuple(
                offset
                for offset, multiplicity in enumerate(composition)
                for _ in range(multiplicity)
            )
            coefficients = parabolic_coefficients(composition)
            for permutation, coefficient in coefficients.items():
                direct = direct_tensor_log_coefficient(
                    tuple(offsets[vertex] for vertex in permutation)
                )
                assert coefficient == direct
                coefficient_checks += 1

            orientation_coefficients = [Fraction() for _ in range(1 << edge_count)]
            for permutation, inversion_mask in zip(all_permutations, inversion_masks):
                orientation_coefficients[inversion_mask] += coefficients[permutation]
            moments = walsh(orientation_coefficients)
            reference_fibres = subset_zeta(orientation_coefficients)
            moment_entries += len(moments)
            for tree in spanning_trees(degree):
                reference_coefficient = reference_fibres[full_mask ^ tree]
                assert moments[tree] == (1 << (degree - 1)) * reference_coefficient
                tree_checks += 1

    assert coefficient_checks == 25_180
    assert moment_entries == 1_065_508
    assert tree_checks == 43_614
    return {
        "direct_log_coefficients": coefficient_checks,
        "moment_entries": moment_entries,
        "C5_C6_trees": tree_checks,
    }


def add_term(polynomial: Polynomial, word: Word, coefficient: int) -> None:
    value = polynomial.get(word, 0) + coefficient
    if value:
        polynomial[word] = value
    else:
        polynomial.pop(word, None)


def bracket(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            coefficient = left_coefficient * right_coefficient
            add_term(result, left_word + right_word, coefficient)
            add_term(result, right_word + left_word, -coefficient)
    return result


@cache
def lie_basis(degree: int) -> tuple[Polynomial, ...]:
    result = []
    for tail in permutations(range(1, degree)):
        polynomial = {(0,): 1}
        for vertex in tail:
            polynomial = bracket(polynomial, {(vertex,): 1})
        assert polynomial[(0, *tail)] == 1
        result.append(polynomial)
    assert len(result) == factorial(degree - 1)
    return tuple(result)


def combine_basis(
    basis: tuple[Polynomial, ...], weights: tuple[int, ...]
) -> Polynomial:
    result: Polynomial = {}
    for weight, polynomial in zip(weights, basis):
        for word, coefficient in polynomial.items():
            add_term(result, word, weight * coefficient)
    assert result
    return result


def primitive_vectors(degree: int, random: Random):
    dimension = factorial(degree - 1)
    if degree <= 4:
        yield from (
            weights for weights in product((-1, 0, 1), repeat=dimension) if any(weights)
        )
        return

    target = 256 if degree == 5 else 64
    seen = set()
    while len(seen) < target:
        weights = tuple(random.randint(-2, 2) for _ in range(dimension))
        if any(weights) and weights not in seen:
            seen.add(weights)
            yield weights


def graph_scope(degree: int, random: Random) -> tuple[int, ...]:
    size = 1 << len(edge_universe(degree))
    if degree <= 5:
        return tuple(range(size))
    edges = edge_universe(degree)
    path = sum(1 << edges.index((vertex, vertex + 1)) for vertex in range(degree - 1))
    star = sum(1 << edges.index((0, vertex)) for vertex in range(1, degree))
    required = {0, size - 1, path, star}
    candidates = [mask for mask in range(size) if mask not in required]
    return tuple(sorted(required | set(random.sample(candidates, 512 - len(required)))))


def submasks(mask: int):
    submask = mask
    while True:
        yield submask
        if not submask:
            return
        submask = (submask - 1) & mask


def check_primitive_campaign() -> dict[str, int]:
    random = Random(SEED)
    vector_count = vector_graph_checks = tree_fibre_checks = 0
    for degree in range(2, 7):
        all_permutations, inversion_masks = permutation_data(degree)
        edge_count = len(edge_universe(degree))
        graph_masks = graph_scope(degree, random)
        basis = lie_basis(degree)
        degree_vectors = degree_checks = 0
        for vector_index, weights in enumerate(primitive_vectors(degree, random)):
            polynomial = combine_basis(basis, weights)
            coefficients = [
                polynomial.get(permutation, 0) for permutation in all_permutations
            ]
            orientation_coefficients = [0 for _ in range(1 << edge_count)]
            for inversion_mask, coefficient in zip(inversion_masks, coefficients):
                orientation_coefficients[inversion_mask] += coefficient
            moments = walsh(orientation_coefficients)
            moment_square_sums = subset_zeta([value * value for value in moments])
            good_trees = tuple(tree for tree in spanning_trees(degree) if moments[tree])

            for dependence_mask in graph_masks:
                trace_fibres = defaultdict(int)
                for inversion_mask, coefficient in zip(inversion_masks, coefficients):
                    trace_fibres[inversion_mask & dependence_mask] += coefficient
                nonzero_fibres = [value for value in trace_fibres.values() if value]
                actual = bool(nonzero_fibres)
                predicted = any(tree & ~dependence_mask == 0 for tree in good_trees)
                assert actual == predicted
                trace_norm = sum(value * value for value in nonzero_fibres)
                moment_norm = Fraction(
                    moment_square_sums[dependence_mask],
                    1 << dependence_mask.bit_count(),
                )
                assert trace_norm == moment_norm
                vector_graph_checks += 1
                degree_checks += 1

            check_tree_fibres = (
                degree <= 4
                or degree == 5
                and vector_index < 64
                or degree == 6
                and vector_index < 8
            )
            if check_tree_fibres:
                for tree in spanning_trees(degree):
                    trace_fibres = defaultdict(int)
                    for inversion_mask, coefficient in zip(
                        inversion_masks, coefficients
                    ):
                        trace_fibres[inversion_mask & tree] += coefficient
                    for orientation in submasks(tree):
                        expected = Fraction(
                            moments[tree] * (-1) ** orientation.bit_count(),
                            1 << (degree - 1),
                        )
                        assert trace_fibres[orientation] == expected
                        tree_fibre_checks += 1

            degree_vectors += 1
            vector_count += 1
        print(
            f"primitive m={degree}: {degree_vectors} vectors x "
            f"{len(graph_masks)} graphs = {degree_checks} cases"
        )

    assert vector_count == 1_058
    assert vector_graph_checks == 341_572
    assert tree_fibre_checks == 553_060
    return {
        "primitive_vectors": vector_count,
        "primitive_graph_cases": vector_graph_checks,
        "primitive_C5_fibres": tree_fibre_checks,
    }


def simple_cycles(vertices: int) -> tuple[Word, ...]:
    result = []
    for width in range(3, vertices + 1):
        for selected in combinations(range(vertices), width):
            root, *others = selected
            for tail in permutations(others):
                if tail[0] < tail[-1]:
                    result.append((root, *tail))
    return tuple(result)


def check_cycle_identity() -> int:
    checks = 0
    for degree in range(3, 7):
        for cycle in simple_cycles(degree):
            width = len(cycle)
            for word in permutations(range(degree)):
                positions = [0] * degree
                for position, vertex in enumerate(word):
                    positions[vertex] = position
                signs = tuple(
                    1
                    if positions[cycle[index]] < positions[cycle[(index + 1) % width]]
                    else -1
                    for index in range(width)
                )
                left = prod(signs)
                right = -sum(
                    prod(signs[index] for index in range(width) if mask >> index & 1)
                    for mask in range((1 << width) - 1)
                    if mask.bit_count() % 2 == width % 2
                )
                assert left == right
                checks += 1
    assert checks == 146_454
    return checks


@cache
def trace_normal_form(
    word: Word, dependence_edges: tuple[tuple[int, int], ...]
) -> Word:
    dependence = frozenset(dependence_edges)
    followers = [[] for _ in word]
    indegree = [0] * len(word)
    for left in range(len(word)):
        for right in range(left + 1, len(word)):
            channels = tuple(sorted((word[left], word[right])))
            if word[left] == word[right] or channels in dependence:
                followers[left].append(right)
                indegree[right] += 1
    available = [
        (letter, position)
        for position, (letter, degree) in enumerate(zip(word, indegree))
        if degree == 0
    ]
    heapq.heapify(available)
    result = []
    while available:
        letter, position = heapq.heappop(available)
        result.append(letter)
        for follower in followers[position]:
            indegree[follower] -= 1
            if indegree[follower] == 0:
                heapq.heappush(available, (word[follower], follower))
    return tuple(result)


def quotient(polynomial: Polynomial, dependence_edges) -> Polynomial:
    edges = tuple(sorted(dependence_edges))
    result: Polynomial = {}
    for word, coefficient in polynomial.items():
        add_term(result, trace_normal_form(word, edges), coefficient)
    return result


def check_negative_controls() -> dict[str, int]:
    # Without primitivity, x0*x1 survives the empty graph although no tree fits.
    nonprimitive = {(0, 1): 1}
    assert quotient(nonprimitive, set())
    assert not any(tree & ~0 == 0 for tree in spanning_trees(2))

    # This full-support primitive has cyclic, not channel-tree, minimal support.
    a, b, c = ({(vertex,): 1} for vertex in range(3))
    cyclic_primitive = bracket(bracket(a, b), bracket(bracket(b, c), bracket(c, a)))
    assert len(cyclic_primitive) == 32
    assert {
        tuple(word.count(channel) for channel in range(3)) for word in cyclic_primitive
    } == {(2, 2, 2)}
    graph_edges = edge_universe(3)
    survival = []
    for mask in range(1 << len(graph_edges)):
        dependence = {edge for bit, edge in enumerate(graph_edges) if mask >> bit & 1}
        survival.append(bool(quotient(cyclic_primitive, dependence)))
    assert survival == [False] * 7 + [True]
    assert cyclic_primitive[(0, 1, 0, 2, 1, 2)] == 1
    return {"nonprimitive_controls": 1, "cyclic_primitive_graphs": len(survival)}


def main() -> None:
    if not __debug__:
        raise RuntimeError("run without -O: these checks require assertions")

    started = perf_counter()
    summary = {}
    summary.update(check_parabolic_c5_c6())
    summary.update(check_primitive_campaign())
    summary["C4_cycle_orders"] = check_cycle_identity()
    summary.update(check_negative_controls())
    print(f"red-team passed (seed={SEED}): {summary}")
    print(f"runtime: {perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()
