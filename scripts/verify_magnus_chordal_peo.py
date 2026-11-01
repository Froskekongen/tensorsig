#!/usr/bin/env python3
"""Exact replay for MAGNUS-SCALAR D.3 and CHORDAL-PEO-DETECT-GLOBAL-v1.

The chordal checker compares the global paired root-subset recurrence with an
explicit scan of increasing PEO-parent trees and Young-block labellings.  The
recurrence is an exact global oracle; this script makes no width-local or
fixed-parameter compactness claim.

Run from the tensorsig repository root:

    uv run python scripts/verify_magnus_chordal_peo.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator, Sequence
from fractions import Fraction
from functools import cache
from itertools import combinations, product
from math import comb, factorial, prod

from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    _labelled_trees,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    _order_polynomial_value,
    _prepare_tree,
    _young_labelings,
    tree_moment,
)

type Covers = tuple[tuple[int, int], ...]
type ParentSets = tuple[frozenset[int], ...]
type Polynomial = dict[tuple[int, int], Fraction]
type RootResponse = dict[tuple[int, int, int, int], Fraction]


def compositions(total: int) -> Iterator[tuple[int, ...]]:
    """Yield the positive ordered compositions of ``total``."""
    if total == 0:
        yield ()
        return
    for first in range(1, total + 1):
        for rest in compositions(total - first):
            yield (first, *rest)


def ordered_block_maps(degree: int) -> Iterator[tuple[int, ...]]:
    """Yield every surjection ``[degree] -> [block_count]``."""
    for block_count in range(1, degree + 1):
        target = set(range(block_count))
        for beta in product(range(block_count), repeat=degree):
            if set(beta) == target:
                yield beta


def _grouped_tree(
    beta: Sequence[int],
    covers: Iterable[tuple[int, int]],
) -> tuple[tuple[int, ...], Covers]:
    """Relabel a PEO-indexed tree into consecutive chronological blocks."""
    vertices = sorted(range(len(beta)), key=lambda vertex: (beta[vertex], vertex))
    relabel = {vertex: index for index, vertex in enumerate(vertices)}
    block_labels = tuple(sorted(set(beta)))
    composition = tuple(sum(value == block for value in beta) for block in block_labels)
    return composition, tuple(
        (relabel[lower], relabel[upper]) for lower, upper in covers
    )


@cache
def explicit_young_order_value(
    beta: tuple[int, ...],
    covers: Covers,
    n: int,
) -> Fraction:
    """Young-average an order polynomial by literal block-label enumeration."""
    composition, grouped_covers = _grouped_tree(beta, covers)
    tree = _prepare_tree(grouped_covers, len(beta))
    return Fraction(
        sum(
            _order_polynomial_value(tree, labeling, n)
            for labeling in _young_labelings(composition)
        ),
        prod(factorial(part) for part in composition),
    )


@cache
def explicit_tree_moment(beta: tuple[int, ...], covers: Covers) -> Fraction:
    """Evaluate one PEO-parent tree with the independent Bridge-B oracle."""
    composition, grouped_covers = _grouped_tree(beta, covers)
    return tree_moment(composition, grouped_covers)


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for (a, a_prime), left_value in left.items():
        for (b, b_prime), right_value in right.items():
            key = (a + b, a_prime + b_prime)
            result[key] = result.get(key, Fraction()) + left_value * right_value
    return result


def global_root_subset_response(
    beta: Sequence[int],
    parent_sets: ParentSets,
    n: int,
    n_prime: int,
) -> Fraction:
    """Return the exact paired response of CHORDAL-PEO-DETECT-GLOBAL-v1.

    Vertices are in numeric PEO order.  Each non-root vertex chooses one
    shared parent from its supplied later-neighbour set.  The recurrence sums
    the two compatible value maps and their Young hook factors without
    enumerating those parent trees.
    """
    blocks = tuple(beta)
    degree = len(blocks)
    assert degree and len(parent_sets) == degree
    assert n >= 1 and n_prime >= 1

    @cache
    def response(mask: int) -> tuple[tuple[tuple[int, int, int, int], Fraction], ...]:
        root = mask.bit_length() - 1
        lower = mask ^ (1 << root)
        if not lower:
            return tuple(
                ((root_value, root_value_prime, 1, 1), Fraction(1))
                for root_value in range(1, n + 1)
                for root_value_prime in range(1, n_prime + 1)
            )

        result: RootResponse = {}
        for root_value in range(1, n + 1):
            for root_value_prime in range(1, n_prime + 1):

                @cache
                def branch(
                    branch_mask: int,
                ) -> tuple[tuple[tuple[int, int], Fraction], ...]:
                    child_root = branch_mask.bit_length() - 1
                    if root not in parent_sets[child_root]:
                        return ()
                    polynomial: Polynomial = {}
                    for (value, value_prime, size, size_prime), weight in response(
                        branch_mask
                    ):
                        compatible = value < root_value or (
                            value == root_value and blocks[child_root] <= blocks[root]
                        )
                        compatible_prime = value_prime < root_value_prime or (
                            value_prime == root_value_prime
                            and blocks[child_root] <= blocks[root]
                        )
                        if not compatible or not compatible_prime:
                            continue
                        exponent = (
                            size
                            if blocks[child_root] == blocks[root]
                            and value == root_value
                            else 0
                        )
                        exponent_prime = (
                            size_prime
                            if blocks[child_root] == blocks[root]
                            and value_prime == root_value_prime
                            else 0
                        )
                        key = (exponent, exponent_prime)
                        polynomial[key] = polynomial.get(key, Fraction()) + Fraction(
                            weight,
                            size * size_prime,
                        )
                    return tuple(polynomial.items())

                @cache
                def anchored_partitions(
                    remaining: int,
                ) -> tuple[tuple[tuple[int, int], Fraction], ...]:
                    if not remaining:
                        return (((0, 0), Fraction(1)),)
                    anchor_bit = remaining & -remaining
                    polynomial: Polynomial = {}
                    branch_mask = remaining
                    while branch_mask:
                        if branch_mask & anchor_bit:
                            branch_polynomial = dict(branch(branch_mask))
                            if branch_polynomial:
                                tail = dict(
                                    anchored_partitions(remaining ^ branch_mask)
                                )
                                for key, value in _multiply(
                                    branch_polynomial,
                                    tail,
                                ).items():
                                    polynomial[key] = (
                                        polynomial.get(
                                            key,
                                            Fraction(),
                                        )
                                        + value
                                    )
                        branch_mask = (branch_mask - 1) & remaining
                    return tuple(polynomial.items())

                for (size_minus_one, size_prime_minus_one), weight in dict(
                    anchored_partitions(lower)
                ).items():
                    key = (
                        root_value,
                        root_value_prime,
                        size_minus_one + 1,
                        size_prime_minus_one + 1,
                    )
                    result[key] = result.get(key, Fraction()) + weight
        return tuple(result.items())

    full_mask = (1 << degree) - 1
    return sum(
        (
            weight / (size * size_prime)
            for (_, _, size, size_prime), weight in response(full_mask)
        ),
        start=Fraction(),
    )


def increasing_parent_trees(parent_sets: ParentSets) -> Iterator[Covers]:
    """Yield the explicit PEO-parent trees used only by the brute oracle."""
    for parents in product(*(tuple(sorted(row)) for row in parent_sets[:-1])):
        yield tuple((vertex, parent) for vertex, parent in enumerate(parents))


def canonical_energy(
    beta: Sequence[int],
    parent_sets: ParentSets,
) -> Fraction:
    """Interpolate the global response to the squared canonical-tree energy."""
    degree = len(beta)
    coefficients = tuple(
        Fraction((-1) ** (n - 1) * comb(degree, n), n) for n in range(1, degree + 1)
    )
    return (1 << (2 * degree - 2)) * sum(
        (
            coefficients[n - 1]
            * coefficients[n_prime - 1]
            * global_root_subset_response(beta, parent_sets, n, n_prime)
            for n in range(1, degree + 1)
            for n_prime in range(1, degree + 1)
        ),
        start=Fraction(),
    )


def _numeric_peo_parent_sets(degree: int, edges: Covers) -> ParentSets | None:
    adjacency = [set() for _ in range(degree)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    later = tuple(frozenset(v for v in adjacency[u] if v > u) for u in range(degree))
    if any(not later[vertex] for vertex in range(degree - 1)):
        return None
    for neighbors in later:
        if any(
            right not in adjacency[left]
            for left, right in combinations(sorted(neighbors), 2)
        ):
            return None
    return later


def chordal_instances(degree: int) -> Iterator[ParentSets]:
    possible_edges = tuple(combinations(range(degree), 2))
    for edge_mask in range(1 << len(possible_edges)):
        edges = tuple(
            edge for index, edge in enumerate(possible_edges) if edge_mask >> index & 1
        )
        parent_sets = _numeric_peo_parent_sets(degree, edges)
        if parent_sets is not None:
            yield parent_sets


def verify_global_root_subset(max_degree: int) -> dict[str, object]:
    response_checks = 0
    energy_checks = 0
    graph_counts: dict[int, int] = {}
    beta_counts: dict[int, int] = {}
    for degree in range(2, max_degree + 1):
        graphs = tuple(chordal_instances(degree))
        betas = tuple(ordered_block_maps(degree))
        graph_counts[degree] = len(graphs)
        beta_counts[degree] = len(betas)
        for parent_sets in graphs:
            trees = tuple(increasing_parent_trees(parent_sets))
            for beta in betas:
                for n in range(1, degree + 1):
                    for n_prime in range(1, degree + 1):
                        expected = sum(
                            (
                                explicit_young_order_value(beta, tree, n)
                                * explicit_young_order_value(beta, tree, n_prime)
                                for tree in trees
                            ),
                            start=Fraction(),
                        )
                        actual = global_root_subset_response(
                            beta,
                            parent_sets,
                            n,
                            n_prime,
                        )
                        assert actual == expected, (
                            degree,
                            parent_sets,
                            beta,
                            n,
                            n_prime,
                            actual,
                            expected,
                        )
                        response_checks += 1

                expected_energy = sum(
                    (explicit_tree_moment(beta, tree) ** 2 for tree in trees),
                    start=Fraction(),
                )
                actual_energy = canonical_energy(beta, parent_sets)
                assert actual_energy == expected_energy, (
                    degree,
                    parent_sets,
                    beta,
                    actual_energy,
                    expected_energy,
                )
                assert actual_energy >= 0
                energy_checks += 1

    all_equal_triangle = (
        frozenset({1, 2}),
        frozenset({2}),
        frozenset(),
    )
    assert global_root_subset_response(
        (0, 0, 0),
        all_equal_triangle,
        1,
        1,
    ) == Fraction(5, 36)

    strict_triangle = (0, 1, 2)
    parent_energy = canonical_energy(strict_triangle, all_equal_triangle)
    all_tree_energy = sum(
        (
            explicit_tree_moment(strict_triangle, tuple(sorted(tree))) ** 2
            for tree in _labelled_trees(3)
        ),
        start=Fraction(),
    )
    assert parent_energy == Fraction(20, 9)
    assert all_tree_energy == Fraction(8, 3)
    assert parent_energy != all_tree_energy

    return {
        "target": "CHORDAL-PEO-DETECT-GLOBAL-v1",
        "terminal_scope": "exact global oracle; no width-local compactness claim",
        "exhaustive_through_degree": max_degree,
        "numeric_peo_graphs": graph_counts,
        "ordered_block_maps": beta_counts,
        "paired_response_checks": response_checks,
        "interpolated_energy_checks": energy_checks,
        "triangle_R_1_1": "5/36",
        "k3_literal_all_tree_energy": str(all_tree_energy),
        "k3_parent_detection_energy": str(parent_energy),
        "literal_energy_distinction_certified": True,
    }


def verify_ch4() -> dict[str, object]:
    fixtures = (
        (
            (0, 3, 2, 4, 5, 1),
            (
                frozenset({5}),
                frozenset({2}),
                frozenset({5}),
                frozenset({5}),
                frozenset({5}),
                frozenset(),
            ),
            Fraction(),
        ),
        (
            (0, 2, 3, 4, 5, 1),
            (
                frozenset({5}),
                frozenset({2}),
                frozenset({5}),
                frozenset({5}),
                frozenset({5}),
                frozenset(),
            ),
            Fraction(1, 900),
        ),
    )
    normalized = []
    for beta, parent_sets, expected in fixtures:
        energy = canonical_energy(beta, parent_sets)
        trees = tuple(increasing_parent_trees(parent_sets))
        assert len(trees) == 1
        assert energy == explicit_tree_moment(beta, trees[0]) ** 2
        value = energy / (1 << (2 * len(beta) - 2))
        assert value == expected, (beta, value, expected)
        normalized.append(str(value))
    return {
        "target": "CH4 inserted rooted pair",
        "normalized_squared_values": normalized,
        "separated": True,
    }


def _is_connected(vertices: frozenset[int], edges: Covers) -> bool:
    if not vertices:
        return False
    adjacency = {vertex: set() for vertex in vertices}
    for left, right in edges:
        if left in vertices and right in vertices:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen = {next(iter(vertices))}
    todo = list(seen)
    while todo:
        for neighbor in adjacency[todo.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                todo.append(neighbor)
    return seen == vertices


def _connected_subsets(degree: int, edges: Covers) -> Iterator[frozenset[int]]:
    for mask in range(1, 1 << degree):
        vertices = frozenset(vertex for vertex in range(degree) if mask >> vertex & 1)
        if _is_connected(vertices, edges):
            yield vertices


def _orientations(tree: frozenset[tuple[int, int]]) -> Iterator[Covers]:
    edges = tuple(sorted(tree))
    for orientation_mask in range(1 << len(edges)):
        yield tuple(
            (right, left) if orientation_mask >> index & 1 else (left, right)
            for index, (left, right) in enumerate(edges)
        )


def _ordered_connected_partitions(
    vertices: frozenset[int],
    part_count: int,
    edges: Covers,
) -> Iterator[tuple[frozenset[int], ...]]:
    ordered_vertices = tuple(sorted(vertices))
    if not ordered_vertices:
        if part_count == 0:
            yield ()
        return
    if not 1 <= part_count <= len(ordered_vertices):
        return
    for assignments in product(range(part_count), repeat=len(ordered_vertices)):
        if set(assignments) != set(range(part_count)):
            continue
        parts = tuple(
            frozenset(
                vertex
                for vertex, assigned in zip(
                    ordered_vertices,
                    assignments,
                    strict=True,
                )
                if assigned == index
            )
            for index in range(part_count)
        )
        if all(_is_connected(part, edges) for part in parts):
            yield parts


def _bernoulli_plus(index: int) -> Fraction:
    values = [Fraction(1)]
    for degree in range(1, index + 1):
        value = -sum(
            (Fraction(comb(degree + 1, k)) * values[k] for k in range(degree)),
            start=Fraction(),
        ) / (degree + 1)
        values.append(value)
    if index == 1:
        return Fraction(1, 2)
    return values[index]


@cache
def _zeta(
    beta: tuple[int, ...],
    reference_edges: Covers,
    vertices: frozenset[int],
) -> Fraction:
    if len(vertices) == 1:
        return Fraction(1)
    induced = tuple(
        (left, right)
        for left, right in reference_edges
        if left in vertices and right in vertices
    )
    local_beta = tuple(beta[vertex] for vertex in sorted(vertices))
    local_index = {vertex: index for index, vertex in enumerate(sorted(vertices))}
    local_edges = tuple(
        (local_index[left], local_index[right]) for left, right in induced
    )
    composition, grouped_edges = _grouped_tree(local_beta, local_edges)
    return tree_moment(composition, grouped_edges) / (1 << (len(vertices) - 1))


def _lambda_sign(
    root: int,
    parts: tuple[frozenset[int], ...],
    reference_edges: Covers,
) -> int:
    suffix = {root}
    sign = 1
    for part in reversed(parts):
        crossings = tuple(
            (left, right)
            for left, right in reference_edges
            if (left in part and right in suffix) or (right in part and left in suffix)
        )
        if len(crossings) != 1:
            return 0
        left, right = crossings[0]
        sign *= 1 if left in part and right in suffix else -1
        suffix.update(part)
    return sign


def magnus_scalar_rhs(
    beta: tuple[int, ...],
    reference_edges: Covers,
    vertices: frozenset[int],
    stage: int,
) -> Fraction:
    total = Fraction()
    for root in vertices:
        if beta[root] != stage:
            continue
        remaining = vertices - {root}
        for part_count in range(len(vertices)):
            bernoulli = _bernoulli_plus(part_count)
            if not bernoulli:
                continue
            for parts in _ordered_connected_partitions(
                remaining,
                part_count,
                reference_edges,
            ):
                sign = _lambda_sign(root, parts, reference_edges)
                if sign:
                    total += (
                        Fraction(bernoulli, factorial(part_count))
                        * sign
                        * prod(_zeta(beta, reference_edges, part) for part in parts)
                    )
    return total


def verify_magnus_scalar(max_degree: int) -> dict[str, object]:
    checks_by_degree: dict[int, int] = {}
    full_support_checks = 0
    for degree in range(1, max_degree + 1):
        checks = 0
        trees = (frozenset(),) if degree == 1 else tuple(_labelled_trees(degree))
        for composition in compositions(degree):
            beta = tuple(
                block for block, size in enumerate(composition) for _ in range(size)
            )
            for undirected in trees:
                reference_edges = tuple(sorted(undirected))
                for vertices in _connected_subsets(
                    degree,
                    reference_edges,
                ):
                    stage = max(beta[vertex] for vertex in vertices)
                    left = sum(beta[vertex] == stage for vertex in vertices) * _zeta(
                        beta,
                        reference_edges,
                        vertices,
                    )
                    right = magnus_scalar_rhs(
                        beta,
                        reference_edges,
                        vertices,
                        stage,
                    )
                    assert left == right, (
                        degree,
                        composition,
                        reference_edges,
                        vertices,
                        stage,
                        left,
                        right,
                    )
                    checks += 1
                    full_support_checks += len(vertices) == degree
        checks_by_degree[degree] = checks

    oriented_full_support_checks = 0
    for degree in range(2, min(max_degree, 4) + 1):
        vertices = frozenset(range(degree))
        for composition in compositions(degree):
            beta = tuple(
                block for block, size in enumerate(composition) for _ in range(size)
            )
            stage = len(composition) - 1
            for tree in _labelled_trees(degree):
                for reference_edges in _orientations(tree):
                    left = composition[-1] * _zeta(
                        beta,
                        reference_edges,
                        vertices,
                    )
                    right = magnus_scalar_rhs(
                        beta,
                        reference_edges,
                        vertices,
                        stage,
                    )
                    assert left == right, (
                        "oriented",
                        degree,
                        composition,
                        reference_edges,
                        left,
                        right,
                    )
                    oriented_full_support_checks += 1
    return {
        "target": "MAGNUS-SCALAR D.3",
        "oracle": "Bridge-B explicit Young average/tree moment",
        "reference_orientation": "ascending occurrence labels",
        "exhaustive_through_degree": max_degree,
        "checks_by_degree": checks_by_degree,
        "connected_subtree_stage_checks": sum(checks_by_degree.values()),
        "full_support_tree_composition_checks": full_support_checks - 1,
        "all_orientation_full_support_checks_through_degree_4": (
            oriented_full_support_checks
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chordal-max-degree", type=int, default=4)
    parser.add_argument("--magnus-max-degree", type=int, default=5)
    parser.add_argument("--skip-ch4", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.chordal_max_degree <= 4:
        parser.error("--chordal-max-degree must lie in 2, ..., 4")
    if not 1 <= args.magnus_max_degree <= 5:
        parser.error("--magnus-max-degree must lie in 1, ..., 5")

    report = {
        "chordal_peo": verify_global_root_subset(args.chordal_max_degree),
        "magnus_scalar": verify_magnus_scalar(args.magnus_max_degree),
    }
    if not args.skip_ch4:
        report["ch4"] = verify_ch4()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
