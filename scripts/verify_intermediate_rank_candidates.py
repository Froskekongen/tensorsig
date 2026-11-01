#!/usr/bin/env python3
"""Verify exact intermediate-rank computations.

Correctness and structural counts live here; wall-clock measurements live in
``benchmark_intermediate_rank_candidates.py``.  The replay triangulates the
implicit Young DP against both the explicit Young average and a literal B.Y
equal-component sum, verifies circuit reconstruction, and checks the exact
positive-support/ZDD upward closure.

Run from the tensorsig repository root:

    uv run python scripts/verify_intermediate_rank_candidates.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from functools import cache
from itertools import combinations, permutations, product
from math import factorial, prod

from tensorsig.experimental.offset_coded_signatures.circuit_coordinates import (
    compile_tree_circuit_coordinates,
)
from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    _labelled_trees,
    cell_survives,
    good_tree_catalogue,
)
from tensorsig.experimental.offset_coded_signatures.support_compiler import (
    compile_projected_support,
)
from tensorsig.offset_coded_signatures import graph_queries
from tensorsig.offset_coded_signatures._graph.adaptive import (
    adaptive_tree_moment,
    composition_is_universally_zero,
)
from tensorsig.offset_coded_signatures._graph.one_centre_gram import (
    one_centre_gram_cell_survives,
    one_centre_gram_energy,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    _order_polynomial_value,
    _prepare_tree,
    _young_labelings,
    tree_moment,
)
from tensorsig.offset_coded_signatures._graph.young_dp import (
    YoungDpMetrics,
    young_dp_order_polynomial_value,
    young_dp_tree_moment,
)
from tensorsig.offset_coded_signatures.graph_queries import (
    compile_chordal_nbc,
    compile_direct_nbc,
)
from tensorsig.offset_coded_signatures.quotients import DependenceGraph

type Covers = tuple[tuple[int, int], ...]


def compositions(total: int):
    """Yield all positive ordered compositions of ``total``."""
    if total == 0:
        yield ()
        return
    for first in range(1, total + 1):
        for rest in compositions(total - first):
            yield (first, *rest)


def block_indices(composition: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(block for block, size in enumerate(composition) for _ in range(size))


def orientations(tree: frozenset[tuple[int, int]]):
    edges = tuple(sorted(tree))
    for mask in range(1 << len(edges)):
        yield tuple(
            (right, left) if mask >> index & 1 else (left, right)
            for index, (left, right) in enumerate(edges)
        )


@cache
def linear_extension_count(
    vertices: tuple[int, ...],
    covers: tuple[tuple[int, int], ...],
) -> int:
    """Count extensions independently by a predecessor-mask subset DP."""
    local = {vertex: index for index, vertex in enumerate(vertices)}
    predecessors = [0] * len(vertices)
    for lower, upper in covers:
        predecessors[local[upper]] |= 1 << local[lower]
    counts = [0] * (1 << len(vertices))
    counts[0] = 1
    for chosen in range(1 << len(vertices)):
        for vertex, required in enumerate(predecessors):
            bit = 1 << vertex
            if not chosen & bit and required & ~chosen == 0:
                counts[chosen | bit] += counts[chosen]
    return counts[-1]


def equal_component_order_value(
    composition: tuple[int, ...],
    covers: Covers,
    n: int,
) -> Fraction:
    """Evaluate proposal B.Y literally, without either production DP."""
    degree = sum(composition)
    blocks = block_indices(composition)
    total = Fraction()
    for values in product(range(1, n + 1), repeat=degree):
        if any(
            values[lower] > values[upper]
            or (values[lower] == values[upper] and blocks[lower] > blocks[upper])
            for lower, upper in covers
        ):
            continue

        adjacency = [set() for _ in range(degree)]
        for lower, upper in covers:
            if blocks[lower] == blocks[upper] and values[lower] == values[upper]:
                adjacency[lower].add(upper)
                adjacency[upper].add(lower)

        weight = Fraction(1)
        unseen = set(range(degree))
        while unseen:
            start = unseen.pop()
            component = {start}
            todo = [start]
            while todo:
                for neighbor in adjacency[todo.pop()]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        todo.append(neighbor)
            if len(component) == 1:
                continue
            vertices = tuple(sorted(component))
            component_covers = tuple(
                sorted(
                    (lower, upper)
                    for lower, upper in covers
                    if lower in component and upper in component
                )
            )
            weight *= Fraction(
                linear_extension_count(vertices, component_covers),
                factorial(len(component)),
            )
        total += weight
    return total


def explicit_young_order_value(
    composition: tuple[int, ...],
    covers: Covers,
    n: int,
) -> Fraction:
    tree = _prepare_tree(covers, sum(composition))
    return Fraction(
        sum(
            _order_polynomial_value(tree, labeling, n)
            for labeling in _young_labelings(composition)
        ),
        prod(factorial(part) for part in composition),
    )


def verify_young_dp() -> dict[str, object]:
    metrics = YoungDpMetrics()
    polynomial_checks = 0
    for degree in range(2, 5):
        for composition in compositions(degree):
            for tree in _labelled_trees(degree):
                for covers in orientations(tree):
                    for n in range(1, degree + 1):
                        explicit = explicit_young_order_value(composition, covers, n)
                        by_components = equal_component_order_value(
                            composition,
                            covers,
                            n,
                        )
                        by_dp = young_dp_order_polynomial_value(
                            composition,
                            covers,
                            n,
                            metrics=metrics,
                        )
                        assert explicit == by_components == by_dp
                        polynomial_checks += 1

    degree_nine_tree = (
        (0, 5),
        (0, 8),
        (1, 2),
        (1, 4),
        (3, 4),
        (3, 6),
        (4, 5),
        (4, 7),
    )
    degree_nine_reoriented = tuple(
        (right, left) if index % 2 else (left, right)
        for index, (left, right) in enumerate(reversed(degree_nine_tree))
    )
    controls = (
        (
            (1, 1, 1, 2),
            ((0, 3), (1, 2), (2, 3), (3, 4)),
        ),
        (
            (2, 2, 2),
            ((0, 4), (1, 4), (2, 1), (3, 1), (4, 5)),
        ),
        (
            (1,) * 9,
            degree_nine_tree,
        ),
        ((1,) * 9, degree_nine_reoriented),
    )
    for composition, covers in controls:
        assert young_dp_tree_moment(composition, covers) == tree_moment(
            composition,
            covers,
        )
        assert adaptive_tree_moment(composition, covers) == tree_moment(
            composition,
            covers,
        )

    zero_controls = ((6,), (1, 3), (3, 1), (1, 5), (5, 1))
    for composition in zero_controls:
        degree = sum(composition)
        covers = tuple((vertex, vertex + 1) for vertex in range(degree - 1))
        assert composition_is_universally_zero(composition)
        assert adaptive_tree_moment(composition, covers) == 0
        assert tree_moment(composition, covers) == 0

    return {
        "exhaustive_polynomials_through_degree": 4,
        "control_degrees": [5, 6, 9],
        "polynomial_checks": polynomial_checks,
        "terminal_moment_controls": len(controls),
        "adaptive_zero_controls": len(zero_controls),
        "polynomial_evaluations": metrics.polynomial_evaluations,
        "state_cells": metrics.state_cells,
        "peak_state_cells": metrics.peak_state_cells,
        "max_numerator_bits": metrics.max_numerator_bits,
        "max_denominator_bits": metrics.max_denominator_bits,
    }


def tree_digest(trees: tuple[frozenset[tuple[int, int]], ...]) -> str:
    payload = repr(tuple(tuple(sorted(tree)) for tree in trees)).encode()
    return hashlib.sha256(payload).hexdigest()


def reducer_digest(reducers: object) -> str:
    return hashlib.sha256(repr(reducers).encode()).hexdigest()


def verify_circuit_coordinates(max_degree: int) -> list[dict[str, object]]:
    summaries = []
    for degree in range(2, max_degree + 1):
        plan = compile_tree_circuit_coordinates(degree)
        assert plan.relation_rank + len(plan.basis_indices) == len(plan.trees)
        assert len(plan.basis_indices) == factorial(degree - 1)

        if degree <= 4:
            check_compositions = tuple(compositions(degree))
        elif degree == 5:
            check_compositions = ((1,) * 5, (2, 3))
        else:
            check_compositions = ((1,) * 6, (2, 2, 2), (1, 4, 1))
        reconstruction_checks = 0
        for composition in check_compositions:
            direct = tuple(tree_moment(composition, tree) for tree in plan.trees)
            basis = tuple(direct[index] for index in plan.basis_indices)
            assert plan.reconstruct(basis) == direct
            reconstruction_checks += len(direct)

        summaries.append(
            {
                "degree": degree,
                "cayley_trees": len(plan.trees),
                "raw_relations": plan.raw_relation_count,
                "raw_nonzeros": plan.raw_relation_nonzeros,
                "rank": plan.relation_rank,
                "basis_coordinates": len(plan.basis_indices),
                "compression_ratio": [len(plan.trees), len(plan.basis_indices)],
                "max_row_width": plan.max_row_width,
                "max_numerator_bits": plan.max_numerator_bits,
                "max_denominator_bits": plan.max_denominator_bits,
                "reconstruction_checks": reconstruction_checks,
                "tree_sha256": tree_digest(plan.trees),
                "reducer_sha256": reducer_digest(plan.reducers),
            }
        )
    return summaries


def verify_chordal_nbc() -> dict[str, object]:
    small_checks = 0
    accepted_graphs = 0
    for degree in range(2, 5):
        edges = tuple(combinations(range(degree), 2))
        channels = tuple(range(degree))
        for graph in graph_family(edges):
            try:
                plan = compile_chordal_nbc(channels, graph)
            except ValueError:
                continue
            assert len(tuple(plan.trees())) == plan.tree_count
            accepted_graphs += 1
            for composition in compositions(degree):
                assert graph_queries.cell_survives(
                    composition,
                    channels,
                    graph,
                    backend="chordal_nbc",
                ) == cell_survives(composition, channels, graph)
                small_checks += 1

    controls = (
        (
            False,
            ((0, 1), (2, 3), (1, 2), (1, 4), (1, 5)),
        ),
        (
            True,
            ((0, 1), (2, 3), (1, 3), (1, 4), (1, 5)),
        ),
    )
    for expected, edges in controls:
        graph = DependenceGraph(edges)
        assert (
            graph_queries.cell_survives(
                (1,) * 6,
                tuple(range(6)),
                graph,
                backend="chordal_nbc",
            )
            is expected
        )

    return {
        "exhaustive_through_degree": 4,
        "accepted_connected_chordal_graphs": accepted_graphs,
        "cell_checks": small_checks,
        "degree_six_ch4_controls": len(controls),
    }


def verify_direct_nbc() -> dict[str, object]:
    connected_graphs = 0
    generated_trees = 0
    cell_checks = 0
    replay_checks = 0
    for degree in range(2, 5):
        edges = tuple(combinations(range(degree), 2))
        channels = tuple(range(degree))
        for graph in graph_family(edges):
            try:
                plan = compile_direct_nbc(channels, graph)
            except ValueError:
                continue
            connected_graphs += 1
            family = tuple(plan.trees())
            assert len(family) == len(set(family))
            assert all(tree <= frozenset(plan.edge_order) for tree in family)
            generated_trees += len(family)

            for composition in compositions(degree):
                assert graph_queries.cell_survives(
                    composition,
                    channels,
                    graph,
                    backend="direct_nbc",
                ) == cell_survives(composition, channels, graph)
                cell_checks += 1

            target = next(
                tree
                for tree in _labelled_trees(degree)
                if tree <= frozenset(plan.edge_order)
            )
            expansion = plan.replay(target)
            composition = (1,) * degree
            assert adaptive_tree_moment(
                composition,
                plan.reference_covers(target),
            ) == sum(
                (
                    coefficient
                    * adaptive_tree_moment(
                        composition,
                        plan.reference_covers(terminal),
                    )
                    for terminal, coefficient in expansion
                ),
                start=Fraction(),
            )
            replay_checks += 1

    return {
        "exhaustive_through_degree": 4,
        "connected_graphs": connected_graphs,
        "generated_nbc_trees": generated_trees,
        "cell_checks": cell_checks,
        "signed_replay_checks": replay_checks,
    }


def verify_one_centre_gram() -> dict[str, object]:
    graph = DependenceGraph(((0, 1), (0, 2)))
    cell_checks = 0
    for degree in range(2, 5):
        for composition in compositions(degree):
            for centre_vertex in range(degree):
                for leaves in product((1, 2), repeat=degree - 1):
                    channels = list(leaves)
                    channels.insert(centre_vertex, 0)
                    energy = one_centre_gram_energy(
                        composition,
                        channels,
                        graph,
                        centre_channel=0,
                    )
                    assert energy >= 0
                    assert bool(energy) == cell_survives(
                        composition,
                        channels,
                        graph,
                    )
                    assert one_centre_gram_cell_survives(
                        composition,
                        channels,
                        graph,
                        centre_channel=0,
                    ) == bool(energy)
                    cell_checks += 1

    assert one_centre_gram_energy(
        (1, 1, 2),
        (0, 1, 1, 1),
        DependenceGraph(((0, 1),)),
        centre_channel=0,
    ) == Fraction(1, 72)
    return {
        "exhaustive_through_degree": 4,
        "cell_checks": cell_checks,
        "augmentation_fixture": [1, 72],
    }


def graph_family(edges: tuple[tuple[int, int], ...]):
    if len(edges) > 10:
        masks = {
            0,
            (1 << len(edges)) - 1,
            sum(1 << index for index in range(0, len(edges), 2)),
            sum(1 << index for index in range(1, len(edges), 2)),
            (1 << (len(edges) // 2)) - 1,
            ((1 << len(edges)) - 1) ^ ((1 << (len(edges) // 2)) - 1),
        }
    else:
        masks = range(1 << len(edges))
    for mask in sorted(masks):
        yield DependenceGraph(
            edge for index, edge in enumerate(edges) if mask >> index & 1
        )


def has_cycle(support: frozenset[tuple[int, int]]) -> bool:
    parent: dict[int, int] = {}

    def root(vertex: int) -> int:
        parent.setdefault(vertex, vertex)
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for left, right in support:
        left_root = root(left)
        right_root = root(right)
        if left_root == right_root:
            return True
        parent[left_root] = right_root
    return False


def clone_orbit_count(
    composition: tuple[int, ...],
    channels: tuple[int, ...],
) -> int:
    groups: dict[tuple[int, int], list[int]] = {}
    for vertex, (block, channel) in enumerate(
        zip(block_indices(composition), channels)
    ):
        groups.setdefault((block, channel), []).append(vertex)
    moving_groups = tuple(tuple(group) for group in groups.values() if len(group) > 1)
    mappings: list[tuple[int, ...]] = []
    mapping = list(range(len(channels)))

    def visit(group_index: int) -> None:
        if group_index == len(moving_groups):
            mappings.append(tuple(mapping))
            return
        group = moving_groups[group_index]
        for image in permutations(group):
            for source, target in zip(group, image, strict=True):
                mapping[source] = target
            visit(group_index + 1)

    visit(0)
    representatives = set()
    for tree in good_tree_catalogue(composition):
        representatives.add(
            min(
                tuple(
                    sorted(
                        (
                            min(mapping[left], mapping[right]),
                            max(mapping[left], mapping[right]),
                        )
                        for left, right in tree
                    )
                )
                for mapping in mappings
            )
        )
    return len(representatives)


def support_digest(
    support_weights: tuple[tuple[frozenset[tuple[int, int]], Fraction], ...],
) -> str:
    payload = repr(
        tuple(
            (
                tuple(sorted(support)),
                weight.numerator,
                weight.denominator,
            )
            for support, weight in support_weights
        )
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_support_compiler(max_degree: int) -> list[dict[str, object]]:
    cases = [
        ("q121_repeated", (1, 2, 1), (0, 1, 1, 2)),
        ("strict4", (1, 1, 1, 1), (0, 1, 2, 3)),
    ]
    if max_degree >= 5:
        cases.append(("strict5", (1,) * 5, tuple(range(5))))
    if max_degree >= 6:
        cases.extend(
            (
                ("strict6", (1,) * 6, tuple(range(6))),
                ("q141_coarse_pair", (1, 4, 1), (4, 1, 2, 3, 3, 0)),
                ("q222_repeated", (2, 2, 2), (0, 0, 1, 1, 2, 2)),
            )
        )

    summaries = []
    for name, composition, channels in cases:
        compiled = compile_projected_support(composition, channels)
        raw_supports = tuple(support for support, _ in compiled.support_weights)
        assert all(weight > 0 for _, weight in compiled.support_weights)
        assert compiled.zdd.enumerate_supports() == compiled.minimal_supports
        assert all(
            not left <= right and not right <= left
            for index, left in enumerate(compiled.minimal_supports)
            for right in compiled.minimal_supports[index + 1 :]
        )

        channel_values = tuple(sorted(set(channels)))
        edges = tuple(combinations(channel_values, 2))
        query_checks = 0
        for graph in graph_family(edges):
            expected = any(support <= graph.edges for support in raw_supports)
            assert expected == compiled.zdd.contains_subset(graph)
            witness = compiled.zdd.one_witness(graph)
            assert (witness is not None) == expected
            if witness is not None:
                assert witness in compiled.minimal_supports
                assert witness <= graph.edges
            query_checks += 1

        cyclic_supports = sum(has_cycle(support) for support in raw_supports)
        weights = tuple(weight for _, weight in compiled.support_weights)
        summaries.append(
            {
                "case": name,
                "composition": composition,
                "channels": channels,
                "trees_scanned": compiled.trees_scanned,
                "good_trees": compiled.good_tree_count,
                "good_tree_clone_orbits": clone_orbit_count(composition, channels),
                "projected_supports": len(compiled.support_weights),
                "minimal_supports": len(compiled.minimal_supports),
                "cyclic_supports": cyclic_supports,
                "zdd_nodes_including_terminals": compiled.zdd.node_count,
                "max_weight_numerator_bits": max(
                    (weight.numerator.bit_length() for weight in weights),
                    default=0,
                ),
                "max_weight_denominator_bits": max(
                    (weight.denominator.bit_length() for weight in weights),
                    default=0,
                ),
                "query_checks": query_checks,
                "query_scope": "all_graphs"
                if len(edges) <= 10
                else "deterministic_sample",
                "support_sha256": support_digest(compiled.support_weights),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-degree",
        type=int,
        default=6,
        choices=range(4, 7),
        help="maximum circuit/support census degree; fixed Young controls reach 9",
    )
    return parser.parse_args()


def main() -> None:
    if not __debug__:
        raise RuntimeError("run without -O: this verifier is assertion-driven")
    args = parse_args()
    circuit_rows = verify_circuit_coordinates(args.max_degree)
    support_rows = verify_support_compiler(args.max_degree)
    summary = {
        "field": "Q",
        "status": "bounded_experimental_evidence",
        "compiler_max_degree": args.max_degree,
        "young_dp": verify_young_dp(),
        "circuit_coordinates": {
            "compiled_through_degree": args.max_degree,
            "rows": circuit_rows,
        },
        "support_compiler": {
            "case_degrees": sorted({sum(row["composition"]) for row in support_rows}),
            "rows": support_rows,
        },
        "chordal_nbc": verify_chordal_nbc(),
        "direct_nbc": verify_direct_nbc(),
        "one_centre_gram": verify_one_centre_gram(),
        "not_implemented": {
            "generic_direct_nbc_certificates": "byte schema and independent checker remain open",
            "width_local_chordal": "repeatedly closed compact state remains open",
            "repeated_centre_star": "outside the proved one-centre Gram theorem",
        },
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
