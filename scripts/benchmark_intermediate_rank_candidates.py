#!/usr/bin/env python3
"""Runtime and memory profiles for the post-IR1 candidates.

This script contains no correctness claims; run
``verify_intermediate_rank_candidates.py`` separately.  Timings are warmed,
calibrated in-process medians, while retained and peak Python allocations are
sampled in a separate ``tracemalloc`` run so tracing does not distort time.

Run from the tensorsig repository root:

    uv run python scripts/benchmark_intermediate_rank_candidates.py
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import tracemalloc
from fractions import Fraction
from itertools import combinations
from math import factorial, prod
from random import Random
from statistics import median
from time import perf_counter_ns

from tensorsig.experimental.offset_coded_signatures.circuit_coordinates import (
    compile_tree_circuit_coordinates,
)
from tensorsig.experimental.offset_coded_signatures.intermediate_rank import (
    _labelled_trees,
    _project_tree,
    cell_survives,
)
from tensorsig.experimental.offset_coded_signatures.support_compiler import (
    LexZDD,
    _minimal_antichain,
    compile_projected_support,
)
from tensorsig.offset_coded_signatures import graph_queries
from tensorsig.offset_coded_signatures._graph._graph_common import (
    ResourceLimit,
)
from tensorsig.offset_coded_signatures._graph.adaptive import (
    adaptive_tree_moment,
    composition_is_universally_zero,
    tree_moment_backend,
)
from tensorsig.offset_coded_signatures._graph.literal_energy import (
    literal_tree_energy,
)
from tensorsig.offset_coded_signatures._graph.one_centre_gram import (
    one_centre_gram_cell_survives,
    one_centre_gram_energy,
)
from tensorsig.offset_coded_signatures._graph.star_gram import (
    star_trace_gram_energy,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    tree_moment,
)
from tensorsig.offset_coded_signatures._graph.young_dp import (
    YoungDpMetrics,
    young_dp_tree_moment,
)
from tensorsig.offset_coded_signatures.graph_queries import (
    CellQueryMetrics,
    compile_chordal_nbc,
    compile_direct_nbc,
)
from tensorsig.offset_coded_signatures.graph_queries import (
    cell_survives as adaptive_cell_survives,
)
from tensorsig.offset_coded_signatures.quotients import DependenceGraph


def measure(call, repeats: int) -> dict[str, object]:
    """Return calibrated per-call medians and separate Python allocations."""
    result = call()
    del result

    loops = 1
    while True:
        gc.collect()
        started = perf_counter_ns()
        for _ in range(loops):
            result = call()
        elapsed = perf_counter_ns() - started
        del result
        if elapsed >= 10_000_000 or loops >= 1 << 16:
            break
        loops *= 2

    durations = []
    for _ in range(repeats):
        gc.collect()
        started = perf_counter_ns()
        for _ in range(loops):
            result = call()
        durations.append((perf_counter_ns() - started) / loops)
        del result

    gc.collect()
    tracemalloc.start()
    result = call()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {
        "median_ms": median(durations) / 1_000_000,
        "samples_ms": [duration / 1_000_000 for duration in durations],
        "timing_loops": loops,
        "retained_python_bytes": retained,
        "peak_python_bytes": peak,
    }


def fraction_row(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def path_tree(degree: int) -> tuple[tuple[int, int], ...]:
    return tuple((vertex, vertex + 1) for vertex in range(degree - 1))


def star_tree(degree: int) -> tuple[tuple[int, int], ...]:
    return tuple((0, vertex) for vertex in range(1, degree))


def mixed_path_tree(degree: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (vertex + 1, vertex) if vertex % 2 else (vertex, vertex + 1)
        for vertex in range(degree - 1)
    )


def adaptive_cayley_cell_survives(
    composition: tuple[int, ...],
    channels: tuple[int, ...],
    graph: DependenceGraph,
) -> bool:
    """Fair Cayley baseline using the same adaptive scalar as NBC."""
    return any(
        _project_tree(tree, channels) <= graph.edges
        and adaptive_tree_moment(composition, tree)
        for tree in _labelled_trees(sum(composition))
    )


def young_benchmarks(max_degree: int, repeats: int) -> list[dict[str, object]]:
    rows = []
    for degree in range(2, max_degree + 1):
        compositions = {(1,) * degree}
        if degree >= 4:
            compositions.update(
                {
                    (degree // 2, degree - degree // 2),
                    (degree - 2, 2),
                    (1, degree - 2, 1),
                }
            )
        compositions = {
            composition
            for composition in compositions
            if not composition_is_universally_zero(composition)
        }
        trees = [
            ("path", path_tree(degree)),
            ("mixed_path", mixed_path_tree(degree)),
        ]
        if degree >= 4:
            trees.append(("star", star_tree(degree)))

        for composition in sorted(compositions):
            young_size = prod(factorial(part) for part in composition)
            for shape, tree in trees:
                metrics = YoungDpMetrics()
                dp_value = young_dp_tree_moment(
                    composition,
                    tree,
                    metrics=metrics,
                )
                explicit_value = tree_moment(composition, tree)
                explicit_profile = measure(
                    lambda c=composition, t=tree: tree_moment(c, t),
                    repeats,
                )
                dp_profile = measure(
                    lambda c=composition, t=tree: young_dp_tree_moment(c, t),
                    repeats,
                )
                adaptive_profile = measure(
                    lambda c=composition, t=tree: adaptive_tree_moment(c, t),
                    repeats,
                )
                explicit_ms = float(explicit_profile["median_ms"])
                dp_ms = float(dp_profile["median_ms"])
                adaptive_ms = float(adaptive_profile["median_ms"])
                rows.append(
                    {
                        "degree": degree,
                        "composition": composition,
                        "tree_shape": shape,
                        "young_group_size": young_size,
                        "explicit_value": fraction_row(explicit_value),
                        "dp_value": fraction_row(dp_value),
                        "explicit": explicit_profile,
                        "young_dp": dp_profile,
                        "adaptive": adaptive_profile,
                        "adaptive_backend": tree_moment_backend(composition),
                        "explicit_over_dp_time": explicit_ms / dp_ms,
                        "explicit_over_adaptive_time": explicit_ms / adaptive_ms,
                        "dp_state_cells": metrics.state_cells,
                        "dp_peak_state_cells": metrics.peak_state_cells,
                        "dp_max_numerator_bits": metrics.max_numerator_bits,
                        "dp_max_denominator_bits": metrics.max_denominator_bits,
                    }
                )
    return rows


def zero_prefilter_benchmarks(
    max_degree: int,
    repeats: int,
) -> list[dict[str, object]]:
    rows = []
    for degree in range(2, max_degree + 1):
        compositions = [(degree,)]
        if degree >= 4 and (degree - 1) % 2:
            compositions.append((1, degree - 1))
        tree = path_tree(degree)
        for composition in compositions:
            explicit = measure(
                lambda c=composition, t=tree: tree_moment(c, t),
                repeats,
            )
            adaptive = measure(
                lambda c=composition, t=tree: adaptive_tree_moment(c, t),
                repeats,
            )
            rows.append(
                {
                    "degree": degree,
                    "composition": composition,
                    "explicit": explicit,
                    "adaptive_zero": adaptive,
                    "explicit_over_adaptive_time": float(explicit["median_ms"])
                    / float(adaptive["median_ms"]),
                }
            )
    return rows


def circuit_benchmarks(max_degree: int, repeats: int) -> list[dict[str, object]]:
    rows = []
    for degree in range(2, max_degree + 1):
        compile_profile = measure(
            lambda d=degree: compile_tree_circuit_coordinates(d),
            repeats,
        )
        plan = compile_tree_circuit_coordinates(degree)
        reconstruct_profile = measure(
            lambda p=plan: p.reconstruct(
                tuple(Fraction(index + 1) for index in range(len(p.basis_indices)))
            ),
            repeats,
        )
        compositions = [(1,) * degree]
        if degree >= 4:
            compositions.append((degree - 2, 2))
        scalar_rows = []
        for composition in compositions:
            full_explicit = measure(
                lambda p=plan, q=composition: tuple(
                    tree_moment(q, tree) for tree in p.trees
                ),
                repeats,
            )
            full_adaptive = measure(
                lambda p=plan, q=composition: tuple(
                    adaptive_tree_moment(q, tree) for tree in p.trees
                ),
                repeats,
            )
            basis_explicit = measure(
                lambda p=plan, q=composition: p.reconstruct(
                    tuple(tree_moment(q, tree) for tree in p.basis_trees)
                ),
                repeats,
            )
            basis_adaptive = measure(
                lambda p=plan, q=composition: p.reconstruct(
                    tuple(adaptive_tree_moment(q, tree) for tree in p.basis_trees)
                ),
                repeats,
            )
            best_reduced_ms = min(
                float(basis_explicit["median_ms"]),
                float(basis_adaptive["median_ms"]),
            )
            full_ms = float(full_explicit["median_ms"])
            full_adaptive_ms = float(full_adaptive["median_ms"])
            basis_adaptive_ms = float(basis_adaptive["median_ms"])
            circuit_saving = full_adaptive_ms - basis_adaptive_ms
            scalar_rows.append(
                {
                    "composition": composition,
                    "adaptive_backend": tree_moment_backend(composition),
                    "full_explicit": full_explicit,
                    "full_adaptive": full_adaptive,
                    "basis_explicit_and_reconstruct": basis_explicit,
                    "basis_adaptive_and_reconstruct": basis_adaptive,
                    "full_explicit_over_best_reduced_time": full_ms / best_reduced_ms,
                    "full_adaptive_over_basis_adaptive_time": full_adaptive_ms
                    / basis_adaptive_ms,
                    "compile_break_even_vectors": (
                        float(compile_profile["median_ms"]) / circuit_saving
                        if circuit_saving > 0
                        else None
                    ),
                }
            )
        rows.append(
            {
                "degree": degree,
                "cayley_trees": len(plan.trees),
                "basis_coordinates": len(plan.basis_indices),
                "raw_relations": plan.raw_relation_count,
                "relation_rank": plan.relation_rank,
                "max_row_width": plan.max_row_width,
                "compile": compile_profile,
                "reconstruct_only": reconstruct_profile,
                "scalar_calls_avoided": len(plan.trees) - len(plan.basis_indices),
                "scalar_rows": scalar_rows,
            }
        )
    return rows


def query_graphs(channels: tuple[int, ...]) -> tuple[DependenceGraph, ...]:
    edges = tuple(combinations(sorted(set(channels)), 2))
    if len(edges) <= 10:
        masks = range(1 << len(edges))
    else:
        random = Random(0)
        sampled = {0, (1 << len(edges)) - 1}
        for size in range(1, len(edges)):
            for _ in range(4):
                chosen = random.sample(range(len(edges)), size)
                sampled.add(sum(1 << index for index in chosen))
        masks = sorted(sampled)
    return tuple(
        DependenceGraph(edge for index, edge in enumerate(edges) if mask >> index & 1)
        for mask in masks
    )


_COMPLETION_TARGET_DENSITIES = (0.30, 0.50, 0.70)


def sampled_connected_nonchordal_graphs(
    degree: int,
    target_density: float,
    sample_count: int,
    seed: int,
) -> tuple[DependenceGraph, ...]:
    """Return seeded connected density samples with an induced four-cycle."""
    graphs = []
    for sample in range(sample_count):
        random = Random(f"{seed}:{degree}:{target_density:.2f}:{sample}")
        cycle = random.sample(range(degree), 4)
        edges = {
            tuple(sorted((cycle[index], cycle[(index + 1) % 4]))) for index in range(4)
        }
        forbidden = {
            tuple(sorted((cycle[0], cycle[2]))),
            tuple(sorted((cycle[1], cycle[3]))),
        }
        connected = list(cycle)
        for vertex in sorted(set(range(degree)) - set(cycle)):
            edges.add(tuple(sorted((vertex, random.choice(connected)))))
            connected.append(vertex)

        candidates = [
            edge
            for edge in combinations(range(degree), 2)
            if edge not in edges and edge not in forbidden
        ]
        possible_edges = degree * (degree - 1) // 2
        target_edges = round(target_density * possible_edges)
        extra_count = min(len(candidates), max(0, target_edges - len(edges)))
        edges.update(random.sample(candidates, extra_count))
        graphs.append(DependenceGraph(edges))
    return tuple(graphs)


def min_fill_completion(
    degree: int,
    edges: frozenset[tuple[int, int]],
) -> tuple[
    frozenset[tuple[int, int]],
    tuple[tuple[int, int], ...],
    tuple[int, ...],
]:
    """Return a deterministic greedy minimum-fill chordal completion."""
    adjacency = [set() for _ in range(degree)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    remaining = set(range(degree))
    fill_edges: set[tuple[int, int]] = set()
    elimination_order = []
    while remaining:

        def score(vertex: int) -> tuple[int, int, int]:
            neighbors = adjacency[vertex] & remaining
            missing = sum(
                right not in adjacency[left]
                for left, right in combinations(sorted(neighbors), 2)
            )
            return missing, len(neighbors), vertex

        vertex = min(remaining, key=score)
        neighbors = sorted(adjacency[vertex] & remaining)
        for left, right in combinations(neighbors, 2):
            if right not in adjacency[left]:
                adjacency[left].add(right)
                adjacency[right].add(left)
                fill_edges.add((left, right))
        remaining.remove(vertex)
        elimination_order.append(vertex)

    return edges | fill_edges, tuple(sorted(fill_edges)), tuple(elimination_order)


def completion_prefilter_then_direct(
    composition: tuple[int, ...],
    channels: tuple[int, ...],
    graph: DependenceGraph,
) -> bool:
    """Decide the original graph exactly via a one-sided completion prefilter."""
    completed_edges, _, _ = min_fill_completion(len(channels), graph.edges)
    completed_graph = DependenceGraph(completed_edges)
    if not graph_queries.cell_survives(
        composition, channels, completed_graph, backend="chordal_nbc"
    ):
        return False
    return graph_queries.cell_survives(
        composition, channels, graph, backend="direct_nbc"
    )


def tuple_witness(
    supports: tuple[frozenset[tuple[int, int]], ...],
    graph: DependenceGraph,
) -> frozenset[tuple[int, int]] | None:
    return next((support for support in supports if support <= graph.edges), None)


def support_benchmarks(max_degree: int, repeats: int) -> list[dict[str, object]]:
    cases = [((1, 2, 1), (0, 1, 1, 2))]
    if max_degree >= 5:
        cases.append(((1,) * 5, tuple(range(5))))
    if max_degree >= 6:
        cases.extend(
            (
                ((1,) * 6, tuple(range(6))),
                ((1, 4, 1), (4, 1, 2, 3, 3, 0)),
                ((2, 2, 2), (0, 0, 1, 1, 2, 2)),
            )
        )

    rows = []
    for composition, channels in cases:
        compile_profile = measure(
            lambda c=composition, h=channels: compile_projected_support(c, h),
            repeats,
        )
        compiled = compile_projected_support(composition, channels)
        raw_supports = tuple(support for support, _ in compiled.support_weights)
        minimize_profile = measure(
            lambda supports=raw_supports: _minimal_antichain(supports),
            repeats,
        )
        zdd_build_profile = measure(
            lambda supports=compiled.minimal_supports: LexZDD.build(supports),
            repeats,
        )
        graphs = query_graphs(channels)
        direct_profile = measure(
            lambda c=composition, h=channels, gs=graphs: tuple(
                cell_survives(c, h, graph) for graph in gs
            ),
            repeats,
        )
        tuple_profile = measure(
            lambda supports=compiled.minimal_supports, gs=graphs: tuple(
                tuple_witness(supports, graph) is not None for graph in gs
            ),
            repeats,
        )
        tuple_witness_profile = measure(
            lambda supports=compiled.minimal_supports, gs=graphs: tuple(
                tuple_witness(supports, graph) for graph in gs
            ),
            repeats,
        )
        zdd_profile = measure(
            lambda zdd=compiled.zdd, gs=graphs: tuple(
                zdd.contains_subset(graph) for graph in gs
            ),
            repeats,
        )
        witness_profile = measure(
            lambda zdd=compiled.zdd, gs=graphs: tuple(
                zdd.one_witness(graph) for graph in gs
            ),
            repeats,
        )
        direct_per_query = float(direct_profile["median_ms"]) / len(graphs)
        tuple_per_query = float(tuple_profile["median_ms"]) / len(graphs)
        zdd_per_query = float(zdd_profile["median_ms"]) / len(graphs)
        tuple_saving = direct_per_query - tuple_per_query
        zdd_saving = direct_per_query - zdd_per_query
        outcomes = tuple(
            tuple_witness(compiled.minimal_supports, graph) is not None
            for graph in graphs
        )
        rows.append(
            {
                "composition": composition,
                "channels": channels,
                "trees_scanned": compiled.trees_scanned,
                "good_trees": compiled.good_tree_count,
                "projected_supports": len(compiled.support_weights),
                "minimal_supports": len(compiled.minimal_supports),
                "scalar_backend": compiled.scalar_backend,
                "zdd_nodes_including_terminals": compiled.zdd.node_count,
                "cayley_fed_compile": compile_profile,
                "minimize_only": minimize_profile,
                "zdd_build_only": zdd_build_profile,
                "query_count": len(graphs),
                "positive_queries": sum(outcomes),
                "negative_queries": len(outcomes) - sum(outcomes),
                "direct_streaming_queries": direct_profile,
                "minimal_tuple_queries": tuple_profile,
                "minimal_tuple_witnesses": tuple_witness_profile,
                "zdd_queries": zdd_profile,
                "zdd_witnesses": witness_profile,
                "tuple_estimated_break_even_queries": (
                    float(compile_profile["median_ms"]) / tuple_saving
                    if tuple_saving > 0
                    else None
                ),
                "zdd_estimated_break_even_queries": (
                    float(compile_profile["median_ms"]) / zdd_saving
                    if zdd_saving > 0
                    else None
                ),
            }
        )
    return rows


def chordal_nbc_benchmarks(
    max_degree: int,
    repeats: int,
) -> list[dict[str, object]]:
    cases = [
        (
            "q121_zero_star",
            (1, 2, 1),
            tuple(range(4)),
            ((0, 1), (0, 2), (0, 3)),
        ),
        (
            "q121_live_tree",
            (1, 2, 1),
            tuple(range(4)),
            ((0, 1), (0, 2), (1, 3)),
        ),
    ]
    if max_degree >= 6:
        cases.extend(
            (
                (
                    "ch4_zero_tree",
                    (1,) * 6,
                    tuple(range(6)),
                    ((0, 1), (2, 3), (1, 2), (1, 4), (1, 5)),
                ),
                (
                    "ch4_live_tree",
                    (1,) * 6,
                    tuple(range(6)),
                    ((0, 1), (2, 3), (1, 3), (1, 4), (1, 5)),
                ),
                (
                    "dense_q42",
                    (4, 2),
                    tuple(range(6)),
                    tuple(combinations(range(6), 2)),
                ),
            )
        )

    rows = []
    for name, composition, channels, edges in cases:
        graph = DependenceGraph(edges)
        plan_profile = measure(
            lambda h=channels, g=graph: compile_chordal_nbc(h, g),
            repeats,
        )
        plan = compile_chordal_nbc(channels, graph)
        metrics = CellQueryMetrics()
        result = graph_queries.cell_survives(
            composition,
            channels,
            graph,
            metrics=metrics,
            backend="chordal_nbc",
        )
        direct_profile = measure(
            lambda c=composition, h=channels, g=graph: cell_survives(c, h, g),
            repeats,
        )
        nbc_profile = measure(
            lambda c=composition, h=channels, g=graph: graph_queries.cell_survives(
                c,
                h,
                g,
                backend="chordal_nbc",
            ),
            repeats,
        )
        rows.append(
            {
                "case": name,
                "composition": composition,
                "channels": channels,
                "survives": result,
                "peo": plan.peo,
                "parent_trees": plan.tree_count,
                "trees_examined": metrics.trees_examined,
                "max_bag_load": plan.max_bag_load,
                "scalar_backend": metrics.scalar_backend,
                "plan_compile": plan_profile,
                "cayley_streaming_query": direct_profile,
                "chordal_nbc_query": nbc_profile,
                "cayley_over_chordal_nbc_time": float(direct_profile["median_ms"])
                / float(nbc_profile["median_ms"]),
            }
        )
    return rows


def direct_nbc_benchmarks(
    max_degree: int,
    repeats: int,
) -> list[dict[str, object]]:
    cases = [
        (
            "q121_zero_star",
            (1, 2, 1),
            tuple(range(4)),
            ((0, 1), (0, 2), (0, 3)),
        ),
        (
            "q121_live_tree",
            (1, 2, 1),
            tuple(range(4)),
            ((0, 1), (0, 2), (1, 3)),
        ),
    ]
    if max_degree >= 6:
        cases.extend(
            (
                (
                    "ch4_zero_tree",
                    (1,) * 6,
                    tuple(range(6)),
                    ((0, 1), (2, 3), (1, 2), (1, 4), (1, 5)),
                ),
                (
                    "ch4_live_tree",
                    (1,) * 6,
                    tuple(range(6)),
                    ((0, 1), (2, 3), (1, 3), (1, 4), (1, 5)),
                ),
                (
                    "nonchordal_cycle",
                    (4, 2),
                    tuple(range(6)),
                    ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)),
                ),
                (
                    "dense_complete",
                    (4, 2),
                    tuple(range(6)),
                    tuple(combinations(range(6), 2)),
                ),
            )
        )

    rows = []
    for name, composition, channels, edges in cases:
        graph = DependenceGraph(edges)
        plan_profile = measure(
            lambda h=channels, g=graph: compile_direct_nbc(h, g),
            repeats,
        )
        plan = compile_direct_nbc(channels, graph)
        generation_profile = measure(
            lambda p=plan: sum(1 for _ in p.trees()),
            repeats,
        )
        family_count = sum(1 for _ in plan.trees())
        result = graph_queries.cell_survives(
            composition, channels, graph, backend="direct_nbc"
        )
        assert result == cell_survives(composition, channels, graph)
        assert result == adaptive_cayley_cell_survives(composition, channels, graph)

        witness_ordinal = next(
            (
                index
                for index, tree in enumerate(plan.trees(), start=1)
                if adaptive_tree_moment(
                    composition,
                    plan.reference_covers(tree),
                )
            ),
            None,
        )
        legacy_profile = measure(
            lambda c=composition, h=channels, g=graph: cell_survives(c, h, g),
            repeats,
        )
        adaptive_profile = measure(
            lambda c=composition, h=channels, g=graph: adaptive_cayley_cell_survives(
                c,
                h,
                g,
            ),
            repeats,
        )
        nbc_profile = measure(
            lambda c=composition, h=channels, g=graph: graph_queries.cell_survives(
                c,
                h,
                g,
                backend="direct_nbc",
            ),
            repeats,
        )

        replay_target = None
        replay_expansion = None
        for tree in _labelled_trees(sum(composition)):
            if tree <= frozenset(plan.edge_order):
                expansion = plan.replay(tree)
                if expansion != ((tree, 1),):
                    replay_target = tree
                    replay_expansion = expansion
                    break
        replay_profile = (
            measure(lambda p=plan, t=replay_target: p.replay(t), repeats)
            if replay_target is not None
            else None
        )
        rows.append(
            {
                "case": name,
                "composition": composition,
                "edge_count": len(plan.edge_order),
                "survives": result,
                "nbc_trees": family_count,
                "witness_ordinal": witness_ordinal,
                "scalar_backend": tree_moment_backend(composition),
                "plan_compile": plan_profile,
                "family_generation": generation_profile,
                "legacy_cayley_explicit": legacy_profile,
                "cayley_adaptive": adaptive_profile,
                "direct_nbc": nbc_profile,
                "adaptive_cayley_over_direct_nbc_time": float(
                    adaptive_profile["median_ms"]
                )
                / float(nbc_profile["median_ms"]),
                "replay": replay_profile,
                "replay_terminal_terms": (
                    len(replay_expansion) if replay_expansion is not None else 0
                ),
                "replay_max_coefficient_bits": max(
                    (
                        abs(coefficient).bit_length()
                        for _, coefficient in replay_expansion or ()
                    ),
                    default=0,
                ),
            }
        )
    return rows


def one_centre_gram_benchmarks(
    max_degree: int,
    repeats: int,
) -> list[dict[str, object]]:
    cases = [
        (
            "strict_zero",
            (1, 1, 1, 1),
            (0, 1, 2, 3),
            0,
            ((0, 1), (0, 2), (0, 3)),
        ),
        (
            "augmentation_live",
            (1, 1, 2),
            (0, 1, 1, 1),
            0,
            ((0, 1),),
        ),
    ]
    if max_degree >= 5:
        cases.append(
            (
                "strict_live",
                (1,) * 5,
                tuple(range(5)),
                0,
                tuple((0, leaf) for leaf in range(1, 5)),
            )
        )
    if max_degree >= 6:
        cases.extend(
            (
                (
                    "clustered_repeated_leaf",
                    (1, 4, 1),
                    (4, 1, 2, 3, 3, 0),
                    4,
                    ((4, 0), (4, 1), (4, 2), (4, 3)),
                ),
                (
                    "spread_repeated_leaf",
                    (2, 2, 2),
                    (0, 1, 1, 2, 1, 3),
                    0,
                    ((0, 1), (0, 2), (0, 3)),
                ),
            )
        )

    rows = []
    for name, composition, channels, centre, edges in cases:
        graph = DependenceGraph(edges)
        energy = one_centre_gram_energy(
            composition,
            channels,
            graph,
            centre_channel=centre,
        )
        result = bool(energy)
        assert result == cell_survives(composition, channels, graph)
        assert result == graph_queries.cell_survives(
            composition, channels, graph, backend="direct_nbc"
        )
        assert result == graph_queries.cell_survives(
            composition, channels, graph, backend="chordal_nbc"
        )

        legacy_profile = measure(
            lambda c=composition, h=channels, g=graph: cell_survives(c, h, g),
            repeats,
        )
        adaptive_profile = measure(
            lambda c=composition, h=channels, g=graph: adaptive_cayley_cell_survives(
                c,
                h,
                g,
            ),
            repeats,
        )
        direct_profile = measure(
            lambda c=composition, h=channels, g=graph: graph_queries.cell_survives(
                c,
                h,
                g,
                backend="direct_nbc",
            ),
            repeats,
        )
        chordal_profile = measure(
            lambda c=composition, h=channels, g=graph: graph_queries.cell_survives(
                c,
                h,
                g,
                backend="chordal_nbc",
            ),
            repeats,
        )
        energy_profile = measure(
            lambda c=composition, h=channels, g=graph, centre=centre: (
                one_centre_gram_energy(
                    c,
                    h,
                    g,
                    centre_channel=centre,
                )
            ),
            repeats,
        )
        decision_profile = measure(
            lambda c=composition, h=channels, g=graph, centre=centre: (
                one_centre_gram_cell_survives(
                    c,
                    h,
                    g,
                    centre_channel=centre,
                )
            ),
            repeats,
        )
        rows.append(
            {
                "case": name,
                "composition": composition,
                "channels": channels,
                "centre_channel": centre,
                "survives": result,
                "energy": fraction_row(energy),
                "legacy_cayley_explicit": legacy_profile,
                "cayley_adaptive": adaptive_profile,
                "direct_nbc": direct_profile,
                "chordal_nbc": chordal_profile,
                "gram_energy": energy_profile,
                "gram_decision": decision_profile,
                "adaptive_cayley_over_gram_time": float(adaptive_profile["median_ms"])
                / float(decision_profile["median_ms"]),
                "direct_nbc_over_gram_time": float(direct_profile["median_ms"])
                / float(decision_profile["median_ms"]),
                "chordal_nbc_over_gram_time": float(chordal_profile["median_ms"])
                / float(decision_profile["median_ms"]),
            }
        )
    return rows


def nonchordal_completion_benchmarks(
    max_degree: int,
    sample_count: int,
    seed: int,
    decision_family_limit: int,
    repeats: int,
) -> list[dict[str, object]]:
    """Compare exact original-graph NBC with non-equivalent completions."""
    rows = []
    for degree in range(4, max_degree + 1):
        composition = (1, 1, degree - 2)
        channels = tuple(range(degree))
        possible_edges = degree * (degree - 1) // 2
        for target_density, sample, graph in (
            (target_density, sample, graph)
            for target_density in _COMPLETION_TARGET_DENSITIES
            for sample, graph in enumerate(
                sampled_connected_nonchordal_graphs(
                    degree,
                    target_density,
                    sample_count,
                    seed,
                )
            )
        ):
            try:
                compile_chordal_nbc(channels, graph)
            except ValueError as error:
                assert "chordal" in str(error)
            else:
                raise AssertionError("sampled graph must be nonchordal")

            completed_edges, fill_edges, min_fill_order = min_fill_completion(
                degree,
                graph.edges,
            )
            completed_graph = DependenceGraph(completed_edges)

            completion_construction = measure(
                lambda d=degree, edges=graph.edges: min_fill_completion(d, edges),
                repeats,
            )

            original_compile = measure(
                lambda h=channels, g=graph: compile_direct_nbc(h, g),
                repeats,
            )
            completed_direct_compile = measure(
                lambda h=channels, g=completed_graph: compile_direct_nbc(h, g),
                repeats,
            )
            completed_chordal_compile = measure(
                lambda h=channels, g=completed_graph: compile_chordal_nbc(h, g),
                repeats,
            )
            original_plan = compile_direct_nbc(channels, graph)
            completed_direct_plan = compile_direct_nbc(channels, completed_graph)
            completed_chordal_plan = compile_chordal_nbc(channels, completed_graph)

            original_generation = measure(
                lambda p=original_plan: sum(1 for _ in p.trees()),
                repeats,
            )
            completed_direct_generation = measure(
                lambda p=completed_direct_plan: sum(1 for _ in p.trees()),
                repeats,
            )
            completed_chordal_generation = measure(
                lambda p=completed_chordal_plan: sum(1 for _ in p.trees()),
                repeats,
            )
            original_family_count = sum(1 for _ in original_plan.trees())
            completed_direct_family_count = sum(
                1 for _ in completed_direct_plan.trees()
            )
            completed_chordal_family_count = sum(
                1 for _ in completed_chordal_plan.trees()
            )
            assert completed_direct_family_count == completed_chordal_family_count

            original_decision = None
            original_survives = None
            if original_family_count <= decision_family_limit:
                original_decision = measure(
                    lambda c=composition, h=channels, g=graph: (
                        graph_queries.cell_survives(c, h, g, backend="direct_nbc")
                    ),
                    repeats,
                )
                original_survives = graph_queries.cell_survives(
                    composition,
                    channels,
                    graph,
                    backend="direct_nbc",
                )

            completed_direct_decision = None
            completed_chordal_decision = None
            completed_survives = None
            if completed_direct_family_count <= decision_family_limit:
                completed_direct_decision = measure(
                    lambda c=composition, h=channels, g=completed_graph: (
                        graph_queries.cell_survives(c, h, g, backend="direct_nbc")
                    ),
                    repeats,
                )
                completed_chordal_decision = measure(
                    lambda c=composition, h=channels, g=completed_graph: (
                        graph_queries.cell_survives(c, h, g, backend="chordal_nbc")
                    ),
                    repeats,
                )
                completed_survives = graph_queries.cell_survives(
                    composition,
                    channels,
                    completed_graph,
                    backend="direct_nbc",
                )
                assert completed_survives == graph_queries.cell_survives(
                    composition,
                    channels,
                    completed_graph,
                    backend="chordal_nbc",
                )

            exact_pipeline_decision = None
            if (
                original_family_count <= decision_family_limit
                and completed_chordal_family_count <= decision_family_limit
            ):
                exact_pipeline_decision = measure(
                    lambda c=composition, h=channels, g=graph: (
                        completion_prefilter_then_direct(c, h, g)
                    ),
                    repeats,
                )
                assert completion_prefilter_then_direct(
                    composition,
                    channels,
                    graph,
                ) == graph_queries.cell_survives(
                    composition, channels, graph, backend="direct_nbc"
                )

            original_cayley = None
            completed_cayley = None
            if degree <= 6:
                original_cayley = measure(
                    lambda c=composition, h=channels, g=graph: (
                        adaptive_cayley_cell_survives(c, h, g)
                    ),
                    repeats,
                )
                completed_cayley = measure(
                    lambda c=composition, h=channels, g=completed_graph: (
                        adaptive_cayley_cell_survives(c, h, g)
                    ),
                    repeats,
                )
                original_cayley_survives = adaptive_cayley_cell_survives(
                    composition,
                    channels,
                    graph,
                )
                completed_cayley_survives = adaptive_cayley_cell_survives(
                    composition,
                    channels,
                    completed_graph,
                )
                assert original_survives in (None, original_cayley_survives)
                assert completed_survives in (None, completed_cayley_survives)
                original_survives = original_cayley_survives
                completed_survives = completed_cayley_survives

            outcome_class = (
                f"{int(original_survives)}{int(completed_survives)}"
                if original_survives is not None and completed_survives is not None
                else None
            )
            assert outcome_class != "10"
            assert outcome_class in (None, "00", "01", "11")

            rows.append(
                {
                    "degree": degree,
                    "sample": sample,
                    "target_density": target_density,
                    "realized_density": len(graph.edges) / possible_edges,
                    "completed_density": len(completed_graph.edges) / possible_edges,
                    "composition": composition,
                    "original_edges": tuple(sorted(graph.edges)),
                    "fill_edges": fill_edges,
                    "completed_edges": tuple(sorted(completed_graph.edges)),
                    "min_fill_elimination_order": min_fill_order,
                    "completion_construction": completion_construction,
                    "completed_graph_semantics": (
                        "different_supergraph_not_equivalent_to_original"
                    ),
                    "original_survives": original_survives,
                    "completed_survives": completed_survives,
                    "monotonic_semantics": (
                        "00/01/11_only; 10_is_impossible_under_edge_addition"
                    ),
                    "outcome_class": outcome_class,
                    "outcome_class_status": (
                        "measured"
                        if outcome_class is not None
                        else "unavailable_due_to_skipped_decision"
                    ),
                    "survival_changed_by_completion": (
                        outcome_class == "01" if outcome_class is not None else None
                    ),
                    "original_direct_nbc": {
                        "family_count": original_family_count,
                        "plan_compile": original_compile,
                        "full_family_generation": original_generation,
                        "decision_status": (
                            "measured"
                            if original_decision is not None
                            else "skipped_family_count_limit"
                        ),
                        "decision": original_decision,
                    },
                    "completed_direct_nbc": {
                        "family_count": completed_direct_family_count,
                        "plan_compile": completed_direct_compile,
                        "full_family_generation": completed_direct_generation,
                        "decision_status": (
                            "measured"
                            if completed_direct_decision is not None
                            else "skipped_family_count_limit"
                        ),
                        "decision": completed_direct_decision,
                    },
                    "completed_chordal_nbc": {
                        "family_count": completed_chordal_family_count,
                        "plan_compile": completed_chordal_compile,
                        "full_family_generation": completed_chordal_generation,
                        "decision_status": (
                            "measured"
                            if completed_chordal_decision is not None
                            else "skipped_family_count_limit"
                        ),
                        "decision": completed_chordal_decision,
                    },
                    "exact_completion_prefilter_then_direct_fallback": {
                        "semantics": (
                            "completion_zero_returns_false; completion_live_falls_"
                            "back_to_direct_nbc_on_original"
                        ),
                        "decision_status": (
                            "measured"
                            if exact_pipeline_decision is not None
                            else "skipped_family_count_limit"
                        ),
                        "decision": exact_pipeline_decision,
                    },
                    "original_adaptive_cayley": original_cayley,
                    "completed_adaptive_cayley": completed_cayley,
                }
            )
    return rows


def new_graph_benchmarks(repeats: int) -> dict[str, object]:
    """The November graph portfolio: exact CPU work, never GPU zero tests."""
    rows = []
    for degree in (3, 5, 7):
        composition = (degree - 1, 1)
        baseline = young_dp_tree_moment if degree >= 7 else tree_moment
        for shape, edges in (("path", path_tree(degree)), ("star", star_tree(degree))):
            expected = baseline(composition, edges)
            assert adaptive_tree_moment(composition, edges) == expected
            rows.append(
                {
                    "case": f"terminal_hook_{degree}_{shape}",
                    "kind": "scalar",
                    "degree": degree,
                    "value": fraction_row(expected),
                    "baseline": measure(lambda: baseline(composition, edges), repeats),
                    "candidate": measure(
                        lambda: adaptive_tree_moment(composition, edges), repeats
                    ),
                }
            )
    for name, composition, channels, edges in (
        ("terminal_cell", (6, 1), tuple(range(7)), path_tree(7)),
        ("nonchordal_cycle", (1, 1, 2), (0, 1, 2, 3), ((0, 1), (1, 2), (2, 3), (0, 3))),
        ("chordal_zero", (1, 1, 1, 1), (0, 1, 2, 3), star_tree(4)),
        ("chordal_dense", (4, 2), tuple(range(6)), tuple(combinations(range(6), 2))),
    ):
        graph = DependenceGraph(edges)
        expected = graph_queries.cell_survives(
            composition, channels, graph, backend="direct_nbc"
        )
        assert adaptive_cell_survives(composition, channels, graph) == expected
        rows.append(
            {
                "case": name,
                "kind": "cell",
                "survives": expected,
                "baseline": measure(
                    lambda: graph_queries.cell_survives(
                        composition, channels, graph, backend="direct_nbc"
                    ),
                    repeats,
                ),
                "candidate": measure(
                    lambda: adaptive_cell_survives(composition, channels, graph),
                    repeats,
                ),
            }
        )
    graph = DependenceGraph(((0, 1), (0, 2)))
    for name, composition, channels in (
        ("one_centre", (1, 1, 1), (0, 1, 2)),
        ("two_equal_centres", (2, 2), (0, 0, 1, 2)),
        ("two_separated_centres", (1, 2, 1), (0, 1, 2, 0)),
        ("three_centres", (1, 1, 1, 1), (0, 0, 0, 1)),
    ):
        value = star_trace_gram_energy(composition, channels, graph, centre_channel=0)
        assert bool(value) == adaptive_cell_survives(composition, channels, graph)
        rows.append(
            {
                "case": name,
                "kind": "star_gram",
                "trace_norm": fraction_row(value),
                "baseline": measure(
                    lambda: adaptive_cell_survives(composition, channels, graph),
                    repeats,
                ),
                "candidate": measure(
                    lambda: star_trace_gram_energy(
                        composition, channels, graph, centre_channel=0
                    ),
                    repeats,
                ),
                "comparison": "same_zero_set_only_not_equal_output_values",
            }
        )
    for name, composition, edges in (
        ("triangle", (1, 1, 1), tuple(combinations(range(3), 2))),
        ("joined_triangle", (2, 2), ((0, 1), (0, 2), (1, 2), (1, 3))),
        ("diamond", (1, 1, 1, 1), ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))),
        ("dense_refusal", (1, 1, 1, 1), tuple(combinations(range(4), 2))),
        (
            "eight_vertex_tree",
            (5, 2, 1),
            ((0, 7), (1, 7), (2, 1), (5, 3), (4, 6), (3, 0), (6, 2)),
        ),
    ):
        graph = DependenceGraph(edges)
        channels = tuple(range(sum(composition)))

        def reference():
            trees = (
                (graph.edges,)
                if len(edges) == len(channels) - 1
                else _labelled_trees(len(channels))
            )
            return sum(
                (
                    adaptive_tree_moment(composition, tree) ** 2
                    for tree in trees
                    if tree <= graph.edges
                ),
                Fraction(),
            )

        started = perf_counter_ns()
        try:
            value = literal_tree_energy(composition, channels, graph)
        except ResourceLimit as error:
            rows.append(
                {
                    "case": name,
                    "kind": "literal_energy",
                    "outcome": "resource_limit",
                    "message": str(error),
                    "refusal_ms": (perf_counter_ns() - started) / 1_000_000,
                }
            )
            continue
        assert value == reference()
        rows.append(
            {
                "case": name,
                "kind": "literal_energy",
                "value": fraction_row(value),
                "baseline": measure(reference, repeats),
                "candidate": measure(
                    lambda: literal_tree_energy(composition, channels, graph), repeats
                ),
            }
        )
    return {
        "schema_version": 1,
        "device": "cpu",
        "arithmetic": "exact_rational",
        "status": "bounded_workload_comparison_not_gpu_runtime",
        "environment": {"python": sys.version, "platform": platform.platform()},
        "limits": {"max_entries": 100_000, "max_terms": 1_000_000},
        "timing": "warmed_calibrated_median_separate_tracemalloc",
        "repeats": repeats,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-graph-only",
        action="store_true",
        help="Run the bounded hook, unified-query, ST4 and LE6 CPU portfolio only.",
    )
    parser.add_argument("--max-degree", type=int, default=6, choices=range(4, 7))
    parser.add_argument(
        "--completion-max-degree",
        type=int,
        default=9,
        choices=range(4, 10),
    )
    parser.add_argument(
        "--completion-samples",
        type=int,
        default=1,
        help="samples per (degree, target density)",
    )
    parser.add_argument("--completion-seed", type=int, default=0)
    parser.add_argument(
        "--completion-decision-family-limit",
        type=int,
        default=10_000,
        help="skip scalar decisions when their full NBC family exceeds this count",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.completion_samples < 1:
        parser.error("--completion-samples must be positive")
    if args.completion_decision_family_limit < 1:
        parser.error("--completion-decision-family-limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.new_graph_only:
        print(json.dumps(new_graph_benchmarks(args.repeats), indent=2))
        return
    young_rows = young_benchmarks(args.max_degree, args.repeats)
    zero_rows = zero_prefilter_benchmarks(args.max_degree, args.repeats)
    circuit_rows = circuit_benchmarks(args.max_degree, args.repeats)
    support_rows = support_benchmarks(args.max_degree, args.repeats)
    chordal_rows = chordal_nbc_benchmarks(args.max_degree, args.repeats)
    direct_nbc_rows = direct_nbc_benchmarks(args.max_degree, args.repeats)
    gram_rows = one_centre_gram_benchmarks(args.max_degree, args.repeats)
    completion_rows = nonchordal_completion_benchmarks(
        args.completion_max_degree,
        args.completion_samples,
        args.completion_seed,
        args.completion_decision_family_limit,
        args.repeats,
    )
    summary = {
        "schema_version": 4,
        "status": "timings_only_no_correctness_claim",
        "timing": "warmed_calibrated_in_process_median",
        "memory": "separate_tracemalloc_retained_and_peak",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "candidate_max_degree": args.max_degree,
        "repeats": args.repeats,
        "young_dp": {
            "benchmarked_degrees": list(range(2, args.max_degree + 1)),
            "rows": young_rows,
        },
        "universal_zero_prefilter": {"rows": zero_rows},
        "circuit_coordinates": {
            "status": "bounded_reusable_control_not_direct_nbc",
            "benchmarked_through_degree": args.max_degree,
            "rows": circuit_rows,
        },
        "support_compiler": {
            "status": "cayley_fed_compiled_query_contender",
            "case_degrees": sorted({sum(row["composition"]) for row in support_rows}),
            "rows": support_rows,
        },
        "chordal_nbc": {
            "status": "direct_peo_parent_zero_nonzero_detector",
            "rows": chordal_rows,
        },
        "direct_nbc": {
            "status": "generic_ordered_nbc_generator_and_signed_replay",
            "rows": direct_nbc_rows,
        },
        "one_centre_gram": {
            "status": "exact_specialized_energy_and_zero_nonzero_detector",
            "rows": gram_rows,
        },
        "nonchordal_completion": {
            "status": "timing_comparison_on_distinct_graphs_not_an_equivalence",
            "semantic_warning": (
                "The min-fill completion is a different supergraph. Its survival "
                "result is not equivalent to the result on the original graph. "
                "Only completion-zero is decisive; completion-live requires the "
                "reported direct-NBC fallback on the original graph."
            ),
            "sampling": {
                "seed": args.completion_seed,
                "samples_per_degree_and_target_density": args.completion_samples,
                "target_densities": _COMPLETION_TARGET_DENSITIES,
                "decision_composition_policy": "(1, 1, degree - 2)",
                "degrees": list(range(4, args.completion_max_degree + 1)),
                "construction": (
                    "seeded fixed-density graph with connectivity and an induced "
                    "four-cycle forced before sampling remaining eligible edges; "
                    "small-degree targets are clamped by those forced edges"
                ),
                "completion": (
                    "deterministic greedy min-fill; ties by remaining degree then "
                    "vertex id"
                ),
            },
            "monotonic_semantics": {
                "00": "both original graph and completed supergraph die",
                "01": "original dies but completed supergraph survives",
                "11": "both original graph and completed supergraph survive",
                "10": "impossible because survival is monotone under edge addition",
            },
            "decision_family_limit": args.completion_decision_family_limit,
            "adaptive_cayley_max_degree": 6,
            "rows": completion_rows,
        },
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
