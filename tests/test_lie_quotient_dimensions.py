"""Independent dimension checks for implemented Lie-quotient families."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from math import factorial, gcd

import torch

from tensorsig.classical.signatures import log_signature
from tensorsig.algebra.hall_bch import SparseBCH
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.plans import LocalRealization
from tensorsig.offset_coded_signatures.quotients import (
    ChannelMultiplicity,
    DependenceGraph,
    LieQuotient,
    OffsetSpan,
    TemporalDegree,
)
from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    LieQuotientPlan,
    compile_lie_quotient,
    quotient_log_signature_from_increment_windows,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import (
    OffsetAlphabet,
    ProjectedAlphabet,
)
from tests.quotient_oracles import project_ambient


def _mobius(value: int) -> int:
    prime_count = 0
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor:
            divisor += 1
            continue
        value //= divisor
        if value % divisor == 0:
            return 0
        prime_count += 1
        while value % divisor == 0:
            value //= divisor
        divisor += 1
    if value > 1:
        prime_count += 1
    return -1 if prime_count % 2 else 1


def _multihomogeneous_witt(multidegree: tuple[int, ...]) -> int:
    level = sum(multidegree)
    common_divisor = 0
    for count in multidegree:
        common_divisor = gcd(common_divisor, count)
    numerator = 0
    for divisor in range(1, common_divisor + 1):
        if common_divisor % divisor:
            continue
        reduced_level = level // divisor
        multinomial = factorial(reduced_level)
        for count in multidegree:
            multinomial //= factorial(count // divisor)
        numerator += _mobius(divisor) * multinomial
    return numerator // level


def _graph_witt_dimensions(
    labels_per_channel: int,
    channel_count: int,
    edges: frozenset[tuple[int, int]],
    max_depth: int,
) -> tuple[int, ...]:
    """Evaluate Theorem 5.1 from the graph independence polynomial."""
    independent_counts = [0] * (channel_count + 1)
    for mask in range(1 << channel_count):
        selected = {channel for channel in range(channel_count) if mask >> channel & 1}
        if all(left not in selected or right not in selected for left, right in edges):
            independent_counts[len(selected)] += 1

    # mu(t) = I_G(-s t), followed by b(t) = -log(mu(t)).
    mu = [Fraction(0)] * (max_depth + 1)
    for degree, count in enumerate(independent_counts[: max_depth + 1]):
        mu[degree] = Fraction(count * (-labels_per_channel) ** degree)
    b = [Fraction(0)] * (max_depth + 1)
    for degree in range(1, max_depth + 1):
        convolution = sum(
            (
                mu[left_degree] * (degree - left_degree) * b[degree - left_degree]
                for left_degree in range(1, degree)
            ),
            Fraction(),
        )
        b[degree] = -mu[degree] - convolution / degree

    dimensions = []
    for degree in range(1, max_depth + 1):
        dimension = sum(
            (
                Fraction(_mobius(divisor), divisor) * b[degree // divisor]
                for divisor in range(1, degree + 1)
                if degree % divisor == 0
            ),
            Fraction(),
        )
        assert dimension.denominator == 1
        dimensions.append(dimension.numerator)
    return tuple(dimensions)


def test_additive_quotient_dimensions_match_multihomogeneous_witt_formula() -> None:
    """Part 1, Section 5: exact primitive counts survive conjunctions."""
    alphabet = ProjectedAlphabet.newton(code_dim=2, in_channels=2)
    quotient = LieQuotient(
        TemporalDegree(2),
        ChannelMultiplicity((2, 1)),
    )
    max_depth = 4
    expected = []
    for level in range(1, max_depth + 1):
        dimension = 0
        for multidegree in product(range(level + 1), repeat=alphabet.width):
            if sum(multidegree) != level:
                continue
            temporal_degree = multidegree[1] + multidegree[3]
            channel_counts = (
                multidegree[0] + multidegree[1],
                multidegree[2] + multidegree[3],
            )
            if (
                temporal_degree <= 2
                and channel_counts[0] <= 2
                and channel_counts[1] <= 1
            ):
                dimension += _multihomogeneous_witt(multidegree)
        expected.append(dimension)

    plan = compile_lie_quotient(quotient, alphabet, max_depth=max_depth)

    assert plan.level_dimensions == tuple(expected)


def test_graph_quotient_dimensions_match_graph_witt_formula() -> None:
    """Part 2, Theorem 5.1, independently of either quotient compiler."""
    channel_count = 3
    pairs = tuple(combinations(range(channel_count), 2))
    for mask in range(1 << len(pairs)):
        edges = frozenset(
            edge for index, edge in enumerate(pairs) if mask & (1 << index)
        )
        plan = compile_lie_quotient(
            LieQuotient(DependenceGraph(edges)),
            ProjectedAlphabet(code_dim=1, in_channels=channel_count),
            max_depth=4,
        )
        assert plan.level_dimensions == _graph_witt_dimensions(
            1,
            channel_count,
            edges,
            4,
        )

    # Also exercise more than one labelled letter per channel.
    edges = frozenset({(0, 1), (1, 2)})
    plan = compile_lie_quotient(
        LieQuotient(DependenceGraph(edges)),
        ProjectedAlphabet(code_dim=2, in_channels=channel_count),
        max_depth=3,
    )
    assert plan.level_dimensions == _graph_witt_dimensions(
        2,
        channel_count,
        edges,
        3,
    )


def test_graph_ideal_meet_is_not_represented_by_the_union_graph() -> None:
    """Part 2, Proposition 3.2: graph ideals are not meet-closed."""
    alphabet = ProjectedAlphabet(code_dim=1, in_channels=3)
    first = compile_lie_quotient(
        LieQuotient(DependenceGraph(((0, 1),))),
        alphabet,
        max_depth=3,
    )
    second = compile_lie_quotient(
        LieQuotient(DependenceGraph(((1, 2),))),
        alphabet,
        max_depth=3,
    )
    union = compile_lie_quotient(
        LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
        alphabet,
        max_depth=3,
    )

    def nested_bracket(plan: LieQuotientPlan) -> torch.Tensor:
        a, b, c = plan.generator_map
        return plan._runtime.bracket(plan._runtime.bracket(a, b), c)

    torch.testing.assert_close(
        nested_bracket(first),
        torch.zeros(first.dimension, dtype=torch.float64),
    )
    torch.testing.assert_close(
        nested_bracket(second),
        torch.zeros(second.dimension, dtype=torch.float64),
    )
    assert nested_bracket(union).norm() > 0.5


def test_graph_span_dimensions_match_exact_ambient_span_formula() -> None:
    """Part 2, Section 5.3: finite differences isolate exact support span."""
    kernel_size = 3
    channel_count = 2
    max_depth = 4
    edges: frozenset[tuple[int, int]] = frozenset()
    interval_dimensions = {
        size: _graph_witt_dimensions(size, channel_count, edges, max_depth)
        for size in range(kernel_size + 1)
    }

    for maximum_span in range(kernel_size):
        expected = [0] * max_depth
        for interval_size in range(1, maximum_span + 2):
            exact_span = tuple(
                interval_dimensions[interval_size][degree]
                - 2 * interval_dimensions[interval_size - 1][degree]
                + (
                    interval_dimensions[interval_size - 2][degree]
                    if interval_size >= 2
                    else 0
                )
                for degree in range(max_depth)
            )
            placements = kernel_size - interval_size + 1
            expected = [
                total + placements * count
                for total, count in zip(expected, exact_span, strict=True)
            ]

        plan = compile_lie_quotient(
            LieQuotient(DependenceGraph(edges), OffsetSpan(maximum_span)),
            OffsetAlphabet(kernel_size, channel_count),
            max_depth=max_depth,
        )
        assert plan.level_dimensions == tuple(expected)


def test_locally_realized_degree_two_graph_span_formula() -> None:
    """Part 2, Section 6.5: local paths realize only cross-offset brackets."""
    kernel_size = 4
    channel_count = 3
    maximum_span = 2
    edges = frozenset({(0, 1), (1, 2)})
    plan = compile_lie_quotient(
        LieQuotient(DependenceGraph(edges), OffsetSpan(maximum_span)),
        OffsetAlphabet(kernel_size, channel_count),
        max_depth=2,
    )
    runtime = SparseBCH(
        dim=plan.dimension,
        depth=2,
        sparse_indices=plan.bracket_indices,
        sparse_values=plan.bracket_values,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    realized_brackets = []
    for left_offset, right_offset in combinations(range(kernel_size), 2):
        for left_channel in range(channel_count):
            for right_channel in range(channel_count):
                realized_brackets.append(
                    runtime.bracket(
                        plan.generator_map[left_offset * channel_count + left_channel],
                        plan.generator_map[
                            right_offset * channel_count + right_channel
                        ],
                    )
                )
    realized_rank = int(torch.linalg.matrix_rank(torch.stack(realized_brackets)))
    support_pairs = sum(kernel_size - span for span in range(1, maximum_span + 1))
    expected = support_pairs * (channel_count + 2 * len(edges))

    assert realized_rank == expected
    assert plan.level_dimensions[1] > realized_rank

    # A single axis segment is one exponential, so kappa_2(2) = 0.
    windows = torch.zeros(1, 1, kernel_size, channel_count, dtype=torch.float64)
    windows[0, 0, 1] = torch.tensor([1.0, -2.0, 3.0])
    output = quotient_log_signature_from_increment_windows(windows, plan)
    torch.testing.assert_close(
        output[..., plan.basis_depths == 2],
        torch.zeros_like(output[..., plan.basis_depths == 2]),
    )


def test_degree_three_graph_span_rank_matches_independent_spanning_paths() -> None:
    """Theorem 7.1 beyond degree two, without the local chart compiler."""
    window = 3
    channels = 2
    quotient = LieQuotient(DependenceGraph(), OffsetSpan(2))
    abstract = compile_lie_quotient(
        quotient,
        OffsetAlphabet(window, channels),
        max_depth=3,
    )
    local = LocalRealization(
        OffsetCode.identity(window, dtype=torch.float64),
        channels=channels,
        max_depth=3,
    ).log_signature(quotient=quotient, depths=(3,))

    generator = torch.Generator().manual_seed(20260804)
    count = 64
    windows = torch.randn(
        count,
        window,
        channels,
        generator=generator,
        dtype=torch.float64,
    )
    lifted = windows.new_zeros(count, window, window * channels)
    for offset in range(window):
        start = offset * channels
        lifted[:, offset, start : start + channels] = windows[:, offset]
    paths = torch.cat(
        (lifted.new_zeros(count, 1, window * channels), lifted.cumsum(dim=1)),
        dim=1,
    )
    ambient = log_signature(paths, depth=3, mode="hall")
    degree_three = project_ambient(ambient, abstract)[:, abstract.basis_depths == 3]
    singular_values = torch.linalg.svdvals(degree_three)
    independent_rank = int((singular_values > 1e-10).sum())

    assert int((abstract.basis_depths == 3).sum()) == 16
    assert independent_rank == local.feature_count == 14
    assert singular_values[13] > 0.1
    assert singular_values[14] < 1e-10
    assert local.quotient_basis is not None
    assert len(local.quotient_basis) == 16
    hash(local.quotient_basis)
