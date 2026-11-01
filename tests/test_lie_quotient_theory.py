"""Theorem-level checks for the implemented Lie-quotient families.

The theory notes also develop Schur blocks, derived/lower-central rules,
arbitrary ideal meets, weighted budgets, projection push/pull, graph--span
Hilbert formulas, and general word automata. Those constructs have no public
implementation yet and are deliberately outside this executable test matrix.
"""

from __future__ import annotations

from math import comb

import pytest
import torch

from tensorsig.classical.signatures import log_signature
from tensorsig.algebra.hall_bch import HallBCH, SparseBCH
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.coordinates import WordSelection
from tensorsig.offset_coded_signatures.plans import LocalRealization
from tensorsig.offset_coded_signatures.quotients import (
    ChannelMultiplicity,
    ChannelSupport,
    DependenceGraph,
    LieQuotient,
    OffsetMultiplicity,
    OffsetSpan,
    OffsetSupport,
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
from tests.quotient_oracles import (
    ambient_quotient_matrix,
    ambient_section_matrix,
    project_ambient,
)


def _path_from_increments(increments: torch.Tensor) -> torch.Tensor:
    origin = torch.zeros(
        increments.shape[0],
        1,
        increments.shape[-1],
        dtype=increments.dtype,
        device=increments.device,
    )
    return torch.cat((origin, increments.cumsum(dim=1)), dim=1)


def _quotient_runtime(plan: LieQuotientPlan) -> SparseBCH:
    return SparseBCH(
        dim=plan.dimension,
        depth=plan.max_depth,
        sparse_indices=plan.bracket_indices,
        sparse_values=plan.bracket_values,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


_UPPER_SET_CASES = (
    pytest.param(
        LieQuotient(TemporalDegree(1)),
        ProjectedAlphabet.newton(2, 1),
        (2, 1, 1),
        id="newton-budget",
    ),
    pytest.param(
        LieQuotient(ChannelMultiplicity(1)),
        ProjectedAlphabet(1, 2),
        (2, 1, 0),
        id="square-free-channels",
    ),
    pytest.param(
        LieQuotient(ChannelSupport(1)),
        ProjectedAlphabet(2, 2),
        (4, 2, 4),
        id="one-channel-support",
    ),
    pytest.param(
        LieQuotient(OffsetMultiplicity(1)),
        OffsetAlphabet(2, 1),
        (2, 1, 0),
        id="square-free-offsets",
    ),
    pytest.param(
        LieQuotient(OffsetSupport(1)),
        OffsetAlphabet(2, 2),
        (4, 2, 4),
        id="one-offset-support",
    ),
    pytest.param(
        LieQuotient(OffsetSpan(0)),
        OffsetAlphabet(2, 2),
        (4, 2, 4),
        id="zero-offset-span",
    ),
)


@pytest.mark.parametrize(
    ("quotient", "alphabet", "_expected"),
    [
        *_UPPER_SET_CASES,
        pytest.param(
            LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
            ProjectedAlphabet(1, 3),
            None,
            id="graph-relations",
        ),
    ],
)
def test_compiled_kernel_is_an_ideal_and_projection_is_a_lie_homomorphism(
    quotient: LieQuotient,
    alphabet: ProjectedAlphabet | OffsetAlphabet,
    _expected: tuple[int, int, int] | None,
) -> None:
    """Theorems 3.1, 4.1, 6.1, and 8.1 hold in the compiled algebra."""
    plan = compile_lie_quotient(quotient, alphabet, max_depth=3)
    ambient = HallBCH(alphabet.width, 3, torch.device("cpu"), torch.float64)
    retained = _quotient_runtime(plan)
    generator = torch.Generator().manual_seed(1789)
    x = torch.randn(4, plan.ambient_dimension, generator=generator, dtype=torch.float64)
    y = torch.randn(4, plan.ambient_dimension, generator=generator, dtype=torch.float64)
    qx = project_ambient(x, plan)
    qy = project_ambient(y, plan)

    projected_bracket = project_ambient(ambient.bracket(x, y), plan)
    torch.testing.assert_close(
        projected_bracket,
        retained.bracket(qx, qy),
        atol=2e-9,
        rtol=2e-9,
    )

    kernel_element = x - qx @ ambient_section_matrix(plan).T
    torch.testing.assert_close(
        project_ambient(kernel_element, plan),
        torch.zeros_like(qx),
        atol=2e-9,
        rtol=0,
    )
    torch.testing.assert_close(
        project_ambient(ambient.bracket(kernel_element, y), plan),
        torch.zeros_like(qx),
        atol=2e-9,
        rtol=0,
    )

    torch.testing.assert_close(
        project_ambient(ambient.bch(x, y), plan),
        retained.bch(qx, qy),
        atol=2e-9,
        rtol=2e-9,
    )


@pytest.mark.parametrize(
    ("quotient", "alphabet", "expected"),
    _UPPER_SET_CASES,
)
def test_upper_set_quotient_dimensions_match_low_degree_witt_counts(
    quotient: LieQuotient,
    alphabet: ProjectedAlphabet | OffsetAlphabet,
    expected: tuple[int, int, int],
) -> None:
    """Independent low-degree counts match additive/support Witt theory."""
    plan = compile_lie_quotient(quotient, alphabet, max_depth=3)

    assert plan.level_dimensions == expected


def test_join_of_upper_set_ideals_keeps_the_intersection_of_standard_sets() -> None:
    alphabet = ProjectedAlphabet.newton(2, 2)
    temporal = LieQuotient(TemporalDegree(1))
    support = LieQuotient(ChannelSupport(1))
    temporal_plan = compile_lie_quotient(temporal, alphabet, max_depth=3)
    support_plan = compile_lie_quotient(support, alphabet, max_depth=3)
    joined_plan = compile_lie_quotient(temporal & support, alphabet, max_depth=3)

    def retained_mask(plan: LieQuotientPlan) -> torch.Tensor:
        projector = ambient_section_matrix(plan) @ ambient_quotient_matrix(plan)
        return torch.diagonal(projector) > 0.5

    assert torch.equal(
        retained_mask(joined_plan),
        retained_mask(temporal_plan) & retained_mask(support_plan),
    )


@pytest.mark.parametrize(
    "edges",
    [
        pytest.param((), id="empty"),
        pytest.param(((0, 1),), id="one-edge"),
        pytest.param(((0, 1), (1, 2)), id="path"),
        pytest.param(((0, 1), (0, 2), (1, 2)), id="complete"),
    ],
)
def test_graph_degree_two_dimension_matches_the_independence_count(
    edges: tuple[tuple[int, int], ...],
) -> None:
    code_dim = 2
    channel_count = 3
    plan = compile_lie_quotient(
        LieQuotient(DependenceGraph(edges)),
        ProjectedAlphabet(code_dim, channel_count),
        max_depth=2,
    )
    nonedges = comb(channel_count, 2) - len(edges)
    expected = comb(code_dim * channel_count, 2) - nonedges * code_dim**2

    assert plan.level_dimensions[1] == expected


def test_graph_connected_support_can_survive_without_being_a_clique() -> None:
    path_plan = compile_lie_quotient(
        LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
        ProjectedAlphabet(1, 3),
        max_depth=3,
    )
    disconnected_plan = compile_lie_quotient(
        LieQuotient(DependenceGraph(((0, 1),))),
        ProjectedAlphabet(1, 3),
        max_depth=3,
    )

    def nested_bracket(plan: LieQuotientPlan) -> torch.Tensor:
        runtime = _quotient_runtime(plan)
        a, b, c = plan.generator_map
        return runtime.bracket(runtime.bracket(a, b), c)

    # {0, 1, 2} is connected in the path graph although {0, 2} is a nonedge.
    assert nested_bracket(path_plan).norm() > 0.5
    # Isolating channel 2 makes the same support disconnected, so it vanishes.
    torch.testing.assert_close(
        nested_bracket(disconnected_plan),
        torch.zeros(disconnected_plan.dimension, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )


def test_graph_join_intersects_allowed_edges() -> None:
    first = LieQuotient(DependenceGraph(((0, 1), (0, 2))))
    second = LieQuotient(DependenceGraph(((0, 1), (1, 2))))
    plan = compile_lie_quotient(
        first & second,
        ProjectedAlphabet(1, 3),
        max_depth=2,
    )
    runtime = _quotient_runtime(plan)
    a, b, c = plan.generator_map

    assert runtime.bracket(a, b).norm() > 0.5
    torch.testing.assert_close(runtime.bracket(a, c), torch.zeros_like(a))
    torch.testing.assert_close(runtime.bracket(b, c), torch.zeros_like(a))


def test_full_offset_pruning_precedes_projection_and_matches_ambient_log() -> None:
    alphabet = OffsetAlphabet(kernel_size=3, in_channels=1)
    plan = compile_lie_quotient(
        LieQuotient(OffsetSupport(2), OffsetSpan(1)),
        alphabet,
        max_depth=3,
    )
    generator = torch.Generator().manual_seed(1204)
    dxw = torch.randn(3, 2, 3, 1, generator=generator, dtype=torch.float64)

    direct = quotient_log_signature_from_increment_windows(dxw, plan)
    flat = dxw.reshape(6, 3, 1)
    ambient_increments = torch.zeros(6, 3, 3, dtype=torch.float64)
    for offset in range(3):
        ambient_increments[:, offset, offset] = flat[:, offset, 0]
    ambient = log_signature(
        _path_from_increments(ambient_increments),
        depth=3,
        method="bch_sparse",
        mode="hall",
    ).reshape(3, 2, -1)

    torch.testing.assert_close(direct, project_ambient(ambient, plan))
    with pytest.raises(ValueError, match="OffsetAlphabet"):
        compile_lie_quotient(
            LieQuotient(OffsetSpan(1)),
            ProjectedAlphabet(2, 1),
            max_depth=3,
        )


@pytest.mark.parametrize(
    "rule",
    [OffsetMultiplicity(1), OffsetSupport(1), OffsetSpan(1)],
)
def test_physical_offset_rules_are_rejected_after_dense_projection(
    rule: OffsetMultiplicity | OffsetSupport | OffsetSpan,
) -> None:
    with pytest.raises(ValueError, match="OffsetAlphabet"):
        compile_lie_quotient(
            LieQuotient(rule),
            ProjectedAlphabet(2, 1),
            max_depth=2,
        )


def test_chen_concatenation_is_bch_in_the_quotient() -> None:
    plan = compile_lie_quotient(
        LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
        ProjectedAlphabet(1, 3),
        max_depth=3,
    )
    generator = torch.Generator().manual_seed(773)
    left = torch.randn(2, 1, 2, 3, generator=generator, dtype=torch.float64)
    right = torch.randn(2, 1, 3, 3, generator=generator, dtype=torch.float64)
    left_phase = torch.ones(2, 1, dtype=torch.float64)
    right_phase = torch.ones(3, 1, dtype=torch.float64)
    left_log = quotient_log_signature_from_increment_windows(left, plan, left_phase)[
        :, 0
    ]
    right_log = quotient_log_signature_from_increment_windows(right, plan, right_phase)[
        :, 0
    ]
    concatenated_log = quotient_log_signature_from_increment_windows(
        torch.cat((left, right), dim=2),
        plan,
        torch.cat((left_phase, right_phase), dim=0),
    )[:, 0]

    torch.testing.assert_close(
        concatenated_log,
        _quotient_runtime(plan).bch(left_log, right_log),
    )


@pytest.mark.parametrize(
    ("quotient", "alphabet"),
    [
        pytest.param(
            LieQuotient(TemporalDegree(0)),
            ProjectedAlphabet(2, 1, temporal_degrees=(1, 2)),
            id="positive-temporal-degrees",
        ),
        pytest.param(
            LieQuotient(ChannelMultiplicity(0)),
            ProjectedAlphabet(1, 2),
            id="channel-multiplicity-zero",
        ),
        pytest.param(
            LieQuotient(ChannelSupport(0)),
            ProjectedAlphabet(1, 2),
            id="channel-support-zero",
        ),
        pytest.param(
            LieQuotient(OffsetMultiplicity(0)),
            OffsetAlphabet(2, 1),
            id="offset-multiplicity-zero",
        ),
        pytest.param(
            LieQuotient(OffsetSupport(0)),
            OffsetAlphabet(2, 1),
            id="offset-support-zero",
        ),
    ],
)
def test_zero_quotients_are_valid_differentiable_terminal_objects(
    quotient: LieQuotient,
    alphabet: ProjectedAlphabet | OffsetAlphabet,
) -> None:
    plan = compile_lie_quotient(quotient, alphabet, max_depth=2)
    kernel_size = alphabet.kernel_size if isinstance(alphabet, OffsetAlphabet) else 2
    dxw = torch.randn(
        2,
        1,
        kernel_size,
        alphabet.in_channels,
        dtype=torch.float64,
        requires_grad=True,
    )
    phase = (
        None
        if isinstance(alphabet, OffsetAlphabet)
        else torch.ones(
            kernel_size,
            alphabet.code_dim,
            dtype=torch.float64,
            requires_grad=True,
        )
    )
    output = quotient_log_signature_from_increment_windows(dxw, plan, phase)

    assert plan.dimension == 0
    assert output.shape == (2, 1, 0)
    output.sum().backward()
    assert dxw.grad is not None
    if phase is not None:
        assert phase.grad is not None


def test_word_selection_is_not_the_graph_quotient_invariant() -> None:
    phase = torch.ones(2, 1, dtype=torch.float64)
    e0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    e1 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    windows = torch.stack((torch.stack((e0, e1)), torch.stack((e1, e0)))).unsqueeze(1)
    word_plan = LocalRealization(
        OffsetCode.constant(2, dtype=torch.float64),
        channels=2,
        max_depth=2,
    ).signature(
        coordinates="chronological",
        depths=(2,),
        selection=WordSelection(descent_sets=((),)),
    )
    word_features = word_plan.evaluate(windows)[:, 0]

    quotient_plan = compile_lie_quotient(
        LieQuotient(DependenceGraph()),
        ProjectedAlphabet(1, 2),
        max_depth=2,
    )
    quotient_features = quotient_log_signature_from_increment_windows(
        windows,
        quotient_plan,
        phase,
    )[:, 0]

    # The no-descent word (0, 1) records order, while the graph quotient makes
    # the two nonadjacent named channels commute.
    assert not torch.allclose(word_features[0], word_features[1])
    torch.testing.assert_close(quotient_features[0], quotient_features[1])


@pytest.mark.parametrize("family", ["upper-set", "graph"])
@pytest.mark.parametrize("method", ["direct", "tensor"])
def test_quotient_methods_have_correct_input_and_phase_gradients(
    family: str,
    method: str,
) -> None:
    if family == "upper-set":
        alphabet = ProjectedAlphabet.newton(2, 1)
        quotient = LieQuotient(TemporalDegree(1))
        channels = 1
        code_dim = 2
    else:
        alphabet = ProjectedAlphabet(1, 3)
        quotient = LieQuotient(DependenceGraph(((0, 1), (1, 2))))
        channels = 3
        code_dim = 1
    plan = compile_lie_quotient(quotient, alphabet, max_depth=3)
    dxw = (torch.randn(1, 1, 2, channels, dtype=torch.float64) * 0.2).requires_grad_()
    phase = (torch.randn(2, code_dim, dtype=torch.float64) * 0.2).requires_grad_()

    assert torch.autograd.gradcheck(
        lambda windows, code: quotient_log_signature_from_increment_windows(
            windows,
            plan,
            code,
            method=method,
        ),
        (dxw, phase),
        fast_mode=True,
    )
