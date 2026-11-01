"""Runtime and boundary checks for compiled Lie quotients."""

import pytest
import torch

from tensorsig.classical.signatures import log_signature
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.coordinates import WordSelection
from tensorsig.offset_coded_signatures.quotients import (
    ChannelMultiplicity,
    ChannelSupport,
    DependenceGraph,
    LieQuotient,
    OffsetSpan,
    TemporalDegree,
)
from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    compile_lie_quotient,
    quotient_log_signature_from_increment_windows,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import (
    OffsetAlphabet,
    ProjectedAlphabet,
)
from tests.quotient_oracles import project_ambient


def _path_from_increments(increments: torch.Tensor) -> torch.Tensor:
    origin = torch.zeros(
        increments.shape[0],
        1,
        increments.shape[-1],
        dtype=increments.dtype,
        device=increments.device,
    )
    return torch.cat((origin, increments.cumsum(dim=1)), dim=1)


def test_temporal_and_channel_quotient_matches_projected_full_hall_log() -> None:
    alphabet = ProjectedAlphabet.newton(code_dim=2, in_channels=2)
    quotient = LieQuotient(TemporalDegree(1), ChannelSupport(1))
    plan = compile_lie_quotient(quotient, alphabet, max_depth=3)
    generator = torch.Generator().manual_seed(410)
    dxw = torch.randn(7, 1, 3, 2, generator=generator, dtype=torch.float64)
    phase = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
    )

    direct = quotient_log_signature_from_increment_windows(dxw, plan, phase)[:, 0]
    lifted = torch.einsum("bnrc,re->bnrce", dxw, phase).flatten(-2)[:, 0]
    ambient = log_signature(
        _path_from_increments(lifted),
        depth=3,
        method="bch_sparse",
        mode="hall",
    )

    assert plan.coordinate_aligned
    assert plan.dimension < plan.ambient_dimension
    torch.testing.assert_close(direct, project_ambient(ambient, plan))


def test_graph_quotient_ignores_nonedge_order_but_retains_edge_order() -> None:
    quotient = LieQuotient(DependenceGraph(((0, 1), (1, 2))))
    plan = compile_lie_quotient(
        quotient,
        ProjectedAlphabet(code_dim=1, in_channels=3),
        max_depth=2,
    )
    phase = torch.ones(2, 1, dtype=torch.float64)
    e0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    e1 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    e2 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    windows = torch.stack(
        (
            torch.stack((e0, e2)),
            torch.stack((e2, e0)),
            torch.stack((e0, e1)),
            torch.stack((e1, e0)),
        )
    ).unsqueeze(1)

    features = quotient_log_signature_from_increment_windows(windows, plan, phase)[:, 0]

    assert not plan.coordinate_aligned
    assert plan.ambient_dimension == 6
    assert plan.dimension == 5
    torch.testing.assert_close(features[0], features[1])
    assert not torch.allclose(features[2], features[3])


def test_graph_quotient_direct_bch_matches_ambient_projection_at_depth_three() -> None:
    plan = compile_lie_quotient(
        LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
        ProjectedAlphabet(code_dim=1, in_channels=3),
        max_depth=3,
    )
    generator = torch.Generator().manual_seed(882)
    dxw = torch.randn(6, 1, 4, 3, generator=generator, dtype=torch.float64)
    phase = torch.ones(4, 1, dtype=torch.float64)

    direct = quotient_log_signature_from_increment_windows(dxw, plan, phase)[:, 0]
    ambient = log_signature(
        _path_from_increments(dxw[:, 0]),
        depth=3,
        method="bch_sparse",
        mode="hall",
    )

    torch.testing.assert_close(direct, project_ambient(ambient, plan))


def test_graph_relation_closure_removes_all_higher_free_lie_directions() -> None:
    plan = compile_lie_quotient(
        LieQuotient(DependenceGraph()),
        ProjectedAlphabet(code_dim=1, in_channels=2),
        max_depth=4,
    )

    assert plan.level_dimensions == (2, 0, 0, 0)
    assert plan.dimension == 2
    assert plan.bracket_indices.shape == (0, 3)


def test_graph_quotient_at_depth_one_is_the_unmodified_generator_space() -> None:
    plan = compile_lie_quotient(
        LieQuotient(DependenceGraph()),
        ProjectedAlphabet(code_dim=2, in_channels=3),
        max_depth=1,
    )

    assert plan.level_dimensions == (6,)
    assert plan.dimension == plan.ambient_dimension == 6


def test_quotient_plan_to_moves_runtime_tensors_without_casting_indices() -> None:
    compiled = compile_lie_quotient(
        LieQuotient(DependenceGraph()),
        ProjectedAlphabet(1, 2),
        max_depth=2,
    )
    assert compiled.to() is compiled
    plan = compiled.to(dtype=torch.float32)

    assert plan.generator_map.dtype == torch.float32
    assert plan.bracket_values.dtype == torch.float32
    assert plan.bracket_indices.dtype == torch.long
    assert plan.basis_depths.dtype == torch.long


def test_device_bound_quotient_plan_reuses_its_validated_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorsig.offset_coded_signatures._lie_quotients.compiler as quotient_module

    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(0)),
        ProjectedAlphabet.newton(2, 2),
        max_depth=3,
    ).to(dtype=torch.float32)

    def fail_rebuild(*args: object, **kwargs: object) -> None:
        raise AssertionError("a device-bound plan must reuse its SparseBCH runtime")

    monkeypatch.setattr(quotient_module, "SparseBCH", fail_rebuild)
    output = quotient_log_signature_from_increment_windows(
        torch.randn(2, 1, 3, 2),
        plan,
        OffsetCode.pascal(3, 2).matrix,
    )

    assert output.shape == (2, 1, plan.dimension)


def test_full_offset_span_is_a_primitive_quotient() -> None:
    plan = compile_lie_quotient(
        LieQuotient(OffsetSpan(1)),
        OffsetAlphabet(kernel_size=3, in_channels=1),
        max_depth=2,
    )

    # Three generators and two retained adjacent-offset commutators. The
    # long-span product is not deleted; only its primitive commutator is.
    assert plan.ambient_dimension == 6
    assert plan.level_dimensions == (3, 2)
    assert plan.dimension == 5


def test_per_channel_multiplicity_cap_is_an_additive_ideal() -> None:
    plan = compile_lie_quotient(
        LieQuotient(ChannelMultiplicity(1)),
        ProjectedAlphabet.newton(code_dim=2, in_channels=1),
        max_depth=3,
    )

    assert plan.level_dimensions == (2, 0, 0)


def test_zero_channel_cap_discards_a_nuisance_generator_exactly() -> None:
    plan = compile_lie_quotient(
        LieQuotient(ChannelMultiplicity((1, 0))),
        ProjectedAlphabet(code_dim=1, in_channels=2),
        max_depth=2,
    )
    dxw = torch.tensor(
        [[[[1.0, 20.0], [2.0, -50.0]]]],
        dtype=torch.float64,
    )
    changed = dxw.clone()
    changed[..., 1] += 1000.0
    phase = torch.ones(2, 1, dtype=torch.float64)

    first = quotient_log_signature_from_increment_windows(dxw, plan, phase)
    second = quotient_log_signature_from_increment_windows(changed, plan, phase)

    assert plan.level_dimensions == (1, 0)
    torch.testing.assert_close(first, second)


def test_zero_dimensional_quotient_has_a_valid_empty_logsignature() -> None:
    plan = compile_lie_quotient(
        LieQuotient(ChannelSupport(0)),
        ProjectedAlphabet(code_dim=1, in_channels=2),
        max_depth=2,
    )
    output = quotient_log_signature_from_increment_windows(
        torch.randn(3, 2, 4, 2),
        plan,
        torch.ones(4, 1),
    )

    assert plan.dimension == 0
    assert output.shape == (3, 2, 0)


def test_quotient_rule_join_operator_imposes_both_constraints() -> None:
    temporal = LieQuotient(TemporalDegree(1))
    channels = LieQuotient(ChannelSupport(1))
    combined = temporal & channels

    direct = compile_lie_quotient(
        combined,
        ProjectedAlphabet.newton(code_dim=2, in_channels=2),
        max_depth=3,
    )
    explicit = compile_lie_quotient(
        LieQuotient(TemporalDegree(1), ChannelSupport(1)),
        ProjectedAlphabet.newton(code_dim=2, in_channels=2),
        max_depth=3,
    )

    assert direct.basis == explicit.basis
    torch.testing.assert_close(direct.generator_map, explicit.generator_map)
    torch.testing.assert_close(direct.bracket_indices, explicit.bracket_indices)
    torch.testing.assert_close(direct.bracket_values, explicit.bracket_values)


def test_join_of_graph_quotients_intersects_allowed_edges() -> None:
    first = LieQuotient(DependenceGraph(((0, 1), (0, 2))))
    second = LieQuotient(DependenceGraph(((0, 1), (1, 2))))
    plan = compile_lie_quotient(
        first & second,
        ProjectedAlphabet(code_dim=1, in_channels=3),
        max_depth=2,
    )

    assert plan.level_dimensions == (3, 1)


def test_quotient_logsignature_is_differentiable() -> None:
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)),
        ProjectedAlphabet.newton(code_dim=2, in_channels=2),
        max_depth=3,
    )
    dxw = torch.randn(2, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    phase = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)

    output = quotient_log_signature_from_increment_windows(dxw, plan, phase)
    output.square().sum().backward()

    assert dxw.grad is not None and torch.isfinite(dxw.grad).all()
    assert phase.grad is not None and torch.isfinite(phase.grad).all()


def test_quotient_runtime_requires_floating_inputs() -> None:
    plan = compile_lie_quotient(
        LieQuotient(),
        ProjectedAlphabet(code_dim=1, in_channels=1),
        max_depth=1,
    )

    with pytest.raises(ValueError, match="dxw must have a floating-point dtype"):
        quotient_log_signature_from_increment_windows(
            torch.ones(1, 1, 2, 1, dtype=torch.int64),
            plan,
            torch.ones(2, 1),
        )


@pytest.mark.parametrize("max_depth", [1.0, True])
def test_quotient_compiler_requires_integer_depth(max_depth: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        compile_lie_quotient(
            LieQuotient(),
            ProjectedAlphabet(1, 1),
            max_depth=max_depth,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TemporalDegree(1.5),  # type: ignore[arg-type]
        lambda: OffsetSpan(True),
        lambda: ProjectedAlphabet(1.5, 2),  # type: ignore[arg-type]
        lambda: DependenceGraph(((0.9, 2.1),)),
    ],
)
def test_pruning_specifications_reject_noninteger_discrete_data(
    factory: object,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        factory()  # type: ignore[operator]


def test_pruning_specifications_normalize_sequence_metadata() -> None:
    multiplicity = ChannelMultiplicity([1, 2])  # type: ignore[arg-type]
    alphabet = ProjectedAlphabet(
        code_dim=2,
        in_channels=2,
        temporal_degrees=[0, 1],  # type: ignore[arg-type]
    )
    selection = WordSelection(
        descent_sets=[[], [1]],  # type: ignore[arg-type,list-item]
    )

    assert multiplicity.maximum == (1, 2)
    assert alphabet.temporal_degrees == (0, 1)
    assert selection.descent_sets == ((), (1,))
    assert hash(multiplicity)
    assert hash(alphabet)
    assert hash(selection)


@pytest.mark.parametrize(
    ("quotient", "alphabet", "message"),
    [
        (LieQuotient(TemporalDegree(1)), OffsetAlphabet(3, 1), "ProjectedAlphabet"),
        (
            LieQuotient(TemporalDegree(1)),
            ProjectedAlphabet(2, 1),
            "explicitly graded",
        ),
        (LieQuotient(OffsetSpan(1)), ProjectedAlphabet(2, 1), "OffsetAlphabet"),
        (
            LieQuotient(DependenceGraph(((0, 2),))),
            ProjectedAlphabet(1, 2),
            "out-of-range",
        ),
    ],
)
def test_quotient_compiler_rejects_rules_in_the_wrong_domain(
    quotient: LieQuotient,
    alphabet: ProjectedAlphabet | OffsetAlphabet,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_lie_quotient(quotient, alphabet, max_depth=2)
