import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from tensorsig.offset_coded_signatures.layers import GapFactorizedVolterra
from tensorsig.offset_coded_signatures.plans import compile_polynomial
from tensorsig.offset_coded_signatures.windows import increment_windows


def _expanded_polynomial_weight(layer: GapFactorizedVolterra) -> torch.Tensor:
    """Expand the structured parameters into the plan's free readout."""
    blocks = []
    for level, depth in enumerate(layer.depths):
        offsets = layer.plan.offset_indices[level]
        channels = layer.plan.channel_indices[level]
        internal = layer.internal_gap_factors[level]
        terminal = layer.terminal_lag_factors[level]
        channel_weight = layer.channel_weights[level]
        temporal = torch.ones(
            layer.temporal_rank,
            offsets.shape[0],
            dtype=channel_weight.dtype,
            device=channel_weight.device,
        )
        runtime_offsets = offsets.to(device=channel_weight.device)
        for position in range(depth - 1):
            gaps = runtime_offsets[:, position + 1] - runtime_offsets[:, position]
            temporal = temporal * internal[:, position, gaps - 1]
        terminal_lags = layer.window - 1 - runtime_offsets[:, -1]
        temporal = temporal * terminal[:, terminal_lags]

        columns = []
        for feature, channel_tuple in enumerate(channels.tolist()):
            selected = channel_weight[
                (slice(None), slice(None), *tuple(int(c) for c in channel_tuple))
            ]
            columns.append((selected * temporal[:, feature]).sum(dim=1))
        blocks.append(torch.stack(columns, dim=1))
    return torch.cat(blocks, dim=1)


def _matching_layers(
    plan,
    *,
    temporal_rank: int = 3,
) -> tuple[GapFactorizedVolterra, GapFactorizedVolterra]:
    dynamic = GapFactorizedVolterra(
        plan,
        out_features=2,
        temporal_rank=temporal_rank,
        backend="dynamic",
        gap_initialization="uniform",
        dtype=torch.float64,
    )
    enumerated = GapFactorizedVolterra(
        plan,
        out_features=2,
        temporal_rank=temporal_rank,
        backend="enumerate",
        gap_initialization="uniform",
        dtype=torch.float64,
    )
    enumerated.load_state_dict(dynamic.state_dict())
    return dynamic, enumerated


def test_strict_plan_is_the_unit_scaled_nonrepeating_volterra_basis() -> None:
    plan = compile_polynomial(
        window=4,
        channels=2,
        max_depth=3,
        depths=(3,),
        max_offset_multiplicity=1,
        scaling="support",
    )
    layer = GapFactorizedVolterra(plan, out_features=2, temporal_rank=1)

    assert layer.plan is plan
    assert plan.feature_count == math.comb(4, 3) * 2**3
    assert torch.all(plan.offset_indices[0][:, :-1] < plan.offset_indices[0][:, 1:])
    torch.testing.assert_close(
        plan.coefficients[0],
        torch.ones(plan.feature_count, dtype=torch.float64),
    )


def test_dynamic_backend_matches_enumeration_in_values_and_gradients() -> None:
    torch.manual_seed(0)
    plan = compile_polynomial(
        window=5,
        channels=2,
        max_depth=3,
        max_offset_multiplicity=1,
    )
    dynamic, enumerated = _matching_layers(plan)
    dynamic_path = torch.randn(2, 11, 2, dtype=torch.float64, requires_grad=True)
    enumerated_path = dynamic_path.detach().clone().requires_grad_()

    dynamic_output = dynamic(dynamic_path)
    enumerated_output = enumerated(enumerated_path)
    projection = torch.randn_like(dynamic_output)
    dynamic_loss = (dynamic_output * projection).sum()
    enumerated_loss = (enumerated_output * projection).sum()
    dynamic_gradients = torch.autograd.grad(
        dynamic_loss,
        (dynamic_path, *dynamic.parameters()),
    )
    enumerated_gradients = torch.autograd.grad(
        enumerated_loss,
        (enumerated_path, *enumerated.parameters()),
    )

    torch.testing.assert_close(
        dynamic_output,
        enumerated_output,
        atol=1e-9,
        rtol=1e-7,
    )
    for actual, expected in zip(
        dynamic_gradients,
        enumerated_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)


def test_structured_parameters_expand_to_plan_in_values_and_gradients() -> None:
    torch.manual_seed(4)
    plan = compile_polynomial(
        window=4,
        channels=2,
        max_depth=3,
        max_offset_multiplicity=1,
        scaling="support",
    )
    layer = GapFactorizedVolterra(
        plan,
        out_features=3,
        temporal_rank=2,
        backend="dynamic",
        gap_initialization="uniform",
        dtype=torch.float64,
    )
    reference = GapFactorizedVolterra(
        plan,
        out_features=3,
        temporal_rank=2,
        backend="enumerate",
        gap_initialization="uniform",
        dtype=torch.float64,
    )
    reference.load_state_dict(layer.state_dict())
    path = torch.randn(2, 9, 2, dtype=torch.float64, requires_grad=True)
    reference_path = path.detach().clone().requires_grad_()
    actual = layer(path)
    windows = increment_windows(reference_path, size=plan.window)
    expected = plan.contract(windows, _expanded_polynomial_weight(reference))
    assert reference.bias is not None
    expected = expected + reference.bias
    projection = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(
        (actual * projection).sum(),
        (path, *layer.parameters()),
    )
    expected_gradients = torch.autograd.grad(
        (expected * projection).sum(),
        (reference_path, *reference.parameters()),
    )

    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=1e-8,
            rtol=1e-6,
        )


def test_span_filtered_plan_uses_enumeration_and_preserves_locality() -> None:
    torch.manual_seed(5)
    plan = compile_polynomial(
        window=5,
        channels=1,
        max_depth=3,
        max_offset_span=2,
        max_offset_multiplicity=1,
    )
    automatic = GapFactorizedVolterra(
        plan,
        out_features=2,
        temporal_rank=2,
        backend="auto",
        gap_initialization="uniform",
        dtype=torch.float64,
    )
    enumerated = GapFactorizedVolterra(
        plan,
        out_features=2,
        temporal_rank=2,
        backend="enumerate",
        gap_initialization="uniform",
        dtype=torch.float64,
    )
    enumerated.load_state_dict(automatic.state_dict())
    path = torch.randn(2, 10, 1, dtype=torch.float64)

    torch.testing.assert_close(automatic(path), enumerated(path))
    with pytest.raises(ValueError, match="span-filtered"):
        GapFactorizedVolterra(
            plan,
            out_features=2,
            temporal_rank=2,
            backend="dynamic",
        )


def test_gap_coordinates_have_no_dead_zero_or_impossible_entries() -> None:
    torch.manual_seed(6)
    temporal_rank = 3
    plan = compile_polynomial(
        window=5,
        channels=2,
        max_depth=3,
        depths=(3,),
        max_offset_multiplicity=1,
    )
    layer = GapFactorizedVolterra(
        plan,
        out_features=2,
        temporal_rank=temporal_rank,
        gap_initialization="uniform",
        dtype=torch.float64,
    )

    assert layer.internal_gap_factors[0].shape == (temporal_rank, 2, 3)
    assert layer.terminal_lag_factors[0].shape == (temporal_rank, 3)
    path = torch.randn(2, 10, 2, dtype=torch.float64)
    layer(path).square().sum().backward()
    for parameter in (
        layer.internal_gap_factors[0],
        layer.terminal_lag_factors[0],
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == parameter.numel()


def test_parameter_count_uses_only_feasible_gap_coordinates() -> None:
    plan = compile_polynomial(
        window=5,
        channels=2,
        max_depth=3,
        max_offset_multiplicity=1,
    )
    layer = GapFactorizedVolterra(
        plan,
        out_features=4,
        temporal_rank=3,
    )

    expected_gap_parameters = sum(3 * depth * (5 - depth + 1) for depth in plan.depths)
    expected_channel_parameters = sum(4 * 3 * 2**depth for depth in plan.depths)
    assert sum(parameter.numel() for parameter in layer.parameters()) == (
        expected_gap_parameters + expected_channel_parameters + 4
    )
    old_padded_gap_parameters = sum(3 * depth * 5 for depth in plan.depths)
    assert old_padded_gap_parameters - expected_gap_parameters == 24


def test_module_is_translation_invariant_and_respects_stride() -> None:
    torch.manual_seed(2)
    plan = compile_polynomial(
        window=4,
        channels=2,
        max_depth=3,
        depths=(1, 3),
        max_offset_multiplicity=1,
    )
    layer = GapFactorizedVolterra(
        plan,
        out_features=3,
        temporal_rank=2,
        stride=2,
        gap_initialization="uniform",
        dtype=torch.float64,
    )
    path = torch.randn(2, 11, 2, dtype=torch.float64)
    shift = torch.tensor([7.0, -4.0], dtype=torch.float64)

    output = layer(path)
    assert output.shape == (2, 4, 3)
    torch.testing.assert_close(output, layer(path + shift))


def test_constructor_requires_a_complete_strict_polynomial_plan() -> None:
    dense = compile_polynomial(window=4, channels=2, max_depth=2)
    with pytest.raises(ValueError, match="max_offset_multiplicity=1"):
        GapFactorizedVolterra(dense, out_features=2, temporal_rank=2)

    incomplete = compile_polynomial(
        window=4,
        channels=2,
        max_depth=2,
        max_offset_support=1,
        max_offset_multiplicity=1,
    )
    with pytest.raises(ValueError, match="no degree-2 features"):
        GapFactorizedVolterra(incomplete, out_features=2, temporal_rank=2)


def test_strict_plan_metadata_is_public_and_immutable() -> None:
    plan = compile_polynomial(
        window=6,
        channels=2,
        max_depth=3,
        max_offset_span=3,
        max_offset_multiplicity=1,
        scaling="support",
    )

    assert plan.max_offset_span == 3
    assert plan.max_offset_support is None
    assert plan.max_offset_multiplicity == 1
    assert plan.scaling == "support"
    with pytest.raises(FrozenInstanceError):
        plan.max_offset_span = None


def test_parameter_count_matches_closed_formula() -> None:
    plan = compile_polynomial(
        window=7,
        channels=3,
        max_depth=3,
        max_offset_multiplicity=1,
    )
    layer = GapFactorizedVolterra(
        plan,
        out_features=5,
        temporal_rank=2,
        bias=False,
    )

    gap_count = sum(2 * depth * (7 - depth + 1) for depth in plan.depths)
    channel_count = sum(5 * 2 * 3**depth for depth in plan.depths)
    assert sum(parameter.numel() for parameter in layer.parameters()) == (
        gap_count + channel_count
    )
    assert plan.feature_count == sum(
        math.comb(7, depth) * 3**depth for depth in plan.depths
    )
