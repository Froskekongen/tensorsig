import itertools
import math

import pytest
import torch

from tensorsig.classical.windows import windowed_signature
from tensorsig.offset_coded_signatures.layers import LocalLinear
from tensorsig.offset_coded_signatures.plans import (
    PolynomialPlan,
    compile_polynomial,
)
from tensorsig.offset_coded_signatures.windows import increment_windows


def _raw_signature_feature_count(
    channels: int,
    window: int,
    max_depth: int,
    depths: tuple[int, ...] | None = None,
    max_offset_span: int | None = None,
    max_offset_support: int | None = None,
    max_offset_multiplicity: int | None = None,
) -> int:
    return compile_polynomial(
        window=window,
        channels=channels,
        max_depth=max_depth,
        depths=depths,
        max_offset_span=max_offset_span,
        max_offset_support=max_offset_support,
        max_offset_multiplicity=max_offset_multiplicity,
    ).feature_count


def _ordinary_signature_readout(plan: PolynomialPlan) -> torch.Tensor:
    """Tie offset-retaining monomials into one ordinary tensor level."""
    assert len(plan.depths) == 1
    depth = plan.depths[0]
    words = tuple(itertools.product(range(plan.channels), repeat=depth))
    readout = torch.zeros(
        len(words),
        plan.feature_count,
        dtype=plan.coefficients[0].dtype,
    )
    for feature, (offset_row, channel_row) in enumerate(
        zip(
            plan.offset_indices[0].tolist(),
            plan.channel_indices[0].tolist(),
            strict=True,
        )
    ):
        for coordinate, word in enumerate(words):
            canonical_channels = []
            run_start = 0
            for stop in range(1, depth + 1):
                if stop == depth or offset_row[stop] != offset_row[stop - 1]:
                    canonical_channels.extend(sorted(word[run_start:stop]))
                    run_start = stop
            if canonical_channels == channel_row:
                readout[coordinate, feature] = 1
    return readout


def test_raw_scalar_feature_count_matches_minimal_polynomial_dimension() -> None:
    channels = 3
    window = 4
    variables = channels * window

    for depth in range(1, 5):
        plan = compile_polynomial(
            window=window,
            channels=channels,
            max_depth=depth,
            depths=(depth,),
        )
        offsets = plan.offset_indices[0]
        channel_indices = plan.channel_indices[0]
        coefficients = plan.coefficients[0]
        expected = math.comb(variables + depth - 1, depth)
        assert offsets.shape == channel_indices.shape == (expected, depth)
        assert coefficients.shape == (expected,)

    assert _raw_signature_feature_count(channels, window, 3) == sum(
        math.comb(variables + depth - 1, depth) for depth in range(1, 4)
    )

    scalar_plan = compile_polynomial(
        window=4,
        channels=1,
        max_depth=4,
        depths=(4,),
    )
    coefficients_by_offset = {
        tuple(offsets): coefficient
        for offsets, coefficient in zip(
            scalar_plan.offset_indices[0].tolist(),
            scalar_plan.coefficients[0].tolist(),
            strict=True,
        )
    }
    assert coefficients_by_offset[(0, 0, 0, 0)] == pytest.approx(1 / 24)
    assert coefficients_by_offset[(0, 0, 2, 3)] == pytest.approx(1 / 2)
    assert coefficients_by_offset[(1, 1, 2, 2)] == pytest.approx(1 / 4)
    assert coefficients_by_offset[(0, 1, 2, 3)] == pytest.approx(1.0)


def test_increment_windows_uses_increment_count_and_stride() -> None:
    increments = torch.arange(2 * 7 * 3, dtype=torch.float64).reshape(2, 7, 3)
    zero = increments.new_zeros(2, 1, 3)
    path = torch.cat((zero, increments.cumsum(dim=1)), dim=1)

    windows = increment_windows(path, size=3, stride=2)

    assert windows.shape == (2, 3, 3, 3)
    for window_index in range(windows.shape[1]):
        start = window_index * 2
        torch.testing.assert_close(
            windows[:, window_index], increments[:, start : start + 3]
        )


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_offset_tying_recovers_the_ordinary_window_signature(depth: int) -> None:
    torch.manual_seed(10 + depth)
    path = torch.randn(2, 8, 2, dtype=torch.float64)
    plan = compile_polynomial(
        window=4,
        channels=2,
        max_depth=depth,
        depths=(depth,),
    )
    actual = plan.contract(
        increment_windows(path, size=4, stride=2),
        _ordinary_signature_readout(plan),
    )
    expected = windowed_signature(path, depth=depth, window_size=5, hop_size=2)[
        ..., -(2**depth) :
    ]

    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)


def test_raw_scalar_features_are_the_explicit_degree_two_monomials() -> None:
    windows = torch.tensor(
        [[[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]]],
        dtype=torch.float64,
    )
    plan = compile_polynomial(
        window=3,
        channels=2,
        max_depth=2,
        depths=(2,),
        scaling="support",
    )
    features = plan.evaluate(windows)
    offsets = plan.offset_indices[0]
    channels = plan.channel_indices[0]

    seen = set()
    for feature_index, (offset_row, channel_row) in enumerate(
        zip(offsets.tolist(), channels.tolist(), strict=True)
    ):
        letters = tuple(
            sorted(
                (int(offset), int(channel))
                for offset, channel in zip(offset_row, channel_row, strict=True)
            )
        )
        seen.add(letters)
        expected = windows.new_tensor(1.0)
        for offset, channel in letters:
            expected = expected * windows[0, 0, offset, channel]
        torch.testing.assert_close(features[0, 0, feature_index], expected)

    assert len(seen) == math.comb(3 * 2 + 2 - 1, 2)


def test_polynomial_plan_rejects_integer_windows() -> None:
    plan = compile_polynomial(window=1, channels=1, max_depth=2)
    with pytest.raises(ValueError, match="floating-point dtype"):
        plan.evaluate(torch.ones(1, 1, 1, 1, dtype=torch.int64))


def test_polynomial_linear_matches_explicit_readout_and_is_translation_invariant() -> (
    None
):
    torch.manual_seed(1)
    path = torch.randn(2, 8, 2, dtype=torch.float64)
    shift = torch.tensor([4.0, -9.0], dtype=torch.float64)
    plan = compile_polynomial(window=3, channels=2, max_depth=3)
    layer = LocalLinear(
        plan,
        out_features=3,
        chunk_size=5,
        dtype=torch.float64,
    )

    expected = torch.nn.functional.linear(
        plan.evaluate(increment_windows(path, size=3), chunk_size=5),
        layer.weight,
        layer.bias,
    )
    torch.testing.assert_close(layer(path), expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(
        layer(path),
        layer(path + shift.view(1, 1, 2)),
        atol=1e-12,
        rtol=1e-10,
    )


def test_multiplicity_filter_removes_same_offset_squares() -> None:
    increments = torch.tensor([[[2.0], [3.0], [5.0]]], dtype=torch.float64)
    windows = increments.unsqueeze(1)
    strict = compile_polynomial(
        window=3,
        channels=1,
        max_depth=2,
        depths=(2,),
        max_offset_multiplicity=1,
        scaling="support",
    )
    assert not any(tuple(row.tolist()) == (1, 1) for row in strict.offset_indices[0])
    assert strict.evaluate(windows).shape[-1] == strict.feature_count

    full = compile_polynomial(
        window=3,
        channels=1,
        max_depth=2,
        depths=(2,),
    )
    square_index = int(
        torch.nonzero((full.offset_indices[0] == torch.tensor([[1, 1]])).all(dim=1))[
            0
        ].item()
    )
    torch.testing.assert_close(
        full.evaluate(windows)[0, 0, square_index],
        torch.tensor(4.5, dtype=increments.dtype),
    )


def test_nonrepeating_feature_count_matches_strict_offset_dimension() -> None:
    channels = 2
    window = 5
    assert _raw_signature_feature_count(
        channels,
        window,
        4,
        max_offset_multiplicity=1,
    ) == sum(math.comb(window, depth) * channels**depth for depth in range(1, 5))
    assert (
        _raw_signature_feature_count(
            channels,
            window,
            3,
            depths=(2, 3),
            max_offset_span=2,
            max_offset_multiplicity=1,
        )
        == 7 * channels**2 + 3 * channels**3
    )


def test_filtered_polynomial_linear_is_translation_invariant() -> None:
    torch.manual_seed(2)
    path = torch.randn(2, 9, 2, dtype=torch.float64)
    shift = torch.tensor([10.0, -3.0], dtype=torch.float64)
    plan = compile_polynomial(
        window=4,
        channels=2,
        max_depth=3,
        max_offset_span=2,
        max_offset_multiplicity=1,
        scaling="support",
    )
    layer = LocalLinear(
        plan,
        out_features=3,
        chunk_size=4,
        dtype=torch.float64,
    )
    windows = increment_windows(path, size=4)
    expected = plan.contract(windows, layer.weight, chunk_size=4) + layer.bias

    torch.testing.assert_close(layer(path), expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(
        layer(path),
        layer(path + shift.view(1, 1, 2)),
        atol=1e-12,
        rtol=1e-10,
    )


def test_filtered_polynomial_plan_gradcheck() -> None:
    torch.manual_seed(3)
    path = torch.randn(1, 5, 2, dtype=torch.float64, requires_grad=True)
    plan = compile_polynomial(
        window=3,
        channels=2,
        max_depth=2,
        max_offset_multiplicity=1,
        scaling="support",
    )
    readout = torch.randn(
        1,
        plan.feature_count,
        dtype=torch.float64,
        requires_grad=True,
    )

    def evaluate(candidate: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return plan.contract(
            increment_windows(candidate, size=3),
            weight,
            chunk_size=2,
        )

    assert torch.autograd.gradcheck(evaluate, (path, readout), eps=1e-6, atol=1e-4)
