import pytest
import torch

from tensorsig.classical.signatures import signature
from tensorsig.classical.windows import windowed_signature
from tensorsig.offset_coded_signatures.codes import AlphabetMap, OffsetCode
from tensorsig.offset_coded_signatures.layers import LocalFeatures
from tensorsig.offset_coded_signatures.plans import LocalRealization
from tensorsig.offset_coded_signatures.windows import increment_windows
from tensorsig.offset_coded_signatures.sliding import (
    _rolling_coded_signature_levels,
)
from tensorsig.tensor_ops.series import unflatten_signature_levels


def _explicit_coded_levels(
    path: torch.Tensor,
    matrix: torch.Tensor,
    depth: int,
    stride: int,
) -> tuple[torch.Tensor, ...]:
    windows = increment_windows(path, size=matrix.shape[0], stride=stride)
    lifted = torch.einsum("bnrc,re->bnrce", windows, matrix).flatten(-2)
    batch, count, steps, width = lifted.shape
    increments = lifted.reshape(batch * count, steps, width)
    zero = increments.new_zeros(batch * count, 1, width)
    lifted_paths = torch.cat((zero, increments.cumsum(dim=1)), dim=1)
    flattened = signature(lifted_paths, depth=depth)
    return tuple(
        level.reshape(batch, count, *level.shape[1:])
        for level in unflatten_signature_levels(flattened, width, depth)
    )


def test_constant_code_is_the_unscaled_offset_forgetting_code() -> None:
    code = OffsetCode.constant(5, dtype=torch.float64)

    torch.testing.assert_close(code.matrix, torch.ones(5, 1, dtype=torch.float64))
    assert code.family == "pascal"
    assert code.temporal_degrees == (0,)

    path = torch.randn(2, 19, 2, dtype=torch.float64)
    levels = _rolling_coded_signature_levels(
        path,
        code.matrix,
        channels=2,
        depth=3,
        stride=2,
    )
    actual = torch.cat([level.flatten(start_dim=2) for level in levels], dim=-1)
    expected = windowed_signature(path, depth=3, window_size=6, hop_size=2)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("orthonormal", [False, True], ids=["raw", "orthonormal"])
@pytest.mark.parametrize("stride", [1, 2])
def test_pascal_shift_operator_is_the_exact_backward_rebase(
    orthonormal: bool,
    stride: int,
) -> None:
    code = OffsetCode.pascal(
        7,
        3,
        orthonormal=orthonormal,
        dtype=torch.float64,
    )

    shift = code.shift_operator(stride)

    torch.testing.assert_close(
        code.matrix[stride:] @ shift,
        code.matrix[:-stride],
        atol=1e-12,
        rtol=1e-11,
    )


@pytest.mark.parametrize(
    "code",
    [
        OffsetCode.pascal(6, 3, orthonormal=False, dtype=torch.float64),
        OffsetCode.pascal(6, 3, orthonormal=True, dtype=torch.float64),
        OffsetCode.identity(6, dtype=torch.float64),
    ],
    ids=["raw-pascal", "orthonormal-pascal", "identity"],
)
@pytest.mark.parametrize("stride", [1, 2])
def test_rolling_levels_match_explicit_window_signatures(
    code: OffsetCode,
    stride: int,
) -> None:
    torch.manual_seed(12)
    increments = 0.1 * torch.randn(2, 35, 2, dtype=torch.float64)
    path = torch.cat(
        (increments.new_zeros(2, 1, 2), increments.cumsum(dim=1)),
        dim=1,
    )

    actual = _rolling_coded_signature_levels(
        path,
        code.matrix,
        channels=2,
        depth=3,
        stride=stride,
    )
    expected = _explicit_coded_levels(path, code.matrix, depth=3, stride=stride)

    for actual_level, expected_level in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_level,
            expected_level,
            atol=2e-9,
            rtol=2e-8,
        )


@pytest.mark.parametrize(
    ("dtype", "relative_tolerance"),
    [(torch.float32, 1e-3), (torch.float64, 1e-10)],
)
def test_periodic_refresh_bounds_long_horizon_drift(
    dtype: torch.dtype,
    relative_tolerance: float,
) -> None:
    torch.manual_seed(4)
    code = OffsetCode.pascal(8, 3, dtype=dtype)
    increments = torch.randn(1, 4096, 1, dtype=dtype)
    path = torch.cat(
        (increments.new_zeros(1, 1, 1), increments.cumsum(dim=1)),
        dim=1,
    )

    actual = _rolling_coded_signature_levels(
        path,
        code.matrix,
        channels=1,
        depth=3,
    )
    expected = _explicit_coded_levels(path, code.matrix, depth=3, stride=1)
    absolute_error = max(
        float((actual_level - expected_level).abs().max())
        for actual_level, expected_level in zip(actual, expected, strict=True)
    )
    scale = max(float(level.abs().max()) for level in expected)

    assert absolute_error / scale < relative_tolerance


def test_shift_operator_rejects_nonclosed_and_trainable_codes() -> None:
    torch.manual_seed(9)
    nonclosed = OffsetCode(torch.randn(6, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="not exactly shift-closed"):
        nonclosed.shift_operator()

    trainable = OffsetCode(torch.nn.Parameter(torch.ones(5, 1, dtype=torch.float64)))
    with pytest.raises(ValueError, match="fixed offset code"):
        trainable.shift_operator()


def test_rolling_rejects_nonclosed_and_trainable_matrices() -> None:
    torch.manual_seed(10)
    path = torch.randn(1, 20, 1, dtype=torch.float64)
    nonclosed = torch.randn(5, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="not exactly shift-closed"):
        _rolling_coded_signature_levels(path, nonclosed, channels=1, depth=2)

    trainable = torch.nn.Parameter(torch.ones(5, 1, dtype=torch.float64))
    with pytest.raises(ValueError, match="fixed offset code"):
        _rolling_coded_signature_levels(path, trainable, channels=1, depth=2)


def test_public_rolling_backend_matches_vectorized_values_and_path_gradients() -> None:
    torch.manual_seed(23)
    code = OffsetCode.pascal(8, 3, dtype=torch.float64)
    plan = LocalRealization(code, channels=2, max_depth=3).signature(
        coordinates="dense"
    )
    rolling_path = torch.randn(2, 41, 2, dtype=torch.float64, requires_grad=True)
    vectorized_path = rolling_path.detach().clone().requires_grad_()

    rolling = LocalFeatures(plan, backend="rolling")(rolling_path)
    vectorized = LocalFeatures(plan, backend="vectorized")(vectorized_path)
    weights = torch.randn_like(rolling)
    rolling_gradient = torch.autograd.grad((rolling * weights).sum(), rolling_path)[0]
    vectorized_gradient = torch.autograd.grad(
        (vectorized * weights).sum(), vectorized_path
    )[0]

    torch.testing.assert_close(rolling, vectorized, atol=2e-9, rtol=2e-8)
    torch.testing.assert_close(
        rolling_gradient,
        vectorized_gradient,
        atol=2e-8,
        rtol=2e-7,
    )


def test_public_rolling_log_signature_matches_dense_values_and_path_gradients() -> None:
    torch.manual_seed(24)
    code = OffsetCode.pascal(7, 2, dtype=torch.float64)
    plan = LocalRealization(code, channels=2, max_depth=3).log_signature()
    rolling_path = torch.randn(2, 31, 2, dtype=torch.float64, requires_grad=True)
    dense_path = rolling_path.detach().clone().requires_grad_()

    rolling = LocalFeatures(plan, backend="rolling")(rolling_path)
    dense = LocalFeatures(plan, backend="dense")(dense_path)
    weights = torch.randn_like(rolling)
    rolling_gradient = torch.autograd.grad((rolling * weights).sum(), rolling_path)[0]
    dense_gradient = torch.autograd.grad((dense * weights).sum(), dense_path)[0]

    torch.testing.assert_close(rolling, dense, atol=2e-9, rtol=2e-8)
    torch.testing.assert_close(
        rolling_gradient,
        dense_gradient,
        atol=2e-8,
        rtol=2e-7,
    )


def test_public_constant_auto_backend_matches_vectorized_values_and_gradients() -> None:
    torch.manual_seed(25)
    plan = LocalRealization(
        OffsetCode.constant(12, dtype=torch.float64),
        channels=2,
        max_depth=3,
    ).signature(coordinates="dense")
    automatic_path = torch.randn(2, 43, 2, dtype=torch.float64, requires_grad=True)
    vectorized_path = automatic_path.detach().clone().requires_grad_()

    automatic = LocalFeatures(plan, backend="auto")(automatic_path)
    vectorized = LocalFeatures(plan, backend="vectorized")(vectorized_path)
    weights = torch.randn_like(automatic)
    automatic_gradient = torch.autograd.grad(
        (automatic * weights).sum(), automatic_path
    )[0]
    vectorized_gradient = torch.autograd.grad(
        (vectorized * weights).sum(), vectorized_path
    )[0]

    torch.testing.assert_close(automatic, vectorized, atol=2e-10, rtol=2e-9)
    torch.testing.assert_close(
        automatic_gradient,
        vectorized_gradient,
        atol=2e-9,
        rtol=2e-8,
    )


def test_public_rolling_backend_has_a_narrow_exact_domain() -> None:
    fixed = OffsetCode.pascal(4, 2, dtype=torch.float64)
    chronological = LocalRealization(fixed, 1, 2).signature()
    LocalFeatures(chronological, backend="rolling")
    numerical = LocalRealization(OffsetCode(fixed.matrix), 1, 2).signature()
    with pytest.raises(ValueError, match="structural Pascal"):
        LocalFeatures(numerical, backend="rolling")

    general = AlphabetMap(torch.ones(4, 1, 1, dtype=torch.float64))
    dense_map = LocalRealization(general, 1, 2).signature()
    with pytest.raises(ValueError, match="separable OffsetCode"):
        LocalFeatures(dense_map, backend="rolling")

    nonclosed = LocalRealization(
        OffsetCode(torch.randn(4, 2, dtype=torch.float64)),
        1,
        2,
    ).signature(coordinates="dense")
    with pytest.raises(ValueError, match="not exactly shift-closed"):
        LocalFeatures(nonclosed, backend="rolling")
