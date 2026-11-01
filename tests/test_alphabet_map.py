import pytest
import torch

from tensorsig.classical.signatures import signature
from tensorsig.classical.windows import windowed_signature
from tensorsig.offset_coded_signatures.codes import AlphabetMap, OffsetCode
from tensorsig.offset_coded_signatures.layers import LocalFeatures, LocalLinear
from tensorsig.offset_coded_signatures.plans import LocalRealization
from tensorsig.offset_coded_signatures.windows import increment_windows
from tensorsig.tensor_ops.series import unflatten_signature_levels


def _explicit_projected_signature(
    path: torch.Tensor,
    projection: torch.Tensor,
    depth: int,
    stride: int = 1,
) -> torch.Tensor:
    windows = increment_windows(path, size=projection.shape[0], stride=stride)
    lifted = torch.einsum("bnrc,rcd->bnrd", windows, projection)
    zero = lifted.new_zeros(*lifted.shape[:2], 1, lifted.shape[-1])
    lifted_paths = torch.cat((zero, lifted.cumsum(dim=2)), dim=2)
    return signature(
        lifted_paths.flatten(0, 1),
        depth=depth,
    ).reshape(*lifted.shape[:2], -1)


def test_constant_code_is_the_ordinary_sliding_window_signature() -> None:
    torch.manual_seed(0)
    path = torch.randn(2, 8, 3, dtype=torch.float64)
    plan = LocalRealization(
        OffsetCode.constant(4, dtype=path.dtype),
        channels=3,
        max_depth=3,
    ).signature(coordinates="dense")

    actual = LocalFeatures(plan)(path)
    expected = windowed_signature(
        path,
        depth=3,
        window_size=5,
        hop_size=1,
    )
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)


def test_general_alphabet_map_matches_explicit_lifted_paths() -> None:
    torch.manual_seed(1)
    path = torch.randn(2, 8, 2, dtype=torch.float64)
    projection = torch.randn(3, 2, 4, dtype=path.dtype)
    plan = LocalRealization(
        AlphabetMap(projection),
        channels=2,
        max_depth=3,
    ).signature()

    torch.testing.assert_close(
        LocalFeatures(plan, stride=2)(path),
        _explicit_projected_signature(path, projection, depth=3, stride=2),
        atol=1e-11,
        rtol=1e-9,
    )


def test_projected_signature_rejects_integer_paths() -> None:
    plan = LocalRealization(
        AlphabetMap(torch.ones(1, 1, 1)),
        channels=1,
        max_depth=1,
    ).signature()
    with pytest.raises(ValueError, match="floating-point dtype"):
        LocalFeatures(plan)(torch.tensor([[[0], [2]]], dtype=torch.int64))


def test_projected_linear_matches_flattened_tensor_readouts() -> None:
    torch.manual_seed(2)
    path = torch.randn(2, 8, 2, dtype=torch.float64)
    projection = torch.randn(3, 2, 3, dtype=path.dtype)
    plan = LocalRealization(
        AlphabetMap(projection),
        channels=2,
        max_depth=3,
    ).signature(depths=(1, 3))
    layer = LocalLinear(plan, out_features=4, stride=2, dtype=path.dtype)
    kernels = (
        torch.randn(4, 3, dtype=path.dtype),
        torch.randn(4, 3, 3, 3, dtype=path.dtype),
    )
    bias = torch.randn(4, dtype=path.dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.cat([kernel.flatten(1) for kernel in kernels], dim=1))
        assert layer.bias is not None
        layer.bias.copy_(bias)

    ambient = _explicit_projected_signature(path, projection, depth=3, stride=2)
    levels = unflatten_signature_levels(ambient, width=3, depth=3)
    selected = torch.cat((levels[0].flatten(2), levels[2].flatten(2)), dim=-1)
    expected = torch.nn.functional.linear(selected, layer.weight, layer.bias)
    torch.testing.assert_close(layer(path), expected, atol=1e-11, rtol=1e-9)


def test_trainable_alphabet_map_propagates_path_and_matrix_gradients() -> None:
    torch.manual_seed(3)
    path = torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True)
    matrix = torch.nn.Parameter(torch.randn(3, 2, 3, dtype=path.dtype))
    plan = LocalRealization(
        AlphabetMap(matrix),
        channels=2,
        max_depth=2,
    ).signature()
    layer = LocalLinear(plan, out_features=2, dtype=path.dtype)

    layer(path).square().mean().backward()
    assert layer.matrix is matrix
    assert path.grad is not None and torch.isfinite(path.grad).all()
    assert matrix.grad is not None and torch.isfinite(matrix.grad).all()
    assert matrix.grad.abs().sum() > 0
