import pytest
import torch

import tensorsig.tensor_ops as tensor_ops
from tensorsig.classical.signatures import signature
from tensorsig.tensor_ops.linear import (
    apply_linear_map_to_signature_levels,
    apply_linear_map_to_tensor_level,
    contract_features_with_readout,
)
from tensorsig.tensor_ops.products import lie_bracket, tensor_product
from tensorsig.tensor_ops.series import (
    batch_restricted_exp,
    flatten_signature_levels,
    signature_inverse,
    signature_multiply,
    unflatten_signature_levels,
)


def test_public_surface_is_organized_by_source_module() -> None:
    assert tensor_ops.__all__ == ["products", "linear", "series"]
    assert tensor_ops.products.__all__ == ["tensor_product", "lie_bracket"]
    assert tensor_ops.linear.__all__ == [
        "apply_linear_map_to_tensor_level",
        "apply_linear_map_to_signature_levels",
        "contract_features_with_readout",
    ]
    assert tensor_ops.series.__all__ == [
        "flatten_signature_levels",
        "unflatten_signature_levels",
        "signature_inverse",
        "signature_multiply",
        "batch_restricted_exp",
    ]

    for old_name in (
        "add_tensor_product",
        "batch_add_tensor_product",
        "batch_lie_brackets",
        "batch_mult_fused_restricted_exp",
        "batch_sequence_tensor_product",
        "batch_tensor_product",
        "lie_brackets",
        "tensor_product_many_shared",
        "tensor_product_shared",
    ):
        assert not hasattr(tensor_ops, old_name)


def test_tensor_product_matches_outer_product() -> None:
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(tensor_product(x, y), torch.outer(x, y))


def test_tensor_product_preserves_exact_shared_axes() -> None:
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    y = torch.randn(2, 3, 5, dtype=torch.float64)

    result = tensor_product(x, y, shared_dims=2)

    assert result.shape == (2, 3, 4, 5)
    torch.testing.assert_close(result, x.unsqueeze(-1) * y.unsqueeze(-2))

    with pytest.raises(ValueError, match="match exactly"):
        tensor_product(x, torch.randn(2, 1, 5), shared_dims=2)

    batched_x = torch.randn(3, 4)
    batched_y = torch.randn(3, 5)
    batched = tensor_product(batched_x, batched_y, shared_dims=1)
    torch.testing.assert_close(
        batched,
        batched_x.unsqueeze(-1) * batched_y.unsqueeze(-2),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_products_preserve_dtype_and_gradients(dtype: torch.dtype) -> None:
    x = torch.randn(2, 3, 3, dtype=dtype, requires_grad=True)
    y = torch.randn(2, 3, 3, 3, dtype=dtype, requires_grad=True)

    product = tensor_product(x, y, shared_dims=1)
    bracket = lie_bracket(x, y, shared_dims=1)

    assert product.shape == (2, 3, 3, 3, 3, 3)
    assert bracket.shape == product.shape
    assert product.dtype == dtype
    assert product.device == x.device
    (product.square().sum() + bracket.square().sum()).backward()
    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    assert y.grad is not None and torch.all(torch.isfinite(y.grad))


def test_tensor_product_supports_scalars() -> None:
    scalar = torch.tensor(2.0)
    vector = torch.tensor([3.0, 4.0])

    torch.testing.assert_close(tensor_product(scalar, vector), 2.0 * vector)
    scalar_product = tensor_product(scalar, torch.tensor(3.0))
    assert scalar_product.shape == ()
    torch.testing.assert_close(scalar_product, torch.tensor(6.0))


def test_tensor_product_validates_shared_dims() -> None:
    x = torch.ones(2)
    with pytest.raises(TypeError, match="integer"):
        tensor_product(x, x, shared_dims=True)
    with pytest.raises(ValueError, match="non-negative"):
        tensor_product(x, x, shared_dims=-1)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        tensor_product(x, x, shared_dims=2)
    with pytest.raises(TypeError):
        tensor_product(x, x, 0)  # type: ignore[misc]


def test_lie_bracket_supports_shared_axes() -> None:
    x = torch.randn(3, 2, dtype=torch.float64)
    y = torch.randn(3, 2, dtype=torch.float64)
    result = lie_bracket(x, y, shared_dims=1)
    expected = x.unsqueeze(-1) * y.unsqueeze(-2) - y.unsqueeze(-1) * x.unsqueeze(-2)
    torch.testing.assert_close(result, expected)

    torch.testing.assert_close(
        lie_bracket(x, x, shared_dims=1),
        torch.zeros_like(result),
    )
    with pytest.raises(ValueError, match="matching product shapes"):
        lie_bracket(torch.ones(2), torch.ones(3))

    plain_x = torch.randn(2)
    plain_y = torch.randn(2)
    assert lie_bracket(plain_x, plain_y).shape == (2, 2)

    sequenced_x = torch.randn(2, 3, 4)
    sequenced_y = torch.randn(2, 3, 4)
    assert lie_bracket(sequenced_x, sequenced_y, shared_dims=2).shape == (
        2,
        3,
        4,
        4,
    )


def test_apply_linear_map_to_tensor_level_maps_each_tensor_axis() -> None:
    level = torch.randn(2, 3, 4, 4, dtype=torch.float64, requires_grad=True)
    matrix = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)

    result = apply_linear_map_to_tensor_level(level, matrix, shared_dims=2)
    expected = torch.einsum("bnij,ip,jq->bnpq", level, matrix, matrix)

    assert result.shape == (2, 3, 5, 5)
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert level.grad is not None
    assert matrix.grad is not None

    with pytest.raises(ValueError, match="same dtype"):
        apply_linear_map_to_tensor_level(
            level.detach(),
            matrix.detach().to(torch.complex128),
            shared_dims=2,
        )


def test_apply_linear_map_to_signature_levels_preserves_degrees() -> None:
    levels = [
        torch.randn(2, 4, dtype=torch.float64),
        torch.randn(2, 4, 4, dtype=torch.float64),
    ]
    matrix = torch.randn(4, 3, dtype=torch.float64)

    mapped = apply_linear_map_to_signature_levels(levels, matrix, shared_dims=1)

    assert mapped[0].shape == (2, 3)
    assert mapped[1].shape == (2, 3, 3)
    torch.testing.assert_close(mapped[0], levels[0] @ matrix)


def test_contract_features_with_readout_preserves_shared_axes() -> None:
    features = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    readout = torch.randn(7, 4, 5, dtype=torch.float64)

    result = contract_features_with_readout(features, readout)
    expected = torch.einsum("bnij,oij->bno", features, readout)

    assert result.shape == (2, 3, 7)
    torch.testing.assert_close(result, expected)

    empty = contract_features_with_readout(
        torch.empty(0, 3, 4, 5, dtype=torch.float64),
        readout,
    )
    assert empty.shape == (0, 3, 7)

    with pytest.raises(ValueError, match="same dtype"):
        contract_features_with_readout(features.float(), readout)


def test_flatten_and_unflatten_signature_levels_round_trip() -> None:
    levels = [
        torch.randn(2, 3, 4, dtype=torch.float64),
        torch.randn(2, 3, 4, 4, dtype=torch.float64),
        torch.randn(2, 3, 4, 4, 4, dtype=torch.float64),
    ]

    flattened = flatten_signature_levels(levels)
    restored = unflatten_signature_levels(flattened, width=4, depth=3)

    assert flattened.shape == (2, 3, 4 + 4**2 + 4**3)
    for actual, expected in zip(restored, levels, strict=True):
        torch.testing.assert_close(actual, expected)

    empty_levels = [torch.empty(0, 4), torch.empty(0, 4, 4)]
    empty_flattened = flatten_signature_levels(empty_levels)
    assert empty_flattened.shape == (0, 4 + 4**2)
    empty_restored = unflatten_signature_levels(empty_flattened, width=4, depth=2)
    assert [level.shape for level in empty_restored] == [
        torch.Size((0, 4)),
        torch.Size((0, 4, 4)),
    ]


def test_level_conversion_interoperates_with_signature_output() -> None:
    path = torch.tensor(
        [[[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]]],
        dtype=torch.float64,
    )
    flattened = signature(path, depth=3)
    levels = unflatten_signature_levels(flattened, width=2, depth=3)
    torch.testing.assert_close(flatten_signature_levels(levels), flattened)


def test_signature_level_layout_is_validated() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        flatten_signature_levels([])
    with pytest.raises(ValueError, match="must have shape"):
        flatten_signature_levels([torch.randn(2, 4), torch.randn(2, 3, 3)])
    with pytest.raises(ValueError, match="trailing dimension"):
        unflatten_signature_levels(torch.randn(2, 7), width=2, depth=3)
    with pytest.raises(TypeError, match="width must be an integer"):
        unflatten_signature_levels(torch.randn(2, 2), width=True, depth=1)


def test_signature_multiply_and_inverse_obey_chen_identity() -> None:
    levels = batch_restricted_exp(
        torch.tensor([[0.5, -1.0]], dtype=torch.float64),
        depth=3,
    )
    identity = signature_multiply(levels, signature_inverse(levels))

    for level in identity:
        torch.testing.assert_close(level, torch.zeros_like(level))


def test_signature_multiplication_is_associative() -> None:
    left = batch_restricted_exp(torch.tensor([[0.2, -0.1]]), depth=3)
    middle = batch_restricted_exp(torch.tensor([[0.3, 0.4]]), depth=3)
    right = batch_restricted_exp(torch.tensor([[-0.5, 0.2]]), depth=3)

    left_grouped = signature_multiply(signature_multiply(left, middle), right)
    right_grouped = signature_multiply(left, signature_multiply(middle, right))
    for actual, expected in zip(left_grouped, right_grouped, strict=True):
        torch.testing.assert_close(actual, expected)


def test_signature_multiply_validates_compatible_series() -> None:
    left = [torch.randn(2, 3), torch.randn(2, 3, 3)]
    with pytest.raises(ValueError, match="same depth"):
        signature_multiply(left, left[:1])
    with pytest.raises(ValueError, match="leading shape and width"):
        signature_multiply(left, [torch.randn(2, 4), torch.randn(2, 4, 4)])
    with pytest.raises(ValueError, match="same dtype"):
        signature_multiply(left, [level.double() for level in left])


def test_batch_restricted_exp_depth_three_and_grad() -> None:
    vector = torch.tensor(
        [[0.5, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    levels = batch_restricted_exp(vector, depth=3)

    assert len(levels) == 3
    torch.testing.assert_close(levels[0], vector)
    torch.testing.assert_close(
        levels[1],
        tensor_product(vector, vector / 2.0, shared_dims=1),
    )
    sum(level.sum() for level in levels).backward()
    assert vector.grad is not None
    assert torch.all(torch.isfinite(vector.grad))


def test_batch_restricted_exp_validates_input() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        batch_restricted_exp(torch.ones(2, 3), depth=0)
    with pytest.raises(ValueError, match="batch, width"):
        batch_restricted_exp(torch.ones(3), depth=2)
    with pytest.raises(ValueError, match="floating-point or complex"):
        batch_restricted_exp(torch.ones(2, 3, dtype=torch.long), depth=2)


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")


@pytest.mark.parametrize("device", DEVICES)
def test_public_tensor_operations_preserve_device(device: str) -> None:
    x = torch.randn(2, 3, device=device, requires_grad=True)
    y = torch.randn(2, 4, device=device, requires_grad=True)
    product = tensor_product(x, y, shared_dims=1)
    assert product.device.type == device

    matrix = torch.randn(3, 2, device=device, requires_grad=True)
    mapped = apply_linear_map_to_tensor_level(x, matrix, shared_dims=1)
    readout = torch.randn(5, 2, device=device, requires_grad=True)
    contracted = contract_features_with_readout(mapped, readout, shared_dims=1)
    assert mapped.device.type == device
    assert contracted.device.type == device

    levels = batch_restricted_exp(x, depth=3)
    restored = unflatten_signature_levels(
        flatten_signature_levels(levels), width=3, depth=3
    )
    multiplied = signature_multiply(restored, signature_inverse(restored))
    assert all(level.device.type == device for level in multiplied)

    (
        product.sum() + contracted.sum() + sum(level.sum() for level in multiplied)
    ).backward()
    assert x.grad is not None
    assert y.grad is not None
    assert matrix.grad is not None
    assert readout.grad is not None
