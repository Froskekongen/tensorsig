import pytest
import torch

from tensorsig.offset_coded_signatures.augmentations import (
    CausalFIRAugmentation,
    CausalWindowedLogSignature,
    PairwiseProductAugmentation,
    causal_fir,
)
from tensorsig.algebra.lyndon import logsigdim_words
from tensorsig.classical.windows import windowed_log_signature


def test_causal_fir_matches_manual_lag_and_padding_convention() -> None:
    path = torch.tensor([[[1.0], [2.0], [4.0], [8.0]]], dtype=torch.float64)
    weight = torch.tensor([[[10.0, 1.0, 2.0]]], dtype=torch.float64)

    repeat = causal_fir(path, weight, padding="repeat")
    zero = causal_fir(path, weight, padding="zero")

    expected_repeat = torch.tensor([[[13.0], [23.0], [44.0], [88.0]]], dtype=path.dtype)
    expected_zero = torch.tensor([[[10.0], [21.0], [44.0], [88.0]]], dtype=path.dtype)
    torch.testing.assert_close(repeat, expected_repeat)
    torch.testing.assert_close(zero, expected_zero)


def test_causal_fir_mixes_channels_and_applies_bias() -> None:
    path = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]],
        dtype=torch.float64,
    )
    weight = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]], dtype=path.dtype)
    bias = torch.tensor([0.5], dtype=path.dtype)

    actual = causal_fir(path, weight, bias, padding="zero")
    expected = torch.tensor([[[5.5], [24.5], [58.5]]], dtype=path.dtype)

    torch.testing.assert_close(actual, expected)


def test_causal_fir_augmentation_preserves_inputs() -> None:
    path = torch.tensor(
        [
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        ],
        dtype=torch.float64,
    )
    augmentation = CausalFIRAugmentation(
        in_channels=2,
        num_filters=2,
        max_lag=1,
        bias=False,
        padding="zero",
    ).double()
    with torch.no_grad():
        augmentation.weight.zero_()
        augmentation.weight[0, 0, 0] = 1.0
        augmentation.weight[1, 1, 1] = 1.0

    actual = augmentation(path)
    expected_filters = torch.tensor(
        [[[1.0, 0.0], [2.0, 10.0], [3.0, 20.0], [4.0, 30.0]]],
        dtype=path.dtype,
    )

    assert augmentation.out_channels == 4
    torch.testing.assert_close(actual[..., :2], path)
    torch.testing.assert_close(actual[..., 2:], expected_filters)
    torch.testing.assert_close(augmentation.filtered(path), expected_filters)


def test_causal_fir_augmentation_ignores_future_perturbations() -> None:
    torch.manual_seed(0)
    path = torch.randn(2, 12, 3, dtype=torch.float64)
    changed = path.clone()
    cutoff = 5
    changed[:, cutoff + 1 :, :] = 100.0 * torch.randn_like(changed[:, cutoff + 1 :, :])

    augmentation = CausalFIRAugmentation(
        in_channels=3,
        num_filters=4,
        max_lag=3,
        bias=True,
        padding="repeat",
    ).double()

    torch.testing.assert_close(
        augmentation(path)[:, : cutoff + 1],
        augmentation(changed)[:, : cutoff + 1],
    )


def test_causal_fir_augmentation_requires_matching_module_state() -> None:
    augmentation = CausalFIRAugmentation(2, 3, 1)
    path = torch.randn(1, 4, 2, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="dtype/device must match"):
        augmentation(path)


def test_pairwise_product_augmentation_uses_distinct_pairs() -> None:
    path = torch.tensor([[[2.0, 3.0, 5.0]]], dtype=torch.float64)
    products = PairwiseProductAugmentation(3, include_inputs=False)
    augmented = PairwiseProductAugmentation(3)

    expected_products = torch.tensor([[[6.0, 10.0, 15.0]]], dtype=path.dtype)
    torch.testing.assert_close(products(path), expected_products)
    torch.testing.assert_close(
        augmented(path),
        torch.cat((path, expected_products), dim=-1),
    )
    assert products.out_channels == 3
    assert augmented.out_channels == 6


def test_path_augmentations_compose_with_sequential() -> None:
    path = torch.randn(2, 8, 2, dtype=torch.float64)
    fir = CausalFIRAugmentation(2, 3, 2, bias=False).double()
    products = PairwiseProductAugmentation(fir.out_channels)

    composed = torch.nn.Sequential(fir, products)

    assert composed(path).shape == (2, 8, products.out_channels)


def test_causal_windowed_logsignature_shapes_and_gradients() -> None:
    torch.manual_seed(1)
    path = torch.randn(2, 14, 2, dtype=torch.float64, requires_grad=True)
    fir = CausalFIRAugmentation(2, 3, 4, bias=False).double()
    module = CausalWindowedLogSignature(
        fir,
        depth=2,
        window_size=5,
        hop_size=2,
        mode="words",
    )

    output = module(path)
    expected_windows = 1 + (path.shape[1] - module.window_size) // module.hop_size
    assert module.embedded_channels == 5
    assert module.signature_channels == 5
    assert output.shape == (2, expected_windows, logsigdim_words(5, 2))

    output.square().sum().backward()
    assert fir.weight.grad is not None
    assert fir.weight.grad.abs().sum() > 0
    assert path.grad is not None
    assert path.grad.abs().sum() > 0


def test_causal_windowed_logsignature_composes_pairwise_products_and_bias() -> None:
    torch.manual_seed(2)
    path = torch.randn(1, 10, 2, dtype=torch.float64)
    fir = CausalFIRAugmentation(2, 3, 2, bias=True).double()
    products = PairwiseProductAugmentation(fir.out_channels)
    module = CausalWindowedLogSignature(
        fir,
        depth=2,
        window_size=4,
        hop_size=3,
        pairwise=products,
        mode="hall",
    )

    output = module(path)
    expected = windowed_log_signature(
        products(fir(path)),
        depth=2,
        window_size=4,
        hop_size=3,
        mode="hall",
    )
    expected_windows = 1 + (path.shape[1] - module.window_size) // module.hop_size
    assert module.embedded_channels == 5
    assert module.signature_channels == 15
    assert output.shape == (1, expected_windows, logsigdim_words(15, 2))
    torch.testing.assert_close(output, expected)

    output.square().sum().backward()
    assert fir.weight.grad is not None
    assert fir.bias is not None and fir.bias.grad is not None
    assert fir.bias.grad.abs().sum() > 0
