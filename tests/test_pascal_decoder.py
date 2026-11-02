import itertools

import pytest
import torch

from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.tensor_ops.series import (
    signature_levels_from_increments,
)
from tensorsig.offset_coded_signatures._charts.pascal import _PascalDecoder


def _words(window: int, dimension: int, channels: int, depth: int) -> torch.Tensor:
    words = []
    for word in itertools.product(range(dimension * channels), repeat=depth):
        temporal = sum(letter % dimension for letter in word)
        descents = sum(
            left // dimension > right // dimension
            for left, right in zip(word, word[1:])
        )
        if temporal + descents < window:
            words.append(word)
    # Reconstruction must not expose or require its private triangular order.
    return torch.tensor(words[::-1], dtype=torch.long)


def _flat_indices(words: torch.Tensor, width: int) -> torch.Tensor:
    return sum(
        words[:, j] * width ** (words.shape[1] - j - 1) for j in range(words.shape[1])
    )


@pytest.mark.parametrize("orthonormal", [False, True], ids=["raw", "qr"])
@pytest.mark.parametrize(
    "window,dimension,channels,depth",
    [(1, 1, 2, 3), (2, 2, 1, 5), (2, 2, 2, 3), (3, 2, 2, 3), (3, 3, 2, 3)],
)
def test_decoder_values_and_vjps_match_dense(
    orthonormal: bool, window: int, dimension: int, channels: int, depth: int
) -> None:
    code = OffsetCode.pascal(
        window, dimension, orthonormal=orthonormal, dtype=torch.float64
    )
    words = _words(window, dimension, channels, depth)
    decoder = _PascalDecoder(code, words)
    width = dimension * channels
    all_words = torch.tensor(list(itertools.product(range(width), repeat=depth)))
    targets = torch.cat((all_words.flip(0), all_words[:3], all_words[:1]))
    generator = torch.Generator().manual_seed(340)
    increments = torch.randn(
        2, 3, window, channels, generator=generator, dtype=torch.float64
    )
    increments.requires_grad_()
    lifted = torch.einsum("bnrc,rp->bnrcp", increments, code.matrix).flatten(-2)
    dense = signature_levels_from_increments(lifted, depth)[-1].flatten(2)
    features = dense.index_select(-1, _flat_indices(words, width))
    expected = dense.index_select(-1, _flat_indices(targets, width))
    actual = decoder.decode(features, targets)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    cotangent = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    actual_vjp = torch.autograd.grad(actual, increments, cotangent, retain_graph=True)[
        0
    ]
    expected_vjp = torch.autograd.grad(expected, increments, cotangent)[0]
    torch.testing.assert_close(actual_vjp, expected_vjp, atol=1e-10, rtol=1e-10)
    assert decoder.decode(features, targets[:0]).shape == (*features.shape[:-1], 0)
    # A fresh feature graph after a prior backward must not encounter cached tapes.
    leaf = features.detach().clone().requires_grad_()
    decoder.decode(leaf, targets).sum().backward()
    assert leaf.grad is not None


def test_raw_decoder_keeps_exact_integer_coefficients() -> None:
    code = OffsetCode.pascal(3, 2, orthonormal=False, dtype=torch.float64)
    words = _words(3, 2, 2, 2)
    decoder = _PascalDecoder(code, words)
    h = decoder._rows
    assert all(type(value) is int for row in h for value in row)
    assert all(h[i][i] == 1 and not any(h[i][i + 1 :]) for i in range(len(h)))
    target = decoder._entries(((3, 1),))[0]
    # Exact integer backward substitution solves c H = H_target, no float oracle.
    coefficients = [0] * len(h)
    for j in reversed(range(len(h))):
        coefficients[j] = target[j] - sum(
            coefficients[k] * h[k][j] for k in range(j + 1, len(h))
        )
    actual = {
        tuple(words[public_index].tolist()): coefficients[private_index]
        for private_index, public_index in enumerate(decoder._order)
        if coefficients[private_index]
    }
    assert actual == {(1, 3): 1, (3, 0): 2, (1, 2): -2}


@pytest.mark.parametrize("orthonormal", [False, True], ids=["raw", "qr"])
def test_pascal_provenance_survives_dtype_conversion(orthonormal: bool) -> None:
    code = OffsetCode.pascal(7, 4, orthonormal=orthonormal, dtype=torch.float64)
    assert code._pascal_raw is (not orthonormal)
    factor = code._pascal_factor
    assert (
        factor is not None
        and factor.dtype == torch.float64
        and factor.device.type == "cpu"
    )
    raw = OffsetCode.pascal(7, 4, orthonormal=False, dtype=torch.float64).matrix
    torch.testing.assert_close(code.matrix @ factor, raw)
    converted = code.to(dtype=torch.float32).to(dtype=torch.float64)
    assert converted._pascal_raw is code._pascal_raw
    torch.testing.assert_close(converted._pascal_factor, factor, atol=0, rtol=0)
    torch.testing.assert_close(
        converted._pascal_shift_operator(),
        code._pascal_shift_operator(),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        code._pascal_shift_operator(-2) @ code._pascal_shift_operator(2),
        torch.eye(4, dtype=torch.float64),
        atol=1e-11,
        rtol=1e-11,
    )
    # One surviving row cannot identify transport numerically; provenance can.
    torch.testing.assert_close(
        code.matrix[6:] @ code.shift_operator(6), code.matrix[:-6]
    )
    constant = OffsetCode.constant(7)
    assert constant._pascal_raw
    torch.testing.assert_close(
        constant._pascal_factor, torch.ones(1, 1, dtype=torch.float64)
    )
    assert OffsetCode(raw)._pascal_factor is None


def test_decoder_uses_distinct_constant_bindings_for_dtypes() -> None:
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    words = _words(3, 2, 1, 2)
    decoder = _PascalDecoder(code, words)
    features = torch.randn(2, len(words), dtype=torch.float64)
    targets = torch.tensor([[1, 1], [0, 1]])
    expected = decoder.decode(features, targets)
    actual = decoder.decode(features.float(), targets)
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)
    assert len(decoder._bound) == 2


@pytest.mark.parametrize("shape", [(), (0,), (2, 0)])
def test_decoder_supports_vectors_and_empty_leading_axes(
    shape: tuple[int, ...],
) -> None:
    code = OffsetCode.pascal(2, 2, dtype=torch.float64)
    words = _words(2, 2, 1, 2)
    decoder = _PascalDecoder(code, words)
    features = torch.randn(*shape, len(words), dtype=torch.float64, requires_grad=True)
    first_targets = torch.tensor([[0, 0], [0, 1]])
    second_targets = torch.tensor([[1, 0], [0, 0]])
    actual = decoder.decode(features, first_targets)
    decoder.decode(features, second_targets)
    assert len(decoder._targets) == 1
    binding = (features.device, features.dtype)
    assert decoder._targets[binding][0] == ((1, 0), (0, 0))
    assert actual.shape == (*shape, 2)
    expected = features.index_select(-1, torch.tensor([2, 1]))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert features.grad is not None and features.grad.shape == features.shape


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("orthonormal", [False, True], ids=["raw", "qr"])
def test_low_precision_decoder_values_and_vjps(
    dtype: torch.dtype, orthonormal: bool
) -> None:
    code = OffsetCode.pascal(3, 2, orthonormal=orthonormal, dtype=dtype)
    words = _words(3, 2, 2, 2)
    decoder = _PascalDecoder(code, words)
    targets = torch.tensor([[3, 1], [1, 3], [3, 1]])
    increments = torch.tensor(
        [[[[0.25, -0.125], [0.125, 0.5], [-0.25, 0.125]]]],
        dtype=dtype,
        requires_grad=True,
    )
    lifted = torch.einsum("bnrc,rp->bnrcp", increments, code.matrix).flatten(-2)
    dense = signature_levels_from_increments(lifted, 2)[-1].flatten(2)
    features = dense.index_select(-1, _flat_indices(words, 4))
    actual = decoder.decode(features, targets)
    expected = dense.index_select(-1, _flat_indices(targets, 4))
    assert actual.dtype == dtype
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    actual_vjp = torch.autograd.grad(actual.sum(), increments, retain_graph=True)[0]
    expected_vjp = torch.autograd.grad(expected.sum(), increments)[0]
    torch.testing.assert_close(actual_vjp, expected_vjp, atol=tolerance, rtol=tolerance)
    decoder.decode(features.detach().float(), targets)
    assert tuple(decoder._bound) == ((torch.device("cpu"), torch.float32),)
