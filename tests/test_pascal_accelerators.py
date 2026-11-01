"""Accelerator values and VJPs checked against ordinary CPU float64 signatures."""

import pytest
import torch

from tensorsig.classical.signatures import signature
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.coordinates import WordSelection
from tensorsig.offset_coded_signatures.layers import LocalFeatures
from tensorsig.offset_coded_signatures.plans import LocalRealization
from tensorsig.offset_coded_signatures.windows import increment_windows
from tensorsig.tensor_ops.series import unflatten_signature_levels


_ACCELERATORS = [
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is unavailable"
        ),
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS is unavailable"
        ),
    ),
]


def _gather_words(level, words, width):
    indices = torch.zeros(len(words), dtype=torch.long)
    for letters in words.T:
        indices = indices * width + letters.cpu()
    return level.flatten(2).index_select(-1, indices)


def _cpu_dense_levels(path, matrix, window, stride, depth):
    """Use ordinary path signatures, independently of the compact word runtime."""
    windows = increment_windows(path, window, stride)
    lifted = torch.einsum("bnrc,re->bnrce", windows, matrix).flatten(-2)
    batch, count, steps, width = lifted.shape
    segments = lifted.reshape(batch * count, steps, width)
    lifted_path = torch.cat(
        (segments.new_zeros(batch * count, 1, width), segments.cumsum(1)), dim=1
    )
    levels = unflatten_signature_levels(
        signature(lifted_path, depth=depth), width=width, depth=depth
    )
    return tuple(level.reshape(batch, count, *level.shape[1:]) for level in levels)


@pytest.mark.parametrize("device", _ACCELERATORS)
@pytest.mark.parametrize("orthonormal", [False, True], ids=["raw", "qr"])
@pytest.mark.parametrize("selected", [False, True], ids=["full", "selected"])
def test_pascal_accelerator_against_cpu_float64(device, orthonormal, selected):
    generator = torch.Generator().manual_seed(421)
    window, dimension, channels, depth, stride, count = 6, 3, 2, 3, 2, 8
    code = OffsetCode.pascal(
        window, dimension, orthonormal=orthonormal, device=device, dtype=torch.float32
    )
    selection = (
        WordSelection(descent_sets=((1,),), max_temporal_degree=3) if selected else None
    )
    plan = LocalRealization(code, channels, depth).signature(
        selection=selection, depths=(2, 3) if selected else None
    )
    increments = (
        torch.randn(
            2,
            window + (count - 1) * stride,
            channels,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.01
    )
    path = torch.cat((increments.new_zeros(2, 1, channels), increments.cumsum(1)), 1)
    runtime_path = path.to(device)
    reference_path = runtime_path.detach().cpu().double().requires_grad_()
    # Promote the values actually used on device, not a separately rebuilt QR code.
    runtime_matrix = code.matrix.detach().cpu().double()
    levels = _cpu_dense_levels(reference_path, runtime_matrix, window, stride, depth)
    width = channels * dimension
    reference = torch.cat(
        [
            _gather_words(levels[level - 1], words, width)
            for level, words in zip(plan.depths, plan.word_indices, strict=True)
        ],
        -1,
    )
    cotangent = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    reference_vjp = torch.autograd.grad(
        reference, reference_path, cotangent.double(), retain_graph=not selected
    )[0]
    requested = torch.tensor([[2, 2, 2], [5, 2, 5], [0, 0, 0], [2, 2, 2]])
    if not selected:
        decoded_reference = _gather_words(levels[depth - 1], requested, width)
        decoded_cotangent = torch.randn(
            decoded_reference.shape, generator=generator, dtype=torch.float32
        )
        decoded_reference_vjp = torch.autograd.grad(
            decoded_reference, reference_path, decoded_cotangent.double()
        )[0]

    for backend in ("vectorized", "rolling"):
        candidate_path = runtime_path.detach().clone().requires_grad_()
        layer = LocalFeatures(plan, stride=stride, backend=backend)
        actual = layer(candidate_path)
        actual_vjp = torch.autograd.grad(
            actual, candidate_path, cotangent.to(device), retain_graph=not selected
        )[0]
        assert actual.dtype == torch.float32 and actual.device.type == device
        torch.testing.assert_close(
            actual.detach().cpu().double(), reference.detach(), atol=3e-6, rtol=3e-4
        )
        torch.testing.assert_close(
            actual_vjp.cpu().double(), reference_vjp, atol=5e-5, rtol=5e-4
        )
        if not selected:
            decoded = plan.reconstruct_words(actual, requested.to(device))
            decoded_vjp = torch.autograd.grad(
                decoded, candidate_path, decoded_cotangent.to(device)
            )[0]
            assert decoded.dtype == actual.dtype and decoded.device == actual.device
            torch.testing.assert_close(
                decoded.detach().cpu().double(),
                decoded_reference.detach(),
                atol=3e-6,
                rtol=3e-4,
            )
            torch.testing.assert_close(
                decoded_vjp.cpu().double(),
                decoded_reference_vjp,
                atol=5e-5,
                rtol=5e-4,
            )


@pytest.mark.parametrize("device", _ACCELERATORS)
@pytest.mark.parametrize("orthonormal", [False, True], ids=["raw", "qr"])
def test_dense_rolling_accelerator_shift_solve_against_cpu_float64(device, orthonormal):
    generator = torch.Generator().manual_seed(422)
    window, dimension, channels, depth, stride, count = 6, 3, 2, 3, 2, 8
    pascal = OffsetCode.pascal(
        window, dimension, orthonormal=orthonormal, device=device, dtype=torch.float32
    )
    # Deliberately remove Pascal provenance to exercise the generic CPU shift solve.
    code = OffsetCode(pascal.matrix)
    plan = LocalRealization(code, channels, depth).signature(coordinates="dense")
    increments = (
        torch.randn(
            2,
            window + (count - 1) * stride,
            channels,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.01
    )
    path = torch.cat((increments.new_zeros(2, 1, channels), increments.cumsum(1)), 1)
    candidate_path = path.to(device).requires_grad_()
    reference_path = candidate_path.detach().cpu().double().requires_grad_()
    reference = torch.cat(
        [
            level.flatten(2)
            for level in _cpu_dense_levels(
                reference_path,
                code.matrix.detach().cpu().double(),
                window,
                stride,
                depth,
            )
        ],
        dim=-1,
    )
    cotangent = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    reference_vjp = torch.autograd.grad(reference, reference_path, cotangent.double())[
        0
    ]
    actual = LocalFeatures(plan, stride=stride, backend="rolling")(candidate_path)
    actual_vjp = torch.autograd.grad(actual, candidate_path, cotangent.to(device))[0]
    assert actual.dtype == torch.float32 and actual.device.type == device
    torch.testing.assert_close(
        actual.detach().cpu().double(), reference.detach(), atol=3e-6, rtol=3e-4
    )
    torch.testing.assert_close(
        actual_vjp.cpu().double(), reference_vjp, atol=5e-5, rtol=5e-4
    )
