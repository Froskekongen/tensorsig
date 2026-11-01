"""Independent dense-window and algebra checks for compact Pascal rolling."""

import pytest
import torch

from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.compact_sliding import _compile_pascal_transport
from tensorsig.offset_coded_signatures.coordinates import WordSelection
from tensorsig.offset_coded_signatures.increment_signature import (
    signature_levels_from_increment_windows,
)
from tensorsig.offset_coded_signatures.layers import LocalFeatures
from tensorsig.offset_coded_signatures.plans import LocalRealization
from tensorsig.offset_coded_signatures.windows import increment_windows
from tensorsig.tensor_ops.linear import apply_linear_map_to_signature_levels


def _dense_reference(path, matrix, plan, stride):
    windows = increment_windows(path, plan.window, stride)
    lifted = torch.einsum("bnrc,re->bnrce", windows, matrix).flatten(-2)
    levels = signature_levels_from_increment_windows(lifted, max(plan.depths))
    width = plan.output_dimension
    outputs = []
    for depth, words in zip(plan.depths, plan.word_indices, strict=True):
        indices = torch.tensor(
            [
                sum(letter * width ** (depth - i - 1) for i, letter in enumerate(word))
                for word in words.tolist()
            ],
            dtype=torch.long,
            device=path.device,
        )
        outputs.append(levels[depth - 1].flatten(2).index_select(-1, indices))
    return torch.cat(outputs, dim=-1)


@pytest.mark.parametrize("orthonormal", [False, True])
@pytest.mark.parametrize(
    "window,dimension,stride,count",
    [
        (5, 3, 1, 12),
        (5, 3, 2, 7),
        (5, 3, 3, 6),
        (4, 4, 1, 9),
        (3, 2, 4, 5),
        (1, 1, 1, 5),
        (5, 3, 1, 1),
        (5, 3, 1, 3),
    ],
)
def test_refresh_blocks_and_strides_match_independent_dense_windows(
    orthonormal, window, dimension, stride, count
):
    torch.manual_seed(314)
    code = OffsetCode.pascal(
        window, dimension, orthonormal=orthonormal, dtype=torch.float64
    )
    plan = LocalRealization(code, 2, 3).signature()
    path = (
        torch.randn(2, window + (count - 1) * stride + 1, 2, dtype=torch.float64) * 0.03
    )
    actual = LocalFeatures(plan, stride=stride, backend="rolling")(path)
    expected = _dense_reference(path, code.matrix, plan, stride)
    assert actual.shape[1] == count
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("orthonormal", [False, True])
@pytest.mark.parametrize(
    "selection",
    [
        WordSelection(max_temporal_degree=2),
        WordSelection(max_channel_descents=0),
        WordSelection(max_complexity=2),
        WordSelection(descent_sets=((1,),)),
        WordSelection(descent_sets=()),
    ],
)
def test_selected_depths_and_hidden_factors_preserve_values_and_vjps(
    orthonormal, selection
):
    torch.manual_seed(315)
    code = OffsetCode.pascal(5, 3, orthonormal=orthonormal, dtype=torch.float64)
    plan = LocalRealization(code, 2, 3).signature(selection=selection, depths=(2, 3))
    actual_path = (torch.randn(2, 18, 2, dtype=torch.float64) * 0.02).requires_grad_()
    expected_path = actual_path.detach().clone().requires_grad_()
    actual = LocalFeatures(plan, stride=2, backend="rolling")(actual_path)
    expected = _dense_reference(expected_path, code.matrix, plan, 2)
    weight = torch.randn_like(actual)
    actual_vjp = torch.autograd.grad((actual * weight).sum(), actual_path)[0]
    expected_vjp = torch.autograd.grad((expected * weight).sum(), expected_path)[0]
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
    torch.testing.assert_close(actual_vjp, expected_vjp, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("orthonormal", [False, True])
def test_triangular_transport_on_arbitrary_states_and_inverse(orthonormal):
    torch.manual_seed(316)
    code = OffsetCode.pascal(5, 3, orthonormal=orthonormal, dtype=torch.float64)
    plan = LocalRealization(code, 2, 3).signature(
        selection=WordSelection(descent_sets=((1,),)), depths=(3,)
    )
    program = plan._rolling_program()
    state = torch.randn(2, program.state_count, dtype=torch.float64)
    backward = _compile_pascal_transport(code, program.words, state.device, state.dtype)
    forward = _compile_pascal_transport(
        code, program.words, state.device, state.dtype, 1
    )
    actual = backward(state)
    torch.testing.assert_close(forward(actual), state, atol=2e-11, rtol=2e-10)

    width = plan.output_dimension
    dense = [torch.zeros(2, width**depth, dtype=state.dtype) for depth in range(1, 4)]
    for index, word in enumerate(program.words[1:], start=1):
        flat = sum(
            letter * width ** (len(word) - i - 1) for i, letter in enumerate(word)
        )
        dense[len(word) - 1][:, flat] = state[:, index]
    dense = [
        level.reshape(2, *([width] * depth)) for depth, level in enumerate(dense, 1)
    ]
    shift = torch.linalg.lstsq(code.matrix[1:], code.matrix[:-1]).solution
    shifted = apply_linear_map_to_signature_levels(
        dense,
        torch.kron(torch.eye(2, dtype=state.dtype), shift.contiguous()),
        shared_dims=1,
    )
    expected = [state[:, :1]]
    for depth in range(1, 4):
        indices = [
            sum(letter * width ** (depth - i - 1) for i, letter in enumerate(word))
            for word in program.words
            if len(word) == depth
        ]
        expected.append(shifted[depth - 1].flatten(1)[:, indices])
    torch.testing.assert_close(actual, torch.cat(expected, -1), atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("orthonormal", [False, True])
def test_module_dtype_moves_keep_chosen_basis(orthonormal):
    torch.manual_seed(317)
    code = OffsetCode.pascal(8, 3, orthonormal=orthonormal, dtype=torch.float64)
    plan = LocalRealization(code, 2, 3).signature(depths=(1, 3))
    layer = LocalFeatures(plan, backend="rolling").to(dtype=torch.float32)
    path = (torch.randn(2, 27, 2) * 0.01).requires_grad_()
    actual = layer(path)
    expected = _dense_reference(path, layer.matrix, plan, 1)
    assert actual.dtype == torch.float32
    weight = torch.randn_like(actual)
    actual_vjp = torch.autograd.grad((actual * weight).sum(), path, retain_graph=True)[
        0
    ]
    expected_vjp = torch.autograd.grad((expected * weight).sum(), path)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_vjp, expected_vjp, atol=2e-5, rtol=2e-4)
    moved_code = code.to(dtype=torch.float32).to(dtype=torch.float64)
    moved_plan = LocalRealization(moved_code, 2, 3).signature(depths=(1, 3))
    moved_layer = LocalFeatures(moved_plan, backend="rolling")
    double_path = path.detach().double()
    torch.testing.assert_close(
        moved_layer(double_path),
        _dense_reference(double_path, moved_layer.matrix, moved_plan, 1),
        atol=2e-8,
        rtol=2e-6,
    )
