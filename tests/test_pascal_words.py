import itertools
import math

import pytest
import torch

from tensorsig.offset_coded_signatures.chronological_words import (
    _admissible_words,
    _compile_chronological_level,
    _compile_chronological_levels,
)
from tensorsig.offset_coded_signatures.colored_descent import (
    cut_rank_target,
    generic_scalar_dimension,
    generic_scalar_dimensions,
    ribbon_dimension,
)
from tensorsig.offset_coded_signatures.coordinates import WordSelection
from tensorsig.offset_coded_signatures.increment_signature import (
    signature_levels_from_increment_windows,
)
from tensorsig.offset_coded_signatures.word_program import (
    _compile_word_program,
    append_word_state,
    evaluate_word_program,
    initialize_word_state,
)


@pytest.mark.parametrize(
    "window,code_dimension,channels,depth",
    [(1, 1, 2, 6), (3, 2, 2, 6), (4, 3, 3, 4), (5, 2, 1, 8)],
)
def test_count_dp_matches_independent_ribbon_formula(
    window, code_dimension, channels, depth
):
    expected = tuple(
        sum(
            cut_rank_target(window, code_dimension, level, cuts)
            * ribbon_dimension(channels, level, cuts)
            for count in range(level)
            for cuts in itertools.combinations(range(1, level), count)
        )
        for level in range(1, depth + 1)
    )
    assert (
        generic_scalar_dimensions(window, code_dimension, channels, depth) == expected
    )
    assert (
        generic_scalar_dimension(window, code_dimension, channels, depth)
        == expected[-1]
    )


def test_counts_use_unbounded_python_integers():
    assert generic_scalar_dimensions(100, 1, 100, 10)[-1] == 100**10


@pytest.mark.parametrize(
    "window,code_dimension,channels", [(1, 1, 3), (3, 2, 2), (3, 3, 2)]
)
def test_pruned_enumeration_preserves_dense_word_order(
    window, code_dimension, channels
):
    levels = _admissible_words(window, code_dimension, channels, 4)
    for depth, words in enumerate(levels, start=1):
        expected = tuple(
            word
            for word in itertools.product(
                range(code_dimension * channels), repeat=depth
            )
            if sum(letter % code_dimension for letter in word)
            + sum(
                left // code_dimension > right // code_dimension
                for left, right in zip(word, word[1:])
            )
            < window
        )
        assert words == expected


def test_compact_metadata_does_not_require_int64_ambient_word_indices():
    level = _compile_chronological_level(1, 1, 2, 64)
    assert level.feature_count == 65
    assert level.words[-1] == (1,) * 64
    assert level._flat_indices is None
    program = _compile_word_program(level.words, 1)
    assert program.state_count == 1 + sum(range(2, 66))
    actual = evaluate_word_program(
        torch.tensor([[0.5, 0.25]], dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.float64),
        program,
    )
    expected = torch.tensor(
        [
            math.prod(0.5 if letter == 0 else 0.25 for letter in word)
            / math.factorial(64)
            for word in level.words
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=1e-12)
    with pytest.raises(OverflowError, match="int64"):
        _ = level.flat_indices


def _dense_words(windows, code, words):
    leading = windows.shape[:-2]
    lifted = (windows[..., :, :, None] * code[:, None, :]).flatten(-2)
    width = lifted.shape[-1]
    levels = signature_levels_from_increment_windows(
        lifted.reshape(-1, 1, *lifted.shape[-2:]),
        max_depth=max(map(len, words)),
    )
    values = []
    for word in words:
        if not word:
            values.append(windows.new_ones(leading))
            continue
        index = sum(
            letter * width**power
            for letter, power in zip(word, reversed(range(len(word))))
        )
        values.append(
            levels[len(word) - 1].flatten(start_dim=2)[..., index].reshape(leading)
        )
    return torch.stack(values, dim=-1)


@pytest.mark.parametrize("leading", [(), (2,), (2, 1, 3)])
def test_shared_program_matches_dense_values_and_vjp_for_arbitrary_batch_shape(leading):
    torch.manual_seed(730)
    windows = torch.randn(*leading, 3, 2, dtype=torch.float64, requires_grad=True)
    code = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    words = ((3, 0, 1), (0,), (3, 0), (), (3, 0, 1), (2, 2))
    program = _compile_word_program(words, 2)
    actual = evaluate_word_program(windows, code, program)
    expected = _dense_words(windows, code, words)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    weight = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(
        (actual * weight).sum(), (windows, code), retain_graph=True
    )
    expected_grad = torch.autograd.grad((expected * weight).sum(), (windows, code))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value, atol=1e-11, rtol=1e-10)


def test_all_levels_share_one_prefix_state_and_keep_selected_output_order():
    levels = _compile_chronological_levels(3, 2, 2, (1, 2, 3))
    words = tuple(word for level in levels for word in level.words)
    program = _compile_word_program(words, 2)
    assert program.state_count == 1 + len(words)
    selected = _compile_chronological_levels(
        3, 2, 2, (1, 3), selection=WordSelection(descent_sets=((2,),))
    )
    assert not selected[0].words
    assert selected[1].words
    outputs = tuple(word for level in selected for word in level.words)
    selected_program = _compile_word_program(outputs, 2)
    assert all(
        word[:length] in selected_program.words
        for word in outputs
        for length in range(len(word))
    )


def test_rolling_closure_and_left_multiplication_match_dense_chen():
    words = ((3, 1, 2), (0, 3))
    program = _compile_word_program(words, 2, rolling=True)
    state_words = set(program.words)
    for word in program.words:
        assert all(
            word[start:stop] in state_words
            for start in range(len(word) + 1)
            for stop in range(start, len(word) + 1)
        )
        assert all(
            (*word[:index], letter - 1, *word[index + 1 :]) in state_words
            for index, letter in enumerate(word)
            if letter % 2
        )
    torch.manual_seed(741)
    increments = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    state = initialize_word_state(
        increments,
        torch.ones(3, 1, dtype=torch.float64),
        _compile_word_program(program.words, 1, rolling=True),
    )
    segment = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    for left in (False, True):
        actual_state = append_word_state(state, segment, program, left=left)
        expanded = torch.cat(
            (segment[:, None], increments) if left else (increments, segment[:, None]),
            dim=-2,
        )
        expected = _dense_words(expanded, torch.ones(4, 1, dtype=torch.float64), words)
        torch.testing.assert_close(
            program.gather(actual_state), expected, atol=1e-12, rtol=1e-11
        )


def test_empty_outputs_keep_zero_gradients_to_inputs_and_code():
    windows = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    code = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    program = _compile_word_program((), 2)
    result = evaluate_word_program(windows, code, program)
    assert result.shape == (2, 0)
    gradients = torch.autograd.grad(result.sum(), (windows, code))
    for value in gradients:
        torch.testing.assert_close(value, torch.zeros_like(value))


@pytest.mark.parametrize("leading", [(2,), (0,), (2, 0)])
@pytest.mark.parametrize("rolling", [False, True])
def test_finite_large_increments_do_not_overflow_zero_graph_connections(
    leading, rolling
):
    windows = torch.full(
        (*leading, 1, 4), 1e38, dtype=torch.float32, requires_grad=True
    )
    code = torch.ones(1, 1, dtype=torch.float32, requires_grad=True)
    words = tuple((letter,) for letter in range(4))
    program = _compile_word_program(words, 1, rolling=rolling)
    actual = evaluate_word_program(windows, code, program)
    expected = _dense_words(windows, code, words)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    weights = torch.full_like(actual, 0.125)
    actual_grad = torch.autograd.grad(
        actual, (windows, code), weights, retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected, (windows, code), weights)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        assert torch.isfinite(actual_value).all()
        torch.testing.assert_close(actual_value, expected_value)
