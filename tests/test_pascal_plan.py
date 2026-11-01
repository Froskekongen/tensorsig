"""Public integration contracts for compact Pascal signature plans."""

import pytest
import torch

from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.coordinates import WordSelection
from tensorsig.offset_coded_signatures.plans import LocalRealization


@pytest.mark.parametrize("orthonormal", [False, True])
@pytest.mark.parametrize("coordinates", ["chronological", "dense", "numerical"])
def test_requested_words_match_dense_values_and_vjps(orthonormal, coordinates):
    torch.manual_seed(381)
    code = OffsetCode.pascal(3, 2, orthonormal=orthonormal, dtype=torch.float64)
    if coordinates == "numerical":
        code = OffsetCode(code.matrix)
    realization = LocalRealization(code, channels=2, max_depth=3)
    plan = realization.signature(
        coordinates="dense" if coordinates == "dense" else "chronological"
    )
    dense = realization.signature(coordinates="dense")
    windows = torch.randn(2, 4, 3, 2, dtype=torch.float64, requires_grad=True)
    reference_windows = windows.detach().requires_grad_()
    words = torch.tensor([[3, 3, 2], [1, 3, 1], [0, 0, 0], [3, 3, 2]])
    actual = plan.reconstruct_words(plan.evaluate(windows), words)
    flat = words @ torch.tensor([16, 4, 1])
    expected = dense.evaluate(reference_windows)[
        ..., dense.level_slices[2]
    ].index_select(-1, flat)
    weights = torch.randn_like(actual)
    actual_vjp = torch.autograd.grad(actual, windows, weights)[0]
    expected_vjp = torch.autograd.grad(expected, reference_windows, weights)[0]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(actual_vjp, expected_vjp, atol=1e-10, rtol=1e-9)


def test_partial_chart_lookup_and_per_level_reconstruction():
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    realization = LocalRealization(code, channels=2, max_depth=3)
    plan = realization.signature(selection=WordSelection(max_complexity=1))
    features = torch.randn(
        2, plan.feature_count, dtype=torch.float64, requires_grad=True
    )
    words = plan.word_indices[-1][[1, 0, 1]]
    actual = plan.reconstruct_words(features, words)
    torch.testing.assert_close(
        actual, features[..., plan.level_slices[-1]][..., [1, 0, 1]]
    )
    with pytest.raises(ValueError, match="complete chart"):
        plan.reconstruct_words(features, torch.tensor([[3, 3, 2]]))
    with pytest.raises(ValueError, match="lossless"):
        plan.reconstruct(features)
    empty = plan.reconstruct_words(features, torch.empty((0, 3), dtype=torch.long))
    assert empty.shape == (2, 0)
    torch.testing.assert_close(
        torch.autograd.grad(empty.sum(), features)[0], torch.zeros_like(features)
    )


@pytest.mark.parametrize(
    "words, message",
    [
        (torch.tensor([[0.0, 1.0]]), "integer"),
        (torch.tensor([0, 1]), "integer"),
        (torch.tensor([[0, -1]]), "alphabet"),
        (torch.tensor([[0, 4]]), "alphabet"),
        (torch.tensor([[0]]), "depth"),
        (torch.empty((1, 0), dtype=torch.long), "depth"),
    ],
)
def test_requested_word_validation(words, message):
    plan = LocalRealization(OffsetCode.pascal(3, 2), 2, 3).signature(depths=(2, 3))
    with pytest.raises(ValueError, match=message):
        plan.reconstruct_words(torch.zeros(plan.feature_count), words)


def test_shared_program_counts_prefixes_once_and_high_depth_has_no_ambient_indices():
    plan = LocalRealization(OffsetCode.pascal(3, 2), 2, 3).signature()
    assert plan.level_dimensions == (4, 15, 44)
    assert plan.state_count == 64
    high = LocalRealization(
        OffsetCode.pascal(1, 1, orthonormal=False), 2, 64
    ).signature(depths=(64,))
    assert high.feature_count == 65
    assert high.state_count == sum(range(1, 66))
    features = torch.randn(2, 65, dtype=torch.float64)
    selected = high.word_indices[0][[0, 64, 0]]
    torch.testing.assert_close(
        high.reconstruct_words(features, selected), features[:, [0, 64, 0]]
    )
    omitted = torch.tensor([[1, 0] * 32])
    decoded = high.reconstruct_words(features, omitted)
    torch.testing.assert_close(decoded[:, 0], features[:, 32])
    with pytest.raises(OverflowError, match="int64"):
        high.reconstruct(features)


def test_structural_operations_do_not_use_numerical_decoder(monkeypatch):
    import tensorsig.offset_coded_signatures.plan as plans
    import tensorsig.offset_coded_signatures.chronological_words as words_module

    def forbidden(*args, **kwargs):
        raise AssertionError("ambient decoder construction")

    monkeypatch.setattr(plans, "_build_chronological_level", forbidden)
    monkeypatch.setattr(words_module, "_canonical_value_basis", forbidden)
    plan = LocalRealization(
        OffsetCode.pascal(3, 2, dtype=torch.float64), 2, 3
    ).signature()
    features = plan.evaluate(torch.ones(1, 1, 3, 2, dtype=torch.float64))
    assert not plan._decoder_cache
    decoded = plan.reconstruct_words(features, torch.tensor([[3, 3, 2]]))
    assert torch.isfinite(decoded).all()
