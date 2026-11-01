import pytest
import torch

from tensorsig.algebra.hall_bch import (
    HallBCH,
    SparseBCH,
    sparse_bch_supports_depth,
)
from tensorsig.classical.signatures import log_signature


@pytest.mark.parametrize("depth", [5, 6])
def test_high_depth_bch_matches_signature_then_log(depth: int) -> None:
    torch.manual_seed(depth)
    path = torch.randn(2, 6, 2, dtype=torch.float64)

    expected = log_signature(path, depth=depth, mode="hall")
    actual = log_signature(path, depth=depth, method="bch_sparse", mode="hall")

    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


def test_high_depth_bch_stream_and_gradients() -> None:
    torch.manual_seed(7)
    path = torch.randn(1, 5, 2, dtype=torch.float64, requires_grad=True)

    actual = log_signature(
        path,
        depth=5,
        stream=True,
        method="bch_sparse",
        mode="hall",
    )
    expected = log_signature(path, depth=5, stream=True, mode="hall")

    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    gradient = torch.autograd.grad(actual.square().sum(), path)[0]
    assert torch.isfinite(gradient).all()


def test_high_depth_bch_is_associative() -> None:
    runtime = HallBCH(2, 6, torch.device("cpu"), torch.float64)
    torch.manual_seed(8)
    x = torch.randn(3, runtime.dim, dtype=torch.float64)
    y = torch.randn(3, runtime.dim, dtype=torch.float64)
    z = torch.randn(3, runtime.dim, dtype=torch.float64)

    left = runtime.bch(runtime.bch(x, y), z)
    right = runtime.bch(x, runtime.bch(y, z))

    torch.testing.assert_close(left, right, atol=2e-8, rtol=2e-8)


def test_empty_bracket_has_no_artificial_depth_limit() -> None:
    runtime = SparseBCH(
        dim=2,
        depth=20,
        sparse_indices=torch.empty((0, 3), dtype=torch.long),
        sparse_values=torch.empty(0),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    x = torch.randn(2, 2, dtype=torch.float64)
    y = torch.randn(2, 2, dtype=torch.float64)

    assert sparse_bch_supports_depth(20)
    torch.testing.assert_close(runtime.bch(x, y), x + y)
