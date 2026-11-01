import pytest
import torch

from tensorsig.algebra.hall_bch import (
    HallBCH,
    SparseBCH,
    _sparse_constants,
)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_sparse_bch_matches_hall_bch(depth: int) -> None:
    indices, values = _sparse_constants(width=2, depth=depth)
    hall = HallBCH(2, depth, torch.device("cpu"), torch.float64)
    generic = SparseBCH(
        dim=hall.dim,
        depth=depth,
        sparse_indices=indices,
        sparse_values=values,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    torch.manual_seed(depth)
    x = torch.randn(3, hall.dim, dtype=torch.float64)
    y = torch.randn(3, hall.dim, dtype=torch.float64)

    torch.testing.assert_close(generic.bracket(x, y), hall.bracket_sparse(x, y))
    torch.testing.assert_close(generic.bch(x, y), hall.bch(x, y))


def test_sparse_bch_accepts_mixed_structure_constants() -> None:
    # [e0, e1] = e2 + 2 e3, with the antisymmetric rows supplied explicitly.
    runtime = SparseBCH(
        dim=4,
        depth=2,
        sparse_indices=torch.tensor([[0, 1, 2], [0, 1, 3], [1, 0, 2], [1, 0, 3]]),
        sparse_values=torch.tensor([1.0, 2.0, -1.0, -2.0]),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    x = torch.tensor([[2.0, -1.0, 0.3, -0.2]], dtype=torch.float64)
    y = torch.tensor([[3.0, 4.0, -0.1, 0.5]], dtype=torch.float64)
    determinant = x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
    expected_bracket = torch.zeros_like(x)
    expected_bracket[:, 2] = determinant
    expected_bracket[:, 3] = 2.0 * determinant

    torch.testing.assert_close(runtime.bracket(x, y), expected_bracket)
    torch.testing.assert_close(runtime.bch(x, y), x + y + 0.5 * expected_bracket)


def test_sparse_bch_gradcheck_through_depth_four() -> None:
    indices, values = _sparse_constants(width=2, depth=4)
    runtime = SparseBCH(
        dim=8,
        depth=4,
        sparse_indices=indices,
        sparse_values=values,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    torch.manual_seed(5)
    x = torch.randn(1, runtime.dim, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, runtime.dim, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(runtime.bch, (x, y))


def test_empty_sparse_bracket_keeps_autograd_connections() -> None:
    runtime = SparseBCH(
        dim=2,
        depth=2,
        sparse_indices=torch.empty((0, 3), dtype=torch.long),
        sparse_values=torch.empty(0),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    x = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)

    runtime.bracket(x, y).sum().backward()

    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    torch.testing.assert_close(y.grad, torch.zeros_like(y))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"depth": 0}, "depth must be positive"),
        ({"dtype": torch.int64}, "dtype must be floating-point"),
        (
            {"sparse_indices": torch.tensor([[0.0, 1.0, 2.0]])},
            "integer dtype",
        ),
        ({"sparse_values": torch.tensor([float("nan")])}, "must be finite"),
    ],
)
def test_sparse_bch_rejects_invalid_runtime_tables(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "dim": 3,
        "depth": 2,
        "sparse_indices": torch.tensor([[0, 1, 2]]),
        "sparse_values": torch.tensor([1.0]),
        "device": torch.device("cpu"),
        "dtype": torch.float64,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        SparseBCH(**arguments)  # type: ignore[arg-type]


def test_sparse_bch_validates_state_shape_and_dtype() -> None:
    runtime = SparseBCH(
        dim=2,
        depth=1,
        sparse_indices=torch.empty((0, 3), dtype=torch.long),
        sparse_values=torch.empty(0),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="trailing dimension 2"):
        runtime.bch(torch.zeros(1, 3), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="dtype torch.float64"):
        runtime.bracket(torch.zeros(1, 2), torch.zeros(1, 2))
