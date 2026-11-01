import torch

from tensorsig.algebra.hall_bch import SparseBCH


def test_sparse_bch_reuses_pre_split_coo_columns() -> None:
    indices = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [1, 0, 2],
            [1, 0, 3],
        ],
        dtype=torch.long,
    )
    values = torch.tensor([2.0, -1.0, -2.0, 1.0], dtype=torch.float64)
    runtime = SparseBCH(
        dim=4,
        depth=2,
        sparse_indices=indices,
        sparse_values=values,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    torch.testing.assert_close(runtime._left_idx, indices[:, 0])
    torch.testing.assert_close(runtime._right_idx, indices[:, 1])
    torch.testing.assert_close(runtime._output_idx, indices[:, 2])
    assert runtime._left_idx.is_contiguous()
    assert runtime._right_idx.is_contiguous()
    assert runtime._output_idx.is_contiguous()

    left = torch.tensor([[2.0, 3.0, 0.0, 0.0]], dtype=torch.float64)
    right = torch.tensor([[5.0, 7.0, 0.0, 0.0]], dtype=torch.float64)
    expected = torch.tensor([[0.0, 0.0, -2.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(runtime.bracket(left, right), expected)
