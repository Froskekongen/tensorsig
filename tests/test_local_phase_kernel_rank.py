import pytest
import torch

from tensorsig.offset_coded_signatures.diagnostics import (
    cumulative_offset_clock_matrix,
    finite_difference_contrast,
    invisible_offset_contrasts,
    no_phase_higher_order_kernel_matrix,
    no_phase_kernel_matrix,
    offset_retention_identity_matrix,
    phase_kernel_rank_summary,
    scalar_time_moment_matrix,
)
from tensorsig.offset_coded_signatures.offset_patterns import (
    raw_offset_weight_fraction,
)


def test_no_phase_matrix_has_rank_one_and_offset_retention_has_full_rank() -> None:
    no_phase = no_phase_kernel_matrix(5, dtype=torch.float64)
    retained = offset_retention_identity_matrix(5, dtype=torch.float64)

    assert phase_kernel_rank_summary(no_phase).rank == 1
    assert phase_kernel_rank_summary(retained).rank == 5
    assert invisible_offset_contrasts(no_phase).shape == (5, 4)
    assert invisible_offset_contrasts(retained).shape == (5, 0)


def test_no_phase_higher_order_kernel_keeps_only_nondecreasing_offsets() -> None:
    kernel_size = 3
    order = 3
    matrix = no_phase_higher_order_kernel_matrix(
        kernel_size,
        order,
        dtype=torch.float64,
    )

    assert matrix.shape == (kernel_size**order, 1)
    for row, offsets in enumerate(
        (
            tuple(values.tolist())
            for values in torch.cartesian_prod(*[torch.arange(kernel_size)] * order)
        )
    ):
        expected = float(raw_offset_weight_fraction(offsets))
        torch.testing.assert_close(
            matrix[row, 0],
            torch.tensor(expected, dtype=torch.float64),
        )

    torch.testing.assert_close(
        no_phase_higher_order_kernel_matrix(4, 1, dtype=torch.float64),
        no_phase_kernel_matrix(4, dtype=torch.float64),
    )


@pytest.mark.parametrize(("kernel_size", "max_depth"), [(5, 1), (5, 3), (3, 5)])
def test_scalar_time_moment_matrix_rank_is_min_r_m(
    kernel_size: int,
    max_depth: int,
) -> None:
    grid = torch.linspace(0.0, 1.0, kernel_size + 1, dtype=torch.float64)
    moments = scalar_time_moment_matrix(grid, max_depth)

    assert phase_kernel_rank_summary(moments).rank == min(kernel_size, max_depth)


def test_scalar_time_equal_grid_finite_difference_contrast_is_invisible() -> None:
    kernel_size = 6
    max_depth = 3
    grid = torch.arange(kernel_size + 1, dtype=torch.float64)
    moments = scalar_time_moment_matrix(grid, max_depth)
    contrast = finite_difference_contrast(max_depth, kernel_size, dtype=torch.float64)

    torch.testing.assert_close(
        contrast @ moments, torch.zeros(max_depth, dtype=torch.float64)
    )


def test_cumulative_offset_clock_matrix_is_full_rank_at_depth_two() -> None:
    matrix = cumulative_offset_clock_matrix(7, dtype=torch.float64)

    assert matrix.shape == (7, 7)
    assert phase_kernel_rank_summary(matrix).rank == 7
    torch.testing.assert_close(
        matrix[0],
        torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(matrix[-1], torch.ones(7, dtype=torch.float64))


def test_invisible_contrast_basis_is_orthogonal_to_visible_columns() -> None:
    grid = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    moments = scalar_time_moment_matrix(grid, max_depth=2)
    null_basis = invisible_offset_contrasts(moments)

    torch.testing.assert_close(
        null_basis.T @ moments, torch.zeros(3, 2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        null_basis.T @ null_basis,
        torch.eye(3, dtype=torch.float64),
        atol=1e-9,
        rtol=1e-7,
    )
