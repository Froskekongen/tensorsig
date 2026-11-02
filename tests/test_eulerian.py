from fractions import Fraction

import torch

from tensorsig.algebra.eulerian import eulerian_permutations, eulerian_project_columns


def test_eulerian_projection_has_exact_coefficients_and_is_idempotent() -> None:
    assert eulerian_permutations(0) == ()
    assert eulerian_permutations(2) == (
        ((0, 1), Fraction(1, 2)),
        ((1, 0), Fraction(-1, 2)),
    )
    for depth in (1, 2, 3, 4):
        matrix = torch.randn(2**depth, 3, dtype=torch.float64, requires_grad=True)
        projected = eulerian_project_columns(matrix, width=2, depth=depth)
        torch.testing.assert_close(
            eulerian_project_columns(projected, width=2, depth=depth), projected
        )
        projected.square().sum().backward()
        assert torch.isfinite(matrix.grad).all()
