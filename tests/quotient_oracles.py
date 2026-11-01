"""Test-local ambient oracles for the private exact quotient compiler."""

from __future__ import annotations

import torch
from torch import Tensor

from tensorsig.algebra.hall_bch import SparseBCH
from tensorsig.algebra.hall import hall_basis
from tensorsig.algebra.hall_projection import HallBasisElement
from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    LieQuotientPlan,
)


def ambient_quotient_matrix(plan: LieQuotientPlan) -> Tensor:
    """Evaluate every ambient Hall tree recursively in ``plan``."""
    basis = tuple(hall_basis(plan.alphabet.width, plan.max_depth))
    runtime = SparseBCH(
        dim=plan.dimension,
        depth=plan.max_depth,
        sparse_indices=plan.bracket_indices,
        sparse_values=plan.bracket_values,
        device=plan.generator_map.device,
        dtype=plan.generator_map.dtype,
    )
    values: dict[HallBasisElement, Tensor] = {
        letter: plan.generator_map[letter - 1]
        for letter in range(1, plan.alphabet.width + 1)
    }
    for element in basis[plan.alphabet.width :]:
        left, right = element
        values[element] = runtime.bracket(values[left], values[right])
    return torch.stack([values[element] for element in basis], dim=1)


def project_ambient(coordinates: Tensor, plan: LieQuotientPlan) -> Tensor:
    """Project ambient Hall coordinates through recursive tree evaluation."""
    matrix = ambient_quotient_matrix(plan).to(
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    return coordinates @ matrix.T


def ambient_section_matrix(plan: LieQuotientPlan) -> Tensor:
    """Return a test-only linear section of the recursive ambient projection."""
    return torch.linalg.pinv(ambient_quotient_matrix(plan))
