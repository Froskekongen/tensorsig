"""Parallel BCH prefixes for degree-one inputs in a graded Lie algebra."""

import torch
from torch import Tensor
from torch.nn.functional import pad

from ..tensor_ops.products import _sparse_bilinear


class DegreeOneBCHPrefix:
    """Scan generator increments through degree three using graded brackets."""

    def __init__(
        self,
        basis_depths: Tensor,
        bracket_indices: Tensor,
        bracket_values: Tensor,
        depth: int,
    ):
        if not 1 <= depth <= 3:
            raise ValueError("Graded BCH prefixes support depths one through three.")
        self.depth = depth
        degrees = basis_depths.cpu()
        groups = [(degrees == k).nonzero().flatten() for k in range(1, depth + 1)]
        device = basis_depths.device
        self.generators = groups[0].to(device)
        self.basis_order = torch.cat(groups).argsort().to(device)

        # Bind compact bracket tables between the grades used by the recurrence.
        local = torch.zeros_like(degrees)
        for group in groups:
            local[group] = torch.arange(group.numel())
        indices = bracket_indices.cpu()
        row_degrees = degrees[indices]
        self._brackets = {}
        for a, b in ((1, 1), (2, 1), (1, 2)):
            if a + b <= depth:
                rows = (
                    ((row_degrees[:, 0] == a) & (row_degrees[:, 1] == b))
                    .nonzero()
                    .flatten()
                )
                self._brackets[a, b] = (
                    local[indices[rows]].to(device).unbind(dim=1),
                    bracket_values.index_select(0, rows.to(device)),
                    groups[a + b - 1].numel(),
                )

    def _bracket(self, x: Tensor, y: Tensor, a: int, b: int) -> Tensor:
        return _sparse_bilinear(x, y, *self._brackets[a, b])

    def __call__(self, generators: Tensor) -> Tensor:
        """Return inclusive prefixes for inputs shaped (..., steps, generators)."""

        def before(values):
            return pad(values[..., :-1, :], (0, 0, 1, 0))

        first = generators.cumsum(dim=-2)
        levels = [first]
        if self.depth >= 2:
            prefix_one = before(first)
            pair = self._bracket(prefix_one, generators, 1, 1)
            second = 0.5 * pair.cumsum(dim=-2)
            levels.append(second)
        if self.depth == 3:
            # BCH(L, u)[3] - L[3] with u homogeneous of degree one.
            third_delta = (
                0.5 * self._bracket(before(second), generators, 2, 1)
                + self._bracket(prefix_one - generators, pair, 1, 2) / 12
            )
            levels.append(third_delta.cumsum(dim=-2))
        return torch.cat(levels, dim=-1).index_select(-1, self.basis_order)
