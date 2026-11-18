"""Batch Pascal windows by joining locally coded tile suffixes and prefixes."""

import torch
from torch import Tensor
from torch.nn.functional import pad

from ...algebra._bch_prefix import DegreeOneBCHPrefix
from ._compile import VariablePascalShift, _pascal_embeddings


class TiledLogSignature:
    """Device-bound evaluator for a certified Pascal quotient through depth three."""

    def __init__(self, plan, code: Tensor):
        self.window = code.shape[0]
        self.plan = plan
        self.prefixes = DegreeOneBCHPrefix(
            plan.basis_depths, plan.bracket_indices, plan.bracket_values, plan.max_depth
        )
        with torch.autocast(code.device.type, enabled=False):
            self.embeddings = _pascal_embeddings(
                code,
                plan.generator_map.index_select(-1, self.prefixes.generators),
                plan.alphabet.in_channels,
            )
        self.shift = VariablePascalShift(plan)

    def _evaluate(self, increments: Tensor, stride: int, count: int) -> Tensor:
        window = self.window
        if not self.plan.dimension or not increments.shape[0]:
            return increments[:, :1, :1].expand(-1, count, self.plan.dimension)
        length = (count - 1) * stride + window
        tiles = pad(increments[:, :length], (0, 0, 0, -length % window))
        tiles = tiles.unflatten(1, (-1, window))
        with torch.autocast(increments.device.type, enabled=False):
            lifted = torch.einsum("bnrc,rcd->bnrd", tiles, self.embeddings)

        starts = torch.arange(count, device=increments.device) * stride
        tile, offset = (
            starts.div(window, rounding_mode="floor"),
            starts.remainder(window),
        )

        # Reversed, negated increments give the inverse of each forward suffix.
        suffixes = -self.prefixes(-lifted.flip(-2)).flip(-2)
        left = self.shift(suffixes[:, tile, offset], -offset)

        prefixes = self.prefixes(lifted)
        next_tile = (tile + 1).clamp_max(tiles.shape[1] - 1)
        right = prefixes[:, next_tile, (offset - 1).clamp_min(0)]
        # Aligned windows use the entire left tile and the identity on the right.
        right = torch.where(offset[:, None] == 0, 0.0, right)
        right = self.shift(right, window - offset)
        return self.plan._runtime.bch(left, right)

    def __call__(
        self, increments: Tensor, stride: int, chunk_windows: int | None
    ) -> Tensor:
        count = 1 + (increments.shape[1] - self.window) // stride
        if chunk_windows is None or chunk_windows >= count:
            return self._evaluate(increments, stride, count)
        return torch.cat(
            [
                self._evaluate(
                    increments[:, start * stride :],
                    stride,
                    min(chunk_windows, count - start),
                )
                for start in range(0, count, chunk_windows)
            ],
            dim=1,
        )
