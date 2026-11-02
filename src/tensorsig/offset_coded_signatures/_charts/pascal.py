"""Triangular Pascal-coordinate reconstruction without an ambient value basis."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..codes import OffsetCode


class _PascalDecoder:
    """Decode one complete admissible level in the code's stored temporal basis.

    ``words`` and ``targets`` contain channel-major letter IDs, one word per
    row. The caller supplies the complete level chart; a lossy selection cannot
    supply the right-hand side of this triangular system. Only constants are
    cached, never features or solved feature-dependent coefficients.
    """

    def __init__(self, code: OffsetCode, words: Tensor) -> None:
        if code.family != "pascal" or code._pascal_factor is None:
            raise ValueError("Pascal decoding requires certified construction data.")
        if words.ndim != 2 or words.dtype != torch.long or words.shape[0] == 0:
            raise ValueError("words must be a nonempty integer word matrix.")
        self._dimension = code.dimension
        self._depth = int(words.shape[1])
        if self._depth == 0 or torch.any(words < 0):
            raise ValueError("words must contain nonnegative, nonempty words.")
        public_words = tuple(map(tuple, words.detach().cpu().tolist()))
        if len(set(public_words)) != len(public_words):
            raise ValueError("The chart must contain each word once.")
        self._width = max(max(word) for word in public_words) + 1
        temporal = [
            tuple(letter % code.dimension for letter in word) for word in public_words
        ]
        channels = [
            tuple(letter // code.dimension for letter in word) for word in public_words
        ]
        self._order = tuple(
            sorted(
                range(len(public_words)),
                key=lambda index: (
                    sum(temporal[index]),
                    temporal[index],
                    channels[index],
                ),
            )
        )
        self._columns = []
        for index in self._order:
            beta, channel_word = temporal[index], channels[index]
            descents = tuple(
                j
                for j in range(self._depth - 1)
                if channel_word[j] > channel_word[j + 1]
            )
            if sum(beta) + len(descents) >= code.window:
                raise ValueError("The chart contains an inadmissible Pascal word.")
            cuts = (0, *(j + 1 for j in descents), self._depth)
            runs = tuple(
                (start, stop, tuple(sorted(channel_word[start:stop])))
                for start, stop in zip(cuts, cuts[1:])
            )
            sigma = tuple(
                sum(beta[:j]) + sum(cut < j for cut in descents)
                for j in range(self._depth)
            )
            self._columns.append((beta, sigma, runs))

        self._raw = code._pascal_raw
        if self._raw:
            self._gauge = None
        else:
            self._gauge = torch.linalg.solve_triangular(
                code._pascal_factor,
                torch.eye(code.dimension, dtype=torch.float64),
                upper=True,
            ).tolist()
        # Cached scalar factors keep the O(N^2) decoder assembly independent of
        # repeated small triangular-basis calculations.
        self._factors: dict[tuple[int, int, int], int | float] = {}
        self._rows = self._entries(tuple(public_words[index] for index in self._order))
        self._bound: dict[tuple[torch.device, torch.dtype], tuple[Tensor, Tensor]] = {}
        self._targets: dict[
            tuple[torch.device, torch.dtype], tuple[tuple[tuple[int, ...], ...], Tensor]
        ] = {}

    def _factor(self, sigma: int, beta: int, alpha: int) -> int | float:
        key = (sigma, beta, alpha)
        if key not in self._factors:
            if self._gauge is None:
                lower = alpha - beta
                value = math.comb(sigma, lower) if 0 <= lower <= sigma else 0
            else:
                value = sum(
                    self._gauge[k][alpha] * math.comb(sigma, k - beta)
                    for k in range(beta, min(alpha, beta + sigma) + 1)
                )
            self._factors[key] = value
        return self._factors[key]

    def _entries(
        self, words: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int | float, ...], ...]:
        rows = []
        for word in words:
            alpha = tuple(letter % self._dimension for letter in word)
            channel_word = tuple(letter // self._dimension for letter in word)
            row = []
            for beta, sigma, runs in self._columns:
                if any(
                    tuple(sorted(channel_word[start:stop])) != multiset
                    for start, stop, multiset in runs
                ):
                    row.append(0)
                else:
                    row.append(
                        math.prod(
                            self._factor(s, b, a) for s, b, a in zip(sigma, beta, alpha)
                        )
                    )
            rows.append(tuple(row))
        return tuple(rows)

    def decode(self, features: Tensor, targets: Tensor) -> Tensor:
        """Return requested coordinates, preserving target order and duplicates."""
        if features.ndim < 1 or features.shape[-1] != len(self._order):
            raise ValueError("features must end in the complete chart dimension.")
        if not features.dtype.is_floating_point:
            raise ValueError("features must have a floating-point dtype.")
        if (
            targets.ndim != 2
            or targets.shape[1] != self._depth
            or targets.dtype != torch.long
            or torch.any(targets < 0)
            or torch.any(targets >= self._width)
        ):
            raise ValueError(
                "targets must contain valid channel-major words at this depth."
            )
        if targets.shape[0] == 0:
            return features[..., :0]
        compute_dtype = (
            torch.float32
            if features.dtype in (torch.float16, torch.bfloat16)
            else features.dtype
        )
        binding = (features.device, compute_dtype)
        if binding not in self._bound:
            self._bound[binding] = (
                torch.tensor(self._rows, device=features.device, dtype=compute_dtype),
                torch.tensor(self._order, device=features.device, dtype=torch.long),
            )
        matrix, order = self._bound[binding]
        words = tuple(map(tuple, targets.detach().cpu().tolist()))
        cached = self._targets.get(binding)
        if cached is None or cached[0] != words:
            self._targets[binding] = (
                words,
                torch.tensor(
                    self._entries(words), device=features.device, dtype=compute_dtype
                ),
            )
        # PyTorch triangular solves do not support half/bfloat16 on CPU; using
        # float32 constants and arithmetic also avoids rounding the QR diagonal.
        rhs = features.to(dtype=compute_dtype).index_select(-1, order)
        # Treat leading axes as multiple right-hand sides, avoiding a broadcast
        # copy of the triangular matrix for every batch/window element.
        coefficients = torch.linalg.solve_triangular(
            matrix,
            rhs.reshape(-1, len(self._order)).T,
            upper=False,
            unitriangular=self._raw,
        ).T.reshape(features.shape)
        return (coefficients @ self._targets[binding][1].T).to(dtype=features.dtype)


__all__: list[str] = []
