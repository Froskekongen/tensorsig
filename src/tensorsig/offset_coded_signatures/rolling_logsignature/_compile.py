"""Compile Pascal transport in intrinsic coordinates using rational arithmetic."""

from fractions import Fraction
from functools import lru_cache
from math import comb

import torch

from .._lie_quotients.trace_compiler import TraceBasisElement


def pascal_transport(plan, shift=-1):
    """Return rows T(e_i) without building an ambient Hall or tensor basis.

    Hall trees are evaluated in the quotient. For trace basis representatives,
    the degree-normalized left Dynkin map reconstructs the primitive before
    substituting generators. This is valid modulo the homogeneous Hopf ideal.
    The caller must establish invariance of that ideal under Pascal translation.
    """
    width = plan.alphabet.code_dim
    table = {}
    for (i, j, k), coefficient in zip(
        plan.bracket_indices.cpu().tolist(), plan.bracket_coefficients, strict=True
    ):
        table.setdefault((i, j), []).append((k, coefficient))

    def add(target, source, scale=Fraction(1)):
        for k, value in source.items():
            target[k] = target.get(k, Fraction()) + scale * value
            if not target[k]:
                del target[k]

    def bracket(left, right):
        result = {}
        for i, a in left.items():
            for j, b in right.items():
                for k, coefficient in table.get((i, j), ()):
                    add(result, {k: coefficient}, a * b)
        return result

    generators = [
        {i: Fraction(float(v)) for i, v in enumerate(row) if v}
        for row in plan.generator_map.cpu().tolist()
    ]
    shifted = []
    for letter in range(plan.alphabet.width):
        channel, degree = divmod(letter, width)
        result = {}
        for p in range(degree, width):
            k = p - degree
            coefficient = (
                comb(shift, k) if shift >= 0 else (-1) ** k * comb(k - shift - 1, k)
            )
            add(result, generators[channel * width + p], Fraction(coefficient))
        shifted.append(result)

    @lru_cache(None)
    def hall(element):
        if isinstance(element, int):
            return shifted[element - 1]
        return bracket(hall(element[0]), hall(element[1]))

    @lru_cache(None)
    def dynkin(word):
        if len(word) == 1:
            return shifted[word[0]]
        return bracket(dynkin(word[:-1]), shifted[word[-1]])

    rows = []
    for element in plan.basis:
        if isinstance(element, TraceBasisElement):
            row = {}
            for word, coefficient in element.terms:
                add(row, dynkin(word), coefficient / element.depth)
            rows.append(row)
        else:
            rows.append(hall(element))
    return rows


class Transport:
    """Two runtime realizations of the same exact grade-preserving linear map."""

    def __init__(self, plan, rows):
        device, dtype = plan.generator_map.device, plan.generator_map.dtype
        triples = [
            (i, j, float(v)) for i, row in enumerate(rows) for j, v in row.items()
        ]
        self.source = torch.tensor([i for i, _, _ in triples], device=device)
        self.target = torch.tensor([j for _, j, _ in triples], device=device)
        self.source = self.source.long()
        self.target = self.target.long()
        self.values = torch.tensor(
            [v for _, _, v in triples], device=device, dtype=dtype
        )
        self.blocks = []
        for grade in range(1, plan.max_depth + 1):
            indices = torch.where(plan.basis_depths == grade)[0]
            positions = {i: j for j, i in enumerate(indices.cpu().tolist())}
            matrix = torch.zeros(len(indices), len(indices), device=device, dtype=dtype)
            for i, local_i in positions.items():
                for j, coefficient in rows[i].items():
                    matrix[local_i, positions[j]] = float(coefficient)
            if len(indices):
                self.blocks.append((indices, matrix))

    def __call__(self, state, backend):
        if backend == "sparse":
            contributions = state[..., self.source] * self.values
            return torch.zeros_like(state).scatter_add(
                -1, self.target.expand_as(contributions), contributions
            )
        result = torch.zeros_like(state)
        for indices, matrix in self.blocks:
            result = result.index_copy(-1, indices, state[..., indices] @ matrix)
        return result
