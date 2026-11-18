"""Compile Pascal transport in intrinsic coordinates using rational arithmetic."""

from fractions import Fraction
from functools import lru_cache
from math import comb

import torch

from .._lie_quotients.rules import (
    ChannelMultiplicity,
    ChannelSupport,
    DependenceGraph,
    ProjectedAlphabet,
    TemporalDegree,
)
from .._lie_quotients.trace_compiler import TraceBasisElement


def _validate_pascal_plan(plan, window):
    alphabet = plan.alphabet
    if not isinstance(alphabet, ProjectedAlphabet):
        raise ValueError("Rolling evaluation requires a ProjectedAlphabet.")
    if alphabet.temporal_degrees != tuple(range(alphabet.code_dim)):
        raise ValueError("Use ProjectedAlphabet.newton for raw Pascal codes.")
    if alphabet.code_dim > window:
        raise ValueError("code_dim must not exceed window.")
    allowed = (TemporalDegree, ChannelMultiplicity, ChannelSupport, DependenceGraph)
    if any(type(rule) not in allowed for rule in plan.quotient.rules):
        raise ValueError("Quotient rule has no certified Pascal-shift invariance.")


def _pascal_embeddings(code, generator_map, channels):
    return torch.einsum(
        "rp,cpd->rcd", code, generator_map.unflatten(0, (channels, code.shape[-1]))
    )


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


def _bind_sparse_rows(rows, like):
    triples = [(i, j, float(v)) for i, row in enumerate(rows) for j, v in row.items()]
    source = torch.tensor(
        [i for i, _, _ in triples], device=like.device, dtype=torch.long
    )
    target = torch.tensor(
        [j for _, j, _ in triples], device=like.device, dtype=torch.long
    )
    values = like.new_tensor([v for _, _, v in triples])
    return source, target, values


class VariablePascalShift:
    """Apply an integer Pascal shift per state using a single sparse correction map."""

    def __init__(self, plan):
        rows = pascal_transport(plan, 1)
        for i, row in enumerate(rows):
            row[i] = row.get(i, Fraction()) - 1
            if not row[i]:
                del row[i]
        self.source, self.target, self.values = _bind_sparse_rows(
            rows, plan.generator_map
        )
        # The correction raises temporal degree, which bounds its nilpotency.
        max_degree = (plan.alphabet.code_dim - 1) * plan.max_depth
        temporal_caps = [
            rule.maximum
            for rule in plan.quotient.rules
            if isinstance(rule, TemporalDegree)
        ]
        self.order = min(max_degree, *temporal_caps) if temporal_caps else max_degree
        if not self.source.numel():
            self.order = 0

    def __call__(self, state, steps):
        # Horner form of (I + N)^steps. This also covers negative integer steps.
        shifted = state
        steps = steps.to(dtype=torch.promote_types(state.dtype, torch.float32))
        for order in range(self.order, 0, -1):
            factor = ((steps - (order - 1)) / order).unsqueeze(-1)
            contribution = (shifted[..., self.source] * self.values * factor).to(
                state.dtype
            )
            shifted = state.scatter_add(
                -1, self.target.expand_as(contribution), contribution
            )
        return shifted


class Transport:
    """Two runtime realizations of the same exact grade-preserving linear map."""

    def __init__(self, plan, rows):
        device, dtype = plan.generator_map.device, plan.generator_map.dtype
        self.source, self.target, self.values = _bind_sparse_rows(
            rows, plan.generator_map
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
