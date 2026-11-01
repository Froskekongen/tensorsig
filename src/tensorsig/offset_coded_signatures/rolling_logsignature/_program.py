"""Packed polynomial DAGs; constants and support decisions are made at setup."""

from fractions import Fraction

import torch

from ._bch import DegreeOneBCH


class BCHProgram:
    """Compile the packed typed endpoint DAG used by the fused FIFO route.

    Only the declared coordinate subspace of y is differentiated. Runtime uses
    packed supports and releases values after their last consumer. PyTorch's
    autograd tape may retain the inputs needed by backward independently.
    """

    def __init__(self, plan, *, endpoint_support=None):
        self.plan = plan
        device = plan.generator_map.device
        self.supports = [
            tuple(range(plan.dimension)),
            tuple(
                torch.where(plan.basis_depths == 1)[0].cpu().tolist()
                if endpoint_support is None
                else endpoint_support
            ),
        ]
        self.nodes = []
        self.cache = {}
        self.rows = [
            (row, c)
            for row, c in zip(
                plan.bracket_indices.cpu().tolist(),
                plan.bracket_coefficients,
                strict=True,
            )
        ]
        self.row_count = 0
        typed = DegreeOneBCH(plan)
        slots = [0, 1]
        terms = [(0, Fraction(1)), (1, Fraction(1))]
        for left, right, coefficient, _, _ in typed.nodes:
            node = self.bracket(slots[left], slots[right])
            slots.append(node)
            if coefficient:
                terms.append((node, Fraction(coefficient)))
        result = self.add(terms)
        self.result = result
        # Whole-node liveness. Coordinate supports are conservative; no claim
        # that this finds all linear dependencies inside an endpoint subspace.
        live = {result}
        for slot in range(len(self.supports) - 1, 1, -1):
            if slot in live:
                live.update(self.nodes[slot - 2][1])
        self.live = live
        self.last = {}
        for slot, (_, parents, _) in enumerate(self.nodes, 2):
            if slot in live:
                for parent in parents:
                    self.last[parent] = slot
        self.row_count = sum(
            len(data[0])
            for slot, (kind, _, data) in enumerate(self.nodes, 2)
            if slot in live and kind == "bracket"
        )
        self.activation_width = sum(len(self.supports[s]) for s in live)
        self.indices = [
            torch.tensor(s, device=device, dtype=torch.long) for s in self.supports
        ]
        # Compilation-only Python table and cache do not belong to a callable graph.
        del self.rows, self.cache

    def _node(self, key, kind, parents, support, data):
        if key in self.cache:
            return self.cache[key]
        slot = len(self.supports)
        self.supports.append(tuple(support))
        self.nodes.append((kind, tuple(parents), data))
        self.cache[key] = slot
        return slot

    def add(self, terms):
        merged = {}
        for slot, weight in terms:
            if self.supports[slot] and weight:
                merged[slot] = merged.get(slot, Fraction()) + weight
        merged = {s: w for s, w in merged.items() if w}
        if len(merged) == 1 and next(iter(merged.values())) == 1:
            return next(iter(merged))
        key = ("add", tuple(sorted(merged.items())))
        support = sorted({i for s in merged for i in self.supports[s]})
        positions = {i: j for j, i in enumerate(support)}
        data = [
            (
                torch.tensor(
                    [positions[i] for i in self.supports[s]],
                    device=self.plan.generator_map.device,
                    dtype=torch.long,
                ),
                float(w),
            )
            for s, w in sorted(merged.items())
        ]
        return self._node(key, "add", sorted(merged), support, data)

    def bracket(self, left, right):
        if left == right or not self.supports[left] or not self.supports[right]:
            return self.add([])
        key = ("bracket", left, right)
        if key in self.cache:
            return self.cache[key]
        ls = {i: j for j, i in enumerate(self.supports[left])}
        rs = {i: j for j, i in enumerate(self.supports[right])}
        rows = [(i, j, k, c) for (i, j, k), c in self.rows if i in ls and j in rs]
        support = sorted({k for _, _, k, _ in rows})
        out = {i: j for j, i in enumerate(support)}
        device = self.plan.generator_map.device
        indices = [
            torch.tensor(values, device=device, dtype=torch.long)
            for values in (
                [ls[i] for i, _, _, _ in rows],
                [rs[j] for _, j, _, _ in rows],
                [out[k] for _, _, k, _ in rows],
            )
        ]
        constants = torch.tensor(
            [float(c) for _, _, _, c in rows],
            device=device,
            dtype=self.plan.generator_map.dtype,
        )
        return self._node(key, "bracket", [left, right], support, (*indices, constants))

    def __call__(self, x, y):
        if not self.plan.dimension:
            return x + y
        values = {0: x[..., self.indices[0]], 1: y[..., self.indices[1]]}
        for slot, (kind, parents, data) in enumerate(self.nodes, 2):
            if slot not in self.live:
                continue
            if kind == "bracket":
                i, j, k, c = data
                contributions = (
                    values[parents[0]][..., i] * values[parents[1]][..., j] * c
                )
                value = x.new_zeros(
                    (*x.shape[:-1], len(self.supports[slot]))
                ).scatter_add(-1, k.expand_as(contributions), contributions)
            else:
                value = x.new_zeros((*x.shape[:-1], len(self.supports[slot])))
                for parent, (indices, coefficient) in zip(parents, data):
                    value = value.index_add(-1, indices, values[parent] * coefficient)
            values[slot] = value
            for parent in parents:
                if self.last[parent] == slot and parent != self.result:
                    values.pop(parent, None)
        return torch.zeros_like(x).index_copy(
            -1, self.indices[self.result], values[self.result]
        )
