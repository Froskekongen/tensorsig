"""Degree-one BCH programs with structural support pruning at compile time."""

import torch

from tensorsig.algebra.hall_bch import _dynkin_bch_schedule


class DegreeOneBCH:
    """BCH(X, y) for y in grade one; never use for two aggregated states."""

    def __init__(self, plan):
        depth = plan.max_depth
        # Inputs are slots 0=X and 1=y; subsequent slots are bracket nodes.
        schedule = []
        if depth > 1:
            schedule = [(0, 1, 0.5)]
        if depth > 2:
            schedule += [(0, 2, 1 / 12), (1, 2, -1 / 12)]
        if depth > 3:
            schedule += [(1, 3, -1 / 24)]
        if depth > 4:
            schedule = [
                (0 if parent < 0 else parent + 2, 1 if parent < 0 else letter - 1, c)
                for parent, letter, c in _dynkin_bch_schedule(depth)
            ]
        supports = [
            set(range(plan.dimension)),
            set(torch.where(plan.basis_depths == 1)[0].cpu().tolist()),
        ]
        rows = plan.bracket_indices.cpu().tolist()
        self.nodes = []
        for left, right, coefficient in schedule:
            keep = [
                n
                for n, (i, j, _) in enumerate(rows)
                if i in supports[left] and j in supports[right]
            ]
            supports.append({rows[n][2] for n in keep})
            selected = torch.tensor(
                keep, device=plan.bracket_indices.device, dtype=torch.long
            )
            indices = plan.bracket_indices[selected]
            self.nodes.append(
                (left, right, coefficient, indices, plan.bracket_values[selected])
            )
        self.row_count = sum(len(node[3]) for node in self.nodes)

    def __call__(self, x, y):
        values = [x, y]
        result = x + y
        for left, right, coefficient, indices, constants in self.nodes:
            if len(indices):
                contributions = (
                    values[left][..., indices[:, 0]]
                    * values[right][..., indices[:, 1]]
                    * constants
                )
                value = torch.zeros_like(x).scatter_add(
                    -1, indices[:, 2].expand_as(contributions), contributions
                )
                if coefficient:
                    result = result + coefficient * value
            else:
                value = torch.zeros_like(x)
            values.append(value)
        return result
