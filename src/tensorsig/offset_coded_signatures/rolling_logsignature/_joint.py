"""Coordinate-pruned polynomial programs with owned first/second derivatives.

Compilation prunes a complete FIFO/forcing expression, including transport.
Pure forward, VJP, and differentiated-VJP kernels may be compiled separately;
autograd never differentiates an AOTAutograd-generated compiled backward.
"""

from fractions import Fraction
from functools import lru_cache
from math import prod
from types import FunctionType, MethodType

import torch

from tensorsig.algebra.bch_coefficients import bch_coeffs


@lru_cache(None)
def _schedule(depth):
    if depth <= 4:
        return tuple(
            node
            for level, node in (
                (2, (0, 1, Fraction(1, 2))),
                (3, (0, 2, Fraction(1, 12))),
                (3, (1, 2, Fraction(-1, 12))),
                (4, (1, 3, Fraction(-1, 24))),
            )
            if level <= depth
        )
    coefficients = {}
    for word, coefficient in bch_coeffs(width=2, depth=depth).items():
        if len(word) < 2 or word[0] == word[1]:
            continue
        key = (1, 2, *word[2:])
        sign = 1 if word[:2] == (1, 2) else -1
        coefficients[key] = coefficients.get(key, 0) + sign * coefficient / len(word)
    coefficients = {word: value for word, value in coefficients.items() if value}
    words = sorted(
        {w[:n] for w in coefficients for n in range(2, len(w) + 1)},
        key=lambda w: (len(w), w),
    )
    slots = {w: i + 2 for i, w in enumerate(words)}
    return tuple(
        (
            0 if len(w) == 2 else slots[w[:-1]],
            w[-1] - 1,
            Fraction(coefficients.get(w, 0)),
        )
        for w in words
    )


class PolynomialProgram:
    """Small linear/bilinear DAG in the supplied fixed quotient coordinates."""

    def __init__(self, plan, supports):
        self.plan = plan
        self.input_count = len(supports)
        self.supports = [set(s) for s in supports]
        self.raw = []
        self.cache = {}
        self.brackets = tuple(
            zip(
                plan.bracket_indices.cpu().tolist(),
                plan.bracket_coefficients,
                strict=True,
            )
        )

    def _node(self, kind, rows, support):
        rows = tuple(rows)
        key = (kind, rows)
        if key not in self.cache:
            self.cache[key] = len(self.supports)
            self.supports.append(set(support))
            self.raw.append((kind, rows))
        return self.cache[key]

    def linear(self, terms):
        # A row is (parent, source coordinate, destination coordinate, weight).
        merged = {}
        for parent, source, target, weight in terms:
            if source in self.supports[parent] and weight:
                key = (parent, source, target)
                merged[key] = merged.get(key, Fraction()) + weight
        rows = tuple((*key, value) for key, value in sorted(merged.items()) if value)
        return self._node("linear", rows, {r[2] for r in rows})

    def add(self, terms):
        return self.linear(
            (parent, i, i, weight)
            for parent, weight in terms
            for i in sorted(self.supports[parent])
        )

    def bracket(self, left, right):
        rows = (
            []
            if left == right
            else [
                (left, i, right, j, k, coefficient)
                for (i, j, k), coefficient in self.brackets
                if i in self.supports[left] and j in self.supports[right]
            ]
        )
        return self._node("bracket", rows, {r[4] for r in rows})

    def bch(self, x, y, depth=None):
        slots, terms = [x, y], [(x, Fraction(1)), (y, Fraction(1))]
        for left, right, weight in _schedule(depth or self.plan.max_depth):
            node = self.bracket(slots[left], slots[right])
            slots.append(node)
            if weight:
                terms.append((node, weight))
        return self.add(terms)

    def transport(self, value, rows):
        return self.linear(
            (value, i, j, c) for i, row in enumerate(rows) for j, c in row.items()
        )

    def finish(self, result, outputs):
        wanted = [set() for _ in self.supports]
        wanted[result] = set(outputs) & self.supports[result]
        retained = {}
        for slot in range(len(self.supports) - 1, self.input_count - 1, -1):
            kind, rows = self.raw[slot - self.input_count]
            rows = [r for r in rows if r[-2] in wanted[slot]]
            retained[slot] = (kind, rows)
            for row in rows:
                wanted[row[0]].add(row[1])
                if kind == "bracket":
                    wanted[row[2]].add(row[3])
        self.output_coordinates = tuple(outputs)
        self.result = result
        self.widths = [len(s) for s in wanted]
        positions = [{v: i for i, v in enumerate(sorted(s))} for s in wanted]
        device, dtype = self.plan.generator_map.device, self.plan.generator_map.dtype
        index_values, coefficient_values = [], []

        def indices(values):
            start = len(index_values)
            index_values.extend(values)
            return slice(start, len(index_values))

        self.input_slices = [indices(sorted(s)) for s in wanted[: self.input_count]]
        self.nodes = []
        for slot, (kind, rows) in retained.items():
            if not rows:
                continue
            groups = {}
            for row in rows:
                parents = (row[0],) if kind == "linear" else (row[0], row[2])
                groups.setdefault(parents, []).append(row)
            packed = []
            for parents, group in groups.items():
                src = indices([positions[r[0]][r[1]] for r in group])
                dst = indices([positions[slot][r[-2]] for r in group])
                coeff = slice(
                    len(coefficient_values), len(coefficient_values) + len(group)
                )
                coefficient_values.extend(float(r[-1]) for r in group)
                other = (
                    indices([positions[r[2]][r[3]] for r in group])
                    if kind == "bracket"
                    else None
                )
                packed.append((parents, src, other, dst, coeff))
            self.nodes.append((slot, kind, tuple(packed)))
        self.nodes.reverse()
        self.result_source = indices(
            [positions[result][v] for v in outputs if v in wanted[result]]
        )
        self.result_target = indices(
            [i for i, v in enumerate(outputs) if v in wanted[result]]
        )
        # Slice inside the traced graph: prebuilt tensor views would become
        # separate input buffers and can exceed Metal's kernel argument limit.
        self.index_buffer = torch.tensor(index_values, device=device, dtype=torch.long)
        self.coefficient_buffer = torch.tensor(
            coefficient_values, device=device, dtype=dtype
        )
        self.row_count = len(coefficient_values)
        self.activation_width = sum(self.widths)
        self._forward = self.raw_forward
        self._vjp = self.raw_vjp
        self._second = self.raw_second
        del self.raw, self.cache, self.brackets, self.supports
        return self

    def compile(self):
        """Compile pure kernels. Their owned derivatives wrap compiled calls."""

        def bind(method):
            # Dynamo's variant budget belongs to a code object. Each fixed DAG
            # needs its own budget, rather than treating every grade as a recompile.
            source = method.__func__
            code = source.__code__.replace(co_name=f"{source.__name__}_{id(self)}")
            function = FunctionType(code, source.__globals__, code.co_name)
            kernel = torch.compile(
                MethodType(function, self), fullgraph=True, dynamic=True
            )

            def evaluate(*arguments):
                # These kernels run inside owned autograd Functions. Save the
                # original connected arguments there, but normalize pure-kernel
                # views to one batch axis to avoid a recompile per tree level.
                shape = arguments[0].shape[:-1]
                flat = tuple(
                    value.detach().reshape(prod(shape), value.shape[-1]).contiguous()
                    for value in arguments
                )
                result = kernel(*flat)
                if isinstance(result, tuple):
                    return tuple(
                        value.reshape(*shape, value.shape[-1]) for value in result
                    )
                return result.reshape(*shape, result.shape[-1])

            return evaluate

        self._forward = bind(self.raw_forward)
        self._vjp = bind(self.raw_vjp)
        self._second = bind(self.raw_second)
        return self

    def _values(self, inputs, tangents=None):
        template = inputs[0]
        values = [None] * len(self.widths)
        dots = [None] * len(self.widths) if tangents is not None else None
        for i, selection in enumerate(self.input_slices):
            ids = self.index_buffer[selection]
            values[i] = inputs[i][..., ids]
            if dots is not None:
                dots[i] = tangents[i][..., ids]
        for slot, kind, groups in self.nodes:
            value = template.new_zeros((*template.shape[:-1], self.widths[slot]))
            dot = torch.zeros_like(value) if dots is not None else None
            for parents, src, other, dst, c in groups:
                src, dst, c = (
                    self.index_buffer[src],
                    self.index_buffer[dst],
                    self.coefficient_buffer[c],
                )
                left = values[parents[0]][..., src]
                contribution = left * c
                if kind == "bracket":
                    other = self.index_buffer[other]
                    right = values[parents[1]][..., other]
                    contribution = contribution * right
                value = value.index_add(-1, dst, contribution)
                if dots is not None:
                    derivative = dots[parents[0]][..., src] * c
                    if kind == "bracket":
                        derivative = (
                            derivative * right + left * c * dots[parents[1]][..., other]
                        )
                    dot = dot.index_add(-1, dst, derivative)
            values[slot] = value
            if dots is not None:
                dots[slot] = dot
        if values[self.result] is None:
            values[self.result] = template[..., :0]
            if dots is not None:
                dots[self.result] = template[..., :0]
        return values, dots

    def _output(self, packed, template):
        return template.new_zeros(
            (*template.shape[:-1], len(self.output_coordinates))
        ).index_add(
            -1,
            self.index_buffer[self.result_target],
            packed[..., self.index_buffer[self.result_source]],
        )

    def raw_forward(self, *inputs):
        values, _ = self._values(inputs)
        return self._output(values[self.result], inputs[0])

    def _reverse(self, inputs, seed, tangents=None):
        values, dots = self._values(inputs, tangents)
        bars = [
            inputs[0].new_zeros((*inputs[0].shape[:-1], width)) for width in self.widths
        ]
        bars[self.result] = bars[self.result].index_add(
            -1,
            self.index_buffer[self.result_source],
            seed[..., self.index_buffer[self.result_target]],
        )
        dbars = [torch.zeros_like(v) for v in bars] if dots is not None else None
        for slot, kind, groups in reversed(self.nodes):
            for parents, src, other, dst, c in groups:
                src, dst, c = (
                    self.index_buffer[src],
                    self.index_buffer[dst],
                    self.coefficient_buffer[c],
                )
                b = bars[slot][..., dst] * c
                if kind == "linear":
                    bars[parents[0]] = bars[parents[0]].index_add(-1, src, b)
                    if dots is not None:
                        dbars[parents[0]] = dbars[parents[0]].index_add(
                            -1, src, dbars[slot][..., dst] * c
                        )
                else:
                    other = self.index_buffer[other]
                    left, right = (
                        values[parents[0]][..., src],
                        values[parents[1]][..., other],
                    )
                    bars[parents[0]] = bars[parents[0]].index_add(-1, src, b * right)
                    bars[parents[1]] = bars[parents[1]].index_add(-1, other, b * left)
                    if dots is not None:
                        db = dbars[slot][..., dst] * c
                        dbars[parents[0]] = dbars[parents[0]].index_add(
                            -1, src, db * right + b * dots[parents[1]][..., other]
                        )
                        dbars[parents[1]] = dbars[parents[1]].index_add(
                            -1, other, db * left + b * dots[parents[0]][..., src]
                        )
        selected = bars if dots is None else dbars
        result = tuple(
            torch.zeros_like(value).index_add(
                -1, self.index_buffer[selection], selected[i]
            )
            for i, (value, selection) in enumerate(
                zip(inputs, self.input_slices, strict=True)
            )
        )
        if dots is not None:
            return (*result, self._output(dots[self.result], inputs[0]))
        return result

    def raw_vjp(self, *arguments):
        return self._reverse(arguments[:-1], arguments[-1])

    def raw_second(self, *arguments):
        n = self.input_count
        return self._reverse(arguments[:n], arguments[n], arguments[n + 1 :])

    def __call__(self, *inputs):
        if not self.output_coordinates:
            return sum((value[..., :0] for value in inputs[1:]), inputs[0][..., :0])
        return _Polynomial.apply(self, *inputs)

    def vjp(self, inputs, seed):
        return _PolynomialVJP.apply(self, *inputs, seed)


class _Polynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, program, *inputs):
        ctx.program = program
        ctx.save_for_backward(*inputs)
        return program._forward(*inputs)

    @staticmethod
    def backward(ctx, seed):
        return (None, *ctx.program.vjp(ctx.saved_tensors, seed))


class _PolynomialVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, program, *arguments):
        ctx.program = program
        ctx.save_for_backward(*arguments)
        return program._vjp(*arguments)

    @staticmethod
    def backward(ctx, *directions):
        arguments = ctx.saved_tensors
        directions = tuple(
            torch.zeros_like(x) if d is None else d
            for x, d in zip(arguments[:-1], directions, strict=True)
        )
        return (None, *ctx.program._second(*arguments, *directions))


def joint_program(owner, grade=None, compiled=False):
    """Compile a FIFO polynomial from a bound ``RollingLogSignature`` context.

    The callable takes state, outgoing and incoming quotient vectors. With a
    grade, it returns that grade's forcing using only lower-grade state inputs.
    Compiled programs preserve their owned first- and second-derivative kernels.
    """
    plan = owner.plan
    degrees = plan.basis_depths.cpu().tolist()
    state = [i for i, d in enumerate(degrees) if grade is None or d < grade]
    endpoints = [
        torch.where(e.any(dim=0))[0].cpu().tolist()
        for e in (owner.embeddings[0], owner.embeddings[-1])
    ]
    program = PolynomialProgram(plan, [state, *endpoints])
    negative = program.add([(0, Fraction(-1))])
    removed = program.add([(program.bch(negative, 1, grade), Fraction(-1))])
    shifted = program.transport(removed, owner.transport_rows)
    result = program.bch(shifted, 2, grade)
    outputs = [i for i, d in enumerate(degrees) if grade is None or d == grade]
    program.finish(result, outputs)
    return program.compile() if compiled else program


def general_bch_program(plan, compiled=False, right_degree_one=False):
    """Compile BCH for two Lie values in the supplied quotient coordinates.

    ``right_degree_one`` restricts the right operand to generator coordinates.
    ``compiled`` compiles the polynomial and its owned derivative kernels.
    """
    support = range(plan.dimension)
    right = (
        torch.where(plan.basis_depths == 1)[0].cpu().tolist()
        if right_degree_one
        else support
    )
    program = PolynomialProgram(plan, [support, right])
    result = program.bch(0, 1)
    program.finish(result, range(plan.dimension))
    return program.compile() if compiled else program
