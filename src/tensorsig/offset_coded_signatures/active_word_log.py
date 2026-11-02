"""Selected tensor logarithms on exact bridge support and its factor closure.

This route reuses the sparse Chen state, polynomial cut recurrence P3--P4,
and the quotient's exact Lyndon bridge. No floating threshold chooses support.
Ordinary PyTorch operations preserve higher derivatives, including at zeros.
Compilation still enumerates ambient Lyndon words for the bridge; this is a
runtime state reduction, not a claim of ambient-free compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ._lie_quotients.compiler import LieQuotientPlan
from ._lie_quotients.rules import OffsetAlphabet, ProjectedAlphabet
from ._word_program import (
    _WordLogReadout,
    _WordProgram,
    _compile_word_log_readout,
    _compile_word_program,
    append_word_state,
    initialize_word_state,
)

__all__ = [
    "ActiveWordLogPlan",
    "compile_active_word_log",
    "active_word_log_signature",
]


@dataclass(frozen=True, eq=False)
class ActiveWordLogPlan:
    """One factor state and its fixed log/quotient readout."""

    program: _WordProgram
    log_readout: _WordLogReadout
    quotient_matrix: Tensor
    requested_words: tuple[tuple[int, ...], ...]
    _bound: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def state_count(self) -> int:
        """Number of factor-closed signature coordinates, including the scalar."""
        return self.program.state_count

    def to(self, device: torch.device, dtype: torch.dtype) -> ActiveWordLogPlan:
        """Bind the immutable state/readout coefficients, reusing prior bindings."""
        device = torch.device(device)
        if (
            device == self.quotient_matrix.device
            and dtype == self.quotient_matrix.dtype
        ):
            return self
        key = (device, dtype)
        if key not in self._bound:
            self._bound[key] = ActiveWordLogPlan(
                self.program.to(device, dtype),
                self.log_readout.to(device, dtype),
                self.quotient_matrix.to(device=device, dtype=dtype),
                self.requested_words,
            )
        return self._bound[key]

    def readout(self, state: Tensor) -> Tensor:
        """Map factor-state signatures to the plan's ordered quotient coordinates."""
        return self.log_readout(state) @ self.quotient_matrix.T


def compile_active_word_log(plan: LieQuotientPlan) -> ActiveWordLogPlan:
    """Compile certified bridge support and its factor closure for independent windows."""
    if not isinstance(plan, LieQuotientPlan):
        raise TypeError("plan must be a LieQuotientPlan.")
    cached = plan._active_word_cache
    if cached is not None:
        return cached
    bridge = plan._word_bridge
    columns = bridge.active_columns
    requested = tuple(
        tuple(letter - 1 for letter in bridge.ambient_words[i]) for i in columns
    )
    code_dim = (
        plan.alphabet.code_dim if isinstance(plan.alphabet, ProjectedAlphabet) else 1
    )
    program = _compile_word_program(requested, code_dim, factors=True)
    readout = _compile_word_log_readout(program, requested)
    matrix = bridge.quotient_matrix.detach().cpu()
    if matrix.layout != torch.strided:
        matrix = matrix.to_dense()
    matrix = matrix.index_select(1, torch.tensor(columns, dtype=torch.long))
    result = ActiveWordLogPlan(program, readout, matrix.to(torch.float64), requested)
    result = result.to(plan.generator_map.device, plan.generator_map.dtype)
    object.__setattr__(plan, "_active_word_cache", result)
    return result


def active_word_log_signature(
    windows: Tensor, plan: LieQuotientPlan, offset_code: Tensor | None = None
) -> Tensor:
    """Evaluate independent windows ``(..., R, C)`` in the quotient basis."""
    if windows.ndim < 2 or not windows.dtype.is_floating_point:
        raise ValueError("windows must be floating point with shape (..., R, C).")
    if not isinstance(plan, LieQuotientPlan):
        raise TypeError("plan must be a LieQuotientPlan.")
    if windows.shape[-1] != plan.alphabet.in_channels:
        raise ValueError("Input channels disagree with the quotient alphabet.")
    compiled = compile_active_word_log(plan).to(windows.device, windows.dtype)
    if isinstance(plan.alphabet, ProjectedAlphabet):
        if offset_code is None:
            raise ValueError("offset_code is required for a ProjectedAlphabet.")
        state = initialize_word_state(windows, offset_code, compiled.program)
    else:
        if offset_code is not None:
            raise ValueError("offset_code is not used with an OffsetAlphabet.")
        if (
            not isinstance(plan.alphabet, OffsetAlphabet)
            or windows.shape[-2] != plan.alphabet.kernel_size
        ):
            raise ValueError("Window size disagrees with the OffsetAlphabet.")
        zero = windows.flatten(-2)[..., :1] * 0
        state = torch.cat(
            (zero + 1, zero.expand(*zero.shape[:-1], compiled.state_count - 1)), -1
        )
        channels = windows.shape[-1]
        for offset in range(windows.shape[-2]):
            before = windows.new_zeros(*windows.shape[:-2], offset * channels)
            after = windows.new_zeros(
                *windows.shape[:-2], (windows.shape[-2] - offset - 1) * channels
            )
            segment = torch.cat((before, windows[..., offset, :], after), -1)
            state = append_word_state(state, segment, compiled.program)
    return compiled.readout(state)
