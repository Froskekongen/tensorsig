"""Private rolling evaluators for dense and chronological signature plans.

Both schedules implement the same fixed-code sliding construction; layers
select the evaluator appropriate to their compiled coordinate plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from functools import lru_cache
from itertools import groupby

import torch
from torch import Tensor

from tensorsig.tensor_ops.linear import apply_linear_map_to_signature_levels
from tensorsig.tensor_ops.series import (
    signature_inverse,
    signature_levels_from_increments,
    signature_multiply,
)

from .codes import (
    OffsetCode,
    _exact_shift_operator,
    _floating_matrix,
)
from .windows import (
    unfold_increments,
    validate_local_signature_args,
    validate_path_points,
)

from ._word_program import (
    _WordProgram,
    append_word_state,
    initialize_word_state,
)


def _lift(increments: Tensor, code_rows: Tensor) -> Tensor:
    """Lift ``(batch, steps, channels)`` increments by matching code rows."""
    lifted = torch.einsum("brc,re->brce", increments, code_rows)
    return lifted.reshape(*increments.shape[:2], -1)


def _segment_signature_levels(
    increments: Tensor,
    start: int,
    code_rows: Tensor,
    depth: int,
) -> list[Tensor]:
    """Return the signature of consecutive increments with prescribed rows."""
    lifted = _lift(
        increments[:, start : start + code_rows.shape[0]],
        code_rows,
    )
    return signature_levels_from_increments(lifted, max_depth=depth)


def _direct_coded_signature_levels(
    increments: Tensor,
    matrix: Tensor,
    depth: int,
    stride: int,
) -> tuple[Tensor, ...]:
    """Evaluate all coded windows independently and in parallel."""
    windows = unfold_increments(increments, matrix.shape[0], stride)
    lifted = torch.einsum("bnrc,re->bnrce", windows, matrix)
    lifted = lifted.reshape(*windows.shape[:3], -1)
    return tuple(signature_levels_from_increments(lifted, max_depth=depth))


def _rolling_coded_signature_levels(
    path: Tensor,
    matrix: Tensor,
    channels: int,
    depth: int,
    stride: int = 1,
) -> tuple[Tensor, ...]:
    """Return dense coded signature levels for every valid increment window.

    ``matrix`` has shape ``(window, code_dimension)`` and acts independently on
    each signal channel. Overlapping windows use exact Chen removal, alphabet
    rebasing, and insertion. The state is recomputed directly after at most one
    window length of accumulated shift to bound floating-point drift.

    The code must be fixed and exactly shift-closed for ``stride``. Nonoverlapping
    windows use the parallel direct evaluator because there is no state to reuse.
    Level ``m`` has shape ``(batch, windows, channels * code_dimension, ..., *)``.
    """
    _floating_matrix(matrix, "matrix", 2)
    if matrix.requires_grad:
        raise ValueError("Rolling evaluation requires a fixed offset code.")
    validate_path_points(path, in_channels=channels)
    validate_local_signature_args(matrix.shape[0], stride, depth)
    if matrix.shape[0] > path.shape[1] - 1:
        raise ValueError("The code window cannot exceed the path increments.")

    increments = path[:, 1:] - path[:, :-1]
    code = matrix.to(device=path.device, dtype=path.dtype)
    window = int(matrix.shape[0])
    if stride >= window:
        return _direct_coded_signature_levels(increments, code, depth, stride)

    shift = _exact_shift_operator(matrix, stride).to(
        device=path.device,
        dtype=path.dtype,
    )
    alphabet_shift = torch.kron(
        torch.eye(channels, device=path.device, dtype=path.dtype),
        shift.contiguous(),
    )
    count = 1 + (increments.shape[1] - window) // stride
    current = _segment_signature_levels(increments, 0, code, depth)
    outputs = [[level] for level in current]
    refresh_every = max(1, window // stride)

    for index in range(1, count):
        start = index * stride
        if index % refresh_every == 0:
            current = _segment_signature_levels(increments, start, code, depth)
        else:
            old_start = start - stride
            prefix = _segment_signature_levels(
                increments,
                old_start,
                code[:stride],
                depth,
            )
            current = signature_multiply(signature_inverse(prefix), current)
            current = apply_linear_map_to_signature_levels(
                current,
                matrix=alphabet_shift,
                shared_dims=1,
            )
            suffix = _segment_signature_levels(
                increments,
                old_start + window,
                code[window - stride :],
                depth,
            )
            current = signature_multiply(current, suffix)
        for level_outputs, level in zip(outputs, current, strict=True):
            level_outputs.append(level)

    return tuple(torch.stack(level_outputs, dim=1) for level_outputs in outputs)


@lru_cache(maxsize=64)
def _shift_chains(words: tuple[tuple[int, ...], ...], code_dimension: int):
    """Partition each position into consecutive temporal-label chains."""
    levels = []
    for depth, rows in groupby(enumerate(words), key=lambda row: len(row[1])):
        rows = tuple(rows)
        if depth == 0:
            continue
        positions = []
        for position in range(depth):
            chains = {}
            for local_index, (_, word) in enumerate(rows):
                channel, temporal = divmod(word[position], code_dimension)
                key = (word[:position], channel, word[position + 1 :])
                chains.setdefault(key, []).append((temporal, local_index))
            by_length = {}
            for chain in chains.values():
                chain.sort()
                if tuple(label for label, _ in chain) != tuple(range(len(chain))):
                    raise ValueError(
                        "Rolling words must be closed under temporal lowering."
                    )
                by_length.setdefault(len(chain), []).append(
                    tuple(index for _, index in chain)
                )
            groups = tuple(
                torch.tensor(chains, dtype=torch.long)
                for _, chains in sorted(by_length.items())
            )
            permutation = torch.cat([indices.flatten() for indices in groups])
            positions.append((groups, torch.argsort(permutation)))
        levels.append((rows[0][0], rows[-1][0] + 1, tuple(positions)))
    return tuple(levels)


@dataclass(frozen=True)
class _PascalTransport:
    levels: tuple
    matrix: Tensor
    raw_backward: bool

    def __call__(self, state: Tensor) -> Tensor:
        outputs = [state[..., :1]]
        for start, stop, positions in self.levels:
            level = state[..., start:stop]
            for groups, inverse in positions:
                transformed = []
                for indices, signs in groups:
                    length = indices.shape[-1]
                    values = level.index_select(-1, indices.flatten()).reshape(
                        *level.shape[:-1], *indices.shape
                    )
                    if self.raw_backward:
                        values = (values * signs).cumsum(-1) * signs
                    else:
                        values = values @ self.matrix[:length, :length]
                    transformed.append(values.flatten(-2))
                level = torch.cat(transformed, dim=-1).index_select(-1, inverse)
            outputs.append(level)
        return torch.cat(outputs, dim=-1)


@lru_cache(maxsize=64)
def _compile_pascal_transport(
    code: OffsetCode,
    words: tuple[tuple[int, ...], ...],
    device: torch.device,
    dtype: torch.dtype,
    shift: int = -1,
) -> _PascalTransport:
    """Bind a triangular Pascal transport without an ambient tensor map."""
    matrix = code._pascal_shift_operator(shift).to(device=device, dtype=dtype)
    levels = []
    for start, stop, positions in _shift_chains(words, code.dimension):
        bound = []
        for groups, inverse in positions:
            bound.append(
                (
                    tuple(
                        (
                            indices.to(device=device),
                            torch.tensor(
                                [
                                    1 if index % 2 == 0 else -1
                                    for index in range(indices.shape[-1])
                                ],
                                device=device,
                                dtype=dtype,
                            ),
                        )
                        for indices in groups
                    ),
                    inverse.to(device=device),
                )
            )
        levels.append((start, stop, tuple(bound)))
    return _PascalTransport(tuple(levels), matrix, code._pascal_raw and shift == -1)


def _roll_word_state(
    state: Tensor,
    outgoing: Tensor,
    incoming: Tensor,
    matrix: Tensor,
    program: _WordProgram,
    transport: _PascalTransport,
) -> Tensor:
    """Remove, rebase, and append one increment; all leading axes are batches."""
    left = (outgoing.unsqueeze(-1) * matrix[0]).flatten(-2)
    right = (incoming.unsqueeze(-1) * matrix[-1]).flatten(-2)
    removed = append_word_state(state, -left, program, left=True)
    return append_word_state(transport(removed), right, program)


def _rolling_chronological_features(
    path: Tensor,
    matrix: Tensor,
    code: OffsetCode,
    program: _WordProgram,
    channels: int,
    stride: int,
) -> Tensor:
    """Evaluate compact windows, batching independent periodic refresh blocks.

    Refresh every ``max(1, window // stride)`` output windows, as in the dense
    backend. A partial final block leaves the active batch when complete. No
    padded trajectory is evaluated. Batched states, vectorized word temporaries,
    output history, and autograd tapes are separate memory costs.
    """
    validate_path_points(path, in_channels=channels)
    if matrix.requires_grad:
        raise ValueError("Rolling evaluation requires a fixed offset code.")
    increments = path[:, 1:] - path[:, :-1]
    window = code.window
    refresh = max(1, window // stride)
    return _rolling_word_readouts(
        increments, matrix, code, program, stride, refresh, program.gather
    )


def _rolling_word_readouts(
    increments: Tensor,
    matrix: Tensor,
    code: OffsetCode,
    program: _WordProgram,
    stride: int,
    refresh: int,
    readout: Callable[[Tensor], Tensor],
) -> Tensor:
    """Shared batched FIFO schedule for signature or selected-log readouts."""
    window = code.window
    count = 1 + (increments.shape[1] - window) // stride
    refresh = min(refresh, count)
    initial = unfold_increments(increments, window, refresh * stride)
    program = program.to(device=increments.device, dtype=increments.dtype)
    state = initialize_word_state(initial, matrix, program)
    if refresh == 1 or count == 1:
        return readout(state)

    transport = _compile_pascal_transport(
        code, program.words, increments.device, increments.dtype
    )
    full_count, tail_count = divmod(count, refresh)
    starts = torch.arange(initial.shape[1], device=increments.device) * refresh * stride
    full_outputs = [readout(state[:, :full_count])] if full_count else []
    tail_outputs = [readout(state[:, -1])] if tail_count else []
    block_length = refresh if full_count else tail_count
    for offset in range(1, block_length):
        if tail_count and offset == tail_count:
            state = state[:, :full_count]
            starts = starts[:full_count]
        for unit in range(stride):
            indices = starts + (offset - 1) * stride + unit
            state = _roll_word_state(
                state,
                increments.index_select(1, indices),
                increments.index_select(1, indices + window),
                matrix,
                program,
                transport,
            )
        if full_count:
            full_outputs.append(readout(state[:, :full_count]))
        if offset < tail_count:
            tail_outputs.append(readout(state[:, -1]))

    outputs = []
    if full_outputs:
        outputs.append(torch.stack(full_outputs, dim=2).flatten(1, 2))
    if tail_outputs:
        outputs.append(torch.stack(tail_outputs, dim=1))
    return torch.cat(outputs, dim=1)
