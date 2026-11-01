"""Shared sparse Chen programs built from actual tensor letters."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class _WordProgram:
    """One dependency state for any collection of requested word coordinates."""

    words: tuple[tuple[int, ...], ...]
    code_dimension: int
    required_width: int
    output_indices: Tensor
    _letters: tuple[Tensor, ...] = field(repr=False)
    _right_sources: tuple[Tensor, ...] = field(repr=False)
    _left_sources: tuple[Tensor, ...] | None = field(repr=False)
    _inverse_factorials: tuple[Tensor, ...] = field(repr=False)
    _dtype: torch.dtype = field(default=torch.float64, repr=False)
    _bound_cache: dict[tuple[torch.device, torch.dtype], _WordProgram] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def state_count(self) -> int:
        return len(self.words)

    @property
    def device(self) -> torch.device:
        return self.output_indices.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> _WordProgram:
        device = self.device if device is None else torch.device(device)
        dtype = self.dtype if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError("word programs require a floating-point dtype.")
        if device == self.device and dtype == self.dtype:
            return self
        key = (device, dtype)
        if key not in self._bound_cache:
            self._bound_cache[key] = _WordProgram(
                words=self.words,
                code_dimension=self.code_dimension,
                required_width=self.required_width,
                output_indices=self.output_indices.to(device=device),
                _letters=tuple(value.to(device=device) for value in self._letters),
                _right_sources=tuple(
                    value.to(device=device) for value in self._right_sources
                ),
                _left_sources=(
                    None
                    if self._left_sources is None
                    else tuple(value.to(device=device) for value in self._left_sources)
                ),
                _inverse_factorials=tuple(
                    value.to(device=device, dtype=dtype)
                    for value in self._inverse_factorials
                ),
                _dtype=dtype,
            )
        return self._bound_cache[key]

    def gather(self, state: Tensor) -> Tensor:
        if state.ndim < 1 or state.shape[-1] != self.state_count:
            raise ValueError(f"Expected state with {self.state_count} coordinates.")
        return state.index_select(-1, self.output_indices.to(device=state.device))


def _compile_word_program(
    words: tuple[tuple[int, ...], ...],
    code_dimension: int,
    *,
    rolling: bool = False,
    factors: bool = False,
) -> _WordProgram:
    """Compile prefix dependencies, or factor/lowering closure for rolling.

    Requested order and duplicates are preserved by the output gather. State
    indices use actual words, so no ambient tensor index needs to fit int64.
    """
    if not isinstance(code_dimension, int) or isinstance(code_dimension, bool):
        raise ValueError("code_dimension must be a positive integer.")
    if code_dimension < 1:
        raise ValueError("code_dimension must be a positive integer.")
    words = tuple(tuple(word) for word in words)
    if any(
        not isinstance(letter, int) or isinstance(letter, bool) or letter < 0
        for word in words
        for letter in word
    ):
        raise ValueError("Words must contain non-negative integer letters.")
    closure = {(), *words}
    if rolling or factors:
        pending = list(closure)
        while pending:
            word = pending.pop()
            if not word:
                continue
            dependencies = [word[:-1], word[1:]]
            dependencies.extend(
                (*word[:position], letter - 1, *word[position + 1 :])
                for position, letter in enumerate(word)
                if rolling and letter % code_dimension
            )
            for dependency in dependencies:
                if dependency not in closure:
                    closure.add(dependency)
                    pending.append(dependency)
    else:
        for word in words:
            closure.update(word[:length] for length in range(1, len(word)))
    ordered = tuple(sorted(closure, key=lambda word: (len(word), word)))
    positions = {word: index for index, word in enumerate(ordered)}
    letters = []
    right_sources = []
    left_sources = []
    inverse_factorials = []
    for depth in range(1, len(ordered[-1]) + 1):
        nodes = tuple(word for word in ordered if len(word) == depth)
        letters.append(torch.tensor(nodes, dtype=torch.long).reshape(-1, depth))
        right_sources.append(
            torch.tensor(
                [
                    [positions[word[: depth - count]] for count in range(depth + 1)]
                    for word in nodes
                ],
                dtype=torch.long,
            ).reshape(-1, depth + 1)
        )
        if rolling:
            left_sources.append(
                torch.tensor(
                    [
                        [positions[word[count:]] for count in range(depth + 1)]
                        for word in nodes
                    ],
                    dtype=torch.long,
                ).reshape(-1, depth + 1)
            )
        inverse_factorials.append(
            torch.tensor(
                [1 / math.factorial(count) for count in range(depth + 1)],
                dtype=torch.float64,
            )
        )
    return _WordProgram(
        words=ordered,
        code_dimension=code_dimension,
        required_width=1
        + max((letter for word in ordered for letter in word), default=-1),
        output_indices=torch.tensor(
            [positions[word] for word in words], dtype=torch.long
        ),
        _letters=tuple(letters),
        _right_sources=tuple(right_sources),
        _left_sources=tuple(left_sources) if rolling else None,
        _inverse_factorials=tuple(inverse_factorials),
    )


@dataclass(frozen=True, eq=False)
class _WordLogReadout:
    """Packed contiguous-cut dynamic program on a factor-closed signature state."""

    initial: Tensor
    outputs: Tensor
    # Each power stores prefix/suffix indices, padding weights, and output gathers.
    powers: tuple[tuple[Tensor, Tensor, Tensor, Tensor], ...]

    def to(self, device: torch.device, dtype: torch.dtype) -> _WordLogReadout:
        return _WordLogReadout(
            self.initial.to(device=device),
            self.outputs.to(device=device),
            tuple(
                (
                    left.to(device=device),
                    right.to(device=device),
                    weights.to(device=device, dtype=dtype),
                    gather.to(device=device),
                )
                for left, right, weights, gather in self.powers
            ),
        )

    def __call__(self, state: Tensor) -> Tensor:
        previous = state.index_select(-1, self.initial)
        result = state.index_select(-1, self.outputs)
        for power, (left, right, weights, gather) in enumerate(self.powers, 2):
            prefix = previous.index_select(-1, left.flatten()).reshape(
                *state.shape[:-1], *left.shape
            )
            suffix = state.index_select(-1, right.flatten()).reshape(
                *state.shape[:-1], *right.shape
            )
            previous = (prefix * suffix * weights).sum(-1)
            padded = torch.cat((previous, state[..., :1] * 0), -1)
            result = result + ((-1) ** (power - 1) / power) * padded.index_select(
                -1, gather
            )
        return result


def _compile_word_log_readout(
    program: _WordProgram, requested: tuple[tuple[int, ...], ...]
) -> _WordLogReadout:
    """Compile P3--P4; only active prefixes carry powers of ``S - 1``."""
    if any(not word for word in requested):
        raise ValueError("Tensor-log outputs must be nonempty words.")
    positions = {word: index for index, word in enumerate(program.words)}
    prefixes = tuple(
        sorted(
            {word[:stop] for word in requested for stop in range(1, len(word) + 1)},
            key=lambda word: (len(word), word),
        )
    )
    missing = {
        word[start:stop]
        for word in requested
        for start in range(len(word))
        for stop in range(start + 1, len(word) + 1)
    } - positions.keys()
    if missing:
        raise ValueError("Selected logarithms require contiguous-factor closure.")
    initial = torch.tensor([positions[word] for word in prefixes], dtype=torch.long)
    outputs = torch.tensor([positions[word] for word in requested], dtype=torch.long)
    depth = max(map(len, requested), default=0)
    powers = []
    previous = prefixes
    for power in range(2, depth + 1):
        previous_positions = {word: index for index, word in enumerate(previous)}
        current = tuple(word for word in prefixes if len(word) >= power)
        current_positions = {word: index for index, word in enumerate(current)}
        left, right, weights = [], [], []
        for word in current:
            cuts = tuple(range(power - 1, len(word)))
            padding = depth - power + 1 - len(cuts)
            left.append(
                [previous_positions[word[:cut]] for cut in cuts] + [0] * padding
            )
            right.append([positions[word[cut:]] for cut in cuts] + [0] * padding)
            weights.append([1] * len(cuts) + [0] * padding)
        powers.append(
            (
                torch.tensor(left, dtype=torch.long),
                torch.tensor(right, dtype=torch.long),
                torch.tensor(weights, dtype=torch.float64),
                torch.tensor(
                    [current_positions.get(word, len(current)) for word in requested],
                    dtype=torch.long,
                ),
            )
        )
        previous = current
    return _WordLogReadout(initial, outputs, tuple(powers))


def append_word_state(
    state: Tensor,
    segment: Tensor,
    program: _WordProgram,
    *,
    left: bool = False,
) -> Tensor:
    """Multiply by ``exp(segment)`` on the right or, for rolling, the left."""
    if state.ndim < 1 or state.shape[-1] != program.state_count:
        raise ValueError(f"Expected state with {program.state_count} coordinates.")
    if segment.ndim < 1 or state.shape[:-1] != segment.shape[:-1]:
        raise ValueError("state and segment must have identical leading dimensions.")
    if state.dtype != segment.dtype or state.device != segment.device:
        raise ValueError("state and segment must share dtype and device.")
    if not state.dtype.is_floating_point:
        raise ValueError("state and segment must have a floating-point dtype.")
    program = program.to(device=state.device, dtype=state.dtype)
    sources = program._left_sources if left else program._right_sources
    if sources is None:
        raise ValueError("Left multiplication requires a rolling word program.")
    next_levels = [state[..., :1] + segment[..., :1].sum(dim=-1, keepdim=True) * 0]
    for depth, (letters, source, factorials) in enumerate(
        zip(program._letters, sources, program._inverse_factorials, strict=True),
        start=1,
    ):
        count = len(letters)
        values = segment.index_select(-1, letters.flatten()).reshape(
            *segment.shape[:-1], count, depth
        )
        products = (values if left else values.flip(-1)).cumprod(-1)
        exponential = torch.cat((torch.ones_like(values[..., :1]), products), dim=-1)
        old_state = state.index_select(-1, source.flatten()).reshape(
            *state.shape[:-1], count, depth + 1
        )
        next_levels.append((old_state * exponential * factorials).sum(dim=-1))
    return torch.cat(next_levels, dim=-1)


def initialize_word_state(
    windows: Tensor,
    code: Tensor,
    program: _WordProgram,
) -> Tensor:
    """Evaluate a full dependency state on ``(..., window, channels)`` input."""
    if windows.ndim < 2 or not windows.dtype.is_floating_point:
        raise ValueError("windows must be floating point with shape (..., R, C).")
    if code.ndim != 2 or tuple(code.shape) != (
        windows.shape[-2],
        program.code_dimension,
    ):
        raise ValueError("code must have shape (window, code_dimension).")
    if not code.dtype.is_floating_point:
        raise ValueError("code must have a floating-point dtype.")
    width = windows.shape[-1] * program.code_dimension
    if program.required_width > width:
        raise ValueError("A program letter exceeds the input alphabet width.")
    program = program.to(device=windows.device, dtype=windows.dtype)
    code = code.to(device=windows.device, dtype=windows.dtype)
    # One value connects each input to empty outputs without overflowing a sum
    # of otherwise finite increments or code coefficients.
    zero = windows.flatten(-2)[..., :1].sum(dim=-1) * 0 + code.reshape(-1)[:1].sum() * 0
    state = torch.cat(
        (
            (zero + 1).unsqueeze(-1),
            zero.unsqueeze(-1).expand(*zero.shape, program.state_count - 1),
        ),
        dim=-1,
    )
    for offset in range(windows.shape[-2]):
        segment = (windows[..., offset, :, None] * code[offset]).flatten(-2)
        state = append_word_state(state, segment, program)
    return state


def evaluate_word_program(
    windows: Tensor,
    code: Tensor,
    program: _WordProgram,
) -> Tensor:
    """Evaluate requested words while sharing every prefix across levels."""
    return program.gather(initialize_word_state(windows, code, program))
