"""Gap-factorized readouts of strict local Volterra plans."""

from __future__ import annotations

import math
from typing import Literal, cast

import torch
from torch import Tensor

from tensorsig.tensor_ops.linear import contract_features_with_readout

from .plans import PolynomialPlan
from .windows import validate_path_points
from .windows import increment_windows


VolterraBackend = Literal["auto", "dynamic", "enumerate"]
GapInit = Literal["ones", "uniform"]


def _validate_backend(backend: str) -> VolterraBackend:
    if not isinstance(backend, str):
        raise TypeError("backend must be a string.")
    if backend not in {"auto", "dynamic", "enumerate"}:
        raise ValueError("backend must be 'auto', 'dynamic', or 'enumerate'.")
    return cast(VolterraBackend, backend)


def _validate_gap_init(initialization: str) -> GapInit:
    if not isinstance(initialization, str):
        raise TypeError("gap_initialization must be a string.")
    if initialization not in {"ones", "uniform"}:
        raise ValueError("gap_initialization must be 'ones' or 'uniform'.")
    return cast(GapInit, initialization)


def _validate_positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _strict_offset_tuples(plan: PolynomialPlan) -> tuple[Tensor, ...]:
    offset_levels = []
    for depth, dictionary in zip(plan.depths, plan._dictionaries, strict=True):
        offsets = torch.unique_consecutive(dictionary.offsets, dim=0)
        if offsets.shape[0] == 0:
            raise ValueError(
                f"The strict polynomial plan has no degree-{depth} features."
            )
        offset_levels.append(offsets)
    return tuple(offset_levels)


def _validate_plan(plan: PolynomialPlan) -> tuple[Tensor, ...]:
    if not isinstance(plan, PolynomialPlan):
        raise TypeError("plan must be a PolynomialPlan.")
    if plan.max_offset_multiplicity != 1:
        raise ValueError(
            "GapFactorizedVolterra requires a plan compiled with "
            "max_offset_multiplicity=1."
        )
    return _strict_offset_tuples(plan)


def _transition_matrix(
    factors: Tensor,
    gap_indices: Tensor,
    valid_gaps: Tensor,
) -> Tensor:
    """Return strict lower-triangular weights indexed by positive gap."""
    selected = factors[:, gap_indices]
    return torch.where(
        valid_gaps.unsqueeze(0),
        selected,
        torch.zeros_like(selected),
    )


def _terminal_weights(
    factors: Tensor,
    lag_indices: Tensor,
    valid_lags: Tensor,
) -> Tensor:
    """Return weights from the final selected offset to the window endpoint."""
    selected = factors[:, lag_indices]
    return torch.where(
        valid_lags.unsqueeze(0),
        selected,
        torch.zeros_like(selected),
    )


def _dynamic_features(
    windows: Tensor,
    internal_gap_factors: Tensor | None,
    terminal_lag_factors: Tensor,
    gap_indices: Tensor | None,
    valid_gaps: Tensor | None,
    lag_indices: Tensor,
    valid_lags: Tensor,
) -> Tensor:
    """Evaluate one degree without enumerating strict offset tuples."""
    batch, count, window, channels = windows.shape
    temporal_rank = terminal_lag_factors.shape[0]
    depth = 1 if internal_gap_factors is None else internal_gap_factors.shape[1] + 1
    state = windows.unsqueeze(2).expand(
        batch,
        count,
        temporal_rank,
        window,
        channels,
    )
    for position in range(depth - 1):
        assert internal_gap_factors is not None
        assert gap_indices is not None and valid_gaps is not None
        transition = _transition_matrix(
            internal_gap_factors[:, position],
            gap_indices,
            valid_gaps,
        )
        accumulated = torch.einsum("lru,bnlu...->bnlr...", transition, state)
        next_factor = windows.reshape(
            batch,
            count,
            1,
            window,
            *([1] * (position + 1)),
            channels,
        )
        state = accumulated.unsqueeze(-1) * next_factor
    terminal = _terminal_weights(
        terminal_lag_factors,
        lag_indices,
        valid_lags,
    )
    return torch.einsum("lr,bnlr...->bnl...", terminal, state)


def _temporal_weights(
    offsets: Tensor,
    internal_gap_factors: Tensor | None,
    terminal_lag_factors: Tensor,
    window: int,
) -> Tensor:
    """Evaluate gap-factorized weights on explicit strict offset tuples."""
    temporal_rank = terminal_lag_factors.shape[0]
    weights = torch.ones(
        temporal_rank,
        offsets.shape[0],
        device=terminal_lag_factors.device,
        dtype=terminal_lag_factors.dtype,
    )
    offsets = offsets.to(device=terminal_lag_factors.device)
    for position in range(offsets.shape[1] - 1):
        assert internal_gap_factors is not None
        gaps = offsets[:, position + 1] - offsets[:, position]
        weights = weights * internal_gap_factors[:, position, gaps - 1]
    terminal_lags = window - 1 - offsets[:, -1]
    return weights * terminal_lag_factors[:, terminal_lags]


class GapFactorizedVolterra(torch.nn.Module):
    """A low-rank gap-factorized readout of a strict polynomial plan.

    For strict offsets ``r_1 < ... < r_m``, the temporal coefficient of one
    rank component is a product of factors indexed by the positive internal
    gaps ``r_{j+1} - r_j`` and the terminal lag ``window - 1 - r_m``. A
    learned channel tensor is then contracted with each component. The
    represented functions are exactly a structured subset of the supplied
    ``PolynomialPlan``.

    Parameters
    ----------
    plan : PolynomialPlan
        A plan compiled with ``max_offset_multiplicity=1``. Span-filtered plans
        are supported through the enumerated backend.
    out_features : int
        Number of output channels.
    temporal_rank : int
        Number of separable temporal/channel components at every degree.
    stride : int, optional
        Window stride measured in path increments.
    bias : bool, optional
        Whether to learn an output bias.
    backend : {"auto", "dynamic", "enumerate"}, optional
        ``"dynamic"`` avoids tuple enumeration and requires an unfiltered
        strict plan. ``"enumerate"`` evaluates the plan's retained tuples.
    gap_initialization : {"ones", "uniform"}, optional
        Initialization for internal-gap and terminal-lag factors.
    device : torch.device or str, optional
        Initial parameter device.
    dtype : torch.dtype, optional
        Initial floating-point parameter dtype.
    """

    def __init__(
        self,
        plan: PolynomialPlan,
        out_features: int,
        temporal_rank: int,
        stride: int = 1,
        bias: bool = True,
        *,
        backend: VolterraBackend = "auto",
        gap_initialization: GapInit = "ones",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        offset_levels = _validate_plan(plan)
        _validate_positive_integer(out_features, "out_features")
        _validate_positive_integer(temporal_rank, "temporal_rank")
        _validate_positive_integer(stride, "stride")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a boolean.")
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError("dtype must be floating-point.")

        self.plan = plan
        self.out_features = int(out_features)
        self.temporal_rank = int(temporal_rank)
        self.stride = int(stride)
        self.backend = _validate_backend(backend)
        self.gap_initialization = _validate_gap_init(gap_initialization)
        self._span_filtered = any(
            offsets.shape[0] < math.comb(plan.window, depth)
            for depth, offsets in zip(plan.depths, offset_levels, strict=True)
        )
        if self.backend == "dynamic" and self._span_filtered:
            raise ValueError("backend='dynamic' does not support a span-filtered plan.")
        self._resolved_backend = (
            "enumerate"
            if self.backend == "enumerate" or self._span_filtered
            else "dynamic"
        )

        internal_factors: list[torch.nn.Parameter | None] = []
        terminal_factors = []
        channel_weights = []
        max_span = plan.max_offset_span
        for index, (depth, offsets) in enumerate(
            zip(plan.depths, offset_levels, strict=True)
        ):
            terminal_width = plan.window - depth + 1
            if depth == 1:
                internal_factors.append(None)
            else:
                internal_width = terminal_width
                if max_span is not None:
                    internal_width = min(
                        internal_width,
                        max_span - (depth - 2),
                    )
                internal_factors.append(
                    torch.nn.Parameter(
                        torch.empty(
                            temporal_rank,
                            depth - 1,
                            internal_width,
                            device=device,
                            dtype=dtype,
                        )
                    )
                )
            terminal_factors.append(
                torch.nn.Parameter(
                    torch.empty(
                        temporal_rank,
                        terminal_width,
                        device=device,
                        dtype=dtype,
                    )
                )
            )
            channel_weights.append(
                torch.nn.Parameter(
                    torch.empty(
                        out_features,
                        temporal_rank,
                        *([plan.channels] * depth),
                        device=device,
                        dtype=dtype,
                    )
                )
            )
            if self._resolved_backend == "dynamic":
                positions = torch.arange(plan.window, device=device)
                gaps = positions[:, None] - positions[None, :]
                if depth > 1:
                    self.register_buffer(
                        f"_gap_indices_{index}",
                        (gaps - 1).clamp(min=0, max=internal_width - 1),
                        persistent=False,
                    )
                    self.register_buffer(
                        f"_valid_gaps_{index}",
                        (gaps > 0) & (gaps <= internal_width),
                        persistent=False,
                    )
                lags = torch.arange(plan.window - 1, -1, -1, device=device)
                self.register_buffer(
                    f"_lag_indices_{index}",
                    lags.clamp(max=terminal_width - 1),
                    persistent=False,
                )
                self.register_buffer(
                    f"_valid_lags_{index}",
                    lags < terminal_width,
                    persistent=False,
                )
            else:
                dictionary = plan._dictionaries[index]
                _, offset_tuple_indices = torch.unique_consecutive(
                    dictionary.offsets,
                    dim=0,
                    return_inverse=True,
                )
                self.register_buffer(
                    f"_offset_tuples_{index}",
                    offsets.to(device=device),
                    persistent=False,
                )
                self.register_buffer(
                    f"_offset_tuple_indices_{index}",
                    offset_tuple_indices.to(device=device),
                    persistent=False,
                )
                self.register_buffer(
                    f"_channel_tuples_{index}",
                    dictionary.channels.to(device=device),
                    persistent=False,
                )

        self.internal_gap_factors = torch.nn.ParameterList(internal_factors)
        self.terminal_lag_factors = torch.nn.ParameterList(terminal_factors)
        self.channel_weights = torch.nn.ParameterList(channel_weights)
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @property
    def window(self) -> int:
        """Number of increments in each local window."""
        return self.plan.window

    @property
    def channels(self) -> int:
        """Expected path channel count."""
        return self.plan.channels

    @property
    def depths(self) -> tuple[int, ...]:
        """Homogeneous polynomial degrees represented by the layer."""
        return self.plan.depths

    def reset_parameters(self) -> None:
        """Initialize gap factors, channel weights, and bias."""
        for depth, internal, terminal, channel_weight in zip(
            self.depths,
            self.internal_gap_factors,
            self.terminal_lag_factors,
            self.channel_weights,
            strict=True,
        ):
            if self.gap_initialization == "ones":
                if internal is not None:
                    torch.nn.init.ones_(internal)
                torch.nn.init.ones_(terminal)
            else:
                if internal is not None:
                    torch.nn.init.uniform_(internal, -0.1, 0.1)
                torch.nn.init.uniform_(terminal, -0.1, 0.1)
            bound = (self.temporal_rank * self.channels**depth) ** -0.5
            torch.nn.init.uniform_(channel_weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _dynamic_level(
        self,
        windows: Tensor,
        index: int,
    ) -> Tensor:
        internal = self.internal_gap_factors[index]
        terminal = self.terminal_lag_factors[index]
        return _dynamic_features(
            windows,
            internal,
            terminal,
            getattr(self, f"_gap_indices_{index}", None),
            getattr(self, f"_valid_gaps_{index}", None),
            getattr(self, f"_lag_indices_{index}"),
            getattr(self, f"_valid_lags_{index}"),
        )

    def _expanded_weight(self) -> Tensor:
        """Expand structured parameters for the plan contraction backend."""
        blocks = []
        for index, (depth, channel_weight) in enumerate(
            zip(self.depths, self.channel_weights, strict=True)
        ):
            offsets = getattr(self, f"_offset_tuples_{index}")
            temporal = _temporal_weights(
                offsets,
                self.internal_gap_factors[index],
                self.terminal_lag_factors[index],
                self.window,
            ).index_select(
                1,
                getattr(self, f"_offset_tuple_indices_{index}"),
            )
            channels = getattr(self, f"_channel_tuples_{index}")
            channel_indices = tuple(channels[:, position] for position in range(depth))
            selected = channel_weight[(slice(None), slice(None), *channel_indices)]
            blocks.append((selected * temporal.unsqueeze(0)).sum(dim=1))
        return torch.cat(blocks, dim=1)

    def forward(self, path: Tensor) -> Tensor:
        """Return one gap-factorized Volterra readout per local window."""
        batch, _, _ = validate_path_points(path, in_channels=self.channels)
        reference = self.channel_weights[0]
        if path.device != reference.device or path.dtype != reference.dtype:
            raise RuntimeError(
                "path dtype/device must match the module parameters; move the "
                "module and input explicitly before evaluation"
            )
        windows = increment_windows(path, size=self.window, stride=self.stride)
        if self._resolved_backend == "enumerate":
            output = self.plan.contract(windows, self._expanded_weight())
        else:
            output = path.new_zeros(batch, windows.shape[1], self.out_features)
            for index, channel_weight in enumerate(self.channel_weights):
                output = output + contract_features_with_readout(
                    self._dynamic_level(windows, index),
                    channel_weight,
                )
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"window={self.window}, channels={self.channels}, depths={self.depths}, "
            f"out_features={self.out_features}, temporal_rank={self.temporal_rank}, "
            f"stride={self.stride}, backend={self.backend!r}, "
            f"bias={self.bias is not None}"
        )


__all__ = ["GapFactorizedVolterra"]
