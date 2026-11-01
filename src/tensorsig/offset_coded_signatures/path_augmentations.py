"""Composable path augmentations for local signature pipelines."""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ..classical.windows import windowed_log_signature


_PaddingMode = Literal["repeat", "zero"]
_FilterInit = Literal["uniform", "normal", "zeros"]
_LogSignatureMode = Literal["words", "hall"]


def _normalize_padding(padding: str) -> _PaddingMode:
    if not isinstance(padding, str):
        raise TypeError("padding must be a string.")
    normalized = padding.lower()
    if normalized not in {"repeat", "zero"}:
        raise ValueError("padding must be either 'repeat' or 'zero'.")
    return cast(_PaddingMode, normalized)


def _normalize_init(init: str) -> _FilterInit:
    if not isinstance(init, str):
        raise TypeError("init must be a string.")
    normalized = init.lower()
    if normalized not in {"uniform", "normal", "zeros"}:
        raise ValueError("init must be one of 'uniform', 'normal', or 'zeros'.")
    return cast(_FilterInit, normalized)


def _normalize_log_signature_mode(mode: str) -> _LogSignatureMode:
    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")
    normalized = mode.lower()
    if normalized not in {"words", "hall"}:
        raise ValueError("mode must be either 'words' or 'hall'.")
    return cast(_LogSignatureMode, normalized)


def _validate_integer(name: str, value: int, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _validate_sequence(
    path: Tensor,
    in_channels: int | None = None,
) -> tuple[int, int, int]:
    if path.ndim != 3:
        raise ValueError(
            f"Expected path shape (batch, length, channels), got {tuple(path.shape)}."
        )
    if not path.dtype.is_floating_point:
        raise ValueError("path must have a floating-point dtype.")
    batch, length, channels = path.shape
    if length < 1:
        raise ValueError("path must contain at least one point.")
    if channels < 1:
        raise ValueError("path must contain at least one channel.")
    if in_channels is not None and channels != in_channels:
        raise ValueError(f"Expected {in_channels} input channels, got {channels}.")
    return int(batch), int(length), int(channels)


def causal_fir(
    path: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    padding: str = "repeat",
) -> Tensor:
    """Apply a causal finite-impulse-response filter bank.

    Parameters
    ----------
    path : Tensor
        Path points with shape ``(batch, length, in_channels)``.
    weight : Tensor
        Filter weights with shape ``(num_filters, in_channels, max_lag + 1)``.
        ``weight[..., 0]`` multiplies the current point and ``weight[..., lag]``
        multiplies the point at that causal lag.
    bias : Tensor, optional
        Per-filter bias with shape ``(num_filters,)``.
    padding : {"repeat", "zero"}, optional
        Values before the first point either repeat that point or are zero.

    Returns
    -------
    Tensor
        Filtered points with shape ``(batch, length, num_filters)``.
    """
    _, _, in_channels = _validate_sequence(path)
    if weight.ndim != 3:
        raise ValueError(
            "weight must have shape (num_filters, in_channels, max_lag + 1)."
        )
    num_filters, weight_channels, kernel_width = weight.shape
    if num_filters < 1:
        raise ValueError("weight must contain at least one filter.")
    if weight_channels != in_channels:
        raise ValueError(
            f"Expected weight.shape[1] == {in_channels}, got {weight_channels}."
        )
    if kernel_width < 1:
        raise ValueError("weight must contain at least one lag coefficient.")
    if bias is not None and tuple(bias.shape) != (num_filters,):
        raise ValueError(f"bias must have shape ({num_filters},).")

    padding = _normalize_padding(padding)
    max_lag = kernel_width - 1
    channel_first = path.movedim(1, 2)
    if max_lag:
        if padding == "zero":
            channel_first = F.pad(channel_first, (max_lag, 0))
        else:
            channel_first = F.pad(channel_first, (max_lag, 0), mode="replicate")

    kernel = weight.to(device=path.device, dtype=path.dtype).flip(-1)
    filter_bias = (
        None if bias is None else bias.to(device=path.device, dtype=path.dtype)
    )
    return F.conv1d(channel_first, kernel, bias=filter_bias).movedim(1, 2)


class CausalFIRAugmentation(torch.nn.Module):
    """Append learned causal FIR channels to a path.

    Each learned filter mixes all input channels and lags from zero through
    ``max_lag``. The output is ``concat(path, filtered_path)`` and therefore
    has ``in_channels + num_filters`` channels.

    A filter bias translates its output channel. Signatures and logsignatures
    directly following this augmentation are translation invariant, so the
    bias only affects them when an intervening nonlinear augmentation, such as
    pairwise products, converts absolute levels into increments.

    Parameters
    ----------
    in_channels : int
        Input path width.
    num_filters : int
        Number of learned FIR channels to append.
    max_lag : int
        Largest causal lag included by each filter.
    bias : bool, optional
        Whether the learned FIR channels have additive biases.
    padding : {"repeat", "zero"}, optional
        Left-boundary values before the first path point.
    init : {"uniform", "normal", "zeros"}, optional
        FIR weight initialization.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        max_lag: int,
        *,
        bias: bool = True,
        padding: str = "repeat",
        init: str = "uniform",
    ) -> None:
        super().__init__()
        self.in_channels = _validate_integer("in_channels", in_channels, minimum=1)
        self.num_filters = _validate_integer("num_filters", num_filters, minimum=1)
        self.max_lag = _validate_integer("max_lag", max_lag, minimum=0)
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool.")
        self.padding = _normalize_padding(padding)
        self.init = _normalize_init(init)
        self.out_channels = self.in_channels + self.num_filters

        self.weight = torch.nn.Parameter(
            torch.empty(self.num_filters, self.in_channels, self.max_lag + 1)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.num_filters))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the filters and optional bias."""
        fan_in = self.in_channels * (self.max_lag + 1)
        if self.init == "zeros":
            torch.nn.init.zeros_(self.weight)
        elif self.init == "normal":
            torch.nn.init.normal_(self.weight, mean=0.0, std=fan_in**-0.5)
        else:
            bound = fan_in**-0.5
            torch.nn.init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            bound = fan_in**-0.5
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def filtered(self, path: Tensor) -> Tensor:
        """Return only the learned FIR channels."""
        _validate_sequence(path, in_channels=self.in_channels)
        if path.device != self.weight.device or path.dtype != self.weight.dtype:
            raise RuntimeError(
                "path dtype/device must match the augmentation parameters; move "
                "the module and input explicitly"
            )
        return causal_fir(path, self.weight, self.bias, padding=self.padding)

    def forward(self, path: Tensor) -> Tensor:
        """Return the input path followed by its learned FIR channels."""
        return torch.cat((path, self.filtered(path)), dim=-1)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, num_filters={self.num_filters}, "
            f"max_lag={self.max_lag}, out_channels={self.out_channels}, "
            f"padding={self.padding!r}, bias={self.bias is not None}"
        )


class PairwiseProductAugmentation(torch.nn.Module):
    """Append all distinct pairwise channel products to a path.

    Parameters
    ----------
    in_channels : int
        Input path width.
    include_inputs : bool, optional
        Whether the input channels precede the product channels in the output.
    """

    def __init__(self, in_channels: int, *, include_inputs: bool = True) -> None:
        super().__init__()
        self.in_channels = _validate_integer("in_channels", in_channels, minimum=1)
        if not isinstance(include_inputs, bool):
            raise TypeError("include_inputs must be a bool.")
        self.include_inputs = include_inputs
        products = self.in_channels * (self.in_channels - 1) // 2
        self.out_channels = products + (self.in_channels if include_inputs else 0)

    def forward(self, path: Tensor) -> Tensor:
        """Return distinct products, optionally preceded by the input channels."""
        _validate_sequence(path, in_channels=self.in_channels)
        first, second = torch.triu_indices(
            self.in_channels,
            self.in_channels,
            offset=1,
            device=path.device,
        )
        products = path[..., first] * path[..., second]
        return torch.cat((path, products), dim=-1) if self.include_inputs else products

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"include_inputs={self.include_inputs}"
        )


class CausalWindowedLogSignature(torch.nn.Module):
    """Apply explicit causal path augmentations before windowed logsignatures.

    Parameters
    ----------
    fir : CausalFIRAugmentation
        Learned causal augmentation applied first.
    depth : int
        Logsignature truncation depth.
    window_size : int
        Number of path points per window.
    hop_size : int, optional
        Number of path points between window starts.
    pairwise : PairwiseProductAugmentation, optional
        Explicit nonlinear augmentation applied after the FIR augmentation.
    mode : {"words", "hall"}, optional
        Logsignature coordinate basis.
    """

    def __init__(
        self,
        fir: CausalFIRAugmentation,
        *,
        depth: int,
        window_size: int,
        hop_size: int = 1,
        pairwise: PairwiseProductAugmentation | None = None,
        mode: str = "words",
    ) -> None:
        super().__init__()
        if not isinstance(fir, CausalFIRAugmentation):
            raise TypeError("fir must be a CausalFIRAugmentation.")
        if pairwise is not None and not isinstance(
            pairwise, PairwiseProductAugmentation
        ):
            raise TypeError("pairwise must be a PairwiseProductAugmentation or None.")
        if pairwise is not None and pairwise.in_channels != fir.out_channels:
            raise ValueError(
                "pairwise input channels must equal the FIR augmentation output "
                f"channels ({fir.out_channels})."
            )

        self.fir = fir
        self.pairwise = pairwise
        self.depth = _validate_integer("depth", depth, minimum=1)
        self.window_size = _validate_integer("window_size", window_size, minimum=2)
        self.hop_size = _validate_integer("hop_size", hop_size, minimum=1)
        self.mode = _normalize_log_signature_mode(mode)

    @property
    def embedded_channels(self) -> int:
        """Return the channel count after FIR augmentation."""
        return self.fir.out_channels

    @property
    def signature_channels(self) -> int:
        """Return the channel count passed to the logsignature."""
        return (
            self.fir.out_channels
            if self.pairwise is None
            else self.pairwise.out_channels
        )

    def augment(self, path: Tensor) -> Tensor:
        """Return the explicitly composed path augmentation."""
        augmented = self.fir(path)
        return augmented if self.pairwise is None else self.pairwise(augmented)

    def forward(self, path: Tensor) -> Tensor:
        """Return windowed logsignatures of the augmented path."""
        return windowed_log_signature(
            self.augment(path),
            depth=self.depth,
            window_size=self.window_size,
            hop_size=self.hop_size,
            mode=self.mode,
        )

    def extra_repr(self) -> str:
        return (
            f"depth={self.depth}, window_size={self.window_size}, "
            f"hop_size={self.hop_size}, embedded_channels={self.embedded_channels}, "
            f"signature_channels={self.signature_channels}, mode={self.mode!r}"
        )


__all__ = [
    "CausalFIRAugmentation",
    "CausalWindowedLogSignature",
    "PairwiseProductAugmentation",
    "causal_fir",
]
