"""General scaled scalar-monomial dictionaries for local windows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import itertools
from typing import Any, Literal, cast

import torch
from torch import Tensor

from .monomials import (
    contract_monomial_readout_from_offset_channels,
    local_monomial_features_from_offset_channels,
)
from .offset_patterns import (
    offset_max_multiplicity,
    offset_run_lengths,
    offset_span,
    offset_support_size,
    raw_offset_weight_fraction,
    symmetric_run_channel_tuples,
    validate_offset_filter_args,
)
from .utils import validate_local_signature_args


RawSignatureScale = Literal["signature", "support"]


def normalize_raw_signature_scale(scale: RawSignatureScale) -> RawSignatureScale:
    """Normalize and validate raw signature monomial scaling."""
    normalized = (scale or "signature").lower()
    if normalized not in {"signature", "support"}:
        raise ValueError("scale must be either 'signature' or 'support'.")
    return cast(RawSignatureScale, normalized)


def _passes_offset_filters(
    offsets: tuple[int, ...],
    max_offset_span: int | None,
    max_offset_support: int | None,
    max_offset_multiplicity: int | None,
) -> bool:
    if max_offset_span is not None and offset_span(offsets) > max_offset_span:
        return False
    if (
        max_offset_support is not None
        and offset_support_size(offsets) > max_offset_support
    ):
        return False
    return not (
        max_offset_multiplicity is not None
        and offset_max_multiplicity(offsets) > max_offset_multiplicity
    )


def filtered_offset_tuples(
    kernel_size: int,
    depth: int,
    max_offset_span: int | None = None,
    max_offset_support: int | None = None,
    max_offset_multiplicity: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Enumerate nondecreasing offset tuples after optional filters.

    Parameters
    ----------
    kernel_size : int
        Number of increments in each local window.
    depth : int
        Homogeneous degree / signature level.
    max_offset_span : int, optional
        Keep only offset tuples whose span is at most this value.
    max_offset_support : int, optional
        Keep only tuples using at most this many distinct offsets.
    max_offset_multiplicity : int, optional
        Keep only tuples whose largest repeated-offset run is at most this
        value.
    device : torch.device or str, optional
        Device for the returned tensor.

    Returns
    -------
    Tensor
        Long tensor with shape ``(num_tuples, depth)``.
    """
    validate_local_signature_args(kernel_size, hop_size=1, depth=depth)
    validate_offset_filter_args(
        max_offset_span=max_offset_span,
        max_offset_support=max_offset_support,
        max_offset_multiplicity=max_offset_multiplicity,
    )
    rows = [
        offsets
        for offsets in itertools.combinations_with_replacement(
            range(kernel_size), depth
        )
        if _passes_offset_filters(
            offsets,
            max_offset_span=max_offset_span,
            max_offset_support=max_offset_support,
            max_offset_multiplicity=max_offset_multiplicity,
        )
    ]
    if rows:
        return torch.tensor(rows, device=device, dtype=torch.long)
    return torch.empty((0, depth), device=device, dtype=torch.long)


def raw_signature_monomial_indices(
    in_channels: int,
    kernel_size: int,
    depth: int,
    max_offset_span: int | None = None,
    max_offset_support: int | None = None,
    max_offset_multiplicity: int | None = None,
    scale: RawSignatureScale = "signature",
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Enumerate scalar-compressed raw signature monomial coordinates."""
    validate_local_signature_args(kernel_size, hop_size=1, depth=depth)
    validate_offset_filter_args(
        max_offset_span=max_offset_span,
        max_offset_support=max_offset_support,
        max_offset_multiplicity=max_offset_multiplicity,
    )
    if in_channels < 1:
        raise ValueError("in_channels must be positive.")
    scale = normalize_raw_signature_scale(scale)

    offset_rows: list[tuple[int, ...]] = []
    channel_rows: list[tuple[int, ...]] = []
    coefficient_values: list[float] = []
    offsets = filtered_offset_tuples(
        kernel_size=kernel_size,
        depth=depth,
        max_offset_span=max_offset_span,
        max_offset_support=max_offset_support,
        max_offset_multiplicity=max_offset_multiplicity,
    )
    for offset_tuple_list in offsets.tolist():
        offset_tuple = tuple(int(value) for value in offset_tuple_list)
        run_lengths = offset_run_lengths(offset_tuple)
        coefficient = (
            float(raw_offset_weight_fraction(offset_tuple))
            if scale == "signature"
            else 1.0
        )
        for channel_tuple in symmetric_run_channel_tuples(in_channels, run_lengths):
            offset_rows.append(offset_tuple)
            channel_rows.append(channel_tuple)
            coefficient_values.append(coefficient)

    if offset_rows:
        offset_tensor = torch.tensor(offset_rows, device=device, dtype=torch.long)
        channel_tensor = torch.tensor(channel_rows, device=device, dtype=torch.long)
    else:
        offset_tensor = torch.empty((0, depth), device=device, dtype=torch.long)
        channel_tensor = torch.empty((0, depth), device=device, dtype=torch.long)
    coefficients = torch.tensor(coefficient_values, device=device, dtype=dtype)
    return offset_tensor, channel_tensor, coefficients


@dataclass(frozen=True)
class ScaledMonomialDictionary:
    """A local scalar-feature dictionary with coordinates ``c_j z**nu_j``.

    Parameters
    ----------
    offsets : Tensor
        Long tensor with shape ``(num_features, depth)``.
    channels : Tensor
        Long tensor with shape ``(num_features, depth)``.
    coefficients : Tensor
        Per-feature scaling coefficients with shape ``(num_features,)``.
    metadata : Mapping[str, Any], optional
        Descriptive construction metadata. It is not used in computation.
    """

    offsets: Tensor
    channels: Tensor
    coefficients: Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _bound_cache: dict[tuple[torch.device, torch.dtype], "ScaledMonomialDictionary"] = (
        field(init=False, default_factory=dict, repr=False, compare=False)
    )

    def __post_init__(self) -> None:
        """Validate dictionary tensor shapes."""
        if self.offsets.ndim != 2 or self.channels.ndim != 2:
            raise ValueError("offsets and channels must have shape (M, depth).")
        if self.offsets.dtype != torch.long or self.channels.dtype != torch.long:
            raise ValueError("offsets and channels must have dtype torch.long.")
        if self.offsets.shape != self.channels.shape:
            raise ValueError("offsets and channels must have identical shapes.")
        if self.coefficients.ndim != 1:
            raise ValueError("coefficients must have shape (M,).")
        if self.coefficients.shape[0] != self.offsets.shape[0]:
            raise ValueError("coefficients must have one value per feature row.")
        if self.offsets.shape[1] < 1:
            raise ValueError("dictionary depth must be positive.")

    @property
    def feature_count(self) -> int:
        """Number of scalar monomial features."""
        return int(self.offsets.shape[0])

    @property
    def depth(self) -> int:
        """Homogeneous degree of every monomial in the dictionary."""
        return int(self.offsets.shape[1])

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ScaledMonomialDictionary":
        """Return a copy on a different device and/or coefficient dtype."""
        target_device = self.offsets.device if device is None else torch.device(device)
        target_dtype = self.coefficients.dtype if dtype is None else dtype
        if not target_dtype.is_floating_point:
            raise ValueError("dtype must be floating-point.")
        if (
            target_device == self.offsets.device
            and target_device == self.channels.device
            and target_device == self.coefficients.device
            and target_dtype == self.coefficients.dtype
        ):
            return self
        cache_key = (target_device, target_dtype)
        cached = self._bound_cache.get(cache_key)
        if cached is not None:
            return cached
        moved = ScaledMonomialDictionary(
            offsets=self.offsets.to(device=target_device),
            channels=self.channels.to(device=target_device),
            coefficients=self.coefficients.to(
                device=target_device,
                dtype=target_dtype,
            ),
            metadata=dict(self.metadata),
        )
        self._bound_cache[cache_key] = moved
        return moved

    def features(self, dxw: Tensor, chunk_size: int | None = None) -> Tensor:
        """Evaluate dictionary features on local increment windows."""
        bound = self.to(device=dxw.device, dtype=dxw.dtype)
        return local_monomial_features_from_offset_channels(
            dxw,
            bound.offsets,
            bound.channels,
            weights=bound.coefficients,
            chunk_size=chunk_size,
        )

    def contract(
        self,
        dxw: Tensor,
        readout: Tensor,
        chunk_size: int | None = None,
    ) -> Tensor:
        """Contract dictionary features with a readout matrix."""
        bound = self.to(device=dxw.device, dtype=dxw.dtype)
        return contract_monomial_readout_from_offset_channels(
            dxw,
            readout,
            bound.offsets,
            bound.channels,
            weights=bound.coefficients,
            chunk_size=chunk_size,
        )

    @classmethod
    def raw_signature(
        cls,
        in_channels: int,
        kernel_size: int,
        depth: int,
        max_offset_span: int | None = None,
        max_offset_support: int | None = None,
        max_offset_multiplicity: int | None = None,
        scale: RawSignatureScale = "signature",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ScaledMonomialDictionary":
        """Build the scalar-compressed raw offset-retaining signature dictionary."""
        offsets, channels, coefficients = raw_signature_monomial_indices(
            in_channels=in_channels,
            kernel_size=kernel_size,
            depth=depth,
            max_offset_span=max_offset_span,
            max_offset_support=max_offset_support,
            max_offset_multiplicity=max_offset_multiplicity,
            scale=scale,
            device=device,
            dtype=dtype,
        )
        return cls(
            offsets=offsets,
            channels=channels,
            coefficients=coefficients,
            metadata={
                "kind": "raw_signature",
                "in_channels": in_channels,
                "kernel_size": kernel_size,
                "scale": scale,
                "max_offset_span": max_offset_span,
                "max_offset_support": max_offset_support,
                "max_offset_multiplicity": max_offset_multiplicity,
            },
        )
