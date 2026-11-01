"""Scalar monomial helpers for local increment windows."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, cast

import torch
from torch import Tensor


def _normalize_alphabet_order(
    alphabet_order: str,
) -> Literal["offset_major", "channel_major"]:
    normalized = (alphabet_order or "offset_major").lower()
    if normalized not in {"offset_major", "channel_major"}:
        raise ValueError(
            "alphabet_order must be either 'offset_major' or 'channel_major'."
        )
    return cast(Literal["offset_major", "channel_major"], normalized)


def chunk_slices(total: int, chunk_size: int | None) -> Iterator[slice]:
    """Yield slices covering ``total`` rows with optional chunking.

    Parameters
    ----------
    total : int
        Number of rows to cover.
    chunk_size : int, optional
        Maximum rows per chunk. Non-positive values disable chunking.

    Yields
    ------
    slice
        Slice selecting one contiguous row chunk.
    """
    if total < 0:
        raise ValueError("total must be non-negative.")
    if total == 0:
        return
    if chunk_size is None or chunk_size <= 0:
        chunk_size = total
    for start in range(0, total, chunk_size):
        yield slice(start, min(start + chunk_size, total))


def flatten_local_increment_letters(
    dxw: Tensor,
    alphabet_order: Literal["offset_major", "channel_major"] = "offset_major",
) -> Tensor:
    """Flatten local increments according to the requested local alphabet order.

    Parameters
    ----------
    dxw : Tensor
        Local increment windows with shape ``(batch, num_windows, kernel_size,
        channels)``.
    alphabet_order : {"offset_major", "channel_major"}, optional
        Flattening convention for the local alphabet.

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, num_windows, kernel_size * channels)``.
    """
    if dxw.ndim != 4:
        raise ValueError(f"Expected dxw shape (B, N, R, C), got {tuple(dxw.shape)}.")

    alphabet_order = _normalize_alphabet_order(alphabet_order)
    width = int(dxw.shape[2] * dxw.shape[3])
    if alphabet_order == "offset_major":
        return dxw.reshape(dxw.shape[0], dxw.shape[1], width)
    return dxw.permute(0, 1, 3, 2).reshape(
        dxw.shape[0],
        dxw.shape[1],
        width,
    )


def flatten_offset_channel_indices(
    offsets: Tensor,
    channels: Tensor,
    in_channels: int,
) -> Tensor:
    """Convert ``(offset, channel)`` index rows into flattened letters.

    Parameters
    ----------
    offsets : Tensor
        Long tensor with shape ``(num_features, depth)``.
    channels : Tensor
        Long tensor with shape ``(num_features, depth)``.
    in_channels : int
        Number of increment channels.

    Returns
    -------
    Tensor
        Flattened offset-major letters with shape ``(num_features, depth)``.
    """
    if offsets.ndim != 2 or channels.ndim != 2:
        raise ValueError("offsets and channels must have shape (M, depth).")
    if offsets.dtype != torch.long or channels.dtype != torch.long:
        raise ValueError("offsets and channels must have dtype torch.long.")
    if offsets.shape != channels.shape:
        raise ValueError("offsets and channels must have identical shapes.")
    if in_channels < 1:
        raise ValueError("in_channels must be positive.")
    return offsets * in_channels + channels


def validate_flat_letter_indices(
    letters: Tensor,
    width: int,
    name: str = "letters",
) -> tuple[int, int]:
    """Validate flattened local-alphabet letter indices.

    Parameters
    ----------
    letters : Tensor
        Long tensor with shape ``(num_features, depth)``.
    width : int
        Local alphabet width.
    name : str, optional
        Name to use in error messages.

    Returns
    -------
    tuple[int, int]
        ``(num_features, depth)``.
    """
    if letters.ndim != 2:
        raise ValueError(f"{name} must have shape (M, depth).")
    if letters.dtype != torch.long:
        raise ValueError(f"{name} must have dtype torch.long.")
    feature_count, depth = letters.shape
    if depth < 1:
        raise ValueError(f"{name} must contain positive-depth rows.")
    if feature_count > 0 and (torch.any(letters < 0) or torch.any(letters >= width)):
        raise ValueError(f"{name} contain entries outside the local window alphabet.")
    return int(feature_count), int(depth)


def local_monomial_features_from_flat(
    flat_dxw: Tensor,
    letters: Tensor,
    weights: Tensor | None = None,
    chunk_size: int | None = None,
) -> Tensor:
    """Compute scalar monomials from flattened local-window letters.

    Parameters
    ----------
    flat_dxw : Tensor
        Local increments with shape ``(batch, num_windows, width)``.
    letters : Tensor
        Long tensor of shape ``(M, depth)`` containing flattened local alphabet
        indices.
    weights : Tensor, optional
        Optional per-monomial weights with shape ``(M,)``.
    chunk_size : int, optional
        Number of monomials to materialize at a time.

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, num_windows, M)``.
    """
    if flat_dxw.ndim != 3:
        raise ValueError(
            f"Expected flat_dxw shape (B, N, width), got {tuple(flat_dxw.shape)}."
        )

    feature_count, _ = validate_flat_letter_indices(letters, width=flat_dxw.shape[-1])
    if weights is not None and (weights.ndim != 1 or weights.shape[0] != feature_count):
        raise ValueError("weights must have shape (M,).")
    if feature_count == 0:
        return torch.empty(
            *flat_dxw.shape[:2],
            0,
            device=flat_dxw.device,
            dtype=flat_dxw.dtype,
        )

    chunks = []
    for chunk in chunk_slices(feature_count, chunk_size):
        letter_chunk = letters[chunk].to(device=flat_dxw.device)
        feature_chunk = flat_dxw[:, :, letter_chunk].prod(dim=-1)
        if weights is not None:
            weight_chunk = weights[chunk].to(
                device=flat_dxw.device,
                dtype=flat_dxw.dtype,
            )
            feature_chunk = feature_chunk * weight_chunk.view(1, 1, -1)
        chunks.append(feature_chunk)
    return torch.cat(chunks, dim=2)


def local_monomial_features(
    dxw: Tensor,
    letters: Tensor,
    weights: Tensor | None = None,
    chunk_size: int | None = None,
) -> Tensor:
    """Compute scalar monomials from offset-major flattened letters.

    Parameters
    ----------
    dxw : Tensor
        Local increment windows with shape ``(batch, num_windows, kernel_size,
        channels)``.
    letters : Tensor
        Long tensor with shape ``(num_features, depth)``.
    weights : Tensor, optional
        Optional per-feature weights with shape ``(num_features,)``.
    chunk_size : int, optional
        Number of monomials to materialize at a time.

    Returns
    -------
    Tensor
        Tensor with shape ``(batch, num_windows, num_features)``.
    """
    flat_dxw = flatten_local_increment_letters(dxw, alphabet_order="offset_major")
    return local_monomial_features_from_flat(
        flat_dxw,
        letters,
        weights=weights,
        chunk_size=chunk_size,
    )


def local_monomial_features_from_offset_channels(
    dxw: Tensor,
    offsets: Tensor,
    channels: Tensor,
    weights: Tensor | None = None,
    chunk_size: int | None = None,
) -> Tensor:
    """Compute scalar monomials indexed by offset and channel rows.

    Parameters
    ----------
    dxw : Tensor
        Local increment windows with shape ``(batch, num_windows, kernel_size,
        channels)``.
    offsets : Tensor
        Long tensor with shape ``(num_features, depth)``.
    channels : Tensor
        Long tensor with shape ``(num_features, depth)``.
    weights : Tensor, optional
        Optional per-feature weights with shape ``(num_features,)``.
    chunk_size : int, optional
        Number of monomials to materialize at a time.

    Returns
    -------
    Tensor
        Tensor with shape ``(batch, num_windows, num_features)``.
    """
    if dxw.ndim != 4:
        raise ValueError(f"Expected dxw shape (B, N, R, C), got {tuple(dxw.shape)}.")

    _, _, kernel_size, in_channels = dxw.shape
    letters = flatten_offset_channel_indices(
        offsets,
        channels,
        in_channels=in_channels,
    )
    feature_count, _ = validate_flat_letter_indices(
        letters,
        width=kernel_size * in_channels,
    )
    if feature_count > 0:
        if torch.any(offsets < 0) or torch.any(offsets >= kernel_size):
            raise ValueError("offsets contain entries outside the local window.")
        if torch.any(channels < 0) or torch.any(channels >= in_channels):
            raise ValueError(
                "channels contain entries outside the input channel range."
            )
    return local_monomial_features(
        dxw,
        letters,
        weights=weights,
        chunk_size=chunk_size,
    )


def contract_monomial_readout_from_flat(
    flat_dxw: Tensor,
    readout: Tensor,
    letters: Tensor,
    weights: Tensor | None = None,
    chunk_size: int | None = None,
) -> Tensor:
    """Contract scalar monomials with a linear readout.

    Parameters
    ----------
    flat_dxw : Tensor
        Local increments with shape ``(batch, num_windows, width)``.
    readout : Tensor
        Readout tensor with shape ``(out_channels, num_features)``.
    letters : Tensor
        Long tensor with shape ``(num_features, depth)``.
    weights : Tensor, optional
        Optional per-feature weights with shape ``(num_features,)``.
    chunk_size : int, optional
        Number of monomials to contract at a time.

    Returns
    -------
    Tensor
        Tensor with shape ``(batch, num_windows, out_channels)``.
    """
    if flat_dxw.ndim != 3:
        raise ValueError(
            f"Expected flat_dxw shape (B, N, width), got {tuple(flat_dxw.shape)}."
        )
    if readout.ndim != 2:
        raise ValueError("readout must have shape (out_channels, feature_count).")

    feature_count, _ = validate_flat_letter_indices(letters, width=flat_dxw.shape[-1])
    if readout.shape[1] != feature_count:
        raise ValueError(
            f"readout has {readout.shape[1]} columns, expected {feature_count}."
        )
    if weights is not None and (weights.ndim != 1 or weights.shape[0] != feature_count):
        raise ValueError("weights must have shape (M,).")

    out = torch.zeros(
        flat_dxw.shape[0],
        flat_dxw.shape[1],
        readout.shape[0],
        device=flat_dxw.device,
        dtype=flat_dxw.dtype,
    )
    if feature_count == 0:
        return out

    readout = readout.to(device=flat_dxw.device, dtype=flat_dxw.dtype)
    for chunk in chunk_slices(feature_count, chunk_size):
        features = local_monomial_features_from_flat(
            flat_dxw,
            letters[chunk],
            weights=None if weights is None else weights[chunk],
            chunk_size=None,
        )
        out = out + torch.matmul(features, readout[:, chunk].T)
    return out


def contract_monomial_readout(
    dxw: Tensor,
    readout: Tensor,
    letters: Tensor,
    weights: Tensor | None = None,
    chunk_size: int | None = None,
) -> Tensor:
    """Contract offset-major scalar monomials with a linear readout.

    Parameters
    ----------
    dxw : Tensor
        Local increment windows with shape ``(batch, num_windows, kernel_size,
        channels)``.
    readout : Tensor
        Readout tensor with shape ``(out_channels, num_features)``.
    letters : Tensor
        Long tensor with shape ``(num_features, depth)``.
    weights : Tensor, optional
        Optional per-feature weights with shape ``(num_features,)``.
    chunk_size : int, optional
        Number of monomials to contract at a time.

    Returns
    -------
    Tensor
        Tensor with shape ``(batch, num_windows, out_channels)``.
    """
    flat_dxw = flatten_local_increment_letters(dxw, alphabet_order="offset_major")
    return contract_monomial_readout_from_flat(
        flat_dxw,
        readout,
        letters,
        weights=weights,
        chunk_size=chunk_size,
    )


def contract_monomial_readout_from_offset_channels(
    dxw: Tensor,
    readout: Tensor,
    offsets: Tensor,
    channels: Tensor,
    weights: Tensor | None = None,
    chunk_size: int | None = None,
) -> Tensor:
    """Contract offset/channel-indexed scalar monomials with a readout.

    Parameters
    ----------
    dxw : Tensor
        Local increment windows with shape ``(batch, num_windows, kernel_size,
        channels)``.
    readout : Tensor
        Readout tensor with shape ``(out_channels, num_features)``.
    offsets : Tensor
        Long tensor with shape ``(num_features, depth)``.
    channels : Tensor
        Long tensor with shape ``(num_features, depth)``.
    weights : Tensor, optional
        Optional per-feature weights with shape ``(num_features,)``.
    chunk_size : int, optional
        Number of monomials to contract at a time.

    Returns
    -------
    Tensor
        Tensor with shape ``(batch, num_windows, out_channels)``.
    """
    if dxw.ndim != 4:
        raise ValueError(f"Expected dxw shape (B, N, R, C), got {tuple(dxw.shape)}.")
    _, _, kernel_size, in_channels = dxw.shape
    letters = flatten_offset_channel_indices(
        offsets,
        channels,
        in_channels=in_channels,
    )
    feature_count, _ = validate_flat_letter_indices(
        letters,
        width=kernel_size * in_channels,
    )
    if feature_count > 0:
        if torch.any(offsets < 0) or torch.any(offsets >= kernel_size):
            raise ValueError("offsets contain entries outside the local window.")
        if torch.any(channels < 0) or torch.any(channels >= in_channels):
            raise ValueError(
                "channels contain entries outside the input channel range."
            )
    return contract_monomial_readout(
        dxw,
        readout,
        letters,
        weights=weights,
        chunk_size=chunk_size,
    )
