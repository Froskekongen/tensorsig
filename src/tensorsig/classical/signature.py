import torch
from torch import Tensor

from ..tensor_ops.products import tensor_product
from ..tensor_ops.series import (
    _infer_width_from_signature_dim,
    _unflatten_stream_signature,
    batch_restricted_exp,
    signature_inverse,
    signature_multiply,
)


def signature(
    path: Tensor,
    depth: int,
    stream: bool = False,
) -> Tensor:
    """Compute signatures for batched paths.

    The signature of a path is a collection of iterated integrals that captures
    the path's geometric properties. It is computed as a truncated tensor series
    up to the specified depth.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, dim)`` representing batched paths.
        For a single path, pass ``path.unsqueeze(0)`` to add a batch dimension.
    depth : int
        Maximum depth to truncate signature computation. The output dimension
        will be ``sum(dim**k for k in range(1, depth+1))``.
    stream : bool, optional
        If True, return signatures at each step along the path. If False,
        return only the final signature. Default is False.
    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, dim + dim² + ... + dim^depth)``
        containing the final signature for each path in the batch.

        If ``stream=True``: Tensor of shape ``(batch, length-1, dim + dim² + ... + dim^depth)``
        containing signatures at each step along each path.

    Raises
    ------
    ValueError
        If ``path`` is not three-dimensional.

    Examples
    --------
    >>> import torch
    >>> from tensorsig.classical.signatures import signature
    >>>
    >>> # Single path (add batch dimension)
    >>> path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
    >>> sig = signature(path, depth=2)
    >>> sig.shape
    torch.Size([1, 6])
    >>>
    >>> # Batched paths
    >>> batch_paths = torch.tensor([
    ...     [[0.0, 0.0], [1.0, 1.0]],
    ...     [[0.0, 0.0], [2.0, 2.0]],
    ... ])
    >>> sig = signature(batch_paths, depth=2)
    >>> sig.shape
    torch.Size([2, 6])
    >>>
    >>> # Streaming signatures
    >>> sig_stream = signature(path, depth=2, stream=True)
    >>> sig_stream.shape
    torch.Size([1, 2, 6])
    """
    if path.ndim != 3:
        msg = (
            f"Path must be of shape (batch, path_length, path_dim); got {path.shape}. "
            "Wrap a single path with path.unsqueeze(0)."
        )
        raise ValueError(msg)

    return _batch_signature(path, depth=depth, stream=stream)


def stream_to_window_signatures(
    signature: Tensor,
    depth: int,
    window_size: int,
    hop_size: int,
) -> Tensor:
    """Compute sliding-window signatures from a stream-computed signature.

    This function applies Chen's identity to a pre-computed stream of signatures
    to obtain signatures for sliding windows.

    Parameters
    ----------
    signature : Tensor
        Tensor of shape ``(batch, length-1, dim_sum)`` containing signatures
        at each step along the path, as returned by
        [`signature(..., stream=True)`][tensorsig.classical.signatures.signature].
    depth : int
        Maximum depth of the signatures.
    window_size : int
        Number of path points per window.
    hop_size : int
        Step between consecutive window starts.

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, num_windows, dim_sum)`` containing the
        signature of each window.
    """
    if signature.ndim != 3:
        raise ValueError(
            "Signature must be a 3D tensor of shape "
            f"(batch, length-1, sigdim) as returned by signature(..., stream=True); "
            f"got {signature.shape}."
        )

    batch_size, stream_len, sig_dim = signature.shape
    seq_len = stream_len + 1  # Original path length

    if window_size < 2:
        raise ValueError("window_size must be at least 2 to form non-empty increments.")
    if hop_size < 1:
        raise ValueError("hop_size must be positive.")
    if seq_len < window_size:
        raise ValueError("window_size cannot exceed the path length.")

    width = _infer_width_from_signature_dim(sig_dim, depth)

    prefix_levels = _unflatten_stream_signature(signature, width=width, depth=depth)
    device = signature.device
    dtype = signature.dtype
    num_windows = 1 + (seq_len - window_size) // hop_size

    # Insert the identity signature at time 0 (all higher levels zero).
    for idx, level in enumerate(prefix_levels):
        zeros_shape = (batch_size, 1) + (width,) * (idx + 1)
        prefix_levels[idx] = torch.cat(
            [torch.zeros(zeros_shape, device=device, dtype=dtype), level], dim=1
        )

    start_indices = torch.arange(num_windows, device=device) * hop_size
    end_indices = start_indices + window_size - 1

    # Gather start/end signatures for all windows and merge (batch, window) into a single axis.
    start_levels: list[Tensor] = []
    end_levels: list[Tensor] = []
    for level in prefix_levels:
        start_levels.append(level[:, start_indices, ...].reshape(-1, *level.shape[2:]))
        end_levels.append(level[:, end_indices, ...].reshape(-1, *level.shape[2:]))

    inv_start = signature_inverse(start_levels)
    window_levels = signature_multiply(inv_start, end_levels)

    # Reshape back to (batch, num_windows, ...) and flatten level blocks.
    flattened = []
    for idx, level in enumerate(window_levels):
        reshaped = level.reshape(batch_size, num_windows, -1)
        flattened.append(reshaped)

    return torch.cat(flattened, dim=2)


def window_signature_levels_from_stream_levels(
    stream_levels: list[Tensor],
    window_size_increments: int,
    hop_size: int,
) -> tuple[list[Tensor], Tensor]:
    """Extract window signature levels from streaming prefix levels.

    Parameters
    ----------
    stream_levels : list[Tensor]
        List where level ``m - 1`` has shape ``(batch, increments, width, ..., width)``.
        Prefix index ``t`` represents increments ``0, ..., t`` as returned by
        ``signature(..., stream=True)``.
    window_size_increments : int
        Number of increments in each local window.
    hop_size : int
        Step between consecutive window starts, measured in increments.

    Returns
    -------
    tuple[list[Tensor], Tensor]
        Window levels and start indices. Level ``m - 1`` has shape
        ``(batch, num_windows, width, ..., width)``. Starts has shape
        ``(num_windows,)``.
    """
    if not stream_levels:
        raise ValueError("stream_levels must be non-empty.")

    if window_size_increments < 1:
        raise ValueError("window_size_increments must be at least 1.")
    if hop_size < 1:
        raise ValueError("hop_size must be positive.")

    first = stream_levels[0]
    if first.ndim < 3:
        raise ValueError("stream_levels[0] must have shape (batch, increments, width).")

    batch_size, stream_len = first.shape[:2]
    if stream_len < window_size_increments:
        raise ValueError(
            "window_size_increments cannot exceed the number of stream increments."
        )

    for depth_index, level in enumerate(stream_levels):
        expected_ndim = 2 + depth_index + 1
        if level.ndim != expected_ndim:
            raise ValueError(
                f"Expected stream_levels[{depth_index}] to have {expected_ndim} "
                f"dimensions, got {level.ndim}."
            )
        if level.shape[:2] != (batch_size, stream_len):
            raise ValueError("All stream levels must share batch and stream axes.")

    num_windows = 1 + (stream_len - window_size_increments) // hop_size
    starts = torch.arange(num_windows, device=first.device) * hop_size
    ends = starts + window_size_increments

    prefix_levels: list[Tensor] = []
    for level in stream_levels:
        zeros_shape = (batch_size, 1, *level.shape[2:])
        zeros = torch.zeros(zeros_shape, device=level.device, dtype=level.dtype)
        prefix_levels.append(torch.cat([zeros, level], dim=1))

    start_levels: list[Tensor] = []
    end_levels: list[Tensor] = []
    for level in prefix_levels:
        start_levels.append(level[:, starts, ...].reshape(-1, *level.shape[2:]))
        end_levels.append(level[:, ends, ...].reshape(-1, *level.shape[2:]))

    inv_start = signature_inverse(start_levels)
    window_flat_levels = signature_multiply(inv_start, end_levels)
    window_levels = [
        level.reshape(batch_size, num_windows, *level.shape[1:])
        for level in window_flat_levels
    ]
    return window_levels, starts


def windowed_signature(
    path: Tensor,
    depth: int,
    window_size: int,
    hop_size: int,
) -> Tensor:
    """Sliding-window signatures using Chen's identity.

    Each window signature is computed from streaming prefix signatures:
    ``Sig(path[s:e]) = Sig(path[:s])^{-1} ⊗ Sig(path[:e])`` where ``e = s + window_size - 1``.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, dim)`` representing batched paths.
    depth : int
        Maximum depth to truncate signature computation.
    window_size : int
        Number of path points per window.
    hop_size : int
        Step between consecutive window starts (``>=1``).

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, num_windows, dim + dim² + ... + dim^depth)``,
        where ``num_windows = 1 + (length - window_size) // hop_size``,
        containing the signature of each window, flattened level-wise.

    Raises
    ------
    ValueError
        If ``path`` is not three-dimensional or if windowing parameters are invalid.

    Notes
    -----
    - Implements the Signatory-style streaming reuse (Chen) without materializing
      every window explicitly.
    - Provides the building block for
      [`windowed_log_signature`][tensorsig.classical.windows.windowed_log_signature];
      both share identical window indexing and batching semantics.
    """
    if path.ndim != 3:
        msg = (
            f"Path must be of shape (batch, path_length, path_dim); got {path.shape}. "
            "Wrap a single path with path.unsqueeze(0)."
        )
        raise ValueError(msg)

    batch_size, seq_len, width = path.shape

    if window_size < 2:
        raise ValueError("window_size must be at least 2 to form non-empty increments.")
    if hop_size < 1:
        raise ValueError("hop_size must be positive.")
    if seq_len < window_size:
        raise ValueError("window_size cannot exceed the path length.")

    stream = signature(path, depth=depth, stream=True)

    return stream_to_window_signatures(stream, depth, window_size, hop_size)


def _batch_signature(
    path: Tensor,
    depth: int,
    stream: bool = False,
) -> Tensor:
    """Compute batched signatures using the fast parallel implementation.

    A memory-intensive but computationally efficient implementation that:
    - Replaces sequential scan operations with parallel tensor operations
    - Pre-computes path increment divisions
    - Uses cumulative sums for parallel sequence processing
    - Trades increased memory usage for reduced sequential operations

    Best suited when:
    - Memory can accommodate larger intermediate tensors
    - Batch/sequence sizes benefit from parallel processing
    - Computation speed is prioritized over memory efficiency

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, seq_len, features)`` representing batched paths.
    depth : int
        Maximum signature truncation depth.
    stream : bool, optional
        If True, returns signatures at each timestep. Default is False.

    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, features + features^2 + ... + features^depth)``
        containing the final signature for each path.

        If ``stream=True``: Tensor of shape ``(batch, seq_len-1, features + features^2 + ... + features^depth)``
        containing signatures at each timestep.

    Examples
    --------
    >>> import torch
    >>> from tensorsig.classical.signature import _batch_signature
    >>>
    >>> path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
    >>> sig = _batch_signature(path, depth=2)
    >>> sig.shape
    torch.Size([1, 6])
    """
    path_increments = torch.diff(path, dim=1)  # Shape: (batch, seq_len-1, features)
    return _batch_signature_from_increments(
        path_increments,
        depth=depth,
        stream=stream,
    )


def _batch_signature_from_increments(
    path_increments: Tensor,
    depth: int,
    stream: bool = False,
) -> Tensor:
    """Compute batched signatures directly from exact path increments.

    Parameters
    ----------
    path_increments : Tensor
        Increment tensor with shape ``(batch, steps, features)``.
    depth : int
        Positive signature truncation depth.
    stream : bool, optional
        Return the signature after every increment when true.

    Returns
    -------
    Tensor
        Final flattened signatures with shape ``(batch, signature_dimension)``
        or streaming signatures with shape
        ``(batch, steps, signature_dimension)``.
    """
    if path_increments.ndim != 3:
        raise ValueError("path_increments must have shape (batch, steps, features).")
    batch_size, steps, n_features = path_increments.shape
    if steps < 1:
        raise ValueError("path_increments must contain at least one step.")

    stacked = [torch.cumsum(path_increments, dim=1)]

    exp_term = batch_restricted_exp(path_increments[:, 0], depth=depth)

    if depth > 1:
        path_increment_divided = torch.stack(
            [path_increments / i for i in range(2, depth + 1)], dim=0
        )

        for depth_index in range(1, depth):
            current = (
                stacked[0][:, :-1] + path_increment_divided[depth_index - 1, :, 1:]
            )
            for j in range(depth_index - 1):
                current = stacked[j + 1][:, :-1] + tensor_product(
                    current,
                    path_increment_divided[depth_index - j - 2, :, 1:],
                    shared_dims=2,
                )
            current = tensor_product(current, path_increments[:, 1:], shared_dims=2)
            current = torch.cat([exp_term[depth_index].unsqueeze(1), current], dim=1)
            stacked.append(torch.cumsum(current, dim=1))

    if not stream:
        return torch.cat(
            [
                c[:, -1].reshape(batch_size, n_features ** (1 + idx))
                for idx, c in enumerate(stacked)
            ],
            dim=1,
        )
    else:
        return torch.cat(
            [
                r.reshape(batch_size, steps, n_features ** (1 + idx))
                for idx, r in enumerate(stacked)
            ],
            dim=2,
        )
