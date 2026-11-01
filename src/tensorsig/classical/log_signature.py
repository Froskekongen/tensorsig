"""Log-signature computation in Hall and Lyndon (\"words\") bases.

This module provides functions to compute the log-signature of a path, with
two coordinate systems:

- Lyndon \"words\" basis (default): Signatory-style ordering where each coefficient
  is the tensor-log coefficient of a Lyndon word; projection reduces to gathers.
- Hall basis: traditional Hall set ordering.

Both bases represent the same free Lie algebra element; a linear change of
basis relates their coordinates.
"""

import torch
from torch import Tensor

from ..algebra.hall_bch import HallBCH, sparse_bch_supports_depth
from ..algebra.hall_projection import _project_to_hall_basis
from ..algebra.lyndon_words import (
    _project_to_words_basis,
)
from .signature import (
    _batch_signature_from_increments,
    signature,
    windowed_signature,
)
from .sparse_signature import signature_sparse
from ..tensor_ops.series import (
    _infer_width_from_signature_dim,
    _signature_to_logsignature_tensor,
    _unflatten_signature,
)


def _batch_log_signature_from_increments(
    increments: Tensor,
    depth: int,
    stream: bool = False,
    mode: str = "words",
) -> Tensor:
    """Compute log-signatures without reconstructing and differencing a path.

    Parameters
    ----------
    increments : Tensor
        Exact path increments with shape ``(batch, steps, width)``.
    depth : int
        Positive truncation depth.
    stream : bool, optional
        Return a log-signature after every increment when true.
    mode : {"words", "hall"}, optional
        Output coordinate basis.

    Returns
    -------
    Tensor
        Final coordinates with shape ``(batch, dimension)`` or streaming
        coordinates with shape ``(batch, steps, dimension)``.
    """
    mode = (mode or "words").lower()
    if mode not in {"hall", "words"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'hall' or 'words'.")
    if increments.ndim != 3:
        raise ValueError("increments must have shape (batch, steps, width).")
    batch_size, steps, width = increments.shape
    sig = _batch_signature_from_increments(
        increments,
        depth=depth,
        stream=stream,
    )
    projector = _project_to_hall_basis if mode == "hall" else _project_to_words_basis

    if not stream:
        sig_tensors = _unflatten_signature(sig, width, depth)
        log_tensors = _signature_to_logsignature_tensor(sig_tensors, width, depth)
        return projector(log_tensors, width, depth)

    flattened = sig.reshape(batch_size * steps, -1)
    sig_tensors = _unflatten_signature(flattened, width, depth)
    log_tensors = _signature_to_logsignature_tensor(sig_tensors, width, depth)
    return projector(log_tensors, width, depth).reshape(batch_size, steps, -1)


def _batch_log_signature(
    path: Tensor,
    depth: int,
    stream: bool = False,
    mode: str = "words",
    sparse: bool = False,
    eps: float = 0.0,
    lengths: Tensor | None = None,
) -> Tensor:
    """Compute log-signatures via signature→log pipeline for batched paths.

    This implementation computes the truncated signature first, converts it to
    a tensor-log via the formal logarithm series, and then projects to either
    Hall or Lyndon (\"words\") coordinates.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, dim)`` representing batched paths.
    depth : int
        Maximum depth to truncate log-signature computation.
    stream : bool, optional
        If True, computed log-signatures are returned for each step. Default is False.
    mode : str, optional
        Basis for the output coordinates: ``"words"`` (default) or ``"hall"``.
    sparse : bool, optional
        If True, use sparse signature computation for paths with repeated points.
        Default is False.
    eps : float, optional
        Threshold for change detection when using sparse mode. Default is 0.0.
    lengths : Tensor, optional
        Tensor of shape ``(batch,)`` with valid lengths for padded batches when
        using sparse mode.

        **Best practice (recommended)**: pad by repeating the last valid point of
        each path (signature-safe padding). In that case, the padded tail has
        zero increments and does not change the signature/log-signature, so you
        can usually leave ``lengths=None``.

        If you pad with zeros/any other values, pass ``lengths`` so the sparse
        path compression can ignore the padded tail.

        Default is None.

    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, D)`` where
        ``D = logsigdim`` for Hall or ``logsigdim_words`` for words mode.

        If ``stream=True``: Tensor of shape ``(batch, length-1, D)`` with the
        same ``D`` definition as above.

    Notes
    -----
    This is the default log-signature computation method. It works for any depth
    and remains preferable when the ambient Hall bracket table would be costly.
    """
    mode = (mode or "words").lower()
    if mode not in {"hall", "words"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'hall' or 'words'.")
    batch_size, seq_len, n_features = path.shape

    if sparse:
        sig = signature_sparse(
            path,
            depth=depth,
            stream=stream,
            eps=eps,
            lengths=lengths,
        )
    else:
        sig = signature(
            path,
            depth=depth,
            stream=stream,
        )

    projector = _project_to_hall_basis if mode == "hall" else _project_to_words_basis

    if not stream:
        sig_tensors = _unflatten_signature(sig, n_features, depth)
        log_sig_tensors = _signature_to_logsignature_tensor(
            sig_tensors, n_features, depth
        )
        return projector(log_sig_tensors, n_features, depth)

    flattened = sig.reshape(batch_size * (seq_len - 1), -1)
    sig_tensors = _unflatten_signature(flattened, n_features, depth)
    log_sig_tensors = _signature_to_logsignature_tensor(sig_tensors, n_features, depth)
    log_sig = projector(log_sig_tensors, n_features, depth)
    return log_sig.reshape(batch_size, seq_len - 1, -1)


def _batch_log_signature_bch(
    path: Tensor,
    depth: int,
    stream: bool = False,
) -> Tensor:
    """Compute a log-signature via incremental BCH in Hall coordinates.

    This avoids materializing the full tensor-algebra signature and
    leverages the fact that each path increment lives in the degree-1
    component of the free Lie algebra. Its value is workload-dependent;
    the vectorized signature→log path is usually faster in the ambient
    free Lie algebra, while sparse BCH is useful for compact quotients.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, width)`` representing batched paths.
    depth : int
        Positive truncation depth for the log-signature.
    stream : bool, optional
        If True, return log-signatures at each step. Default is False.

    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, logsigdim(width, depth))``
        containing the final log-signature for each path.

        If ``stream=True``: Tensor of shape ``(batch, length-1, logsigdim(width, depth))``
        containing log-signatures at each step.

    Notes
    -----
    This method uses the Baker-Campbell-Hausdorff formula to incrementally
    update the log-signature. Compact formulas are used through depth four;
    higher depths use a universal Dynkin expansion and can be substantially
    slower for an ambient free Lie algebra.

    Examples
    --------
    >>> import torch
    >>> from tensorsig.classical.log_signature import _batch_log_signature_bch
    >>>
    >>> path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
    >>> log_sig = _batch_log_signature_bch(path, depth=2)
    >>> log_sig.shape
    torch.Size([1, 3])
    """
    batch_size, seq_len, width = path.shape
    increments = torch.diff(path, dim=1)
    bch = HallBCH(width=width, depth=depth, device=path.device, dtype=path.dtype)
    steps = increments.shape[1]

    # Vectorize embedding of increments into Hall coordinates.
    hall_increments = torch.zeros(
        batch_size,
        steps,
        bch.dim,
        device=path.device,
        dtype=path.dtype,
    )
    hall_increments[:, :, :width] = increments

    state = torch.zeros(batch_size, bch.dim, device=path.device, dtype=path.dtype)
    if not stream:
        for step in range(steps):
            state = bch.bch(state, hall_increments[:, step])
        return state

    history = []
    for step in range(steps):
        state = bch.bch(state, hall_increments[:, step])
        history.append(state)
    return torch.stack(history, dim=1)


def log_signature(
    path: Tensor,
    depth: int,
    stream: bool = False,
    method: str = "default",
    mode: str = "words",
    sparse: bool = False,
    eps: float = 0.0,
    lengths: Tensor | None = None,
) -> Tensor:
    """Compute log-signatures for batched paths.

    The log-signature is a compressed representation of the signature. Two bases are
    supported:

    - ``mode=\"words\"`` (default): Signatory-style Lyndon words basis (triangular/gather projection)
    - ``mode=\"hall\"``: classic Hall basis

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, dim)`` representing batched paths.
        For a single path, pass ``path.unsqueeze(0)`` to add a batch dimension.
    depth : int
        Maximum depth to truncate log-signature computation. The output dimension
        is ``logsigdim(dim, depth)`` for ``mode="hall"`` and
        ``logsigdim_words(dim, depth)`` for ``mode="words"``.
    stream : bool, optional
        If True, computed log-signatures are returned for each step. Default is False.
    method : str, optional
        Computation method: "default" (signature then log) or "bch_sparse"
        (sparse Hall-BCH). The default remains signature then log because the
        ambient Hall bracket table and higher-order BCH expansion can be more
        expensive than tensor-algebra computation. Default is "default".
    mode : str, optional
        Basis for the log-signature coordinates: "words" (default) or "hall".
        "words" is only available with ``method=\"default\"``.
    sparse : bool, optional
        If True, use sparse signature computation for paths with repeated points.
        Only applies when ``method=\"default\"``. Default is False.
    eps : float, optional
        Threshold for change detection when using sparse mode. Default is 0.0.
    lengths : Tensor, optional
        Tensor of shape ``(batch,)`` with valid lengths for padded batches when
        using sparse mode. Default is None.

    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, D)`` where
        ``D = logsigdim(dim, depth)`` for ``mode=\"hall\"`` and
        ``D = logsigdim_words(dim, depth)`` for ``mode=\"words\"``.

        If ``stream=True``: Tensor of shape ``(batch, length-1, D)`` with
        the same ``D`` definition as above.

    Raises
    ------
    ValueError
        If ``path`` is not three-dimensional, if ``method`` is not
        "default" or "bch_sparse", or if an unsupported ``mode``/``method``
        combination is requested.

    Examples
    --------
    >>> import torch
    >>> from tensorsig.algebra.hall import logsigdim
    >>> from tensorsig.classical.signatures import log_signature
    >>> from tensorsig.algebra.lyndon import logsigdim_words
    >>>
    >>> # Single path (add batch dimension)
    >>> path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
    >>> log_sig = log_signature(path, depth=2)
    >>> log_sig.shape
    torch.Size([1, 3])
    >>> logsigdim_words(2, 2)
    3
    >>>
    >>> # Batched paths
    >>> batch_paths = torch.tensor([
    ...     [[0.0, 0.0], [1.0, 1.0]],
    ...     [[0.0, 0.0], [2.0, 2.0]],
    ... ])
    >>> log_sig = log_signature(batch_paths, depth=2)
    >>> log_sig.shape
    torch.Size([2, 3])
    >>>
    >>> # Streaming log-signatures
    >>> log_sig_stream = log_signature(path, depth=2, stream=True)
    >>> log_sig_stream.shape
    torch.Size([1, 2, 3])
    >>>
    >>> # Using the direct BCH method in Hall coordinates
    >>> log_sig_bch = log_signature(path, depth=2, method="bch_sparse", mode="hall")
    >>> log_sig_bch.shape
    torch.Size([1, 3])
    """
    if path.ndim != 3:
        msg = (
            f"Path must be of shape (batch, path_length, path_dim); got {path.shape}. "
            "Wrap a single path with path.unsqueeze(0)."
        )
        raise ValueError(msg)

    mode = (mode or "words").lower()
    if mode not in {"hall", "words"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'hall' or 'words'.")

    method = (method or "default").lower()
    if method == "bch_sparse":
        if mode != "hall":
            raise ValueError("mode='words' is only supported with method='default'.")
        if not sparse_bch_supports_depth(depth):
            log_sig = _batch_log_signature(
                path,
                depth=depth,
                stream=stream,
                mode=mode,
                sparse=sparse,
                eps=eps,
                lengths=lengths,
            )
        else:
            log_sig = _batch_log_signature_bch(
                path,
                depth=depth,
                stream=stream,
            )
    elif method in {"default", None}:
        log_sig = _batch_log_signature(
            path,
            depth=depth,
            stream=stream,
            mode=mode,
            sparse=sparse,
            eps=eps,
            lengths=lengths,
        )
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'default' or 'bch_sparse'."
        )

    return log_sig


def windowed_log_signature(
    path: Tensor,
    depth: int,
    window_size: int,
    hop_size: int,
    mode: str = "words",
) -> Tensor:
    """Sliding-window log-signatures via windowed signatures + projection.

    Windows are formed with ``size=window_size`` and ``step=hop_size``.
    Signatures are obtained using
    [`windowed_signature`][tensorsig.classical.windows.windowed_signature]
    (Chen reuse of streaming prefixes), then converted to log-signatures and
    projected to the requested basis.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, dim)`` representing batched paths.
        For a single path, pass ``path.unsqueeze(0)`` to add a batch dimension.
    depth : int
        Maximum depth to truncate log-signature computation. The output dimension
        is ``logsigdim(dim, depth)`` for ``mode="hall"`` and
        ``logsigdim_words(dim, depth)`` for ``mode="words"``.
    window_size : int
        Size of the window to use for the signature.
    hop_size : int
        Hop size to use for the signature.
    mode : str, optional
        Basis for the log-signature coordinates: "words" (default) or "hall".
        "words" is only available with ``method=\"default\"``.

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, num_windows, D)`` where
        ``num_windows = 1 + (length - window_size) // hop_size`` and
        ``D = logsigdim_words(dim, depth)`` if ``mode=\"words\"`` else
        ``logsigdim(dim, depth)`` for Hall.

    Raises
    ------
    ValueError
        If ``path`` is not three-dimensional or if ``mode`` is unsupported.
    """
    if path.ndim != 3:
        msg = (
            f"Path must be of shape (batch, path_length, path_dim); got {path.shape}. "
            "Wrap a single path with path.unsqueeze(0)."
        )
        raise ValueError(msg)

    mode = (mode or "words").lower()
    if mode not in {"hall", "words"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'hall' or 'words'.")

    batch, _, width = path.shape

    # Reuse efficient streaming→windowed signature computation.
    window_sig = windowed_signature(
        path,
        depth=depth,
        window_size=window_size,
        hop_size=hop_size,
    )  # (batch, num_windows, sigdim)

    batch_windows, num_windows = batch, window_sig.shape[1]

    flattened = window_sig.reshape(batch_windows * num_windows, -1)
    sig_tensors = _unflatten_signature(flattened, width, depth)
    log_sig_tensors = _signature_to_logsignature_tensor(sig_tensors, width, depth)

    projector = _project_to_hall_basis if mode == "hall" else _project_to_words_basis
    projected = projector(log_sig_tensors, width, depth)

    return projected.reshape(batch_windows, num_windows, -1)


def signature_to_logsignature(
    signature: Tensor,
    depth: int,
    mode: str = "words",
) -> Tensor:
    """Convert signature to log-signature.

    Parameters
    ----------
    signature: Tensor
        Signature tensor with arbitrary leading batch/window dimensions and
        trailing dimension equal to the flattened signature size
        ``width + width^2 + ... + width^depth``. This includes outputs from
        :func:`signature`, :func:`windowed_signature`, or any precomputed
        flattened signature with that final dimension.
    depth: int
        Depth of the log-signature.
    mode: str, optional
        Basis for the log-signature coordinates: "words" (default) or "hall".

    Returns
    -------
    Tensor
        Log-signature tensor with the same leading shape as ``signature`` but
        with the last dimension replaced by the log-signature dimension
        (``logsigdim`` or ``logsigdim_words`` depending on ``mode``).
    """
    mode = (mode or "words").lower()
    if mode not in {"hall", "words"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'hall' or 'words'.")

    if signature.ndim < 2:
        raise ValueError(
            "signature tensor must have at least one batch dimension "
            "and a trailing signature dimension."
        )

    sigdim = signature.shape[-1]
    width = _infer_width_from_signature_dim(sigdim, depth)

    leading_shape = signature.shape[:-1]
    flat = signature.reshape(-1, sigdim)

    sig_tensors = _unflatten_signature(flat, width, depth)
    log_sig_tensors = _signature_to_logsignature_tensor(sig_tensors, width, depth)

    projector = _project_to_hall_basis if mode == "hall" else _project_to_words_basis
    projected = projector(log_sig_tensors, width, depth)

    return projected.reshape(*leading_shape, projected.shape[-1])
