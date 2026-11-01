# %% [markdown]
# # Using the API
#
# Classical signature functions accept a path of points with shape
# `(batch, points, channels)`. The output conventions are:
#
# | Operation | Output shape |
# | --- | --- |
# | Final signature | `(batch, signature_features)` |
# | Streaming signature | `(batch, points - 1, signature_features)` |
# | Final logsignature | `(batch, logsignature_features)` |
# | Streaming logsignature | `(batch, points - 1, logsignature_features)` |
# | Windowed operation | `(batch, windows, features)` |
#
# A streaming position is produced after each increment. Classical
# `window_size` counts **path points**, so a window of three points contains two
# increments. `hop_size` shifts the start by that many points.

# %%
import torch

from tensorsig.classical.signatures import log_signature, signature
from tensorsig.classical.sparse import pad_paths_correctly, signature_sparse
from tensorsig.classical.windows import windowed_log_signature, windowed_signature

path = torch.tensor(
    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]],
    dtype=torch.float64,
)

stream = signature(path, depth=2, stream=True)
sig_windows = windowed_signature(path, depth=2, window_size=3, hop_size=1)
log_windows = windowed_log_signature(path, depth=2, window_size=3, hop_size=1)

assert stream.shape == (1, 3, 6)
assert sig_windows.shape == (1, 2, 6)
assert log_windows.shape == (1, 2, 3)
torch.testing.assert_close(
    stream[:, 1],
    torch.tensor([[1.0, 1.0, 0.5, 1.0, 0.0, 0.5]], dtype=path.dtype),
)
torch.testing.assert_close(
    log_windows,
    torch.tensor([[[1.0, 1.0, 0.5], [1.0, 1.0, -0.5]]], dtype=path.dtype),
)

# %% [markdown]
# `windowed_signature` obtains windows from a streaming prefix signature using
# Chen removal. `stream_to_window_signatures` exposes that conversion when a
# prefix stream has already been computed.
#
# ## Logsignature options
#
# `log_signature` supports `mode="words"` (the default Lyndon-word basis) and
# `mode="hall"`. `method="default"` computes a signature, takes its tensor
# logarithm, and supports either mode. `method="bch_sparse"` uses incremental
# BCH where supported and requires Hall mode; for unsupported depths it falls
# back to the default Hall computation.
#
# `sparse=True` applies repeated-point compression in the default method.
# `eps` is the change threshold used only by sparse compression: increments
# whose largest absolute component is at most `eps` are treated as unchanged.

# %%
repeated = torch.cat((path[:, :2], path[:, 1:]), dim=1)
dense_repeated = signature(repeated, depth=2)
sparse_repeated = signature_sparse(repeated, depth=2)
hall_bch = log_signature(path[:, :3], depth=2, method="bch_sparse", mode="hall")

assert dense_repeated.shape == sparse_repeated.shape == (1, 6)
assert hall_bch.shape == (1, 3)
torch.testing.assert_close(dense_repeated, signature(path, depth=2))
torch.testing.assert_close(sparse_repeated, dense_repeated)
torch.testing.assert_close(
    hall_bch,
    torch.tensor([[1.0, 1.0, 0.5]], dtype=path.dtype),
)

# %% [markdown]
# For variable-length batches, repeat each final valid point. The padded tail
# then has zero increments and is signature-safe. `pad_paths_correctly` returns
# both the padded tensor and original lengths. Pass `lengths` when some other
# padding value could introduce a false final move.

# %%
short = path[0, :3]
long = path[0]
padded, lengths = pad_paths_correctly([short, long])
padded_sig = signature_sparse(padded, depth=2)

assert padded.shape == (2, 4, 2)
assert lengths.tolist() == [3, 4]
assert padded_sig.shape == (2, 6)
torch.testing.assert_close(padded_sig[0], signature(short.unsqueeze(0), 2)[0])

# %% [markdown]
# ## PyTorch behavior
#
# Outputs keep the input dtype and device. Operations are differentiable with
# respect to floating-point path values; move inputs to the desired device and
# dtype before evaluation.

# %%
differentiable_path = path.clone().requires_grad_()
loss = signature(differentiable_path, depth=2).square().sum()
loss.backward()

assert differentiable_path.grad is not None
assert differentiable_path.grad.shape == path.shape
assert signature(path, depth=2).dtype == torch.float64
assert torch.isfinite(differentiable_path.grad).all()

# %% [markdown]
# ## API reference
#
# - [`signature`](../api/classical/signatures/signature.md) and
#   [`log_signature`](../api/classical/signatures/log-signature.md)
# - [`windowed_signature`](../api/classical/windows/windowed-signature.md),
#   [`windowed_log_signature`](../api/classical/windows/windowed-log-signature.md),
#   and
#   [`stream_to_window_signatures`](../api/classical/windows/stream-to-window-signatures.md)
# - [`compress_path`](../api/classical/sparse/compress-path.md),
#   [`signature_sparse`](../api/classical/sparse/signature-sparse.md), and
#   [`pad_paths_correctly`](../api/classical/sparse/pad-paths-correctly.md)
#
# ## Guarantees and limitations
#
# Final-point padding and exact repeated points preserve the mathematical
# signature. A positive `eps` is deliberately approximate because it discards
# small changes. Window APIs return only complete valid windows; they do not pad
# a partial final window.
