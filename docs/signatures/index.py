# %% [markdown]
# # Signatures
#
# A path signature turns an ordered sequence into features of its increments.
# It records displacement first, then ordered interactions, then progressively
# higher-order interactions up to a chosen depth. A logsignature describes the
# same truncated path information in fewer, Lie-algebra coordinates.
#
# This guide assumes basic familiarity with signatures, PyTorch, and tensor
# shapes. It focuses on the conventions and choices needed to use `tensorsig`.
#
# ## Quick start

# %%
import torch

from tensorsig.classical.signatures import log_signature, signature

path = torch.tensor(
    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]],
    dtype=torch.float64,
)
first_turn = path[:, :3]

sig = signature(first_turn, depth=2)
logsig = log_signature(first_turn, depth=2)

assert sig.shape == (1, 6)
assert logsig.shape == (1, 3)
torch.testing.assert_close(
    sig,
    torch.tensor([[1.0, 1.0, 0.5, 1.0, 0.0, 0.5]], dtype=path.dtype),
)
torch.testing.assert_close(
    logsig,
    torch.tensor([[1.0, 1.0, 0.5]], dtype=path.dtype),
)

# %% [markdown]
# Inputs have shape `(batch, points, channels)`. Here the first two output
# coordinates are the displacement `(1, 1)`. The remaining signature values
# are the flattened second level; the final logsignature coordinate records the
# orientation of the turn.
#
# ## Which representation should I use?
#
# | Need | Start with |
# | --- | --- |
# | Direct tensor coordinates or Chen multiplication | Signature |
# | Fewer independent coordinates | Logsignature |
# | A prefix feature after every increment | Streaming signature or logsignature |
# | One feature per overlapping point window | Windowed signature or logsignature |
# | Relative position inside a local increment window | [Offset-coded signatures](../offset_coded_signatures/index.md) |
#
# A linear model on signatures is not generally equivalent to a linear model on
# logsignatures: converting between the two representations is nonlinear.
#
# Continue with:
#
# - [Signature concepts](signature_concepts.md)
# - [Logsignature concepts](logsignature_concepts.md)
# - [Using the API](using_the_api.md)
#
# ## Guarantees and limitations
#
# The classical operations compute truncated piecewise-linear signatures with
# PyTorch tensor arithmetic and support autograd. Truncation deliberately drops
# levels above `depth`; finite precision still applies. Coordinate ordering,
# shapes, window conventions, and supported options are part of the current
# public implementation and are demonstrated by executable assertions here.
