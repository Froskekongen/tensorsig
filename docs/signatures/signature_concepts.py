# %% [markdown]
# # Signature concepts
#
# This page fixes the notation and output ordering used by `tensorsig`.
# A batched path has shape `(B, T, C)`: `B` paths, `T` sampled points, and `C`
# channels. Consecutive points define `T - 1` increments
#
# \[
# \Delta x_t = x_{t+1}-x_t.
# \]
#
# In plain language, the signature depends on the ordered moves between points,
# not on the absolute starting position.
#
# ## Levels and ordering
#
# At truncation depth `M`, `signature` returns levels one through `M`. Level
# `m` has `C^m` coordinates and the flattened feature count is
#
# \[
# \sum_{m=1}^{M} C^m.
# \]
#
# Level one is total displacement. Level two records ordered pairs of channel
# movements. With two channels its flattened order is `(11, 12, 21, 22)`.
# The antisymmetric difference between `12` and `21` carries signed-area
# intuition: it distinguishes going right-then-up from up-then-right.

# %%
import torch

from tensorsig.classical.signatures import signature

path = torch.tensor(
    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]],
    dtype=torch.float64,
)
first_turn = path[:, :3]
features = signature(first_turn, depth=2)

level_one = features[:, :2]
level_two = features[:, 2:].reshape(1, 2, 2)

assert features.shape == (1, 6)
torch.testing.assert_close(
    level_one,
    torch.tensor([[1.0, 1.0]], dtype=path.dtype),
)
torch.testing.assert_close(
    level_two,
    torch.tensor([[[0.5, 1.0], [0.0, 0.5]]], dtype=path.dtype),
)
assert ((level_two[:, 0, 1] - level_two[:, 1, 0]) / 2).item() == 0.5

# %% [markdown]
# The path therefore has displacement `(1, 1)`, ordered interaction `12 = 1`,
# and `21 = 0`. The signed antisymmetric contribution is `1/2`.
#
# ## Invariances and concatenation
#
# Adding the same vector to every point leaves every increment unchanged, so it
# leaves the signature unchanged. Repeating a point inserts a zero increment
# and also leaves the final signature unchanged.

# %%
translated = first_turn + torch.tensor([10.0, -4.0], dtype=path.dtype)
with_repeat = torch.cat((first_turn[:, :2], first_turn[:, 1:]), dim=1)

translated_sig = signature(translated, depth=2)
repeated_sig = signature(with_repeat, depth=2)

assert translated_sig.shape == features.shape
assert repeated_sig.shape == features.shape
torch.testing.assert_close(translated_sig, features)
torch.testing.assert_close(repeated_sig, features)

# %% [markdown]
# If a path is split into consecutive pieces `X` and `Y`, Chen concatenation
# states
#
# \[
# S(X * Y)=S(X)\otimes S(Y).
# \]
#
# In plain language, the signature of the whole path can be assembled in order
# from the signatures of its pieces. This identity is what makes streaming and
# efficient window removal possible.
#
# ## Guarantees and limitations
#
# Translation and exact repeated-point invariance follow from using increments.
# Chen concatenation holds for the represented truncated tensor series. A
# truncated signature is intentionally incomplete above depth `M`, and sampled
# points are interpreted through their piecewise-linear segments.
