# %% [markdown]
# # Logsignature concepts
#
# A truncated signature is group-like: its tensor coordinates satisfy algebraic
# relations rather than varying independently. The logsignature is its formal
# tensor logarithm,
#
# \[
# \operatorname{LogSig}(X)=\log S(X),
# \]
#
# represented in a basis of the free Lie algebra. In plain language, it removes
# redundant tensor coordinates while preserving the same truncated object.
# Applying the corresponding truncated exponential reconstructs the signature.
#
# Level one remains displacement. At level two, the independent Lie coordinate
# measures the orientation of interactions between distinct channels.

# %%
import torch

from tensorsig.algebra.hall import logsigdim
from tensorsig.algebra.lyndon import logsigdim_words
from tensorsig.classical.signatures import log_signature, signature

path = torch.tensor(
    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]],
    dtype=torch.float64,
)
first_turn = path[:, :3]

sig = signature(first_turn, depth=2)
words = log_signature(first_turn, depth=2, mode="words")
hall = log_signature(first_turn, depth=2, mode="hall")

assert sig.shape == (1, 6)
assert words.shape == hall.shape == (1, 3)
assert logsigdim_words(2, 2) == logsigdim(2, 2) == 3
torch.testing.assert_close(
    words,
    torch.tensor([[1.0, 1.0, 0.5]], dtype=path.dtype),
)
torch.testing.assert_close(words, hall)

# %% [markdown]
# ## Hall and Lyndon-word coordinates
#
# `mode="words"` is the default and returns Lyndon-word coordinates.
# `mode="hall"` returns coordinates in a Hall basis. They are two bases of the
# same Lie element, related by a linear basis change; coordinate labels and
# values need not coincide beyond simple cases such as the example above.
#
# The reduction in size is visible already at depth two: the signature has six
# coordinates while either logsignature basis has three. Higher depths remove
# more tensor coordinates that are fixed by the group-like relations.
#
# ## Consequences for models
#
# The maps `signature -> logsignature` and `logsignature -> signature` use a
# truncated logarithm and exponential. They are nonlinear. Therefore a linear
# prediction head on signature coordinates and a linear head on logsignature
# coordinates express different model classes, even though both representations
# describe the same truncated path information.
#
# ## Guarantees and limitations
#
# Hall and Lyndon-word modes span the same truncated free Lie algebra. Choosing
# a basis is lossless at the selected depth; truncating at that depth is not.
# Conversion uses floating-point arithmetic, and the compact coordinates do not
# make a linear downstream model representation-invariant.
