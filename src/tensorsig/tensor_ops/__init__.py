"""Public tensor-algebra operations organized by mathematical concept.

Products make shared leading axes explicit, linear operations transform or
contract homogeneous levels, and series operations implement multiplication,
inversion, flattening, and restricted exponentials. These are low-level
building blocks; path-level signature evaluation lives in
``tensorsig.classical``.
"""

from . import linear as linear
from . import products as products
from . import series as series


__all__ = ["products", "linear", "series"]
