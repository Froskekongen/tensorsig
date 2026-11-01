"""Classical path signatures, log-signatures, and window transforms.

The classical API treats a path as batched points with shape
``(batch, length, channels)``. Full-path coordinates live in ``signatures``,
overlapping-window transforms in ``windows``, and repeated-point compression
in ``sparse``. All computations remain native
PyTorch operations and are differentiable with respect to the path.
"""

from . import signatures as signatures
from . import windows as windows
from . import sparse as sparse

__all__ = [
    "signatures",
    "windows",
    "sparse",
]
