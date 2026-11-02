"""Differentiable tensor-signature tools for PyTorch.

The public API is organized into domain modules:

- ``tensorsig.classical`` for path signatures and log-signatures;
- ``tensorsig.augmentations`` for composable path lifts;
- ``tensorsig.algebra`` for Hall and Lyndon basis utilities;
- ``tensorsig.tensor_ops`` for low-level tensor-algebra operations; and
- ``tensorsig.offset_coded_signatures`` for local coded features.
"""

from . import classical as classical
from . import augmentations as augmentations
from . import algebra as algebra
from . import tensor_ops as tensor_ops
from . import offset_coded_signatures as offset_coded_signatures

__all__ = [
    "classical",
    "augmentations",
    "algebra",
    "tensor_ops",
    "offset_coded_signatures",
]
