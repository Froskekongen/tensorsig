"""Full-path signatures and log-signatures.

A signature stores truncated iterated-integral tensors and obeys Chen
concatenation. A log-signature stores the corresponding primitive Lie element
in Hall or Lyndon-word coordinates. Inputs are batched path points; ``depth``
is the highest retained homogeneous degree.
"""

from .log_signature import log_signature as log_signature
from .signature import signature as signature

__all__ = [
    "signature",
    "log_signature",
]
