"""Signatures of paths with repeated points.

Consecutive repeated points contribute zero increments. These helpers expose
the compression and padding conventions used to evaluate such paths without
changing their signatures.
"""

from .sparse_signature import compress_path as compress_path
from .sparse_signature import pad_paths_correctly as pad_paths_correctly
from .sparse_signature import signature_sparse as signature_sparse

__all__ = [
    "compress_path",
    "signature_sparse",
    "pad_paths_correctly",
]
