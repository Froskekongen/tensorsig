"""Sliding-window signatures and log-signatures.

Window sizes count path points, while hops count the shift between successive
point windows. The helpers either evaluate windows directly or convert a
streaming prefix signature into valid window signatures using Chen removal.
"""

from .log_signature import windowed_log_signature as windowed_log_signature
from .signature import stream_to_window_signatures as stream_to_window_signatures
from .signature import windowed_signature as windowed_signature

__all__ = [
    "windowed_signature",
    "windowed_log_signature",
    "stream_to_window_signatures",
]
