"""Lyndon-word construction, dimensions, and coordinate labels.

Lyndon basis elements are canonical words over one-based generator labels.
The helpers use the ordering returned by word-mode log-signatures, making the
keys suitable for aligning model features with their algebraic coordinates.
"""

from .lyndon_words import logsigdim_words as logsigdim_words
from .lyndon_words import logsigkeys_words as logsigkeys_words
from .lyndon_words import lyndon_words as lyndon_words

__all__ = [
    "lyndon_words",
    "logsigdim_words",
    "logsigkeys_words",
]
