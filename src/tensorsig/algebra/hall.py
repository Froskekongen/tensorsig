"""Hall-basis construction, dimensions, and coordinate labels.

Hall basis elements are nested bracket trees with one-based generator labels.
The dimension and key helpers use the same deterministic ordering as Hall-mode
log-signatures.
"""

from .hall_projection import hall_basis as hall_basis
from .hall_projection import logsigdim as logsigdim
from .hall_projection import logsigkeys as logsigkeys

__all__ = [
    "hall_basis",
    "logsigdim",
    "logsigkeys",
]
