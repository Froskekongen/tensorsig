"""Free Lie coordinates and Eulerian tensor-word projection.

Hall trees and Lyndon words are two coordinate systems for the same truncated
free Lie algebra. Their modules expose basis construction, dimensions, and
stable human-readable coordinate labels; numerical log-signature evaluation
lives in ``tensorsig.classical.signatures``.
"""

from . import hall as hall
from . import lyndon as lyndon
from . import eulerian as eulerian

__all__ = [
    "hall",
    "lyndon",
    "eulerian",
]
