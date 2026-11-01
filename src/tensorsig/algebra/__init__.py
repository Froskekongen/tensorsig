"""Hall and Lyndon coordinates for free Lie algebras.

Hall trees and Lyndon words are two coordinate systems for the same truncated
free Lie algebra. Their modules expose basis construction, dimensions, and
stable human-readable coordinate labels; numerical log-signature evaluation
lives in ``tensorsig.classical.signatures``.
"""

from . import hall as hall
from . import lyndon as lyndon

__all__ = [
    "hall",
    "lyndon",
]
