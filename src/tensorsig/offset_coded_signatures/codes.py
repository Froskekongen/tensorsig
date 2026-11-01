"""Maps from physical offsets and channels to coded alphabets.

``OffsetCode`` represents the separable map ``A ⊗ I``: each offset row
of ``A`` codes time while leaving physical channels distinct. Certified Pascal
codes carry a chronological grading; constant codes forget relative offset;
identity codes retain it. Arbitrary matrices remain explicitly ungraded.

``AlphabetMap`` is the general offset-dependent map and may mix channels.
Dense signature plans support it, while chronological charts and structured
quotients require the separable semantics of ``OffsetCode``. Coding
increments is not the same construction as appending a deterministic phase
channel to a path.
"""

from .alphabet import AlphabetMap as AlphabetMap
from .alphabet import OffsetCode as OffsetCode

__all__ = [
    "AlphabetMap",
    "OffsetCode",
]
