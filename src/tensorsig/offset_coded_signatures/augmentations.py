"""Composable causal and nonlinear path augmentations.

Causal FIR augmentation appends learned lagged mixtures of path channels with
an explicit left-boundary convention. Pairwise products are a separate
nonlinear module so filtering and nonlinear lifting remain visible and
reusable. A constant FIR bias is invisible to an immediately applied signature
because of translation invariance, but becomes observable when a nonlinear
point augmentation intervenes.
"""

from .path_augmentations import CausalFIRAugmentation as CausalFIRAugmentation
from .path_augmentations import (
    CausalWindowedLogSignature as CausalWindowedLogSignature,
)
from .path_augmentations import (
    PairwiseProductAugmentation as PairwiseProductAugmentation,
)
from .path_augmentations import causal_fir as causal_fir

__all__ = [
    "causal_fir",
    "CausalFIRAugmentation",
    "CausalWindowedLogSignature",
    "PairwiseProductAugmentation",
]
