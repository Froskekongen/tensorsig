"""Compilation plans for local signatures, log-signatures, and polynomials.

``LocalRealization`` binds an offset code, channel count, and maximum
depth, then compiles reusable signature or log-signature plans. Signature
plans may retain dense tensor coordinates or a smaller chronological chart.
Log-signature plans select a minimal chart on the locally realized primitive
image and can reconstruct the represented ambient coordinates.

``compile_polynomial`` instead builds the physical-offset scalar monomial
dictionary. Its span, support, and multiplicity bounds are deliberately lossy
coordinate restrictions, not Lie quotients. Plans are immutable compiled
descriptions: construct them through these factories and reuse them rather
than instantiating plan classes directly.

Pascal chronological charts use structural counting, shared word evaluation,
and lazy triangular reconstruction, including requested ordinary words through
``SignaturePlan.reconstruct_words``. General
fixed-code charts and minimal local primitive charts use float64 rank and
pivot decisions controlled by ``rank_tolerance``; their margins remain
inspectable on the compiled plan.
"""

from .plan import LocalRealization as LocalRealization
from .plan import LogSignaturePlan as LogSignaturePlan
from .plan import PolynomialPlan as PolynomialPlan
from .plan import SignaturePlan as SignaturePlan
from .plan import compile_polynomial as compile_polynomial

__all__ = [
    "LocalRealization",
    "LogSignaturePlan",
    "PolynomialPlan",
    "SignaturePlan",
    "compile_polynomial",
]
