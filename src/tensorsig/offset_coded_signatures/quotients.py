"""Declarative Lie quotients and stable quotient-coordinate labels.

A ``LieQuotient`` generates an ideal before evaluation, so the reduced
algebra remains compatible with brackets, BCH, exponentials, and Chen
concatenation. Rules can constrain temporal degree, channel or offset support
and multiplicity, physical offset span, or commutation through a dependence
graph; rules compose with ``&``.

The abstract quotient is compiled exactly. A local plan then composes it with
the finite-window realization and numerically selects a minimal local chart.
Its abstract ``quotient_lie_dimension`` and realized
``local_quotient_dimension`` therefore answer different questions.
``QuotientBasisElement`` supplies stable public labels without exposing
private compiler objects. Physical-offset rules must be applied before any
lower-dimensional offset projection.
"""

from ._lie_quotients.rules import ChannelMultiplicity as ChannelMultiplicity
from ._lie_quotients.rules import ChannelSupport as ChannelSupport
from ._lie_quotients.rules import DependenceGraph as DependenceGraph
from ._lie_quotients.rules import LieQuotient as LieQuotient
from ._lie_quotients.rules import OffsetMultiplicity as OffsetMultiplicity
from ._lie_quotients.rules import OffsetSpan as OffsetSpan
from ._lie_quotients.rules import OffsetSupport as OffsetSupport
from ._lie_quotients.rules import TemporalDegree as TemporalDegree
from .quotient_basis import QuotientBasisElement as QuotientBasisElement
from ._lie_quotients.compiler import LieQuotientPlan as LieQuotientPlan
from ._lie_quotients.compiler import compile_lie_quotient as compile_lie_quotient
from ._lie_quotients.rules import OffsetAlphabet as OffsetAlphabet
from ._lie_quotients.rules import ProjectedAlphabet as ProjectedAlphabet
from .window_logsignature import WindowLogSignature as WindowLogSignature
from .window_logsignature import WindowSelection as WindowSelection

__all__ = [
    "LieQuotient",
    "DependenceGraph",
    "QuotientBasisElement",
    "ChannelMultiplicity",
    "ChannelSupport",
    "OffsetMultiplicity",
    "OffsetSpan",
    "OffsetSupport",
    "TemporalDegree",
    "LieQuotientPlan",
    "compile_lie_quotient",
    "OffsetAlphabet",
    "ProjectedAlphabet",
    "WindowLogSignature",
    "WindowSelection",
]
