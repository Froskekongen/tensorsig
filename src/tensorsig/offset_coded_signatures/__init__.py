"""Offset-coded signature plans, layers, and algebraic quotients.

The API keeps four modeling choices separate: increment-window extraction, a
linear map on offsets and channels, a coordinate chart or Lie quotient, and an
optional PyTorch readout. Plans compile the fixed algebra once; layers reuse
that plan across every valid sliding window of a batched path.

Start with ``codes.OffsetCode``, ``plans.LocalRealization``, and
``layers.LocalFeatures``. ``coordinates`` selects returned functionals;
``quotients`` defines algebraic relations. ``windows`` provides increment-window
extraction and direct quotient evaluation. The remaining public modules expose
advanced evaluation engines and offline rank or graph certificates. Private
modules beginning with ``_`` implement these constructions.

``window`` and ``stride`` count increments rather than path points. Dense
operations are differentiable with respect to both the path and any trainable
alphabet map.
"""

from . import codes as codes
from . import coordinates as coordinates
from . import plans as plans
from . import quotients as quotients
from . import layers as layers
from . import windows as windows
from . import graph_queries as graph_queries

from . import active_word_log as active_word_log
from . import rolling_logsignature as rolling_logsignature
from . import pascal_rank as pascal_rank
from . import pascal_certificate as pascal_certificate

__all__ = [
    "codes",
    "coordinates",
    "plans",
    "quotients",
    "layers",
    "windows",
    "graph_queries",
    "active_word_log",
    "rolling_logsignature",
    "pascal_rank",
    "pascal_certificate",
]
