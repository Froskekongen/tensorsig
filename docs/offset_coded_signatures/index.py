# %% [markdown]
# # Offset-coded signatures
#
# Start with `OffsetCode`, `LocalRealization`, and `LocalFeatures` to turn a
# path into local features that retain relative position inside each window.
# Choose the construction by the information you need, then choose its
# evaluation algorithm separately.
#
# ## Choose a construction
#
# | Need | Construction | Public module |
# | --- | --- | --- |
# | Ordinary path signatures or logsignatures | `signature`, `log_signature` | `tensorsig.classical.signatures` |
# | Ordinary sliding logsignatures after path lifts | `nn.Sequential(..., WindowedLogSignature(...))` | `tensorsig.augmentations`, `tensorsig.classical.windows` |
# | Relative offset information in local signatures | `LocalRealization(code, ...).signature()` | `offset_coded_signatures.codes`, `.plans` |
# | Minimal local primitive coordinates | `LocalRealization(code, ...).log_signature()` | `offset_coded_signatures.plans` |
# | Primitive features with structural relations | `.log_signature(quotient=LieQuotient(...))` | `offset_coded_signatures.quotients` |
# | Scalar polynomials in physical increments | `compile_polynomial(...)` | `offset_coded_signatures.plans` |
# | A trainable head on any of these local plans | `LocalLinear(plan, out_features=...)` | `offset_coded_signatures.layers` |
#
# Paths have shape `(batch, points, channels)`. Offset-coded window sizes and
# strides count **increments**: a window of size 3 uses 4 consecutive points.
# Classical `window_size` instead counts **points**. All windows are valid
# windows, without padding.
#
# ## Compile once, evaluate many paths

# %%
import torch

from tensorsig.augmentations import CausalFIRAugmentation, PairwiseProductAugmentation
from tensorsig.classical.windows import WindowedLogSignature, windowed_log_signature
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.layers import LocalFeatures, LocalLinear
from tensorsig.offset_coded_signatures.plans import LocalRealization
from tensorsig.offset_coded_signatures.quotients import (
    LieQuotient,
    ProjectedAlphabet,
    TemporalDegree,
    compile_lie_quotient,
)
from tensorsig.offset_coded_signatures.windows import (
    WindowLogSignature,
    increment_windows,
)

path = torch.tensor(
    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0], [3.0, 0.0]]],
    dtype=torch.float64,
)
code = OffsetCode.pascal(window=3, dimension=2, dtype=path.dtype)
realization = LocalRealization(code, channels=2, max_depth=2)
plan = realization.signature()
features = LocalFeatures(plan)
values = features(path)

assert values.shape == (1, 3, plan.feature_count)
torch.testing.assert_close(
    values,
    plan.evaluate(increment_windows(path, size=3)),
)
head = LocalLinear(plan, out_features=4)
assert head(path).shape == (1, 3, 4)

# %% [markdown]
# A plan describes the compiled coordinates; a layer applies them to sliding
# windows. Reuse plans and layers rather than compiling them in `forward`.
# Fixed code matrices become module buffers, and trainable matrices become
# parameters. Move layers and input tensors to matching dtypes and devices with
# the usual PyTorch `.to(...)` operation.
#
# `OffsetCode.constant(window)` forgets relative offset, `pascal(window,
# dimension)` retains a chosen chronological code, and `identity(window)` keeps
# every physical offset. `AlphabetMap` additionally allows channel mixing;
# currently it supports dense signature plans.
#
# The default signature coordinates use a chronological chart where supported.
# Request `coordinates="dense"` for ordinary tensor coordinates. Inspect
# `plan.feature_count`, `plan.level_dimensions`, and `plan.is_lossless` when
# choosing a chart or a coordinate selection. `WordSelection` and
# `ChronologicalFiltration` restrict coordinates; `LieQuotient` changes the
# represented algebra. They are distinct choices.
#
# ## Compose path augmentations
#
# Path lifts belong to `tensorsig.augmentations` and can precede classical or
# offset-coded features. Use ordinary `nn.Sequential` to specify their order.
# Each lift receives the full path before windows are extracted. When a lift
# changes the number of channels, use its `out_channels` to configure the next
# lift or an offset-coded `LocalRealization`.

# %%
fir = CausalFIRAugmentation(in_channels=2, num_filters=1, max_lag=1)
products = PairwiseProductAugmentation(in_channels=fir.out_channels)
pipeline = torch.nn.Sequential(
    fir,
    products,
    WindowedLogSignature(depth=2, window_size=4),
).to(dtype=path.dtype)
lifted_values = pipeline(path)

assert lifted_values.shape[:2] == (1, 3)
torch.testing.assert_close(
    lifted_values,
    windowed_log_signature(products(fir(path)), depth=2, window_size=4, hop_size=1),
)

# %% [markdown]
# This replaces the former `CausalWindowedLogSignature` wrapper. The pipeline
# is causal only when its preceding lifts are causal; the window transform
# itself does not make that guarantee for arbitrary modules.
#
# ## Advanced runtime and offline analysis
#
# Use `windows.WindowLogSignature` when you already have a `LieQuotientPlan`
# and need explicit independent, rolling, or scan algorithms. It consumes
# **increments**, and returns the abstract quotient coordinates. In contrast,
# `LocalFeatures` consumes **path points** and returns its plan's local chart.
# These coordinate systems need not have the same dimension.

# %%
quotient_plan = compile_lie_quotient(
    LieQuotient(TemporalDegree(1)),
    ProjectedAlphabet.newton(code_dim=2, in_channels=2),
    max_depth=2,
)
runtime = WindowLogSignature(quotient_plan, window=3)
quotient_values = runtime(path.diff(dim=1), algorithm="direct")
assert quotient_values.shape == (1, 3, quotient_plan.dimension)

# %% [markdown]
# `WindowLogSignature` generates raw Pascal codes for a certified projected
# alphabet. Its default is different from `OffsetCode.pascal`, which
# orthonormalizes by default. Do not compare coordinates from different codes
# without the corresponding basis conversion.
#
# | Advanced task | Module |
# | --- | --- |
# | Choose or inspect a window evaluation policy | `windows` |
# | Work directly with selected tensor-log states | `active_word_log` |
# | Work directly with FIFO/scan transport programs | `rolling_logsignature` |
# | Inspect proved binary Pascal ranks | `pascal_rank` |
# | Build exact offline Pascal rank witnesses | `pascal_certificate` |
# | Run exact offline graph-block queries | `graph_queries` |
#
# The last three modules provide mathematical diagnostics and certificates;
# they are not alternative PyTorch feature layers. Modules beginning with `_`
# and implementation helpers are not user entry points.
#
# Shared low-level operations live outside this package:
# `tensorsig.tensor_ops.series.signature_levels_from_increments` accepts
# arbitrary leading batch axes, and `tensorsig.algebra.eulerian` provides the
# common homogeneous Eulerian projection.
#
# See the [offset-coded API reference](../api/offset-coded-signatures/index.md)
# for plan options, quotient rules, and the limits of individual backends.
