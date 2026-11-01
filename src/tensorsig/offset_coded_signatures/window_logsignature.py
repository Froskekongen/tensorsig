"""Reusable quotient-window evaluation with explicit scheduling choices.

Selection reads fixed algebra and workload metadata only. The algorithms share
the same ordered quotient coordinates; none selects support from tensor values.
"""

from dataclasses import dataclass
from math import comb

import torch
from torch import Tensor

from ._lie_quotients.compiler import (
    LieQuotientPlan,
    quotient_log_signature_from_increment_windows,
)
from ._lie_quotients.rules import (
    DependenceGraph,
    LieQuotient,
    OffsetAlphabet,
    ProjectedAlphabet,
    TemporalDegree,
)
from .alphabet import _positive_integer
from .utils import unfold_increments


@dataclass(frozen=True)
class WindowSelection:
    """Inspectable policy result, including the requested derivative order."""

    algorithm: str
    derivative_order: int
    backend: str | None = None
    chunk_windows: int | None = None
    reason: str = "explicit override"
    shift_backend: str | None = None


class WindowLogSignature:
    """Map increment streams ``(B,T,C)`` to quotient windows ``(B,N,D)``.

    A projected alphabet without ``offset_code`` uses certified raw Pascal
    codes. A supplied code tensor supports independent windows and gradients
    in that tensor. Rolling transport requires the certified default code.
    ``derivative_order`` is selection metadata, not a switch that disables
    autograd. The default is 1; use 2 when requesting higher-order derivatives.
    Automatic choices use fixed schedules; choose an explicit algorithm to set
    backend, shift_backend, chunk_windows or refresh_every. Explicit rolling
    transport defaults to dense; the automatic typed rule records sparse.
    GPU auto is restricted to the typed FIFO region and raises NotImplementedError
    elsewhere; explicit algorithms retain their existing capabilities. CPU auto
    uses direct independent windows. Selection is not an accuracy certificate.
    """

    def __init__(
        self, plan: LieQuotientPlan, window: int, *, offset_code: Tensor | None = None
    ):
        if not isinstance(plan, LieQuotientPlan):
            raise TypeError("plan must be a LieQuotientPlan.")
        _positive_integer(window, "window")
        self.plan, self.window = plan, window
        self._pascal = False
        if isinstance(plan.alphabet, ProjectedAlphabet):
            if offset_code is None:
                if plan.alphabet.temporal_degrees != tuple(
                    range(plan.alphabet.code_dim)
                ):
                    raise ValueError(
                        "Supply offset_code for an uncertified projected alphabet."
                    )
                if plan.alphabet.code_dim > window:
                    raise ValueError("Pascal width must not exceed window.")
                self._pascal = True
                offset_code = torch.tensor(
                    [
                        [comb(r, p) for p in range(plan.alphabet.code_dim)]
                        for r in range(window)
                    ],
                    dtype=torch.float64,
                )
            if (
                not isinstance(offset_code, Tensor)
                or not offset_code.dtype.is_floating_point
            ):
                raise TypeError("offset_code must be a floating-point tensor.")
            if tuple(offset_code.shape) != (window, plan.alphabet.code_dim):
                raise ValueError("offset_code shape must be (window, code_dim).")
        elif isinstance(plan.alphabet, OffsetAlphabet):
            if offset_code is not None or window != plan.alphabet.kernel_size:
                raise ValueError(
                    "An offset alphabet requires its own window size and no code."
                )
        self._offset_code = offset_code
        self._pascal_codes = {}
        self._rolling = {}
        self._auto_graph = None
        if self._pascal and plan.alphabet.code_dim == 2 and plan.max_depth == 3:
            if plan.quotient == LieQuotient(TemporalDegree(1)):
                self._auto_graph = False
            elif plan.quotient == LieQuotient(
                TemporalDegree(1),
                DependenceGraph(
                    (i, i + 1) for i in range(plan.alphabet.in_channels - 1)
                ),
            ):
                self._auto_graph = True

    @property
    def offset_code(self) -> Tensor | None:
        """A copy for inspection; provide a code tensor at construction to train it."""
        return None if self._offset_code is None else self._offset_code.clone()

    def select(
        self,
        increments: Tensor,
        *,
        stride: int = 1,
        algorithm: str = "auto",
        derivative_order: int = 1,
        backend: str | None = None,
        shift_backend: str | None = None,
        chunk_windows: int | None = None,
    ) -> WindowSelection:
        _positive_integer(stride, "stride")
        if type(derivative_order) is not int or derivative_order not in {0, 1, 2}:
            raise ValueError("derivative_order must be 0, 1, or 2.")
        if (
            increments.ndim != 3
            or increments.shape[-1] != self.plan.alphabet.in_channels
        ):
            raise ValueError("Expected increments with shape (batch, time, channels).")
        if not increments.dtype.is_floating_point or increments.shape[1] < self.window:
            raise ValueError(
                "Expected floating increments containing at least one window."
            )
        if chunk_windows is not None:
            _positive_integer(chunk_windows, "chunk_windows")
        independent = {"direct", "pbw", "tensor", "active"}
        rolling = {"rolling", "scan"}
        if algorithm not in independent | rolling | {"auto"}:
            raise ValueError(f"Unknown window algorithm: {algorithm!r}.")
        reason = "explicit override"
        if algorithm == "auto":
            if (
                backend is not None
                or shift_backend is not None
                or chunk_windows is not None
            ):
                raise ValueError(
                    "Use an explicit algorithm to override backend, transport or chunk_windows."
                )
            if increments.device.type == "cpu":
                algorithm = "direct"
                reason = "CPU independent direct baseline"
            else:
                channels = self.plan.alphabet.in_channels
                count = 1 + (increments.shape[1] - self.window) // stride
                supported = (
                    increments.device.type in {"mps", "cuda"}
                    and increments.dtype == torch.float32
                    and increments.shape[0] == 1
                    and stride == 1
                    and self.window == 8
                    and self._auto_graph is not None
                )
                if (
                    supported
                    and self._auto_graph
                    and derivative_order in {0, 1}
                    and 8 <= channels <= 16
                    and 1024 <= count <= 4096
                ):
                    algorithm, backend, shift_backend, chunk_windows = (
                        "rolling",
                        "typed",
                        "sparse",
                        8,
                    )
                    reason = "GPU typed FIFO region; MPS proxy assumption"
                else:
                    raise NotImplementedError(
                        "Unsupported GPU auto configuration. Choose an explicit algorithm; "
                        "GPU auto supports only float32, batch=1, stride=1, default "
                        "Pascal P=2, depth=3, TD1, window=8: path channels=8..16, "
                        "windows=1024..4096, derivative_order=0 or 1."
                    )
        if algorithm in rolling:
            if not self._pascal:
                raise ValueError(
                    "Rolling algorithms require the certified default Pascal code."
                )
            backend = "joint" if backend is None else backend
            if backend not in {"typed", "fused", "joint", "joint_fused"}:
                raise ValueError("backend must be typed, fused, joint, or joint_fused.")
            shift_backend = "dense" if shift_backend is None else shift_backend
            if shift_backend not in {"dense", "sparse"}:
                raise ValueError("shift_backend must be 'dense' or 'sparse'.")
            if algorithm == "scan" and shift_backend != "dense":
                raise ValueError("Affine scan supports only shift_backend='dense'.")
            if (
                algorithm == "rolling"
                and backend in {"joint", "joint_fused"}
                and shift_backend != "dense"
            ):
                raise ValueError("Joint chunks support only shift_backend='dense'.")
            if derivative_order == 2 and backend == "fused":
                raise ValueError(
                    "Use joint_fused for second-order compiled differentiation."
                )
        elif backend is not None or shift_backend is not None:
            raise ValueError(
                "backend and shift_backend apply only to rolling and scan algorithms."
            )
        return WindowSelection(
            algorithm, derivative_order, backend, chunk_windows, reason, shift_backend
        )

    def __call__(
        self,
        increments: Tensor,
        *,
        stride: int = 1,
        algorithm: str = "auto",
        derivative_order: int = 1,
        backend: str | None = None,
        shift_backend: str | None = None,
        chunk_windows: int | None = None,
        refresh_every: int | None = None,
    ) -> Tensor:
        if algorithm == "auto" and refresh_every is not None:
            raise ValueError("Use an explicit algorithm to override refresh_every.")
        selected = self.select(
            increments,
            stride=stride,
            algorithm=algorithm,
            derivative_order=derivative_order,
            backend=backend,
            shift_backend=shift_backend,
            chunk_windows=chunk_windows,
        )
        if selected.algorithm in {"direct", "pbw", "tensor", "active"}:
            if refresh_every is not None:
                raise ValueError("refresh_every applies only to rolling algorithms.")
            windows = unfold_increments(increments, self.window, stride)
            if self._offset_code is None:
                code = None
            elif self._pascal:
                key = (increments.device, increments.dtype)
                code = self._pascal_codes.get(key)
                if code is None:
                    code = self._offset_code.to(increments)
                    self._pascal_codes[key] = code
            else:
                # Custom codes can change or carry a new autograd graph each call.
                code = self._offset_code.to(increments)
            count = windows.shape[1]
            chunk = selected.chunk_windows or count
            if chunk >= count:
                return quotient_log_signature_from_increment_windows(
                    windows, self.plan, code, method=selected.algorithm
                )
            return torch.cat(
                [
                    quotient_log_signature_from_increment_windows(
                        windows[:, start : start + chunk],
                        self.plan,
                        code,
                        method=selected.algorithm,
                    )
                    for start in range(0, count, chunk)
                ],
                dim=1,
            )
        key = (increments.device, increments.dtype)
        evaluator = self._rolling.get(key)
        if evaluator is None:
            from .rolling_logsignature import RollingLogSignature

            bound = self.plan.to(device=increments.device, dtype=increments.dtype)
            evaluator = RollingLogSignature(bound, self.window)
            self._rolling[key] = evaluator
        return evaluator(
            increments,
            algorithm="chunks"
            if selected.algorithm == "rolling"
            else selected.algorithm,
            backend=selected.backend,
            shift_backend=selected.shift_backend,
            stride=stride,
            derivative_order=derivative_order,
            chunk_windows=selected.chunk_windows,
            refresh_every=refresh_every,
        )


__all__ = ["WindowLogSignature", "WindowSelection"]
