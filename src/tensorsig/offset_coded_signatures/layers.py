"""PyTorch layers and structured readouts for compiled local feature plans.

``LocalFeatures`` evaluates a plan on every valid increment window;
``LocalLinear`` adds a trainable linear readout. Fixed code matrices are
buffers and trainable maps are parameters, so device moves, state dictionaries,
optimizers, and autograd follow ordinary PyTorch conventions. Inputs must
match registered state dtype and device.

``GapFactorizedVolterra`` is a compact structured alternative to a free
linear head on a strict polynomial plan. It factors temporal weights over
positive internal gaps and terminal lag while retaining a learned channel
tensor. Complete strict plans can use dynamic programming; genuinely
span-filtered plans use exact tuple enumeration.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from tensorsig.classical.windows import windowed_signature
from tensorsig.tensor_ops.series import unflatten_signature_levels

from .codes import OffsetCode
from ._sliding import _rolling_chronological_features
from .gap_factorized_volterra import GapFactorizedVolterra as GapFactorizedVolterra
from .plans import LogSignaturePlan, PolynomialPlan, SignaturePlan
from ._sliding import _rolling_coded_signature_levels
from .windows import validate_path_points
from .windows import increment_windows


FeaturePlan = SignaturePlan | LogSignaturePlan | PolynomialPlan


def _validate_stride(stride: int) -> int:
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    return stride


def _validate_chunk_size(chunk_size: int | None) -> int | None:
    if chunk_size is not None and (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer or None.")
    return chunk_size


class LocalFeatures(torch.nn.Module):
    """Evaluate a compiled local feature plan on sliding path windows.

    A trainable code or alphabet-projection matrix is registered as a parameter;
    a fixed matrix is registered as a buffer. Polynomial plans have no matrix
    state. ``stride`` is measured in path increments.
    """

    def __init__(
        self,
        plan: FeaturePlan,
        stride: int = 1,
        *,
        backend: str = "auto",
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(plan, (SignaturePlan, LogSignaturePlan, PolynomialPlan)):
            raise TypeError(
                "plan must be a SignaturePlan, LogSignaturePlan, or PolynomialPlan."
            )
        if not isinstance(backend, str):
            raise TypeError("backend must be a string.")

        self.plan = plan
        self.stride = _validate_stride(stride)
        self.backend = backend
        self.chunk_size = _validate_chunk_size(chunk_size)

        if isinstance(plan, SignaturePlan):
            allowed = {"auto", "vectorized", "rolling"}
            if self.backend not in allowed:
                raise ValueError("backend must be 'auto', 'vectorized', or 'rolling'.")
            if self.backend == "rolling":
                if not isinstance(plan.mapping, OffsetCode):
                    raise ValueError("rolling requires a separable OffsetCode.")
                if plan.coordinates == "chronological":
                    if (
                        plan.certification != "structural"
                        or plan.mapping.family != "pascal"
                    ):
                        raise ValueError(
                            "Chronological rolling requires a structural Pascal plan."
                        )
                elif self.stride < plan.window:
                    plan.mapping.shift_operator(self.stride)
        elif isinstance(plan, LogSignaturePlan):
            allowed = (
                {"auto", "direct", "pbw", "tensor"}
                if plan.quotient is not None
                else {"auto", "dense", "rolling", "sparse"}
            )
            if self.backend not in allowed:
                choices = ", ".join(sorted(allowed - {"auto"}))
                raise ValueError(f"backend must be 'auto' or one of: {choices}.")
            if self.backend == "rolling" and self.stride < plan.window:
                plan.code.shift_operator(self.stride)
        elif self.backend != "auto":
            raise ValueError("PolynomialPlan supports only backend='auto'.")
        if not isinstance(plan, PolynomialPlan) and self.chunk_size is not None:
            raise ValueError("chunk_size applies only to PolynomialPlan.")

        if isinstance(plan, PolynomialPlan):
            self.register_parameter("matrix", None)
        elif plan._matrix_tensor().requires_grad:
            source_matrix = plan._matrix_tensor()
            matrix = (
                source_matrix
                if isinstance(source_matrix, torch.nn.Parameter)
                else torch.nn.Parameter(source_matrix.detach().clone())
            )
            self.register_parameter("matrix", matrix)
        else:
            self.register_buffer("matrix", plan._matrix_tensor().detach().clone())

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        key = prefix + "matrix"
        if "matrix" in self._buffers and key in state_dict:
            current = self._buffers["matrix"]
            incoming = state_dict[key]
            if current is not None and incoming.shape == current.shape:
                candidate = incoming.to(device=current.device, dtype=current.dtype)
                if not torch.equal(candidate, current):
                    error_msgs.append(
                        f"{key} is compiled plan architecture and cannot be "
                        "replaced through state_dict; construct the module with "
                        "the matching plan"
                    )
                    return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def window(self) -> int:
        """Number of increments in each local window."""
        return self.plan.window

    @property
    def channels(self) -> int:
        """Expected path channel count."""
        return self.plan.channels

    @property
    def feature_count(self) -> int:
        """Number of output features per window."""
        return self.plan.feature_count

    def _evaluate_windows(self, windows: Tensor) -> Tensor:
        if self.matrix is not None and (
            windows.device != self.matrix.device or windows.dtype != self.matrix.dtype
        ):
            raise RuntimeError(
                "path dtype/device must match the module matrix; move the module "
                "and input explicitly before evaluation"
            )
        if isinstance(self.plan, SignaturePlan):
            assert self.matrix is not None
            return self.plan._evaluate(windows, self.matrix)
        if isinstance(self.plan, LogSignaturePlan):
            assert self.matrix is not None
            return self.plan._evaluate(
                windows,
                self.matrix,
                backend=self.backend,
            )
        return self.plan.evaluate(windows, chunk_size=self.chunk_size)

    def _constant_code_signature(self, path: Tensor) -> Tensor | None:
        if not isinstance(self.plan, SignaturePlan):
            return None
        if not isinstance(self.plan.mapping, OffsetCode):
            return None
        matrix = self.matrix
        assert matrix is not None
        if (
            matrix.requires_grad
            or self.plan.mapping.family != "pascal"
            or self.plan.mapping.temporal_degrees != (0,)
            or matrix.shape[1] != 1
            or not torch.equal(matrix.detach(), matrix[:1].detach().expand_as(matrix))
        ):
            return None
        flattened = windowed_signature(
            path,
            depth=max(self.plan.depths),
            window_size=self.window + 1,
            hop_size=self.stride,
        )
        levels = unflatten_signature_levels(
            flattened,
            width=self.channels,
            depth=max(self.plan.depths),
        )
        scale = matrix[0, 0]
        coded_levels = tuple(
            level * scale.pow(depth) for depth, level in enumerate(levels, start=1)
        )
        return self.plan._coordinates_from_levels(coded_levels)

    def forward(self, path: Tensor) -> Tensor:
        """Return ``(batch, windows, features)`` local coordinates."""
        validate_path_points(path, in_channels=self.channels)
        if self.matrix is not None and (
            path.device != self.matrix.device or path.dtype != self.matrix.dtype
        ):
            raise RuntimeError(
                "path dtype/device must match the module matrix; move the module "
                "and input explicitly before evaluation"
            )
        if isinstance(self.plan, SignaturePlan):
            if self.backend == "auto":
                constant = self._constant_code_signature(path)
                if constant is not None:
                    return constant
            if self.backend == "rolling":
                assert self.matrix is not None
                if self.plan.coordinates == "chronological":
                    assert isinstance(self.plan.mapping, OffsetCode)
                    return _rolling_chronological_features(
                        path,
                        self.matrix,
                        self.plan.mapping,
                        self.plan._rolling_program(),
                        channels=self.channels,
                        stride=self.stride,
                    )
                levels = _rolling_coded_signature_levels(
                    path,
                    self.matrix,
                    channels=self.channels,
                    depth=max(self.plan.depths),
                    stride=self.stride,
                )
                return self.plan._coordinates_from_levels(levels)
        if isinstance(self.plan, LogSignaturePlan) and self.backend == "rolling":
            assert self.matrix is not None
            levels = _rolling_coded_signature_levels(
                path,
                self.matrix,
                channels=self.channels,
                depth=max(self.plan.depths),
                stride=self.stride,
            )
            return self.plan._coordinates_from_signature_levels(levels)
        windows = increment_windows(path, size=self.window, stride=self.stride)
        return self._evaluate_windows(windows)

    def extra_repr(self) -> str:
        details = (
            f"plan={type(self.plan).__name__}, window={self.window}, "
            f"channels={self.channels}, features={self.feature_count}, "
            f"stride={self.stride}"
        )
        if isinstance(self.plan, (SignaturePlan, LogSignaturePlan)):
            details += f", backend={self.backend!r}"
        if isinstance(self.plan, PolynomialPlan):
            details += f", chunk_size={self.chunk_size}"
        return details


class LocalLinear(torch.nn.Module):
    """Apply a learned linear map to a compiled local feature plan."""

    def __init__(
        self,
        plan: FeaturePlan,
        out_features: int,
        stride: int = 1,
        bias: bool = True,
        *,
        backend: str = "auto",
        chunk_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(out_features, int)
            or isinstance(out_features, bool)
            or out_features < 1
        ):
            raise ValueError("out_features must be a positive integer.")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a boolean.")
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError("dtype must be floating-point.")

        self.features = LocalFeatures(
            plan,
            stride=stride,
            backend=backend,
            chunk_size=chunk_size,
        )
        if device is not None or dtype is not None:
            self.features.to(device=device, dtype=dtype)

        parameter_device = device
        parameter_dtype = dtype
        if not isinstance(plan, PolynomialPlan):
            plan_matrix = plan._matrix_tensor()
            if parameter_device is None:
                parameter_device = plan_matrix.device
            if parameter_dtype is None:
                parameter_dtype = plan_matrix.dtype

        self.in_features = plan.feature_count
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(
                out_features,
                self.in_features,
                device=parameter_device,
                dtype=parameter_dtype,
            )
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(
                    out_features,
                    device=parameter_device,
                    dtype=parameter_dtype,
                )
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @property
    def plan(self) -> FeaturePlan:
        """The compiled feature plan."""
        return self.features.plan

    @property
    def stride(self) -> int:
        """Window stride measured in increments."""
        return self.features.stride

    @property
    def matrix(self) -> Tensor | None:
        """Registered code/projection matrix, if the plan has one."""
        return self.features.matrix

    def reset_parameters(self) -> None:
        """Initialize the linear readout like ``torch.nn.Linear``."""
        if self.in_features:
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = self.in_features**-0.5
            if self.bias is not None:
                torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            torch.nn.init.zeros_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

    def forward(self, path: Tensor) -> Tensor:
        """Return ``(batch, windows, out_features)`` local readouts."""
        if isinstance(self.plan, PolynomialPlan):
            windows = increment_windows(
                path,
                size=self.plan.window,
                stride=self.stride,
            )
            if (
                windows.device != self.weight.device
                or windows.dtype != self.weight.dtype
            ):
                raise RuntimeError(
                    "path dtype/device must match the module readout; move the "
                    "module and input explicitly before evaluation"
                )
            output = self.plan.contract(
                windows,
                self.weight,
                chunk_size=self.features.chunk_size,
            )
            if self.bias is not None:
                output = output + self.bias
            return output
        features = self.features(path)
        return F.linear(features, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"stride={self.stride}, bias={self.bias is not None}"
        )


__all__ = ["LocalFeatures", "LocalLinear", "GapFactorizedVolterra"]
