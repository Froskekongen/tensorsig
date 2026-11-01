"""Linear maps from physical offsets and channels to local alphabets."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import math
from typing import Literal

import torch
from torch import Tensor

_CodeFamily = Literal["pascal", "identity", "structured", "matrix"]

_STRUCTURED_KINDS = {
    "constant",
    "identity",
    "cumulative_clock",
    "dct",
    "polynomial",
    "sinusoidal",
    "minirocket",
}


def _floating_matrix(matrix: Tensor, name: str, ndim: int) -> None:
    if not isinstance(matrix, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if matrix.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if not matrix.dtype.is_floating_point:
        raise ValueError(f"{name} must have a floating-point dtype.")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")


def _owned_matrix(matrix: Tensor) -> Tensor:
    """Own fixed storage and normalize trainable storage to a leaf parameter."""
    if not matrix.requires_grad:
        return matrix.detach().clone()
    if isinstance(matrix, torch.nn.Parameter):
        return matrix
    return torch.nn.Parameter(matrix.detach().clone())


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _output_dtype(dtype: torch.dtype | None) -> torch.dtype:
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating-point.")
    return dtype


def _pascal_matrix(
    window: int,
    dimension: int,
    *,
    orthonormal: bool,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> tuple[Tensor, Tensor]:
    dtype = _output_dtype(dtype)
    raw = torch.tensor(
        [
            [math.comb(offset, degree) for degree in range(dimension)]
            for offset in range(window)
        ],
        dtype=torch.float64,
    )
    if not orthonormal:
        return raw.to(device=device, dtype=dtype), torch.eye(
            dimension, dtype=torch.float64
        )
    code, triangular = torch.linalg.qr(raw, mode="reduced")
    signs = torch.where(
        torch.diagonal(triangular) < 0,
        -torch.ones(dimension, dtype=torch.float64),
        torch.ones(dimension, dtype=torch.float64),
    )
    return (code * signs).to(device=device, dtype=dtype), signs[:, None] * triangular


def _center_columns(matrix: Tensor) -> Tensor:
    return matrix if matrix.numel() == 0 else matrix - matrix.mean(0, keepdim=True)


def _normalize_columns(matrix: Tensor) -> Tensor:
    if matrix.numel() == 0:
        return matrix
    return matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-12)


def _structured_columns(
    kind: str,
    window: int,
    count: int,
    *,
    include_constant: bool,
    positive_count: int | None,
    seed: int,
    dtype: torch.dtype,
) -> Tensor:
    if count == 0:
        return torch.empty(window, 0, dtype=dtype)
    if kind == "constant":
        raise ValueError("constant codes do not accept non-constant columns.")
    if kind == "identity":
        return torch.eye(window, dtype=dtype)[:, :count]
    if kind == "cumulative_clock":
        start = int(include_constant)
        if start + count > window:
            raise ValueError("Too many cumulative-clock columns for the window.")
        offsets = torch.arange(window).view(-1, 1)
        clocks = torch.arange(window).view(1, -1)
        matrix = (offsets > clocks).to(dtype)
        matrix = matrix + 0.5 * (offsets == clocks).to(dtype)
        matrix[:, 0] = 1
        return matrix[:, start : start + count]
    if kind == "dct":
        offsets = torch.arange(window, dtype=dtype).view(-1, 1)
        start = int(include_constant)
        frequencies = torch.arange(start, start + count, dtype=dtype).view(1, -1)
        return torch.cos(math.pi * (offsets + 0.5) * frequencies / window)
    if kind == "polynomial":
        grid = torch.linspace(-1.0, 1.0, window, dtype=dtype)
        start = int(include_constant)
        return torch.stack(
            [grid.pow(degree) for degree in range(start, start + count)],
            dim=1,
        )
    if kind == "sinusoidal":
        offsets = torch.arange(window, dtype=dtype)
        columns = []
        frequency = 1
        while len(columns) < count:
            angle = 2 * math.pi * frequency * offsets / window
            columns.append(torch.cos(angle))
            if len(columns) < count:
                columns.append(torch.sin(angle))
            frequency += 1
        return torch.stack(columns, dim=1)

    if window == 1:
        return torch.ones(1, count, dtype=dtype)
    if positive_count is None:
        positive_count = max(1, min(window - 1, window // 3))
    if positive_count < 1 or positive_count >= window:
        raise ValueError("positive_count must lie in [1, window - 1].")
    positive_value = float(window - positive_count) / positive_count
    columns = []
    for subset in itertools.combinations(range(window), positive_count):
        column = -torch.ones(window, dtype=dtype)
        column[list(subset)] = positive_value
        columns.append(column)
        if len(columns) == count:
            return torch.stack(columns, dim=1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    while len(columns) < count:
        selected = torch.randperm(window, generator=generator)[:positive_count]
        column = -torch.ones(window, dtype=dtype)
        column[selected] = positive_value
        columns.append(column)
    return torch.stack(columns, dim=1)


def _exact_shift_operator(matrix: Tensor, stride: int) -> Tensor:
    """Return ``L`` with ``matrix[stride:] @ L == matrix[:-stride]``.

    The solve is performed in float64 on CPU and accepted only when the
    residual is consistent with rounding in the matrix's stored dtype.
    """
    _floating_matrix(matrix, "matrix", 2)
    _positive_integer(stride, "stride")
    if matrix.requires_grad:
        raise ValueError("Shift operators require a fixed offset code.")
    if stride >= matrix.shape[0]:
        raise ValueError("stride must be smaller than the code window.")

    # Transfer before promoting: MPS cannot materialize a float64 intermediate.
    source = matrix[stride:].detach().cpu().double()
    target = matrix[:-stride].detach().cpu().double()
    operator = torch.linalg.lstsq(source, target).solution
    rounded = operator.to(dtype=matrix.dtype).to(dtype=torch.float64)
    residual = (source @ rounded - target).norm()
    scale = source.norm() * rounded.norm() + target.norm()
    tolerance = (
        64
        * torch.finfo(matrix.dtype).eps
        * max(matrix.shape)
        * scale.clamp_min(torch.finfo(torch.float64).tiny)
    )
    if residual > tolerance:
        relative = float(residual / target.norm().clamp_min(1e-30))
        raise ValueError(
            "Offset code is not exactly shift-closed for "
            f"stride={stride}; relative residual is {relative:.3e}."
        )
    return rounded.to(device=matrix.device, dtype=matrix.dtype)


@dataclass(frozen=True, eq=False, init=False)
class OffsetCode:
    """A separable offset code ``A: U_R -> W``.

    The matrix has shape ``(window, dimension)`` and acts identically on every
    signal channel. Temporal degrees are present only when the code carries a
    certified Pascal/Newton filtration.
    """

    _matrix: Tensor = field(repr=False)
    _family: _CodeFamily
    _temporal_degrees: tuple[int, ...] | None
    _pascal_factor: Tensor | None = field(repr=False)
    _pascal_raw: bool

    def __init__(self, matrix: Tensor) -> None:
        """Bind an arbitrary matrix as an ungraded matrix code."""
        _floating_matrix(matrix, "matrix", 2)
        owned = _owned_matrix(matrix)
        object.__setattr__(self, "_matrix", owned)
        object.__setattr__(self, "_family", "matrix")
        object.__setattr__(self, "_temporal_degrees", None)
        object.__setattr__(self, "_pascal_factor", None)
        object.__setattr__(self, "_pascal_raw", False)
        self.__post_init__()

    @classmethod
    def _certified(
        cls,
        matrix: Tensor,
        family: _CodeFamily,
        temporal_degrees: tuple[int, ...] | None = None,
        *,
        pascal_factor: Tensor | None = None,
        pascal_raw: bool = False,
    ) -> "OffsetCode":
        _floating_matrix(matrix, "matrix", 2)
        code = object.__new__(cls)
        owned = _owned_matrix(matrix)
        object.__setattr__(code, "_matrix", owned)
        object.__setattr__(code, "_family", family)
        object.__setattr__(code, "_temporal_degrees", temporal_degrees)
        object.__setattr__(
            code,
            "_pascal_factor",
            None
            if pascal_factor is None
            else pascal_factor.detach().to(device="cpu", dtype=torch.float64).clone(),
        )
        object.__setattr__(code, "_pascal_raw", pascal_raw)
        code.__post_init__()
        return code

    def __post_init__(self) -> None:
        _floating_matrix(self._matrix, "matrix", 2)
        window, dimension = self._matrix.shape
        if window < 1 or dimension < 1 or dimension > window:
            raise ValueError("matrix must have shape (R, P) with 1 <= P <= R.")
        if not isinstance(self._family, str) or self._family not in {
            "pascal",
            "identity",
            "structured",
            "matrix",
        }:
            raise ValueError("Unknown offset-code family.")
        if self._family == "pascal" and self._temporal_degrees is None:
            raise ValueError("Pascal codes require certified temporal degrees.")
        if self._family == "pascal":
            factor = self._pascal_factor
            if (
                factor is None
                or factor.shape != (dimension, dimension)
                or not torch.isfinite(factor).all()
                or not torch.equal(factor, factor.triu())
                or not torch.all(factor.diagonal() > 0)
            ):
                raise ValueError("Pascal codes require a positive triangular factor.")
        elif self._pascal_factor is not None or self._pascal_raw:
            raise ValueError("Only Pascal codes can carry Pascal construction data.")
        if self._temporal_degrees is not None:
            degrees = tuple(self._temporal_degrees)
            if self._family != "pascal":
                raise ValueError("Only certified Pascal codes carry temporal degrees.")
            if degrees != tuple(range(dimension)):
                raise ValueError("Pascal temporal degrees must be 0, ..., P - 1.")
            object.__setattr__(self, "_temporal_degrees", degrees)
        if self._family == "identity":
            identity = torch.eye(
                window, device=self._matrix.device, dtype=self._matrix.dtype
            )
            if dimension != window or not torch.equal(self._matrix, identity):
                raise ValueError(
                    "identity codes must contain the exact identity matrix."
                )

    @classmethod
    def pascal(
        cls,
        window: int,
        dimension: int,
        *,
        orthonormal: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "OffsetCode":
        """Return a certified Pascal/Newton offset code."""
        _positive_integer(window, "window")
        _positive_integer(dimension, "dimension")
        if not isinstance(orthonormal, bool):
            raise TypeError("orthonormal must be a boolean.")
        if dimension > window:
            raise ValueError("dimension must not exceed window.")
        matrix, factor = _pascal_matrix(
            window,
            dimension,
            orthonormal=orthonormal,
            device=device,
            dtype=dtype,
        )
        return cls._certified(
            matrix,
            family="pascal",
            temporal_degrees=tuple(range(dimension)),
            pascal_factor=factor,
            pascal_raw=not orthonormal,
        )

    @classmethod
    def constant(
        cls,
        window: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "OffsetCode":
        """Return the degree-zero Pascal code that forgets relative offsets."""
        _positive_integer(window, "window")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating-point.")
        return cls._certified(
            torch.ones(window, 1, device=device, dtype=dtype),
            family="pascal",
            temporal_degrees=(0,),
            pascal_factor=torch.ones(1, 1, dtype=torch.float64),
            pascal_raw=True,
        )

    @classmethod
    def identity(
        cls,
        window: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "OffsetCode":
        """Return full physical offset retention."""
        _positive_integer(window, "window")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating-point.")
        return cls._certified(
            torch.eye(window, device=device, dtype=dtype),
            family="identity",
        )

    @classmethod
    def structured(
        cls,
        window: int,
        dimension: int,
        *,
        kind: str = "dct",
        include_constant: bool = True,
        center: bool = False,
        normalize: bool = False,
        positive_count: int | None = None,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "OffsetCode":
        """Return a fixed structured code without inferred Newton grading."""
        _positive_integer(window, "window")
        _positive_integer(dimension, "dimension")
        if not isinstance(kind, str):
            raise TypeError("kind must be a string.")
        for name, value in (
            ("include_constant", include_constant),
            ("center", center),
            ("normalize", normalize),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer.")
        if positive_count is not None and (
            not isinstance(positive_count, int) or isinstance(positive_count, bool)
        ):
            raise TypeError("positive_count must be an integer or None.")
        if window < 1 or dimension < 1 or dimension > window:
            raise ValueError(
                "window and dimension must satisfy 1 <= dimension <= window."
            )
        if kind not in _STRUCTURED_KINDS:
            choices = ", ".join(sorted(_STRUCTURED_KINDS))
            raise ValueError(f"kind must be one of: {choices}.")
        dtype = _output_dtype(dtype)
        nonconstant = dimension - int(include_constant)
        if nonconstant < 0:
            raise ValueError("dimension is too small for a constant column.")
        if kind == "constant" and dimension != 1:
            raise ValueError("constant codes have dimension one.")
        columns = []
        if include_constant:
            columns.append(torch.ones(window, 1, dtype=dtype))
        structured = _structured_columns(
            kind,
            window,
            nonconstant,
            include_constant=include_constant,
            positive_count=positive_count,
            seed=seed,
            dtype=dtype,
        )
        if center:
            structured = _center_columns(structured)
        if normalize:
            structured = _normalize_columns(structured)
        if structured.shape[1]:
            columns.append(structured)
        matrix = torch.cat(columns, dim=1).to(device=device)
        return cls._certified(matrix, family="structured")

    @property
    def family(self) -> _CodeFamily:
        """Construction family and associated semantic guarantees."""
        return self._family

    @property
    def matrix(self) -> Tensor:
        """Code matrix; fixed matrices are returned as defensive copies."""
        return self._matrix if self._matrix.requires_grad else self._matrix.clone()

    def _runtime_matrix(self) -> Tensor:
        """Return owned storage for plan and module execution."""
        return self._matrix

    @property
    def temporal_degrees(self) -> tuple[int, ...] | None:
        """Certified Pascal/Newton filtration degrees, when available."""
        return self._temporal_degrees

    @property
    def window(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def retains_physical_offsets(self) -> bool:
        return self.family == "identity"

    @property
    def is_graded(self) -> bool:
        return self.temporal_degrees is not None

    @property
    def rank(self) -> int:
        matrix = self._matrix.detach().to(device="cpu", dtype=torch.float64)
        return int(torch.linalg.matrix_rank(matrix).item())

    @property
    def condition_number(self) -> float:
        matrix = self._matrix.detach().to(device="cpu", dtype=torch.float64)
        return float(torch.linalg.cond(matrix).item())

    def shift_operator(self, stride: int = 1) -> Tensor:
        """Return the exact backward rebase for a sliding-window update.

        If ``A`` is this code's matrix, the returned operator ``L`` satisfies
        ``A[stride:] @ L == A[:-stride]`` up to storage-dtype rounding. Thus a
        surviving letter formerly at offset ``r + stride`` is relabelled to
        offset ``r``. Codes without this finite shift closure are rejected.
        """
        if self._pascal_factor is None:
            return _exact_shift_operator(self._matrix, stride).clone()
        _positive_integer(stride, "stride")
        if stride >= self.window:
            raise ValueError("stride must be smaller than the code window.")
        return self._pascal_shift_operator(-stride).to(
            device=self._matrix.device, dtype=self._matrix.dtype
        )

    def _pascal_shift_operator(self, shift: int = -1) -> Tensor:
        """Return canonical float64 transport for a signed Pascal translation."""
        if not isinstance(shift, int) or isinstance(shift, bool):
            raise ValueError("shift must be an integer.")
        factor = self._pascal_factor
        if factor is None:
            raise ValueError("Pascal transport requires certified construction data.")
        coefficients = [
            math.comb(shift, degree)
            if shift >= 0 and degree <= shift
            else (-1) ** degree * math.comb(-shift + degree - 1, degree)
            if shift < 0
            else 0
            for degree in range(self.dimension)
        ]
        translation = torch.tensor(
            [
                [coefficients[j - i] if j >= i else 0 for j in range(self.dimension)]
                for i in range(self.dimension)
            ],
            dtype=torch.float64,
        )
        return torch.linalg.solve_triangular(
            factor.T, (factor @ translation).T, upper=False
        ).T

    @property
    def fingerprint(self) -> str:
        raw = (
            self._matrix.detach()
            .to(device="cpu")
            .contiguous()
            .view(torch.uint8)
            .flatten()
            .tolist()
        )
        metadata = (
            tuple(self._matrix.shape),
            str(self._matrix.dtype),
            self.family,
            self.temporal_degrees,
        )
        digest = hashlib.sha256(repr(metadata).encode())
        digest.update(bytes(raw))
        if self._pascal_factor is not None:
            digest.update(bytes([self._pascal_raw]))
            digest.update(
                bytes(
                    self._pascal_factor.contiguous()
                    .view(torch.uint8)
                    .flatten()
                    .tolist()
                )
            )
        return digest.hexdigest()[:16]

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "OffsetCode":
        """Return the same semantic code on another device or dtype."""
        matrix = self._matrix.to(device=device, dtype=dtype)
        return OffsetCode._certified(
            matrix,
            self.family,
            self.temporal_degrees,
            pascal_factor=self._pascal_factor,
            pascal_raw=self._pascal_raw,
        )

    def as_alphabet_map(self, channels: int) -> "AlphabetMap":
        """Return the channel-equivariant ``A tensor I`` alphabet map."""
        _positive_integer(channels, "channels")
        eye = torch.eye(channels, device=self._matrix.device, dtype=self._matrix.dtype)
        matrix = self._matrix[:, None, None, :] * eye[None, :, :, None]
        return AlphabetMap(
            matrix.reshape(self.window, channels, channels * self.dimension)
        )


@dataclass(frozen=True, eq=False, init=False)
class AlphabetMap:
    """An arbitrary offset-dependent map ``B_r: E -> V``.

    Unlike ``OffsetCode``, this object may mix channels and therefore does
    not inherit colored-descent or channel-equivariant guarantees.
    """

    _matrix: Tensor = field(repr=False)

    def __init__(self, matrix: Tensor) -> None:
        _floating_matrix(matrix, "matrix", 3)
        owned = _owned_matrix(matrix)
        object.__setattr__(self, "_matrix", owned)
        self.__post_init__()

    def __post_init__(self) -> None:
        _floating_matrix(self._matrix, "matrix", 3)
        window, channels, output = self._matrix.shape
        if min(window, channels, output) < 1:
            raise ValueError("matrix dimensions must be positive.")

    @property
    def window(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def channels(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def output_dimension(self) -> int:
        return int(self._matrix.shape[2])

    @property
    def matrix(self) -> Tensor:
        """Projection matrix; fixed matrices are returned as defensive copies."""
        return self._matrix if self._matrix.requires_grad else self._matrix.clone()

    def _runtime_matrix(self) -> Tensor:
        """Return owned storage for plan and module execution."""
        return self._matrix

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "AlphabetMap":
        return AlphabetMap(self._matrix.to(device=device, dtype=dtype))


__all__ = ["AlphabetMap", "OffsetCode"]
