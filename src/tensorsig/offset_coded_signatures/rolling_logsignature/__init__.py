"""Explicit quotient-log FIFO kernels and differentiable grade scans."""

from math import comb

import torch
from torch import Tensor

from .._lie_quotients.compiler import (
    LieQuotientPlan,
    quotient_log_signature_from_increment_windows,
)
from .._lie_quotients.rules import (
    ChannelMultiplicity,
    ChannelSupport,
    DependenceGraph,
    ProjectedAlphabet,
    TemporalDegree,
)
from ..codes import _positive_integer
from ..windows import unfold_increments
from ._bch import DegreeOneBCH
from ._compile import Transport, pascal_transport
from ._program import BCHProgram
from ._scan import optimized_scan_interval, prepare_powers
from ._joint import general_bch_program, joint_program

__all__ = [
    "RollingLogSignature",
    "Transport",
    "pascal_transport",
    "general_bch_program",
    "joint_program",
]


class RollingLogSignature:
    """Device-bound rolling evaluation on fixed raw Pascal codes.

    Input: increments (batch, time, channels). Output: quotient coordinates
    (batch, windows, dimension), in exactly the supplied LieQuotientPlan basis.
    Compilation is separate from evaluation and never enumerates ambient words.
    Only known Pascal-stable rules are accepted, including their conjunctions.
    """

    def __init__(self, plan: LieQuotientPlan, window: int):
        _positive_integer(window, "window")
        alphabet = plan.alphabet
        if not isinstance(alphabet, ProjectedAlphabet):
            raise ValueError("Rolling evaluation requires a ProjectedAlphabet.")
        if alphabet.temporal_degrees != tuple(range(alphabet.code_dim)):
            raise ValueError("Use ProjectedAlphabet.newton for raw Pascal codes.")
        if alphabet.code_dim > window:
            raise ValueError("code_dim must not exceed window.")
        allowed = (TemporalDegree, ChannelMultiplicity, ChannelSupport, DependenceGraph)
        if any(type(rule) not in allowed for rule in plan.quotient.rules):
            raise ValueError("Quotient rule has no certified Pascal-shift invariance.")
        self.plan = plan
        self.window = window
        self.code = torch.tensor(
            [[comb(r, p) for p in range(alphabet.code_dim)] for r in range(window)],
            dtype=plan.generator_map.dtype,
            device=plan.generator_map.device,
        )
        self.transport_rows = pascal_transport(plan)
        self.transport = Transport(plan, self.transport_rows)
        self.append = DegreeOneBCH(plan)
        # Precontract lifting with the quotient generator map once per offset.
        self.embeddings = torch.einsum(
            "rp,cpd->rcd",
            self.code,
            plan.generator_map.reshape(
                alphabet.in_channels, alphabet.code_dim, plan.dimension
            ),
        )

        self._programs = {}
        self._powers = [self.transport]
        self._joint_programs = {}
        self._general_programs = {}
        self._scan_stages = {}

    def _joint_program(self, grade=None, compiled=False):
        key = (grade, compiled)
        if key not in self._joint_programs:
            self._joint_programs[key] = joint_program(self, grade, compiled)
        return self._joint_programs[key]

    def _general_program(self, compiled=False, right_degree_one=False):
        key = (compiled, right_degree_one)
        if key not in self._general_programs:
            self._general_programs[key] = general_bch_program(
                self.plan, compiled, right_degree_one
            )
        return self._general_programs[key]

    @staticmethod
    def _derivative_order(order):
        if type(order) is not int or order not in {0, 1, 2}:
            raise ValueError("derivative_order must be 0, 1, or 2.")

    def _endpoints(self, backend):
        if backend in {"joint", "joint_fused"}:
            return (
                self._general_program(backend == "joint_fused", right_degree_one=True),
            ) * 3
        if backend == "typed":
            return (self.append,) * 3
        if backend != "fused":
            raise ValueError("backend must be typed, fused, joint, or joint_fused.")
        if backend not in self._programs:
            programs = [BCHProgram(self.plan)]
            for embedding in (self.embeddings[0], self.embeddings[-1]):
                support = torch.where(embedding.any(dim=0))[0].cpu().tolist()
                programs.append(BCHProgram(self.plan, endpoint_support=support))
            self._programs[backend] = tuple(
                torch.compile(program, fullgraph=True) for program in programs
            )
        return self._programs[backend]

    def _scan_powers(self, count):
        return prepare_powers(self, count)

    def scan(
        self,
        increments,
        *,
        stride=1,
        backend="joint",
        shift_backend="dense",
        refresh_every=None,
        chunk_windows=None,
        derivative_order=1,
    ):
        """Scan unit windows then select stride. Chunk length counts unit windows.

        A fixed refresh/chunk partition is initialized independently. Zero
        refresh disables resets; default refresh is one window length.
        Transport uses dense per-grade power blocks; sparse is unsupported.
        Powered matrices are lazy setup, available via _scan_powers for timing.
        """
        self._validate(increments, stride)
        self._derivative_order(derivative_order)
        if derivative_order == 2 and backend == "fused":
            raise ValueError(
                "Use joint_fused for owned second derivatives; fused uses AOTAutograd."
            )
        if shift_backend != "dense":
            raise ValueError("Affine scan supports only shift_backend='dense'.")
        self._endpoints(backend)
        refresh = self.window if refresh_every is None else refresh_every
        if not isinstance(refresh, int) or isinstance(refresh, bool) or refresh < 0:
            raise ValueError("refresh_every must be a nonnegative integer or None.")
        count = increments.shape[1] - self.window + 1
        length = count
        if chunk_windows is not None:
            _positive_integer(chunk_windows, "chunk_windows")
            length = min(length, chunk_windows)
        if refresh:
            length = min(length, refresh)
        starts = torch.arange(0, count, length, device=increments.device)
        states = optimized_scan_interval(self, increments, starts, length, backend)
        return states.flatten(1, 2)[:, :count:stride]

    def _validate(self, increments, stride):
        _positive_integer(stride, "stride")
        if (
            increments.ndim != 3
            or increments.shape[-1] != self.plan.alphabet.in_channels
        ):
            raise ValueError("Expected increments with shape (batch, time, channels).")
        if increments.shape[1] < self.window:
            raise ValueError("The stream must contain at least window increments.")
        if increments.device != self.code.device or increments.dtype != self.code.dtype:
            raise ValueError(
                "Bind the quotient plan to the input device and dtype before compilation."
            )

    def reference(self, increments: Tensor, *, stride=1, method="direct") -> Tensor:
        """Existing independent-window baseline, including lifting and readout."""
        self._validate(increments, stride)
        return quotient_log_signature_from_increment_windows(
            unfold_increments(increments, self.window, stride),
            self.plan,
            self.code,
            method=method,
        )

    def __call__(
        self,
        increments: Tensor,
        *,
        stride=1,
        backend=None,
        shift_backend="dense",
        chunk_windows: int | None = None,
        refresh_every: int | None = None,
        algorithm="chunks",
        derivative_order=1,
    ) -> Tensor:
        """Roll typed, fused, or joint BCH circuits in independent chunks.

        An omitted backend means typed for chunks and joint for scan.
        Joint chunks integrate transport into their polynomial and accept
        only the default dense spelling; typed/fused chunks also support sparse.
        chunk_windows=None uses one sequential lane; positive values initialize
        independent chunks together and then roll lanes in parallel. Refresh is
        every window increments by default. Zero disables refresh (for drift
        experiments). Strides use successive unit updates between outputs.
        """
        self._derivative_order(derivative_order)
        if backend is None:
            backend = "typed" if algorithm == "chunks" else "joint"
        if algorithm == "scan":
            return self.scan(
                increments,
                stride=stride,
                backend=backend,
                shift_backend=shift_backend,
                refresh_every=refresh_every,
                chunk_windows=chunk_windows,
                derivative_order=derivative_order,
            )
        if algorithm != "chunks":
            raise ValueError("algorithm must be chunks or scan.")
        if derivative_order == 2 and backend == "fused":
            raise ValueError(
                "Use joint_fused for owned second derivatives; fused uses AOTAutograd."
            )
        self._validate(increments, stride)
        if shift_backend not in {"dense", "sparse"}:
            raise ValueError("shift_backend must be 'dense' or 'sparse'.")
        if backend in {"joint", "joint_fused"} and shift_backend != "dense":
            raise ValueError("Joint chunks support only shift_backend='dense'.")
        append, remove, insert = self._endpoints(backend)
        if refresh_every is None:
            refresh_every = self.window
        if (
            not isinstance(refresh_every, int)
            or isinstance(refresh_every, bool)
            or refresh_every < 0
        ):
            raise ValueError("refresh_every must be a nonnegative integer or None.")
        count = 1 + (increments.shape[1] - self.window) // stride
        if chunk_windows is None:
            chunk_windows = count
        _positive_integer(chunk_windows, "chunk_windows")
        chunk_windows = min(chunk_windows, count)
        starts = (
            torch.arange(0, count, chunk_windows, device=increments.device) * stride
        )
        last_start = (count - 1) * stride

        def initialize(positions):
            state = increments.new_zeros(
                increments.shape[0], len(starts), self.plan.dimension
            )
            for r in range(self.window):
                state = append(state, increments[:, positions + r] @ self.embeddings[r])
            return state

        state = initialize(starts)
        outputs = [state]
        age = 0
        for index in range(1, chunk_windows):
            if stride >= self.window or (
                refresh_every and age + stride >= refresh_every
            ):
                state = initialize((starts + index * stride).clamp_max(last_start))
                age = 0
            else:
                for step in range(stride):
                    positions = starts + (index - 1) * stride + step
                    outgoing = (
                        increments[:, positions.clamp_max(increments.shape[1] - 1)]
                        @ self.embeddings[0]
                    )
                    incoming = (
                        increments[
                            :,
                            (positions + self.window).clamp_max(
                                increments.shape[1] - 1
                            ),
                        ]
                        @ self.embeddings[-1]
                    )
                    if backend in {"joint", "joint_fused"}:
                        state = self._joint_program(compiled=backend == "joint_fused")(
                            state, outgoing, incoming
                        )
                    else:
                        state = -remove(-state, outgoing)
                        state = self.transport(state, shift_backend)
                        state = insert(state, incoming)
                age += stride
            outputs.append(state)
        # The final lane may contain padded outputs; they are never returned.
        return torch.stack(outputs, dim=2).flatten(1, 2)[:, :count]
