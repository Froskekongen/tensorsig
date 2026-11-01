"""Integration checks for quotient-native compilation."""

from __future__ import annotations

import pytest
import torch

from tensorsig.classical.signatures import log_signature
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.quotients import (
    DependenceGraph,
    LieQuotient,
    OffsetSpan,
    TemporalDegree,
)
from tensorsig.offset_coded_signatures._lie_quotients.compiler import (
    compile_lie_quotient,
    quotient_log_signature_from_increment_windows,
)
from tensorsig.offset_coded_signatures._lie_quotients.rules import (
    OffsetAlphabet,
    ProjectedAlphabet,
)
from tests.quotient_oracles import project_ambient


def _path_from_increments(increments: torch.Tensor) -> torch.Tensor:
    origin = torch.zeros(
        increments.shape[0],
        1,
        increments.shape[-1],
        device=increments.device,
        dtype=increments.dtype,
    )
    return torch.cat((origin, increments.cumsum(dim=1)), dim=1)


def test_public_compiler_and_runtime_do_not_touch_ambient_hall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorsig.algebra.hall_bch as hall_bch
    import tensorsig.offset_coded_signatures._lie_quotients.compiler as quotient_module
    import tensorsig.offset_coded_signatures._lie_quotients.upper_set_normal_form as upper

    quotient_module._compile_lie_quotient_cached.cache_clear()
    upper._compile_upper_set_hall_normal_form_cached.cache_clear()

    def fail_ambient(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "quotient-native execution must not build ambient Hall data"
        )

    monkeypatch.setattr(hall_bch, "_sparse_constants", fail_ambient)

    cases = (
        (
            LieQuotient(TemporalDegree(1)),
            ProjectedAlphabet.newton(3, 2),
        ),
        (
            LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
            ProjectedAlphabet(1, 3),
        ),
    )
    for quotient, alphabet in cases:
        plan = compile_lie_quotient(quotient, alphabet, max_depth=3)
        windows = torch.randn(2, 1, 3, alphabet.in_channels)
        code = torch.randn(3, alphabet.code_dim)
        output = quotient_log_signature_from_increment_windows(windows, plan, code)
        tensor_output = quotient_log_signature_from_increment_windows(
            windows,
            plan,
            code,
            method="tensor",
        )

        assert output.shape == (2, 1, plan.dimension)
        torch.testing.assert_close(output, tensor_output, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    ("quotient", "alphabet"),
    [
        (
            LieQuotient(TemporalDegree(1)),
            ProjectedAlphabet.newton(2, 1),
        ),
        (
            LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
            ProjectedAlphabet(1, 3),
        ),
    ],
)
def test_depth_five_public_runtime_matches_full_signature_then_projection(
    quotient: LieQuotient,
    alphabet: ProjectedAlphabet,
) -> None:
    plan = compile_lie_quotient(quotient, alphabet, max_depth=5)
    generator = torch.Generator().manual_seed(20260803)
    windows = torch.randn(
        2,
        1,
        3,
        alphabet.in_channels,
        generator=generator,
        dtype=torch.float64,
    )
    code = (
        OffsetCode.pascal(3, alphabet.code_dim, dtype=torch.float64).matrix
        if alphabet.temporal_degrees is not None
        else torch.ones(3, alphabet.code_dim, dtype=torch.float64)
    )

    direct = quotient_log_signature_from_increment_windows(windows, plan, code)[:, 0]
    tensor = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        code,
        method="tensor",
    )[:, 0]
    torch.testing.assert_close(direct, tensor, atol=2e-9, rtol=2e-9)
    lifted = torch.einsum("bnrc,re->bnrce", windows, code).flatten(-2)[:, 0]
    full = log_signature(
        _path_from_increments(lifted),
        depth=5,
        method="default",
        mode="hall",
    )

    torch.testing.assert_close(
        direct,
        project_ambient(full, plan),
        atol=2e-9,
        rtol=2e-9,
    )


def test_ambient_dimension_uses_witt_formula_without_hall_enumeration() -> None:
    import tensorsig.offset_coded_signatures._lie_quotients.compiler as quotient_module

    quotient_module._compile_lie_quotient_cached.cache_clear()
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(0)),
        ProjectedAlphabet.newton(2, 1),
        max_depth=6,
    )

    assert not hasattr(quotient_module, "hall_basis")
    assert plan.ambient_dimension == 23


def test_bound_plan_moves_precompiled_word_bridge() -> None:
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)),
        ProjectedAlphabet.newton(2, 1),
        max_depth=4,
    )
    original_bridge = plan._word_bridge

    moved = plan.to(dtype=torch.float32)

    assert moved._word_bridge_cache is not None
    assert moved._word_bridge_cache.ambient_words == original_bridge.ambient_words
    assert moved._word_bridge_cache.quotient_matrix.dtype == torch.float32


def test_words_bridge_uses_sparse_storage_only_for_strong_pruning() -> None:
    strong = compile_lie_quotient(
        LieQuotient(TemporalDegree(0)),
        ProjectedAlphabet.newton(2, 3),
        max_depth=4,
    )
    weak = compile_lie_quotient(
        LieQuotient(),
        ProjectedAlphabet(1, 2),
        max_depth=3,
    )

    assert strong._word_bridge.quotient_matrix.layout == torch.sparse_coo
    assert weak._word_bridge.quotient_matrix.layout == torch.strided

    coordinates = torch.randn(
        2,
        len(strong._word_bridge.ambient_words),
        dtype=torch.float64,
        requires_grad=True,
    )
    strong._word_bridge.project(coordinates).square().sum().backward()

    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()


def test_tensor_runtime_supports_full_offset_quotients_and_gradients() -> None:
    plan = compile_lie_quotient(
        LieQuotient(OffsetSpan(1)),
        OffsetAlphabet(kernel_size=3, in_channels=2),
        max_depth=3,
    )
    windows = torch.randn(
        2,
        2,
        3,
        2,
        dtype=torch.float64,
        requires_grad=True,
    )

    direct = quotient_log_signature_from_increment_windows(windows, plan)
    tensor = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        method="tensor",
    )

    torch.testing.assert_close(direct, tensor, atol=2e-9, rtol=2e-9)
    tensor.square().sum().backward()
    assert windows.grad is not None and torch.isfinite(windows.grad).all()


def test_runtime_device_dtype_binding_is_memoized_for_both_methods() -> None:
    plan = compile_lie_quotient(
        LieQuotient(TemporalDegree(1)),
        ProjectedAlphabet.newton(2, 1),
        max_depth=3,
    )
    windows = torch.randn(1, 1, 3, 1, dtype=torch.float32)
    phase = torch.randn(3, 2, dtype=torch.float32)

    direct = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        phase,
        method="direct",
    )
    bound = plan.to(dtype=torch.float32)
    assert bound is plan.to(dtype=torch.float32)

    tensor = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        phase,
        method="tensor",
    )
    bridge = bound._word_bridge
    quotient_log_signature_from_increment_windows(
        windows,
        plan,
        phase,
        method="tensor",
    )

    assert bound._word_bridge is bridge
    torch.testing.assert_close(direct, tensor, atol=2e-5, rtol=2e-5)


def test_tensor_runtime_does_not_reconstruct_large_dynamic_range_path() -> None:
    plan = compile_lie_quotient(
        LieQuotient(),
        ProjectedAlphabet(code_dim=1, in_channels=2),
        max_depth=2,
    )
    windows = torch.tensor(
        [[[[1.0e8, 1.0e8], [1.0, -1.0], [-1.0e8, -1.0e8]]]],
        dtype=torch.float32,
    )
    phase = torch.ones(3, 1)

    direct = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        phase,
        method="direct",
    )
    tensor = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        phase,
        method="tensor",
    )

    assert float(direct[..., 2].abs().max()) > 1.0e7
    torch.testing.assert_close(direct, tensor, atol=8.0, rtol=1e-6)


@pytest.mark.parametrize("depth", range(1, 6))
@pytest.mark.parametrize(
    ("quotient", "alphabet"),
    [
        (LieQuotient(), ProjectedAlphabet(1, 2)),
        (
            LieQuotient(TemporalDegree(1)),
            ProjectedAlphabet.newton(2, 1),
        ),
        (
            LieQuotient(DependenceGraph(((0, 1), (1, 2)))),
            ProjectedAlphabet(1, 3),
        ),
        (
            LieQuotient(OffsetSpan(0)),
            OffsetAlphabet(2, 1),
        ),
    ],
)
def test_direct_and_tensor_methods_agree_across_quotient_families_and_depths(
    quotient: LieQuotient,
    alphabet: ProjectedAlphabet | OffsetAlphabet,
    depth: int,
) -> None:
    plan = compile_lie_quotient(quotient, alphabet, max_depth=depth)
    kernel_size = alphabet.kernel_size if isinstance(alphabet, OffsetAlphabet) else 3
    generator = torch.Generator().manual_seed(1000 + 10 * alphabet.width + depth)
    windows = torch.randn(
        2,
        1,
        kernel_size,
        alphabet.in_channels,
        generator=generator,
        dtype=torch.float64,
    )
    phase = (
        None
        if isinstance(alphabet, OffsetAlphabet)
        else torch.randn(
            kernel_size,
            alphabet.code_dim,
            generator=generator,
            dtype=torch.float64,
        )
    )

    direct = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        phase,
        method="direct",
    )
    tensor = quotient_log_signature_from_increment_windows(
        windows,
        plan,
        phase,
        method="tensor",
    )

    torch.testing.assert_close(direct, tensor, atol=2e-9, rtol=2e-9)
