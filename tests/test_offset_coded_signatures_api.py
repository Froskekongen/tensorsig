import pytest
import torch

import tensorsig.offset_coded_signatures as offset_coded_signatures
from tensorsig.classical.signatures import log_signature, signature
from tensorsig.offset_coded_signatures.codes import AlphabetMap, OffsetCode
from tensorsig.offset_coded_signatures.coordinates import (
    ChronologicalFiltration,
    WordSelection,
)
from tensorsig.offset_coded_signatures.layers import LocalFeatures, LocalLinear
from tensorsig.offset_coded_signatures.plans import (
    LocalRealization,
    LogSignaturePlan,
    PolynomialPlan,
    SignaturePlan,
    compile_polynomial,
)
from tensorsig.offset_coded_signatures.quotients import (
    ChannelSupport,
    DependenceGraph,
    LieQuotient,
    OffsetSpan,
    QuotientBasisElement,
    TemporalDegree,
)
from tensorsig.offset_coded_signatures.windows import increment_windows
from tensorsig.tensor_ops.series import unflatten_signature_levels


def _path(*, batch: int = 2, increments: int = 5, channels: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260804)
    dx = 0.2 * torch.randn(
        batch,
        increments,
        channels,
        generator=generator,
        dtype=torch.float64,
    )
    origin = torch.zeros(batch, 1, channels, dtype=torch.float64)
    return torch.cat((origin, dx.cumsum(dim=1)), dim=1)


def _explicit_lifted_paths(
    windows: torch.Tensor,
    code: OffsetCode,
) -> torch.Tensor:
    """Lift increment windows explicitly for an independent global oracle."""
    lifted = torch.einsum("bnrc,rp->bnrcp", windows, code.matrix).flatten(-2)
    batch, count, window, width = lifted.shape
    paths = torch.cat(
        (lifted.new_zeros(batch, count, 1, width), lifted.cumsum(dim=2)),
        dim=2,
    )
    return paths.reshape(batch * count, window + 1, width)


def _explicit_local_signature_levels(
    windows: torch.Tensor,
    code: OffsetCode,
    depth: int,
) -> tuple[torch.Tensor, ...]:
    """Evaluate dense local levels via the public global signature."""
    batch, count = windows.shape[:2]
    width = code.dimension * windows.shape[-1]
    levels = unflatten_signature_levels(
        signature(_explicit_lifted_paths(windows, code), depth=depth),
        width=width,
        depth=depth,
    )
    return tuple(level.reshape(batch, count, *level.shape[1:]) for level in levels)


def _flat_word_indices(words: torch.Tensor, width: int) -> torch.Tensor:
    indices = torch.zeros(words.shape[0], dtype=torch.long)
    for column in range(words.shape[1]):
        indices = indices * width + words[:, column]
    return indices


def test_public_surface_contains_only_the_clean_plan_api() -> None:
    expected = {
        "codes",
        "coordinates",
        "layers",
        "plans",
        "quotients",
        "windows",
        "graph_queries",
        "active_word_log",
        "rolling_logsignature",
        "pascal_rank",
        "pascal_certificate",
    }

    assert set(offset_coded_signatures.__all__) == expected
    for legacy_name in (
        "AlphabetProjection",
        "CausalEmbeddingWindowedLogSignature",
        "KernelizedLocalLogSignature",
        "LocalLogSignatureBasis",
        "LocalPolynomialSignature",
        "LearnableCausalFilterBank",
        "LogCoordinateSelection",
        "SignatureWordSelection",
        "build_local_log_signature_basis",
        "causal_filter_embedding",
        "causal_fir_filter_bank",
        "local_log_signature",
        "kernelized_local_polynomial_signature",
        "pascal_offset_code",
        "pairwise_product_channel_count",
        "pairwise_product_channels",
    ):
        assert not hasattr(offset_coded_signatures, legacy_name)
    assert not hasattr(OffsetCode, "custom")
    assert not hasattr(OffsetCode, "as_projection")
    assert not hasattr(LocalRealization, "logsignature")
    for plan_type in (
        SignaturePlan,
        LogSignaturePlan,
        PolynomialPlan,
    ):
        with pytest.raises(TypeError, match="objects are compiled"):
            plan_type()


def test_offset_code_metadata_alphabet_map_and_validation() -> None:
    code = OffsetCode.pascal(4, 2, dtype=torch.float64)

    assert code.family == "pascal"
    assert code.temporal_degrees == (0, 1)
    assert code.window == 4
    assert code.dimension == 2
    assert code.is_graded
    assert not code.retains_physical_offsets
    assert code.rank == 2
    assert code.condition_number == pytest.approx(1.0)
    assert code.fingerprint == OffsetCode.pascal(4, 2, dtype=torch.float64).fingerprint

    projection = code.as_alphabet_map(channels=3)
    assert projection.matrix.shape == (4, 3, 6)
    assert projection.window == 4
    assert projection.channels == 3
    assert projection.output_dimension == 6

    identity = OffsetCode.identity(3, dtype=torch.float64)
    assert identity.retains_physical_offsets
    assert not identity.is_graded
    torch.testing.assert_close(identity.matrix, torch.eye(3, dtype=torch.float64))

    with pytest.raises(ValueError, match="2 dimensions"):
        OffsetCode(torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="floating-point"):
        OffsetCode(torch.ones(3, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        OffsetCode(torch.tensor([[1.0], [float("nan")]]))
    with pytest.raises(ValueError, match="1 <= P <= R"):
        OffsetCode(torch.ones(1, 2))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        OffsetCode(
            torch.ones(2, 1),
            family="pascal",
            temporal_degrees=(0,),
        )
    assert OffsetCode(torch.ones(2, 1)).family == "matrix"

    source = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    owned = OffsetCode(source)
    source.zero_()
    exposed = owned.matrix
    exposed.zero_()
    torch.testing.assert_close(
        owned.matrix,
        torch.tensor([[1.0], [2.0]], dtype=torch.float64),
    )


def test_graded_selections_refuse_an_ungraded_matrix_code() -> None:
    code = OffsetCode(
        torch.tensor(
            [[1.0, 0.0], [0.5, 1.0], [-1.0, 0.25]],
            dtype=torch.float64,
        )
    )
    realization = LocalRealization(code, channels=2, max_depth=3)

    with pytest.raises(ValueError, match="graded OffsetCode"):
        realization.signature(selection=WordSelection(max_temporal_degree=1))
    with pytest.raises(ValueError, match="graded OffsetCode"):
        realization.log_signature(filtration=ChronologicalFiltration(1))


def test_clean_api_rejects_integer_inputs_and_legacy_backend_names() -> None:
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    realization = LocalRealization(code, channels=2, max_depth=2)
    plan = realization.log_signature()

    with pytest.raises(ValueError, match="floating-point dtype"):
        increment_windows(torch.ones(1, 5, 2, dtype=torch.long), size=3)
    with pytest.raises(ValueError, match="floating-point dtype"):
        plan.evaluate(torch.ones(1, 1, 3, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="dense.*sparse"):
        plan.evaluate(increment_windows(_path(), size=3), backend="signature")
    with pytest.raises(ValueError, match="backend"):
        LocalFeatures(plan, backend="DENSE")

    quotient = realization.log_signature(quotient=LieQuotient(ChannelSupport(1)))
    with pytest.raises(ValueError, match="direct.*pbw.*tensor"):
        quotient.evaluate(
            increment_windows(_path(), size=3),
            backend="bch_sparse",
        )


def test_chronological_signature_matches_global_signature_and_reconstructs_dense() -> (
    None
):
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    realization = LocalRealization(code, channels=2, max_depth=3)
    plan = realization.signature()
    dense_plan = realization.signature(coordinates="dense")
    windows = increment_windows(_path(), size=3)

    features = plan.evaluate(windows)
    explicit_levels = _explicit_local_signature_levels(windows, code, depth=3)
    explicit_features = torch.cat(
        [
            level.flatten(start_dim=2).index_select(
                -1,
                _flat_word_indices(words, plan.output_dimension),
            )
            for level, words in zip(
                explicit_levels,
                plan.word_indices,
                strict=True,
            )
        ],
        dim=-1,
    )

    assert plan.coordinates == "chronological"
    assert plan.certification == "structural"
    assert plan.is_lossless
    assert features.shape[-1] == plan.feature_count
    torch.testing.assert_close(features, explicit_features, atol=1e-12, rtol=1e-10)
    for actual, expected in zip(
        plan.reconstruct(features),
        explicit_levels,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)
    dense_features = dense_plan.evaluate(windows)
    torch.testing.assert_close(
        dense_features,
        torch.cat([level.flatten(start_dim=2) for level in explicit_levels], dim=-1),
        atol=1e-12,
        rtol=1e-10,
    )
    for actual, expected in zip(
        dense_plan.reconstruct(dense_features),
        explicit_levels,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)


def test_no_op_and_cross_depth_word_selections_keep_correct_losslessness() -> None:
    realization = LocalRealization(
        OffsetCode.pascal(3, 2, dtype=torch.float64),
        channels=2,
        max_depth=3,
    )
    windows = increment_windows(_path(), size=3)
    full = realization.signature()
    no_op = realization.signature(selection=WordSelection())
    mixed_depth_sets = realization.signature(
        selection=WordSelection(descent_sets=((), (1,)))
    )

    assert no_op.is_lossless
    torch.testing.assert_close(no_op.evaluate(windows), full.evaluate(windows))
    assert not mixed_depth_sets.is_lossless
    assert mixed_depth_sets.level_dimensions[0] == full.level_dimensions[0]


def test_fixed_matrix_code_uses_a_numerical_lossless_chart() -> None:
    code = OffsetCode(
        torch.tensor(
            [[1.0, 0.0], [0.5, 1.0], [-1.0, 0.25]],
            dtype=torch.float64,
        )
    )
    realization = LocalRealization(code, channels=2, max_depth=3)
    chronological = realization.signature()
    dense = realization.signature(coordinates="dense")
    windows = increment_windows(_path(), size=3)

    features = chronological.evaluate(windows)
    reconstructed = torch.cat(
        [level.flatten(start_dim=2) for level in chronological.reconstruct(features)],
        dim=-1,
    )

    assert chronological.certification == "numerical"
    assert chronological.is_lossless
    torch.testing.assert_close(reconstructed, dense.evaluate(windows))


def test_general_projection_agrees_with_the_separable_code_pushforward() -> None:
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    projection = AlphabetMap(code.as_alphabet_map(channels=2).matrix)
    windows = increment_windows(_path(), size=3)
    coded = LocalRealization(code, channels=2, max_depth=3).signature(
        coordinates="dense"
    )
    projected = LocalRealization(projection, channels=2, max_depth=3).signature()

    torch.testing.assert_close(coded.evaluate(windows), projected.evaluate(windows))


def test_channel_mixing_projection_is_the_tensor_algebra_pushforward() -> None:
    windows = increment_windows(_path(), size=3)
    generator = torch.Generator().manual_seed(41)
    matrix = torch.randn(3, 2, 3, generator=generator, dtype=torch.float64)
    plan = LocalRealization(
        AlphabetMap(matrix),
        channels=2,
        max_depth=3,
    ).signature()

    physical = windows.new_zeros(*windows.shape[:3], 3 * 2)
    for offset in range(3):
        physical[:, :, offset, 2 * offset : 2 * (offset + 1)] = windows[:, :, offset]
    zero = physical.new_zeros(*physical.shape[:2], 1, physical.shape[-1])
    physical_paths = torch.cat((zero, physical.cumsum(dim=2)), dim=2)
    physical_signature = signature(
        physical_paths.flatten(0, 1),
        depth=3,
    ).reshape(*physical.shape[:2], -1)
    physical_levels = unflatten_signature_levels(
        physical_signature,
        width=6,
        depth=3,
    )
    alphabet_map = matrix.reshape(6, 3)
    expected_levels = []
    tensor_map = torch.ones(1, 1, dtype=torch.float64)
    for depth, physical_level in enumerate(physical_levels, start=1):
        tensor_map = torch.kron(tensor_map, alphabet_map)
        expected_levels.append(physical_level.flatten(start_dim=2) @ tensor_map)

    torch.testing.assert_close(
        plan.evaluate(windows),
        torch.cat(expected_levels, dim=-1),
        atol=1e-11,
        rtol=1e-9,
    )


def test_pascal_gauge_change_pushes_forward_every_dense_level() -> None:
    channels = 2
    raw_code = OffsetCode.pascal(
        3,
        2,
        orthonormal=False,
        dtype=torch.float64,
    )
    qr_code = OffsetCode.pascal(3, 2, orthonormal=True, dtype=torch.float64)
    raw = LocalRealization(raw_code, channels, 3).signature(coordinates="dense")
    qr = LocalRealization(qr_code, channels, 3).signature(coordinates="dense")
    windows = increment_windows(_path(), size=3)

    gauge = torch.linalg.lstsq(raw_code.matrix, qr_code.matrix).solution
    alphabet_map = torch.kron(
        torch.eye(channels, dtype=torch.float64),
        gauge.contiguous(),
    )
    raw_features = raw.evaluate(windows)
    qr_features = qr.evaluate(windows)
    tensor_map = torch.ones(1, 1, dtype=torch.float64)
    for raw_slice, qr_slice in zip(raw.level_slices, qr.level_slices, strict=True):
        tensor_map = torch.kron(tensor_map, alphabet_map)
        torch.testing.assert_close(
            raw_features[..., raw_slice] @ tensor_map,
            qr_features[..., qr_slice],
            atol=1e-11,
            rtol=1e-9,
        )


@pytest.mark.parametrize(("window", "channels"), ((2, 1), (3, 2), (4, 3)))
def test_offset_forgetting_kernel_is_exactly_the_contrast_space(
    window: int,
    channels: int,
) -> None:
    projection = OffsetCode(torch.ones(window, 1, dtype=torch.float64)).as_alphabet_map(
        channels
    )
    alphabet_map = projection.matrix.reshape(window * channels, channels)
    contrasts = []
    for channel in range(channels):
        for offset in range(1, window):
            contrast = torch.zeros(window * channels, dtype=torch.float64)
            contrast[channel] = -1.0
            contrast[offset * channels + channel] = 1.0
            contrasts.append(contrast)
    contrast_matrix = torch.stack(contrasts)

    nullity = alphabet_map.shape[0] - int(torch.linalg.matrix_rank(alphabet_map))
    assert nullity == channels * (window - 1)
    assert int(torch.linalg.matrix_rank(contrast_matrix)) == nullity
    torch.testing.assert_close(
        contrast_matrix @ alphabet_map,
        torch.zeros(nullity, channels, dtype=torch.float64),
    )


def test_minimal_logsignature_reconstructs_the_global_logsignature() -> None:
    path = _path()
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    plan = LocalRealization(code, channels=2, max_depth=3).log_signature()

    windows = increment_windows(path, size=3)
    actual = plan.evaluate(windows, backend="dense")
    sparse = plan.evaluate(windows, backend="sparse")

    assert plan.quotient is None
    assert plan.is_lossless
    assert plan.feature_count == actual.shape[-1]
    torch.testing.assert_close(sparse, actual, atol=1e-11, rtol=1e-9)

    ambient = plan.reconstruct(actual)
    explicit = log_signature(
        _explicit_lifted_paths(windows, code),
        depth=3,
        mode="words",
    ).reshape_as(ambient)
    torch.testing.assert_close(ambient, explicit, atol=1e-11, rtol=1e-9)


def test_filtered_logsignature_is_backend_independent_and_explicitly_lossy() -> None:
    path = _path()
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    plan = LocalRealization(code, channels=2, max_depth=3).log_signature(
        filtration=ChronologicalFiltration(1)
    )
    windows = increment_windows(path, size=3)

    actual = plan.evaluate(windows, backend="dense")
    sparse = plan.evaluate(windows, backend="sparse")

    assert not plan.is_lossless
    torch.testing.assert_close(sparse, actual, atol=1e-11, rtol=1e-9)
    with pytest.raises(ValueError, match="not reconstructive"):
        plan.reconstruct(actual)

    full = LocalRealization(code, channels=2, max_depth=3).log_signature()
    no_op = LocalRealization(code, channels=2, max_depth=3).log_signature(
        filtration=ChronologicalFiltration(99)
    )
    full_features = full.evaluate(windows)
    no_op_features = no_op.evaluate(windows)
    assert no_op.is_lossless
    torch.testing.assert_close(no_op_features, full_features)
    torch.testing.assert_close(
        no_op.reconstruct(no_op_features),
        full.reconstruct(full_features),
    )


def test_rank_tolerance_controls_numerical_chart_discovery() -> None:
    nearly_singular = OffsetCode(
        torch.tensor(
            [[1.0, 1.0], [0.0, 1e-7], [1.0, 1.0 + 1e-7]],
            dtype=torch.float64,
        )
    )
    realization = LocalRealization(nearly_singular, channels=2, max_depth=2)
    fine = realization.log_signature(rank_tolerance=1e-12)
    coarse = realization.log_signature(rank_tolerance=1e-4)

    assert fine.level_dimensions == (4, 6)
    assert coarse.level_dimensions == (2, 1)
    assert coarse.feature_count < fine.feature_count
    assert coarse.rank_tolerance == pytest.approx(1e-4)
    assert fine.chart_margin is not None
    assert coarse.chart_margin is not None
    assert coarse.chart_margin > fine.chart_margin

    signature = realization.signature(rank_tolerance=1e-4)
    assert signature.certification == "numerical"
    assert signature.rank_tolerance == pytest.approx(1e-4)
    assert signature.chart_margin is not None

    for invalid in (True, float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="finite non-negative"):
            realization.log_signature(rank_tolerance=invalid)
        with pytest.raises(ValueError, match="finite non-negative"):
            realization.signature(rank_tolerance=invalid)


def test_rank_tolerance_is_applied_to_the_global_singular_spectrum_once() -> None:
    from tensorsig.offset_coded_signatures._charts.primitive import (
        _ordered_independent_rows,
        _rank_revealing_subspaces,
    )

    matrix = torch.tensor([[0.4], [0.4]], dtype=torch.float64)
    row_subspace, column_subspace, tolerance = _rank_revealing_subspaces(
        matrix,
        tolerance=0.52,
    )

    assert tolerance == pytest.approx(0.52)
    assert row_subspace.shape == (2, 1)
    assert column_subspace.shape == (1, 1)
    assert _ordered_independent_rows(row_subspace, range(2)) == (0,)


def test_coarse_rank_quotient_chart_selects_deterministic_actual_words() -> None:
    realization = LocalRealization(
        OffsetCode.pascal(
            3,
            3,
            orthonormal=False,
            dtype=torch.float64,
        ),
        channels=1,
        max_depth=3,
    )
    quotient = LieQuotient(TemporalDegree(1))

    first = realization.log_signature(
        quotient=quotient,
        rank_tolerance=0.52,
    )
    second = realization.log_signature(
        quotient=quotient,
        rank_tolerance=0.52,
    )

    assert first.level_dimensions == (2, 1, 0)
    assert first.primitive_dimension == 5
    assert first.chronological_dimension == 15
    assert first.rank_tolerance == pytest.approx(0.52)
    assert first.local_quotient_residual == pytest.approx(6**-0.5)
    torch.testing.assert_close(
        first.quotient_coordinate_indices,
        torch.tensor([0, 1, 2]),
    )
    torch.testing.assert_close(
        first.quotient_coordinate_indices,
        second.quotient_coordinate_indices,
    )


def test_zero_rank_quotient_reports_discarded_map_as_residual() -> None:
    plan = LocalRealization(
        OffsetCode.pascal(3, 3, dtype=torch.float64),
        channels=1,
        max_depth=3,
    ).log_signature(
        quotient=LieQuotient(TemporalDegree(1)),
        depths=(3,),
        rank_tolerance=0.5,
    )

    assert plan.feature_count == 0
    assert plan.chronological_dimension > 0
    assert plan.quotient_lie_dimension > 0
    assert plan.local_quotient_residual is not None
    assert plan.local_quotient_residual > 0.0


def test_selected_depths_bound_every_compiled_plan() -> None:
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    deep = LocalRealization(code, channels=2, max_depth=7)
    shallow = LocalRealization(code, channels=2, max_depth=1)
    windows = increment_windows(_path(), size=3)

    deep_dense = deep.signature(coordinates="dense", depths=(1,))
    deep_log = deep.log_signature(depths=(1,))
    deep_quotient = deep.log_signature(
        quotient=LieQuotient(ChannelSupport(1)),
        depths=(1,),
    )
    assert deep_dense.max_depth == deep_log.max_depth == deep_quotient.max_depth == 1
    torch.testing.assert_close(
        deep_dense.evaluate(windows),
        shallow.signature(coordinates="dense").evaluate(windows),
    )
    torch.testing.assert_close(
        deep_log.evaluate(windows),
        shallow.log_signature().evaluate(windows),
    )


def test_local_quotient_composes_runs_all_backends_and_reconstructs() -> None:
    code = OffsetCode.pascal(2, 2, dtype=torch.float64)
    temporal = TemporalDegree(1)
    support = ChannelSupport(1)
    quotient = LieQuotient(temporal) & LieQuotient(support)
    assert quotient.rules == (temporal, support)

    realization = LocalRealization(code, channels=2, max_depth=3)
    plan = realization.log_signature(quotient=quotient)
    windows = increment_windows(_path(increments=4), size=2)
    outputs = {
        backend: plan.evaluate(windows, backend=backend)
        for backend in ("direct", "pbw", "tensor")
    }

    torch.testing.assert_close(outputs["direct"], outputs["pbw"], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(
        outputs["direct"], outputs["tensor"], atol=1e-10, rtol=1e-9
    )
    assert plan.quotient_lie_dimension is not None
    assert plan.feature_count == plan.quotient_lie_dimension
    assert (
        plan.feature_count
        < realization.log_signature(quotient=LieQuotient(temporal)).feature_count
    )
    assert (
        plan.feature_count
        < realization.log_signature(quotient=LieQuotient(support)).feature_count
    )

    represented = plan.reconstruct(outputs["direct"])
    represented_indices = plan.represented_quotient_indices.tolist()
    chart_positions = torch.tensor(
        [
            represented_indices.index(index)
            for index in plan.quotient_coordinate_indices.tolist()
        ],
        dtype=torch.long,
    )
    assert plan.quotient_basis is not None
    assert len(plan.quotient_basis) == represented.shape[-1]
    torch.testing.assert_close(
        represented.index_select(-1, chart_positions),
        outputs["direct"],
    )


def test_local_graph_span_dimension_matches_the_degree_two_formula() -> None:
    window = 4
    channels = 3
    maximum_span = 2
    edges = ((0, 1), (1, 2))
    plan = LocalRealization(
        OffsetCode.identity(window, dtype=torch.float64),
        channels=channels,
        max_depth=2,
    ).log_signature(
        quotient=LieQuotient(
            DependenceGraph(edges),
            OffsetSpan(maximum_span),
        ),
        depths=(2,),
    )
    support_pairs = sum(window - span for span in range(1, maximum_span + 1))
    expected = support_pairs * (channels + 2 * len(edges))

    assert plan.feature_count == expected
    assert plan.local_quotient_dimension == expected
    assert plan.quotient_lie_dimension is not None
    assert plan.quotient_lie_dimension > expected

    generator = torch.Generator().manual_seed(73)
    windows = torch.randn(
        2,
        3,
        window,
        channels,
        generator=generator,
        dtype=torch.float64,
    )
    direct = plan.evaluate(windows, backend="direct")
    torch.testing.assert_close(
        direct,
        plan.evaluate(windows, backend="pbw"),
        atol=1e-10,
        rtol=1e-9,
    )
    torch.testing.assert_close(
        direct,
        plan.evaluate(windows, backend="tensor"),
        atol=1e-10,
        rtol=1e-9,
    )
    assert plan.reconstruct(direct).shape[-1] == plan.quotient_lie_dimension


def test_public_quotient_basis_labels_are_stable_and_reconstruction_aligned() -> None:
    hall_plan = LocalRealization(
        OffsetCode.identity(2, dtype=torch.float64),
        channels=1,
        max_depth=2,
    ).log_signature(quotient=LieQuotient(OffsetSpan(1)))

    assert hall_plan.quotient_basis == (
        QuotientBasisElement(kind="hall", depth=1, tree=1),
        QuotientBasisElement(kind="hall", depth=1, tree=2),
        QuotientBasisElement(kind="hall", depth=2, tree=(1, 2)),
    )

    trace_plan = LocalRealization(
        OffsetCode.constant(2, dtype=torch.float64),
        channels=3,
        max_depth=3,
    ).log_signature(
        quotient=LieQuotient(DependenceGraph(((0, 1),))),
        depths=(2, 3),
    )
    expected_trace = (
        QuotientBasisElement(kind="trace", depth=2, word=(1, 2)),
        QuotientBasisElement(kind="trace", depth=3, word=(1, 1, 2)),
        QuotientBasisElement(kind="trace", depth=3, word=(1, 2, 2)),
    )

    assert trace_plan.quotient_basis == expected_trace
    assert hash(expected_trace)
    windows = torch.randn(2, 3, 2, 3, dtype=torch.float64)
    reconstructed = trace_plan.reconstruct(trace_plan.evaluate(windows))
    assert reconstructed.shape[-1] == len(expected_trace)


def test_polynomial_plan_matches_explicit_scaled_monomials_and_contracts() -> None:
    path = _path()
    windows = increment_windows(path, size=3)
    plan = compile_polynomial(
        window=3,
        channels=2,
        max_depth=3,
        depths=(1, 3),
        max_offset_span=2,
    )

    features = plan.evaluate(windows, chunk_size=3)
    expected_blocks = []
    for offsets, channels, coefficients in zip(
        plan.offset_indices,
        plan.channel_indices,
        plan.coefficients,
        strict=True,
    ):
        columns = []
        for offset_row, channel_row, coefficient in zip(
            offsets,
            channels,
            coefficients,
            strict=True,
        ):
            column = windows.new_ones(*windows.shape[:2])
            for offset, channel in zip(offset_row, channel_row, strict=True):
                column = column * windows[..., int(offset), int(channel)]
            columns.append(coefficient * column)
        expected_blocks.append(torch.stack(columns, dim=-1))
    expected = torch.cat(expected_blocks, dim=-1)
    weight = torch.randn(4, plan.feature_count, dtype=torch.float64)

    assert features.shape[-1] == plan.feature_count
    torch.testing.assert_close(features, expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(
        plan.contract(windows, weight, chunk_size=3),
        features @ weight.T,
        atol=1e-12,
        rtol=1e-10,
    )


def test_modules_persist_fixed_state_and_propagate_path_and_readout_gradients() -> None:
    plan = LocalRealization(
        OffsetCode.pascal(3, 2, dtype=torch.float64),
        channels=2,
        max_depth=2,
    ).log_signature()
    features = LocalFeatures(plan, stride=2, backend="sparse")
    path = _path().requires_grad_()

    assert dict(features.named_parameters()) == {}
    assert set(features.state_dict()) == {"matrix"}
    torch.testing.assert_close(
        features(path),
        plan.evaluate(increment_windows(path, size=3, stride=2), backend="sparse"),
        atol=1e-11,
        rtol=1e-9,
    )

    layer = LocalLinear(
        plan,
        out_features=3,
        stride=2,
        backend="sparse",
    )
    output = layer(path)
    output.square().mean().backward()

    assert output.shape == (2, 2, 3)
    assert layer.weight.dtype == plan.matrix.dtype == torch.float64
    assert set(layer.state_dict()) == {"weight", "bias", "features.matrix"}
    assert path.grad is not None and torch.isfinite(path.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert layer.bias is not None
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()

    with pytest.raises(RuntimeError, match="dtype/device must match"):
        layer(_path().float())


def test_module_registers_the_original_trainable_code_and_differentiates_it() -> None:
    matrix = torch.nn.Parameter(
        torch.tensor(
            [[1.0, 0.0], [0.5, 1.0], [-0.5, 0.25]],
            dtype=torch.float64,
        )
    )
    plan = LocalRealization(
        OffsetCode(matrix),
        channels=2,
        max_depth=2,
    ).signature(coordinates="dense")
    module = LocalFeatures(plan)
    path = _path().requires_grad_()

    assert module.matrix is matrix
    assert dict(module.named_parameters()) == {"matrix": matrix}
    module(path).square().mean().backward()

    assert matrix.grad is not None and torch.isfinite(matrix.grad).all()
    assert matrix.grad.abs().sum() > 0
    assert path.grad is not None and torch.isfinite(path.grad).all()


def test_trainable_mapping_owns_a_reusable_leaf_parameter() -> None:
    base = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    code = OffsetCode(base * 2.0)
    assert isinstance(code.matrix, torch.nn.Parameter)
    assert code.matrix.is_leaf
    assert code.matrix is not base

    plan = LocalRealization(code, channels=2, max_depth=2).signature(
        coordinates="dense"
    )
    module = LocalFeatures(plan)
    for _ in range(2):
        module.zero_grad(set_to_none=True)
        module(_path()).square().mean().backward()
        assert module.matrix is not None
        assert module.matrix.grad is not None
    assert base.grad is None


def test_fixed_plan_state_rejects_an_incompatible_code_matrix() -> None:
    first_plan = LocalRealization(
        OffsetCode.pascal(3, 2, dtype=torch.float64),
        channels=2,
        max_depth=2,
    ).signature()
    second_plan = LocalRealization(
        OffsetCode(
            torch.tensor(
                [[1.0, 0.0], [0.5, 1.0], [-0.25, 0.75]],
                dtype=torch.float64,
            )
        ),
        channels=2,
        max_depth=2,
    ).signature()
    first = LocalFeatures(first_plan)
    compatible = LocalFeatures(first_plan)
    incompatible = LocalFeatures(second_plan)

    compatible.load_state_dict(first.state_dict())
    with pytest.raises(RuntimeError, match="compiled plan architecture"):
        incompatible.load_state_dict(first.state_dict())


def test_polynomial_linear_fuses_contraction_with_state_and_gradients() -> None:
    plan = compile_polynomial(window=3, channels=2, max_depth=3)
    layer = LocalLinear(
        plan,
        out_features=4,
        stride=2,
        chunk_size=3,
        dtype=torch.float64,
    )
    path = _path().requires_grad_()

    actual = layer(path)
    windows = increment_windows(path, size=3, stride=2)
    expected = torch.nn.functional.linear(
        plan.evaluate(windows, chunk_size=3),
        layer.weight,
        layer.bias,
    )

    assert layer.matrix is None
    assert set(layer.state_dict()) == {"weight", "bias"}
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)

    actual.square().mean().backward()
    assert path.grad is not None and torch.isfinite(path.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert layer.bias is not None
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()

    with pytest.raises(RuntimeError, match="dtype/device must match"):
        layer(_path().float())


def test_polynomial_plan_supports_empty_axes_and_validates_clean_boundaries() -> None:
    plan = compile_polynomial(window=3, channels=2, max_depth=2)
    assert plan.evaluate(torch.empty(0, 2, 3, 2)).shape == (
        0,
        2,
        plan.feature_count,
    )
    assert plan.evaluate(torch.empty(2, 0, 3, 2)).shape == (
        2,
        0,
        plan.feature_count,
    )
    assert LocalFeatures(plan)(torch.empty(0, 6, 2)).shape == (
        0,
        3,
        plan.feature_count,
    )

    with pytest.raises(ValueError, match="size must be a positive integer"):
        increment_windows(_path(), size=True)
    with pytest.raises(ValueError, match="window must be a positive integer"):
        compile_polynomial(window=True, channels=2, max_depth=2)
    with pytest.raises(ValueError, match="chunk_size"):
        plan.evaluate(torch.empty(1, 1, 3, 2), chunk_size=0)
    with pytest.raises(ValueError, match="scaling"):
        compile_polynomial(window=3, channels=2, max_depth=2, scaling=[])

    realization = LocalRealization(
        OffsetCode.pascal(3, 2),
        channels=2,
        max_depth=2,
    )
    with pytest.raises(TypeError, match="selection"):
        realization.signature(selection="bad")
    with pytest.raises(TypeError, match="filtration"):
        realization.log_signature(filtration="bad")
