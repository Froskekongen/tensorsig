"""Independent theory checks for compiled local log-signature plans."""

import math

import pytest
import torch

from tensorsig.algebra.lyndon import lyndon_words
from tensorsig.classical.signatures import log_signature, signature
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.coordinates import ChronologicalFiltration
from tensorsig.offset_coded_signatures.layers import LocalFeatures, LocalLinear
from tensorsig.offset_coded_signatures.plans import LocalRealization
from tensorsig.offset_coded_signatures.windows import increment_windows


def _path(
    *,
    seed: int,
    batch: int = 2,
    increments: int = 6,
    channels: int = 2,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    dx = torch.randn(
        batch,
        increments,
        channels,
        generator=generator,
        dtype=torch.float64,
    )
    origin = torch.zeros(batch, 1, channels, dtype=torch.float64)
    return torch.cat((origin, dx.cumsum(dim=1)), dim=1)


def _explicit_projected_log_signature(
    windows: torch.Tensor,
    code: OffsetCode,
    depth: int,
) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
    """Lift windows directly and call the independent global logsignature."""
    lifted = torch.einsum("bnrc,rp->bnrcp", windows, code.matrix).flatten(-2)
    batch, count, window, width = lifted.shape
    paths = torch.cat(
        (lifted.new_zeros(batch, count, 1, width), lifted.cumsum(dim=2)),
        dim=2,
    )
    values = log_signature(
        paths.reshape(batch * count, window + 1, width),
        depth=depth,
        mode="words",
    )
    return values.reshape(batch, count, -1), lyndon_words(width, depth)


def _compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts == 1:
        return ((total,),)
    return tuple(
        (first, *rest)
        for first in range(1, total - parts + 2)
        for rest in _compositions(total - first, parts - 1)
    )


def _explicit_projected_tensor_log(
    windows: torch.Tensor,
    code: OffsetCode,
    depth: int,
) -> torch.Tensor:
    """Compute one tensor-log level from the defining formal logarithm."""
    levels = _explicit_projected_signature_levels(windows, code, depth)
    batch, count = windows.shape[:2]

    result = levels[depth - 1].clone()
    for order in range(2, depth + 1):
        coefficient = (-1.0) ** (order + 1) / order
        for composition in _compositions(depth, order):
            term = levels[composition[0] - 1]
            for part in composition[1:]:
                right = levels[part - 1]
                term = (term.unsqueeze(2) * right.unsqueeze(1)).flatten(start_dim=1)
            result = result + coefficient * term
    return result.reshape(batch, count, -1)


def _explicit_projected_signature_levels(
    windows: torch.Tensor,
    code: OffsetCode,
    depth: int,
) -> list[torch.Tensor]:
    """Lift every window explicitly and evaluate the ordinary signature."""
    lifted = torch.einsum("bnrc,rp->bnrcp", windows, code.matrix).flatten(-2)
    batch, count, window, width = lifted.shape
    paths = torch.cat(
        (lifted.new_zeros(batch, count, 1, width), lifted.cumsum(dim=2)),
        dim=2,
    )
    flattened = signature(
        paths.reshape(batch * count, window + 1, width),
        depth=depth,
    )
    levels = []
    start = 0
    for current_depth in range(1, depth + 1):
        stop = start + width**current_depth
        levels.append(flattened[:, start:stop])
        start = stop
    return levels


def _flat_word_index(word: tuple[int, ...], width: int) -> int:
    result = 0
    for letter in word:
        result = result * width + letter - 1
    return result


def _fit_product_map(
    left: torch.Tensor,
    right: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Recover the unique product map from full-rank function values."""
    products = (left.unsqueeze(2) * right.unsqueeze(1)).flatten(start_dim=1)
    solution = torch.linalg.lstsq(target, products).solution
    torch.testing.assert_close(target @ solution, products, atol=1e-9, rtol=1e-8)
    return solution.T.reshape(left.shape[1], right.shape[1], target.shape[1])


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_public_primitive_chart_has_the_tensor_log_image_rank(depth: int) -> None:
    code = OffsetCode.pascal(3, 2, orthonormal=False, dtype=torch.float64)
    plan = LocalRealization(code, channels=2, max_depth=depth).log_signature(
        depths=(depth,)
    )
    generator = torch.Generator().manual_seed(100 + depth)
    windows = torch.randn(
        1,
        max(2 * plan.feature_count, 32),
        code.window,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    tensor_log = _explicit_projected_tensor_log(windows, code, depth)[0]

    assert plan.ambient_words is not None
    indices = torch.tensor(
        [_flat_word_index(word, plan.output_dimension) for word in plan.ambient_words]
    )
    ambient = tensor_log.index_select(-1, indices)
    features = plan.evaluate(windows)[0]

    assert int(torch.linalg.matrix_rank(ambient)) == plan.primitive_dimension
    assert int(torch.linalg.matrix_rank(features)) == plan.feature_count
    torch.testing.assert_close(
        plan.reconstruct(features),
        ambient,
        atol=1e-9,
        rtol=1e-7,
    )


def test_local_word_spaces_are_shuffle_closed_and_associative_through_degree_three() -> (
    None
):
    code = OffsetCode.pascal(3, 2, orthonormal=False, dtype=torch.float64)
    realization = LocalRealization(code, channels=2, max_depth=3)
    generator = torch.Generator().manual_seed(1701)
    windows = torch.randn(
        1,
        96,
        code.window,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    explicit_levels = _explicit_projected_signature_levels(windows, code, depth=3)
    coordinates = {}
    for depth in (1, 2, 3):
        plan = realization.signature(depths=(depth,))
        words = plan.word_indices[0]
        flat_indices = torch.zeros(words.shape[0], dtype=torch.long)
        for column in range(depth):
            flat_indices = flat_indices * plan.output_dimension + words[:, column]
        values = explicit_levels[depth - 1].index_select(-1, flat_indices)
        assert int(torch.linalg.matrix_rank(values)) == plan.feature_count
        torch.testing.assert_close(
            plan.evaluate(windows)[0],
            values,
            atol=1e-11,
            rtol=1e-9,
        )
        coordinates[depth] = values

    multiply_11 = _fit_product_map(
        coordinates[1],
        coordinates[1],
        coordinates[2],
    )
    multiply_12 = _fit_product_map(
        coordinates[1],
        coordinates[2],
        coordinates[3],
    )
    multiply_21 = _fit_product_map(
        coordinates[2],
        coordinates[1],
        coordinates[3],
    )

    left_associated = torch.einsum("ije,ekf->ijkf", multiply_11, multiply_21)
    right_associated = torch.einsum("jke,ief->ijkf", multiply_11, multiply_12)
    torch.testing.assert_close(
        left_associated,
        right_associated,
        atol=1e-8,
        rtol=1e-7,
    )


@pytest.mark.parametrize(
    ("window", "code_dimension", "channels", "expected"),
    [
        (3, 2, 1, 1),
        (3, 2, 2, 6),
        (4, 3, 2, 15),
        (4, 2, 3, 15),
    ],
)
def test_degree_two_primitive_rank_matches_closed_formula(
    window: int,
    code_dimension: int,
    channels: int,
    expected: int,
) -> None:
    code = OffsetCode.pascal(
        window,
        code_dimension,
        orthonormal=False,
        dtype=torch.float64,
    )
    plan = LocalRealization(code, channels, 2).log_signature(depths=(2,))

    alternating_code = math.comb(code_dimension, 2)
    symmetric_code = min(
        math.comb(window, 2),
        math.comb(code_dimension + 1, 2),
    )
    formula = alternating_code * math.comb(channels + 1, 2) + (
        symmetric_code * math.comb(channels, 2)
    )

    assert plan.primitive_dimension == plan.feature_count == expected == formula


@pytest.mark.parametrize(
    ("window", "channels", "expected"),
    [(2, 2, 12), (3, 2, 44), (3, 3, 135)],
)
def test_degree_three_full_retention_rank_matches_closed_formula(
    window: int,
    channels: int,
    expected: int,
) -> None:
    plan = LocalRealization(
        OffsetCode.identity(window, dtype=torch.float64),
        channels,
        3,
    ).log_signature(depths=(3,))

    symmetric = math.comb(window + 2, 3) - window
    standard = window * (window**2 - 1) // 3
    alternating = math.comb(window, 3)
    formula = (
        symmetric * math.comb(channels + 2, 3)
        + standard * channels * (channels**2 - 1) // 3
        + alternating * math.comb(channels, 3)
    )

    assert plan.primitive_dimension == plan.feature_count == expected == formula


def test_projected_degree_three_rank_detects_coupled_deficit() -> None:
    code = OffsetCode.pascal(4, 3, orthonormal=False, dtype=torch.float64)
    plan = LocalRealization(code, channels=2, max_depth=3).log_signature(depths=(3,))

    assert plan.ambient_words is not None
    assert len(plan.ambient_words) == 70
    assert plan.chronological_dimension == 108
    assert plan.primitive_dimension == plan.feature_count == 68


@pytest.mark.parametrize(("window", "expected"), [(2, 1), (3, 6)])
def test_degree_four_scalar_full_retention_rank_matches_closed_formula(
    window: int,
    expected: int,
) -> None:
    plan = LocalRealization(
        OffsetCode.identity(window, dtype=torch.float64),
        channels=1,
        max_depth=4,
    ).log_signature(depths=(4,))

    formula = math.comb(window + 3, 4) - window**2
    assert plan.primitive_dimension == plan.feature_count == expected == formula


@pytest.mark.parametrize(
    ("window", "code_dimension", "channels"),
    [(2, 2, 2), (3, 2, 2), (3, 3, 2), (4, 3, 3)],
)
def test_degree_two_pascal_value_profiles_match_closed_formulas(
    window: int,
    code_dimension: int,
    channels: int,
) -> None:
    code = OffsetCode.pascal(
        window,
        code_dimension,
        orthonormal=False,
        dtype=torch.float64,
    )
    plan = LocalRealization(code, channels=channels, max_depth=2).log_signature(
        depths=(2,)
    )
    generator = torch.Generator().manual_seed(200 + 10 * window + channels)
    windows = torch.randn(
        1,
        max(4 * plan.feature_count, 32),
        window,
        channels,
        generator=generator,
        dtype=torch.float64,
    )
    primitive = _explicit_projected_tensor_log(windows, code, depth=2)[0]
    assert int(torch.linalg.matrix_rank(primitive)) == plan.primitive_dimension
    width = code_dimension * channels
    temporal_degree = torch.tensor(
        [
            sum(
                (flat // width**power) % width % code_dimension
                for power in reversed(range(2))
            )
            for flat in range(width**2)
        ]
    )
    value_profile = tuple(
        int(torch.linalg.matrix_rank(primitive[:, temporal_degree <= degree]))
        for degree in range(2 * code_dimension - 1)
    )

    exact_degrees = []
    for left in range(code_dimension):
        for right in range(left + 1, code_dimension):
            exact_degrees.extend([left + right] * math.comb(channels + 1, 2))
    if window == code_dimension:
        pairs = (
            (left, right)
            for left in range(code_dimension)
            for right in range(left + 1, code_dimension)
        )
        exact_degrees.extend(
            left + right - 1
            for left, right in pairs
            for _ in range(math.comb(channels, 2))
        )
    else:
        pairs = (
            (left, right)
            for left in range(code_dimension)
            for right in range(left, code_dimension)
        )
        exact_degrees.extend(
            left + right for left, right in pairs for _ in range(math.comb(channels, 2))
        )
    expected = tuple(
        sum(exact_degree <= degree for exact_degree in exact_degrees)
        for degree in range(2 * code_dimension - 1)
    )

    assert value_profile == expected


def test_degree_two_pascal_coordinate_profile_regression() -> None:
    realization = LocalRealization(
        OffsetCode.pascal(3, 2, orthonormal=False, dtype=torch.float64),
        channels=2,
        max_depth=2,
    )
    coordinate_profile = tuple(
        realization.log_signature(
            depths=(2,),
            filtration=ChronologicalFiltration(degree),
        ).feature_count
        for degree in range(3)
    )

    assert coordinate_profile == (1, 6, 6)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_minimal_logsignature_reconstructs_explicit_projected_path(
    depth: int,
) -> None:
    code = OffsetCode.pascal(3, 2, orthonormal=False, dtype=torch.float64)
    plan = LocalRealization(code, channels=2, max_depth=depth).log_signature(
        depths=(depth,)
    )
    windows = increment_windows(_path(seed=10 + depth), size=3, stride=2)

    dense = plan.evaluate(windows, backend="dense")
    sparse = plan.evaluate(windows, backend="sparse")
    ambient = plan.reconstruct(dense)
    explicit, all_words = _explicit_projected_log_signature(windows, code, depth)
    positions = {word: index for index, word in enumerate(all_words)}
    assert plan.ambient_words is not None
    expected = explicit.index_select(
        -1,
        torch.tensor([positions[word] for word in plan.ambient_words]),
    )

    torch.testing.assert_close(sparse, dense, atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(ambient, expected, atol=1e-9, rtol=1e-7)


def test_coordinate_filtration_is_nested_and_explicitly_lossy() -> None:
    code = OffsetCode.pascal(4, 3, orthonormal=False, dtype=torch.float64)
    realization = LocalRealization(code, channels=2, max_depth=3)
    full = realization.log_signature(depths=(3,))
    filtered = realization.log_signature(
        depths=(3,),
        filtration=ChronologicalFiltration(1),
    )
    dimensions = tuple(
        realization.log_signature(
            depths=(3,),
            filtration=ChronologicalFiltration(degree),
        ).feature_count
        for degree in range(7)
    )
    windows = increment_windows(_path(seed=25, increments=7), size=4)

    assert list(dimensions) == sorted(dimensions)
    assert dimensions[-1] == full.feature_count
    assert filtered.feature_count == dimensions[1] < full.feature_count
    assert not filtered.is_lossless
    torch.testing.assert_close(
        filtered.evaluate(windows, backend="dense"),
        filtered.evaluate(windows, backend="sparse"),
        atol=1e-9,
        rtol=1e-7,
    )
    with pytest.raises(ValueError, match="not reconstructive"):
        filtered.reconstruct(filtered.evaluate(windows))


@pytest.mark.parametrize("depth", [2, 3])
def test_coordinate_filtration_matches_an_independent_subspace_intersection(
    depth: int,
) -> None:
    code = OffsetCode.pascal(3, 2, orthonormal=False, dtype=torch.float64)
    realization = LocalRealization(code, channels=2, max_depth=depth)
    source_plan = realization.signature(depths=(depth,))
    primitive_plan = realization.log_signature(depths=(depth,))
    generator = torch.Generator().manual_seed(300 + depth)
    windows = torch.randn(
        1,
        4 * (source_plan.feature_count + primitive_plan.primitive_dimension),
        code.window,
        2,
        generator=generator,
        dtype=torch.float64,
    )

    signature_level = _explicit_projected_signature_levels(windows, code, depth)[
        depth - 1
    ]
    tensor_log = _explicit_projected_tensor_log(windows, code, depth)[0]
    assert primitive_plan.ambient_words is not None
    primitive_indices = torch.tensor(
        [
            _flat_word_index(word, primitive_plan.output_dimension)
            for word in primitive_plan.ambient_words
        ]
    )
    primitive_functions = tensor_log.index_select(-1, primitive_indices)
    assert (
        int(torch.linalg.matrix_rank(primitive_functions))
        == primitive_plan.primitive_dimension
    )

    letters = source_plan.word_indices[0]
    flat_indices = torch.zeros(letters.shape[0], dtype=torch.long)
    for column in range(depth):
        flat_indices = flat_indices * source_plan.output_dimension + letters[:, column]
    source_functions = signature_level.index_select(-1, flat_indices)
    source_degrees = source_plan.temporal_indices[0].sum(dim=1)

    for cutoff in range(int(source_degrees.max()) + 1):
        filtered_functions = source_functions[:, source_degrees <= cutoff]
        intersection_dimension = (
            int(torch.linalg.matrix_rank(primitive_functions))
            + int(torch.linalg.matrix_rank(filtered_functions))
            - int(
                torch.linalg.matrix_rank(
                    torch.cat((primitive_functions, filtered_functions), dim=1)
                )
            )
        )
        filtered_plan = realization.log_signature(
            depths=(depth,),
            filtration=ChronologicalFiltration(cutoff),
        )
        assert filtered_plan.feature_count == intersection_dimension


def test_public_layers_match_feature_contraction() -> None:
    torch.manual_seed(30)
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    plan = LocalRealization(code, channels=2, max_depth=3).log_signature()
    path = _path(seed=30)
    windows = increment_windows(path, size=3)
    weight = torch.randn(4, plan.feature_count, dtype=torch.float64)
    bias = torch.randn(4, dtype=torch.float64)
    expected_features = plan.evaluate(windows)
    expected = expected_features @ weight.T + bias

    features = LocalFeatures(plan)
    linear = LocalLinear(plan, out_features=4)
    with torch.no_grad():
        linear.weight.copy_(weight)
        assert linear.bias is not None
        linear.bias.copy_(bias)

    torch.testing.assert_close(features(path), expected_features)
    torch.testing.assert_close(linear(path), expected)


def test_local_logsignature_is_translation_invariant_and_window_equivariant() -> None:
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    features = LocalFeatures(LocalRealization(code, 2, 3).log_signature())
    path = _path(seed=31, increments=7)
    shift = torch.tensor([10.0, -3.0], dtype=torch.float64)

    output = features(path)
    translated = features(path + shift.view(1, 1, 2))
    shifted_window = features(path[:, 1:])

    torch.testing.assert_close(output, translated, atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(output[:, 1:], shifted_window, atol=1e-9, rtol=1e-7)


def test_public_logsignature_contraction_gradcheck() -> None:
    code = OffsetCode.pascal(2, 2, dtype=torch.float64)
    plan = LocalRealization(code, channels=2, max_depth=2).log_signature()
    path = _path(seed=32, batch=1, increments=4).requires_grad_()
    weight = torch.randn(
        2,
        plan.feature_count,
        dtype=torch.float64,
        requires_grad=True,
    )

    def func(points: torch.Tensor, readout: torch.Tensor) -> torch.Tensor:
        windows = increment_windows(points, size=2)
        return plan.contract(windows, readout, backend="sparse")

    assert torch.autograd.gradcheck(func, (path, weight), eps=1e-6, atol=1e-4)


def test_minimal_logsignature_rejects_a_trainable_offset_code() -> None:
    matrix = torch.eye(2, dtype=torch.float64, requires_grad=True)
    code = OffsetCode(matrix)

    with pytest.raises(ValueError, match="fixed OffsetCode"):
        LocalRealization(code, channels=2, max_depth=2).log_signature()
