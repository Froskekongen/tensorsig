from dataclasses import FrozenInstanceError
import itertools

import pytest
import torch

from tensorsig.classical.signatures import signature
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.coordinates import WordSelection
from tensorsig.offset_coded_signatures.plans import LocalRealization
from tensorsig.offset_coded_signatures._charts.colored_descent import (
    generic_scalar_dimension,
    ribbon_dimension,
)
from tensorsig.tensor_ops.series import unflatten_signature_levels


def _explicit_levels(
    windows: torch.Tensor,
    code: torch.Tensor,
    depth: int,
) -> tuple[torch.Tensor, ...]:
    lifted = torch.einsum("bnrc,re->bnrce", windows, code).flatten(-2)
    batch, count, steps, width = lifted.shape
    increments = lifted.reshape(batch * count, steps, width)
    paths = torch.cat(
        (increments.new_zeros(batch * count, 1, width), increments.cumsum(1)),
        dim=1,
    )
    return tuple(
        level.reshape(batch, count, *level.shape[1:])
        for level in unflatten_signature_levels(
            signature(paths, depth=depth),
            width=width,
            depth=depth,
        )
    )


@pytest.mark.parametrize(
    ("window", "code_dimension", "channels", "depth"),
    [
        (1, 1, 2, 3),
        (3, 1, 2, 3),
        (3, 2, 2, 3),
        (3, 3, 2, 3),
        (4, 2, 2, 2),
        (4, 4, 2, 3),
    ],
)
def test_public_word_count_and_labels_match_the_colored_descent_dimension(
    window: int,
    code_dimension: int,
    channels: int,
    depth: int,
) -> None:
    plan = LocalRealization(
        OffsetCode.pascal(window, code_dimension, dtype=torch.float64),
        channels,
        depth,
    ).signature(depths=(depth,))
    temporal = plan.temporal_indices[0]
    channel_words = plan.channel_indices[0]

    expected = generic_scalar_dimension(window, code_dimension, channels, depth)
    assert plan.feature_count == expected
    assert temporal.shape == channel_words.shape == (expected, depth)
    assert torch.all(
        temporal.sum(1) + (channel_words[:, :-1] > channel_words[:, 1:]).sum(1) < window
    )
    torch.testing.assert_close(
        plan.word_indices[0],
        channel_words * code_dimension + temporal,
    )


@pytest.mark.parametrize(
    ("window", "code_dimension", "channels", "depth"),
    [(3, 2, 1, 3), (3, 2, 2, 3), (4, 2, 2, 2)],
)
def test_public_chart_losslessly_reconstructs_explicit_global_signatures(
    window: int,
    code_dimension: int,
    channels: int,
    depth: int,
) -> None:
    code = OffsetCode.pascal(window, code_dimension, dtype=torch.float64)
    plan = LocalRealization(code, channels, depth).signature(depths=(depth,))
    generator = torch.Generator().manual_seed(1800 + window + depth)
    windows = torch.randn(
        8,
        2,
        window,
        channels,
        generator=generator,
        dtype=torch.float64,
    )
    expected = _explicit_levels(windows, code.matrix, depth)[depth - 1]
    features = plan.evaluate(windows)

    assert plan.is_lossless
    torch.testing.assert_close(
        plan.reconstruct(features)[0],
        expected,
        atol=1e-11,
        rtol=1e-9,
    )


def test_public_decoder_transfers_every_dense_linear_readout() -> None:
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    plan = LocalRealization(code, channels=2, max_depth=3).signature(depths=(3,))
    generator = torch.Generator().manual_seed(1811)
    windows = torch.randn(
        5,
        2,
        3,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    ambient_weights = torch.randn(
        4,
        plan.output_dimension**3,
        generator=generator,
        dtype=torch.float64,
    )

    feature_basis = plan.reconstruct(
        torch.eye(plan.feature_count, dtype=torch.float64)
    )[0].flatten(start_dim=1)
    transferred_weights = ambient_weights @ feature_basis.T
    features = plan.evaluate(windows)
    dense = plan.reconstruct(features)[0].flatten(start_dim=2)

    torch.testing.assert_close(
        plan.contract(windows, transferred_weights),
        dense @ ambient_weights.T,
        atol=1e-11,
        rtol=1e-9,
    )


def test_random_invertible_code_gauge_preserves_the_chronological_function_space() -> (
    None
):
    raw = OffsetCode.pascal(
        3,
        2,
        orthonormal=False,
        dtype=torch.float64,
    )
    gauge = torch.tensor(
        [[1.2, -0.3], [0.4, 0.9]],
        dtype=torch.float64,
    )
    transformed = OffsetCode(raw.matrix @ gauge)
    raw_plan = LocalRealization(raw, channels=2, max_depth=3).signature(depths=(3,))
    transformed_plan = LocalRealization(
        transformed,
        channels=2,
        max_depth=3,
    ).signature(depths=(3,))
    generator = torch.Generator().manual_seed(1812)
    windows = torch.randn(
        96,
        1,
        3,
        2,
        generator=generator,
        dtype=torch.float64,
    )

    raw_features = raw_plan.evaluate(windows)[:, 0]
    transformed_features = transformed_plan.evaluate(windows)[:, 0]
    change = torch.linalg.lstsq(raw_features, transformed_features).solution
    reconstructed = raw_features @ change

    assert raw_plan.feature_count == transformed_plan.feature_count
    assert int(torch.linalg.matrix_rank(raw_features)) == raw_plan.feature_count
    assert int(torch.linalg.matrix_rank(transformed_features)) == raw_plan.feature_count
    assert int(torch.linalg.matrix_rank(change)) == raw_plan.feature_count
    torch.testing.assert_close(
        reconstructed,
        transformed_features,
        atol=2e-10,
        rtol=2e-9,
    )

    alphabet_gauge = torch.kron(torch.eye(2, dtype=torch.float64), gauge.contiguous())
    tensor_gauge = torch.kron(
        torch.kron(alphabet_gauge, alphabet_gauge),
        alphabet_gauge,
    )
    raw_dense = raw_plan.reconstruct(raw_features)[0].flatten(start_dim=1)
    transformed_dense = transformed_plan.reconstruct(transformed_features)[0].flatten(
        start_dim=1
    )
    torch.testing.assert_close(
        raw_dense @ tensor_gauge,
        transformed_dense,
        atol=2e-10,
        rtol=2e-9,
    )


def test_public_coordinates_are_actual_dense_tensor_words() -> None:
    code = OffsetCode.pascal(3, 2, dtype=torch.float64)
    realization = LocalRealization(code, channels=2, max_depth=3)
    chronological = realization.signature(depths=(3,))
    dense = realization.signature(coordinates="dense", depths=(3,))
    windows = torch.randn(2, 3, 3, 2, dtype=torch.float64)

    dense_features = dense.evaluate(windows)
    width = dense.output_dimension
    powers = torch.tensor([width**power for power in reversed(range(3))])
    flat_indices = chronological.word_indices[0] @ powers

    torch.testing.assert_close(
        chronological.evaluate(windows),
        dense_features.index_select(-1, flat_indices),
        atol=1e-12,
        rtol=1e-10,
    )


def test_raw_pascal_reconstruction_encodes_the_flat_straightening_relation() -> None:
    code = OffsetCode.pascal(
        3,
        2,
        orthonormal=False,
        dtype=torch.float64,
    )
    plan = LocalRealization(code, channels=2, max_depth=2).signature(depths=(2,))
    decoder = plan.reconstruct(torch.eye(plan.feature_count, dtype=torch.float64))[0]
    decoder = decoder.reshape(plan.feature_count, -1).T
    width = plan.output_dimension
    flat_indices = plan.word_indices[0] @ torch.tensor([width, 1])

    expected = torch.zeros(plan.feature_count, dtype=torch.float64)
    for flat_index, coefficient in ((7, 1.0), (6, -2.0), (12, 2.0)):
        position = (flat_indices == flat_index).nonzero().item()
        expected[position] = coefficient
    torch.testing.assert_close(decoder[13], expected, atol=1e-12, rtol=0)


def test_raw_and_orthonormal_pascal_share_words_but_not_coordinates() -> None:
    raw = OffsetCode.pascal(4, 2, orthonormal=False, dtype=torch.float64)
    orthonormal = OffsetCode.pascal(4, 2, dtype=torch.float64)
    raw_plan = LocalRealization(raw, 2, 3).signature(depths=(3,))
    orthonormal_plan = LocalRealization(orthonormal, 2, 3).signature(depths=(3,))
    windows = torch.randn(5, 2, 4, 2, dtype=torch.float64)
    raw_features = raw_plan.evaluate(windows)
    orthonormal_features = orthonormal_plan.evaluate(windows)

    torch.testing.assert_close(
        raw_plan.word_indices[0], orthonormal_plan.word_indices[0]
    )
    assert not torch.allclose(raw_features, orthonormal_features)
    for plan, features, code in (
        (raw_plan, raw_features, raw),
        (orthonormal_plan, orthonormal_features, orthonormal),
    ):
        torch.testing.assert_close(
            plan.reconstruct(features)[0],
            _explicit_levels(windows, code.matrix, 3)[2],
            atol=1e-10,
            rtol=1e-9,
        )


@pytest.mark.parametrize(
    ("channels", "depth", "cuts"),
    [(2, 2, (1,)), (2, 3, (2,)), (3, 4, (1, 3))],
)
def test_exact_descent_sections_are_the_canonical_channel_orbit_sums(
    channels: int,
    depth: int,
    cuts: tuple[int, ...],
) -> None:
    import tensorsig.offset_coded_signatures._charts.chronological as words

    section = words._descent_orbit_section(channels, depth, cuts)
    representatives = [
        word
        for word in itertools.product(range(channels), repeat=depth)
        if tuple(
            position
            for position in range(1, depth)
            if word[position - 1] > word[position]
        )
        == cuts
    ]
    boundaries = (0, *cuts, depth)

    assert section.shape == (
        channels**depth,
        ribbon_dimension(channels, depth, cuts),
    )
    assert len(representatives) == section.shape[1]
    for column, representative in zip(section.T, representatives, strict=True):
        block_orbits = [
            set(itertools.permutations(representative[start:stop]))
            for start, stop in zip(boundaries, boundaries[1:])
        ]
        expected_rows = {
            sum(
                channel * channels ** (depth - position - 1)
                for position, channel in enumerate(itertools.chain(*blocks))
            )
            for blocks in itertools.product(*block_orbits)
        }
        actual_rows = set(column.nonzero(as_tuple=False).flatten().tolist())
        assert actual_rows == expected_rows
        assert torch.equal(column[sorted(actual_rows)], torch.ones(len(actual_rows)))


def test_selection_is_directly_visible_in_public_word_metadata() -> None:
    realization = LocalRealization(
        OffsetCode.pascal(4, 2, dtype=torch.float64),
        channels=2,
        max_depth=3,
    )
    full = realization.signature(depths=(3,))
    selected = realization.signature(
        depths=(3,),
        selection=WordSelection(
            max_temporal_degree=1,
            max_channel_descents=1,
            max_complexity=1,
            descent_sets=((), (1,)),
        ),
    )
    temporal = full.temporal_indices[0]
    channels = full.channel_indices[0]
    descents = channels[:, :-1] > channels[:, 1:]
    mask = (
        (temporal.sum(1) <= 1)
        & (descents.sum(1) <= 1)
        & (temporal.sum(1) + descents.sum(1) <= 1)
        & ((~descents).all(1) | (descents[:, 0] & ~descents[:, 1]))
    )

    torch.testing.assert_close(selected.word_indices[0], full.word_indices[0][mask])
    assert not selected.is_lossless


def test_structural_compilation_does_not_construct_the_dense_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorsig.offset_coded_signatures._charts.chronological as words

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("dense reference construction was called")

    monkeypatch.setattr(words, "_canonical_value_basis", fail_if_called)
    plan = LocalRealization(
        OffsetCode.pascal(4, 2, dtype=torch.float64),
        channels=2,
        max_depth=3,
    ).signature(selection=WordSelection(max_complexity=1), depths=(3,))

    assert plan.feature_count == 20
    assert plan.state_count == 35
    with pytest.raises(FrozenInstanceError):
        plan.depths = (2,)  # type: ignore[misc]


def test_structural_decoder_is_built_once_per_level_and_owned_by_the_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorsig.offset_coded_signatures.plans as plans

    original = plans._PascalDecoder
    built_depths = []

    class TrackedDecoder(original):
        def __init__(self, code, words):
            built_depths.append(words.shape[1])
            super().__init__(code, words)

    monkeypatch.setattr(plans, "_PascalDecoder", TrackedDecoder)
    plan = LocalRealization(
        OffsetCode.pascal(3, 2, dtype=torch.float64),
        channels=2,
        max_depth=3,
    ).signature(depths=(2, 3))
    features = torch.randn(plan.feature_count, dtype=torch.float64)

    first = plan.reconstruct(features)
    second = plan.reconstruct(features)

    assert built_depths == [2, 3]
    assert set(plan._decoder_cache) == {2, 3}
    assert all(level.reconstruction_matrix is None for level in plan._levels)
    other = LocalRealization(plan.mapping, channels=2, max_depth=3).signature(
        depths=(2, 3)
    )
    assert other._decoder_cache == {}
    assert other._decoder_cache is not plan._decoder_cache
    for first_level, second_level in zip(first, second, strict=True):
        torch.testing.assert_close(first_level, second_level)
    with pytest.raises(FrozenInstanceError):
        plan._decoder_cache = {}  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_temporal_degree": -1}, "max_temporal_degree"),
        ({"max_channel_descents": -1}, "max_channel_descents"),
        ({"max_complexity": -1}, "max_complexity"),
        ({"descent_sets": ((0,),)}, "descent-set positions"),
        ({"descent_sets": ((1, 1),)}, "duplicates"),
    ],
)
def test_public_chart_rejects_invalid_selections(
    kwargs: dict[str, object],
    message: str,
) -> None:
    realization = LocalRealization(
        OffsetCode.pascal(3, 2, dtype=torch.float64),
        channels=2,
        max_depth=3,
    )
    with pytest.raises(ValueError, match=message):
        realization.signature(selection=WordSelection(**kwargs))
