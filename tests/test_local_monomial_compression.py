from collections import Counter
import itertools
import math

import torch

from tensorsig.offset_coded_signatures.monomials import (
    local_monomial_features_from_offset_channels,
)
from tensorsig.offset_coded_signatures.offset_patterns import (
    log_signature_offset_coefficient,
    offset_run_lengths,
)
from tensorsig.offset_coded_signatures.scaled_monomials import (
    raw_signature_monomial_indices,
)
from tensorsig.classical.signatures import signature
from tensorsig.tensor_ops.series import (
    _signature_to_logsignature_tensor,
    unflatten_signature_levels,
)


def _lifted_path_from_window(z: torch.Tensor) -> torch.Tensor:
    kernel_size, channels = z.shape
    increments = torch.zeros(
        1,
        kernel_size,
        kernel_size * channels,
        dtype=z.dtype,
        device=z.device,
    )
    for offset in range(kernel_size):
        start = offset * channels
        increments[0, offset, start : start + channels] = z[offset]
    zero = torch.zeros(
        1,
        1,
        kernel_size * channels,
        dtype=z.dtype,
        device=z.device,
    )
    return torch.cat([zero, increments.cumsum(dim=1)], dim=1)


def _raw_weight_for_offsets(offsets: tuple[int, ...]) -> float:
    denominator = 1
    for offset in set(offsets):
        denominator *= math.factorial(offsets.count(offset))
    return 1.0 / denominator


def _raw_tensor_feature_rows(
    in_channels: int,
    kernel_size: int,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset_rows = []
    channel_rows = []
    for offset_tuple in itertools.combinations_with_replacement(
        range(kernel_size),
        depth,
    ):
        for channel_tuple in itertools.product(range(in_channels), repeat=depth):
            offset_rows.append(offset_tuple)
            channel_rows.append(channel_tuple)
    return (
        torch.tensor(offset_rows, dtype=torch.long),
        torch.tensor(channel_rows, dtype=torch.long),
    )


def _canonical_channels_for_offsets(
    offset_tuple: tuple[int, ...],
    channel_tuple: tuple[int, ...],
) -> tuple[int, ...]:
    canonical = []
    start = 0
    for run_length in offset_run_lengths(offset_tuple):
        stop = start + run_length
        canonical.extend(sorted(channel_tuple[start:stop]))
        start = stop
    return tuple(canonical)


def _raw_weight_tensor(offsets: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            _raw_weight_for_offsets(tuple(int(value) for value in row))
            for row in offsets.tolist()
        ],
        dtype=dtype,
    )


def _compressed_row_index(
    compressed_offsets: torch.Tensor,
    compressed_channels: torch.Tensor,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], int]:
    return {
        (
            tuple(int(value) for value in offset_row),
            tuple(int(value) for value in channel_row),
        ): index
        for index, (offset_row, channel_row) in enumerate(
            zip(compressed_offsets.tolist(), compressed_channels.tolist())
        )
    }


def _raw_to_compressed_fibre_map(
    raw_offsets: torch.Tensor,
    raw_channels: torch.Tensor,
    compressed_offsets: torch.Tensor,
    compressed_channels: torch.Tensor,
) -> torch.Tensor:
    compressed_lookup = _compressed_row_index(
        compressed_offsets,
        compressed_channels,
    )
    return torch.tensor(
        [
            compressed_lookup[
                (
                    tuple(int(value) for value in offset_row),
                    _canonical_channels_for_offsets(
                        tuple(int(value) for value in offset_row),
                        tuple(int(value) for value in channel_row),
                    ),
                )
            ]
            for offset_row, channel_row in zip(
                raw_offsets.tolist(),
                raw_channels.tolist(),
            )
        ],
        dtype=torch.long,
    )


def _fibre_multiplicity_formula(
    offset_tuple: tuple[int, ...],
    channel_tuple: tuple[int, ...],
) -> int:
    total = 1
    start = 0
    for run_length in offset_run_lengths(offset_tuple):
        stop = start + run_length
        counts = Counter(channel_tuple[start:stop])
        numerator = math.factorial(run_length)
        denominator = math.prod(math.factorial(count) for count in counts.values())
        total *= numerator // denominator
        start = stop
    return total


def test_offset_labelled_signature_has_ordered_block_support_and_values() -> None:
    z = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]],
        dtype=torch.float64,
    )
    kernel_size, channels = z.shape
    width = kernel_size * channels
    depth = 3
    path = _lifted_path_from_window(z)
    sig = signature(path, depth=depth)
    levels = unflatten_signature_levels(sig, width=width, depth=depth)

    for current_depth in range(1, depth + 1):
        level = levels[current_depth - 1][0]
        for word in itertools.product(range(width), repeat=current_depth):
            offsets = tuple(letter // channels for letter in word)
            channel_tuple = tuple(letter % channels for letter in word)
            got = level[word]
            if any(left > right for left, right in zip(offsets, offsets[1:])):
                torch.testing.assert_close(got, torch.tensor(0.0, dtype=z.dtype))
                continue

            expected = torch.tensor(_raw_weight_for_offsets(offsets), dtype=z.dtype)
            for offset, channel in zip(offsets, channel_tuple):
                expected = expected * z[offset, channel]
            torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-10)


def test_tensor_log_offset_coefficients_match_direct_offset_labelled_lift() -> None:
    z = torch.tensor(
        [[0.4, -0.2], [1.5, 0.25], [-0.75, 2.0]],
        dtype=torch.float64,
    )
    kernel_size, channels = z.shape
    width = kernel_size * channels
    depth = 3
    path = _lifted_path_from_window(z)
    sig = signature(path, depth=depth)
    levels = unflatten_signature_levels(sig, width=width, depth=depth)
    log_levels = _signature_to_logsignature_tensor(levels, width=width, depth=depth)

    for current_depth in range(1, depth + 1):
        level = log_levels[current_depth - 1][0]
        for word in itertools.product(range(width), repeat=current_depth):
            offsets = tuple(letter // channels for letter in word)
            channel_tuple = tuple(letter % channels for letter in word)
            expected = torch.tensor(
                log_signature_offset_coefficient(offsets),
                dtype=z.dtype,
            )
            for offset, channel in zip(offsets, channel_tuple):
                expected = expected * z[offset, channel]
            torch.testing.assert_close(level[word], expected, atol=1e-12, rtol=1e-10)


def test_scalar_monomial_fibre_map_collapses_duplicate_tensor_coordinates() -> None:
    dxw = torch.tensor(
        [[[[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]]]],
        dtype=torch.float64,
    )
    raw_offsets, raw_channels = _raw_tensor_feature_rows(
        in_channels=2,
        kernel_size=3,
        depth=3,
    )
    compressed_offsets, compressed_channels, compressed_weights = (
        raw_signature_monomial_indices(
            in_channels=2,
            kernel_size=3,
            depth=3,
            dtype=dxw.dtype,
        )
    )
    fibre_map = _raw_to_compressed_fibre_map(
        raw_offsets,
        raw_channels,
        compressed_offsets,
        compressed_channels,
    )

    raw_features = local_monomial_features_from_offset_channels(
        dxw,
        raw_offsets,
        raw_channels,
        weights=_raw_weight_tensor(raw_offsets, dxw.dtype),
    )
    compressed_features = local_monomial_features_from_offset_channels(
        dxw,
        compressed_offsets,
        compressed_channels,
        weights=compressed_weights,
    )

    torch.testing.assert_close(
        raw_features,
        compressed_features.index_select(dim=2, index=fibre_map),
        atol=1e-12,
        rtol=1e-10,
    )


def test_duplicate_scalar_monomial_fibres_are_null_directions() -> None:
    torch.manual_seed(0)
    dxw = torch.randn(2, 4, 3, 2, dtype=torch.float64)
    raw_offsets, raw_channels = _raw_tensor_feature_rows(
        in_channels=2,
        kernel_size=3,
        depth=3,
    )
    compressed_offsets, compressed_channels, _ = raw_signature_monomial_indices(
        in_channels=2,
        kernel_size=3,
        depth=3,
        dtype=dxw.dtype,
    )
    fibre_map = _raw_to_compressed_fibre_map(
        raw_offsets,
        raw_channels,
        compressed_offsets,
        compressed_channels,
    )
    duplicate_fibre = int(torch.mode(fibre_map).values.item())
    raw_indices = torch.nonzero(fibre_map == duplicate_fibre).flatten()

    raw_features = local_monomial_features_from_offset_channels(
        dxw,
        raw_offsets,
        raw_channels,
        weights=_raw_weight_tensor(raw_offsets, dxw.dtype),
    )
    null_readout = torch.zeros(1, raw_offsets.shape[0], dtype=dxw.dtype)
    null_readout[0, raw_indices[0]] = 1.0
    null_readout[0, raw_indices[1]] = -1.0

    torch.testing.assert_close(
        torch.matmul(raw_features, null_readout.T),
        torch.zeros(dxw.shape[0], dxw.shape[1], 1, dtype=dxw.dtype),
        atol=1e-12,
        rtol=1e-10,
    )


def test_scalar_monomial_fibre_multiplicities_match_raw_coordinate_count() -> None:
    raw_offsets, raw_channels = _raw_tensor_feature_rows(
        in_channels=2,
        kernel_size=2,
        depth=2,
    )
    compressed_offsets, compressed_channels, _ = raw_signature_monomial_indices(
        in_channels=2,
        kernel_size=2,
        depth=2,
    )
    compressed_lookup = _compressed_row_index(compressed_offsets, compressed_channels)
    counts = Counter()
    for offset_row, channel_row in zip(raw_offsets.tolist(), raw_channels.tolist()):
        offset_tuple = tuple(int(value) for value in offset_row)
        counts[
            compressed_lookup[
                (
                    offset_tuple,
                    _canonical_channels_for_offsets(
                        offset_tuple,
                        tuple(int(value) for value in channel_row),
                    ),
                )
            ]
        ] += 1

    expected = []
    for offset_row, channel_row in zip(
        compressed_offsets.tolist(),
        compressed_channels.tolist(),
    ):
        expected.append(
            _fibre_multiplicity_formula(
                tuple(int(value) for value in offset_row),
                tuple(int(value) for value in channel_row),
            )
        )
    assert [counts[index] for index in range(len(expected))] == expected


def test_scalar_monomial_incidence_map_has_exact_rank_and_nullity() -> None:
    channels = 2
    kernel_size = 3
    depth = 3
    raw_offsets, raw_channels = _raw_tensor_feature_rows(
        in_channels=channels,
        kernel_size=kernel_size,
        depth=depth,
    )
    compressed_offsets, compressed_channels, _ = raw_signature_monomial_indices(
        in_channels=channels,
        kernel_size=kernel_size,
        depth=depth,
    )
    fibre_map = _raw_to_compressed_fibre_map(
        raw_offsets,
        raw_channels,
        compressed_offsets,
        compressed_channels,
    )
    incidence = torch.zeros(
        raw_offsets.shape[0],
        compressed_offsets.shape[0],
        dtype=torch.float64,
    )
    incidence[torch.arange(raw_offsets.shape[0]), fibre_map] = 1

    rank = int(torch.linalg.matrix_rank(incidence))
    raw_dimension = math.comb(kernel_size + depth - 1, depth) * channels**depth
    compressed_dimension = math.comb(kernel_size * channels + depth - 1, depth)

    assert raw_offsets.shape[0] == raw_dimension
    assert compressed_offsets.shape[0] == compressed_dimension
    assert rank == compressed_dimension
    assert raw_dimension - rank == raw_dimension - compressed_dimension


def test_scalar_monomial_readouts_have_sparse_and_minimum_norm_lifts() -> None:
    generator = torch.Generator().manual_seed(812)
    windows = torch.randn(
        2,
        4,
        3,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    raw_offsets, raw_channels = _raw_tensor_feature_rows(2, 3, 3)
    compressed_offsets, compressed_channels, compressed_scaling = (
        raw_signature_monomial_indices(
            in_channels=2,
            kernel_size=3,
            depth=3,
            dtype=windows.dtype,
        )
    )
    fibre_map = _raw_to_compressed_fibre_map(
        raw_offsets,
        raw_channels,
        compressed_offsets,
        compressed_channels,
    )
    incidence = windows.new_zeros(raw_offsets.shape[0], compressed_offsets.shape[0])
    incidence[torch.arange(raw_offsets.shape[0]), fibre_map] = 1
    raw_features = local_monomial_features_from_offset_channels(
        windows,
        raw_offsets,
        raw_channels,
        weights=_raw_weight_tensor(raw_offsets, windows.dtype),
    )
    compressed_features = local_monomial_features_from_offset_channels(
        windows,
        compressed_offsets,
        compressed_channels,
        weights=compressed_scaling,
    )

    raw_readout = torch.randn(
        3,
        raw_offsets.shape[0],
        generator=generator,
        dtype=windows.dtype,
    )
    transferred = raw_readout @ incidence
    torch.testing.assert_close(
        raw_features @ raw_readout.T,
        compressed_features @ transferred.T,
        atol=1e-12,
        rtol=1e-10,
    )

    compressed_readout = torch.randn(
        3,
        compressed_offsets.shape[0],
        generator=generator,
        dtype=windows.dtype,
    )
    fibre_sizes = torch.bincount(
        fibre_map,
        minlength=compressed_offsets.shape[0],
    )
    minimum_norm_lift = compressed_readout.index_select(1, fibre_map) / (
        fibre_sizes.index_select(0, fibre_map)
    )
    sparse_lift = torch.zeros_like(minimum_norm_lift)
    for compressed_index in range(compressed_offsets.shape[0]):
        raw_index = int(torch.nonzero(fibre_map == compressed_index)[0])
        sparse_lift[:, raw_index] = compressed_readout[:, compressed_index]

    for lift in (sparse_lift, minimum_norm_lift):
        torch.testing.assert_close(lift @ incidence, compressed_readout)
        torch.testing.assert_close(
            raw_features @ lift.T,
            compressed_features @ compressed_readout.T,
            atol=1e-12,
            rtol=1e-10,
        )
    assert minimum_norm_lift.norm() < sparse_lift.norm()


def test_raw_scalar_monomial_dictionary_has_full_dimension_through_degree() -> None:
    kernel_size = 3
    channels = 2
    max_depth = 3
    feature_count = 1
    for depth in range(1, max_depth + 1):
        offsets, _, _ = raw_signature_monomial_indices(
            in_channels=channels,
            kernel_size=kernel_size,
            depth=depth,
        )
        feature_count += offsets.shape[0]

    assert feature_count == math.comb(kernel_size * channels + max_depth, max_depth)
