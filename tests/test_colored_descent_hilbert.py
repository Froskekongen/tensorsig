import itertools
import math

import pytest
import torch

from tensorsig.classical.signatures import signature
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.colored_descent import (
    bounded_gap_count,
    cut_rank_target,
    generic_scalar_dimension,
    ribbon_dimension,
)
from tensorsig.tensor_ops.series import unflatten_signature_levels


def _coded_levels(
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


def _required_cut_rank(
    code: torch.Tensor,
    depth: int,
    cuts: tuple[int, ...],
) -> int:
    rows = []
    for offsets in itertools.combinations_with_replacement(
        range(code.shape[0]),
        depth,
    ):
        if not all(offsets[cut - 1] < offsets[cut] for cut in cuts):
            continue
        row = code[offsets[0]]
        for offset in offsets[1:]:
            row = torch.kron(row, code[offset])
        rows.append(row)
    return int(torch.linalg.matrix_rank(torch.stack(rows)).item())


def _block_symmetric_channel_basis(
    in_channels: int,
    depth: int,
    cuts: tuple[int, ...],
) -> torch.Tensor:
    """Return orbit sums for tensors symmetric inside the cut blocks."""
    boundaries = (0, *cuts, depth)
    blocks = tuple(zip(boundaries, boundaries[1:]))
    words = tuple(itertools.product(range(in_channels), repeat=depth))
    keys = [
        tuple(tuple(sorted(word[start:stop])) for start, stop in blocks)
        for word in words
    ]
    positions = {key: index for index, key in enumerate(dict.fromkeys(keys))}
    basis = torch.zeros(len(words), len(positions), dtype=torch.float64)
    for row, key in enumerate(keys):
        basis[row, positions[key]] = 1
    return basis


def _runtime_signature_rank(
    kernel_size: int,
    code_dim: int,
    in_channels: int,
    depth: int,
    expected_rank: int,
) -> int:
    generator = torch.Generator().manual_seed(
        1000 + 100 * kernel_size + 10 * code_dim + in_channels
    )
    sample_count = max(64, expected_rank + 24)
    windows = torch.randn(
        sample_count,
        1,
        kernel_size,
        in_channels,
        generator=generator,
        dtype=torch.float64,
    )
    code = OffsetCode.pascal(
        kernel_size,
        code_dim,
        dtype=torch.float64,
    ).matrix
    level = _coded_levels(windows, code, depth)[depth - 1]
    features = level[:, 0].reshape(sample_count, -1)
    return int(torch.linalg.matrix_rank(features).item())


@pytest.mark.parametrize(
    ("kernel_size", "code_dim", "in_channels", "depth", "expected"),
    [
        (1, 1, 2, 3, 4),
        (3, 1, 2, 3, 8),
        (3, 2, 2, 3, 44),
        (3, 3, 2, 3, 56),
        (4, 2, 2, 3, 60),
        (4, 4, 2, 3, 120),
    ],
)
def test_runtime_signature_rank_matches_colored_descent_dimension(
    kernel_size: int,
    code_dim: int,
    in_channels: int,
    depth: int,
    expected: int,
) -> None:
    predicted = generic_scalar_dimension(
        kernel_size,
        code_dim,
        in_channels,
        depth,
    )
    assert predicted == expected
    assert (
        _runtime_signature_rank(
            kernel_size,
            code_dim,
            in_channels,
            depth,
            expected_rank=predicted,
        )
        == predicted
    )


@pytest.mark.parametrize(
    ("kernel_size", "code_dim", "depth"),
    [(3, 2, 3), (5, 3, 3), (6, 3, 4)],
)
def test_pascal_required_cut_ranks_match_bounded_gap_targets(
    kernel_size: int,
    code_dim: int,
    depth: int,
) -> None:
    code = OffsetCode.pascal(
        kernel_size,
        code_dim,
        dtype=torch.float64,
    ).matrix
    for cut_count in range(depth):
        for cuts in itertools.combinations(range(1, depth), cut_count):
            assert _required_cut_rank(code, depth, cuts) == cut_rank_target(
                kernel_size,
                code_dim,
                depth,
                cuts,
            )


@pytest.mark.parametrize(
    ("in_channels", "depth"),
    [(2, 3), (3, 3), (2, 4), (3, 4)],
)
def test_higher_degree_ribbon_dimensions_match_channel_quotients_and_descents(
    in_channels: int,
    depth: int,
) -> None:
    words = tuple(itertools.product(range(in_channels), repeat=depth))
    cut_positions = tuple(range(1, depth))
    for cut_count in range(depth):
        for cuts in itertools.combinations(cut_positions, cut_count):
            channel_space = _block_symmetric_channel_basis(
                in_channels,
                depth,
                cuts,
            )
            proper_subspaces = [
                _block_symmetric_channel_basis(in_channels, depth, subset)
                for subset_size in range(len(cuts))
                for subset in itertools.combinations(cuts, subset_size)
            ]
            lower_rank = (
                int(torch.linalg.matrix_rank(torch.cat(proper_subspaces, dim=1)))
                if proper_subspaces
                else 0
            )
            quotient_dimension = channel_space.shape[1] - lower_rank
            descent_word_count = sum(
                tuple(
                    position + 1
                    for position in range(depth - 1)
                    if word[position] > word[position + 1]
                )
                == cuts
                for word in words
            )

            assert (
                quotient_dimension
                == descent_word_count
                == ribbon_dimension(in_channels, depth, cuts)
            )


def test_runtime_degree_two_level_matches_offset_code_expansion() -> None:
    code = OffsetCode.pascal(2, 2, dtype=torch.float64).matrix
    window = torch.tensor(
        [[[[2.0, -1.0], [3.0, 5.0]]]],
        dtype=torch.float64,
    )
    actual = _coded_levels(window, code, 2)[1][0, 0]

    expected = torch.zeros_like(actual)
    for left_letter, right_letter in itertools.product(range(4), repeat=2):
        left_channel, left_code = divmod(left_letter, 2)
        right_channel, right_code = divmod(right_letter, 2)
        for left_offset in range(2):
            for right_offset in range(left_offset, 2):
                scale = 0.5 if left_offset == right_offset else 1.0
                expected[left_letter, right_letter] += (
                    scale
                    * code[left_offset, left_code]
                    * code[right_offset, right_code]
                    * window[0, 0, left_offset, left_channel]
                    * window[0, 0, right_offset, right_channel]
                )

    torch.testing.assert_close(actual, expected)


def test_combinatorics_recover_signature_endpoints() -> None:
    assert bounded_gap_count(3, 2, 3) == 7
    assert ribbon_dimension(2, 2, ()) == math.comb(3, 2)
    assert ribbon_dimension(2, 2, (1,)) == math.comb(2, 2)

    for kernel_size, in_channels, depth in [(2, 3, 2), (3, 2, 3)]:
        assert generic_scalar_dimension(
            kernel_size,
            kernel_size,
            in_channels,
            depth,
        ) == math.comb(
            kernel_size * in_channels + depth - 1,
            depth,
        )
