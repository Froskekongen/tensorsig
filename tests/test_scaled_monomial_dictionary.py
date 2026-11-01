import torch

from tensorsig.offset_coded_signatures.windows import increment_windows
from tensorsig.offset_coded_signatures.scaled_monomials import (
    ScaledMonomialDictionary,
)


def test_raw_signature_dictionary_matches_explicit_scaled_monomials() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 7, 2, dtype=torch.float64)
    dxw = increment_windows(x, size=3, stride=2)
    dictionary = ScaledMonomialDictionary.raw_signature(
        in_channels=2,
        kernel_size=3,
        depth=3,
        scale="signature",
        dtype=x.dtype,
    )

    actual = dictionary.features(dxw, chunk_size=4)
    columns = []
    for offsets, channels, coefficient in zip(
        dictionary.offsets,
        dictionary.channels,
        dictionary.coefficients,
        strict=True,
    ):
        column = torch.ones(*dxw.shape[:2], dtype=dxw.dtype)
        for offset, channel in zip(offsets, channels, strict=True):
            column = column * dxw[..., int(offset), int(channel)]
        columns.append(coefficient * column)
    expected = torch.stack(columns, dim=-1)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)


def test_dictionary_contract_matches_materialized_features() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 8, 3, dtype=torch.float64)
    dxw = increment_windows(x, size=4)
    dictionary = ScaledMonomialDictionary.raw_signature(
        in_channels=3,
        kernel_size=4,
        depth=2,
        max_offset_span=2,
        scale="support",
        dtype=x.dtype,
    )
    readout = torch.randn(5, dictionary.feature_count, dtype=x.dtype)

    actual = dictionary.contract(dxw, readout, chunk_size=3)
    expected = torch.matmul(dictionary.features(dxw, chunk_size=3), readout.T)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
