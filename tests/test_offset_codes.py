import pytest
import torch

from tensorsig.offset_coded_signatures.codes import OffsetCode


def test_orthonormal_pascal_code_has_a_canonical_positive_qr_gauge() -> None:
    raw = OffsetCode.pascal(
        8,
        4,
        orthonormal=False,
        dtype=torch.float64,
    ).matrix
    code = OffsetCode.pascal(8, 4, dtype=torch.float64).matrix
    triangular = torch.linalg.lstsq(code, raw).solution

    torch.testing.assert_close(code.T @ code, torch.eye(4, dtype=torch.float64))
    torch.testing.assert_close(code @ triangular, raw)
    assert torch.all(torch.diagonal(triangular) > 0)
    assert torch.linalg.cond(code) < torch.linalg.cond(raw)


def test_structured_polynomials_and_pascal_span_the_same_temporal_space() -> None:
    pascal = OffsetCode.pascal(8, 4, dtype=torch.float64).matrix
    polynomial = OffsetCode.structured(
        8,
        4,
        kind="polynomial",
        include_constant=True,
        dtype=torch.float64,
    ).matrix

    torch.testing.assert_close(
        pascal @ pascal.T,
        polynomial @ torch.linalg.pinv(polynomial),
        atol=1e-12,
        rtol=1e-10,
    )


def test_structured_code_families_have_the_requested_geometry() -> None:
    identity = OffsetCode.structured(
        5,
        5,
        kind="identity",
        include_constant=False,
        dtype=torch.float64,
    )
    dct = OffsetCode.structured(
        6,
        6,
        kind="dct",
        include_constant=True,
        dtype=torch.float64,
    )
    minirocket = OffsetCode.structured(
        9,
        4,
        kind="minirocket",
        include_constant=False,
        positive_count=3,
        dtype=torch.float64,
    )

    torch.testing.assert_close(identity.matrix, torch.eye(5, dtype=torch.float64))
    assert dct.rank == 6
    assert torch.all((minirocket.matrix == 2) | (minirocket.matrix == -1))
    torch.testing.assert_close(
        minirocket.matrix.mean(0),
        torch.zeros(4, dtype=torch.float64),
    )


def test_structured_code_validation_is_exact_and_case_sensitive() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        OffsetCode.structured(3, 2, kind="DCT")
    with pytest.raises(ValueError, match="constant codes"):
        OffsetCode.structured(3, 2, kind="constant")
    with pytest.raises(ValueError, match="positive_count"):
        OffsetCode.structured(
            4,
            2,
            kind="minirocket",
            include_constant=False,
            positive_count=4,
        )
