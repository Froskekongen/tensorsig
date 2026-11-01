"""Exercise installed distributions outside the source checkout."""

from pathlib import Path

import torch
import tensorsig
from tensorsig.classical.signatures import log_signature, signature
from tensorsig.classical.windows import windowed_signature
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.layers import LocalFeatures
from tensorsig.offset_coded_signatures.plans import LocalRealization


def main() -> None:
    # A release must exercise the installed package, never an editable checkout.
    checkout = Path(__file__).resolve().parents[1]
    assert not Path(tensorsig.__file__).resolve().is_relative_to(checkout)

    path = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    first_turn = path[:, :3]
    sig = signature(first_turn, depth=2)
    logsig = log_signature(first_turn, depth=2)
    torch.testing.assert_close(sig, path.new_tensor([[1.0, 1.0, 0.5, 1.0, 0.0, 0.5]]))
    torch.testing.assert_close(logsig, path.new_tensor([[1.0, 1.0, 0.5]]))
    windows = windowed_signature(path, depth=2, window_size=3, hop_size=1)
    assert windows.shape == (1, 2, 6)
    torch.testing.assert_close(windows[:, 0], sig)

    # A constant one-dimensional code is the ordinary sliding-window signature.
    code = OffsetCode.constant(window=2, dtype=path.dtype)
    plan = LocalRealization(code, channels=2, max_depth=2).signature(
        coordinates="dense"
    )
    features = LocalFeatures(plan)(path)
    torch.testing.assert_close(features, windows)
    loss = sig.square().sum() + logsig.square().sum() + features.square().sum()
    loss.backward()
    assert path.grad is not None and torch.isfinite(path.grad).all()
    assert path.grad.abs().sum() > 0
    print(f"Installed tensorsig smoke test passed: {tensorsig.__file__}")


if __name__ == "__main__":
    main()
