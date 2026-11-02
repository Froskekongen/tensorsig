"""Sliding-window signatures and log-signatures.

Window sizes count path points, while hops count the shift between successive
point windows. The helpers either evaluate windows directly or convert a
streaming prefix signature into valid window signatures using Chen removal.
"""

from torch import Tensor, nn

from .log_signature import windowed_log_signature as windowed_log_signature
from .signature import stream_to_window_signatures as stream_to_window_signatures
from .signature import windowed_signature as windowed_signature


class WindowedLogSignature(nn.Module):
    """A windowed logsignature transform for ``torch.nn.Sequential``.

    Compose arbitrary path augmentations before this module, for example
    ``nn.Sequential(fir, products, WindowedLogSignature(2, 5))``. Augmentations
    see the full path before windows are extracted. The transform itself does
    not impose causality on preceding modules.

    Parameters
    ----------
    depth : int
        Positive logsignature truncation depth.
    window_size : int
        Number of path points per window (at least two).
    hop_size : int, optional
        Number of points between window starts.
    mode : {"words", "hall"}, optional
        Logsignature coordinate basis.
    """

    def __init__(
        self,
        depth: int,
        window_size: int,
        hop_size: int = 1,
        mode: str = "words",
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("depth", depth, 1),
            ("window_size", window_size, 2),
            ("hop_size", hop_size, 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        if not isinstance(mode, str):
            raise TypeError("mode must be a string.")
        mode = mode.lower()
        if mode not in {"words", "hall"}:
            raise ValueError("mode must be either 'words' or 'hall'.")
        self.depth = depth
        self.window_size = window_size
        self.hop_size = hop_size
        self.mode = mode

    def forward(self, path: Tensor) -> Tensor:
        """Map path points ``(B, L, C)`` to window coordinates ``(B, N, D)``."""
        return windowed_log_signature(
            path,
            depth=self.depth,
            window_size=self.window_size,
            hop_size=self.hop_size,
            mode=self.mode,
        )

    def extra_repr(self) -> str:
        return (
            f"depth={self.depth}, window_size={self.window_size}, "
            f"hop_size={self.hop_size}, mode={self.mode!r}"
        )


__all__ = [
    "windowed_signature",
    "windowed_log_signature",
    "stream_to_window_signatures",
    "WindowedLogSignature",
]
