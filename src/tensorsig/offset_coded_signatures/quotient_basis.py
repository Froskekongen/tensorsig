"""Stable labels for reconstructed Lie-quotient coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


type _HallTree = int | tuple[_HallTree, _HallTree]


def _hall_tree_depth(tree: _HallTree) -> int:
    if isinstance(tree, bool) or not isinstance(tree, (int, tuple)):
        raise TypeError("A Hall tree must be a positive letter or a binary tuple.")
    if isinstance(tree, int):
        if tree < 1:
            raise ValueError("Hall-tree letters must be positive.")
        return 1
    if len(tree) != 2:
        raise ValueError("A Hall-tree bracket must contain two children.")
    return _hall_tree_depth(tree[0]) + _hall_tree_depth(tree[1])


@dataclass(frozen=True)
class QuotientBasisElement:
    """One immutable label for a reconstructed quotient coordinate.

    ``kind="hall"`` uses ``tree``: positive one-based letters and nested
    ``(left, right)`` tuples encode generators and Lie brackets.  ``kind="trace"``
    uses ``word``: a one-based canonical word labels an exact trace-normal-form
    basis element. Exactly one of ``tree`` and ``word`` is present.

    Parameters
    ----------
    kind : {"hall", "trace"}
        Normal-form family used by the exact quotient compiler.
    depth : int
        Homogeneous Lie degree.
    tree : int or tuple, optional
        Recursive Hall tree for ``kind="hall"``.
    word : tuple[int, ...], optional
        Canonical one-based trace word for ``kind="trace"``.
    """

    kind: Literal["hall", "trace"]
    depth: int
    tree: _HallTree | None = None
    word: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.depth, int) or isinstance(self.depth, bool):
            raise TypeError("depth must be an integer.")
        if self.depth < 1:
            raise ValueError("depth must be positive.")
        if self.kind == "hall":
            if self.tree is None or self.word is not None:
                raise ValueError("A Hall label requires only tree.")
            if _hall_tree_depth(self.tree) != self.depth:
                raise ValueError("depth does not match the Hall tree.")
            return
        if self.kind == "trace":
            if self.tree is not None or self.word is None:
                raise ValueError("A trace label requires only word.")
            if not isinstance(self.word, tuple):
                raise TypeError("A trace word must be a tuple.")
            if len(self.word) != self.depth:
                raise ValueError("depth does not match the trace word.")
            if any(
                not isinstance(letter, int) or isinstance(letter, bool) or letter < 1
                for letter in self.word
            ):
                raise ValueError("Trace-word letters must be positive integers.")
            return
        raise ValueError("kind must be 'hall' or 'trace'.")


__all__ = ["QuotientBasisElement"]
