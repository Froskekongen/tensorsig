"""Execute code embedded in public docstrings and canonical guides."""

from __future__ import annotations

import doctest
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.generate_docs import discover_api_model


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

API = discover_api_model()
DOC_MODULES = sorted(
    {section.module for section in API.sections}
    | {
        symbol.canonical_path.rsplit(".", 1)[0]
        for section in API.sections
        for symbol in section.symbols
    }
)
GUIDE_SOURCES = sorted((ROOT / "docs").rglob("*.py"))


@pytest.mark.parametrize("module_name", DOC_MODULES)
def test_public_docstrings_run(module_name: str) -> None:
    module = importlib.import_module(module_name)
    failures, _ = doctest.testmod(
        module,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert failures == 0, f"Doctests failed for {module_name}"


@pytest.mark.parametrize(
    "source", GUIDE_SOURCES, ids=lambda path: str(path.relative_to(ROOT))
)
def test_jupytext_guide_runs(source: Path) -> None:
    """Each percent guide must execute without state from another guide."""
    subprocess.run(
        [sys.executable, str(source)],
        cwd=ROOT,
        check=True,
        timeout=120,
    )


def test_readme_quick_start_runs() -> None:
    blocks = re.findall(
        r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.DOTALL
    )
    assert blocks, "README must contain a runnable quick start"
    for block in blocks:
        exec(compile(block, "README.md", "exec"), {})
