"""Runtime APIs must stay independent of optional verification modules."""

import ast
from importlib.util import resolve_name
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_maintained_source_has_no_research_imports():
    source = ROOT / "src"
    for path in (source / "tensorsig").rglob("*.py"):
        if "experimental" in path.relative_to(source).parts:
            continue
        parts = path.relative_to(source).with_suffix("").parts
        package = ".".join(parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                base = resolve_name(name, package) if node.level else name
                names = [base, *(base + "." + alias.name for alias in node.names)]
            assert not any("experimental" in name.split(".") for name in names), (
                path,
                getattr(node, "lineno", None),
                names,
            )


def test_maintained_import_does_not_load_research():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, tensorsig; "
            "assert not any('experimental' in name.split('.') for name in sys.modules "
            "if name.startswith('tensorsig'))",
        ],
        check=True,
        cwd=ROOT,
    )
