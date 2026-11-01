"""Tests for the source-owned documentation build."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest

from scripts.generate_docs import (
    PACKAGE_NAME,
    build_page_model,
    discover_api_model,
    literal_exports,
    render_guides,
    write_generated_tree,
)


ROOT = Path(__file__).resolve().parents[1]
DOC_SOURCES = tuple(sorted((ROOT / "docs").rglob("*.py")))

EXPECTED_PUBLIC_TREE = {
    "tensorsig": (
        ("classical", "algebra", "tensor_ops", "offset_coded_signatures"),
        (),
    ),
    "tensorsig.classical": (("signatures", "windows", "sparse"), ()),
    "tensorsig.classical.signatures": ((), ("signature", "log_signature")),
    "tensorsig.classical.windows": (
        (),
        (
            "windowed_signature",
            "windowed_log_signature",
            "stream_to_window_signatures",
        ),
    ),
    "tensorsig.classical.sparse": (
        (),
        ("compress_path", "signature_sparse", "pad_paths_correctly"),
    ),
    "tensorsig.algebra": (("hall", "lyndon"), ()),
    "tensorsig.algebra.hall": ((), ("hall_basis", "logsigdim", "logsigkeys")),
    "tensorsig.algebra.lyndon": (
        (),
        ("lyndon_words", "logsigdim_words", "logsigkeys_words"),
    ),
    "tensorsig.tensor_ops": (("products", "linear", "series"), ()),
    "tensorsig.tensor_ops.products": ((), ("tensor_product", "lie_bracket")),
    "tensorsig.tensor_ops.linear": (
        (),
        (
            "apply_linear_map_to_tensor_level",
            "apply_linear_map_to_signature_levels",
            "contract_features_with_readout",
        ),
    ),
    "tensorsig.tensor_ops.series": (
        (),
        (
            "flatten_signature_levels",
            "unflatten_signature_levels",
            "signature_inverse",
            "signature_multiply",
            "batch_restricted_exp",
        ),
    ),
    "tensorsig.offset_coded_signatures": (
        (
            "codes",
            "coordinates",
            "plans",
            "quotients",
            "layers",
            "augmentations",
            "windows",
            "graph_queries",
            "active_word_log",
            "rolling_logsignature",
            "pascal_rank",
            "pascal_certificate",
        ),
        (),
    ),
    "tensorsig.offset_coded_signatures.codes": ((), ("AlphabetMap", "OffsetCode")),
    "tensorsig.offset_coded_signatures.coordinates": (
        (),
        ("ChronologicalFiltration", "WordSelection"),
    ),
    "tensorsig.offset_coded_signatures.plans": (
        (),
        (
            "LocalRealization",
            "LogSignaturePlan",
            "PolynomialPlan",
            "SignaturePlan",
            "compile_polynomial",
        ),
    ),
    "tensorsig.offset_coded_signatures.quotients": (
        (),
        (
            "LieQuotient",
            "DependenceGraph",
            "QuotientBasisElement",
            "ChannelMultiplicity",
            "ChannelSupport",
            "OffsetMultiplicity",
            "OffsetSpan",
            "OffsetSupport",
            "TemporalDegree",
            "LieQuotientPlan",
            "compile_lie_quotient",
            "OffsetAlphabet",
            "ProjectedAlphabet",
            "WindowLogSignature",
            "WindowSelection",
        ),
    ),
    "tensorsig.offset_coded_signatures.layers": (
        (),
        ("LocalFeatures", "LocalLinear", "GapFactorizedVolterra"),
    ),
    "tensorsig.offset_coded_signatures.augmentations": (
        (),
        (
            "causal_fir",
            "CausalFIRAugmentation",
            "CausalWindowedLogSignature",
            "PairwiseProductAugmentation",
        ),
    ),
    "tensorsig.offset_coded_signatures.windows": ((), ("increment_windows",)),
    "tensorsig.offset_coded_signatures.active_word_log": (
        (),
        ("ActiveWordLogPlan", "compile_active_word_log", "active_word_log_signature"),
    ),
    "tensorsig.offset_coded_signatures.rolling_logsignature": (
        (),
        (
            "RollingLogSignature",
            "Transport",
            "pascal_transport",
            "general_bch_program",
            "joint_program",
        ),
    ),
    "tensorsig.offset_coded_signatures.pascal_rank": (
        (),
        ("binary_pascal_rank", "binary_pascal_profile", "binary_lie_weight_dimension"),
    ),
    "tensorsig.offset_coded_signatures.pascal_certificate": (
        (),
        ("PascalRankCertificate", "binary_pascal_rank_certificate"),
    ),
    "tensorsig.offset_coded_signatures.graph_queries": (
        (),
        (
            "tree_moment",
            "CellQueryMetrics",
            "DirectNbcPlan",
            "ChordalNbcPlan",
            "compile_direct_nbc",
            "compile_chordal_nbc",
            "cell_survives",
            "cell_witness",
            "ResourceLimit",
            "literal_tree_energy",
            "star_trace_gram_energy",
            "one_centre_gram_energy",
            "compile_graph_block_coordinates",
            "decode_coordinates",
            "rational",
            "replay",
        ),
    ),
}


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_public_tree_and_pages_come_from_source() -> None:
    api = discover_api_model()
    assert api.package == PACKAGE_NAME
    actual = {
        section.module: (
            tuple(child.rsplit(".", 1)[-1] for child in section.children),
            tuple(symbol.name for symbol in section.symbols),
        )
        for section in api.sections
    }
    assert actual == EXPECTED_PUBLIC_TREE
    assert all(section.summary for section in api.sections)

    symbols = [symbol for section in api.sections for symbol in section.symbols]
    assert all(
        symbol.public_paths == (f"{symbol.owner_module}.{symbol.name}",)
        for symbol in symbols
    )
    assert len({symbol.public_paths[0] for symbol in symbols}) == len(symbols)
    assert len({symbol.canonical_path for symbol in symbols}) == len(symbols)

    paths = {page.path for page in build_page_model(api).pages}
    assert Path("api/index.md") in paths
    assert Path("api/classical/index.md") in paths
    assert Path("api/classical/signatures/log-signature.md") in paths
    assert Path("api/offset-coded-signatures/plans/local-realization.md") in paths


def test_intermediate_packages_export_modules_only_and_old_flat_exports_are_absent() -> (
    None
):
    for module_name, (children, symbols) in EXPECTED_PUBLIC_TREE.items():
        module = import_module(module_name)
        assert tuple(module.__all__) == (*children, *symbols)
        if children:
            assert not symbols
            assert all(
                isinstance(getattr(module, name), ModuleType) for name in children
            )

    root = import_module("tensorsig")
    old_root_exports = {
        "signature",
        "signature_inverse",
        "signature_multiply",
        "windowed_signature",
        "stream_to_window_signatures",
        "log_signature",
        "windowed_log_signature",
        "hall_basis",
        "logsigdim",
        "logsigkeys",
        "logsigdim_words",
        "logsigkeys_words",
        "lyndon_words",
        "compress_path",
        "signature_sparse",
        "pad_paths_correctly",
    }
    offset = import_module("tensorsig.offset_coded_signatures")
    tensor_ops = import_module("tensorsig.tensor_ops")
    assert old_root_exports.isdisjoint(vars(root))
    assert {
        symbol
        for module, (_, symbols) in EXPECTED_PUBLIC_TREE.items()
        if module.startswith("tensorsig.offset_coded_signatures.")
        for symbol in symbols
    }.isdisjoint(vars(offset))
    assert {
        symbol
        for module, (_, symbols) in EXPECTED_PUBLIC_TREE.items()
        if module.startswith("tensorsig.tensor_ops.")
        for symbol in symbols
    }.isdisjoint(vars(tensor_ops))
    assert "lie_brackets" not in vars(import_module("tensorsig.tensor_ops.products"))


def test_static_discovery_preserves_all_order_without_importing(tmp_path: Path) -> None:
    package = tmp_path / "example"
    package.mkdir()
    (package / "__init__.py").write_text(
        '"""Example API."""\nfrom . import tools as tools\n__all__ = ["tools"]\n',
        encoding="utf-8",
    )
    (package / "tools.py").write_text(
        '"""Public tools."""\n'
        'raise RuntimeError("must not import")\n\n'
        '__all__ = ["second", "first"]\n\n'
        'def first():\n    """First tool."""\n\n'
        'def second():\n    """Second tool."""\n',
        encoding="utf-8",
    )

    api = discover_api_model(tmp_path, "example")
    assert [section.module for section in api.sections] == ["example", "example.tools"]
    assert [symbol.name for symbol in api.sections[1].symbols] == ["second", "first"]


def test_duplicate_final_targets_are_rejected(tmp_path: Path) -> None:
    package = tmp_path / "example"
    package.mkdir()
    (package / "__init__.py").write_text(
        '"""Example API."""\nfrom . import left, right\n__all__ = ["left", "right"]\n',
        encoding="utf-8",
    )
    (package / "impl.py").write_text(
        'def shared():\n    """Shared implementation."""\n', encoding="utf-8"
    )
    for name in ("left", "right"):
        (package / f"{name}.py").write_text(
            f'"""{name.title()} API."""\n'
            f"from .impl import shared as {name}\n"
            f'__all__ = ["{name}"]\n',
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Duplicate public target"):
        discover_api_model(tmp_path, "example")


def test_public_nodes_cannot_mix_modules_and_symbols(tmp_path: Path) -> None:
    package = tmp_path / "example"
    package.mkdir()
    (package / "__init__.py").write_text(
        '"""Example API."""\n'
        "from . import tools as tools\n"
        'def public():\n    """A public function."""\n'
        '__all__ = ["tools", "public"]\n',
        encoding="utf-8",
    )
    (package / "tools.py").write_text(
        '"""Public tools."""\ndef tool():\n    """A tool."""\n__all__ = ["tool"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot mix"):
        discover_api_model(tmp_path, "example")


def test_public_modules_require_summaries_and_resolved_exports(tmp_path: Path) -> None:
    package = tmp_path / "example"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text('""""""\n__all__ = ["missing"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing a docstring"):
        discover_api_model(tmp_path, "example")

    init.write_text('"""Example API."""\n__all__ = ["missing"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="undefined"):
        discover_api_model(tmp_path, "example")


@pytest.mark.parametrize(
    "definition",
    (
        'NAMES = ["api"]\n__all__ = NAMES\n',
        '__all__ = ["api", "api"]\n',
        '__all__ = ["api"]\n__all__ += ["other"]\n',
    ),
)
def test_exports_must_be_one_literal_unique_sequence(
    tmp_path: Path, definition: str
) -> None:
    source = tmp_path / "__init__.py"
    source.write_text(definition, encoding="utf-8")
    with pytest.raises(ValueError, match="__all__"):
        literal_exports(source)


def test_rendered_guides_and_generated_tree_are_deterministic(tmp_path: Path) -> None:
    guides = render_guides()
    assert Path("index.md") in guides
    assert Path("development/mathematical_verification.md") in guides
    signature_guides = (
        Path("signatures/index.md"),
        Path("signatures/signature_concepts.md"),
        Path("signatures/logsignature_concepts.md"),
        Path("signatures/using_the_api.md"),
    )
    assert set(signature_guides) <= guides.keys()
    assert guides[Path("index.md")] == (ROOT / "README.md").read_text()
    assert not any(path.parts[0] == "offset_coded_signatures" for path in guides)

    model = build_page_model(discover_api_model())
    output = tmp_path / "generated-docs"
    config = tmp_path / ".generated-site.yml"
    write_generated_tree(model, guides, output, config)
    first_docs = _snapshot(output)
    first_config = config.read_text(encoding="utf-8")
    assert all(path.as_posix() in first_config for path in signature_guides)
    positions = [first_config.index(path.as_posix()) for path in signature_guides]
    assert positions == sorted(positions)
    assert "offset_coded_signatures/" not in first_config
    assert "api/offset-coded-signatures/plans/local-realization.md" in first_config

    generated_api = "\n".join(page.body for page in model.pages)
    private_paths = {
        symbol.canonical_path
        for section in discover_api_model().sections
        for symbol in section.symbols
        if symbol.canonical_path not in symbol.public_paths
    }
    assert all(path not in generated_api for path in private_paths)

    (output / "stale.md").write_text("stale", encoding="utf-8")
    write_generated_tree(model, guides, output, config)
    assert _snapshot(output) == first_docs
    assert config.read_text(encoding="utf-8") == first_config


@pytest.mark.parametrize(
    "source",
    DOC_SOURCES,
    ids=lambda source: source.relative_to(ROOT).as_posix(),
)
def test_python_guide_executes(source: Path) -> None:
    runpy.run_path(source)


def test_generator_refuses_unsafe_output_paths(tmp_path: Path) -> None:
    guides = render_guides()
    model = build_page_model(discover_api_model())
    with pytest.raises(ValueError, match="unsafe generated path"):
        write_generated_tree(
            model,
            guides,
            tmp_path / "docs",
            tmp_path / ".generated-site.yml",
        )


def test_generation_does_not_rewrite_sources(tmp_path: Path) -> None:
    sources = (ROOT / "README.md", *DOC_SOURCES)
    before = {path: path.read_bytes() for path in sources}
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_docs.py"),
            "--output",
            str(tmp_path / "generated-docs"),
            "--config",
            str(tmp_path / ".generated-site.yml"),
        ],
        check=True,
        cwd=ROOT,
    )
    assert {path: path.read_bytes() for path in sources} == before
    assert not (ROOT / "tests/mathematical_verification_guide.md").exists()


def test_raw_python_fences_in_guide_markdown_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "index.py").write_text(
        "# %% [markdown]\n# ```python\n# print('not executed')\n# ```\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="executable code cells"):
        render_guides(tmp_path)


def test_generated_docs_build_with_zensical_in_strict_mode(tmp_path: Path) -> None:
    output = tmp_path / "generated-docs"
    config = tmp_path / ".generated-site.yml"
    site = tmp_path / "site"
    guides = render_guides()
    write_generated_tree(build_page_model(discover_api_model()), guides, output, config)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "zensical",
            "build",
            "--clean",
            "--strict",
            "-f",
            str(config),
        ],
        cwd=ROOT,
        check=True,
    )
    assert (site / "index.html").is_file()
    api_page = site / "api" / "classical" / "signatures" / "signature" / "index.html"
    api_html = api_page.read_text(encoding="utf-8")
    assert "doc-object doc-function" in api_html
    assert "signature(" in api_html
    assert "headerlink" in api_html

    crossref_html = (
        site / "api" / "tensor-ops" / "products" / "lie-bracket" / "index.html"
    ).read_text(encoding="utf-8")
    assert 'href="../tensor-product/#tensorsig.tensor_ops.products.tensor_product"' in (
        crossref_html
    )

    guide_html = (site / "signatures" / "signature_concepts" / "index.html").read_text(
        encoding="utf-8"
    )
    assert 'class="arithmatex"' in guide_html
    assert 'class="highlight"' in guide_html
    assert 'name="viewport"' in guide_html
    config_script = guide_html.index("javascripts/mathjax.js")
    runtime_script = guide_html.index("mathjax@3")
    assert config_script < runtime_script

    mathjax = (output / "javascripts" / "mathjax.js").read_text(encoding="utf-8")
    assert 'processHtmlClass: "arithmatex"' in mathjax
    assert "document$.subscribe" in mathjax
    assert "MathJax.typesetPromise" in mathjax

    search = (site / "search.json").read_text(encoding="utf-8")
    assert "signature_inverse" in search
    assert "LocalFeatures" in search
    assert all(
        not entry["location"].startswith("offset_coded_signatures/")
        for entry in json.loads(search)["items"]
    )
    assert not (site / "offset_coded_signatures").exists()


def test_docs_workflow_runs_the_validated_commands() -> None:
    workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    commands = (
        "uv sync --locked --only-group docs --no-install-project",
        "uv run --no-sync python scripts/generate_docs.py",
        "uv run --no-sync zensical build --clean --strict -f .generated-site.yml",
    )
    positions = [workflow.index(command) for command in commands]
    assert positions == sorted(positions)
    assert "pull_request:" in workflow
    assert "if: github.event_name == 'release'" in workflow
