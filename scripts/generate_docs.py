"""Build disposable documentation from README, docstrings, and Python guides.

The public API comes from literal ``__all__`` declarations and is resolved
statically with Griffe.  Guides are Jupytext percent scripts under ``docs/``.
Nothing imports :mod:`tensorsig` while the documentation tree is generated.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import griffe
import jupytext


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "src"
PACKAGE_NAME = "tensorsig"
GUIDE_SOURCE_DIR = ROOT / "docs"
BASE_CONFIG = ROOT / "zensical.yml"
# Zensical skips a docs root whose own name starts with a dot.
DEFAULT_OUTPUT = ROOT / "generated-docs"
DEFAULT_CONFIG = ROOT / ".generated-site.yml"

STATIC_TREES = {
    GUIDE_SOURCE_DIR / "javascripts": Path("javascripts"),
}


@dataclass(frozen=True)
class ApiSymbol:
    """One documented public symbol and all supported import paths."""

    name: str
    owner_module: str
    canonical_path: str
    public_paths: tuple[str, ...]
    kind: str

    @property
    def module(self) -> str:
        """Return the public source module that owns this symbol."""
        return self.owner_module


@dataclass(frozen=True)
class ApiSection:
    """One source-owned public module in the API tree."""

    module: str
    summary: str
    children: tuple[str, ...]
    symbols: tuple[ApiSymbol, ...]


@dataclass(frozen=True)
class ApiModel:
    """The complete statically discovered public API."""

    package: str
    sections: tuple[ApiSection, ...]


@dataclass(frozen=True)
class Page:
    """A generated Markdown page relative to the generated docs root."""

    path: Path
    title: str
    body: str


@dataclass(frozen=True)
class NavNode:
    """One hierarchical navigation node."""

    label: str
    path: Path
    children: tuple[NavNode, ...] = ()


@dataclass(frozen=True)
class PageModel:
    """Generated API pages and their navigation entries."""

    pages: tuple[Page, ...]
    api_nav: NavNode


def literal_exports(source: Path) -> tuple[str, ...] | None:
    """Return one literal ``__all__`` assignment, or ``None`` if absent."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    assignments: list[ast.Assign | ast.AnnAssign | ast.AugAssign] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            assignments.append(node)

    if not assignments:
        return None
    if len(assignments) != 1 or isinstance(assignments[0], ast.AugAssign):
        raise ValueError(f"{source}: __all__ must have one literal assignment")
    try:
        value = ast.literal_eval(assignments[0].value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source}: __all__ must be a literal list or tuple"
        ) from error
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(name, str) and name.isidentifier() for name in value
    ):
        raise ValueError(f"{source}: __all__ must contain only valid names")
    if len(value) != len(set(value)):
        raise ValueError(f"{source}: __all__ contains duplicate names")
    return tuple(value)


def _source_file(module: griffe.Module) -> Path | None:
    """Return the Python source file represented by a Griffe module."""
    filepath = module.filepath
    if filepath is None:
        return None
    path = Path(filepath)
    return path if path.suffix == ".py" else None


def discover_api_model(
    source_dir: Path = SOURCE_DIR,
    package_name: str = PACKAGE_NAME,
) -> ApiModel:
    """Discover the public API statically from source-owned ``__all__`` trees.

    Public modules recurse through their exported child modules. Symbols stay
    under the leaf facade that declares them, even when implementation aliases
    resolve to private modules.
    """
    package = griffe.load(
        package_name,
        search_paths=[source_dir],
        submodules=True,
        resolve_aliases=True,
        allow_inspection=False,
    )
    if not isinstance(package, griffe.Module):
        raise TypeError(f"{package_name} did not resolve to a package")

    symbols_by_module: dict[str, list[ApiSymbol]] = {}
    children_by_module: dict[str, list[str]] = {}
    module_summaries: dict[str, str] = {}
    visited: set[str] = set()
    module_order: list[str] = []
    final_targets: dict[str, str] = {}

    def visit(module: griffe.Module) -> None:
        if module.path in visited:
            return
        visited.add(module.path)
        module_order.append(module.path)
        source = _source_file(module)
        if source is None:
            raise ValueError(f"Public API module {module.path} has no Python source")
        exports = literal_exports(source)
        if exports is None:
            raise ValueError(
                f"{source}: public API modules must define literal __all__"
            )
        if not module.docstring or not module.docstring.value.strip():
            raise ValueError(f"{source}: public API module is missing a docstring")
        module_summaries[module.path] = module.docstring.value.strip()
        owned: list[ApiSymbol] = []
        children: list[str] = []
        for name in exports:
            try:
                member = module.all_members[name]
            except KeyError as error:
                raise ValueError(
                    f"{source}: {name!r} is in __all__ but is undefined"
                ) from error
            target = member.final_target if isinstance(member, griffe.Alias) else member
            if isinstance(target, griffe.Module):
                if not target.path.startswith(f"{package_name}."):
                    raise ValueError(
                        f"{source}: public module {target.path} is external"
                    )
                children.append(target.path)
                visit(target)
                continue
            public_path = f"{module.path}.{name}"
            if not target.docstring:
                raise ValueError(
                    f"{source}: public symbol {public_path} is missing a docstring"
                )
            previous = final_targets.get(member.canonical_path)
            if previous is not None:
                raise ValueError(
                    f"Duplicate public target {member.canonical_path}: "
                    f"{previous} and {public_path}"
                )
            final_targets[member.canonical_path] = public_path
            owned.append(
                ApiSymbol(
                    name=name,
                    owner_module=module.path,
                    canonical_path=member.canonical_path,
                    public_paths=(public_path,),
                    kind=str(target.kind.value),
                )
            )
        if children and owned:
            raise ValueError(
                f"{source}: public API modules cannot mix child modules and symbols"
            )
        symbols_by_module[module.path] = owned
        children_by_module[module.path] = children

    visit(package)

    sections = []
    for module_name in module_order:
        sections.append(
            ApiSection(
                module=module_name,
                summary=module_summaries.get(module_name, ""),
                children=tuple(children_by_module[module_name]),
                symbols=tuple(symbols_by_module[module_name]),
            )
        )
    if not any(section.symbols for section in sections):
        raise ValueError(
            f"No literal public exports found below {source_dir / package_name}"
        )
    return ApiModel(package=package_name, sections=tuple(sections))


def _module_dir(module: str, package: str) -> Path:
    if module == package:
        return Path("api")
    relative = module.removeprefix(f"{package}.")
    if relative == module:
        raise ValueError(f"{module!r} is outside package {package!r}")
    return Path("api", *(_slug(part) for part in relative.split(".")))


def _slug(name: str) -> str:
    """Use readable, stable URL segments for Python identifiers."""
    words = re.sub(r"(.)([A-Z][a-z]+)", r"\1-\2", name)
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", words)
    return words.replace("_", "-").lower()


def _symbol_filename(symbol: ApiSymbol, section: ApiSection) -> str:
    """Keep function URLs stable when a class name has the same slug."""
    slug = _slug(symbol.name)
    if symbol.kind == "class" and any(
        other.name != symbol.name and _slug(other.name) == slug
        for other in section.symbols
    ):
        slug += "-class"
    return f"{slug}.md"


def build_page_model(api: ApiModel) -> PageModel:
    """Build one stable page per symbol plus source-module landing pages."""
    pages: list[Page] = []
    sections = {section.module: section for section in api.sections}
    for section in api.sections:
        directory = _module_dir(section.module, api.package)
        symbol_links = []
        for symbol in section.symbols:
            filename = _symbol_filename(symbol, section)
            path = directory / filename
            preferred = symbol.public_paths[0]
            body = (
                f"# `{symbol.name}` {{ #{preferred} }}\n\n"
                f"```python\nfrom {section.module} import {symbol.name}\n```\n\n"
                f"::: {preferred}\n"
                "    options:\n"
                "      show_root_heading: false\n"
                "      show_root_toc_entry: false\n"
            )
            pages.append(Page(path=path, title=symbol.name, body=body))
            symbol_links.append(f"- [`{symbol.name}`]({filename})")

        overview = f"# {section.module}\n\n"
        overview += f"{section.summary}\n\n"
        if section.children:
            overview += "## Modules\n\n"
            for child in section.children:
                child_dir = _module_dir(child, api.package)
                relative = child_dir.relative_to(directory)
                overview += f"- [`{child}`]({relative.as_posix()}/index.md)\n"
            overview += "\n"
        if symbol_links:
            overview += "## Public API\n\n" + "\n".join(symbol_links) + "\n"
        pages.append(
            Page(path=directory / "index.md", title=section.module, body=overview)
        )

    def nav_node(module: str) -> NavNode:
        section = sections[module]
        directory = _module_dir(module, api.package)
        children = [nav_node(child) for child in section.children]
        children.extend(
            NavNode(symbol.name, directory / _symbol_filename(symbol, section))
            for symbol in section.symbols
        )
        label = "API reference" if module == api.package else module.rsplit(".", 1)[-1]
        return NavNode(label, directory / "index.md", tuple(children))

    return PageModel(pages=tuple(pages), api_nav=nav_node(api.package))


def render_guides(source_dir: Path = GUIDE_SOURCE_DIR) -> dict[Path, str]:
    """Read the hand-written home page and render executable Python guides."""
    rendered = {Path("index.md"): (ROOT / "README.md").read_text(encoding="utf-8")}
    for source in sorted(source_dir.rglob("*.py")):
        relative = source.relative_to(source_dir).with_suffix(".md")
        notebook = jupytext.read(source, fmt="py:percent")
        for cell in notebook.cells:
            if cell.cell_type == "markdown" and re.search(
                r"^```(?:py|python)(?:\s|$)", cell.source, re.MULTILINE
            ):
                raise ValueError(
                    f"{source}: Python examples must be executable code cells"
                )
        markdown = jupytext.writes(notebook, fmt="md")
        if relative in rendered:
            raise ValueError(f"Duplicate guide page: {relative}")
        rendered[relative] = markdown.rstrip() + "\n"
    return rendered


def _validate_generated_path(path: Path, expected_name: str) -> Path:
    """Resolve one narrowly named disposable output path."""
    if path.name != expected_name or path.is_symlink():
        raise ValueError(f"Refusing unsafe generated path: {path}")
    resolved = path.resolve()
    protected = {ROOT.resolve(), SOURCE_DIR.resolve(), GUIDE_SOURCE_DIR.resolve()}
    if resolved in protected or resolved == resolved.parent:
        raise ValueError(f"Refusing unsafe generated path: {path}")
    return resolved


def _guide_section(
    guides: dict[Path, str],
    overview: Path,
    section_label: str,
) -> list[str]:
    """Render one guide section, ordered by links from its overview."""
    if overview not in guides:
        return []
    linked = [
        overview.parent / target
        for target in re.findall(r"\]\(([^)#]+\.md)\)", guides[overview])
        if (overview.parent / target).parent == overview.parent
        and (overview.parent / target) in guides
    ]
    pages = [overview, *dict.fromkeys(linked)]
    pages.extend(
        path
        for path in sorted(guides)
        if path.parent == overview.parent and path not in pages
    )
    lines = [f"  - {section_label}:"]
    for path in pages:
        heading = re.search(r"^# (.+)$", guides[path], flags=re.MULTILINE)
        label = "Overview" if path == overview else heading.group(1)
        lines.append(f"      - {json.dumps(label)}: {json.dumps(path.as_posix())}")
    return lines


def _navigation(guides: dict[Path, str], model: PageModel) -> str:
    """Render deterministic site navigation without another schema layer."""
    lines = ["nav:", "  - Home: index.md"]
    lines.extend(_guide_section(guides, Path("signatures/index.md"), "Signatures"))
    lines.extend(
        _guide_section(
            guides,
            Path("offset_coded_signatures/index.md"),
            "Offset-coded signatures",
        )
    )
    development_pages = [path for path in guides if path.parts[:1] == ("development",)]
    if development_pages:
        lines.append("  - Development:")
        for path in development_pages:
            heading = re.search(r"^# (.+)$", guides[path], flags=re.MULTILINE)
            label = heading.group(1) if heading else path.stem.replace("_", " ").title()
            lines.append(f"      - {json.dumps(label)}: {json.dumps(path.as_posix())}")

    def append_node(node: NavNode, indent: int) -> None:
        prefix = " " * indent
        if not node.children:
            lines.append(
                f"{prefix}- {json.dumps(node.label)}: {json.dumps(node.path.as_posix())}"
            )
            return
        lines.append(f"{prefix}- {json.dumps(node.label)}:")
        lines.append(f"{prefix}    - Overview: {json.dumps(node.path.as_posix())}")
        for child in node.children:
            append_node(child, indent + 4)

    append_node(model.api_nav, 2)
    return "\n".join(lines) + "\n"


def _site_config(
    output_dir: Path,
    config_path: Path,
    guides: dict[Path, str],
    model: PageModel,
) -> str:
    """Combine the human-maintained site settings with generated paths/nav."""
    config = BASE_CONFIG.read_text(encoding="utf-8")
    relative_docs = Path(os.path.relpath(output_dir, start=config_path.parent))
    replacement = f"docs_dir: {json.dumps(relative_docs.as_posix())}"
    lines = [
        replacement if line.startswith("docs_dir:") else line
        for line in config.splitlines()
    ]
    if not any(line.startswith("docs_dir:") for line in config.splitlines()):
        lines.insert(0, replacement)
    if any(line.startswith("nav:") for line in lines):
        raise ValueError(
            f"{BASE_CONFIG}: nav is generated and must not be declared here"
        )
    return "\n".join(lines).rstrip() + "\n\n" + _navigation(guides, model)


def write_generated_tree(
    model: PageModel,
    guides: dict[Path, str],
    output_dir: Path = DEFAULT_OUTPUT,
    config_path: Path = DEFAULT_CONFIG,
) -> None:
    """Atomically replace the disposable docs tree and generated site config."""
    output_dir = _validate_generated_path(output_dir, "generated-docs")
    config_path = _validate_generated_path(config_path, ".generated-site.yml")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".generated-docs-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "generated-docs"
        staging.mkdir()
        contents = dict(guides)
        for page in model.pages:
            if page.path in contents:
                raise ValueError(f"Generated page collision: {page.path}")
            contents[page.path] = page.body
        for relative, body in contents.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        for source, relative in STATIC_TREES.items():
            if source.is_dir():
                shutil.copytree(source, staging / relative)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)

    config = _site_config(output_dir, config_path, guides, model)
    temporary_config = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary_config.write_text(config, encoding="utf-8")
    temporary_config.replace(config_path)


def main() -> None:
    """Generate and strictly build the site, optionally opening a local preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--serve", action="store_true", help="Serve the built docs and open a browser"
    )
    args = parser.parse_args()

    api = discover_api_model()
    model = build_page_model(api)
    guides = render_guides()
    write_generated_tree(model, guides, args.output, args.config)
    print(f"Generated {len(model.pages)} API pages in {args.output}", flush=True)
    zensical = [sys.executable, "-m", "zensical"]
    config = str(args.config.resolve())
    subprocess.run(
        [*zensical, "build", "--clean", "--strict", "-f", config],
        cwd=ROOT,
        check=True,
    )
    if args.serve:
        subprocess.run(
            [*zensical, "serve", "--open", "-f", config], cwd=ROOT, check=True
        )


if __name__ == "__main__":
    main()
