"""Release gates reject invalid artifacts and stop before any publication."""

import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

import pytest

from scripts import check_distributions as distributions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def archives(tmp_path, monkeypatch):
    monkeypatch.setattr(distributions, "ROOT", tmp_path)
    (tmp_path / "LICENSE").write_bytes(b"license")
    (tmp_path / "README.md").write_bytes(b"readme")
    source = tmp_path / "src/tensorsig"
    source.mkdir(parents=True)
    (source / "__init__.py").write_bytes(b'"""Maintained package."""\n')
    package = {"tensorsig/__init__.py": (source / "__init__.py").read_bytes()}
    metadata = b"Name: tensorsig\nVersion: 0.1.0\n"
    wheel = {
        **package,
        "tensorsig-0.1.0.dist-info/METADATA": metadata,
        "tensorsig-0.1.0.dist-info/licenses/LICENSE": b"license",
    }
    sdist = {
        **{"src/" + name: data for name, data in package.items()},
        "PKG-INFO": metadata,
        "LICENSE": b"license",
        "README.md": b"readme",
    }
    directory = tmp_path / "dist"
    directory.mkdir()

    def write():
        with zipfile.ZipFile(
            directory / "tensorsig-0.1.0-py3-none-any.whl", "w"
        ) as stream:
            for name, data in wheel.items():
                stream.writestr(name, data)
        with tarfile.open(directory / "tensorsig-0.1.0.tar.gz", "w:gz") as stream:
            for name, data in sdist.items():
                info = tarfile.TarInfo("tensorsig-0.1.0/" + name)
                info.size = len(data)
                stream.addfile(info, io.BytesIO(data))

    return directory, source, wheel, sdist, write


def test_distribution_contents_and_source_parity(archives):
    directory, source, wheel, sdist, write = archives
    write()
    distributions.check_distributions(directory, "0.1.0", source)
    sdist["src/tensorsig/__init__.py"] = b"different implementation"
    write()
    with pytest.raises(ValueError, match="differ from maintained source"):
        distributions.check_distributions(directory, "0.1.0", source)


@pytest.mark.parametrize(
    "name",
    (
        "tensorsig/experimental/antialiased/__init__.py",
        "tensorsig/README.research.sqlite",
        "tensorsig/.DS_Store",
        "tensorsig/rolling/README.md",
        "../secret.py",
    ),
)
def test_distribution_rejects_research_and_unsafe_paths(archives, name):
    directory, source, wheel, sdist, write = archives
    wheel[name] = b"must not ship"
    write()
    with pytest.raises(
        ValueError, match="forbidden release artifact|unsafe archive path"
    ):
        distributions.check_distributions(directory, "0.1.0", source)


def test_distribution_rejects_missing_modules_and_wrong_version(archives):
    directory, source, wheel, sdist, write = archives
    del wheel["tensorsig/__init__.py"]
    write()
    with pytest.raises(ValueError, match="differ from maintained source"):
        distributions.check_distributions(directory, "0.1.0", source)
    wheel["tensorsig/__init__.py"] = sdist["src/tensorsig/__init__.py"]
    wheel["tensorsig-0.1.0.dist-info/METADATA"] = b"Name: tensorsig\nVersion: 9.9.9\n"
    write()
    with pytest.raises(ValueError, match="metadata name/version mismatch"):
        distributions.check_distributions(directory, "0.1.0", source)


@pytest.fixture
def release_checkout(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts/release.sh", scripts / "release.sh")
    (tmp_path / "CITATION.cff").write_text(
        'version: "0.1.0"\ndate-released: "2026-01-01"\n'
    )
    (tmp_path / ".gitignore").write_text("/bin/\n/calls.log\n")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "--initial-branch=main")
    git(
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.invalid",
        "add",
        ".",
    )
    git(
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    git("remote", "add", "origin", "https://github.com/example/tensorsig.git")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {
        "uv": """if [[ $1 == version ]]; then echo 0.1.0; exit 0; fi
 if [[ $1 == run ]]; then echo 3.14; exit 0; fi
 echo "injected check failure" >&2
 exit 42""",
        "gh": """case "$1 $2" in
 "auth status") exit 0;;
 "api repos/"*)
 if [[ $2 == *"/git/matching-refs/"* ]]; then
     if [[ ${RELEASE_TEST_TAG_EXISTS:-0} == 1 ]]; then echo refs/tags/v0.1.0; fi
 else echo "$RELEASE_TEST_COMMIT"; fi
 exit 0;;
 "release view") exit "${RELEASE_TEST_EXISTS:-1}";;
 *) echo "unexpected GitHub action" >&2; exit 99;;
 esac""",
    }.items():
        path = bin_dir / name
        path.write_text(
            '#!/usr/bin/env bash\nprintf "%s %s\\n" "${0##*/}" "$*" >> "$RELEASE_TEST_LOG"\n'
            + body
            + "\n"
        )
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "RELEASE_TEST_LOG": str(tmp_path / "calls.log"),
        "RELEASE_TEST_COMMIT": git("rev-parse", "HEAD"),
    }

    def run(*args):
        return subprocess.run(
            ["bash", str(scripts / "release.sh"), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )

    return tmp_path, git, env, run


def test_check_accepts_detached_head_and_propagates_failed_checks(release_checkout):
    root, git, env, run = release_checkout
    git("checkout", "--detach")
    result = run("--check", "0.1.0")
    assert result.returncode == 42
    assert (
        "uv sync --locked --group dev --python 3.14" in (root / "calls.log").read_text()
    )
    assert "gh " not in (root / "calls.log").read_text()


def test_release_rejects_dirty_tree_and_version_mismatch(release_checkout):
    root, git, env, run = release_checkout
    (root / "untracked.py").write_text("dirty")
    assert "commit or stash" in run("--check", "0.1.0").stderr
    (root / "untracked.py").unlink()
    result = run("--check", "0.2.0")
    assert result.returncode != 0
    assert "does not have version 0.2.0" in result.stderr


def test_release_rejects_duplicate_github_release(release_checkout):
    root, git, env, run = release_checkout
    env["RELEASE_TEST_EXISTS"] = "0"
    result = run("0.1.0")
    assert result.returncode != 0
    assert "GitHub release v0.1.0 already exists" in result.stderr
    assert "uv sync" not in (root / "calls.log").read_text()
    assert "gh release create" not in (root / "calls.log").read_text()


def test_release_rejects_remote_tag_without_a_release(release_checkout):
    root, git, env, run = release_checkout
    env["RELEASE_TEST_TAG_EXISTS"] = "1"
    result = run("0.1.0")
    assert result.returncode != 0
    assert "GitHub tag v0.1.0 already exists" in result.stderr
    assert "gh release create" not in (root / "calls.log").read_text()
