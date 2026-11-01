"""Verify release contents and package parity between the wheel and source archive."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def check_distributions(directory: Path, version: str, source: Path) -> None:
    """Reject excluded artifacts, missing modules, and inconsistent distributions."""
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one wheel and one source distribution")
    expected = {
        "tensorsig/" + path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*.py")
        if "experimental" not in path.relative_to(source).parts
    }
    if "tensorsig/__init__.py" not in expected:
        raise ValueError("maintained package source is missing")
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_files = {
            name: wheel.read(name)
            for name in wheel.namelist()
            if not name.endswith("/")
        }
    prefix = f"tensorsig-{version}/"
    source_files = {}
    with tarfile.open(sdists[0]) as sdist:
        for member in sdist:
            if member.isdir():
                continue
            if not member.isfile() or not member.name.startswith(prefix):
                raise ValueError(f"unexpected source archive member: {member.name}")
            stream = sdist.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable source archive member: {member.name}")
            source_files[member.name.removeprefix(prefix)] = stream.read()

    metadata_dir = f"tensorsig-{version}.dist-info"
    for files, metadata, license_path in (
        (wheel_files, f"{metadata_dir}/METADATA", f"{metadata_dir}/licenses/LICENSE"),
        (source_files, "PKG-INFO", "LICENSE"),
    ):
        for name in files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive path: {name}")
            if (
                "experimental" in path.parts
                or path.name.startswith(".")
                or ".sqlite" in path.name
                or any(
                    part in {"benchmarks", "site", "generated-docs"}
                    for part in path.parts
                )
                or (
                    name.startswith(("tensorsig/", "src/tensorsig/"))
                    and path.suffix != ".py"
                )
            ):
                raise ValueError(f"forbidden release artifact: {name}")
        package = {
            name.removeprefix("src/"): data
            for name, data in files.items()
            if name.startswith(("tensorsig/", "src/tensorsig/"))
        }
        if package != expected:
            raise ValueError("packaged modules differ from maintained source")
        if files.get(license_path) != (ROOT / "LICENSE").read_bytes():
            raise ValueError("release license is missing or incorrect")
        if metadata not in files:
            raise ValueError("package metadata is missing")
        headers = BytesParser().parsebytes(files[metadata])
        if headers["Name"] != "tensorsig" or headers["Version"] != version:
            raise ValueError("package metadata name/version mismatch")
    if source_files.get("README.md") != (ROOT / "README.md").read_bytes():
        raise ValueError("source distribution README differs from maintained README")
    print(f"Verified wheel and source distribution for tensorsig {version}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=ROOT / "dist")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    check_distributions(args.directory, version, ROOT / "src/tensorsig")


if __name__ == "__main__":
    main()
