#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--check] X.Y.Z" >&2
    exit 2
}

die() {
    echo "release: $*" >&2
    exit 1
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
root=$PWD
check_only=false
if [[ ${1:-} == "--check" ]]; then
    check_only=true
    shift
fi
[[ $# -eq 1 ]] || usage
version=${1#v}
[[ $version =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || usage
tag="v$version"

command -v git >/dev/null || die "git is required"
command -v uv >/dev/null || die "uv is required"
[[ -z $(git status --porcelain) ]] || die "commit or stash all changes first"
[[ $(uv version --short) == "$version" ]] ||
    die "pyproject.toml does not have version $version"
grep -Fxq "version: \"$version\"" CITATION.cff ||
    die "set CITATION.cff version to \"$version\""
# The date belongs to the release commit, so an older tag can be checked again.
grep -Eq '^date-released: "[0-9]{4}-[0-9]{2}-[0-9]{2}"$' CITATION.cff ||
    die "set date-released in CITATION.cff"

if [[ $check_only == false ]]; then
    [[ $(git branch --show-current) == main ]] || die "run from the main branch"
    command -v gh >/dev/null || die "GitHub CLI is required"
    git remote get-url origin >/dev/null 2>&1 || die "origin is not configured"
    gh auth status >/dev/null 2>&1 || die "authenticate with 'gh auth login'"
    remote_commit=$(gh api 'repos/{owner}/{repo}/commits/main' --jq .sha)
    [[ $(git rev-parse HEAD) == "$remote_commit" ]] ||
        die "local main must exactly match GitHub main"
    git show-ref --verify --quiet "refs/tags/$tag" &&
        die "tag $tag already exists"
    remote_tag=$(gh api "repos/{owner}/{repo}/git/matching-refs/tags/$tag" \
        --jq ".[] | select(.ref == \"refs/tags/$tag\") | .ref")
    [[ -z $remote_tag ]] || die "GitHub tag $tag already exists"
    gh release view "$tag" >/dev/null 2>&1 &&
        die "GitHub release $tag already exists"
fi

minimum_python=$(uv run --no-sync python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["requires-python"].removeprefix(">="))')
uv sync --locked --group dev --python "$minimum_python"
python=$(uv run --no-sync python -c 'import sys; print(sys.executable)')
uvx --python "$python" --from cffconvert==2.0.0 cffconvert --validate
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
uv run --no-sync python scripts/generate_docs.py
uv run --no-sync zensical build --clean --strict -f .generated-site.yml

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/tensorsig-release.XXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
uv build --no-sources --out-dir "$build_dir"
uv run --no-sync python scripts/check_distributions.py "$build_dir"
shopt -s nullglob
wheels=("$build_dir"/*.whl)
sdists=("$build_dir"/*.tar.gz)
# Running outside the checkout also prevents accidental imports from the source tree.
(
    cd "$build_dir"
    uv run --isolated --no-project --python "$python" --with "${wheels[0]}" "$root/tests/smoke_test.py"
    uv run --isolated --no-project --python "$python" --with "${sdists[0]}" "$root/tests/smoke_test.py"
)
[[ -z $(git status --porcelain) ]] || die "release checks changed the worktree"
mkdir -p dist
rm -f dist/*.whl dist/*.tar.gz dist/*.publish.attestation
cp -- "${wheels[0]}" "${sdists[0]}" dist/

if [[ $check_only == true ]]; then
    echo "Release checks passed for $tag; verified distributions are in dist/"
    exit 0
fi

commit=$(git rev-parse HEAD)
printf 'Publish GitHub release %s from %s? [y/N] ' "$tag" "$commit"
read -r answer
[[ $answer == y || $answer == Y ]] || die "cancelled"
gh release create "$tag" --target "$commit" --title "$tag" --generate-notes
echo "Published $tag; monitor the Release and Docs workflows with 'gh run list'."
