# tensorsig

`tensorsig` provides differentiable signature, log-signature, and offset-coded
signature features for PyTorch.

The first release is being prepared. The public API may change between `0.x`
releases.

## Installation

`tensorsig` requires Python 3.14 or newer and PyTorch 2.14 or newer.

Until the first PyPI release, install from a checkout of this repository:

```console
python -m pip install .
```

After publication, install with `python -m pip install tensorsig`.

## Quick start

```python
import torch

from tensorsig.classical.signatures import log_signature, signature

path = torch.tensor(
    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]],
    dtype=torch.float64,
)

first_turn = path[:, :3]
sig = signature(first_turn, depth=2)
logsig = log_signature(first_turn, depth=2)

assert sig.shape == (1, 6)
assert logsig.shape == (1, 3)
torch.testing.assert_close(
    sig,
    torch.tensor([[1.0, 1.0, 0.5, 1.0, 0.0, 0.5]], dtype=path.dtype),
)
torch.testing.assert_close(
    logsig,
    torch.tensor([[1.0, 1.0, 0.5]], dtype=path.dtype),
)
```

Start with the practitioner guides to
[signatures and logsignatures](https://froskekongen.github.io/tensorsig/signatures/).
The complete callable surface is in the
[API reference](https://froskekongen.github.io/tensorsig/api/).
The hosted documentation will become available with the first release. Until
then, use the local preview described under [Development](#development).

## Choosing local features

| Need | Start with |
| --- | --- |
| Classical logsignatures after path lifts | `tensorsig.augmentations` followed by `classical.windows.WindowedLogSignature` in `torch.nn.Sequential` |
| Signatures or logsignatures retaining local offset information | `offset_coded_signatures.plans.LocalRealization` with a `codes.OffsetCode` |
| Scalar polynomials in physical increments | `offset_coded_signatures.plans.compile_polynomial` |
| Apply a compiled local plan to a path | `offset_coded_signatures.layers.LocalFeatures` or `LocalLinear` |
| Explicit quotient window algorithms on increment streams | `offset_coded_signatures.windows.WindowLogSignature` |

The [offset-coded signatures guide](https://froskekongen.github.io/tensorsig/offset_coded_signatures/)
shows these constructions and their input conventions. Offset-coded windows
count increments; classical windows count points. Path augmentations are shared
across both families and run before window extraction.

## Citing
If you use this package in your research, please cite as follows:

```bibtex
@software{aune2026tensorsig,
  author    = {Aune, Erlend},
  title     = {tensorsig},
  version   = {0.1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23886748},
  url       = {https://doi.org/10.5281/zenodo.23886748}
}
```

```text
Aune, E. (2026). tensorsig (Version 0.1.0) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.23886748
```

See [CITATION.cff](https://github.com/Froskekongen/tensorsig/blob/main/CITATION.cff).

## Development

The project uses [uv](https://docs.astral.sh/uv/) and keeps its development
environment in the committed `uv.lock`. Python requirements are defined in
`pyproject.toml`.

```console
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Generate and strictly build the complete HTML documentation with:

```console
uv run python scripts/generate_docs.py
```

To build, start a local preview, and open it in your browser:

```console
uv run python scripts/generate_docs.py --serve
```

The HTML site is written to `site/`. Stop the preview with Ctrl+C. After editing
Python guides or docstrings, rerun the script to regenerate the documentation.

Validate a prepared release with `scripts/release.sh --check X.Y.Z`, then publish
it from clean `main` with `scripts/release.sh X.Y.Z`. Keep the package version and
`CITATION.cff` version aligned, and set `date-released` in `CITATION.cff` to the
actual release date before running the release checks. GitHub Actions publishes
to PyPI and deploys the documentation when the GitHub Release is published. The
PyPI Trusted Publisher and GitHub Pages Actions source must be configured before
the first release.

## License

`tensorsig` is released under the
[MIT License](https://github.com/Froskekongen/tensorsig/blob/main/LICENSE).
