# Installation

## Requirements

- **Python 3.12+** (the package is tested with 3.12 and 3.13).
- Enough disk for the model cache: a base SigLIP 2 checkpoint is about
  **1.5 GB**; large is about 3.5 GB and so400m about 4.6 GB.
- Linux, macOS or Windows. CUDA and Apple MPS are optional; CPU works
  everywhere.

The only mandatory dependencies are installed automatically: `torch`,
`transformers`, `huggingface_hub`, `numpy`, `Pillow`, `pillow-heif`, `PyYAML`
and `tqdm`.

!!! warning "transformers is capped below 5"
    `siglip2-toolkit` pins `transformers>=4.51,<5`. transformers 5 changed
    `SiglipModel.get_image_features`/`get_text_features` to return the tower
    outputs instead of the projected embeddings, which breaks the vector
    space. The cap is intentional; see the
    [changelog](changelog.md#003-2026-10-20).

## From PyPI

```bash
pip install siglip2-toolkit             # library + CLI
pip install "siglip2-toolkit[server]"   # + REST server (FastAPI/uvicorn)
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add siglip2-toolkit
uv add "siglip2-toolkit[server]"
# one-off CLI, without adding it to a project:
uvx --from siglip2-toolkit siglip2 --help
```

Verify:

```bash
siglip2 --version
siglip2 --help
```

## From a source checkout

```bash
git clone https://github.com/gia-uh/siglip2-toolkit
cd siglip2-toolkit
uv sync --all-extras
uv run siglip2 --help
```

`uv sync` creates the project `.venv` from `uv.lock` and installs the
development and server extras. The checkpoints are **not** downloaded by
`uv sync`.

### CPU-only development

The default torch wheel from PyPI can include CUDA libraries on Linux. If you
only need CPU, use the CPU wheels:

```bash
uv venv --python 3.12
uv pip install --torch-backend cpu -e ".[dev,server]"
```

### Editable install without uv

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,server]"
```

## Verify the install

```bash
siglip2 --version
siglip2 models --help
```

Then download a model (the only networked command):

```bash
siglip2 models download google/siglip2-base-patch16-naflex
siglip2 models use google/siglip2-base-patch16-naflex
siglip2 models current
```

Continue with the [Quickstart](quickstart.md).

## Offline and air-gapped machines

The library and CLI work offline once a model is in the toolkit cache. The
only step that needs the network is `models download`.

On a machine **with** network access:

```bash
siglip2 models download google/siglip2-base-patch16-naflex
# copy the toolkit home (default ~/.local/share/siglip2-toolkit)
```

On the offline machine, copy that directory or import a model from a local
folder:

```bash
siglip2 --home /path/to/toolkit models import /path/to/model-directory \
  --name google/siglip2-base-patch16-naflex
siglip2 --home /path/to/toolkit models use google/siglip2-base-patch16-naflex
```

For Python packages, build a wheelhouse on a connected machine and install
with `pip install --no-index --find-links wheelhouse "siglip2-toolkit[server]"`.
See [Models](models.md#offline-workflow) for details.

## Where things live

| Thing | Default |
|---|---|
| Toolkit home | `~/.local/share/siglip2-toolkit/` |
| Model cache | `<home>/models/` |
| Configuration | `<home>/config.json` |
| HF model names | `google/siglip2-*` (HuggingFace) |

There are **no environment variables** for configuration. Pass `--home DIR`
(before the subcommand) or `home=` in the Python API for non-default
installations.

## Next steps

- [Quickstart](quickstart.md) — embed, export, inspect, classify.
- [Models](models.md) — cache, download/import, device selection.
- [Library guide](library.md) — the full Python API.
