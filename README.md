# siglip2-toolkit

Local-first toolkit around Google's **SigLIP 2** models: text/image embeddings, zero-shot and fine-tuned classification, incremental corpus export, and retrieval math.

It is **not a vector database**: it produces embeddings and portable exports, and stays agnostic about where you store or search your vectors.

Co-developed by [GIA-UH](https://github.com/gia-uh) and [Syalia S.R.L.](https://github.com/syalia-srl).

[![PyPI](https://img.shields.io/pypi/v/siglip2-toolkit.svg)](https://pypi.org/project/siglip2-toolkit/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://pypi.org/project/siglip2-toolkit/)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://uhgia.org/siglip2-toolkit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Documentation:** <https://uhgia.org/siglip2-toolkit/>
(source in [`docs/`](https://github.com/gia-uh/siglip2-toolkit/tree/main/docs);
build it locally with the commands in [Development](#development)).

## Installation

```bash
pip install siglip2-toolkit             # library + CLI
pip install "siglip2-toolkit[server]"   # + REST server (FastAPI)
```

## Model management (the toolkit's own cache)

Models live in the toolkit's **own cache**, independent from HuggingFace's
cache. The only command that uses the network is `models download`;
everything else works with what you already have locally.

```bash
siglip2 models                       # your cache (★ model in use, origin, size)
siglip2 models --hf                  # SigLIP-2 catalog on HuggingFace: ✓ have it / ☁ downloadable
siglip2 models current               # the model in use (+ path, origin; --json)
siglip2 models use <id>              # set the model in use (persistent)
siglip2 models download <org/name> [--revision REV]   # into the toolkit cache (network)
siglip2 models import <path> [--name ALIAS]           # copy a local model into the cache
siglip2 models delete <id>           # delete from the cache (refuses the model in use)
```

Configuration lives in `<home>/config.json` (default home
`~/.local/share/siglip2-toolkit/`, created on first use):

```json
{
  "cache_dir": "~/.local/share/siglip2-toolkit/models",
  "current": "google/siglip2-base-patch16-naflex",
  "device": "auto"
}
```

- **`cache_dir`**: where the cache lives; change it to relocate it (e.g. another disk).
- **`current`**: the model in use.
- **`device`**: `auto` (CUDA → MPS → CPU), or `cuda`, `cpu`, `mps`, `cuda:1`, …
  Any command's `--device` overrides it for that run.

There are **no environment variables**; for non-default installations pass
`--home DIR` (global option, before the command).

`--model` accepts either an identifier from your cache or a path to a local
model directory. The implicit default is
`google/siglip2-base-patch16-naflex`: on a machine that does not have it yet,
the first command tells you the exact `models download` to run. The toolkit
never picks a model for you.

Incremental exports refuse to run when the existing export was produced by a
different model: mixing embedding spaces would silently corrupt the vectors.

## Examples and documentation

- [Documentation](https://uhgia.org/siglip2-toolkit/) — installation,
  quickstart, models, library guide, CLI, REST, export format, recipes,
  troubleshooting and the API reference.
- [`examples/`](https://github.com/gia-uh/siglip2-toolkit/tree/main/examples) — runnable scripts for embeddings, export/inspect,
  search, classification and the REST client.

## Three ways to use it

**1. As a library (primary interface)**

```python
from siglip2_toolkit import Embedder, models

# Resolve from your cache (download/import first) or pass a local path.
resolved = models.resolve_model("google/siglip2-base-patch16-naflex")
embedder = Embedder(model=resolved.path, model_id=resolved.identifier, device="auto")

vectors = embedder.embed_texts(["a wedding at sunset"])
results = embedder.classify(["photo.jpg"], labels=["beach", "mountain", "wedding"])

from siglip2_toolkit.filters import field, matches

matches({"width": 2400}, field("width") >= 2000)  # -> True
```

**2. As a CLI**

```bash
siglip2 models download google/siglip2-base-patch16-naflex
siglip2 models use google/siglip2-base-patch16-naflex
siglip2 export -i /path/to/photos -o embeddings.jsonl --incremental
siglip2 classify --image photo.jpg --labels "beach,mountain,wedding"
```

**3. As a REST service (optional extra)**

```bash
pip install "siglip2-toolkit[server]"
siglip2 serve --host 0.0.0.0 --port 8000
```

Endpoints: `/v1/embed/text`, `/v1/embed/image`, `/v1/classify`, `/v1/health`, `/v1/info`.

## Highlights

- **Library-first design**: the CLI and the REST server are thin adapters over the same public API.
- **Zero-shot classification**: prompt-based labels using the model's calibrated SigLIP logits, plus fine-tuned classification heads (`ImageClassifier`).
- **Own model cache**: downloads and imports live in the toolkit cache (independent from HuggingFace's); only `models download` uses the network.
- **GPU-ready**: `device: "auto"` picks CUDA when available (FP16; FP32 on CPU); override per command with `--device`.
- **Incremental export**: new, modified and deleted files are detected (modification detection requires metadata).
- **Resumable runs**: interrupted exports continue where they left off.
- **Optional metadata**: rich file metadata (EXIF, dimensions, timestamps, thumbnails) — or ids and vectors only, your choice.
- **Versioned export format** (`jsonl` / `pkl`) with an embedding-space manifest.
- **Retrieval math**: top-k scoring, RRF fusion, similarity-drop filtering and serializable metadata filters, composable with any store.

> **Status:** first preview (0.0.4, October 2026). The public API and the
export format may still change before 1.0; pin the version you depend on.

## Development

Managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras     # dev + server extras
uv run pytest
uv run siglip2 --help
uv lock                   # refresh uv.lock
uv build                  # wheel + sdist
```

CPU-only development (avoids the CUDA torch wheels):

```bash
uv venv --python 3.12
uv pip install --torch-backend cpu -e ".[dev,server]"
```

## License

MIT — see [LICENSE](LICENSE).
