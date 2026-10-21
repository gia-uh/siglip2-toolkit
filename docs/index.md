# siglip2-toolkit

**Local-first toolkit around Google's SigLIP 2 models**: text and image
embeddings, zero-shot and fine-tuned classification, incremental corpus
export, and retrieval math.

[Get started](installation.md){ .md-button .md-button--primary }
[Quickstart](quickstart.md){ .md-button }

[![PyPI](https://img.shields.io/pypi/v/siglip2-toolkit.svg)](https://pypi.org/project/siglip2-toolkit/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://pypi.org/project/siglip2-toolkit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/gia-uh/siglip2-toolkit/blob/main/LICENSE)

## What it is

- A **library** around SigLIP 2 that produces L2-normalised `float32`
  embeddings for text and images.
- A **CLI** (`siglip2`) for embedding, exporting a corpus, inspecting exports,
  classification and model-cache management.
- An optional **REST server** (FastAPI) with the same behaviour as the
  library.
- A set of **retrieval primitives**: similarity ranking, RRF fusion,
  similarity-drop filtering and serialisable metadata filters.
- A portable, versioned **export format** (JSONL or pickle) with an
  embedding-space manifest.

## What it is not

- It is **not a vector database**: it does not store or index vectors for you.
  It produces embeddings and portable exports; use any store you like.
- It does **not** download images or run a web UI.
- It only uses the network for `siglip2 models download` (and, for the
  optional server, nothing beyond serving local requests).

## Highlights

- **Own model cache**, independent from HuggingFace's cache.
- **Three interfaces** over one implementation: library, CLI, REST.
- **Incremental exports** with resumable runs.
- **Embedding-space guard**: exports from different models are never mixed.
- **CPU, CUDA and Apple MPS** support (`device: auto`).
- **Portable exports** with ids, vectors, metadata and an optional
  embedding-space manifest.
- **MIT licensed**, co-developed by GIA-UH and Syalia S.R.L.

## Install in one line

```bash
pip install "siglip2-toolkit[server]"
```

or with uv:

```bash
uv add "siglip2-toolkit[server]"
```

See [Installation](installation.md) for source checkouts, CPU-only torch and
offline installs.

## 30-second example

```bash
siglip2 models download google/siglip2-base-patch16-naflex
siglip2 models use google/siglip2-base-patch16-naflex
```

```python
from siglip2_toolkit import Embedder, models

resolved = models.resolve_model("google/siglip2-base-patch16-naflex")
embedder = Embedder(
    model=resolved.path,
    model_id=resolved.identifier,
    device="auto",
)
vectors = embedder.embed_texts([
    "a wedding at sunset",
    "a dog on the beach",
])
print(vectors.shape)  # (2, dimensions)
```

## Documentation map

| If you want to… | Read |
|---|---|
| Go from zero to search | [Tutorial](tutorial.md) |
| Run the examples | [Examples](examples.md) |
| Install it, including offline | [Installation](installation.md) |
| Embed, export, inspect and classify in minutes | [Quickstart](quickstart.md) |
| Manage models and the cache | [Models](models.md) |
| Use the Python API in depth | [Library guide](library.md) |
| Use the command line | [CLI](cli.md) |
| Call the REST service | [REST API](rest.md) |
| Understand the export file | [Export format](formats.md) |
| Copy a recipe (search, RRF, filters) | [Recipes](recipes.md) |
| Fix a problem | [Troubleshooting](troubleshooting.md) · [FAQ](faq.md) · [Glossary](glossary.md) |
| Contribute or build docs | [Development](development.md) |

## Status

This is the **0.0.x preview**. The public API and the export format may still
change before 1.0; pin the version you depend on (`siglip2-toolkit==0.0.3`).
See the [changelog](changelog.md).

## License

MIT — see [LICENSE](https://github.com/gia-uh/siglip2-toolkit/blob/main/LICENSE).
