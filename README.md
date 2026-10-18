# siglip2-toolkit

Local-first toolkit around Google's **SigLIP 2** models: text/image embeddings, zero-shot and fine-tuned classification, incremental corpus export, and retrieval math.

It is **not a vector database**: it produces embeddings and portable exports, and stays agnostic about where you store or search your vectors.

Co-developed by [GIA-UH](https://github.com/gia-uh) and [Syalia S.R.L.](https://github.com/syalia-srl).

## Installation

```bash
pip install siglip2-toolkit             # library + CLI
pip install "siglip2-toolkit[server]"   # + REST server (FastAPI)
```

## Three ways to use it

**1. As a library (primary interface)**

```python
from siglip2_toolkit import Embedder

embedder = Embedder(model="google/siglip2-base-patch16-naflex", device="auto")
vectors = embedder.embed_texts(["a wedding at sunset"])
results = embedder.classify(["photo.jpg"], labels=["beach", "mountain", "wedding"])

from siglip2_toolkit.filters import field, matches

matches({"width": 2400}, field("width") >= 2000)  # -> True
```

**2. As a CLI**

```bash
siglip2 export -i /path/to/photos -o embeddings.jsonl --incremental
siglip2 classify --image photo.jpg --labels "beach,mountain,wedding"
```

**3. As a REST service (optional extra)**

```bash
pip install "siglip2-toolkit[server]"
siglip2 serve --host 0.0.0.0 --port 8000
```

Endpoints: `/v1/embed/text`, `/v1/embed/image`, `/v1/classify`, `/v1/health`, `/v1/info`.

## Model selection & caching

Models are always chosen explicitly (`--model` / `Embedder(model=...)`);
the toolkit never picks one for you. Recommended starting points:

| Model | When |
|---|---|
| `google/siglip2-base-patch16-naflex` | default: dynamic resolution, good all-rounder |
| `google/siglip2-so400m-patch16-naflex` | higher accuracy, GPU recommended |
| `google/siglip2-base-patch16-224` | fixed resolution, fastest on CPU |
| `google/siglip2-giant-opt-patch16-384` | best accuracy, needs a strong GPU |

Models are cached by Hugging Face in `~/.cache/huggingface/hub`
(`HF_HOME` / `HF_HUB_CACHE` override it), persist across reboots and are
reused offline. To guarantee no network access at all:

```bash
siglip2 models                                            # what is already cached
siglip2 models --check google/siglip2-base-patch16-naflex # exit code 0/1
siglip2 export -i photos -o embeddings.jsonl --offline    # cache-only; fails fast if missing
```

To pre-seed a machine without internet: download the model on a connected
machine and copy its `models--google--siglip2-...` folder from that
machine's `hub` directory into the target's cache.

Incremental exports refuse to run when the existing export was produced by a
different model: mixing embedding spaces would silently corrupt the vectors.

## Highlights

- **Library-first design**: the CLI and the REST server are thin adapters over the same public API.
- **Zero-shot classification**: prompt-based labels using the model's calibrated SigLIP logits, plus fine-tuned classification heads (`ImageClassifier`).
- **Incremental export**: new, modified and deleted files are detected (modification detection requires metadata).
- **Resumable runs**: interrupted exports continue where they left off.
- **Optional metadata**: rich file metadata (EXIF, dimensions, timestamps, thumbnails) — or ids and vectors only, your choice.
- **Versioned export format** (`jsonl` / `pkl`) with an embedding-space manifest.
- **Retrieval math**: top-k scoring, RRF fusion, similarity-drop filtering and serializable metadata filters, composable with any store.
- **Offline-first**: `--offline` runs cache-only (`local_files_only`), `siglip2 models` shows what is available locally; no telemetry, no external services.

> **Status:** early scaffold (October 2026). The public API and the export format are being stabilised before the first release.

## License

MIT — see [LICENSE](LICENSE).
