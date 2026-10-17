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

## Highlights

- **Library-first design**: the CLI and the REST server are thin adapters over the same public API.
- **Zero-shot classification**: prompt-based labels using the model's calibrated SigLIP logits, plus fine-tuned classification heads (`ImageClassifier`).
- **Incremental export**: new, modified and deleted files are detected (modification detection requires metadata).
- **Resumable runs**: interrupted exports continue where they left off.
- **Optional metadata**: rich file metadata (EXIF, dimensions, timestamps, thumbnails) — or ids and vectors only, your choice.
- **Versioned export format** (`jsonl` / `pkl`) with an embedding-space manifest.
- **Retrieval math**: top-k scoring, RRF fusion, similarity-drop filtering and serializable metadata filters, composable with any store.
- **100% local and offline**: no telemetry, no external services.

> **Status:** early scaffold (October 2026). The public API and the export format are being stabilised before the first release.

## License

MIT — see [LICENSE](LICENSE).
