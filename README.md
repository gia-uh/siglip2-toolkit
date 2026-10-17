# siglip2-toolkit

Local-first toolkit around Google's **SigLIP 2** vision-language models for image retrieval: text and image embeddings, incremental image indexing, and retrieval primitives.

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
```

**2. As a CLI**

```bash
siglip2 index -i /path/to/photos -o images.jsonl --reindex
```

**3. As a REST service (optional extra)**

```bash
pip install "siglip2-toolkit[server]"
siglip2 serve --host 0.0.0.0 --port 8000
```

## Highlights

- **Library-first design**: the CLI and the REST server are thin adapters over the same public API.
- **Incremental re-indexing**: new, modified and deleted files are detected.
- **Resumable indexing**: interrupted runs continue where they left off.
- **Rich metadata**: EXIF, dimensions, file times, optional thumbnails.
- **Versioned index format** (`jsonl` / `pkl`) with an embedding-space manifest.
- **Retrieval primitives**: top-k scoring, RRF fusion, similarity-drop filtering.
- **100% local and offline**: no telemetry, no external services.

> **Status:** early scaffold (October 2026). The public API and the index format are being stabilised before the first release.

## License

MIT — see [LICENSE](LICENSE).
