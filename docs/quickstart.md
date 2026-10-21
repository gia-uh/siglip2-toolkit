# Quickstart

This page takes you from a fresh install to embeddings, an export, a search
and a classification in a few commands.

## 1. Get a model

The toolkit uses its **own model cache**; it never picks a model silently.

```bash
siglip2 models download google/siglip2-base-patch16-naflex
siglip2 models use google/siglip2-base-patch16-naflex
siglip2 models current
```

The base checkpoint is about **1.5 GB**. If you already have a model
directory, use `siglip2 models import /path/to/model` instead. See
[Models](models.md) for CPU/GPU, offline and cache details.

## 2. Embed text

```bash
siglip2 embed --text "a wedding at sunset" --text "a dog on the beach"
```

JSON is printed to stdout. With `-o embeddings.json` the vectors are written
to a file (`--format pickle` is also accepted).

The same thing with the library:

```python
from siglip2_toolkit import Embedder, models

resolved = models.resolve_model("google/siglip2-base-patch16-naflex")
embedder = Embedder(
    model=resolved.path,
    model_id=resolved.identifier,
    device="auto",          # cuda, mps or cpu are chosen automatically
)
vectors = embedder.embed_texts(["a wedding at sunset", "a dog on the beach"])
print(vectors.shape)        # (2, dimensions), float32, L2-normalised
```

## 3. Export an image folder

```bash
siglip2 export -i ./photos -o embeddings.jsonl
siglip2 inspect embeddings.jsonl
```

The export contains a format header (with the embedding-space manifest) and
one record per image: `id`, `vector` and optional `metadata`. The file is the
single source of truth for downstream search.

Incremental run (only new/modified files; deleted files are dropped):

```bash
siglip2 export -i ./photos -o embeddings.jsonl --incremental
```

!!! note
    Incremental modification detection relies on stored metadata. With
    `--no-metadata`, only new and deleted files are detected.

## 4. Search an export

```python
from siglip2_toolkit import Embedder, formats, models
from siglip2_toolkit.retrieval import apply_similarity_drop, rank_by_similarity, results_matrix

records = formats.load_records("embeddings.jsonl")
matrix = results_matrix(records)

resolved = models.resolve_model("google/siglip2-base-patch16-naflex")
embedder = Embedder(model=resolved.path, model_id=resolved.identifier)
query = embedder.embed_texts(["a cat on a sofa"])[0]

ranked = rank_by_similarity(query, matrix, top_k=20, metric="cosine")
ranked = apply_similarity_drop(ranked, max_drop_percent=15)
for row, score in ranked:
    print(score, records[row].get("metadata", {}).get("path") or records[row].get("id"))
```

`rank_by_similarity` returns **distances** (lower is better) for cosine,
euclidean and manhattan metrics. Use `apply_similarity_drop` to trim the tail
relative to the best score. See [Retrieval](api/retrieval.md).

## 5. Classify

Zero-shot with candidate labels:

```bash
siglip2 classify --image examples/images/cat.jpg \
  --labels "cat,dog,astronaut,coffee,rocket" --top-k 2
```

The library equivalent is `embedder.classify(images, labels, top_k=...)`; for
a fine-tuned classification head use `ImageClassifier` with `--finetuned`
(CLI) or the class directly. See [Classification](api/classification.md).

## 6. Serve the REST API (optional)

```bash
pip install "siglip2-toolkit[server]"
siglip2 serve --host 127.0.0.1 --port 8000
```

- Interactive docs: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`
- Endpoints: `/v1/health`, `/v1/info`, `/v1/embed/text`, `/v1/embed/image`,
  `/v1/classify`.

The model is loaded lazily on the first embedding request. See
[REST API](rest.md) for `curl` examples and request/response shapes.

## 7. Runnable examples

The repository contains small scripts under [`examples/`](https://github.com/gia-uh/siglip2-toolkit/tree/main/examples)
that use the public API end to end: embeddings, export/inspect, search,
classification, model listing and a REST client.
