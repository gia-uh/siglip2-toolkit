# Recipes

Short, runnable patterns built on the public API. All examples assume a model
is in the toolkit cache (`siglip2 models use …`) and the package is installed.

## Semantic search over an export

```python
from siglip2_toolkit import Embedder, formats, models
from siglip2_toolkit.retrieval import (
    apply_similarity_drop, fuse_rrf, rank_by_similarity, results_matrix,
)

records = formats.load_records("embeddings.jsonl")
matrix = results_matrix(records)

resolved = models.resolve_model()
embedder = Embedder(model=resolved.path, model_id=resolved.identifier)
query = embedder.embed_texts(["a cat on a sofa"])[0]

ranked = rank_by_similarity(query, matrix, top_k=50, metric="cosine")
for row, distance in apply_similarity_drop(ranked, max_drop_percent=15):
    metadata = records[row].get("metadata") or {}
    print(f"{distance:.3f}  {metadata.get('path', records[row].get('id'))}")
```

## Fuse a text query and an image query

```python
text_query = embedder.embed_texts(["a sunny beach"])[0]
image_query = embedder.embed_images(["reference.jpg"])[0]

text_ranked = rank_by_similarity(text_query, matrix, top_k=50)
image_ranked = rank_by_similarity(image_query, matrix, top_k=50)

fused = fuse_rrf(
    [text_ranked, image_ranked],
    weights=[1.0, 0.7],
    k=60,
)
for payload, score in fused[:20]:
    print(score, payload.get("metadata", {}).get("path") if isinstance(payload, dict) else payload)
```

`rank_by_similarity` returns `(row_index, distance)` pairs, not payloads. Pass
the records as payloads if you want `fuse_rrf` to deduplicate by path:

```python
text_payloads = [(records[row], score) for row, score in text_ranked]
image_payloads = [(records[row], score) for row, score in image_ranked]
fused = fuse_rrf([text_payloads, image_payloads], weights=[1.0, 0.7])
```

## Filter by metadata before ranking

```python
from siglip2_toolkit.filters import field, matches

predicate = (field("width") >= 2000) & field("filename").contains("cat")
kept = [record for record in records if matches(record.get("metadata") or {}, predicate)]
```

Filters are JSON-serialisable (`filters.to_dict` / `from_dict`), so they can
be stored alongside a search profile.

## Incremental export from a cron job

```bash
siglip2 export \
  --config /etc/siglip2/photos.yaml \
  --incremental \
  --log-level INFO
```

`export.yaml`:

```yaml
input_dirs: [/srv/photos]
output_file: /srv/siglip2/photos.jsonl
model: google/siglip2-base-patch16-naflex
batch_size: 32
incremental: true
```

Unchanged files are skipped, modified files are re-embedded and deleted files
are removed. The export is replaced atomically at the end.

## Classify and keep only high-confidence labels

```python
from siglip2_toolkit import Embedder, models

resolved = models.resolve_model()
embedder = Embedder(model=resolved.path, model_id=resolved.identifier)
labels = ["cat", "dog", "coffee", "rocket"]

for path, scored in zip(paths, embedder.classify(paths, labels, top_k=3)):
    best = [(label, p) for label, p in scored if p >= 0.5]
    print(path, best)
```

## Use the REST server from another process

```bash
siglip2 serve --host 127.0.0.1 --port 8000 &
curl -s http://127.0.0.1:8000/v1/embed/text \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["a wedding at sunset"]}'
```

See [REST API](rest.md) for image/classify uploads and error handling.

## Custom vector store

The toolkit is storage-agnostic. Keep the vector and the record id in your
store; use `results_matrix()` and `rank_by_similarity()` only if you also
keep vectors in memory. For a persistent store, compute the query with
`Embedder.embed_texts` and pass it to the store's own similarity search; the
embedding-space manifest is the only contract you must preserve.

## Batch embeddings on CPU

Lower the batch size when memory is tight; on CPU, float32 is used
automatically:

```bash
siglip2 export -i ./photos -o out.jsonl --device cpu --batch-size 8
```

On CUDA, `device="auto"` uses float16; pass `dtype="float32"` to `Embedder`
if a downstream component requires it.
