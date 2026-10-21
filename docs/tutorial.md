# Tutorial: from zero to search

This is the shortest path from a fresh install to indexing photos, searching
them and classifying images. It is a **learning path**, not a second copy of
the reference: each step links to the page that documents it completely.

You need:

- Linux, macOS or Windows, Python 3.12+.
- ~2 GB free for the base model and the environment.
- A folder with a few images (the examples use openly licensed images).

## 1. Install

```bash
pip install "siglip2-toolkit[server]"
```

With uv:

```bash
uv add "siglip2-toolkit[server]"
# or, in a source checkout:
uv sync --all-extras
```

`[server]` adds FastAPI/uvicorn for the REST surface. The core already includes
torch/transformers; see [Installation](installation.md) for CPU-only torch and
source checkouts.

Verify:

```bash
siglip2 --version
siglip2 --help
```

## 2. Get a model

The toolkit has its own model cache; it never chooses a model silently.

```bash
siglip2 models download google/siglip2-base-patch16-naflex
siglip2 models use google/siglip2-base-patch16-naflex
siglip2 models current
```

The base checkpoint is about 1.5 GB. See [Models](models.md) for sizes,
devices, offline import and the rest of the family.

## 3. Embed text

```bash
siglip2 embed --text "a wedding at sunset" --text "a dog on the beach"
```

The same with the library:

```python
from siglip2_toolkit import Embedder, models

resolved = models.resolve_model("google/siglip2-base-patch16-naflex")
embedder = Embedder(
    model=resolved.path,
    model_id=resolved.identifier,
    device="auto",
)
vectors = embedder.embed_texts(["a wedding at sunset", "a dog on the beach"])
print(vectors.shape)  # (2, dimensions)
```

Embeddings are L2-normalised `float32` arrays. See the
[library guide](library.md#embedder) and [API reference](api/embedder.md).

## 4. Embed images and compare

The toolkit can compare text and images in the same space:

```bash
siglip2 embed --image examples/images/cat.jpg --image examples/images/coffee.jpg
```

Full runnable example: [`examples/07_text_image_similarity.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/07_text_image_similarity.py).

```python
import numpy as np
from pathlib import Path

from siglip2_toolkit import Embedder, models

resolved = models.resolve_model()
embedder = Embedder(model=resolved.path, model_id=resolved.identifier)

images = sorted(str(p) for p in Path("examples/images").glob("*.jpg"))
texts = ["a cat", "an astronaut", "a cup of coffee"]

image_vectors = embedder.embed_images(images)   # (n_images, dim)
text_vectors = embedder.embed_texts(texts)      # (n_texts, dim)
similarity = image_vectors @ text_vectors.T     # normalized → cosine
print(similarity.round(3))
```

## 5. Export a corpus

```bash
siglip2 export -i ./photos -o embeddings.jsonl
siglip2 inspect embeddings.jsonl
```

The export has a format header with the embedding-space manifest and one
record per image (`id`, `vector`, optional `metadata`). Incremental updates:

```bash
siglip2 export -i ./photos -o embeddings.jsonl --incremental
```

See [Export format](formats.md) and
[`examples/02_export_and_inspect.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/02_export_and_inspect.py).

## 6. Search the export

```python
from siglip2_toolkit import Embedder, formats, models
from siglip2_toolkit.retrieval import (
    apply_similarity_drop, rank_by_similarity, results_matrix,
)

records = formats.load_records("embeddings.jsonl")
matrix = results_matrix(records)

resolved = models.resolve_model()
embedder = Embedder(model=resolved.path, model_id=resolved.identifier)
query = embedder.embed_texts(["a cat on a sofa"])[0]

ranked = rank_by_similarity(query, matrix, top_k=20, metric="cosine")
for row, distance in apply_similarity_drop(ranked, max_drop_percent=15):
    metadata = records[row].get("metadata") or {}
    print(distance, metadata.get("path") or records[row].get("id"))
```

`rank_by_similarity` returns **distances** (lower is better). See
[Retrieval](api/retrieval.md), [Recipes](recipes.md) and
[`examples/03_search_export.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/03_search_export.py).

## 7. Classify

```bash
siglip2 classify --image examples/images/cat.jpg \
  --labels "cat,dog,astronaut,coffee" --top-k 2
```

With a threshold, the library can abstain instead of forcing a weak label:

```python
labels = ["cat", "dog", "astronaut", "coffee"]
results = embedder.classify(
    ["examples/images/cat.jpg"],
    labels,
    template="a photo of a {}",
    top_k=None,
)
best_label, best_probability = results[0][0]
print(best_label, best_probability, best_probability < 0.5)  # abstain below 0.5
```

Full example:
[`examples/08_classify_threshold.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/08_classify_threshold.py).
See [Classification](api/classification.md).

## 8. Serve the REST API

```bash
siglip2 serve --host 127.0.0.1 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/v1/embed/text \
  -H "Content-Type: application/json" \
  -d '{"texts": ["a wedding at sunset"]}'
```

- Interactive docs: `http://127.0.0.1:8000/docs`
- Image/classify uploads: `curl -F files=@examples/images/cat.jpg ...`
- The model loads lazily on the first request.

See [REST API](rest.md) and
[`examples/06_rest_client.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/06_rest_client.py).

## 9. Offline

Only `models download` uses the network. Copy the toolkit home or import a
local model:

```bash
siglip2 --home /data/toolkit models import /media/model \
  --name google/siglip2-base-patch16-naflex
siglip2 --home /data/toolkit models use google/siglip2-base-patch16-naflex
```

See [Models → Offline workflow](models.md#offline-workflow).

## 10. What next

- [Library guide](library.md) — the full Python API.
- [CLI](cli.md) and [REST](rest.md) — the other two surfaces.
- [Recipes](recipes.md) — search, fusion, filters and batch.
- [Examples](https://github.com/gia-uh/siglip2-toolkit/tree/main/examples) —
  runnable scripts.
- [Troubleshooting](troubleshooting.md) and [API reference](api/index.md).
