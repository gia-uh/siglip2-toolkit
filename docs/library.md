# Library guide

This guide covers the public Python API. For every signature and docstring,
see the [API reference](api/index.md).

## Embedder

`Embedder` turns one local model directory into text and image embeddings.
It loads the model lazily on the first call, is safe to share across threads,
and is **not** safe to share across processes.

```python
from siglip2_toolkit import Embedder, models

resolved = models.resolve_model("google/siglip2-base-patch16-naflex")
embedder = Embedder(
    model=resolved.path,
    model_id=resolved.identifier,   # recorded in the manifest
    device="auto",                  # cuda → mps → cpu
)

texts = ["a cat on a sofa", "a dog on the beach"]
text_vectors = embedder.embed_texts(texts, batch_size=32, max_length=64)
print(text_vectors.shape, text_vectors.dtype)   # (2, dim), float32

image_vectors = embedder.embed_images(
    ["examples/images/cat.jpg", "examples/images/coffee.jpg"],
    batch_size=16,
)
```

- Embeddings are **L2-normalised float32 numpy arrays** (one row per input).
- `embed_images` accepts paths, bytes, file-like objects or PIL images.
  `on_error="skip"` skips unreadable inputs; with `return_indices=True` you
  also get the positions that were kept.
- `embedding_space()` returns an `EmbeddingSpace` manifest
  (`model`, `dimensions`, `normalization`, optional `revision`/`preprocess`).
- `dimensions` and `device` are `None` until the model is loaded.
- `load()` is idempotent; you normally do not need to call it.

### Zero-shot classification

```python
labels = ["cat", "dog", "astronaut", "coffee", "rocket"]
results = embedder.classify(
    ["examples/images/cat.jpg", "examples/images/coffee.jpg"],
    labels,
    template="a photo of a {}",   # "{}" is replaced by each label
    top_k=3,
)
for row in results:
    print(row)   # [("cat", 0.93), ...] sorted by probability
```

Use `zero_shot_logits` when you need raw scores instead of softmax
probabilities. The scores follow the SigLIP recipe: cosine similarity scaled
by the model's learned `logit_scale` plus `logit_bias`.

### Fine-tuned classification

`ImageClassifier` loads a fine-tuned image-classification checkpoint. Labels
come from `config.id2label`:

```python
from siglip2_toolkit import ImageClassifier

classifier = ImageClassifier("/path/to/finetuned-model", device="auto")
for image, labels in zip(
    ["examples/images/cat.jpg"],
    classifier.classify(["examples/images/cat.jpg"], top_k=3),
):
    print(image, labels)
```

`num_labels` overrides the head size when a checkpoint requires it.

## Exporter

`Exporter` scans image directories, embeds the images and writes a portable
export. The configuration is a dataclass:

```python
from siglip2_toolkit import ExportConfig, Exporter

config = ExportConfig(
    input_dirs=["./photos"],
    output_file="embeddings.jsonl",
    model="google/siglip2-base-patch16-naflex",  # cache id or local path
    device="auto",
    batch_size=32,
    incremental=True,
    store_metadata=True,
    create_thumbnails=False,
    thumbnail_dir="thumbnails",
    thumbnail_size=(128, 128),
)
result = Exporter(config).run()
print(result.processed, result.skipped, result.removed, result.total)
print(result.embedding_space)
```

| Field | Meaning |
|---|---|
| `input_dirs` | One or more directories to scan (recursive). |
| `output_file` | `.jsonl` or `.pkl`; the format is inferred. |
| `output_format` | Override `jsonl`/`pkl`. |
| `model` | Toolkit cache identifier or local model directory; `None` = current. |
| `home` | Non-default toolkit home. |
| `device` | `auto`, `cpu`, `cuda`, `mps`, `cuda:1`… |
| `batch_size` | Images per embedding batch. |
| `incremental` | Only new/modified files; drop deleted files. |
| `source_file` | Base export for an incremental run (defaults to `output_file`). |
| `store_metadata` | Store EXIF/dimensions/timestamps/filename. |
| `create_thumbnails` | Also write WEBP thumbnails. |
| `thumbnail_dir`, `thumbnail_size` | Thumbnail destination and size. |

Progress callbacks:

```python
def progress(done, total, path):
    print(f"{done}/{total} {path}")

def stage(name):
    print("stage:", name)   # scanning / embedding / done

Exporter(config, on_progress=progress, on_stage=stage).run()
```

With `incremental=True` the exporter keeps the ids and metadata of unchanged
files, embeds new/modified ones and removes deleted files. Modification
detection needs metadata (`store_metadata=True`); with metadata disabled only
new/deleted files are detected.

Failed runs write to `<output>.tmp` and are only moved into place when the
export succeeds, so interrupted runs do not leave a half-written output.

## Filters

Filters are composable, serialisable predicates over metadata dictionaries:

```python
from siglip2_toolkit.filters import field, matches, to_dict, from_dict

predicate = (field("width") >= 2000) & (field("filename").contains("cat"))
metadata = {"width": 2400, "filename": "cat_01.jpg"}
assert matches(metadata, predicate)

payload = to_dict(predicate)          # JSON-serialisable
same = from_dict(payload)
```

Operators: `==`, `!=`, `<`, `<=`, `>`, `>=`, `between(low, high)`, `in_`,
`contains(value, case_sensitive=False)`, `startswith(...)`, `exists()`.
Combine with `&`, `|`, `~`. See [Filters](api/filters.md).

## Retrieval

Storage-agnostic helpers that operate on numpy arrays and `(payload, score)`
lists:

```python
import numpy as np
from siglip2_toolkit.retrieval import (
    rank_by_similarity, results_matrix, fuse_rrf, apply_similarity_drop,
)
from siglip2_toolkit import formats

records = formats.load_records("embeddings.jsonl")
matrix = results_matrix(records)             # (n, dimensions)
query = embedder.embed_texts(["a cat"])[0]

ranked = rank_by_similarity(query, matrix, top_k=50, metric="cosine")
ranked = apply_similarity_drop(ranked, max_drop_percent=15)

# Fuse several ranked lists (for example text and image queries)
fused = fuse_rrf([ranked_text, ranked_image], weights=[1.0, 0.7], k=60)
```

- Metrics: `cosine` (1 − cosine similarity), `euclid`, `manhattan`.
  Scores are **distances** (lower is better).
- `fuse_rrf` accepts lists of `(payload, score)`; payloads can be ids, paths
  or dicts with an `id`/`path` key. Ties prefer the better original score.
- `apply_similarity_drop` keeps results within a percentage of the best score.
  Use `lower_is_better=False` for RRF/higher-is-better scores.

## Formats

```python
from siglip2_toolkit import EmbeddingSpace, formats

header = formats.read_header("embeddings.jsonl")
space = formats.header_embedding_space(header)
if space:
    print(space.model, space.dimensions, space.id)

for record in formats.iter_records("embeddings.jsonl"):
    print(record["id"], len(record["vector"]))

records = formats.load_records("embeddings.jsonl")
by_id = formats.load_by_id("embeddings.jsonl")

formats.write_export(
    "copy.jsonl",
    records,
    embedding_space=space,
)
```

`make_record(vector, metadata, id=...)` builds a record. See
[Export format](formats.md).

## Utilities

- `utils.find_image_files(dirs)` — recursive image discovery
  (JPEG/PNG/WebP/TIFF/HEIF, depending on Pillow support).
- `utils.extract_metadata(path)` — EXIF/dimensions/timestamps/size.
- `utils.create_thumbnail(...)` — WEBP thumbnails.
- `utils.register_heif_support()` — enable HEIF/HEIC if `pillow-heif` is
  installed (it is a dependency, but registration is explicit).
- `utils.resolve_torch_device(...)` — the device policy used internally.

## Logging

The library uses the standard `logging` module and never configures handlers
at import time. The CLI calls `setup_logging(level)`; libraries can configure
their own logger hierarchy.
