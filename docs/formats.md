# Export format

An export is a portable file with **one embedding-space header** and zero or
more **records**. Two containers are supported:

- **JSONL** (`.jsonl`): the first line is the header; each following line is
  one record.
- **Pickle** (`.pkl`): the header and records are pickled sequentially.

The current format name is `siglip2-export` and the version is `2`.

## Header

```json
{
  "format": "siglip2-export",
  "version": 2,
  "created_at": "2026-10-20T12:34:56.789012+00:00",
  "embedding_space": {
    "model": "google/siglip2-base-patch16-naflex",
    "dimensions": 768,
    "normalization": "l2",
    "revision": null,
    "preprocess": null
  }
}
```

| Field | Meaning |
|---|---|
| `format` | Always `siglip2-export`. |
| `version` | Format version (`2`). |
| `created_at` | UTC ISO-8601 timestamp. |
| `embedding_space` | Manifest identifying the vector space. Optional when unknown (legacy merges). |

### Embedding space

`EmbeddingSpace` is the identity of the vectors:

| Field | Meaning |
|---|---|
| `model` | Model identifier or local path recorded by the embedder. |
| `dimensions` | Vector dimension, when known. |
| `normalization` | `l2` for the normal toolkit output. |
| `revision` | Optional model revision. |
| `preprocess` | Optional preprocessing marker. |

Two vectors are only comparable when they come from the same model **and**
the same preprocessing. `EmbeddingSpace.compatible_with` rejects different
models or different known dimensions; unknown dimensions are tolerated.

!!! danger "Never mix embedding spaces"
    Searching a matrix with a query embedded by another model produces
    nonsense silently. Incremental exports refuse to run when the existing
    file was produced by a different model. Use the manifest to validate
    inputs in your own store.

## Record

```json
{
  "id": "/photos/cat.jpg",
  "vector": [0.0123, -0.0456, "..."],
  "metadata": {
    "path": "/photos/cat.jpg",
    "filename": "cat.jpg",
    "size_bytes": 18233,
    "creation_time": "2026-10-19T18:00:00+00:00",
    "modification_time": "2026-10-19T18:00:00+00:00",
    "width": 320,
    "height": 240,
    "exif": {},
    "thumbnail_path": "thumbnails/cat.webp"
  }
}
```

| Field | Meaning |
|---|---|
| `id` | Stable record identifier. Optional, but **required for incremental updates and deduplication**. The exporter uses the absolute file path. |
| `vector` | JSON array of floats (or a numpy array in pickle). |
| `metadata` | Optional JSON-serialisable dictionary. |

With `store_metadata=False`, records contain only `id` and `vector`.
The metadata schema is stable and part of the format: `path`, `filename`,
`size_bytes`, `creation_time`, `modification_time`, `width`, `height`,
`exif`, and optionally `thumbnail_path`.

Legacy records without a top-level `id` fall back to `metadata["path"]`
(`formats.record_id`).

## Python API

```python
from siglip2_toolkit import EmbeddingSpace, formats

# Read the header without loading vectors
header = formats.read_header("embeddings.jsonl")
space = formats.header_embedding_space(header)

# Stream records (the header is skipped)
for record in formats.iter_records("embeddings.jsonl"):
    ...

records = formats.load_records("embeddings.jsonl")
by_id = formats.load_by_id("embeddings.jsonl")

# Build records and write a new file
record = formats.make_record(
    [1.0, 0.0, 0.0],
    {"filename": "a.jpg"},
    id="/photos/a.jpg",
)
formats.write_export("copy.jsonl", [record], embedding_space=space)
```

| Function | Purpose |
|---|---|
| `make_header(embedding_space=None, created_at=None)` | Header object. |
| `make_record(vector, metadata=None, *, id=None)` | Record object. |
| `read_header(path)` | Header or `None`. |
| `iter_records(path)` | Iterator of records (streaming). |
| `load_records(path)` | List of records. |
| `load_by_id(path)` | Dict id → record. |
| `write_export(path, records, embedding_space=None, fmt=None)` | New/rewritten export. |
| `infer_format(path)` | `jsonl` or `pkl` from the extension. |
| `record_id(record)` | Top-level `id` or legacy `metadata["path"]`. |
| `header_embedding_space(header)` | Manifest or `None`. |

## Incremental updates

`Exporter` with `incremental=True` reads the existing output (or
`source_file`), keeps unchanged records, embeds new/modified files and drops
records whose files disappeared.

- Modification detection compares the recorded `modification_time` with the
  file's mtime; it requires metadata (`store_metadata=True`).
- Records without metadata can only be detected as new/deleted.
- The output is written to `<output>.tmp` and moved into place at the end, so
  an interrupted run does not corrupt the previous export.
- The exporter refuses to mix models: the existing header's embedding space
  must be compatible with the current model.

## Thumbnails

`create_thumbnails=True` writes WEBP thumbnails and adds
`metadata["thumbnail_path"]`. The export file itself contains only the path,
not the image bytes.
