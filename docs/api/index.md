# API reference

The package is deliberately lazy: importing `siglip2_toolkit` does not import
torch, transformers or FastAPI. Heavy modules load on first use.

The most common entry points are re-exported at the top level:

```python
from siglip2_toolkit import (
    Embedder, Exporter, ExportConfig, ExportResult,
    EmbeddingSpace, ImageClassifier, Filter, field,
)
```

| Page | Contents |
|---|---|
| [Embedder](embedder.md) | Text/image embeddings, zero-shot logits and classification. |
| [Exporter](exporter.md) | `ExportConfig`, `Exporter`, `ExportResult`. |
| [Models](models.md) | Cache, catalog, download/import/resolve, device policy. |
| [Formats](formats.md) | Export format v2, `EmbeddingSpace`, read/write helpers. |
| [Retrieval](retrieval.md) | Similarity ranking, RRF fusion, similarity drop. |
| [Filters](filters.md) | Composable, serialisable metadata predicates. |
| [Classification](classification.md) | Fine-tuned `ImageClassifier`. |
| [Utilities](utils.md) | Image discovery, metadata, thumbnails, device helpers. |
