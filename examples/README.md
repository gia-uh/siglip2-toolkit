# Examples

Runnable examples for the public API. They use the three openly licensed
images in `images/`; no private photos.

## Before you start

```bash
pip install "siglip2-toolkit[server]"
siglip2 models download google/siglip2-base-patch16-naflex
siglip2 models use google/siglip2-base-patch16-naflex
```

If your toolkit home is not the default, add `--home /path/to/toolkit` to the
scripts (or pass `--model /path/to/model`).

From a source checkout:

```bash
uv sync --all-extras
uv run siglip2 models download google/siglip2-base-patch16-naflex
uv run siglip2 models use google/siglip2-base-patch16-naflex
```

## Scripts

| Script | What it shows |
|---|---|
| `01_embed_text.py` | Text embeddings and pairwise cosine similarity. |
| `02_export_and_inspect.py` | Export `images/` and inspect the format header/records. |
| `03_search_export.py` | Rank an export with a text query and a similarity drop. |
| `04_classify.py` | Zero-shot classification with candidate labels. |
| `05_models.py` | List the cache and the informational model catalog. |
| `06_rest_client.py` | Call a running `siglip2 serve` instance. |
| `07_text_image_similarity.py` | Compare text labels with images. |
| `08_classify_threshold.py` | Zero-shot classification with abstention. |

Run them from the repository root:

```bash
uv run python examples/01_embed_text.py
uv run python examples/02_export_and_inspect.py -o /tmp/examples.jsonl
uv run python examples/03_search_export.py --export /tmp/examples.jsonl --query "a cat"
uv run python examples/04_classify.py
uv run python examples/05_models.py --catalog
```

For the REST example:

```bash
# terminal 1
siglip2 serve --host 127.0.0.1 --port 8000
# terminal 2
uv run python examples/06_rest_client.py --text "a cat on a sofa"
```

## Image licenses

See [`images/SOURCES.md`](images/SOURCES.md). Every example/test image must be
openly licensed and attributed; never add personal photos.
