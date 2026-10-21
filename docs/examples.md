# Examples

The repository ships small, runnable scripts for the three surfaces. They use
the openly licensed images in `examples/images/`; no private photos.

Before running them:

```bash
uv sync --all-extras
uv run siglip2 models download google/siglip2-base-patch16-naflex
uv run siglip2 models use google/siglip2-base-patch16-naflex
```

If the model is in a non-default cache, pass `--home /path/to/toolkit` to the
examples that accept it (or set `--model`).

| # | Script | What it shows |
|---|---|---|
| 01 | [`01_embed_text.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/01_embed_text.py) | Text embeddings and pairwise cosine similarity. |
| 02 | [`02_export_and_inspect.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/02_export_and_inspect.py) | Export a folder, read the header and inspect records. |
| 03 | [`03_search_export.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/03_search_export.py) | Rank an export with a text query and a similarity drop. |
| 04 | [`04_classify.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/04_classify.py) | Zero-shot classification with top-k labels. |
| 05 | [`05_models.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/05_models.py) | List the cache and the informational model catalog. |
| 06 | [`06_rest_client.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/06_rest_client.py) | Call a running `siglip2 serve` instance. |
| 07 | [`07_text_image_similarity.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/07_text_image_similarity.py) | Compare text labels with images in the same space. |
| 08 | [`08_classify_threshold.py`](https://github.com/gia-uh/siglip2-toolkit/blob/main/examples/08_classify_threshold.py) | Zero-shot classification with abstention. |

Run from the repository root:

```bash
uv run python examples/01_embed_text.py
uv run python examples/02_export_and_inspect.py -o /tmp/examples.jsonl
uv run python examples/03_search_export.py --export /tmp/examples.jsonl --query "a cat"
uv run python examples/04_classify.py
uv run python examples/05_models.py --catalog
uv run python examples/07_text_image_similarity.py
uv run python examples/08_classify_threshold.py --threshold 0.5
```

For the REST client, start the server in another terminal:

```bash
uv run siglip2 serve --host 127.0.0.1 --port 8000
uv run python examples/06_rest_client.py --text "a cat on a sofa"
```

The [tutorial](tutorial.md) builds the same flow step by step. For deeper
patterns, see [Recipes](recipes.md).

## Running in CI

The scripts are checked by `tests/test_examples.py`: every example must expose
`--help` and the tutorial must only link to existing files. The tests do not
download a model; the model-dependent integration suite remains opt-in.
