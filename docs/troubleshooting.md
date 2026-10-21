# Troubleshooting

## “The model is not in the toolkit cache”

The toolkit never downloads a model implicitly. Run:

```bash
siglip2 models
siglip2 models download google/siglip2-base-patch16-naflex
siglip2 models use google/siglip2-base-patch16-naflex
```

If you already have a model directory, import it:

```bash
siglip2 models import /path/to/model --name google/siglip2-base-patch16-naflex
```

For a non-default toolkit home, put `--home DIR` **before** the subcommand.

## `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`

You are on **transformers 5.x**. transformers 5 changed
`SiglipModel.get_image_features`/`get_text_features` to return the tower
outputs instead of the projected embeddings. `siglip2-toolkit` pins
`transformers>=4.51,<5` from 0.0.3 onward.

```bash
pip install --upgrade "siglip2-toolkit>=0.0.3"
python -c "import transformers; print(transformers.__version__)"   # 4.x
```

## `Using use_fast=True but torchvision is not available`

The image processor falls back to the slow path. It still works. To use the
fast processor, install a matching `torchvision` (CPU or CUDA) for your torch
build, for example:

```bash
uv pip install --torch-backend cpu torchvision
```

## The model download is slow or blocked

Download on a connected machine, then copy the toolkit home or import the
model directory on the offline machine:

```bash
# connected
siglip2 models download google/siglip2-base-patch16-naflex

# offline
siglip2 --home /data/toolkit models import /media/model \
  --name google/siglip2-base-patch16-naflex
siglip2 --home /data/toolkit models use google/siglip2-base-patch16-naflex
```

Only `models download` needs the network. The library, CLI and server work
offline.

## `CUDA out of memory` or the process is killed

- Reduce `--batch-size` (for example 8 or 4).
- Use `--device cpu` to confirm it is a GPU-memory issue.
- On CUDA, the toolkit uses float16 by default; pass `dtype="float32"` to
  `Embedder` only if required (it uses more memory).
- Close other GPU processes (`nvidia-smi`).

## `CUDA` is not used even though a GPU exists

```bash
siglip2 models current
nvidia-smi
```

Check that the installed torch is a CUDA build:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

If `torch.cuda.is_available()` is `False`, install a CUDA torch build or use
`--device cpu`.

## Apple MPS

`device: auto` selects MPS when available. Some operations are unsupported on
MPS; if a model fails there, use `--device cpu`. MPS memory is shared with the
system, so lower the batch size on large images.

## Incremental export does not detect modified files

Modification detection uses the stored `modification_time`; it requires
metadata. If the export was created with `--no-metadata`, only new and
deleted files are detected. Recreate the export with metadata if you need
mtime-based incremental runs.

## “Cannot mix embedding spaces” / unexpected search results

Vectors from different models (or different preprocessing) are not
comparable. Check the export header:

```bash
siglip2 inspect embeddings.jsonl
```

The `embedding_space.model` and `dimensions` must match the embedder used for
the query. Re-export the corpus with the query model, or use a separate index
per model. The toolkit refuses incremental exports across different models.

## A corrupted or truncated export

`inspect` fails or reports a bad header. Exports are written to
`<output>.tmp` and moved into place only on success; an interrupted run
leaves the previous file intact. If a file was edited manually, restore it
from a backup or re-export.

## HEIF/HEIC images fail to open

`pillow-heif` is a dependency, but the plugin may need registration in custom
Pillow code:

```python
from siglip2_toolkit.utils import register_heif_support
register_heif_support()
```

The exporter and CLI register it automatically.

## The REST server says the `[server]` extra is missing

```bash
pip install "siglip2-toolkit[server]"
```

If you installed from source with uv:

```bash
uv sync --extra server
```

## A new release is not visible to `uv`

uv caches index metadata. Refresh it:

```bash
uv lock --refresh
# or
uv --no-cache lock
```

For an install:

```bash
uv pip install --refresh "siglip2-toolkit==0.0.3"
```

## No environment variables

There are deliberately no environment variables for configuration. Use
`--home DIR` / `home=` for a non-default toolkit home, `config.json` for the
cache/device, and explicit CLI arguments or Python parameters for everything
else.
