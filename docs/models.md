# Models

The toolkit has its **own model cache**, independent from HuggingFace's cache.
Only `siglip2 models download` uses the network; every other command works
with what is already local.

## The toolkit home

By default:

```
~/.local/share/siglip2-toolkit/
├── config.json
└── models/
    └── google--siglip2-base-patch16-naflex/
        ├── config.json
        ├── model.safetensors
        ├── preprocessor_config.json
        ├── tokenizer.json
        └── model.json            # toolkit metadata (origin, size…)
```

`config.json` contains visible defaults:

```json
{
  "cache_dir": "models",
  "current": "google/siglip2-base-patch16-naflex",
  "device": "auto"
}
```

- `cache_dir` may be absolute or relative to the toolkit home.
- `current` is the model used when you do not pass `--model`.
- `device` is `auto` (CUDA → MPS → CPU), `cpu`, `cuda`, `mps`, `cuda:1`…
  A command's `--device` overrides it for that run.

There are **no environment variables**. For a non-default location pass
`--home DIR` before the subcommand, or `home=` in the Python API.

## Commands

| Command | What it does |
|---|---|
| `siglip2 models` | List the models in the cache. |
| `siglip2 models --hf` | Show the SigLIP 2 catalog on HuggingFace and whether you have each model. Nothing is downloaded. |
| `siglip2 models current [--json]` | Show the model in use, its path and origin. |
| `siglip2 models use <id>` | Set the model in use. |
| `siglip2 models download <org/name> [--revision REV]` | Download from HuggingFace into the cache (network). |
| `siglip2 models import <path> [--name ID]` | Copy a local model directory into the cache. |
| `siglip2 models delete <id>` | Delete a cached model (refuses the model in use). |

Examples:

```bash
siglip2 models
siglip2 models --hf
siglip2 models current --json
siglip2 models use google/siglip2-base-patch16-naflex
siglip2 models download google/siglip2-large-patch16-256
siglip2 models import /mnt/models/siglip2-base --name google/siglip2-base-patch16-naflex
```

## Catalog and size

The catalog is informational; the toolkit never chooses a model for you.
Checkpoint sizes are the actual HuggingFace downloads (fp32 safetensors):

| Family | Approx. size |
|---|---|
| `google/siglip2-base-*` | ≈1.5 GB |
| `google/siglip2-large-*` | ≈3.5 GB |
| `google/siglip2-so400m-*` | ≈4.6 GB |
| `google/siglip2-giant-opt-*` | ≈7.5 GB |

Inside a family, patch size and resolution change speed and quality, not the
weight class. For a first install, `google/siglip2-base-patch16-naflex` is the
reference base model.

## Which model?

There is no universally best checkpoint. Choose by **task, hardware and
measured quality on your data**:

| Family | Approx. size | When to consider it |
|---|---:|---|
| `google/siglip2-base-*` | ≈1.5 GB | First install, CPU, most collections; the reference quality |
| `google/siglip2-large-*` | ≈3.5 GB | You measured a quality gain over base and have the RAM/time |
| `google/siglip2-so400m-*` | ≈4.6 GB | Quality-first workloads with more CPU/GPU budget |
| `google/siglip2-giant-opt-*` | ≈7.5 GB | Maximum capacity; not a default and expensive on CPU |

Within the base family:

- `patch16-naflex` (toolkit reference) supports flexible resolution; start here.
- `patch16-224/256/384/512` use fixed resolutions; higher resolution preserves
  detail but costs compute.
- `patch32-256` uses larger patches; it can be faster, but measure retrieval
  quality on your images.

Practical selection:

1. Start with base `naflex`; build a small labelled query set.
2. Measure top-k accuracy/recall and latency on **your** hardware.
3. Try one larger family only if the gain justifies its size and time.
4. Keep one model per library/export; never mix embedding spaces.

Model licenses are separate from the toolkit's MIT license; check each model
card on HuggingFace before redistribution. See
[Export format](formats.md#embedding-space) for compatibility.

## Choosing a device

```bash
siglip2 models use google/siglip2-base-patch16-naflex
# edit ~/.local/share/siglip2-toolkit/config.json if you want a persistent device
```

Or per command:

```bash
siglip2 embed --text "hello" --device cuda
siglip2 export -i ./photos -o out.jsonl --device cpu
```

The library uses the same policy:

```python
from siglip2_toolkit import Embedder, models

resolved = models.resolve_model()           # current model
device = models.default_device()             # config.json "device"
embedder = Embedder(model=resolved.path, model_id=resolved.identifier, device=device)
```

`device="auto"` picks CUDA when available, then Apple MPS, then CPU. On CPU
the toolkit uses float32; on CUDA it uses float16 by default. Override with
`dtype="float32"` on `Embedder` if you need to.

## Offline workflow

1. On a connected machine, download the model:
   ```bash
   siglip2 models download google/siglip2-base-patch16-naflex
   ```
2. Copy the whole toolkit home (or just `models/`) to the offline machine,
   or copy the model directory and import it:
   ```bash
   siglip2 --home /data/toolkit models import /media/model \
     --name google/siglip2-base-patch16-naflex
   siglip2 --home /data/toolkit models use google/siglip2-base-patch16-naflex
   ```
3. `siglip2 models current` should show the model and its local path.

The download command is the only one that requires the network. The library,
CLI and REST server otherwise work entirely offline.

## Importing an existing HuggingFace snapshot

If you already have a HuggingFace snapshot, import it into the toolkit cache
(the import copies the directory):

```bash
siglip2 models import \
  ~/.cache/huggingface/hub/models--google--siglip2-base-patch16-naflex/snapshots/<revision> \
  --name google/siglip2-base-patch16-naflex
```

The copy is deliberate: the toolkit cache is self-contained and can be moved
or backed up independently.

## Per-export model choice

An export records its **embedding space**. Incremental exports refuse to run
when the existing file was produced by a different model, and Clipper-style
consumers can use `EmbeddingSpace.compatible_with` to reject mixed vectors.
Always use the same model (and preprocessing) for vectors you intend to
compare. See [Export format](formats.md#embedding-space).
