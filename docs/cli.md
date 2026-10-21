# CLI reference

The `siglip2` command is a thin adapter over the library. Global options must
come **before** the subcommand:

```bash
siglip2 [--home DIR] [--version] <command> [options]
```

| Command | Purpose |
|---|---|
| `embed` | Text or image embeddings. |
| `export` | Create or update an embeddings export from image folders. |
| `inspect` | Validate an export and print a summary. |
| `classify` | Zero-shot or fine-tuned image classification. |
| `models` | Manage the toolkit model cache. |
| `serve` | Start the optional REST server. |

## `siglip2 embed`

```bash
siglip2 embed --text "a wedding at sunset"
siglip2 embed --text "one" --text "two" -o vectors.json
siglip2 embed --image photo.jpg --image other.jpg -o vectors.pkl --format pickle
siglip2 embed --text "hello" --model /path/to/model --device cpu
```

| Option | Meaning |
|---|---|
| `--text TEXT` | Text to embed; repeat for several (mutually exclusive with `--image`). |
| `--image PATH` | Image file to embed; repeat for several. |
| `-o, --output FILE` | Write to a file instead of stdout. |
| `--format {json,pickle}` | Output format for `-o` (default `json`). |
| `--model ID` | Cache identifier or local model directory (default: model in use). |
| `--device DEVICE` | `auto`, `cpu`, `cuda`, `mps`, `cuda:1`… |
| `--batch-size N` | Inputs per batch (default 32). |

## `siglip2 export`

```bash
siglip2 export -i ./photos -o embeddings.jsonl
siglip2 export -i ./photos -i ./more -o corpus.pkl --format pkl
siglip2 export -i ./photos -o embeddings.jsonl --incremental
siglip2 export --config export.yaml
```

| Option | Meaning |
|---|---|
| `-c, --config FILE` | YAML config; CLI flags override it. |
| `-i, --input-dirs DIR` | Directory to scan; repeat for several. |
| `-o, --output-file FILE` | Output `.jsonl` or `.pkl`. |
| `--format {jsonl,pkl}` | Override the format inferred from the extension. |
| `--model ID` | Cache identifier or local model directory. |
| `--batch-size N` | Images per batch. |
| `--device DEVICE` | Device override. |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `--incremental` | Only new/modified files; drop deleted files. |
| `--source-file FILE` | Base export for incremental (default: output file). |
| `--no-metadata` | Store only ids and vectors; disables modification detection. |
| `--create-thumbnails` | Also generate WEBP thumbnails. |
| `--thumbnail-dir DIR` | Thumbnail directory. |
| `--thumbnail-size W H` | Thumbnail size in pixels. |

Example YAML:

```yaml
input_dirs:
  - ./photos
  - ./more
output_file: embeddings.jsonl
output_format: null        # inferred
model: google/siglip2-base-patch16-naflex
batch_size: 32
device: auto
log_level: INFO
incremental: false
source_file: null
store_metadata: true
create_thumbnails: false
thumbnail_dir: thumbnails
thumbnail_size: [128, 128]
```

## `siglip2 inspect`

```bash
siglip2 inspect embeddings.jsonl
```

Prints the format header, embedding space, number of records and a summary of
ids/metadata. It validates that the file is readable and that the header is
present. It does not load a model.

## `siglip2 classify`

```bash
# zero-shot
siglip2 classify --image cat.jpg --labels "cat,dog,coffee" --top-k 2

# prompt template
siglip2 classify --image cat.jpg --labels "cat,dog" --template "a photo of a {}"

# fine-tuned classification head
siglip2 classify --image cat.jpg --finetuned --model /path/to/model
```

| Option | Meaning |
|---|---|
| `--image PATH` | Image file; repeat for several (required). |
| `--labels L1,L2,...` | Zero-shot candidate labels (comma-separated). |
| `--template TEMPLATE` | Prompt template; `{}` is replaced by each label. |
| `--finetuned` | Use a fine-tuned classification head instead of `--labels`. |
| `--model ID` | Model identifier or local model path. |
| `--device DEVICE` | Device override. |
| `--batch-size N` | Images per batch (default 32). |
| `--top-k N` | Only report the best k labels per image. |

## `siglip2 models`

```bash
siglip2 models                  # list cache
siglip2 models --hf             # catalog on HuggingFace + what you have
siglip2 models current --json
siglip2 models use google/siglip2-base-patch16-naflex
siglip2 models download google/siglip2-base-patch16-naflex --revision b53b807d
siglip2 models import /mnt/models/siglip2-base --name google/siglip2-base-patch16-naflex
siglip2 models delete google/siglip2-large-patch16-256
```

`download` is the only networked command. `delete` refuses to delete the
model currently in use. See [Models](models.md).

## `siglip2 serve`

```bash
pip install "siglip2-toolkit[server]"
siglip2 serve --host 127.0.0.1 --port 8000
siglip2 serve --model google/siglip2-base-patch16-naflex --device cpu
```

| Option | Meaning |
|---|---|
| `--host HOST` | Bind address (default `0.0.0.0`). |
| `--port PORT` | Port (default 8000). |
| `--model ID` | Model identifier or local path. |
| `--device DEVICE` | Device override. |

OpenAPI docs are served at `/docs` and `/openapi.json`. See
[REST API](rest.md).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | Runtime/server failure (for example, a failed screenshot in dev tooling). |
| `2` | Invalid arguments, missing model/file, or validation error. |
