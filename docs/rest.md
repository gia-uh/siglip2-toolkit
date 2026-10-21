# REST API

The optional REST server is a thin adapter over `Embedder`: the same model,
the same preprocessing and the same vector semantics as the library.

## Install and run

```bash
pip install "siglip2-toolkit[server]"
siglip2 serve --host 127.0.0.1 --port 8000
```

Options:

| Option | Default |
|---|---|
| `--host` | `0.0.0.0` |
| `--port` | `8000` |
| `--model` | model in use |
| `--device` | `config.json` device, else `auto` |

The model is loaded **lazily** on the first embedding request;
`/v1/health` and `/v1/info` never trigger a load.

Interactive docs: `http://127.0.0.1:8000/docs`  
OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

!!! warning "No authentication"
    The server has no authentication or CORS configuration. Bind it to
    `127.0.0.1`, or put it behind your own authenticated reverse proxy.

## `GET /v1/health`

```bash
curl http://127.0.0.1:8000/v1/health
```

```json
{"status": "ok", "model": "google/siglip2-base-patch16-naflex", "loaded": false, "device": null}
```

## `GET /v1/info`

```bash
curl http://127.0.0.1:8000/v1/info
```

```json
{"model": "google/siglip2-base-patch16-naflex", "dimensions": null, "embedding_space": null}
```

After the first request, `dimensions` and `embedding_space` are populated.

## `POST /v1/embed/text`

```bash
curl -s http://127.0.0.1:8000/v1/embed/text \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["a wedding at sunset", "a dog on the beach"]}'
```

Response:

```json
{
  "model": "google/siglip2-base-patch16-naflex",
  "dimensions": 768,
  "vectors": [[0.01, -0.02, "..."]]
}
```

- Empty/whitespace texts are dropped; at least one non-empty text is required
  (`400`).
- More than `max_batch_size` texts returns `413` (default 256).

## `POST /v1/embed/image`

Multipart upload with one or more `files` fields:

```bash
curl -s http://127.0.0.1:8000/v1/embed/image \
  -F files=@examples/images/cat.jpg \
  -F files=@examples/images/coffee.jpg
```

The response has the same `model`/`dimensions`/`vectors` shape. Non-image
content types are rejected with `400`; too many files returns `413`.

## `POST /v1/classify`

Multipart upload. `labels` accepts a JSON array or a comma-separated list;
`template` and `top_k` are optional:

```bash
curl -s http://127.0.0.1:8000/v1/classify \
  -F files=@examples/images/cat.jpg \
  -F labels='["cat","dog","coffee"]' \
  -F template='a photo of a {}' \
  -F top_k=2
```

```json
{
  "model": "google/siglip2-base-patch16-naflex",
  "labels": ["cat", "dog", "coffee"],
  "results": [[
    {"label": "cat", "score": 0.93},
    {"label": "coffee", "score": 0.04}
  ]]
}
```

An empty label list returns `400`; too many files returns `413`.

## Errors

| Status | When |
|---|---|
| `400` | No non-empty texts, no files, non-image upload, empty labels, invalid classify input. |
| `413` | More than `max_batch_size` texts or files (default 256). |
| `503` / startup error | The `[server]` extra is not installed or the model cannot be resolved. |

## Mount it in your own FastAPI app

```python
from fastapi import FastAPI
from siglip2_toolkit.server import create_app

api = FastAPI()
api.mount("/siglip2", create_app(model="google/siglip2-base-patch16-naflex"))
```

`create_app` accepts an existing `Embedder`, a `model`, `device`,
`max_batch_size` and `home`:

```python
app = create_app(
    model="google/siglip2-base-patch16-naflex",
    device="cpu",
    max_batch_size=64,
    home="/data/toolkit",
)
```

## Python client

The server is plain HTTP; use `httpx`, `requests` or `urllib`:

```python
import json, urllib.request

payload = json.dumps({"texts": ["a cat on a sofa"]}).encode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/embed/text",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    data = json.load(response)
print(data["dimensions"], len(data["vectors"][0]))
```

For images, send multipart form data; `examples/06_rest_client.py` shows a
minimal text client, and the [Quickstart](quickstart.md#6-serve-the-rest-api-optional)
has a runnable server setup.
