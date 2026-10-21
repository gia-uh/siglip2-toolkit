# FAQ

## Is this a search engine or a vector database?

No. It produces embeddings, exports and retrieval helpers. You decide where to
store vectors and how to serve search. [Clipper](https://github.com/syalia-srl/clipper)
is an example of a product built on top of it.

## Does it run offline?

Yes. Only `siglip2 models download` uses the network. After the model is in the
cache, the library, CLI and REST work offline. See
[Models → Offline workflow](models.md#offline-workflow).

## Do I need a GPU?

No. It runs on CPU with torch/transformers. CUDA and Apple MPS are optional
(`device: auto`); CPU is the reference. Larger models need more RAM and time.

## Which model should I start with?

`google/siglip2-base-patch16-naflex` (base, ~1.5 GB) is the reference for a
first install. Large/so400m/giant improve capacity but cost memory and time;
measure on your data before switching. See
[Models → Which model?](models.md#which-model).

## Can I mix models inside a library?

No. Vectors are only comparable within the same model and preprocessing.
Exports carry an embedding-space manifest; incremental exports refuse to mix
models. Use one library/index per model. See
[Export format](formats.md#embedding-space).

## Why are embeddings L2-normalised?

The toolkit returns L2-normalised `float32` vectors, so cosine similarity is a
dot product and euclidean/cosine rankings agree up to a monotone transform. The
retrieval helpers document their score semantics; `rank_by_similarity` returns
**distances** (lower is better).

## Does it store my photos?

No. The exporter scans files and stores ids/vectors/metadata; thumbnails are
optional. The originals stay where they are. The library database is the
consumer's choice.

## What is in an export?

A header with the embedding-space manifest and one record per image (`id`,
`vector`, optional `metadata`). JSONL and pickle are supported. See
[Export format](formats.md).

## Why are thumbnails missing after importing an export?

Exports do not include image bytes or thumbnails. The toolkit regenerates
missing thumbnails on demand from the original files. If the originals are not
available, thumbnails cannot be generated.

## Does incremental export detect modified files?

Yes, if metadata is stored: it compares modification times. With
`--no-metadata`, only new/deleted files are detected. See
[Export format → Incremental updates](formats.md#incremental-updates).

## What is the difference between zero-shot classification and fine-tuned classification?

- `Embedder.classify` / `siglip2 classify` use a SigLIP 2 model with candidate
  labels (zero-shot; no task training).
- `ImageClassifier` loads a fine-tuned image-classification checkpoint with its
  own labels.

See [Classification](api/classification.md).

## Can I use it from a web app?

Yes: install the `[server]` extra and call the REST API, or use the Python
library in your backend. The toolkit does not ship a web UI.

## Is the REST server authenticated?

No. It is a local-first, development/serving adapter. Bind it to localhost or
put it behind your own authenticated proxy/gateway if it is exposed. See
[REST API](rest.md).

## Why does a model download say it is large?

Base checkpoints are ~1.5 GB, large ~3.5 GB, so400m ~4.6 GB and giant ~7.5 GB.
The catalog shows the actual sizes; `models download` is the only network path.

## Can I use a local/HuggingFace model directory?

Yes: `siglip2 models import /path --name my-model`, then `models use my-model`.
`Embedder(model=path)` also accepts a local directory directly.

## What about HEIF/HEIC?

`pillow-heif` is a dependency. The CLI/exporter register support automatically;
custom Pillow code can call `utils.register_heif_support()`.

## Why do I see a `torchvision` warning?

The fast image processor is unavailable; the toolkit falls back to the slow
processor. It still works. Install a matching `torchvision` if you want the
fast path.

## `AttributeError: BaseModelOutputWithPooling ...`?

You are on transformers 5.x. The toolkit pins `transformers<5` from 0.0.3; see
[Troubleshooting](troubleshooting.md#attributeerror-basemodeloutputwithpooling-object-has-no-attribute-norm).

## Is the toolkit MIT?

Yes. Each model has its own license; see the model cards and the
[THIRD_PARTY notices](https://github.com/gia-uh/siglip2-toolkit/blob/main/THIRD_PARTY_NOTICES.md).
