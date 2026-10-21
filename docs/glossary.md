# Glossary

| Term | Meaning |
|---|---|
| **Embedding** | A numeric vector representing the meaning of a text or image. |
| **Model** | A SigLIP 2 checkpoint loaded from a local directory (usually the toolkit cache). |
| **Checkpoint** | The model weights, configuration, preprocessor and tokenizer files. |
| **Embedding space** | The model + preprocessing identity that makes vectors comparable. |
| **Manifest** | The `embedding_space` header stored in an export. |
| **Export** | A portable JSONL/pickle file with a header and records. |
| **Record** | One vector plus optional `id` and `metadata`. |
| **`id`** | Stable record identifier; needed for incremental updates and deduplication. |
| **Metadata** | Optional JSON data: path, filename, dimensions, EXIF, timestamps, thumbnail path. |
| **Thumbnail** | An optional derived image; never embedded in the export. |
| **Normalization** | L2 normalisation of embeddings; the toolkit default. |
| **Cosine similarity** | Dot product of normalised vectors; the toolkit reports cosine **distance** in ranking. |
| **L2² / euclidean** | Squared/euclidean distance; lower is better. |
| **RRF** | Reciprocal Rank Fusion; combines ranked lists without comparing raw scores. |
| **Zero-shot classification** | Classify against candidate labels without task-specific training. |
| **Template** | Prompt wrapping each label, e.g. `a photo of a {}`. |
| **Fine-tuned classifier** | A model with a trained classification head and its own labels (`ImageClassifier`). |
| **Logits** | Raw model scores before softmax; available via `zero_shot_logits`. |
| **Toolkit home** | Directory containing `config.json` and the model cache. |
| **Model cache** | Downloaded/imported checkpoints under the toolkit home. |
| **Revision** | The immutable model revision/hash recorded in the manifest. |
| **Preprocess** | The image/text preprocessing contract that must match for vectors to be comparable. |
| **Incremental export** | Update only new/modified files and drop deleted ones; needs ids/metadata. |
| **Exporter** | The class/CLI that scans images and writes an export. |
| **Embedder** | The class that turns texts/images into vectors and runs zero-shot classification. |
| **Filters** | Composable metadata predicates (`field`, `matches`, `to_dict`, `from_dict`). |
| **CLI** | The `siglip2` command. |
| **REST** | The optional FastAPI server (`siglip2 serve`). |
| **Toolkit** | The library, CLI and REST surfaces over the same core. |
