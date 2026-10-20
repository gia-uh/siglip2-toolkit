# Changelog

## 0.0.3 — 2026-10-20

- Cap `transformers<5`: 5.x changed `SiglipModel.get_image_features` /
  `get_text_features` to return the tower outputs instead of the projected
  embeddings (and removed the projection heads), which broke indexing and
  search in fresh installs. A fresh install of 0.0.2 resolved 5.20.2.

## 0.0.2 — 2026-10-20

- The project is now managed with **uv**: committed `uv.lock`, pinned
  `.python-version` (3.12) and documented `uv sync` / `uv run` development.
- Corrected the SigLIP 2 catalog's checkpoint sizes (base ≈1.5 GB,
  large ≈3.5 GB, so400m ≈4.6 GB, giant ≈7.5 GB).

## 0.0.1 — 2026-10-20

- First PyPI preview: library, CLI and optional REST server.
