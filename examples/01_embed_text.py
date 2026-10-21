#!/usr/bin/env python3
"""Embed texts and show pairwise cosine similarity.

Run after downloading a model:

    siglip2 models download google/siglip2-base-patch16-naflex
    siglip2 models use google/siglip2-base-patch16-naflex
    uv run python examples/01_embed_text.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from siglip2_toolkit import Embedder, models

DEFAULT_TEXTS = [
    "a wedding at sunset",
    "a bride and groom on the beach",
    "a photo of a cat",
    "a spaceship orbiting the earth",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", help="Toolkit home (default: ~/.local/share/siglip2-toolkit)")
    parser.add_argument("--model", help="Model id or local path (default: model in use)")
    parser.add_argument("--device", help="auto, cpu, cuda, mps (default: config.json)")
    parser.add_argument("--text", action="append", help="Text to embed; repeat for several")
    args = parser.parse_args()

    texts = args.text or DEFAULT_TEXTS
    try:
        resolved = models.resolve_model(args.model, home=args.home)
        embedder = Embedder(
            model=resolved.path,
            model_id=resolved.identifier,
            device=args.device or models.default_device(args.home),
        )
        vectors = embedder.embed_texts(texts)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Download a model with `siglip2 models download …` first.", file=sys.stderr)
        return 2

    print(f"model: {resolved.identifier}")
    print(f"shape: {vectors.shape}  (texts, dimensions)")

    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    print("\npairwise cosine similarity:")
    for index, text in enumerate(texts):
        print(f"  [{index}] {text}")
    print()
    for row in similarity:
        print("  " + "  ".join(f"{value:6.3f}" for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
