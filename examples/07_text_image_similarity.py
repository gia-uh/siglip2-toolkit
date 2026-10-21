#!/usr/bin/env python3
"""Compare text descriptions against images in the same embedding space.

    uv run python examples/07_text_image_similarity.py
    uv run python examples/07_text_image_similarity.py --text "a cat" --text "coffee"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from siglip2_toolkit import Embedder, models

HERE = Path(__file__).resolve().parent
DEFAULT_TEXTS = ["a cat", "an astronaut", "a cup of coffee"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", help="Toolkit home (default: ~/.local/share/siglip2-toolkit)")
    parser.add_argument("--model", help="Model id or local path (default: model in use)")
    parser.add_argument("--device", help="auto, cpu, cuda, mps")
    parser.add_argument("--text", action="append", help="Text label; repeat for several")
    parser.add_argument("--image", action="append", dest="images",
                        help="Image file; repeat for several (default: examples/images/*.jpg)")
    args = parser.parse_args()

    texts = args.text or DEFAULT_TEXTS
    images = args.images or sorted(str(path) for path in (HERE / "images").glob("*.jpg"))
    if not images:
        print("error: no images found", file=sys.stderr)
        return 2

    try:
        resolved = models.resolve_model(args.model, home=args.home)
        embedder = Embedder(
            model=resolved.path,
            model_id=resolved.identifier,
            device=args.device or models.default_device(args.home),
        )
        image_vectors = embedder.embed_images(images)
        text_vectors = embedder.embed_texts(texts)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    similarity = image_vectors @ text_vectors.T  # both are L2-normalised
    print(f"model: {resolved.identifier}")
    print(f"images: {len(images)} · texts: {len(texts)}\n")
    print("cosine similarity:")
    print("  " + " " * 28 + "".join(f"{text[:14]:>16}" for text in texts))
    for image, row in zip(images, similarity):
        print(f"  {Path(image).name:28}" + "".join(f"{value:16.3f}" for value in row))

    print("\nbest text per image:")
    for image, row in zip(images, similarity):
        order = np.argsort(row)[::-1]
        best = texts[int(order[0])]
        print(f"  {Path(image).name:28} {best!r} ({row[order[0]]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
