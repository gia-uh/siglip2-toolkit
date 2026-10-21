#!/usr/bin/env python3
"""Zero-shot classification with candidate labels.

    uv run python examples/04_classify.py
    uv run python examples/04_classify.py --image photo.jpg --labels "cat,dog,beach"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from siglip2_toolkit import Embedder, models

HERE = Path(__file__).resolve().parent
DEFAULT_LABELS = ["cat", "astronaut", "coffee", "rocket", "horse"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        help="Image file; repeat for several (default: examples/images/*.jpg)",
    )
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS),
                        help="Comma-separated candidate labels")
    parser.add_argument("--template", default="a photo of a {}",
                        help="Prompt template; {} is replaced by each label")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--home", help="Toolkit home")
    parser.add_argument("--model", help="Model id or local path (default: model in use)")
    parser.add_argument("--device", help="auto, cpu, cuda, mps")
    args = parser.parse_args()

    images = args.images or sorted(str(path) for path in (HERE / "images").glob("*.jpg"))
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not images or not labels:
        print("error: provide at least one image and one label", file=sys.stderr)
        return 2

    try:
        resolved = models.resolve_model(args.model, home=args.home)
        embedder = Embedder(
            model=resolved.path,
            model_id=resolved.identifier,
            device=args.device or models.default_device(args.home),
        )
        results = embedder.classify(
            images, labels, template=args.template, top_k=args.top_k
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"model: {resolved.identifier} · labels: {', '.join(labels)}")
    for image, scored in zip(images, results):
        print(f"\n{image}")
        for label, probability in scored:
            print(f"  {probability:6.3f}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
