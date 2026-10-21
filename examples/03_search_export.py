#!/usr/bin/env python3
"""Search an embeddings export with a text query.

First create an export (see ``02_export_and_inspect.py``), then run:

    uv run python examples/03_search_export.py --export embeddings.jsonl --query "a cat"
"""

from __future__ import annotations

import argparse
import sys

from siglip2_toolkit import Embedder, formats, models
from siglip2_toolkit.retrieval import (
    apply_similarity_drop,
    rank_by_similarity,
    results_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default="embeddings.jsonl", help="Export file")
    parser.add_argument("--query", required=True, help="Text query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--drop", type=float, default=15.0, help="Max %% drop from the best score")
    parser.add_argument("--home", help="Toolkit home")
    parser.add_argument("--model", help="Model id or local path (default: model in use)")
    parser.add_argument("--device", help="auto, cpu, cuda, mps")
    args = parser.parse_args()

    try:
        records = formats.load_records(args.export)
        matrix = results_matrix(records)
        resolved = models.resolve_model(args.model, home=args.home)
        embedder = Embedder(
            model=resolved.path,
            model_id=resolved.identifier,
            device=args.device or models.default_device(args.home),
        )
        query = embedder.embed_texts([args.query])[0]
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ranked = rank_by_similarity(query, matrix, top_k=args.top_k, metric="cosine")
    ranked = apply_similarity_drop(ranked, max_drop_percent=args.drop)
    print(f"model: {resolved.identifier} · records: {len(records)} · results: {len(ranked)}")
    for index, distance in ranked:
        record = records[index]
        metadata = record.get("metadata") or {}
        label = metadata.get("path") or record.get("id") or index
        print(f"{distance:8.4f}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
