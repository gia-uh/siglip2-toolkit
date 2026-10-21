#!/usr/bin/env python3
"""Export a small image folder and inspect the resulting embeddings file.

The default folder is ``examples/images`` (three openly licensed images).

    uv run python examples/02_export_and_inspect.py
    uv run python examples/02_export_and_inspect.py -i ~/Pictures -o /tmp/photos.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from siglip2_toolkit import ExportConfig, Exporter, formats

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", default=str(HERE / "images"), help="Image directory")
    parser.add_argument("-o", "--output", default="embeddings.jsonl", help="Output .jsonl/.pkl")
    parser.add_argument("--home", help="Toolkit home")
    parser.add_argument("--model", help="Model id or local path (default: model in use)")
    parser.add_argument("--device", help="auto, cpu, cuda, mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser()
    if not input_dir.is_dir():
        print(f"error: no such directory: {input_dir}", file=sys.stderr)
        return 2

    config = ExportConfig(
        input_dirs=[str(input_dir)],
        output_file=args.output,
        model=args.model,
        home=args.home,
        device=args.device,
        batch_size=args.batch_size,
        incremental=args.incremental,
    )
    try:
        result = Exporter(config).run()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"processed={result.processed} skipped={result.skipped} "
          f"removed={result.removed} total={result.total}")
    if result.embedding_space:
        space = result.embedding_space
        print(
            f"embedding space: {space.model} "
            f"({space.dimensions} dimensions, {space.normalization})"
        )

    header = formats.read_header(args.output)
    records = formats.load_records(args.output)
    print(f"header: {header}")
    print(f"records: {len(records)}")
    for record in records[:3]:
        metadata = record.get("metadata") or {}
        print(f"  {record.get('id')}  {metadata.get('filename', '')}  dim={len(record['vector'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
