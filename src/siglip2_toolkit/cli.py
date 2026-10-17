"""Command-line interface: ``siglip2 embed|index|inspect|serve``.

The CLI is a thin adapter over the public library API: it contains no
embedding or indexing logic of its own.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any, Optional

from . import __version__
from .defaults import DEFAULT_MODEL

DEFAULT_INDEX_CONFIG: dict[str, Any] = {
    "input_dirs": [],
    "output_file": "image_index.jsonl",
    "output_format": None,
    "model_name": DEFAULT_MODEL,
    "batch_size": 32,
    "device": "auto",
    "log_level": "INFO",
    "reindex": False,
    "source_index": None,
    "create_thumbnails": False,
    "thumbnail_dir": "thumbnails",
    "thumbnail_size": [128, 128],
}

_INDEX_KEYS = tuple(DEFAULT_INDEX_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siglip2",
        description="SigLIP 2 toolkit: embeddings, image indexing and retrieval.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ embed
    embed = sub.add_parser(
        "embed",
        help="Generate text or image embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = embed.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", action="append", metavar="TEXT", help="Text to embed; repeat for several.")
    group.add_argument("--image", action="append", metavar="PATH", help="Image file to embed; repeat for several.")
    embed.add_argument("-o", "--output", help="Write embeddings to a file instead of stdout.")
    embed.add_argument("--format", choices=["json", "pickle"], default="json", help="Output file format.")
    embed.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name.")
    embed.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps...")
    embed.add_argument("--batch-size", type=int, default=32)

    # ------------------------------------------------------------------ index
    index = sub.add_parser(
        "index",
        help="Create or update an image index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    index.add_argument("-c", "--config", dest="config_file", help="YAML configuration file; CLI flags override it.")
    index.add_argument("-i", "--input-dirs", action="append", help="Directory to scan; repeat for several.")
    index.add_argument("-o", "--output-file", help="Output index file (.jsonl or .pkl).")
    index.add_argument("--format", dest="output_format", choices=["jsonl", "pkl"], help="Output format (inferred from the output file when omitted).")
    index.add_argument("--model-name", help="Hugging Face model name for embeddings.")
    index.add_argument("--batch-size", type=int, help="Images per batch.")
    index.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], help="Device to use.")
    index.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    index.add_argument("--reindex", action="store_true", help="Only process new or modified files, and drop deleted ones.")
    index.add_argument("--source-index", help="Existing index used as the re-index base (defaults to the output file).")
    index.add_argument("--create-thumbnails", action="store_true", help="Also generate WEBP thumbnails.")
    index.add_argument("--thumbnail-dir", help="Directory for generated thumbnails.")
    index.add_argument("--thumbnail-size", type=int, nargs=2, metavar=("W", "H"), help="Thumbnail size in pixels.")

    # ---------------------------------------------------------------- inspect
    inspect = sub.add_parser("inspect", help="Validate an index file and print a summary.")
    inspect.add_argument("index_file")

    # --------------------------------------------------------------- classify
    classify = sub.add_parser(
        "classify",
        help="Classify images: zero-shot with --labels, or with a fine-tuned model (--finetuned).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    classify.add_argument("--image", action="append", required=True, metavar="PATH", help="Image file; repeat for several.")
    classify.add_argument("--labels", metavar="L1,L2,...", help="Comma-separated candidate labels (zero-shot mode).")
    classify.add_argument("--template", default="{}", help="Prompt template; '{}' is replaced by each label.")
    classify.add_argument("--finetuned", action="store_true", help="Use a fine-tuned classification head instead of --labels.")
    classify.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name.")
    classify.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps...")
    classify.add_argument("--batch-size", type=int, default=32)
    classify.add_argument("--top-k", type=int, help="Only report the best k labels per image.")

    # ------------------------------------------------------------------ serve
    serve = sub.add_parser("serve", help="Start the optional REST server (requires the [server] extra).")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name.")
    serve.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps...")

    return parser


# ---------------------------------------------------------------------- #
# Command implementations
# ---------------------------------------------------------------------- #


def _cmd_embed(args: argparse.Namespace) -> int:
    from .embedder import Embedder
    from .utils import setup_logging

    setup_logging("INFO")
    embedder = Embedder(model=args.model, device=args.device)

    if args.text:
        matrix = embedder.embed_texts(args.text, batch_size=args.batch_size)
    else:
        matrix = embedder.embed_images(args.image, batch_size=args.batch_size)

    if args.output:
        if args.format == "json":
            payload = {
                "model": embedder.model_name,
                "dimensions": embedder.dimensions,
                "vectors": matrix.tolist(),
            }
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        else:
            with open(args.output, "wb") as handle:
                pickle.dump(matrix, handle)
        print(f"Saved {matrix.shape[0]} embedding(s) to {args.output}")
    else:
        for row in matrix:
            print(json.dumps(row.tolist()))
    return 0


def _merge_index_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_INDEX_CONFIG)
    if args.config_file:
        import yaml

        with open(args.config_file, "r", encoding="utf-8") as handle:
            yaml_config = yaml.safe_load(handle) or {}
        config.update(yaml_config)

    for key in _INDEX_KEYS:
        value = getattr(args, key, None)
        if value is None:
            continue
        if key in ("reindex", "create_thumbnails") and not value:
            continue  # store_true flags: only override when enabled
        config[key] = value
    return config


def _cmd_index(args: argparse.Namespace) -> int:
    from .indexer import Indexer, IndexerConfig
    from .utils import setup_logging

    config = _merge_index_config(args)
    setup_logging(config.get("log_level", "INFO"))

    if not config.get("input_dirs"):
        print("error: no input directories specified (-i/--input-dirs or config file)", file=sys.stderr)
        return 2

    indexer_config = IndexerConfig(
        input_dirs=list(config["input_dirs"]),
        output_file=str(config["output_file"]),
        output_format=config.get("output_format"),
        model=config.get("model_name", DEFAULT_MODEL),
        device=config.get("device", "auto"),
        batch_size=int(config.get("batch_size", 32)),
        reindex=bool(config.get("reindex")),
        source_index=config.get("source_index"),
        create_thumbnails=bool(config.get("create_thumbnails")),
        thumbnail_dir=str(config.get("thumbnail_dir", "thumbnails")),
        thumbnail_size=tuple(config.get("thumbnail_size") or (128, 128)),
    )

    bar: dict[str, Any] = {"pbar": None}

    def on_progress(done: int, total: int, path: str) -> None:
        if bar["pbar"] is None:
            from tqdm import tqdm

            bar["pbar"] = tqdm(total=total, desc="Indexing", unit="image")
        pbar = bar["pbar"]
        pbar.update(done - pbar.n)

    try:
        result = Indexer(indexer_config, on_progress=on_progress).run()
    finally:
        if bar["pbar"] is not None:
            bar["pbar"].close()

    print(
        f"Done. processed={result.processed} skipped={result.skipped} "
        f"removed={result.removed} total={result.total}"
    )
    if result.embedding_space is not None:
        space = result.embedding_space
        print(f"Embedding space: {space.model} (d={space.dimensions}, {space.normalization})")
    print(f"Index written to: {result.output_file}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from . import formats

    path = os.fspath(args.index_file)
    if not os.path.exists(path):
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    header = formats.read_header(path)
    records = formats.load_records(path)

    print(f"File: {path}")
    print(f"Format: {formats.infer_format(path)}")
    if header is None:
        print("Header: none (legacy v1 index)")
    else:
        print(f"Header: {header.get('format')} v{header.get('version')} (created {header.get('created_at', 'unknown')})")
        space = formats.header_embedding_space(header)
        if space is not None:
            print(
                f"Embedding space: model={space.model} dimensions={space.dimensions} "
                f"normalization={space.normalization} id={space.id}"
            )
    print(f"Records: {len(records)}")

    if records:
        vector = records[0].get("vector") or []
        print(f"Vector dimension: {len(vector)}")
        metadata = records[0].get("metadata")
        if isinstance(metadata, dict):
            print(f"Metadata keys: {', '.join(sorted(metadata))}")
        missing = sum(1 for record in records if formats.record_path(record) is None)
        if missing:
            print(f"Records without id (metadata.path): {missing}")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    from .utils import setup_logging

    setup_logging("INFO")

    if args.finetuned:
        from .classification import ImageClassifier

        classifier = ImageClassifier(model=args.model, device=args.device)
        results = classifier.classify(args.image, top_k=args.top_k, batch_size=args.batch_size)
    else:
        if not args.labels:
            print(
                "error: pass --labels for zero-shot classification or --finetuned for a trained head",
                file=sys.stderr,
            )
            return 2
        labels = [label.strip() for label in args.labels.split(",") if label.strip()]
        if not labels:
            print("error: --labels must contain at least one non-empty label", file=sys.stderr)
            return 2
        from .embedder import Embedder

        embedder = Embedder(model=args.model, device=args.device)
        results = embedder.classify(
            args.image, labels, template=args.template, top_k=args.top_k, batch_size=args.batch_size
        )

    for path, row in zip(args.image, results):
        scores = ", ".join(f"{label} {probability:.3f}" for label, probability in row)
        print(f"{path}: {scores}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import run_server

    try:
        run_server(host=args.host, port=args.port, model=args.model, device=args.device)
    except ModuleNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "embed":
        return _cmd_embed(args)
    if args.command == "index":
        return _cmd_index(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "classify":
        return _cmd_classify(args)
    if args.command == "serve":
        return _cmd_serve(args)

    parser.error(f"unknown command: {args.command!r}")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
