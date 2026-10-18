"""Command-line interface: ``siglip2 embed|export|inspect|classify|models|serve``.

The CLI is a thin adapter over the public library API: it contains no
embedding or export logic of its own.
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

DEFAULT_EXPORT_CONFIG: dict[str, Any] = {
    "input_dirs": [],
    "output_file": "embeddings.jsonl",
    "output_format": None,
    "model_name": DEFAULT_MODEL,
    "batch_size": 32,
    "device": "auto",
    "local_files_only": False,
    "log_level": "INFO",
    "incremental": False,
    "source_file": None,
    "store_metadata": True,
    "create_thumbnails": False,
    "thumbnail_dir": "thumbnails",
    "thumbnail_size": [128, 128],
}

_EXPORT_KEYS = tuple(DEFAULT_EXPORT_CONFIG)


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
    embed.add_argument("--offline", action="store_true", help="Use only models already in the local cache (no network).")

    # ----------------------------------------------------------------- export
    export = sub.add_parser(
        "export",
        help="Create or update an embeddings export from image directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    export.add_argument("-c", "--config", dest="config_file", help="YAML configuration file; CLI flags override it.")
    export.add_argument("-i", "--input-dirs", action="append", help="Directory to scan; repeat for several.")
    export.add_argument("-o", "--output-file", help="Output export file (.jsonl or .pkl).")
    export.add_argument("--format", dest="output_format", choices=["jsonl", "pkl"], help="Output format (inferred from the output file when omitted).")
    export.add_argument("--model-name", help="Hugging Face model name for embeddings.")
    export.add_argument("--batch-size", type=int, help="Images per batch.")
    export.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], help="Device to use.")
    export.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    export.add_argument("--incremental", action="store_true", help="Only process new or modified files, and drop deleted ones.")
    export.add_argument("--source-file", help="Existing export used as the base for an incremental run (defaults to the output file).")
    export.add_argument("--no-metadata", action="store_true", help="Store only ids and vectors (no metadata); incremental modification detection is disabled.")
    export.add_argument("--offline", action="store_true", help="Use only models already in the local cache (no network).")
    export.add_argument("--create-thumbnails", action="store_true", help="Also generate WEBP thumbnails.")
    export.add_argument("--thumbnail-dir", help="Directory for generated thumbnails.")
    export.add_argument("--thumbnail-size", type=int, nargs=2, metavar=("W", "H"), help="Thumbnail size in pixels.")

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
    classify.add_argument("--offline", action="store_true", help="Use only models already in the local cache (no network).")

    # ------------------------------------------------------------------ serve
    serve = sub.add_parser("serve", help="Start the optional REST server (requires the [server] extra).")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name.")
    serve.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps...")
    serve.add_argument("--offline", action="store_true", help="Use only models already in the local cache (no network).")

    # ------------------------------------------------------------------ models
    models = sub.add_parser(
        "models",
        help="List or check SigLIP models in the local Hugging Face cache (offline).",
    )
    models.add_argument("--all", action="store_true", help="List every cached model, not only SigLIP-family ones.")
    models.add_argument("--check", metavar="ORG/NAME", help="Check whether a specific model is cached (exit code 0/1).")

    return parser


# ---------------------------------------------------------------------- #
# Command implementations
# ---------------------------------------------------------------------- #


def _cmd_embed(args: argparse.Namespace) -> int:
    from .embedder import Embedder
    from .utils import setup_logging

    setup_logging("INFO")
    embedder = Embedder(model=args.model, device=args.device, local_files_only=args.offline)

    try:
        if args.text:
            matrix = embedder.embed_texts(args.text, batch_size=args.batch_size)
        else:
            matrix = embedder.embed_images(args.image, batch_size=args.batch_size)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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


def _merge_export_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_EXPORT_CONFIG)
    if args.config_file:
        import yaml

        with open(args.config_file, "r", encoding="utf-8") as handle:
            yaml_config = yaml.safe_load(handle) or {}
        config.update(yaml_config)

    for key in _EXPORT_KEYS:
        value = getattr(args, key, None)
        if value is None:
            continue
        if key in ("incremental", "create_thumbnails") and not value:
            continue  # store_true flags: only override when enabled
        config[key] = value

    if getattr(args, "no_metadata", False):
        config["store_metadata"] = False
    if getattr(args, "offline", False):
        config["local_files_only"] = True
    return config


def _cmd_export(args: argparse.Namespace) -> int:
    from .exporter import ExportConfig, Exporter
    from .utils import setup_logging

    config = _merge_export_config(args)
    setup_logging(config.get("log_level", "INFO"))

    if not config.get("input_dirs"):
        print("error: no input directories specified (-i/--input-dirs or config file)", file=sys.stderr)
        return 2

    export_config = ExportConfig(
        input_dirs=list(config["input_dirs"]),
        output_file=str(config["output_file"]),
        output_format=config.get("output_format"),
        model=config.get("model_name", DEFAULT_MODEL),
        device=config.get("device", "auto"),
        local_files_only=bool(config.get("local_files_only")),
        batch_size=int(config.get("batch_size", 32)),
        incremental=bool(config.get("incremental")),
        source_file=config.get("source_file"),
        store_metadata=bool(config.get("store_metadata", True)),
        create_thumbnails=bool(config.get("create_thumbnails")),
        thumbnail_dir=str(config.get("thumbnail_dir", "thumbnails")),
        thumbnail_size=tuple(config.get("thumbnail_size") or (128, 128)),
    )

    bar: dict[str, Any] = {"pbar": None}

    def on_progress(done: int, total: int, path: str) -> None:
        if bar["pbar"] is None:
            from tqdm import tqdm

            bar["pbar"] = tqdm(total=total, desc="Exporting", unit="image")
        pbar = bar["pbar"]
        pbar.update(done - pbar.n)

    try:
        result = Exporter(export_config, on_progress=on_progress).run()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
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
    print(f"Export written to: {result.output_file}")
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
        if isinstance(metadata, dict) and metadata:
            print(f"Metadata keys: {', '.join(sorted(metadata))}")
        else:
            print("Metadata: none")
        missing = sum(1 for record in records if formats.record_id(record) is None)
        if missing:
            print(f"Records without id: {missing}")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    from .utils import setup_logging

    setup_logging("INFO")

    try:
        if args.finetuned:
            from .classification import ImageClassifier

            classifier = ImageClassifier(
                model=args.model, device=args.device, local_files_only=args.offline
            )
            results = classifier.classify(
                args.image, top_k=args.top_k, batch_size=args.batch_size
            )
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

            embedder = Embedder(
                model=args.model, device=args.device, local_files_only=args.offline
            )
            results = embedder.classify(
                args.image,
                labels,
                template=args.template,
                top_k=args.top_k,
                batch_size=args.batch_size,
            )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for path, row in zip(args.image, results):
        scores = ", ".join(f"{label} {probability:.3f}" for label, probability in row)
        print(f"{path}: {scores}")
    return 0


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _cmd_models(args: argparse.Namespace) -> int:
    from . import utils

    hub = utils.default_hub_cache()
    if args.check:
        if utils.is_model_cached(args.check):
            print(f"cached: {args.check}")
            return 0
        print(f"not cached: {args.check} (would need a download or cache seeding)")
        return 1

    cached = utils.list_cached_models(include_all=args.all)
    print(f"Cache: {hub}")
    if not cached:
        kind = "models" if args.all else "SigLIP models"
        print(f"No {kind} found in the cache. Use --all to list every model.")
        return 0
    for model in cached:
        print(f"{model.repo_id:<60} {_format_bytes(model.size_bytes):>10}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import run_server

    try:
        run_server(
            host=args.host,
            port=args.port,
            model=args.model,
            device=args.device,
            local_files_only=args.offline,
        )
    except ModuleNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "embed":
        return _cmd_embed(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "classify":
        return _cmd_classify(args)
    if args.command == "models":
        return _cmd_models(args)
    if args.command == "serve":
        return _cmd_serve(args)

    parser.error(f"unknown command: {args.command!r}")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
