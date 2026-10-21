"""Command-line interface: ``siglip2 embed|export|inspect|classify|models|serve``.

The CLI is a thin adapter over the public library API: it contains no
embedding or export logic of its own. Model references are resolved through
the toolkit's own model cache (`siglip2_toolkit.models`).
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
    "model": None,
    "batch_size": 32,
    "device": None,
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
        description=(
            "SigLIP 2 toolkit: embeddings, incremental image export, "
            "classification and model management."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--home",
        metavar="DIR",
        help="Toolkit home (models cache + config). Default: ~/.local/share/siglip2-toolkit.",
    )
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
    embed.add_argument("--model", help="Model identifier in your cache or a local model directory (default: model in use).")
    embed.add_argument("--device", help="Device: auto, cpu, cuda, mps, cuda:1... (default: config.json 'device', else auto).")
    embed.add_argument("--batch-size", type=int, default=32)

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
    export.add_argument("--model", help="Model identifier in your cache or a local model directory (default: model in use).")
    export.add_argument("--batch-size", type=int, help="Images per batch.")
    export.add_argument("--device", help="Device: auto, cpu, cuda, mps, cuda:1... (default: config.json 'device', else auto).")
    export.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    export.add_argument("--incremental", action="store_true", help="Only process new or modified files, and drop deleted ones.")
    export.add_argument("--source-file", help="Existing export used as the base for an incremental run (defaults to the output file).")
    export.add_argument("--no-metadata", action="store_true", help="Store only ids and vectors (no metadata); incremental modification detection is disabled.")
    export.add_argument("--create-thumbnails", action="store_true", help="Also generate WEBP thumbnails.")
    export.add_argument("--thumbnail-dir", help="Directory for generated thumbnails.")
    export.add_argument("--thumbnail-size", type=int, nargs=2, metavar=("W", "H"), help="Thumbnail size in pixels.")

    # ---------------------------------------------------------------- inspect
    inspect = sub.add_parser("inspect", help="Validate an export file and print a summary.")
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
    classify.add_argument("--model", help="Model identifier in your cache or a local model directory (default: model in use).")
    classify.add_argument("--device", help="Device: auto, cpu, cuda, mps, cuda:1... (default: config.json 'device', else auto).")
    classify.add_argument("--batch-size", type=int, default=32)
    classify.add_argument("--top-k", type=int, help="Only report the best k labels per image.")

    # ----------------------------------------------------------------- models
    models_parser = sub.add_parser(
        "models",
        help="Manage the toolkit's own model cache (independent from HuggingFace's).",
    )
    models_parser.add_argument(
        "--hf",
        action="store_true",
        help="Show the SigLIP-2 catalog available on HuggingFace and its status (nothing is downloaded).",
    )
    models_sub = models_parser.add_subparsers(dest="models_command")

    current = models_sub.add_parser("current", help="Show the model in use.")
    current.add_argument("--json", dest="as_json", action="store_true", help="Machine-readable output.")

    use = models_sub.add_parser("use", help="Set the model in use.")
    use.add_argument("identifier", help="Identifier of a model in the cache.")

    download = models_sub.add_parser(
        "download",
        help="Download a HuggingFace model into the toolkit cache (the only networked command).",
    )
    download.add_argument("identifier", metavar="ORG/NAME")
    download.add_argument("--revision")

    imp = models_sub.add_parser("import", help="Copy a model from a local path into the toolkit cache.")
    imp.add_argument("path")
    imp.add_argument("--name", help="Identifier to register it under (default: folder name).")

    delete = models_sub.add_parser("delete", help="Delete a model from the toolkit cache.")
    delete.add_argument("identifier")

    # ------------------------------------------------------------------ serve
    serve = sub.add_parser("serve", help="Start the optional REST server (requires the [server] extra).")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", help="Model identifier in your cache or a local model directory (default: model in use).")
    serve.add_argument("--device", help="Device: auto, cpu, cuda, mps, cuda:1... (default: config.json 'device', else auto).")

    return parser


# ---------------------------------------------------------------------- #
# Shared helpers
# ---------------------------------------------------------------------- #


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


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
    return config


# ---------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------- #


def _cmd_embed(args: argparse.Namespace) -> int:
    from .embedder import Embedder
    from .models import default_device, resolve_model
    from .utils import setup_logging

    setup_logging("INFO")
    try:
        resolved = resolve_model(args.model, home=args.home)
        device = args.device or default_device(args.home)
        embedder = Embedder(
            model=resolved.path, model_id=resolved.identifier, device=device
        )
        if args.text:
            matrix = embedder.embed_texts(args.text, batch_size=args.batch_size)
        else:
            matrix = embedder.embed_images(args.image, batch_size=args.batch_size)
    except OSError as exc:
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


def _cmd_export(args: argparse.Namespace) -> int:
    from .exporter import ExportConfig, Exporter
    from .models import default_device
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
        model=config.get("model"),
        home=args.home,
        device=config.get("device") or default_device(args.home),
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
    except OSError as exc:
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
    from .models import default_device, resolve_model
    from .utils import setup_logging

    setup_logging("INFO")

    if not args.finetuned and not args.labels:
        print(
            "error: pass --labels for zero-shot classification or --finetuned for a trained head",
            file=sys.stderr,
        )
        return 2

    try:
        resolved = resolve_model(args.model, home=args.home)
        device = args.device or default_device(args.home)
        if args.finetuned:
            from .classification import ImageClassifier

            classifier = ImageClassifier(model=resolved.path, device=device)
            results = classifier.classify(
                args.image, top_k=args.top_k, batch_size=args.batch_size
            )
        else:
            labels = [label.strip() for label in args.labels.split(",") if label.strip()]
            if not labels:
                print("error: --labels must contain at least one non-empty label", file=sys.stderr)
                return 2
            from .embedder import Embedder

            embedder = Embedder(
                model=resolved.path, model_id=resolved.identifier, device=device
            )
            results = embedder.classify(
                args.image,
                labels,
                template=args.template,
                top_k=args.top_k,
                batch_size=args.batch_size,
            )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for path, row in zip(args.image, results):
        scores = ", ".join(f"{label} {probability:.3f}" for label, probability in row)
        print(f"{path}: {scores}")
    return 0


# ---------------------------------------------------------------------- #
# models commands
# ---------------------------------------------------------------------- #


def _models_list(home: Optional[str]) -> int:
    from . import models

    entries = models.list_models(home=home)
    current = models.current_identifier(home=home)

    print(f"Caché:  {models.cache_dir(home=home)}")
    print(f"Config: {models.config_path(home=home)}")
    if not entries:
        print()
        print("La caché está vacía. Trae un modelo con:")
        print("  siglip2 models download google/siglip2-base-patch16-naflex")
        print("o importa uno desde una ruta local con:")
        print("  siglip2 models import /ruta/al/modelo")
        return 0

    print()
    for entry in entries:
        star = "★" if entry.identifier == current else " "
        kind = {"download": "descargado", "import": "importado"}.get(entry.source, entry.source)
        print(f"{star} {entry.identifier:<52} [{kind}] {_format_bytes(entry.size_bytes):>9}")
    if current is None:
        print()
        print(f"Sin modelo en uso (se usará el default: {DEFAULT_MODEL})")
    return 0


def _models_catalog(home: Optional[str]) -> int:
    from . import models

    available = {entry.identifier for entry in models.list_models(home=home)}
    print("Catálogo SigLIP-2 en HuggingFace — informativo: nada se descarga solo")
    print()
    for item in models.CATALOG:
        mark = "✓" if item.id in available else "☁"
        print(
            f"{mark} {item.id:<46} {item.params:>4}  "
            f"patch{item.patch:<4}{item.resolution:<8}{item.approx_size:>8}"
        )
    print()
    print("✓ ya está en tu caché    ☁ disponible para descargar (siglip2 models download <id>)")
    return 0


def _models_current(args: argparse.Namespace, home: Optional[str]) -> int:
    from . import models

    identifier = models.current_identifier(home=home)
    entry = models.find_model(identifier, home=home) if identifier else None

    if args.as_json:
        print(
            json.dumps(
                {
                    "identifier": identifier,
                    "path": str(entry.path) if entry else None,
                    "source": entry.source if entry else None,
                    "origin": entry.origin if entry else None,
                }
            )
        )
        return 0

    if identifier is None:
        print(f"Sin modelo en uso. Se usará el default: {DEFAULT_MODEL}")
        return 0
    print(identifier)
    if entry is not None:
        print(f"  ruta:   {entry.path}")
        print(f"  origen: {entry.origin} ({entry.source})")
    else:
        print("  aviso: el modelo en uso no está en la caché; usa `siglip2 models use` o descárgalo")
    return 0


def _models_use(args: argparse.Namespace, home: Optional[str]) -> int:
    from . import models

    entry = models.set_current(args.identifier, home=home)
    print(f"Modelo en uso: {entry.identifier}")
    return 0


def _models_download(args: argparse.Namespace, home: Optional[str]) -> int:
    from . import models

    entry = models.download_model(args.identifier, revision=args.revision, home=home)
    print(f"Descargado: {entry.identifier} → {entry.path} ({_format_bytes(entry.size_bytes)})")
    print(f"Para usarlo:  siglip2 models use {entry.identifier}")
    return 0


def _models_import(args: argparse.Namespace, home: Optional[str]) -> int:
    from . import models

    entry = models.import_model(args.path, name=args.name, home=home)
    print(f"Importado: {entry.identifier} → {entry.path} (desde {entry.origin})")
    print(f"Para usarlo:  siglip2 models use {entry.identifier}")
    return 0


def _models_delete(args: argparse.Namespace, home: Optional[str]) -> int:
    from . import models

    entry = models.delete_model(args.identifier, home=home)
    print(f"Borrado: {entry.identifier} ({_format_bytes(entry.size_bytes)} liberados)")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    home = getattr(args, "home", None)
    command = getattr(args, "models_command", None)
    try:
        if command == "current":
            return _models_current(args, home)
        if command == "use":
            return _models_use(args, home)
        if command == "download":
            return _models_download(args, home)
        if command == "import":
            return _models_import(args, home)
        if command == "delete":
            return _models_delete(args, home)
        if getattr(args, "hf", False):
            return _models_catalog(home)
        return _models_list(home)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import run_server

    try:
        run_server(
            host=args.host,
            port=args.port,
            model=args.model,
            device=args.device,
            home=args.home,
        )
    except ModuleNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
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
