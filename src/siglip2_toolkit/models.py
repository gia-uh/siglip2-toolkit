"""Toolkit model cache: the toolkit's own store of models.

Everything the toolkit uses lives here, or is a local path you provide. The
HuggingFace cache is unrelated: this toolkit never reads it. The only network
access in the whole toolkit is :func:`download_model` (the
``siglip2 models download`` command).

Layout::

    <home>/                      # default: ~/.local/share/siglip2-toolkit
    ├── config.json              # {"cache_dir": "...", "current": "..."}
    └── models/
        └── google--siglip2-base-patch16-naflex/
            ├── model.json       # provenance: identifier, source, origin, date
            └── ...              # model files (config.json, weights, processor)

``config.json`` is created on first use with the defaults already written, so
you can see and edit them (e.g. point ``cache_dir`` at another disk). There
are no environment variables.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .defaults import DEFAULT_MODEL

logger = logging.getLogger(__name__)

#: Default toolkit home (no environment variables; pass ``home=`` / ``--home``
#: explicitly for non-default installations such as tests).
DEFAULT_HOME = Path.home() / ".local/share/siglip2-toolkit"

_CONFIG_NAME = "config.json"
_ENTRY_META = "model.json"


@dataclass(frozen=True)
class CatalogModel:
    """One entry of the informational SigLIP 2 catalog (nothing is automatic)."""

    id: str
    params: str
    patch: int
    resolution: str  # "224" | "256" | "384" | "512" | "naflex"
    approx_size: str


#: Curated, informational list of the SigLIP 2 family published on
#: HuggingFace. The toolkit never picks a model for you; this only shows what
#: exists, so you can see what you already have and what you could download.
#: Sizes are the actual HuggingFace checkpoint downloads (fp32 safetensors),
#: not just the vision encoder.
CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel("google/siglip2-base-patch32-256", "86M", 32, "256", "≈1.5 GB"),
    CatalogModel("google/siglip2-base-patch16-224", "86M", 16, "224", "≈1.5 GB"),
    CatalogModel("google/siglip2-base-patch16-256", "86M", 16, "256", "≈1.5 GB"),
    CatalogModel("google/siglip2-base-patch16-384", "86M", 16, "384", "≈1.5 GB"),
    CatalogModel("google/siglip2-base-patch16-512", "86M", 16, "512", "≈1.5 GB"),
    CatalogModel("google/siglip2-base-patch16-naflex", "86M", 16, "naflex", "≈1.5 GB"),
    CatalogModel("google/siglip2-large-patch16-256", "303M", 16, "256", "≈3.5 GB"),
    CatalogModel("google/siglip2-large-patch16-384", "303M", 16, "384", "≈3.5 GB"),
    CatalogModel("google/siglip2-large-patch16-512", "303M", 16, "512", "≈3.5 GB"),
    CatalogModel("google/siglip2-so400m-patch14-224", "400M", 14, "224", "≈4.6 GB"),
    CatalogModel("google/siglip2-so400m-patch14-384", "400M", 14, "384", "≈4.6 GB"),
    CatalogModel("google/siglip2-so400m-patch16-256", "400M", 16, "256", "≈4.6 GB"),
    CatalogModel("google/siglip2-so400m-patch16-384", "400M", 16, "384", "≈4.6 GB"),
    CatalogModel("google/siglip2-so400m-patch16-512", "400M", 16, "512", "≈4.6 GB"),
    CatalogModel("google/siglip2-so400m-patch16-naflex", "400M", 16, "naflex", "≈4.6 GB"),
    CatalogModel("google/siglip2-giant-opt-patch16-256", "1B", 16, "256", "≈7.5 GB"),
    CatalogModel("google/siglip2-giant-opt-patch16-384", "1B", 16, "384", "≈7.5 GB"),
)


@dataclass(frozen=True)
class ModelEntry:
    """A model stored in the toolkit cache."""

    identifier: str
    path: Path
    source: str  # "download" | "import"
    origin: str  # "huggingface:<id>@<rev>" | "<absolute path>"
    created_at: str
    size_bytes: int


@dataclass(frozen=True)
class ResolvedModel:
    """A model resolved to a local directory, plus its identifier."""

    path: Path
    identifier: str


# --------------------------------------------------------------------- #
# Home, config and cache directory
# --------------------------------------------------------------------- #


def toolkit_home(home: str | Path | None = None) -> Path:
    """The toolkit home directory (explicit for non-default installations)."""
    return Path(home).expanduser() if home else DEFAULT_HOME


def config_path(home: str | Path | None = None) -> Path:
    """Path of the toolkit ``config.json``."""
    return toolkit_home(home) / _CONFIG_NAME


def ensure_config(home: str | Path | None = None) -> dict:
    """Read ``config.json``, creating it with visible defaults on first use."""
    path = config_path(home)
    if path.is_file():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted toolkit config at {path}: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"Corrupted toolkit config at {path}: expected an object")
        return config

    config = {
        "cache_dir": str(toolkit_home(home) / "models"),
        "current": None,
        "device": "auto",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    logger.info("Created toolkit config: %s", path)
    return config


def save_config(config: dict, home: str | Path | None = None) -> None:
    """Write the toolkit ``config.json``."""
    path = config_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def cache_dir(home: str | Path | None = None) -> Path:
    """The model cache directory, as configured in ``config.json``.

    Accepts absolute paths or paths relative to the toolkit home.
    """
    configured = ensure_config(home).get("cache_dir") or "models"
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = toolkit_home(home) / path
    return path


def default_device(home: str | Path | None = None) -> str:
    """The device configured in ``config.json`` (``"auto"`` when unset)."""
    value = ensure_config(home).get("device")
    return str(value) if value else "auto"


# --------------------------------------------------------------------- #
# Cache inventory
# --------------------------------------------------------------------- #


def flatten(identifier: str) -> str:
    """Cache entry directory name for an identifier (``org/name`` → ``org--name``)."""
    return str(identifier).replace("/", "--")


def _dir_size(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            try:
                total += file.stat().st_size
            except OSError:
                continue
    return total


def _write_entry_meta(directory: Path, identifier: str, source: str, origin: str) -> None:
    meta = {
        "identifier": identifier,
        "source": source,
        "origin": origin,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / _ENTRY_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _read_entry_meta(directory: Path) -> Optional[dict]:
    meta_path = directory / _ENTRY_META
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Ignoring corrupted entry metadata: %s", meta_path)
        return None
    return meta if isinstance(meta, dict) else None


def list_models(home: str | Path | None = None) -> list[ModelEntry]:
    """List the models stored in the toolkit cache."""
    root = cache_dir(home)
    if not root.is_dir():
        return []
    entries: list[ModelEntry] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        meta = _read_entry_meta(directory)
        if meta is None:
            continue
        entries.append(
            ModelEntry(
                identifier=str(meta.get("identifier", directory.name)),
                path=directory,
                source=str(meta.get("source", "unknown")),
                origin=str(meta.get("origin", "")),
                created_at=str(meta.get("created_at", "")),
                size_bytes=_dir_size(directory),
            )
        )
    return entries


def find_model(identifier: str, home: str | Path | None = None) -> Optional[ModelEntry]:
    """Find a cache entry by identifier (or by its flattened directory name)."""
    wanted = str(identifier)
    for entry in list_models(home):
        if entry.identifier == wanted or entry.path.name == flatten(wanted):
            return entry
    return None


def current_identifier(home: str | Path | None = None) -> Optional[str]:
    """The identifier of the model in use, or ``None``."""
    value = ensure_config(home).get("current")
    return str(value) if value else None


def set_current(identifier: str, home: str | Path | None = None) -> ModelEntry:
    """Set the model in use (it must already be in the cache)."""
    entry = find_model(identifier, home)
    if entry is None:
        raise FileNotFoundError(
            f"'{identifier}' is not in the toolkit cache ({cache_dir(home)}). "
            "Download or import it first."
        )
    config = ensure_config(home)
    config["current"] = entry.identifier
    save_config(config, home)
    return entry


# --------------------------------------------------------------------- #
# Fetching models into the cache
# --------------------------------------------------------------------- #


def is_model_directory(path: str | Path) -> bool:
    """Whether ``path`` looks like a loadable transformers model directory."""
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    for pattern in ("*.safetensors", "pytorch_model.bin", "*.bin", "*.gguf"):
        if any(path.glob(pattern)):
            return True
    return False


def download_model(
    identifier: str,
    revision: Optional[str] = None,
    home: str | Path | None = None,
) -> ModelEntry:
    """Download a HuggingFace model into the toolkit cache (the network path).

    This is the only function in the toolkit that uses the network.
    """
    identifier = str(identifier)
    if "/" not in identifier or Path(identifier).exists():
        raise ValueError(
            f"'{identifier}' is not a HuggingFace model id (expected org/name)."
        )
    existing = find_model(identifier, home)
    if existing is not None:
        raise FileExistsError(
            f"'{identifier}' is already in the toolkit cache ({existing.path}). "
            "Delete it first to download it again."
        )

    target = cache_dir(home) / flatten(identifier)
    target.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    logger.info("Downloading '%s' from HuggingFace into %s ...", identifier, target)
    snapshot_download(repo_id=identifier, revision=revision, local_dir=str(target))

    origin = f"huggingface:{identifier}" + (f"@{revision}" if revision else "")
    _write_entry_meta(target, identifier, "download", origin)
    entry = find_model(identifier, home)
    assert entry is not None
    logger.info("Downloaded: %s (%d bytes)", entry.identifier, entry.size_bytes)
    return entry


def import_model(
    path: str | Path,
    name: Optional[str] = None,
    home: str | Path | None = None,
) -> ModelEntry:
    """Copy a model from a local path into the toolkit cache.

    The source directory is never modified. ``name`` sets the identifier
    (defaults to the folder name).
    """
    source = Path(path).expanduser()
    if not source.is_dir() or not is_model_directory(source):
        raise ValueError(
            f"'{path}' is not a loadable model directory "
            "(it needs config.json and weight files)."
        )
    identifier = str(name) if name else source.name
    existing = find_model(identifier, home)
    if existing is not None:
        raise FileExistsError(
            f"'{identifier}' is already in the toolkit cache ({existing.path}). "
            "Use --name to register it under a different identifier."
        )

    target = cache_dir(home) / flatten(identifier)
    if target.exists():
        raise FileExistsError(f"Cache directory already exists: {target}")

    logger.info("Copying model from %s into the cache at %s ...", source, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)  # dereferences symlinks: self-contained copy
    _write_entry_meta(target, identifier, "import", str(source.resolve()))
    entry = find_model(identifier, home)
    assert entry is not None
    logger.info("Imported: %s (%d bytes)", entry.identifier, entry.size_bytes)
    return entry


def delete_model(identifier: str, home: str | Path | None = None) -> ModelEntry:
    """Delete a model from the toolkit cache (refuses the model in use)."""
    entry = find_model(identifier, home)
    if entry is None:
        raise FileNotFoundError(f"'{identifier}' is not in the toolkit cache.")
    if current_identifier(home) == entry.identifier:
        raise ValueError(
            f"'{entry.identifier}' is the model in use; switch first with: "
            "siglip2 models use <other-model>"
        )
    shutil.rmtree(entry.path)
    logger.info("Deleted: %s (%d bytes)", entry.identifier, entry.size_bytes)
    return entry


# --------------------------------------------------------------------- #
# Resolution: what the embedder/CLI actually loads
# --------------------------------------------------------------------- #


def _identifier_for_path(path: Path, home: str | Path | None = None) -> Optional[str]:
    """Reverse lookup: identifier of a cache entry, when ``path`` is one."""
    try:
        root = cache_dir(home).resolve()
    except Exception:  # noqa: BLE001 - resolution must stay usable
        return None
    try:
        if path.resolve().parent == root:
            meta = _read_entry_meta(path)
            if meta and meta.get("identifier"):
                return str(meta["identifier"])
    except OSError:
        return None
    return None


def resolve_model(
    model: Optional[str] = None,
    home: str | Path | None = None,
) -> ResolvedModel:
    """Resolve a model reference to a local directory.

    Accepted references: an identifier in the toolkit cache, or a path to a
    local model directory. With ``model=None`` the model in use is used, or
    the built-in default. Never touches the network.
    """
    if model:
        candidate = Path(model).expanduser()
        if candidate.is_dir():
            identifier = _identifier_for_path(candidate, home) or str(candidate)
            return ResolvedModel(path=candidate.resolve(), identifier=identifier)
        entry = find_model(model, home)
        if entry is not None:
            return ResolvedModel(path=entry.path, identifier=entry.identifier)
        if "/" in str(model):
            hint = f"Download it first with:  siglip2 models download {model}"
        else:
            hint = "Import it from a local path with:  siglip2 models import <path> [--name ...]"
        raise FileNotFoundError(
            f"Model '{model}' is not a local directory and is not in the toolkit "
            f"cache ({cache_dir(home)}). {hint}"
        )

    identifier = current_identifier(home) or DEFAULT_MODEL
    entry = find_model(identifier, home)
    if entry is not None:
        return ResolvedModel(path=entry.path, identifier=entry.identifier)
    raise FileNotFoundError(
        f"No model is in use and the default '{identifier}' is not in the toolkit "
        f"cache ({cache_dir(home)}). Download it with:  "
        f"siglip2 models download {identifier}"
    )
