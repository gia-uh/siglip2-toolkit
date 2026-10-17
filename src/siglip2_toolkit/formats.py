"""Embeddings export format (v2): specification and read/write helpers.

An export file is a sequence of objects, optionally preceded by a header::

    {"format": "siglip2-export", "version": 2,
     "embedding_space": {"model": "...", "dimensions": 768, "normalization": "l2"},
     "created_at": "2026-10-17T12:00:00+00:00"}
    {"id": "/photos/a.jpg", "vector": [...], "metadata": {...}}
    {"id": "/photos/b.jpg", "vector": [...]}
    ...

``jsonl`` files store one object per line; ``pkl`` files store a sequence of
pickled objects. Record-only files (written before format v2) are still
readable.

A record carries its identity in ``id`` (any stable string; the exporter
uses the absolute image path). ``metadata`` is optional and free-form: store
EXIF, timestamps, tags — or nothing at all. Records without an ``id`` fall
back to ``metadata["path"]`` when read back.

This is an interchange format, not a database: consumers load the vectors
with whatever tool they prefer (a vector store, numpy, another pipeline).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

logger = logging.getLogger(__name__)

FORMAT_NAME = "siglip2-export"
FORMAT_VERSION = 2

JSONL_SUFFIXES = (".jsonl", ".jsonlines", ".ndjson")
PKL_SUFFIXES = (".pkl", ".pickle")
_TEMP_SUFFIXES = (".tmp", ".part")


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EmbeddingSpace:
    """Identity of the vector space produced by an embedder.

    Two vectors are only comparable when they were produced by the same model
    (and the same preprocessing). This manifest is stored in the export header
    so consumers can detect mismatches.
    """

    model: str
    dimensions: Optional[int] = None
    normalization: str = "l2"
    revision: Optional[str] = None
    preprocess: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "model": self.model,
            "dimensions": self.dimensions,
            "normalization": self.normalization,
            "revision": self.revision,
            "preprocess": self.preprocess,
        }
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingSpace":
        return cls(
            model=str(data["model"]),
            dimensions=data.get("dimensions"),
            normalization=data.get("normalization", "l2"),
            revision=data.get("revision"),
            preprocess=data.get("preprocess"),
        )

    @property
    def id(self) -> str:
        """Stable short identifier derived from the manifest contents."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

    def compatible_with(self, other: "EmbeddingSpace") -> bool:
        """Whether vectors from both spaces can be compared.

        Unknown dimensions on either side are tolerated; a different model or
        a different declared dimension is not.
        """
        if self.model != other.model:
            return False
        if self.dimensions is not None and other.dimensions is not None:
            return self.dimensions == other.dimensions
        return True


def make_header(
    embedding_space: Optional[EmbeddingSpace] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Build the export header object."""
    header: dict[str, Any] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "created_at": created_at or now_iso(),
    }
    if embedding_space is not None:
        header["embedding_space"] = embedding_space.to_dict()
    return header


def is_header(obj: Any) -> bool:
    """Whether an object read from an export file is a format header."""
    return isinstance(obj, dict) and obj.get("format") == FORMAT_NAME


def make_record(
    vector: Sequence[float],
    metadata: Optional[dict[str, Any]] = None,
    *,
    id: Optional[str] = None,
) -> dict[str, Any]:
    """Build an export record.

    ``id`` identifies the vector (enable it for incremental updates and
    deduplication); ``metadata`` is optional and may hold any
    JSON-serializable data.
    """
    record: dict[str, Any] = {}
    if id is not None:
        record["id"] = str(id)
    record["vector"] = list(vector)
    if metadata is not None:
        record["metadata"] = dict(metadata)
    return record


def record_id(record: dict[str, Any]) -> Optional[str]:
    """The record id: top-level ``id``, or legacy ``metadata["path"]``."""
    if not isinstance(record, dict):
        return None
    identifier = record.get("id")
    if isinstance(identifier, str) and identifier:
        return identifier
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        path = metadata.get("path")
        if isinstance(path, str) and path:
            return path
    return None


def infer_format(path: str | Path) -> str:
    """Infer the export format (``jsonl`` or ``pkl``) from a file name.

    Temporary suffixes (``.tmp``/``.part``) are stripped first, so the
    interrupted-run convention ``<export>.jsonl.tmp`` also resolves.
    """
    name = Path(path).name.lower()
    for temp_suffix in _TEMP_SUFFIXES:
        if name.endswith(temp_suffix):
            name = name[: -len(temp_suffix)]
            break
    suffix = Path(name).suffix
    if suffix in JSONL_SUFFIXES:
        return "jsonl"
    if suffix in PKL_SUFFIXES:
        return "pkl"
    raise ValueError(
        f"Cannot infer export format from '{path}'. "
        f"Use one of {JSONL_SUFFIXES + PKL_SUFFIXES}."
    )


def read_header(path: str | Path) -> Optional[dict[str, Any]]:
    """Read the export header, or ``None`` for legacy or missing files."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        if infer_format(path) == "jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    return obj if is_header(obj) else None
            return None
        with path.open("rb") as handle:
            obj = pickle.load(handle)
            return obj if is_header(obj) else None
    except (EOFError, json.JSONDecodeError, pickle.UnpicklingError, ValueError):
        return None


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield export records, skipping the header if present."""
    path = Path(path)
    fmt = infer_format(path)
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not is_header(obj):
                    yield obj
    else:
        with path.open("rb") as handle:
            while True:
                try:
                    obj = pickle.load(handle)
                except EOFError:
                    break
                if not is_header(obj):
                    yield obj


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load all records of an export file into memory."""
    return list(iter_records(path))


def load_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load an export file into a dictionary keyed by record id.

    Records without an id (neither ``id`` nor a legacy ``metadata.path``) are
    skipped with a warning. Later duplicates win, matching the behaviour of
    incremental runs.
    """
    index: dict[str, dict[str, Any]] = {}
    skipped = 0
    for record in iter_records(path):
        key = record_id(record)
        if key:
            index[key] = record
        else:
            skipped += 1
    if skipped:
        logger.warning("Skipped %d record(s) without id in %s", skipped, path)
    return index


def write_export(
    path: str | Path,
    records: Sequence[dict[str, Any]],
    embedding_space: Optional[EmbeddingSpace] = None,
    fmt: Optional[str] = None,
) -> None:
    """Write records (with a header) to a new export file.

    The file is created (or fully rewritten); parent directories are created
    as needed. A format header is always written; pass
    ``embedding_space=None`` when the space is unknown (e.g. when merging
    legacy files).
    """
    path = Path(path)
    fmt = fmt or infer_format(path)
    if fmt not in ("jsonl", "pkl"):
        raise ValueError(f"Unsupported export format: {fmt!r}")
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(make_header(embedding_space)) + "\n")
            for record in records:
                handle.write(json.dumps(record) + "\n")
    else:
        with path.open("wb") as handle:
            pickle.dump(make_header(embedding_space), handle)
            for record in records:
                pickle.dump(record, handle)


def header_embedding_space(header: Optional[dict[str, Any]]) -> Optional[EmbeddingSpace]:
    """Extract the embedding space manifest from a header, if present."""
    if not header:
        return None
    space = header.get("embedding_space")
    if isinstance(space, dict) and "model" in space:
        return EmbeddingSpace.from_dict(space)
    return None
