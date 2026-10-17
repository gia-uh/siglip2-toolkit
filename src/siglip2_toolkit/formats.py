"""Index file format (v2): specification and read/write helpers.

An index file is a sequence of objects, optionally preceded by a header::

    {"format": "siglip2-index", "version": 2,
     "embedding_space": {"model": "...", "dimensions": 768, "normalization": "l2"},
     "created_at": "2026-10-17T12:00:00+00:00"}
    {"vector": [...], "metadata": {"path": "/abs/path.jpg", ...}}
    {"vector": [...], "metadata": {"path": "/abs/path2.jpg", ...}}
    ...

``jsonl`` files store one object per line; ``pkl`` files store a sequence of
pickled objects. Record-only files produced before format v2 (a.k.a. v1) are
still readable: they are treated as records with an unknown embedding space.

The record id is ``metadata["path"]`` (the absolute path of the image).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

FORMAT_NAME = "siglip2-index"
FORMAT_VERSION = 2

JSONL_SUFFIXES = (".jsonl", ".jsonlines", ".ndjson")
PKL_SUFFIXES = (".pkl", ".pickle")


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EmbeddingSpace:
    """Identity of the vector space produced by an embedder.

    Two vectors are only comparable when they were produced by the same model
    (and the same preprocessing). This manifest is stored in the index header
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
    """Build the index header object."""
    header: dict[str, Any] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "created_at": created_at or now_iso(),
    }
    if embedding_space is not None:
        header["embedding_space"] = embedding_space.to_dict()
    return header


def is_header(obj: Any) -> bool:
    """Whether an object read from an index file is a format header."""
    return isinstance(obj, dict) and obj.get("format") == FORMAT_NAME


def make_record(vector: Sequence[float], metadata: dict[str, Any]) -> dict[str, Any]:
    """Build an index record from a vector and its metadata."""
    return {"vector": list(vector), "metadata": dict(metadata)}


def record_path(record: dict[str, Any]) -> Optional[str]:
    """The record id: the absolute path stored in ``metadata["path"]``."""
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        path = metadata.get("path")
        if isinstance(path, str) and path:
            return path
    return None


_TEMP_SUFFIXES = (".tmp", ".part")


def infer_format(path: str | Path) -> str:
    """Infer the index format (``jsonl`` or ``pkl``) from a file name.

    Temporary suffixes (``.tmp``/``.part``) are stripped first, so the
    interrupted-run convention ``<index>.jsonl.tmp`` also resolves.
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
        f"Cannot infer index format from '{path}'. "
        f"Use one of {JSONL_SUFFIXES + PKL_SUFFIXES}."
    )


def read_header(path: str | Path) -> Optional[dict[str, Any]]:
    """Read the index header, or ``None`` for legacy (v1) or missing files."""
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
    """Yield index records, skipping the header if present."""
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
    """Load all records of an index file into memory."""
    return list(iter_records(path))


def load_index_map(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load an index file into a dictionary keyed by record id (image path).

    Later duplicates win, matching the historical behaviour of the importer
    and indexer tooling.
    """
    index: dict[str, dict[str, Any]] = {}
    for record in iter_records(path):
        key = record_path(record)
        if key:
            index[key] = record
    return index


def write_index(
    path: str | Path,
    records: Sequence[dict[str, Any]],
    embedding_space: Optional[EmbeddingSpace] = None,
    fmt: Optional[str] = None,
) -> None:
    """Write records (with an optional header) to a new index file.

    The file is created (or fully rewritten); parent directories are created
    as needed. A format header is always written; pass
    ``embedding_space=None`` when the space is unknown (e.g. when merging
    legacy files).
    """
    path = Path(path)
    fmt = fmt or infer_format(path)
    if fmt not in ("jsonl", "pkl"):
        raise ValueError(f"Unsupported index format: {fmt!r}")
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
