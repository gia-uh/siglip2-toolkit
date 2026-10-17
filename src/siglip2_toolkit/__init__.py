"""siglip2-toolkit — local-first toolkit around Google's SigLIP 2 models.

The package is deliberately lazy: importing :mod:`siglip2_toolkit` does not
import torch, transformers or FastAPI. Heavy modules are loaded on first
attribute access, so tooling and metadata inspection stay fast::

    from siglip2_toolkit import Embedder

    embedder = Embedder(model="google/siglip2-base-patch16-naflex", device="auto")
    vectors = embedder.embed_texts(["a wedding at sunset"])
"""

from importlib import import_module

try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("siglip2-toolkit")
except Exception:  # pragma: no cover - source checkout without installed metadata
    __version__ = "0.0.1"

#: Lazy attributes: name -> (relative module, attribute or None for the module itself)
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    # Public API
    "Embedder": (".embedder", "Embedder"),
    "Exporter": (".exporter", "Exporter"),
    "ExportConfig": (".exporter", "ExportConfig"),
    "ExportResult": (".exporter", "ExportResult"),
    "EmbeddingSpace": (".formats", "EmbeddingSpace"),
    "ImageClassifier": (".classification", "ImageClassifier"),
    "Filter": (".filters", "Filter"),
    "field": (".filters", "field"),
    # Submodules
    "formats": (".formats", None),
    "retrieval": (".retrieval", None),
    "filters": (".filters", None),
    "utils": (".utils", None),
    "embedder": (".embedder", None),
    "exporter": (".exporter", None),
    "classification": (".classification", None),
    "cli": (".cli", None),
    "server": (".server", None),
}

__all__ = [
    "Embedder",
    "Exporter",
    "ExportConfig",
    "ExportResult",
    "EmbeddingSpace",
    "ImageClassifier",
    "Filter",
    "field",
    "__version__",
]


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name, __name__)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
