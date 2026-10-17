"""Test doubles: a deterministic embedder that does not need torch."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from siglip2_toolkit.formats import EmbeddingSpace


class FakeEmbedder:
    """Deterministic stand-in for :class:`siglip2_toolkit.embedder.Embedder`.

    Produces unit-norm vectors seeded by the bytes of each image, so runs are
    reproducible and no model is required. Mirrors the public surface used by
    the indexer: ``embed_images(..., on_error=, return_indices=)`` and
    ``embedding_space()``.
    """

    def __init__(self, dimensions: int = 8, model: str = "fake/deterministic") -> None:
        self.dimensions = dimensions
        self.model_name = model
        self.calls = 0

    def embedding_space(self) -> EmbeddingSpace:
        return EmbeddingSpace(model=self.model_name, dimensions=self.dimensions)

    def _vector(self, data: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha1(data).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dimensions).astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-12)

    def embed_images(
        self,
        images,
        batch_size: int = 32,
        on_error: str = "raise",
        return_indices: bool = False,
    ):
        vectors: list[np.ndarray] = []
        kept: list[int] = []
        for index, item in enumerate(images):
            try:
                if isinstance(item, (str, Path)):
                    data = Path(item).read_bytes()
                elif isinstance(item, (bytes, bytearray, memoryview)):
                    data = bytes(item)
                else:
                    raise TypeError(f"unsupported input: {type(item).__name__}")
            except Exception as exc:
                if on_error == "raise":
                    raise ValueError(f"Could not load image at position {index}: {exc}") from exc
                continue
            vectors.append(self._vector(data))
            kept.append(index)

        matrix = (
            np.stack(vectors) if vectors else np.zeros((0, self.dimensions), dtype=np.float32)
        )
        self.calls += 1
        if return_indices:
            return matrix, kept
        return matrix
