"""SigLIP 2 embedder: L2-normalised text and image embeddings.

The model is loaded lazily on first use (``load()`` is called automatically by
every ``embed_*`` method), so importing this module and constructing an
:class:`Embedder` is cheap.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from . import utils
from .defaults import DEFAULT_MODEL
from .formats import EmbeddingSpace

logger = logging.getLogger(__name__)

#: Path, bytes, file-like object or an already-open PIL image.
ImageInput = Union[str, Path, bytes, Any]


class Embedder:
    """Generates text and image embeddings with a SigLIP 2 model.

    Embeddings are L2-normalised ``float32`` numpy arrays. The instance is
    safe to share across threads (inference is serialised with an internal
    lock); it is not safe to share across processes.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model
        self.requested_device = device
        self.requested_dtype = dtype
        self.revision = revision
        self.trust_remote_code = trust_remote_code

        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._dimensions: Optional[int] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> Optional[str]:
        """Device string (e.g. ``"cuda"``/``"cpu"``), or ``None`` if not loaded."""
        return str(self._device) if self._device is not None else None

    @property
    def dimensions(self) -> Optional[int]:
        """Embedding dimension, or ``None`` if the model has not been loaded."""
        return self._dimensions

    def load(self) -> "Embedder":
        """Load the model and processor (idempotent)."""
        if self._model is not None:
            return self

        import torch
        from transformers import AutoModel, AutoProcessor

        device, dtype = self._resolve_device(torch)
        logger.info("Loading '%s' on %s (%s)...", self.model_name, device, dtype)
        # NOTE: torch_dtype= keeps compatibility with transformers 4.51-4.55;
        # newer versions prefer dtype= and only emit a deprecation warning.
        model = AutoModel.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=dtype,
        )
        model = model.to(device).eval()
        processor = AutoProcessor.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )

        self._torch = torch
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype

        # Probe the embedding dimension with a tiny forward pass.
        probe = self._embed_text_batch(["siglip2-toolkit"], max_length=8)
        self._dimensions = int(probe.shape[1])
        logger.info("Model ready: %d dimensions ✅", self._dimensions)
        return self

    def _resolve_device(self, torch: Any) -> tuple[Any, Any]:
        choice = str(self.requested_device or "auto").lower()
        if choice == "auto":
            if torch.cuda.is_available():
                choice = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                choice = "mps"
            else:
                choice = "cpu"

        device = torch.device(choice)
        if self.requested_dtype:
            dtype = getattr(torch, self.requested_dtype)
        elif device.type in ("cuda", "mps"):
            dtype = torch.float16
        else:
            dtype = torch.float32
        return device, dtype

    def embedding_space(self) -> EmbeddingSpace:
        """The embedding space manifest for the loaded model."""
        self.load()
        return EmbeddingSpace(
            model=self.model_name,
            dimensions=self._dimensions,
            revision=self.revision,
        )

    # ------------------------------------------------------------------ #
    # Text embeddings
    # ------------------------------------------------------------------ #

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        max_length: int = 64,
    ) -> np.ndarray:
        """Embed a batch of texts. Returns a ``(n, dimensions)`` float32 array."""
        self.load()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self._dimensions or 0), dtype=np.float32)

        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = [text if isinstance(text, str) else str(text) for text in texts[start : start + batch_size]]
            chunks.append(self._embed_text_batch(batch, max_length=max_length))
        return np.concatenate(chunks, axis=0)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text. Returns a ``(dimensions,)`` float32 array."""
        return self.embed_texts([text])[0]

    def _embed_text_batch(self, texts: list[str], max_length: int = 64) -> np.ndarray:
        torch = self._torch
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).to(self._device)
        with self._lock, torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return self._normalize(features)

    # ------------------------------------------------------------------ #
    # Image embeddings
    # ------------------------------------------------------------------ #

    def embed_images(
        self,
        images: Sequence[ImageInput],
        batch_size: int = 32,
        on_error: str = "raise",
        return_indices: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, list[int]]]:
        """Embed a batch of images.

        ``images`` items may be paths, bytes, file-like objects or PIL images.
        With ``on_error="skip"`` unreadable items are logged and skipped; the
        kept positions are returned alongside the matrix when
        ``return_indices=True``.
        """
        if on_error not in ("raise", "skip"):
            raise ValueError("on_error must be 'raise' or 'skip'")
        self.load()
        images = list(images)

        loaded: list[Any] = []
        kept_indices: list[int] = []
        for index, item in enumerate(images):
            try:
                loaded.append(utils.load_image(item))
                kept_indices.append(index)
            except Exception as exc:  # noqa: BLE001 - per-item robustness
                if on_error == "raise":
                    raise ValueError(f"Could not load image at position {index}: {exc}") from exc
                logger.warning("Skipping unreadable image at position %d: %s", index, exc)

        chunks = []
        for start in range(0, len(loaded), batch_size):
            chunks.append(self._embed_image_batch(loaded[start : start + batch_size]))
        if chunks:
            matrix = np.concatenate(chunks, axis=0)
        else:
            matrix = np.zeros((0, self._dimensions or 0), dtype=np.float32)

        if return_indices:
            return matrix, kept_indices
        return matrix

    def embed_image(self, image: ImageInput) -> np.ndarray:
        """Embed a single image. Returns a ``(dimensions,)`` float32 array."""
        return self.embed_images([image])[0]

    def _embed_image_batch(self, images: list[Any]) -> np.ndarray:
        torch = self._torch
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with self._lock, torch.no_grad():
            features = self._model.get_image_features(**inputs)
        return self._normalize(features)

    # ------------------------------------------------------------------ #

    def _normalize(self, features: Any) -> np.ndarray:
        torch = self._torch
        normalized = features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        return normalized.detach().cpu().to(torch.float32).numpy()

    def __repr__(self) -> str:
        return (
            f"Embedder(model={self.model_name!r}, "
            f"device={self.requested_device!r}, loaded={self.is_loaded})"
        )
