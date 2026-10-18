"""Fine-tuned image classification with SigLIP-family checkpoints.

Zero-shot classification (prompt-based) lives on the embedder:
:meth:`siglip2_toolkit.embedder.Embedder.classify`. This module covers
checkpoints that ship a trained classification head, consumed through
``AutoModelForImageClassification`` (e.g. ``SiglipForImageClassification``).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import utils
from .embedder import ImageInput, softmax_rows

logger = logging.getLogger(__name__)


class ImageClassifier:
    """Classifies images with a fine-tuned image-classification model.

    The model and its image processor are loaded lazily on first use. Labels
    come from the checkpoint's ``config.id2label``; ``num_labels`` overrides
    the head size when the checkpoint requires it.
    """

    def __init__(
        self,
        model: str | Path,
        device: str = "auto",
        dtype: Optional[str] = None,
        num_labels: Optional[int] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_path = str(Path(model).expanduser())
        self.model_name = self.model_path
        self.requested_device = device
        self.requested_dtype = dtype
        self.num_labels = num_labels
        self.trust_remote_code = trust_remote_code

        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._id2label: dict[int, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def labels(self) -> dict[int, str]:
        """``id -> label`` mapping (empty until the model is loaded)."""
        return dict(self._id2label)

    @property
    def device(self) -> Optional[str]:
        return str(self._device) if self._device is not None else None

    def load(self) -> "ImageClassifier":
        """Load the model and its image processor from a local directory (idempotent)."""
        if self._model is not None:
            return self

        path = Path(self.model_path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"ImageClassifier expects a local model directory, got: "
                f"'{self.model_path}'. Get one with `siglip2 models download` or "
                "`siglip2 models import`, or pass the path to one."
            )

        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        device, dtype = utils.resolve_torch_device(
            torch, self.requested_device, self.requested_dtype
        )
        logger.info("Loading classifier '%s' on %s (%s)...", self.model_name, device, dtype)

        extra: dict[str, Any] = {}
        if self.num_labels is not None:
            extra["num_labels"] = self.num_labels

        try:
            model = AutoModelForImageClassification.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=dtype,
                **extra,
            )
            processor = AutoImageProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception as exc:
            raise OSError(
                f"Could not load a classifier from '{self.model_path}': {exc}"
            ) from exc
        model = model.to(device).eval()

        self._torch = torch
        self._model = model
        self._processor = processor
        self._device = device
        self._id2label = {
            int(key): str(value)
            for key, value in (getattr(model.config, "id2label", None) or {}).items()
        }
        logger.info("Classifier ready: %d label(s) ✅", len(self._id2label))
        return self

    # ------------------------------------------------------------------ #

    def classify(
        self,
        images: Sequence[ImageInput],
        top_k: Optional[int] = None,
        batch_size: int = 16,
    ) -> list[list[tuple[str, float]]]:
        """Classify images; returns (label, probability) pairs sorted by score."""
        self.load()
        images = list(images)
        results: list[list[tuple[str, float]]] = []

        for start in range(0, len(images), batch_size):
            batch = [utils.load_image(item) for item in images[start : start + batch_size]]
            if not batch:
                continue
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with self._lock, self._torch.no_grad():
                logits = self._model(**inputs).logits
            probabilities = softmax_rows(
                logits.detach().cpu().to(self._torch.float32).numpy()
            )
            for row in probabilities:
                order = np.argsort(row)[::-1]
                if top_k is not None:
                    order = order[: max(0, int(top_k))]
                results.append(
                    [
                        (self._id2label.get(int(index), f"LABEL_{int(index)}"), float(row[index]))
                        for index in order
                    ]
                )
        return results

    def __repr__(self) -> str:
        return (
            f"ImageClassifier(model={self.model_name!r}, "
            f"device={self.requested_device!r}, loaded={self.is_loaded})"
        )
