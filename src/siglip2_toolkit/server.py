"""Optional REST server for embeddings (requires the ``server`` extra).

The server is a thin adapter over `Embedder`, so third
parties using the library programmatically see exactly the same behaviour as
REST clients. ``create_app()`` also allows mounting the API inside an
existing FastAPI application.
"""

# NOTE: deliberately no ``from __future__ import annotations`` in this module:
# FastAPI resolves route annotations at runtime, and the FastAPI symbols are
# imported inside ``create_app`` (the [server] extra is optional).

import json
import logging
from typing import Any, Optional

from . import models
from .embedder import Embedder

logger = logging.getLogger(__name__)

DEFAULT_MAX_BATCH = 256
_SERVER_EXTRA_HINT = (
    "The REST server requires the [server] extra:\n"
    "    pip install 'siglip2-toolkit[server]'"
)


def _parse_labels(raw: str) -> list[str]:
    """Accept candidate labels as a JSON array or a comma-separated list."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            values = None
        if isinstance(values, list):
            return [str(value).strip() for value in values if str(value).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def create_app(
    embedder: Optional[Embedder] = None,
    *,
    model: Optional[str] = None,
    device: Optional[str] = None,
    max_batch_size: int = DEFAULT_MAX_BATCH,
    home: Optional[str] = None,
) -> Any:
    """Build the FastAPI application.

    The model is loaded lazily on the first embedding request; ``/v1/health``
    and ``/v1/info`` never trigger a load.
    """
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from pydantic import BaseModel
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:
        raise ModuleNotFoundError(_SERVER_EXTRA_HINT) from exc

    from . import __version__

    if embedder is None:
        resolved = models.resolve_model(model, home=home)
        embedder = Embedder(
            model=resolved.path,
            model_id=resolved.identifier,
            device=device or models.default_device(home),
        )
    state = {"embedder": embedder}
    app = FastAPI(
        title="siglip2-toolkit",
        version=__version__,
        description=(
            "Local-first SigLIP 2 embeddings and classification. "
            "The model is loaded lazily on the first embedding request. "
            "Interactive docs are available at /docs."
        ),
    )

    class TextRequest(BaseModel):
        texts: list[str]

    def _vector_response(matrix: Any) -> dict[str, Any]:
        active = state["embedder"]
        return {
            "model": active.model_name,
            "dimensions": active.dimensions,
            "vectors": matrix.tolist(),
        }

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        """Liveness and model state; never loads the model."""
        active = state["embedder"]
        return {
            "status": "ok",
            "model": active.model_name,
            "loaded": active.is_loaded,
            "device": active.device,
        }

    @app.get("/v1/info")
    async def info() -> dict[str, Any]:
        """Model and embedding-space metadata; dimensions may be null until the first load."""
        active = state["embedder"]
        space = active.embedding_space() if active.is_loaded else None
        return {
            "model": active.model_name,
            "dimensions": active.dimensions,
            "embedding_space": space.to_dict() if space else None,
        }

    @app.post("/v1/embed/text")
    async def embed_text(request: TextRequest) -> dict[str, Any]:
        """Embed one or more texts and return L2-normalised float32 vectors."""
        texts = [text.strip() for text in request.texts if text and text.strip()]
        if not texts:
            raise HTTPException(status_code=400, detail="Provide at least one non-empty text")
        if len(texts) > max_batch_size:
            raise HTTPException(status_code=413, detail=f"Too many texts (max {max_batch_size})")
        matrix = await run_in_threadpool(state["embedder"].embed_texts, texts)
        return _vector_response(matrix)

    @app.post("/v1/embed/image")
    async def embed_image(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        """Embed one or more uploaded images (multipart field ``files``)."""
        if not files:
            raise HTTPException(status_code=400, detail="Upload at least one image file")
        if len(files) > max_batch_size:
            raise HTTPException(status_code=413, detail=f"Too many files (max {max_batch_size})")

        payloads = []
        for upload in files:
            if upload.content_type and not upload.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"Not an image: {upload.filename}")
            payloads.append(await upload.read())

        try:
            matrix = await run_in_threadpool(state["embedder"].embed_images, payloads)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _vector_response(matrix)

    @app.post("/v1/classify")
    async def classify_images(
        files: list[UploadFile] = File(...),
        labels: str = Form(...),
        template: str = Form("{}"),
        top_k: Optional[int] = Form(None),
    ) -> dict[str, Any]:
        """Zero-shot classification with candidate labels (JSON array or comma-separated)."""
        parsed_labels = _parse_labels(labels)
        if not parsed_labels:
            raise HTTPException(
                status_code=400,
                detail="Provide candidate labels as a JSON array or a comma-separated list",
            )
        if not files:
            raise HTTPException(status_code=400, detail="Upload at least one image file")
        if len(files) > max_batch_size:
            raise HTTPException(status_code=413, detail=f"Too many files (max {max_batch_size})")

        payloads = []
        for upload in files:
            if upload.content_type and not upload.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"Not an image: {upload.filename}")
            payloads.append(await upload.read())

        try:
            results = await run_in_threadpool(
                state["embedder"].classify,
                payloads,
                parsed_labels,
                template=template,
                top_k=top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "model": state["embedder"].model_name,
            "labels": parsed_labels,
            "results": [
                [{"label": label, "score": score} for label, score in row] for row in results
            ],
        }

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    model: Optional[str] = None,
    device: Optional[str] = None,
    home: Optional[str] = None,
) -> None:
    """Create the app and serve it with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ModuleNotFoundError(_SERVER_EXTRA_HINT) from exc

    app = create_app(model=model, device=device, home=home)
    uvicorn.run(app, host=host, port=port)
