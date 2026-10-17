"""Optional REST server for embeddings (requires the ``server`` extra).

The server is a thin adapter over :class:`~.embedder.Embedder`, so third
parties using the library programmatically see exactly the same behaviour as
REST clients. ``create_app()`` also allows mounting the API inside an
existing FastAPI application.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .embedder import DEFAULT_MODEL, Embedder

logger = logging.getLogger(__name__)

DEFAULT_MAX_BATCH = 256
_SERVER_EXTRA_HINT = (
    "The REST server requires the [server] extra:\n"
    "    pip install 'siglip2-toolkit[server]'"
)


def create_app(
    embedder: Optional[Embedder] = None,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    max_batch_size: int = DEFAULT_MAX_BATCH,
) -> Any:
    """Build the FastAPI application.

    The model is loaded lazily on the first embedding request; ``/v1/health``
    and ``/v1/info`` never trigger a load.
    """
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from pydantic import BaseModel
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:
        raise ModuleNotFoundError(_SERVER_EXTRA_HINT) from exc

    from . import __version__

    state = {"embedder": embedder or Embedder(model=model, device=device)}
    app = FastAPI(title="siglip2-toolkit", version=__version__)

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
        active = state["embedder"]
        return {
            "status": "ok",
            "model": active.model_name,
            "loaded": active.is_loaded,
            "device": active.device,
        }

    @app.get("/v1/info")
    async def info() -> dict[str, Any]:
        active = state["embedder"]
        space = active.embedding_space() if active.is_loaded else None
        return {
            "model": active.model_name,
            "dimensions": active.dimensions,
            "embedding_space": space.to_dict() if space else None,
        }

    @app.post("/v1/embed/text")
    async def embed_text(request: TextRequest) -> dict[str, Any]:
        texts = [text.strip() for text in request.texts if text and text.strip()]
        if not texts:
            raise HTTPException(status_code=400, detail="Provide at least one non-empty text")
        if len(texts) > max_batch_size:
            raise HTTPException(status_code=413, detail=f"Too many texts (max {max_batch_size})")
        matrix = await run_in_threadpool(state["embedder"].embed_texts, texts)
        return _vector_response(matrix)

    @app.post("/v1/embed/image")
    async def embed_image(files: list[UploadFile] = File(...)) -> dict[str, Any]:
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

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
) -> None:
    """Create the app and serve it with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ModuleNotFoundError(_SERVER_EXTRA_HINT) from exc

    app = create_app(model=model, device=device)
    uvicorn.run(app, host=host, port=port)
