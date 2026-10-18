"""Shared, dependency-light helpers: filesystem scanning, image loading,
metadata (EXIF) extraction, thumbnails and incremental-run planning.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}

_HEIF_REGISTERED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a compact, consistent format."""
    log_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def register_heif_support() -> None:
    """Register the Pillow HEIF opener once, supporting HEIC/HEIF images."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:  # pragma: no cover - pillow-heif is a core dependency
        logger.debug("pillow-heif is not installed; HEIC/HEIF images will not be supported")
    _HEIF_REGISTERED = True


def find_image_files(directories: list[str | os.PathLike[str]]) -> Iterator[str]:
    """Recursively yield the image files found in the given directories.

    Non-existent or non-directory inputs are skipped with a warning. Yields
    absolute paths.
    """
    for directory in directories:
        directory = os.fspath(directory)
        if not os.path.isdir(directory):
            logger.warning("Input path is not a directory, skipping: %s", directory)
            continue
        logger.debug("Scanning directory: %s", directory)
        for root, _dirs, files in os.walk(directory):
            for name in sorted(files):
                if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield os.path.abspath(os.path.join(root, name))


def sanitize_filename(filename: str) -> str:
    """Remove characters that are invalid in file names."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)


def resolve_torch_device(torch: Any, device_choice: str = "auto", dtype_choice: Optional[str] = None):
    """Resolve ``(device, dtype)`` from a device choice string.

    ``"auto"`` prefers CUDA, then MPS, then CPU. Half precision is selected
    automatically on accelerated devices unless ``dtype_choice`` overrides it.
    """
    choice = str(device_choice or "auto").lower()
    if choice == "auto":
        if torch.cuda.is_available():
            choice = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            choice = "mps"
        else:
            choice = "cpu"

    device = torch.device(choice)
    if dtype_choice:
        dtype = getattr(torch, str(dtype_choice))
    elif device.type in ("cuda", "mps"):
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def load_image(item: Any) -> Any:
    """Load an image from a path, bytes, a file-like object or a PIL image.

    Returns an RGB ``PIL.Image.Image``. Raises ``ValueError``/``OSError`` for
    unreadable input.
    """
    from PIL import Image

    register_heif_support()

    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(item))).convert("RGB")
    if isinstance(item, (str, os.PathLike)):
        return Image.open(os.fspath(item)).convert("RGB")
    if hasattr(item, "read"):  # file-like object
        return Image.open(item).convert("RGB")
    raise TypeError(f"Unsupported image input of type {type(item).__name__}")


def create_thumbnail(
    image_path: str,
    image: Any,
    thumbnail_dir: str,
    thumbnail_size: tuple[int, int] = (128, 128),
) -> Optional[str]:
    """Save a WEBP thumbnail next to the configured thumbnail directory.

    Returns the absolute thumbnail path, or ``None`` when creation fails.
    """
    try:
        thumb = image.copy()
        thumb.thumbnail(tuple(thumbnail_size))
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        thumb_path = os.path.join(thumbnail_dir, f"{sanitize_filename(base_name)}.webp")
        thumb.save(thumb_path, "WEBP", quality=80)
        return os.path.abspath(thumb_path)
    except Exception as exc:  # noqa: BLE001 - thumbnails are best-effort
        logger.warning("Could not create thumbnail for %s: %s", image_path, exc)
        return None


def extract_metadata(
    image_path: str,
    *,
    create_thumbnails: bool = False,
    thumbnail_dir: str = "thumbnails",
    thumbnail_size: tuple[int, int] = (128, 128),
) -> Optional[dict[str, Any]]:
    """Extract filesystem/EXIF metadata (and optionally a thumbnail) for an image.

    Returns ``None`` when the image cannot be read. The metadata schema is
    stable and part of the index format: ``path``, ``filename``, ``size_bytes``,
    ``creation_time``, ``modification_time``, ``width``, ``height``, ``exif``
    and (optionally) ``thumbnail_path``.
    """
    try:
        from PIL import ExifTags, Image

        register_heif_support()
        with Image.open(image_path) as img:
            stat = os.stat(image_path)
            metadata: dict[str, Any] = {
                "path": os.path.abspath(image_path),
                "filename": os.path.basename(image_path),
                "size_bytes": stat.st_size,
                "creation_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modification_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "width": img.width,
                "height": img.height,
                "exif": {},
            }

            if hasattr(img, "_getexif"):
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        try:
                            if isinstance(value, bytes):
                                value = value.decode("utf-8", errors="replace").strip("\x00")
                            elif not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                                value = str(value)
                            json.dumps(value)
                            metadata["exif"][str(tag)] = value
                        except (TypeError, OverflowError):
                            metadata["exif"][str(tag)] = str(value)

            if create_thumbnails:
                metadata["thumbnail_path"] = create_thumbnail(
                    image_path, img, thumbnail_dir, thumbnail_size
                )

        return metadata
    except Exception as exc:  # noqa: BLE001 - a single bad image must not stop the run
        logger.warning("Could not extract metadata for %s: %s", image_path, exc)
        return None


def filter_files_for_update(
    all_files_on_disk: set[str],
    existing: dict[str, dict[str, Any]],
) -> set[str]:
    """Plan an incremental run.

    Files to process are the files on disk that are either new (not in the
    export) or whose modification time differs from the recorded one by more
    than one second. Records without metadata (metadata was not exported)
    cannot be compared, so only new files are detected for them. Missing or
    unparseable timestamps cause reprocessing.
    """
    files_to_process: set[str] = set()
    existing_ids = set(existing)

    files_to_process.update(all_files_on_disk - existing_ids)

    for path in existing_ids & all_files_on_disk:
        metadata = existing[path].get("metadata")
        if not isinstance(metadata, dict) or not metadata:
            continue  # nothing to compare: metadata was not exported
        try:
            disk_mtime = os.path.getmtime(path)
            recorded_mtime = metadata.get("modification_time")
            if not recorded_mtime:
                files_to_process.add(path)
                continue
            recorded = datetime.fromisoformat(recorded_mtime).timestamp()
            if abs(float(disk_mtime) - recorded) > 1:
                files_to_process.add(path)
        except Exception as exc:  # noqa: BLE001 - be conservative and reprocess
            logger.warning(
                "Could not compare modification time for %s (%s); re-processing", path, exc
            )
            files_to_process.add(path)

    logger.info("Found %d new or modified file(s) to process.", len(files_to_process))
    return files_to_process
