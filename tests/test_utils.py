"""Unit tests for filesystem and metadata helpers."""

from datetime import datetime
from pathlib import Path

from siglip2_toolkit import utils


def test_find_image_files_filters_extensions_and_recurses(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("not an image", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "c.png").write_bytes(b"y")

    found = sorted(Path(path).name for path in utils.find_image_files([str(tmp_path)]))
    assert found == ["a.jpg", "c.png"]


def test_find_image_files_skips_missing_directories(tmp_path):
    found = list(utils.find_image_files([str(tmp_path / "does-not-exist")]))
    assert found == []


def test_filter_files_for_reindex_detects_new_and_modified(tmp_path):
    first = tmp_path / "a.jpg"
    first.write_bytes(b"x")
    second = tmp_path / "b.jpg"
    second.write_bytes(b"y")

    stamp = datetime.fromtimestamp(first.stat().st_mtime).isoformat()
    existing = {str(first): {"metadata": {"modification_time": stamp}}}

    to_process = utils.filter_files_for_reindex({str(first), str(second)}, existing)
    assert to_process == {str(second)}


def test_extract_metadata_shape(tmp_path, image_paths):
    source = image_paths[0]
    metadata = utils.extract_metadata(str(source))

    assert metadata is not None
    assert metadata["path"] == str(source)
    assert metadata["filename"] == source.name
    assert metadata["size_bytes"] > 0
    assert metadata["width"] > 0
    assert metadata["height"] > 0
    assert isinstance(metadata["exif"], dict)
    assert "thumbnail_path" not in metadata


def test_extract_metadata_with_thumbnail(tmp_path, image_paths):
    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir()
    metadata = utils.extract_metadata(
        str(image_paths[0]),
        create_thumbnails=True,
        thumbnail_dir=str(thumb_dir),
        thumbnail_size=(64, 64),
    )

    assert metadata is not None
    thumbnail = Path(metadata["thumbnail_path"])
    assert thumbnail.exists()
    assert thumbnail.suffix == ".webp"


def test_extract_metadata_returns_none_for_unreadable_file(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"this is not an image")
    assert utils.extract_metadata(str(broken)) is None
