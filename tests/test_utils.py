"""Unit tests for filesystem and metadata helpers."""

import os
import time
from datetime import datetime
from pathlib import Path

from siglip2_toolkit import formats, utils


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


def test_filter_files_for_update_detects_new_and_modified(tmp_path):
    first = tmp_path / "a.jpg"
    first.write_bytes(b"x")
    second = tmp_path / "b.jpg"
    second.write_bytes(b"y")

    stamp = datetime.fromtimestamp(first.stat().st_mtime).isoformat()
    existing = {str(first): {"metadata": {"modification_time": stamp}}}

    to_process = utils.filter_files_for_update({str(first), str(second)}, existing)
    assert to_process == {str(second)}


def test_filter_files_for_update_skips_records_without_metadata(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    future = time.time() + 10
    os.utime(image, (future, future))

    existing = {str(image): formats.make_record([1.0], id=str(image))}
    assert utils.filter_files_for_update({str(image)}, existing) == set()


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


def test_list_cached_models_filters_and_counts_sizes(tmp_path):
    cache = tmp_path / "hub"
    siglip = cache / "models--google--siglip2-base-patch16-naflex"
    (siglip / "blobs").mkdir(parents=True)
    (siglip / "snapshots" / "abc").mkdir(parents=True)
    (siglip / "blobs" / "model.bin").write_bytes(b"x" * 1000)
    (siglip / "snapshots" / "abc" / "model.bin").symlink_to(siglip / "blobs" / "model.bin")

    other = cache / "models--Qwen--Qwen3"
    (other / "blobs").mkdir(parents=True)
    (other / "blobs" / "w.bin").write_bytes(b"y" * 10)

    only_siglip = utils.list_cached_models(cache_dir=cache)
    assert [m.repo_id for m in only_siglip] == ["google/siglip2-base-patch16-naflex"]
    assert only_siglip[0].size_bytes == 1000  # blobs only: snapshots are symlinks

    everything = utils.list_cached_models(include_all=True, cache_dir=cache)
    assert {m.repo_id for m in everything} == {
        "google/siglip2-base-patch16-naflex",
        "Qwen/Qwen3",
    }


def test_is_model_cached(tmp_path):
    cache = tmp_path / "hub"
    entry = cache / "models--google--siglip2-base-patch16-naflex"
    (entry / "snapshots" / "abc").mkdir(parents=True)

    assert utils.is_model_cached("google/siglip2-base-patch16-naflex", cache_dir=cache)
    assert not utils.is_model_cached("google/siglip2-so400m-patch16-naflex", cache_dir=cache)


def test_default_hub_cache_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "direct"))
    assert utils.default_hub_cache() == tmp_path / "direct"

    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert utils.default_hub_cache() == tmp_path / "home" / "hub"
