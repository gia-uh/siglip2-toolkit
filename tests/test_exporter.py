"""Exporter tests with a deterministic fake embedder (no model, no torch)."""

import os
import shutil
import time
from pathlib import Path

from siglip2_toolkit import formats
from siglip2_toolkit.exporter import ExportConfig, Exporter

from fakes import FakeEmbedder


def _dataset(tmp_path: Path, image_paths: list[Path]) -> Path:
    """Copy the test images into a scratch directory (tests may mutate them)."""
    data_dir = tmp_path / "photos"
    data_dir.mkdir()
    for path in image_paths:
        shutil.copy2(path, data_dir / path.name)
    return data_dir


def _config(tmp_path: Path, input_dir: Path, **overrides) -> ExportConfig:
    config = ExportConfig(
        input_dirs=[str(input_dir)],
        output_file=str(tmp_path / "export.jsonl"),
        model="fake/deterministic",
        batch_size=4,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_export_creates_one_record_per_image(tmp_path, image_paths):
    data_dir = _dataset(tmp_path, image_paths)
    result = Exporter(_config(tmp_path, data_dir), embedder=FakeEmbedder()).run()

    assert result.processed == len(image_paths)
    assert result.skipped == 0
    assert result.total == len(image_paths)

    records = formats.load_records(result.output_file)
    assert len(records) == len(image_paths)
    first = records[0]
    assert first["id"].startswith(str(data_dir))
    metadata = first["metadata"]
    assert {"path", "filename", "size_bytes", "width", "height", "exif"} <= set(metadata)

    space = formats.header_embedding_space(formats.read_header(result.output_file))
    assert space.model == "fake/deterministic"
    assert space.dimensions == 8


def test_vectors_only_without_metadata(tmp_path, image_paths):
    data_dir = _dataset(tmp_path, image_paths[:2])
    config = _config(tmp_path, data_dir, store_metadata=False)
    result = Exporter(config, embedder=FakeEmbedder()).run()

    assert result.processed == 2
    for record in formats.load_records(result.output_file):
        assert "metadata" not in record
        assert "vector" in record
        assert record["id"].startswith(str(data_dir))


def test_incremental_processes_modified_and_removes_deleted(tmp_path, image_paths):
    data_dir = _dataset(tmp_path, image_paths)
    config = _config(tmp_path, data_dir)
    Exporter(config, embedder=FakeEmbedder()).run()

    # "Modify" one image: only that file must be reprocessed.
    modified = Path(sorted(str(path) for path in data_dir.glob("*.jpg"))[0])
    future = time.time() + 10
    os.utime(modified, (future, future))

    config.incremental = True
    result = Exporter(config, embedder=FakeEmbedder()).run()
    assert result.processed == 1
    assert result.removed == 0
    assert result.total == len(image_paths)

    # "Delete" one image: its record must be dropped from the export.
    hidden = modified.with_name(modified.name + ".hidden")
    modified.rename(hidden)
    try:
        result = Exporter(config, embedder=FakeEmbedder()).run()
        assert result.processed == 0
        assert result.removed == 1
        assert result.total == len(image_paths) - 1
    finally:
        hidden.rename(modified)


def test_incremental_without_metadata_only_detects_new_and_deleted(tmp_path, image_paths):
    data_dir = _dataset(tmp_path, image_paths[:3])
    config = _config(tmp_path, data_dir, store_metadata=False)
    Exporter(config, embedder=FakeEmbedder()).run()

    # Without metadata there is nothing to compare: modifications go unnoticed.
    modified = sorted(data_dir.glob("*.jpg"))[0]
    future = time.time() + 10
    os.utime(modified, (future, future))
    config.incremental = True
    result = Exporter(config, embedder=FakeEmbedder()).run()
    assert result.processed == 0
    assert result.total == 3

    # New files are detected and deleted files are dropped.
    shutil.copy2(image_paths[5], data_dir / "extra.jpg")
    hidden = modified.with_name(modified.name + ".hidden")
    modified.rename(hidden)
    try:
        result = Exporter(config, embedder=FakeEmbedder()).run()
        assert result.processed == 1  # extra.jpg
        assert result.removed == 1  # the hidden one
        assert result.total == 3
    finally:
        hidden.rename(modified)


def test_resume_from_temporary_file(tmp_path, image_paths):
    data_dir = _dataset(tmp_path, image_paths)
    config = _config(tmp_path, data_dir)
    Exporter(config, embedder=FakeEmbedder()).run()

    output = Path(config.output_file)
    temp = Path(str(output) + ".tmp")
    records = formats.load_records(output)
    space = formats.header_embedding_space(formats.read_header(output))

    # Simulate an interrupted run: a .tmp with all but the last record.
    formats.write_export(temp, records[:-1], embedding_space=space)
    output.unlink()

    result = Exporter(config, embedder=FakeEmbedder()).run()
    assert result.processed == 1
    assert result.total == len(records)
    assert not temp.exists()
    assert output.exists()


def test_thumbnails_are_created(tmp_path, image_paths):
    data_dir = _dataset(tmp_path, image_paths[:2])
    thumbnail_dir = tmp_path / "thumbs"
    config = _config(
        tmp_path,
        data_dir,
        create_thumbnails=True,
        thumbnail_dir=str(thumbnail_dir),
    )
    result = Exporter(config, embedder=FakeEmbedder()).run()

    assert result.processed == 2
    for record in formats.load_records(result.output_file):
        thumbnail = record["metadata"].get("thumbnail_path")
        assert thumbnail is not None
        assert Path(thumbnail).exists()
        assert Path(thumbnail).suffix == ".webp"


def test_corrupted_images_are_skipped(tmp_path, image_paths):
    data_dir = _dataset(tmp_path, image_paths[:2])
    (data_dir / "broken.jpg").write_bytes(b"this is not an image")

    result = Exporter(_config(tmp_path, data_dir), embedder=FakeEmbedder()).run()
    assert result.processed == 2
    assert result.skipped == 1
    assert result.total == 2


def test_empty_input_directory_produces_empty_export(tmp_path, image_paths):
    data_dir = _dataset(tmp_path, [])
    result = Exporter(_config(tmp_path, data_dir), embedder=FakeEmbedder()).run()

    assert result.processed == 0
    assert result.total == 0
    assert Path(result.output_file).exists()
    assert formats.load_records(result.output_file) == []


def test_progress_callback_reports_batches(tmp_path, image_paths):
    data_dir = _dataset(tmp_path, image_paths[:3])
    events: list[tuple[int, int, str]] = []
    stages: list[str] = []

    Exporter(
        _config(tmp_path, data_dir, batch_size=2),
        embedder=FakeEmbedder(),
        on_progress=lambda done, total, path: events.append((done, total, path)),
        on_stage=stages.append,
    ).run()

    assert stages == ["scanning", "embedding", "done"]
    assert events[-1][0] == events[-1][1] == 3
