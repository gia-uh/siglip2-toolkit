"""Unit tests for the embeddings export format (no model, no torch)."""

import logging
from pathlib import Path

import pytest

from siglip2_toolkit import formats


def _records():
    return [
        formats.make_record([0.1, 0.2], {"path": "/a/x.jpg", "width": 10}),
        formats.make_record([0.3, 0.4], {"path": "/a/y.jpg", "width": 20}),
    ]


@pytest.mark.parametrize("suffix", [".jsonl", ".pkl"])
def test_write_read_roundtrip(tmp_path, suffix):
    path = tmp_path / f"export{suffix}"
    space = formats.EmbeddingSpace(model="test/model", dimensions=2)
    formats.write_export(path, _records(), embedding_space=space)

    header = formats.read_header(path)
    assert header["format"] == formats.FORMAT_NAME
    assert header["version"] == formats.FORMAT_VERSION
    assert formats.header_embedding_space(header) == space
    assert formats.load_records(path) == _records()


def test_make_record_with_and_without_metadata():
    with_id = formats.make_record([1.0], id="/a.jpg")
    assert with_id == {"id": "/a.jpg", "vector": [1.0]}
    assert formats.record_id(with_id) == "/a.jpg"

    bare = formats.make_record([1.0])
    assert bare == {"vector": [1.0]}
    assert formats.record_id(bare) is None

    with_metadata = formats.make_record([1.0], {"width": 10}, id="x")
    assert with_metadata == {"id": "x", "vector": [1.0], "metadata": {"width": 10}}


def test_record_id_prefers_top_level_id_with_legacy_fallback():
    explicit = {"id": "/explicit.jpg", "vector": [1.0], "metadata": {"path": "/legacy.jpg"}}
    assert formats.record_id(explicit) == "/explicit.jpg"

    legacy = {"vector": [1.0], "metadata": {"path": "/legacy.jpg"}}
    assert formats.record_id(legacy) == "/legacy.jpg"

    assert formats.record_id({"vector": [1.0]}) is None
    assert formats.record_id({"metadata": {"path": ""}}) is None


def test_legacy_record_only_file_is_readable(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text('{"vector": [1.0], "metadata": {"path": "/a.jpg"}}\n', encoding="utf-8")
    assert formats.read_header(path) is None
    assert set(formats.load_by_id(path)) == {"/a.jpg"}
    assert len(formats.load_records(path)) == 1


def test_infer_format_handles_temp_suffixes():
    assert formats.infer_format("export.jsonl.tmp") == "jsonl"
    assert formats.infer_format("export.pkl.part") == "pkl"
    assert formats.infer_format(Path("EXPORT.PKL")) == "pkl"
    with pytest.raises(ValueError):
        formats.infer_format("export.nope")


def test_load_by_id_keys_and_last_wins(tmp_path):
    path = tmp_path / "export.jsonl"
    records = [
        formats.make_record([0.1], {"width": 1}, id="/a.jpg"),
        formats.make_record([0.2], {"width": 2}, id="/b.jpg"),
        formats.make_record([0.9], {"width": 99}, id="/a.jpg"),
    ]
    formats.write_export(path, records)

    loaded = formats.load_by_id(path)
    assert set(loaded) == {"/a.jpg", "/b.jpg"}
    assert loaded["/a.jpg"]["metadata"]["width"] == 99


def test_load_by_id_skips_records_without_id(tmp_path, caplog):
    path = tmp_path / "export.jsonl"
    formats.write_export(path, [formats.make_record([1.0])])
    with caplog.at_level(logging.WARNING):
        assert formats.load_by_id(path) == {}
    assert "without id" in caplog.text


def test_write_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dirs" / "export.jsonl"
    formats.write_export(path, _records())
    assert path.exists()


def test_empty_export_has_header_without_space(tmp_path):
    path = tmp_path / "export.jsonl"
    formats.write_export(path, [], embedding_space=None)
    header = formats.read_header(path)
    assert header is not None
    assert formats.header_embedding_space(header) is None
    assert formats.load_records(path) == []


def test_embedding_space_id_and_compatibility():
    a = formats.EmbeddingSpace(model="m", dimensions=4)
    b = formats.EmbeddingSpace(model="m", dimensions=4)
    c = formats.EmbeddingSpace(model="m", dimensions=5)
    other = formats.EmbeddingSpace(model="n", dimensions=4)

    assert a.id == b.id
    assert a.id != c.id
    assert a.compatible_with(b)
    assert not a.compatible_with(c)
    assert not a.compatible_with(other)
