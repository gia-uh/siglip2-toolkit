"""Unit tests for the index file format (no model, no torch)."""

from pathlib import Path

import pytest

from siglip2_toolkit import formats


def _records():
    return [
        {"vector": [0.1, 0.2], "metadata": {"path": "/a/x.jpg", "width": 10}},
        {"vector": [0.3, 0.4], "metadata": {"path": "/a/y.jpg", "width": 20}},
    ]


@pytest.mark.parametrize("suffix", [".jsonl", ".pkl"])
def test_write_read_roundtrip(tmp_path, suffix):
    path = tmp_path / f"index{suffix}"
    space = formats.EmbeddingSpace(model="test/model", dimensions=2)
    formats.write_index(path, _records(), embedding_space=space)

    header = formats.read_header(path)
    assert header["format"] == formats.FORMAT_NAME
    assert header["version"] == formats.FORMAT_VERSION
    assert formats.header_embedding_space(header) == space
    assert formats.load_records(path) == _records()


def test_legacy_v1_file_has_no_header(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text('{"vector": [1.0], "metadata": {"path": "/a.jpg"}}\n', encoding="utf-8")
    assert formats.read_header(path) is None
    assert len(formats.load_records(path)) == 1


def test_infer_format_handles_temp_suffixes(tmp_path):
    assert formats.infer_format("index.jsonl.tmp") == "jsonl"
    assert formats.infer_format("index.pkl.part") == "pkl"
    assert formats.infer_format(Path("INDEX.PKL")) == "pkl"
    with pytest.raises(ValueError):
        formats.infer_format("index.nope")


def test_load_index_map_keys_by_path_and_last_wins(tmp_path):
    path = tmp_path / "index.jsonl"
    records = _records() + [{"vector": [9.0], "metadata": {"path": "/a/x.jpg", "width": 99}}]
    formats.write_index(path, records)

    index_map = formats.load_index_map(path)
    assert set(index_map) == {"/a/x.jpg", "/a/y.jpg"}
    assert index_map["/a/x.jpg"]["metadata"]["width"] == 99


def test_write_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dirs" / "index.jsonl"
    formats.write_index(path, _records())
    assert path.exists()


def test_make_record_and_record_path():
    record = formats.make_record([0.5], {"path": "/a.jpg"})
    assert formats.record_path(record) == "/a.jpg"
    assert formats.record_path({"vector": [1.0]}) is None
    assert formats.record_path({"metadata": {"path": ""}}) is None


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
