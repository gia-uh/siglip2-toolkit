"""Unit tests for the generic filter expressions (numpy-only, no torch)."""

import json

import pytest

from siglip2_toolkit import filters
from siglip2_toolkit.filters import And, Not, Or, field, from_dict, matches, to_dict

SUBJECT = {
    "width": 2000,
    "size_bytes": 3_500_000,
    "filename": "IMG_0042.JPG",
    "exif": {"Make": "NIKON CORPORATION", "FNumber": "2.8", "ISO": 160},
    "tags": ["wedding", "beach"],
}


def test_comparison_operators():
    assert matches(SUBJECT, field("width") == 2000)
    assert matches(SUBJECT, field("width") != 100)
    assert matches(SUBJECT, field("width") > 1000)
    assert matches(SUBJECT, field("width") >= 2000)
    assert matches(SUBJECT, field("width") < 3000)
    assert matches(SUBJECT, field("width") <= 2000)
    assert not matches(SUBJECT, field("width") > 5000)


def test_dotted_paths_and_missing_fields_never_match():
    assert matches(SUBJECT, field("exif.Make").contains("nikon"))
    assert not matches(SUBJECT, field("exif.Model") == "X100")
    assert not matches(SUBJECT, field("missing") != 5)
    assert matches(SUBJECT, Not(field("missing") == 5))


def test_numeric_coercion_for_string_numbers():
    assert matches(SUBJECT, field("exif.FNumber") >= 2.0)
    assert matches(SUBJECT, field("exif.FNumber") < 3)
    assert not matches(SUBJECT, field("exif.FNumber") >= 3.0)
    assert matches({"exposure": "0.005"}, field("exposure") <= 0.005)


def test_string_operators_case_insensitive_by_default():
    assert matches(SUBJECT, field("filename").startswith("img_"))
    assert matches(SUBJECT, field("filename").contains(".jpg"))
    assert not matches(SUBJECT, field("filename").startswith("img_", case_sensitive=True))
    assert matches(SUBJECT, field("filename").startswith("IMG_", case_sensitive=True))


def test_in_and_exists():
    assert matches(SUBJECT, field("exif.ISO").in_([100, 160, 200]))
    assert not matches(SUBJECT, field("exif.ISO").in_([100, 200]))
    assert matches(SUBJECT, field("tags").exists())
    assert not matches(SUBJECT, field("exif.Flash").exists())
    assert not matches({"exif": {}}, field("exif").exists())
    assert not matches({"exif": None}, field("exif").exists())


def test_between():
    assert matches(SUBJECT, field("width").between(1000, 4000))
    assert not matches(SUBJECT, field("width").between(3000, 4000))


def test_combinators():
    assert matches(SUBJECT, And(field("width") > 1000, field("size_bytes") < 5_000_000))
    assert matches(SUBJECT, Or(field("width") > 5000, field("exif.Make").contains("nikon")))
    assert matches(SUBJECT, Not(field("exif.Make").contains("canon")))
    assert not matches(SUBJECT, And(field("width") > 1000, field("width") < 1500))


def test_matches_rejects_non_filters():
    with pytest.raises(TypeError):
        matches(SUBJECT, {"not": "a filter"})
    with pytest.raises(ValueError):
        filters.Filter("x", "nope")


def test_leaf_serialization():
    assert to_dict(field("exif.Model").exists()) == {"path": "exif.Model", "op": "exists"}
    assert to_dict(field("name").contains("x", case_sensitive=True)) == {
        "path": "name",
        "op": "contains",
        "value": "x",
        "case_sensitive": True,
    }


def test_serialization_roundtrip_is_json_compatible():
    node = And(
        field("width").between(1000, 4000),
        Or(field("exif.Make").contains("NikOn", case_sensitive=True), field("tags").in_(["x", "y"])),
        Not(field("exif.Model").exists()),
    )
    data = to_dict(node)
    json.dumps(data)  # must be JSON-serializable

    rebuilt = from_dict(data)
    assert to_dict(rebuilt) == data
    assert matches(SUBJECT, rebuilt) == matches(SUBJECT, node)


def test_from_dict_rejects_unknown_shapes():
    with pytest.raises(ValueError):
        from_dict({"op": "wat"})
