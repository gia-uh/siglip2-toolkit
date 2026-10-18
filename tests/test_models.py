"""Unit tests for the toolkit model cache (no torch, no network)."""

import sys
import types
from pathlib import Path

import pytest

from siglip2_toolkit import models


def _fake_model_dir(tmp_path: Path, name: str = "source-model") -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{"model_type": "siglip2"}', encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"fake-weights")
    return directory


def _home(tmp_path: Path) -> Path:
    return tmp_path / "home"


def test_config_is_created_with_visible_defaults(tmp_path):
    home = _home(tmp_path)
    config = models.ensure_config(home=home)

    assert models.config_path(home=home).is_file()
    assert config["current"] is None
    assert config["device"] == "auto"
    assert Path(config["cache_dir"]) == home / "models"
    assert models.cache_dir(home=home) == home / "models"


def test_device_default_can_be_changed_in_config(tmp_path):
    home = _home(tmp_path)
    assert models.default_device(home=home) == "auto"

    config = models.ensure_config(home=home)
    config["device"] = "cuda"
    models.save_config(config, home=home)
    assert models.default_device(home=home) == "cuda"


def test_cache_dir_can_be_relocated_from_config(tmp_path):
    home = _home(tmp_path)
    config = models.ensure_config(home=home)

    config["cache_dir"] = "otra-cache"
    models.save_config(config, home=home)
    assert models.cache_dir(home=home) == home / "otra-cache"

    elsewhere = tmp_path / "big-disk" / "siglip2-models"
    config["cache_dir"] = str(elsewhere)
    models.save_config(config, home=home)
    assert models.cache_dir(home=home) == elsewhere


def test_import_copies_model_into_the_cache(tmp_path):
    home = _home(tmp_path)
    source = _fake_model_dir(tmp_path)

    entry = models.import_model(source, home=home)

    assert entry.identifier == source.name
    assert entry.source == "import"
    assert entry.path.parent == home / "models"
    assert (entry.path / "config.json").is_file()
    assert (entry.path / "model.safetensors").read_bytes() == b"fake-weights"
    assert (entry.path / "model.json").is_file()

    assert [item.identifier for item in models.list_models(home=home)] == [entry.identifier]
    # The source directory is untouched.
    assert (source / "model.safetensors").is_file()


def test_import_with_alias_and_collision(tmp_path):
    home = _home(tmp_path)
    source = _fake_model_dir(tmp_path)
    entry = models.import_model(source, name="mi-modelo", home=home)
    assert entry.identifier == "mi-modelo"

    with pytest.raises(FileExistsError, match="already in the toolkit cache"):
        models.import_model(source, name="mi-modelo", home=home)


def test_import_rejects_invalid_directory(tmp_path):
    home = _home(tmp_path)
    invalid = tmp_path / "not-a-model"
    invalid.mkdir()
    with pytest.raises(ValueError, match="not a loadable model directory"):
        models.import_model(invalid, home=home)


def test_download_uses_huggingface_into_cache(tmp_path, monkeypatch):
    home = _home(tmp_path)

    def fake_snapshot_download(repo_id, revision=None, local_dir=None):
        target = Path(local_dir)
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"weights")
        return str(target)

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    entry = models.download_model("google/siglip2-base-patch16-naflex", revision="abc", home=home)
    assert entry.source == "download"
    assert entry.origin == "huggingface:google/siglip2-base-patch16-naflex@abc"
    assert (entry.path / "model.safetensors").is_file()

    with pytest.raises(FileExistsError):
        models.download_model("google/siglip2-base-patch16-naflex", home=home)


def test_download_rejects_non_hf_identifier(tmp_path):
    home = _home(tmp_path)
    with pytest.raises(ValueError, match="not a HuggingFace model id"):
        models.download_model("just-a-name", home=home)


def test_use_and_current(tmp_path):
    home = _home(tmp_path)
    source = _fake_model_dir(tmp_path)
    entry = models.import_model(source, home=home)

    assert models.current_identifier(home=home) is None
    assert models.set_current(entry.identifier, home=home).identifier == entry.identifier
    assert models.current_identifier(home=home) == entry.identifier

    with pytest.raises(FileNotFoundError):
        models.set_current("missing-model", home=home)


def test_delete_refuses_model_in_use(tmp_path):
    home = _home(tmp_path)
    entry = models.import_model(_fake_model_dir(tmp_path), home=home)
    models.set_current(entry.identifier, home=home)

    with pytest.raises(ValueError, match="model in use"):
        models.delete_model(entry.identifier, home=home)

    second_entry = models.import_model(_fake_model_dir(tmp_path, "second-model"), home=home)
    models.set_current(second_entry.identifier, home=home)

    removed = models.delete_model(entry.identifier, home=home)
    assert removed.identifier == entry.identifier
    assert not entry.path.exists()
    assert [item.identifier for item in models.list_models(home=home)] == [second_entry.identifier]


def test_resolve_model_by_identifier_path_and_current(tmp_path):
    home = _home(tmp_path)
    source = _fake_model_dir(tmp_path)
    entry = models.import_model(source, home=home)

    # By cache identifier.
    resolved = models.resolve_model(entry.identifier, home=home)
    assert resolved.path == entry.path
    assert resolved.identifier == entry.identifier

    # By path outside the cache: the identifier is the path string.
    resolved_path = models.resolve_model(str(source), home=home)
    assert resolved_path.path == source.resolve()
    assert resolved_path.identifier == str(source)

    # A path inside the cache reverse-resolves to the identifier.
    resolved_cached = models.resolve_model(str(entry.path), home=home)
    assert resolved_cached.identifier == entry.identifier

    # The model in use.
    models.set_current(entry.identifier, home=home)
    assert models.resolve_model(home=home).identifier == entry.identifier


def test_resolve_model_errors_include_actionable_hints(tmp_path):
    home = _home(tmp_path)
    models.ensure_config(home=home)

    with pytest.raises(FileNotFoundError, match="siglip2 models download"):
        models.resolve_model("google/siglip2-not-downloaded", home=home)
    with pytest.raises(FileNotFoundError, match="siglip2 models import"):
        models.resolve_model("mi-alias", home=home)
    with pytest.raises(FileNotFoundError, match="No model is in use"):
        models.resolve_model(home=home)


def test_catalog_is_siglip2_only():
    assert models.CATALOG
    assert all(item.id.startswith("google/siglip2-") for item in models.CATALOG)
    assert len({item.id for item in models.CATALOG}) == len(models.CATALOG)
