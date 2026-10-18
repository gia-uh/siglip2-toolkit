"""Shared fixtures and markers for the test suite."""

import os
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = DATA_DIR / "images"

#: Environment variable that opts into model-dependent (integration) tests.
INTEGRATION_ENV = "SIGLIP2_INTEGRATION"

DEFAULT_TEST_MODEL = "google/siglip2-base-patch16-naflex"


def pytest_collection_modifyitems(config, items):
    if os.environ.get(INTEGRATION_ENV):
        return
    skip = pytest.mark.skip(
        reason=f"set {INTEGRATION_ENV}=1 to run (needs torch and a local SigLIP 2 model)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def images_dir() -> Path:
    """Directory with the small, openly licensed test images."""
    assert IMAGES_DIR.is_dir(), f"missing test images directory: {IMAGES_DIR}"
    return IMAGES_DIR


@pytest.fixture(scope="session")
def image_paths(images_dir: Path) -> list[Path]:
    """Paths of the test images, sorted by file name."""
    paths = sorted(images_dir.glob("*.jpg"))
    assert len(paths) >= 6, "expected the documented test images in tests/data/images"
    return paths


@pytest.fixture(scope="session")
def test_model_path() -> Path:
    """A local model directory for integration tests.

    Uses ``SIGLIP2_TEST_MODEL_PATH`` when set; otherwise looks for a cached
    HuggingFace snapshot of ``SIGLIP2_TEST_MODEL`` (default: the toolkit's
    default model). Integration tests never download anything.
    """
    override = os.environ.get("SIGLIP2_TEST_MODEL_PATH")
    if override:
        path = Path(override).expanduser()
        if (path / "config.json").is_file():
            return path
        pytest.skip(f"SIGLIP2_TEST_MODEL_PATH is not a model directory: {path}")

    repo_id = os.environ.get("SIGLIP2_TEST_MODEL", DEFAULT_TEST_MODEL)
    snapshots = (
        Path.home()
        / ".cache/huggingface/hub"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
    )
    if snapshots.is_dir():
        for candidate in sorted(snapshots.iterdir()):
            if (candidate / "config.json").is_file():
                return candidate
    pytest.skip(
        "no local SigLIP 2 model found; set SIGLIP2_TEST_MODEL_PATH to a model directory"
    )


@pytest.fixture(scope="session")
def test_device() -> str:
    """Device for integration tests (CPU by default for reproducibility).

    Set ``SIGLIP2_TEST_DEVICE=cuda`` to run them on GPU.
    """
    return os.environ.get("SIGLIP2_TEST_DEVICE", "cpu")
