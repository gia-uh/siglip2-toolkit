"""Shared fixtures and markers for the test suite."""

import os
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = DATA_DIR / "images"

#: Environment variable that opts into model-dependent (integration) tests.
INTEGRATION_ENV = "SIGLIP2_INTEGRATION"


def pytest_collection_modifyitems(config, items):
    if os.environ.get(INTEGRATION_ENV):
        return
    skip = pytest.mark.skip(
        reason=f"set {INTEGRATION_ENV}=1 to run (needs torch and a cached SigLIP 2 model)"
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
def test_model() -> str:
    """Model used by integration tests (override with SIGLIP2_TEST_MODEL)."""
    return os.environ.get("SIGLIP2_TEST_MODEL", "google/siglip2-base-patch16-naflex")
