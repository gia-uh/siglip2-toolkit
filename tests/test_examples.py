"""Validate the runnable examples and the documentation links to them."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_example_help(example: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(example), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("page", ["docs/tutorial.md", "docs/examples.md"])
def test_docs_example_links_exist(page: str) -> None:
    text = (ROOT / page).read_text(encoding="utf-8")
    names = set(re.findall(r"examples/(\d+_[A-Za-z0-9_]+\.py)", text))
    assert names, f"no example links found in {page}"
    for name in names:
        assert (ROOT / "examples" / name).is_file(), f"missing example {name}"
