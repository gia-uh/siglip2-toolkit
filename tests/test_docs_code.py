"""Validate every fenced code block in the README and docs.

Python blocks must compile (top-level await allowed), JSON/YAML must parse and
shell blocks must pass ``bash -n``. Text trees and illustrative output are
explicitly marked as ``text`` instead of pretending to be executable code.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```\s*$", re.M | re.S)
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
LANGUAGES = {"python", "py", "bash", "sh", "shell", "json", "yaml", "yml", "text", "console", "output"}


def _blocks():
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        for match in FENCE.finditer(text):
            raw = match.group(1).strip()
            language = raw.split()[0].lower() if raw else ""
            line = text[: match.start()].count("\n") + 1
            yield pytest.param(path, line, language, match.group(2), id=f"{path.name}:{line}")


@pytest.mark.parametrize("path,line,language,code", list(_blocks()))
def test_code_block(path: Path, line: int, language: str, code: str) -> None:
    where = f"{path.relative_to(ROOT)}:{line}"
    if language in {"python", "py"}:
        compile(code, where, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        return
    if language == "json":
        json.loads(code)
        return
    if language in {"yaml", "yml"}:
        list(yaml.safe_load_all(code))
        return
    if language in {"bash", "sh", "shell"}:
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
            handle.write(code)
            script = Path(handle.name)
        try:
            result = subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, text=True, timeout=30
            )
        finally:
            script.unlink()
        assert result.returncode == 0, f"{where}: {result.stderr.strip()}"
        return
    if language in {"text", "console", "output"}:
        return
    pytest.fail(f"{where}: fenced block without a supported language ({language!r})")
