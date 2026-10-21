# Development

The project is managed with [uv](https://docs.astral.sh/uv/). The Python
version is pinned in `.python-version` (3.12) and dependencies are locked in
`uv.lock`.

## Setup

```bash
git clone https://github.com/gia-uh/siglip2-toolkit
cd siglip2-toolkit
uv sync --all-extras
uv run siglip2 --help
```

`--all-extras` installs the `dev` and `server` extras. For CPU-only work:

```bash
uv venv --python 3.12
uv pip install --torch-backend cpu -e ".[dev,server]"
```

## Tests

```bash
uv run pytest -q                 # unit tests; no model needed
uv run pytest -q --cov=siglip2_toolkit
```

The integration tests load a real model and are opt-in:

```bash
HF_HUB_OFFLINE=1 SIGLIP2_INTEGRATION=1 uv run pytest -q
```

They use the open-licensed images in `tests/data/images/`. **Never** add
private photos to the repository; every test/example image must list its
source and an open license in `tests/data/SOURCES.md` or
`examples/images/SOURCES.md`.

## Lint and style

```bash
uv run ruff check src tests examples
uv run ruff format src tests examples    # optional
```

Line length is 100; target Python 3.12. Public functions and classes should
have docstrings (they are rendered in the [API reference](api/index.md)).

## Documentation

The docs use MkDocs Material and mkdocstrings. Build them in a separate
environment so the docs toolchain does not install torch:

```bash
uv venv .venv-docs --python 3.12
uv pip install --python .venv-docs/bin/python mkdocs-material "mkdocstrings[python]" ruff
.venv-docs/bin/mkdocs serve          # live preview at http://127.0.0.1:8000
.venv-docs/bin/mkdocs build --strict # CI build
```

The site sources live in `docs/`; the configuration is `mkdocs.yml`. API
pages use `::: siglip2_toolkit.module.Class` and are generated from the
docstrings with **static** analysis (`allow_inspection: false`), so the docs
build does not need the runtime dependencies.

CI builds the docs with `mkdocs build --strict`. The deploy job is gated by
the repository variable `DEPLOY_DOCS`: enable GitHub Pages (source: GitHub
Actions), set `DEPLOY_DOCS=true`, and the `docs` workflow publishes on main.

## Build and publish

```bash
uv build                       # wheel + sdist in dist/
uv run twine check dist/*
```

Before a release:

1. Update `version` in `pyproject.toml`.
2. Add an entry to `CHANGELOG.md`.
3. Run the unit and integration tests.
4. `uv lock` if dependencies changed.
5. `uv build` and check the artifacts.
6. Publish (maintainers) with a PyPI token or trusted publishing:
   ```bash
   uv publish
   # or: twine upload dist/*
   ```
7. Tag the release (`git tag vX.Y.Z`) and push the tag.

## Project layout

```
src/siglip2_toolkit/
├── __init__.py        # lazy public API
├── embedder.py        # Embedder (text/image embeddings, zero-shot)
├── exporter.py        # ExportConfig, Exporter, ExportResult
├── classification.py  # ImageClassifier (fine-tuned heads)
├── formats.py         # export format v2, EmbeddingSpace, read/write helpers
├── filters.py         # serialisable metadata predicates
├── retrieval.py       # similarity ranking, RRF, similarity drop
├── models.py          # toolkit cache, catalog, download/import/resolve
├── utils.py           # images, metadata, thumbnails, device selection
├── cli.py             # siglip2 command
└── server.py          # optional FastAPI app
```

`__init__.py` is deliberately lazy: importing the package does not import
torch, transformers or FastAPI.

## Contributing

- Open an issue or a pull request with a clear description.
- Keep the public API and the export format stable; document breaking changes
  in the changelog.
- Add tests for new behaviour, including edge cases (unreadable images,
  missing ids, mixed embedding spaces).
- Do not add private or non-redistributable images or models.
- Update the docs when you add a public function, CLI flag or endpoint.
- Run `uv run pytest -q` and `mkdocs build --strict` before opening the PR.
