"""Incremental corpus export.

Scans image directories, embeds images with a SigLIP 2
`Embedder` and writes a portable embeddings export (see
`formats`). Supports incremental runs (new/modified/deleted files) and
automatic resume from an interrupted run via a ``.tmp`` file.

The export is an interchange artifact: it is not a database, and this toolkit
does not serve searches over it. Where the vectors live afterwards — a vector
store, numpy, another pipeline — is the consumer's choice.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from . import formats, models, utils
from .embedder import Embedder
from .formats import EmbeddingSpace

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]
StageCallback = Callable[[str], None]


@dataclass
class ExportConfig:
    """Configuration for a single export run."""

    input_dirs: Sequence[str] = ()
    output_file: str = "embeddings.jsonl"
    output_format: Optional[str] = None  # None -> inferred from output_file
    model: Optional[str] = None  # toolkit cache identifier or local path; None -> model in use
    home: Optional[str] = None  # toolkit home, for non-default installations
    device: Optional[str] = None  # defaults to config.json "device" (else "auto")
    batch_size: int = 32
    incremental: bool = False
    source_file: Optional[str] = None
    store_metadata: bool = True
    create_thumbnails: bool = False
    thumbnail_dir: str = "thumbnails"
    thumbnail_size: tuple[int, int] = (128, 128)


@dataclass
class ExportResult:
    """Summary of an export run."""

    processed: int = 0
    skipped: int = 0
    removed: int = 0
    total: int = 0
    output_file: str = ""
    embedding_space: Optional[EmbeddingSpace] = None


class Exporter:
    """Runs the export workflow for one configuration.

    Progress can be followed with two callbacks: ``on_progress(done, total,
    path)`` after each processed batch and ``on_stage(stage)`` for coarse
    phase changes (``"scanning"``, ``"embedding"``, ``"done"``).

    With ``incremental`` enabled, only new or modified files are embedded and
    deleted files are dropped from the export. Modification detection relies
    on file metadata, so ``store_metadata=False`` limits incremental runs to
    new/deleted files.
    """

    def __init__(
        self,
        config: ExportConfig,
        embedder: Optional[Embedder] = None,
        on_progress: Optional[ProgressCallback] = None,
        on_stage: Optional[StageCallback] = None,
    ) -> None:
        self.config = config
        self._embedder = embedder
        self._on_progress = on_progress
        self._on_stage = on_stage

    @property
    def embedder(self) -> Embedder:
        """The embedder to use, created from the config if none was provided."""
        if self._embedder is None:
            resolved = models.resolve_model(self.config.model, home=self.config.home)
            device = self.config.device or models.default_device(self.config.home)
            self._embedder = Embedder(
                model=resolved.path,
                model_id=resolved.identifier,
                device=device,
            )
        return self._embedder

    # ------------------------------------------------------------------ #

    def run(self) -> ExportResult:
        """Execute the run and return its summary."""
        cfg = self.config
        if not cfg.input_dirs:
            raise ValueError("input_dirs must contain at least one directory")
        if not cfg.output_file:
            raise ValueError("output_file must be set")
        if cfg.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        output = os.fspath(cfg.output_file)
        temp = output + ".tmp"
        fmt = cfg.output_format or formats.infer_format(output)
        if cfg.create_thumbnails:
            os.makedirs(cfg.thumbnail_dir, exist_ok=True)

        self._stage("scanning")
        disk_files = sorted(utils.find_image_files(cfg.input_dirs))
        disk_set = set(disk_files)
        logger.info("Found %d image file(s) on disk.", len(disk_files))

        existing: dict[str, dict] = {}
        source: Optional[str] = None
        if cfg.incremental:
            source = cfg.source_file or output
            if os.path.exists(source):
                existing = formats.load_by_id(source)
                logger.info("Loaded %d record(s) from %s for an incremental run.", len(existing), source)
        elif os.path.exists(temp):
            source = temp
            existing = formats.load_by_id(temp)
            logger.info("Resuming interrupted run from %s (%d record(s)).", temp, len(existing))

        if source is not None and existing:
            existing_space = self._header_space(source)
            if existing_space is not None:
                self._assert_embedding_space(existing_space, source)

        to_process = utils.filter_files_for_update(disk_set, existing)

        if not to_process and not cfg.incremental:
            if source == temp and existing:
                # Resumed run with nothing left to embed: finalise it.
                space = self._header_space(temp) or self._header_space(output)
                formats.write_export(output, list(existing.values()), embedding_space=space, fmt=fmt)
                os.remove(temp)
                logger.info("Interrupted run finalised: %s", output)
            elif not existing and not os.path.exists(output):
                # Nothing to do at all: still leave a valid, empty export.
                formats.write_export(output, [], embedding_space=None, fmt=fmt)
                logger.info("No images found; empty export written: %s", output)
            else:
                logger.info("No new or modified images to process.")
            self._stage("done")
            return ExportResult(
                total=len(existing), output_file=output, embedding_space=self._header_space(output)
            )

        records = list(existing.values())
        position: dict[str, int] = {}
        for index, record in enumerate(records):
            identifier = formats.record_id(record)
            if identifier:
                position[identifier] = index

        self._stage("embedding")
        space: Optional[EmbeddingSpace] = None
        processed = 0
        skipped = 0
        done = 0
        total = len(to_process)

        for batch in _chunks(sorted(to_process), cfg.batch_size):
            vectors, kept = self.embedder.embed_images(
                batch, batch_size=cfg.batch_size, on_error="skip", return_indices=True
            )
            if space is None:
                space = self.embedder.embedding_space()

            for row, item_index in enumerate(kept):
                path = batch[item_index]
                if cfg.store_metadata:
                    metadata = utils.extract_metadata(
                        path,
                        create_thumbnails=cfg.create_thumbnails,
                        thumbnail_dir=cfg.thumbnail_dir,
                        thumbnail_size=cfg.thumbnail_size,
                    )
                    if metadata is None:
                        skipped += 1
                        continue
                    record = formats.make_record(vectors[row].tolist(), metadata, id=path)
                else:
                    record = formats.make_record(vectors[row].tolist(), id=path)

                if path in position:
                    records[position[path]] = record
                else:
                    position[path] = len(records)
                    records.append(record)
                processed += 1

            skipped += len(batch) - len(kept)
            formats.write_export(temp, records, embedding_space=space, fmt=fmt)
            done += len(batch)
            self._progress(done, total, batch[-1])

        removed = 0
        if cfg.incremental:
            kept_records = [
                record
                for record in records
                if (identifier := formats.record_id(record)) is not None and identifier in disk_set
            ]
            removed = len(records) - len(kept_records)
            records = kept_records

        if space is None:
            space = self._header_space(temp) or self._header_space(source)

        formats.write_export(output, records, embedding_space=space, fmt=fmt)
        if os.path.exists(temp):
            os.remove(temp)

        logger.info(
            "Export complete: %d processed, %d skipped, %d removed; %d record(s) in %s.",
            processed,
            skipped,
            removed,
            len(records),
            output,
        )
        self._stage("done")
        return ExportResult(
            processed=processed,
            skipped=skipped,
            removed=removed,
            total=len(records),
            output_file=output,
            embedding_space=space,
        )

    # ------------------------------------------------------------------ #

    def _header_space(self, path: Optional[str]) -> Optional[EmbeddingSpace]:
        if not path or not os.path.exists(path):
            return None
        return formats.header_embedding_space(formats.read_header(path))

    def _assert_embedding_space(self, existing_space: EmbeddingSpace, source: str) -> None:
        configured = self.embedder.model_name
        if existing_space.model != configured:
            raise ValueError(
                f"The existing export at '{source}' was produced by "
                f"'{existing_space.model}' but the configured model is '{configured}'. "
                "Incremental updates would mix embedding spaces; use the same model or "
                "write to a different output file."
            )

    def _progress(self, done: int, total: int, path: str) -> None:
        if self._on_progress is not None:
            self._on_progress(done, total, path)

    def _stage(self, stage: str) -> None:
        logger.debug("Stage: %s", stage)
        if self._on_stage is not None:
            self._on_stage(stage)


def _chunks(items: Sequence[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]
