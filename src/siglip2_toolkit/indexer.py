"""Incremental image indexer.

Scans directories, embeds images with a SigLIP 2 :class:`~.embedder.Embedder`
and writes a versioned index file (see :mod:`.formats`). Supports incremental
re-indexing (new/modified/deleted files) and automatic resume from an
interrupted run via a ``.tmp`` file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from . import formats, utils
from .embedder import DEFAULT_MODEL, Embedder
from .formats import EmbeddingSpace

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]
StageCallback = Callable[[str], None]


@dataclass
class IndexerConfig:
    """Configuration for a single indexing run."""

    input_dirs: Sequence[str] = ()
    output_file: str = "image_index.jsonl"
    output_format: Optional[str] = None  # None -> inferred from output_file
    model: str = DEFAULT_MODEL
    device: str = "auto"
    batch_size: int = 32
    reindex: bool = False
    source_index: Optional[str] = None
    create_thumbnails: bool = False
    thumbnail_dir: str = "thumbnails"
    thumbnail_size: tuple[int, int] = (128, 128)


@dataclass
class IndexResult:
    """Summary of an indexing run."""

    processed: int = 0
    skipped: int = 0
    removed: int = 0
    total: int = 0
    output_file: str = ""
    embedding_space: Optional[EmbeddingSpace] = None


class Indexer:
    """Runs the indexing workflow for one configuration.

    Progress can be followed with two callbacks: ``on_progress(done, total,
    path)`` after each processed batch and ``on_stage(stage)`` for coarse
    phase changes (``"scanning"``, ``"indexing"``, ``"done"``).
    """

    def __init__(
        self,
        config: IndexerConfig,
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
            self._embedder = Embedder(model=self.config.model, device=self.config.device)
        return self._embedder

    # ------------------------------------------------------------------ #

    def run(self) -> IndexResult:
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
        if cfg.reindex:
            source = cfg.source_index or output
            if os.path.exists(source):
                existing = formats.load_index_map(source)
                logger.info("Loaded %d record(s) from %s for re-indexing.", len(existing), source)
        elif os.path.exists(temp):
            source = temp
            existing = formats.load_index_map(temp)
            logger.info("Resuming interrupted run from %s (%d record(s)).", temp, len(existing))

        to_process = utils.filter_files_for_reindex(disk_set, existing)

        if not to_process and not cfg.reindex:
            if source == temp and existing:
                # Resumed run with nothing left to embed: finalise it.
                space = self._header_space(temp) or self._header_space(output)
                formats.write_index(output, list(existing.values()), embedding_space=space, fmt=fmt)
                os.remove(temp)
                logger.info("Interrupted run finalised: %s", output)
            else:
                logger.info("No new or modified images to process.")
            self._stage("done")
            return IndexResult(
                total=len(existing), output_file=output, embedding_space=self._header_space(output)
            )

        records = list(existing.values())
        position: dict[str, int] = {}
        for index, record in enumerate(records):
            path = formats.record_path(record)
            if path:
                position[path] = index

        self._stage("indexing")
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
                metadata = utils.extract_metadata(
                    path,
                    create_thumbnails=cfg.create_thumbnails,
                    thumbnail_dir=cfg.thumbnail_dir,
                    thumbnail_size=cfg.thumbnail_size,
                )
                if metadata is None:
                    skipped += 1
                    continue
                record = formats.make_record(vectors[row].tolist(), metadata)
                if path in position:
                    records[position[path]] = record
                else:
                    position[path] = len(records)
                    records.append(record)
                processed += 1

            skipped += len(batch) - len(kept)
            formats.write_index(temp, records, embedding_space=space, fmt=fmt)
            done += len(batch)
            self._progress(done, total, batch[-1])

        removed = 0
        if cfg.reindex:
            kept_records = [
                record
                for record in records
                if (path := formats.record_path(record)) is not None and path in disk_set
            ]
            removed = len(records) - len(kept_records)
            records = kept_records

        if space is None:
            space = self._header_space(temp) or self._header_space(source)

        formats.write_index(output, records, embedding_space=space, fmt=fmt)
        if os.path.exists(temp):
            os.remove(temp)

        logger.info(
            "Indexing complete: %d processed, %d skipped, %d removed; %d record(s) in %s.",
            processed,
            skipped,
            removed,
            len(records),
            output,
        )
        self._stage("done")
        return IndexResult(
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
