"""Storage-agnostic retrieval primitives.

These helpers operate on numpy arrays and on ``(payload, score)`` result
lists, so they can be reused with any vector store or directly over vectors
loaded from an embeddings export (see `formats`).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Hashable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

METRICS = ("cosine", "euclid", "manhattan")


def rank_by_similarity(
    query: Any,
    matrix: Any,
    top_k: int = 50,
    metric: str = "cosine",
) -> list[tuple[int, float]]:
    """Rank the rows of ``matrix`` against ``query``.

    Returns ``(row_index, score)`` pairs ordered from best to worst, where
    scores are distances (lower is better) for every supported metric.
    """
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError("matrix must be a 2-D array of row vectors")
    if m.shape[1] != q.shape[0]:
        raise ValueError(
            f"dimension mismatch: query has {q.shape[0]}, matrix rows have {m.shape[1]}"
        )

    if metric == "cosine":
        q_norm = q / max(float(np.linalg.norm(q)), 1e-12)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        m_norm = m / np.clip(norms, 1e-12, None)
        scores = 1.0 - (m_norm @ q_norm)
    elif metric == "euclid":
        scores = np.linalg.norm(m - q, axis=1)
    elif metric == "manhattan":
        scores = np.abs(m - q).sum(axis=1)
    else:
        raise ValueError(f"unknown metric: {metric!r} (expected one of {METRICS})")

    order = np.argsort(scores)[: max(0, int(top_k))]
    return [(int(index), float(scores[index])) for index in order]


def results_matrix(records: Sequence[dict[str, Any]], key: str = "vector") -> np.ndarray:
    """Stack the vectors of index records into a ``(n, dimensions)`` matrix."""
    return np.asarray([record[key] for record in records], dtype=np.float32)


def _default_key(item: Any) -> Hashable:
    if isinstance(item, dict):
        # Index records: {"vector": [...], "metadata": {"path": ...}}
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            nested = metadata.get("path") or metadata.get("id")
            if isinstance(nested, str) and nested:
                return nested
        # Flat metadata dictionaries.
        for candidate in ("path", "id"):
            value = item.get(candidate)
            if isinstance(value, str) and value:
                return value
    try:
        hash(item)
    except TypeError as exc:
        raise ValueError(
            "Items of this type need an explicit key= function to be fused."
        ) from exc
    return item


def fuse_rrf(
    result_lists: Sequence[Sequence[tuple[Any, float]]],
    weights: Optional[Sequence[float]] = None,
    k: int = 60,
    key: Optional[Callable[[Any], Hashable]] = None,
) -> list[tuple[Any, float]]:
    """Reciprocal Rank Fusion over several ranked lists.

    Each list contains ``(payload, score)`` pairs ordered from best to worst;
    the score semantics do not matter for the fusion. Returns ``(payload,
    rrf_score)`` pairs sorted by decreasing fused score. Ties prefer the
    better original score, assuming lower-is-better (distance-like) scores.
    """
    result_lists = list(result_lists)
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)
    weights = [float(weight) for weight in weights]
    if len(weights) != len(result_lists):
        raise ValueError("weights must have one entry per result list")

    key_fn = key or _default_key
    fused: dict[Hashable, dict[str, Any]] = {}

    for list_index, results in enumerate(result_lists):
        weight = weights[list_index]
        for rank, entry in enumerate(results):
            if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
                continue
            payload, score = entry
            item_key = key_fn(payload)
            state = fused.setdefault(
                item_key, {"payload": payload, "rrf": 0.0, "best": float("inf")}
            )
            state["rrf"] += weight * (1.0 / (k + rank))
            try:
                state["best"] = min(state["best"], float(score))
            except (TypeError, ValueError):
                pass

    ranked = sorted(fused.values(), key=lambda state: (state["rrf"], -state["best"]), reverse=True)
    return [(state["payload"], state["rrf"]) for state in ranked]


def apply_similarity_drop(
    results: Sequence[tuple[Any, float]],
    max_drop_percent: float,
    lower_is_better: bool = True,
    score_of: Optional[Callable[[tuple[Any, float]], float]] = None,
) -> list[tuple[Any, float]]:
    """Keep only the results within ``max_drop_percent`` of the best score.

    The threshold is relative to the magnitude of the best score:
    ``best + |best| * pct/100`` when lower is better (distances), or
    ``best - |best| * pct/100`` when higher is better (e.g. RRF scores).
    """
    results = list(results)
    if not results:
        return results

    getter = score_of or (lambda entry: float(entry[1]))
    best = min(getter(entry) for entry in results) if lower_is_better else max(
        getter(entry) for entry in results
    )
    delta = abs(best) * (float(max_drop_percent) / 100.0)

    if lower_is_better:
        threshold = best + delta
        return [entry for entry in results if getter(entry) <= threshold]
    threshold = best - delta
    return [entry for entry in results if getter(entry) >= threshold]
