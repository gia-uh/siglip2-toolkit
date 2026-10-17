"""Unit tests for the retrieval primitives (numpy only)."""

import numpy as np
import pytest

from siglip2_toolkit import retrieval


def test_rank_by_similarity_cosine():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.70710678, 0.70710678]], dtype=np.float32)
    ranked = retrieval.rank_by_similarity([1.0, 0.0], matrix, top_k=3)

    assert [index for index, _ in ranked][:2] == [0, 2]
    assert ranked[0][1] == pytest.approx(0.0, abs=1e-6)
    assert ranked[0][1] <= ranked[1][1] <= ranked[2][1]


def test_rank_by_similarity_euclid_and_validation():
    matrix = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    ranked = retrieval.rank_by_similarity([1.0, 0.0], matrix, metric="euclid")
    assert ranked[0][0] == 0

    with pytest.raises(ValueError):
        retrieval.rank_by_similarity([1.0, 2.0, 3.0], matrix)
    with pytest.raises(ValueError):
        retrieval.rank_by_similarity([1.0, 0.0], matrix, metric="nope")


def test_results_matrix_stacks_vectors():
    records = [
        {"vector": [0.1, 0.2], "metadata": {"path": "/a.jpg"}},
        {"vector": [0.3, 0.4], "metadata": {"path": "/b.jpg"}},
    ]
    matrix = retrieval.results_matrix(records)
    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.float32


def test_fuse_rrf_prefers_documents_ranked_high_in_both_lists():
    list_a = [({"path": "/a.jpg"}, 0.1), ({"path": "/b.jpg"}, 0.2)]
    list_b = [({"path": "/b.jpg"}, 0.9), ({"path": "/c.jpg"}, 1.0)]

    fused = retrieval.fuse_rrf([list_a, list_b])
    assert [payload["path"] for payload, _ in fused] == ["/b.jpg", "/a.jpg", "/c.jpg"]


def test_fuse_rrf_weights_and_index_records_are_keyed_by_metadata_path():
    record = {"vector": [1.0], "metadata": {"path": "/a.jpg"}}
    fused = retrieval.fuse_rrf([[(record, 0.5)]], weights=[2.0], k=10)

    assert fused[0][0] is record
    assert fused[0][1] == pytest.approx(2.0 / 10.0)

    with pytest.raises(ValueError):
        retrieval.fuse_rrf([[({}, 1.0)]], weights=[1.0, 2.0])


def test_apply_similarity_drop_lower_is_better():
    distances = [("a", 0.10), ("b", 0.11), ("c", 0.30)]
    kept = retrieval.apply_similarity_drop(distances, max_drop_percent=15)
    assert [name for name, _ in kept] == ["a", "b"]


def test_apply_similarity_drop_higher_is_better():
    scores = [("x", 0.90), ("y", 0.80), ("z", 0.10)]
    kept = retrieval.apply_similarity_drop(scores, max_drop_percent=15, lower_is_better=False)
    assert [name for name, _ in kept] == ["x", "y"]


def test_apply_similarity_drop_empty_input():
    assert retrieval.apply_similarity_drop([], 20) == []
