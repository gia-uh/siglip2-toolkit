"""Run the tutorial's main flow against a real model (opt-in integration)."""

from __future__ import annotations

from pathlib import Path

import pytest

from siglip2_toolkit import Embedder, ExportConfig, Exporter, formats
from siglip2_toolkit.retrieval import apply_similarity_drop, rank_by_similarity, results_matrix

pytestmark = pytest.mark.integration

IMAGES = sorted((Path(__file__).resolve().parents[1] / "examples" / "images").glob("*.jpg"))


def test_tutorial_flow(tmp_path, test_model_path, test_device):
    embedder = Embedder(model=test_model_path, device=test_device)

    # 3. text embeddings
    vectors = embedder.embed_texts(["a cat", "an astronaut", "a cup of coffee"])
    assert vectors.shape == (3, embedder.dimensions)

    # 4. image embeddings and text/image similarity
    image_vectors = embedder.embed_images([str(path) for path in IMAGES])
    similarity = image_vectors @ vectors.T
    texts = ["a cat", "an astronaut", "a cup of coffee"]
    best_text = {path.name: texts[int(row.argmax())] for path, row in zip(IMAGES, similarity)}
    assert best_text == {
        "astronaut.jpg": "an astronaut",
        "cat.jpg": "a cat",
        "coffee.jpg": "a cup of coffee",
    }

    # 5-6. export + inspect + search
    output = tmp_path / "embeddings.jsonl"
    result = Exporter(ExportConfig(
        input_dirs=[str(IMAGES[0].parent)],
        output_file=str(output),
        model=str(test_model_path),
        device=test_device,
        batch_size=2,
    )).run()
    assert result.total == len(IMAGES)

    records = formats.load_records(output)
    matrix = results_matrix(records)
    query = embedder.embed_texts(["a cat"])[0]
    ranked = apply_similarity_drop(
        rank_by_similarity(query, matrix, top_k=10, metric="cosine"),
        max_drop_percent=15,
    )
    top = Path((records[ranked[0][0]].get("metadata") or {})["path"]).name
    assert top == "cat.jpg"

    # 7. zero-shot classification with an explicit threshold
    cat = next(path for path in IMAGES if path.name == "cat.jpg")
    scored = embedder.classify([str(cat)], ["cat", "astronaut", "coffee"], template="a photo of a {}")[0]
    assert scored[0][0] == "cat"
    assert scored[0][1] >= 0.5
