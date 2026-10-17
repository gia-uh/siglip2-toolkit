"""Semantic tests with a real SigLIP 2 model.

These are integration tests: they need torch and a locally available model.
Enable them with ``SIGLIP2_INTEGRATION=1`` (see ``conftest.py``). Ground truth
below follows ``tests/data/SOURCES.md``: every query or label describes what
is actually visible in the referenced image.
"""

import numpy as np
import pytest

from siglip2_toolkit import retrieval

pytestmark = pytest.mark.integration

#: Text query -> image that must rank first.
RETRIEVAL_CASES = {
    "a cat": "cat.jpg",
    "an astronaut in a spacesuit": "astronaut.jpg",
    "a cup of coffee": "coffee.jpg",
    "a rocket on a launch pad": "rocket.jpg",
    "a horse": "horse.jpg",
    "a man taking a photo with a camera": "camera.jpg",
    "galaxies in space": "hubble.jpg",
    "a woman in a military uniform": "grace_hopper.jpg",
}

#: Image -> correct zero-shot label (plus fixed distractors below).
CLASSIFICATION_CASES = {
    "cat.jpg": "a close-up of a cat",
    "astronaut.jpg": "an astronaut in an orange spacesuit",
    "coffee.jpg": "a cup of coffee on a table",
    "rocket.jpg": "a rocket standing on a launch pad",
    "horse.jpg": "a silhouette of a horse",
    "camera.jpg": "a man using a camera",
    "hubble.jpg": "galaxies in deep space",
    "grace_hopper.jpg": "a woman in a navy uniform",
}

DISTRACTORS = ["a dog", "a beach", "a bicycle"]


@pytest.fixture(scope="module")
def embedder(test_model):
    from siglip2_toolkit.embedder import Embedder

    return Embedder(model=test_model, device="cpu")


def test_embeddings_are_normalized_and_deterministic(embedder, image_paths):
    paths = [str(path) for path in image_paths[:3]]
    first = embedder.embed_images(paths)
    second = embedder.embed_images(paths)

    assert first.shape[0] == 3
    assert first.shape[1] == embedder.dimensions
    norms = np.linalg.norm(first, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
    assert np.allclose(first, second, atol=1e-6)


def test_text_to_image_retrieval_top1(embedder, image_paths):
    names = [path.name for path in image_paths]
    matrix = embedder.embed_images([str(path) for path in image_paths])

    failures = []
    for query, expected in RETRIEVAL_CASES.items():
        ranked = retrieval.rank_by_similarity(embedder.embed_text(query), matrix, top_k=1)
        best = names[ranked[0][0]]
        if best != expected:
            failures.append(f"{query!r} -> {best} (expected {expected})")
    assert not failures, "retrieval mismatches: " + "; ".join(failures)


def test_zero_shot_classification_top1(embedder, images_dir):
    failures = []
    for filename, correct in CLASSIFICATION_CASES.items():
        labels = [correct] + DISTRACTORS
        top = embedder.classify([str(images_dir / filename)], labels, top_k=1)[0][0]
        if top[0] != correct:
            failures.append(f"{filename}: got {top[0]!r} ({top[1]:.3f})")
    assert not failures, "classification mismatches: " + "; ".join(failures)


def test_zero_shot_probabilities_sum_to_one(embedder, images_dir):
    labels = ["a cat", "a dog", "a cup of coffee"]
    row = embedder.classify([str(images_dir / "cat.jpg")], labels)[0]
    assert sum(score for _, score in row) == pytest.approx(1.0, abs=1e-4)


def test_classify_with_template(embedder, images_dir):
    results = embedder.classify(
        [str(images_dir / "cat.jpg")],
        ["a cat", "a dog"],
        template="This is a photo of {}.",
        top_k=1,
    )
    assert results[0][0][0] == "a cat"


def test_zero_shot_logits_shape(embedder, images_dir):
    logits = embedder.zero_shot_logits(
        [str(images_dir / "cat.jpg"), str(images_dir / "coffee.jpg")],
        ["a cat", "a cup of coffee", "a rocket"],
    )
    assert logits.shape == (2, 3)
    assert logits.dtype == np.float32


def test_indexer_with_real_model_smoke(tmp_path, image_paths, embedder):
    """End-to-end: index the test images with the real model, then search."""
    from siglip2_toolkit.indexer import Indexer, IndexerConfig

    output = tmp_path / "semantic_index.jsonl"
    config = IndexerConfig(
        input_dirs=[str(image_paths[0].parent)],
        output_file=str(output),
        model=embedder.model_name,
        device="cpu",
        batch_size=4,
    )
    result = Indexer(config, embedder=embedder).run()
    assert result.processed >= 6
    assert output.exists()
