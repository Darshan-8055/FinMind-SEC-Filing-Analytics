
"""Tests for semantic retrieval using a fake embedding model."""

import pytest

from src.retrieval.semantic_search import semantic_search


class FakeEmbeddingModel:
    """Return predictable vectors so tests need no model download."""

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return [self.vectors[text] for text in texts]


@pytest.fixture
def model():
    return FakeEmbeddingModel(
        {
            "query": [1.0, 0.0],
            "similar passage": [0.9, 0.1],
            "unrelated passage": [0.0, 1.0],
            "another similar passage": [0.8, 0.2],
        }
    )


@pytest.fixture
def chunks():
    return [
        {
            "chunk_id": "A",
            "text": "unrelated passage",
            "ticker": "MSFT",
        },
        {
            "chunk_id": "B",
            "text": "similar passage",
            "ticker": "AAPL",
        },
        {
            "chunk_id": "C",
            "text": "another similar passage",
            "ticker": "AAPL",
        },
    ]


def test_ranks_most_similar_chunk_first(model, chunks):
    results = semantic_search("query", chunks, model, top_k=3)

    assert results[0]["chunk_id"] == "B"
    assert results[1]["chunk_id"] == "C"
    assert results[2]["chunk_id"] == "A"


def test_adds_similarity_score(model, chunks):
    results = semantic_search("query", chunks, model)

    assert "semantic_score" in results[0]
    assert -1.0 <= results[0]["semantic_score"] <= 1.0


def test_preserves_chunk_metadata(model, chunks):
    results = semantic_search("query", chunks, model)

    best_match = next(item for item in results if item["chunk_id"] == "B")
    assert best_match["ticker"] == "AAPL"


def test_respects_top_k(model, chunks):
    results = semantic_search("query", chunks, model, top_k=2)

    assert len(results) == 2


def test_returns_empty_for_blank_query(model, chunks):
    assert semantic_search("   ", chunks, model) == []


def test_returns_empty_when_no_valid_text(model):
    chunks_without_text = [
        {"chunk_id": "A"},
        {"chunk_id": "B", "text": "   "},
    ]

    assert semantic_search("query", chunks_without_text, model) == []


def test_rejects_invalid_top_k(model, chunks):
    with pytest.raises(ValueError):
        semantic_search("query", chunks, model, top_k=0)


def test_rejects_non_string_query(model, chunks):
    with pytest.raises(TypeError):
        semantic_search(123, chunks, model)
