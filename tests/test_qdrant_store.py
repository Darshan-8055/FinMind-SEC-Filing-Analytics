
"""Automated tests for the Qdrant chunk store."""

import pytest
from qdrant_client import QdrantClient

from src.retrieval.qdrant_store import QdrantChunkStore


@pytest.fixture
def store():
    """Create an isolated in-memory Qdrant store for each test."""
    client = QdrantClient(":memory:")
    chunk_store = QdrantChunkStore(
        client,
        collection_name="test_chunks",
        vector_size=3,
    )
    yield chunk_store
    client.close()


@pytest.fixture
def sample_chunk():
    return {
        "chunk_id": "chunk-001",
        "company_name": "Example Corporation",
        "ticker": "EXM",
        "form_type": "10-K",
        "filing_date": "2026-02-01",
        "section": "Item 1 - Business",
        "text": "Example Corporation develops financial analytics software.",
        "source_url": "https://example.com/filing/001",
    }


def test_upsert_and_search_preserves_metadata(store, sample_chunk):
    count = store.upsert_chunks([sample_chunk], [[1.0, 0.0, 0.0]])

    results = store.search([1.0, 0.0, 0.0], top_k=1)

    assert count == 1
    assert len(results) == 1
    assert results[0]["chunk_id"] == sample_chunk["chunk_id"]
    assert results[0]["ticker"] == "EXM"
    assert results[0]["text"] == sample_chunk["text"]
    assert results[0]["source_url"] == sample_chunk["source_url"]
    assert results[0]["semantic_score"] == pytest.approx(1.0)


def test_search_filters_by_ticker(store, sample_chunk):
    second_chunk = {
        **sample_chunk,
        "chunk_id": "chunk-002",
        "ticker": "SMP",
        "company_name": "Sample Industries",
    }

    store.upsert_chunks(
        [sample_chunk, second_chunk],
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
    )

    results = store.search(
        [1.0, 0.0, 0.0],
        top_k=5,
        filters={"ticker": "SMP"},
    )

    assert len(results) == 1
    assert results[0]["ticker"] == "SMP"


def test_returns_empty_when_collection_does_not_exist(store):
    results = store.search([1.0, 0.0, 0.0])

    assert results == []


def test_rejects_mismatched_chunk_and_embedding_counts(store, sample_chunk):
    with pytest.raises(ValueError, match="exactly one embedding"):
        store.upsert_chunks([sample_chunk], [])


def test_rejects_wrong_embedding_dimension(store, sample_chunk):
    with pytest.raises(ValueError, match="dimensions"):
        store.upsert_chunks([sample_chunk], [[1.0, 0.0]])


def test_rejects_duplicate_chunk_ids_in_batch(store, sample_chunk):
    duplicate = {**sample_chunk}

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        store.upsert_chunks(
            [sample_chunk, duplicate],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )


def test_rejects_invalid_top_k(store):
    with pytest.raises(ValueError, match="at least 1"):
        store.search([1.0, 0.0, 0.0], top_k=0)


def test_empty_batch_writes_nothing(store):
    assert store.upsert_chunks([], []) == 0


def test_rejects_empty_required_chunk_field(store, sample_chunk):
    invalid_chunk = {**sample_chunk, "text": " "}

    with pytest.raises(ValueError, match="text"):
        store.upsert_chunks([invalid_chunk], [[1.0, 0.0, 0.0]])
