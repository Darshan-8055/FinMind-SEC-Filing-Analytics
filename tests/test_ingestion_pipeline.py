
"""Tests for the FinMind filing ingestion pipeline."""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.ingestion.pipeline import FilingIngestionPipeline
from src.retrieval.qdrant_store import QdrantChunkStore


class FakeEmbeddingModel:
    """Deterministic 384-dimensional model for fast tests."""

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = 1.0
            vectors.append(vector)
        return np.array(vectors)


@pytest.fixture
def pipeline():
    client = QdrantClient(":memory:")
    store = QdrantChunkStore(
        client,
        collection_name="pipeline_test",
        vector_size=384,
    )
    model = FakeEmbeddingModel()
    instance = FilingIngestionPipeline(store, model)

    yield instance
    client.close()


def test_ingests_filing_and_preserves_metadata(pipeline):
    filing_text = """
ITEM 1. BUSINESS
Example Corporation provides financial analytics software.
The company serves business customers.

ITEM 1A. RISK FACTORS
Competition may affect future revenue.
"""

    count = pipeline.ingest_filing(
        filing_text,
        filename="example-10k.txt",
        filing_type="10-K",
        ticker="EXM",
        company_name="Example Corporation",
        filing_date="2026-02-01",
        source_url="https://example.com/filing/001",
    )

    assert count > 0

    results = pipeline.search("financial analytics", top_k=5)

    assert results
    assert results[0]["ticker"] == "EXM"
    assert results[0]["company_name"] == "Example Corporation"
    assert results[0]["form_type"] == "10-K"
    assert results[0]["source_url"] == "https://example.com/filing/001"
    assert results[0]["text"].strip()


def test_empty_filing_returns_zero(pipeline):
    count = pipeline.ingest_filing(
        "",
        filename="empty.txt",
        filing_type="10-K",
        ticker="EXM",
        company_name="Example Corporation",
        filing_date="2026-02-01",
        source_url="https://example.com/empty",
    )

    assert count == 0


def test_search_can_filter_by_ticker(pipeline):
    pipeline.ingest_filing(
        "ITEM 1. BUSINESS\nExample Corporation provides analytics software.",
        filename="example.txt",
        filing_type="10-K",
        ticker="EXM",
        company_name="Example Corporation",
        filing_date="2026-02-01",
        source_url="https://example.com/example",
    )

    results = pipeline.search(
        "analytics software",
        top_k=5,
        filters={"ticker": "OTHER"},
    )

    assert results == []


def test_rejects_model_dimension_mismatch():
    client = QdrantClient(":memory:")
    store = QdrantChunkStore(
        client,
        collection_name="wrong_dimension_test",
        vector_size=3,
    )

    with pytest.raises(ValueError, match="does not match"):
        FilingIngestionPipeline(store, FakeEmbeddingModel())

    client.close()


def test_blank_search_returns_empty(pipeline):
    assert pipeline.search("   ") == []
