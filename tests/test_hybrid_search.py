import pytest

from src.retrieval.hybrid_search import hybrid_search


class FakeModel:
    """Simple deterministic model for testing semantic retrieval."""

    def encode(self, texts, **kwargs):
        import numpy as np

        vectors = []
        for text in texts:
            text = str(text).lower()
            vectors.append(
                [
                    text.count("revenue") + 0.1,
                    text.count("risk") + 0.1,
                    text.count("debt") + 0.1,
                ]
            )
        return np.array(vectors, dtype=float)


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "c1",
            "text": "The company reported strong revenue growth.",
            "ticker": "AAA",
            "chunk_index": 0,
        },
        {
            "chunk_id": "c2",
            "text": "The company faces significant financial risk.",
            "ticker": "BBB",
            "chunk_index": 1,
        },
        {
            "chunk_id": "c3",
            "text": "Total debt increased during the fiscal year.",
            "ticker": "CCC",
            "chunk_index": 2,
        },
        {
            "chunk_id": "c4",
            "text": "Revenue and financial risk both increased.",
            "ticker": "DDD",
            "chunk_index": 3,
        },
    ]


def test_hybrid_search_returns_results(sample_chunks):
    results = hybrid_search(
        "revenue growth",
        sample_chunks,
        FakeModel(),
        top_k=3,
    )

    assert results
    assert len(results) <= 3
    assert all("hybrid_score" in result for result in results)


def test_hybrid_search_includes_score_fields(sample_chunks):
    results = hybrid_search("revenue", sample_chunks, FakeModel())

    assert results
    for result in results:
        assert "semantic_score" in result
        assert "bm25_score" in result
        assert "hybrid_score" in result


def test_hybrid_search_preserves_chunk_metadata(sample_chunks):
    results = hybrid_search("revenue", sample_chunks, FakeModel())

    assert results
    assert all("ticker" in result for result in results)
    assert all("chunk_id" in result for result in results)


def test_hybrid_search_returns_empty_for_blank_query(sample_chunks):
    results = hybrid_search("   ", sample_chunks, FakeModel())

    assert results == []


def test_hybrid_search_returns_empty_for_empty_chunks():
    results = hybrid_search("revenue", [], FakeModel())

    assert results == []


def test_hybrid_search_respects_top_k(sample_chunks):
    results = hybrid_search(
        "revenue risk",
        sample_chunks,
        FakeModel(),
        top_k=2,
    )

    assert len(results) <= 2


def test_hybrid_search_rejects_invalid_top_k(sample_chunks):
    with pytest.raises(ValueError):
        hybrid_search("revenue", sample_chunks, FakeModel(), top_k=0)


def test_hybrid_search_rejects_negative_weights(sample_chunks):
    with pytest.raises(ValueError):
        hybrid_search(
            "revenue",
            sample_chunks,
            FakeModel(),
            dense_weight=-0.1,
        )


def test_hybrid_search_does_not_modify_input_chunks(sample_chunks):
    original = [dict(chunk) for chunk in sample_chunks]

    hybrid_search("revenue", sample_chunks, FakeModel())

    assert sample_chunks == original
