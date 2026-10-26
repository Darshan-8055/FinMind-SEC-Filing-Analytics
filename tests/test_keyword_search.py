
import pytest

from src.retrieval.keyword_search import keyword_search


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "AAPL_001",
            "ticker": "AAPL",
            "section": "Risk Factors",
            "text": "Supply chain risks may affect product availability.",
        },
        {
            "chunk_id": "AAPL_002",
            "ticker": "AAPL",
            "section": "Financial Results",
            "text": "Revenue growth increased as revenue from services grew.",
        },
        {
            "chunk_id": "MSFT_001",
            "ticker": "MSFT",
            "section": "Financial Results",
            "text": "Revenue growth was driven by cloud services.",
        },
    ]


def test_finds_matching_chunks(sample_chunks):
    results = keyword_search("revenue growth", sample_chunks)

    assert len(results) == 2
    assert all("revenue" in result["text"].lower() for result in results)


def test_ranks_stronger_match_first(sample_chunks):
    results = keyword_search("revenue growth", sample_chunks)

    assert results[0]["chunk_id"] == "AAPL_002"


def test_preserves_original_metadata(sample_chunks):
    results = keyword_search("supply chain", sample_chunks)

    assert results[0]["ticker"] == "AAPL"
    assert results[0]["section"] == "Risk Factors"
    assert results[0]["chunk_id"] == "AAPL_001"


def test_returns_match_details(sample_chunks):
    results = keyword_search("revenue growth", sample_chunks)

    assert results[0]["keyword_score"] > 0
    assert set(results[0]["matched_terms"]) == {"revenue", "growth"}


def test_respects_top_k(sample_chunks):
    results = keyword_search("revenue growth", sample_chunks, top_k=1)

    assert len(results) == 1


def test_returns_empty_for_no_matches(sample_chunks):
    assert keyword_search("bankruptcy", sample_chunks) == []


@pytest.mark.parametrize("query", ["", "   ", "!!!"])
def test_returns_empty_for_empty_or_punctuation_query(
    sample_chunks, query
):
    assert keyword_search(query, sample_chunks) == []


def test_rejects_invalid_top_k(sample_chunks):
    with pytest.raises(ValueError):
        keyword_search("revenue", sample_chunks, top_k=0)


def test_rejects_non_string_query(sample_chunks):
    with pytest.raises(TypeError):
        keyword_search(None, sample_chunks)


def test_handles_chunk_without_text():
    results = keyword_search("revenue", [{"chunk_id": "EMPTY"}])

    assert results == []


def test_ignores_non_string_text():
    results = keyword_search(
        "revenue",
        [{"chunk_id": "BAD", "text": None}],
    )

    assert results == []
