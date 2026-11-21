"""Tests for FinMind Phase 2 Step 2 SEC benchmark dataset validation."""

import json
from pathlib import Path
import pytest

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "sec_filings"
SOURCE_FILINGS_PATH = BENCHMARK_DIR / "source_filings.json"
CORPUS_CHUNKS_PATH = BENCHMARK_DIR / "corpus_chunks.json"
EVALUATION_QUERIES_PATH = BENCHMARK_DIR / "evaluation_queries.json"


@pytest.fixture(scope="module")
def source_filings():
    assert SOURCE_FILINGS_PATH.exists(), f"Missing {SOURCE_FILINGS_PATH}"
    with open(SOURCE_FILINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def corpus_chunks():
    assert CORPUS_CHUNKS_PATH.exists(), f"Missing {CORPUS_CHUNKS_PATH}"
    with open(CORPUS_CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def evaluation_queries():
    assert EVALUATION_QUERIES_PATH.exists(), f"Missing {EVALUATION_QUERIES_PATH}"
    with open(EVALUATION_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_source_filings_schema_and_minimums(source_filings):
    """Source filings must contain at least 4 companies and both 10-K and 10-Q forms."""
    assert len(source_filings) >= 8
    tickers = {f["ticker"] for f in source_filings}
    forms = {f["form"] for f in source_filings}

    assert {"AAPL", "MSFT", "AMZN", "NVDA"}.issubset(tickers)
    assert {"10-K", "10-Q"}.issubset(forms)

    for item in source_filings:
        for required_field in (
            "ticker", "company_name", "cik", "form",
            "filing_date", "accession", "doc", "url", "sha256", "bytes"
        ):
            assert required_field in item
            assert item[required_field], f"Field {required_field} is empty in {item}"
        assert item["url"].startswith("https://www.sec.gov/")
        assert len(item["sha256"]) == 64


def test_corpus_chunks_minimum_count_and_uniqueness(corpus_chunks):
    """Corpus must have at least 500 valid, non-empty chunks with unique chunk_ids."""
    assert len(corpus_chunks) >= 500
    seen_ids = set()
    for chunk in corpus_chunks:
        cid = chunk["chunk_id"]
        assert cid not in seen_ids, f"Duplicate chunk_id detected: {cid}"
        seen_ids.add(cid)

        assert isinstance(chunk["text"], str) and len(chunk["text"].strip()) > 0
        assert chunk["ticker"] in {"AAPL", "MSFT", "AMZN", "NVDA"}
        assert chunk["form_type"] in {"10-K", "10-Q"}
        assert chunk["source_url"].startswith("https://www.sec.gov/")


def test_corpus_chunk_ids_are_deterministic(corpus_chunks):
    """Chunk IDs must follow the format TICKER:FILING_DATE:DOC:CHUNK_INDEX."""
    for chunk in corpus_chunks:
        expected_prefix = f"{chunk['ticker']}:{chunk['filing_date']}:{chunk['filename']}:{chunk['chunk_index']}"
        assert chunk["chunk_id"] == expected_prefix


def test_evaluation_queries_counts_and_separation(evaluation_queries):
    """Evaluation queries must have at least 50 queries: >=40 answerable, >=10 unanswerable."""
    assert len(evaluation_queries) >= 50

    answerable = [q for q in evaluation_queries if q["answerable"]]
    unanswerable = [q for q in evaluation_queries if not q["answerable"]]

    assert len(answerable) >= 40
    assert len(unanswerable) >= 10

    # Ensure query IDs are unique
    qids = [q["query_id"] for q in evaluation_queries]
    assert len(qids) == len(set(qids))


def test_answerable_queries_have_valid_supporting_evidence(evaluation_queries, corpus_chunks):
    """Every answerable query must reference existing chunks and valid evidence quotes."""
    chunk_map = {c["chunk_id"]: c for c in corpus_chunks}

    for q in evaluation_queries:
        if q["answerable"]:
            assert len(q["relevant_chunk_ids"]) > 0, f"Answerable query {q['query_id']} has no relevant_chunk_ids"
            assert q["evidence_quote"] is not None and len(q["evidence_quote"].strip()) > 0
            assert q["rationale"] is not None and len(q["rationale"].strip()) > 0

            # Verify every relevant chunk exists in corpus and contains or supports the query topic
            for cid in q["relevant_chunk_ids"]:
                assert cid in chunk_map, f"Query {q['query_id']} references nonexistent chunk {cid}"

            # Verify that the evidence quote is an exact, character-for-character substring of a cited chunk
            quote = q["evidence_quote"]
            assert any(
                quote in chunk_map[cid]["text"] for cid in q["relevant_chunk_ids"]
            ), f"evidence_quote for {q['query_id']} is not an exact substring of any cited chunk"
        else:
            assert len(q["relevant_chunk_ids"]) == 0, f"Unanswerable query {q['query_id']} must have empty relevant_chunk_ids"
            assert q["evidence_quote"] is None
            assert "Unanswerable" in q["rationale"] or "not in the corpus" in q["rationale"]
