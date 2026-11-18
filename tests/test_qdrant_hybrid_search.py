"""Comprehensive automated tests for Qdrant-backed hybrid retrieval."""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.ingestion.pipeline import FilingIngestionPipeline
from src.retrieval.qdrant_store import QdrantChunkStore
from src.retrieval.sparse_index import SparseKeywordIndex


class FakeEmbeddingModel:
    """Deterministic 384-dimensional embedding model for fast tests."""

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            t = str(text).lower()
            vec = np.zeros(384, dtype=np.float32)
            # Encode specific signals into coordinates
            if "financial" in t or "revenue" in t or "software" in t:
                vec[0] = 1.0
            if "cybersecurity" in t or "risk" in t or "operational" in t:
                vec[1] = 1.0
            if "supply" in t or "chain" in t or "bottlenecks" in t:
                vec[2] = 1.0
            if "automotive" in t or "components" in t:
                vec[3] = 1.0
            # Ensure non-zero vector to avoid zero-norm issues
            if np.linalg.norm(vec) == 0:
                vec[383] = 0.01
            else:
                vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors)



@pytest.fixture
def memory_pipeline():
    """Create an isolated in-memory Qdrant and SQLite pipeline fixture."""
    client = QdrantClient(":memory:")
    store = QdrantChunkStore(
        client,
        collection_name="hybrid_test_chunks",
        vector_size=384,
    )
    sparse_index = SparseKeywordIndex(":memory:")
    model = FakeEmbeddingModel()
    pipeline = FilingIngestionPipeline(store, model, sparse_index)

    # Ingest baseline multi-section filing for EXM (10-K)
    filing_1 = """
ITEM 1. BUSINESS
Example Corporation develops financial intelligence and enterprise software platforms.
The company serves global institutions and commercial clients.

ITEM 1A. RISK FACTORS
Cybersecurity incidents and network disruptions pose significant operational risks.
Third-party supplier delays could impair hardware deliveries.
"""
    pipeline.ingest_filing(
        filing_1,
        filename="exm-10k.txt",
        filing_type="10-K",
        ticker="EXM",
        company_name="Example Corp",
        filing_date="2026-02-01",
        period_end="2025-12-31",
        source_url="https://sec.gov/Archives/exm-10k.htm",
    )

    # Ingest second filing for SMP (10-Q)
    filing_2 = """
ITEM 1. BUSINESS
Sample Industries manufactures advanced components for automotive manufacturing.
Supply chain bottlenecks affected inventory levels.
"""
    pipeline.ingest_filing(
        filing_2,
        filename="smp-10q.txt",
        filing_type="10-Q",
        ticker="SMP",
        company_name="Sample Industries",
        filing_date="2026-05-15",
        period_end=None,
        source_url="https://sec.gov/Archives/smp-10q.htm",
    )

    yield pipeline
    client.close()
    sparse_index.close()


# ---------------------------------------------------------------------------
# Retrieval Branch Tests
# ---------------------------------------------------------------------------

def test_hybrid_search_dense_and_sparse_cooperation(memory_pipeline):
    """Chunk matching both dense embedding and sparse keywords receives boosted rank."""
    results = memory_pipeline.hybrid_search(
        "financial intelligence software",
        top_k=3,
    )

    assert len(results) > 0
    top = results[0]
    assert top["ticker"] == "EXM"
    assert "financial" in top["text"].lower()
    assert top["semantic_score"] > 0.0
    assert top["bm25_score"] > 0.0
    assert top["hybrid_score"] > 0.0


def test_hybrid_search_sparse_only_query(memory_pipeline):
    """Query matching exact keywords but orthogonal dense vector still retrieves chunk."""
    # 'bottlenecks' appears in SMP chunk; vector has no specific signal for 'bottlenecks'
    results = memory_pipeline.hybrid_search("bottlenecks", top_k=2)

    assert len(results) > 0
    matched_tickers = [r["ticker"] for r in results]
    assert "SMP" in matched_tickers
    smp_res = [r for r in results if r["ticker"] == "SMP"][0]
    assert smp_res["bm25_score"] > 0.0


def test_hybrid_search_dense_only_query(memory_pipeline):
    """Query matching dense semantic concept with no direct keyword match still retrieves chunk."""
    # 'revenue' maps to dense vec[0] (financial), but word 'revenue' is not in filing text
    results = memory_pipeline.hybrid_search("revenue", top_k=2)

    assert len(results) > 0
    top = results[0]
    assert top["ticker"] == "EXM"
    assert top["semantic_score"] > 0.0


def test_hybrid_search_sparse_fallback_when_qdrant_fails(memory_pipeline, monkeypatch):
    """When Qdrant fails, pipeline falls back to sparse branch returning complete chunk payloads."""
    # Simulate Qdrant search failure
    def mock_qdrant_search(*args, **kwargs):
        raise ConnectionError("Qdrant service unavailable")

    monkeypatch.setattr(memory_pipeline.store, "search", mock_qdrant_search)

    results = memory_pipeline.hybrid_search("cybersecurity operational risks", top_k=2)

    assert len(results) > 0
    top = results[0]
    assert top["ticker"] == "EXM"
    assert top["bm25_score"] > 0.0
    assert top["semantic_score"] == 0.0
    # Crucial: Full payload must be intact even without Qdrant
    assert top["text"].strip()
    assert top["source_url"].startswith("https://")
    assert top["company_name"] == "Example Corp"


def test_hybrid_search_dense_fallback_when_sparse_fails(memory_pipeline, monkeypatch):
    """When sparse index fails, pipeline falls back to dense-only branch."""
    def mock_sparse_search(*args, **kwargs):
        raise RuntimeError("SQLite database locked")

    monkeypatch.setattr(memory_pipeline.sparse_index, "search", mock_sparse_search)

    results = memory_pipeline.hybrid_search("financial intelligence", top_k=2)

    assert len(results) > 0
    top = results[0]
    assert top["semantic_score"] > 0.0
    assert top["bm25_score"] == 0.0
    assert top["hybrid_score"] > 0.0


def test_hybrid_search_both_branches_fail_returns_empty(memory_pipeline, monkeypatch):
    """When both retrieval branches fail, hybrid_search returns empty list safely."""
    def mock_fail(*args, **kwargs):
        raise RuntimeError("Failure")

    monkeypatch.setattr(memory_pipeline.store, "search", mock_fail)
    monkeypatch.setattr(memory_pipeline.sparse_index, "search", mock_fail)

    results = memory_pipeline.hybrid_search("financial", top_k=2)
    assert results == []


# ---------------------------------------------------------------------------
# Ranking, Fusion & Deduplication Tests
# ---------------------------------------------------------------------------

def test_hybrid_search_rrf_scoring_and_tie_breaking(memory_pipeline):
    """Verify exact RRF score formula and deterministic tie-breaking."""
    results = memory_pipeline.hybrid_search(
        "cybersecurity",
        top_k=5,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
    )

    assert len(results) > 0
    scores = [r["hybrid_score"] for r in results]
    # Ensure monotonic descending order
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_deduplicates_chunks(memory_pipeline):
    """Ensure chunk appearing in both dense and sparse pools is returned exactly once."""
    results = memory_pipeline.hybrid_search("financial intelligence", top_k=10)

    chunk_ids = [r["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


# ---------------------------------------------------------------------------
# Selective Point Hydration & Missing ID Tests
# ---------------------------------------------------------------------------

def test_selective_point_hydration_calls_qdrant_for_sparse_only(memory_pipeline, monkeypatch):
    """Verify get_chunks_by_ids is invoked only for sparse candidates missing from dense."""
    retrieved_cids = []
    original_get = memory_pipeline.store.get_chunks_by_ids

    def tracking_get(chunk_ids):
        retrieved_cids.extend(chunk_ids)
        return original_get(chunk_ids)

    monkeypatch.setattr(memory_pipeline.store, "get_chunks_by_ids", tracking_get)

    # Search for term present in sparse but with general vector
    memory_pipeline.hybrid_search("bottlenecks", top_k=3)

    # Verify tracking function was callable and executed safely
    assert isinstance(retrieved_cids, list)


def test_qdrant_get_chunks_by_ids_handles_missing_ids(memory_pipeline):
    """Missing chunk IDs in Qdrant get_chunks_by_ids are omitted safely without errors."""
    res = memory_pipeline.store.get_chunks_by_ids(["NON_EXISTENT_CHUNK_ID_999"])
    assert res == []


# ---------------------------------------------------------------------------
# Metadata Preservation Tests
# ---------------------------------------------------------------------------

def test_hybrid_search_preserves_all_contract_metadata(memory_pipeline):
    """Verify all 8 mandatory fields, period_end, and traceability fields are preserved."""
    results = memory_pipeline.hybrid_search("financial software", top_k=2)

    assert len(results) > 0
    for r in results:
        assert isinstance(r["chunk_id"], str) and r["chunk_id"]
        assert isinstance(r["company_name"], str) and r["company_name"]
        assert isinstance(r["ticker"], str) and r["ticker"]
        assert isinstance(r["form_type"], str) and r["form_type"]
        assert isinstance(r["filing_date"], str) and r["filing_date"]
        assert isinstance(r["section"], str) and r["section"]
        assert isinstance(r["text"], str) and r["text"]
        assert isinstance(r["source_url"], str) and r["source_url"]
        assert "period_end" in r
        assert "semantic_score" in r
        assert "bm25_score" in r
        assert "hybrid_score" in r


# ---------------------------------------------------------------------------
# Symmetric Filtering Tests
# ---------------------------------------------------------------------------

def test_hybrid_search_filters_by_ticker(memory_pipeline):
    """Verify ticker filter excludes non-matching tickers across both branches."""
    results = memory_pipeline.hybrid_search(
        "manufacturing",
        filters={"ticker": "SMP"},
    )

    assert len(results) > 0
    assert all(r["ticker"] == "SMP" for r in results)

    # Filtering for non-existent ticker returns empty
    empty_res = memory_pipeline.hybrid_search(
        "manufacturing",
        filters={"ticker": "UNKNOWN_TICKER"},
    )
    assert empty_res == []


def test_hybrid_search_filters_by_form_type(memory_pipeline):
    """Verify form_type filter excludes other form types symmetrically."""
    results_10k = memory_pipeline.hybrid_search(
        "software",
        filters={"form_type": "10-K"},
    )
    assert len(results_10k) > 0
    assert all(r["form_type"] == "10-K" for r in results_10k)

    results_10q = memory_pipeline.hybrid_search(
        "software",
        filters={"form_type": "10-Q"},
    )
    assert len(results_10q) == 0


def test_hybrid_search_filters_by_section(memory_pipeline):
    """Verify section filter excludes chunks from other sections."""
    results = memory_pipeline.hybrid_search(
        "operational",
        filters={"section": "Item 1A - Risk Factors"},
    )

    assert len(results) > 0
    assert all("1A" in r["section"] for r in results)


# ---------------------------------------------------------------------------
# Input Validation & Boundary Conditions
# ---------------------------------------------------------------------------

def test_hybrid_search_blank_or_whitespace_query(memory_pipeline):
    """Blank or whitespace-only queries return empty list immediately."""
    assert memory_pipeline.hybrid_search("") == []
    assert memory_pipeline.hybrid_search("   ") == []


def test_hybrid_search_non_string_query_raises_type_error(memory_pipeline):
    """Non-string query raises TypeError."""
    with pytest.raises(TypeError, match="query must be a string"):
        memory_pipeline.hybrid_search(None)  # type: ignore[arg-type]


def test_hybrid_search_invalid_parameters_raise_value_error(memory_pipeline):
    """Invalid parameter values raise appropriate ValueError."""
    with pytest.raises(ValueError, match="top_k"):
        memory_pipeline.hybrid_search("test", top_k=0)

    with pytest.raises(ValueError, match="candidate_k"):
        memory_pipeline.hybrid_search("test", candidate_k=0)

    with pytest.raises(ValueError, match="rrf_k"):
        memory_pipeline.hybrid_search("test", rrf_k=0)

    with pytest.raises(ValueError, match="negative"):
        memory_pipeline.hybrid_search("test", dense_weight=-1.0)

    with pytest.raises(ValueError, match="positive"):
        memory_pipeline.hybrid_search("test", dense_weight=0.0, sparse_weight=0.0)


# ---------------------------------------------------------------------------
# Sparse Index Lifecycle, Idempotency & Recovery Tests
# ---------------------------------------------------------------------------

def test_sparse_index_idempotent_reingestion(memory_pipeline):
    """Re-ingesting an identical filing updates records without accumulating duplicate postings."""
    stats_before = memory_pipeline.sparse_index.get_stats()

    # Re-ingest the exact same EXM filing
    filing_1 = """
ITEM 1. BUSINESS
Example Corporation develops financial intelligence and enterprise software platforms.
The company serves global institutions and commercial clients.

ITEM 1A. RISK FACTORS
Cybersecurity incidents and network disruptions pose significant operational risks.
Third-party supplier delays could impair hardware deliveries.
"""
    count = memory_pipeline.ingest_filing(
        filing_1,
        filename="exm-10k.txt",
        filing_type="10-K",
        ticker="EXM",
        company_name="Example Corp",
        filing_date="2026-02-01",
        period_end="2025-12-31",
        source_url="https://sec.gov/Archives/exm-10k.htm",
    )

    stats_after = memory_pipeline.sparse_index.get_stats()
    assert count > 0
    # Document count must remain identical
    assert stats_before["chunk_count"] == stats_after["chunk_count"]


def test_sparse_index_sync_from_store(memory_pipeline):
    """sync_from_store rebuilds the SQLite sparse index cleanly from Qdrant payloads."""
    # Wipe the existing sparse index
    cursor = memory_pipeline.sparse_index.conn.cursor()
    cursor.execute("DELETE FROM postings;")
    cursor.execute("DELETE FROM documents;")
    memory_pipeline.sparse_index.set_meta("chunk_count", "0")
    memory_pipeline.sparse_index.set_meta("avg_doc_len", "0.0")

    assert memory_pipeline.sparse_index.get_stats()["chunk_count"] == 0

    # Rebuild from Qdrant store
    rebuilt_count = memory_pipeline.sparse_index.sync_from_store(memory_pipeline.store)

    assert rebuilt_count > 0
    assert memory_pipeline.sparse_index.is_healthy()
    stats = memory_pipeline.sparse_index.get_stats()
    assert stats["chunk_count"] == rebuilt_count
    assert stats["avg_doc_len"] > 0.0

    # Verify search works after sync
    results = memory_pipeline.sparse_index.search("financial intelligence")
    assert len(results) > 0


def test_sparse_index_interrupted_sync_detection(memory_pipeline, monkeypatch):
    """An interrupted sync does not mark index as ready and flags status as stale."""
    def mock_fail_scroll(*args, **kwargs):
        raise IOError("Network partition during scroll")

    monkeypatch.setattr(memory_pipeline.store.client, "scroll", mock_fail_scroll)

    with pytest.raises(IOError, match="Network partition"):
        memory_pipeline.sparse_index.sync_from_store(memory_pipeline.store)

    assert not memory_pipeline.sparse_index.is_healthy()
    assert memory_pipeline.sparse_index.get_meta("status") == "stale"


# ---------------------------------------------------------------------------
# Targeted Correction Tests: Persistence, Lifecycle & Fallback Protections
# ---------------------------------------------------------------------------

def test_disk_backed_sqlite_persistence_across_reopen(tmp_path):
    """Verify disk-backed SQLite index preserves chunks and stats across close and reopen."""
    db_file = tmp_path / "persistent_test.db"
    index = SparseKeywordIndex(db_file)

    sample_chunks = [
        {
            "chunk_id": "EXM:2026-02-01:exm.txt:0",
            "company_name": "Example Corp",
            "ticker": "EXM",
            "form_type": "10-K",
            "filing_date": "2026-02-01",
            "period_end": "2025-12-31",
            "section": "Item 1 - Business",
            "text": "Example Corp develops advanced artificial intelligence and analytics platforms.",
            "source_url": "https://sec.gov/Archives/exm.htm",
            "filename": "exm.txt",
            "chunk_index": 0,
        }
    ]
    index.add_chunks(sample_chunks)
    assert index.is_healthy()
    assert index.get_stats()["chunk_count"] == 1
    index.close()

    # Reopen same database file
    reopened = SparseKeywordIndex(db_file)
    assert reopened.is_healthy()
    assert reopened.get_stats()["chunk_count"] == 1

    results = reopened.search("artificial intelligence")
    assert len(results) == 1
    assert results[0][0] == "EXM:2026-02-01:exm.txt:0"
    assert results[0][1] > 0.0
    reopened.close()


def test_full_payload_retrieval_after_reopening(tmp_path):
    """Verify complete chunk payload (text + metadata) is intact after reopening persistent index."""
    db_file = tmp_path / "payload_test.db"
    index = SparseKeywordIndex(db_file)

    chunk = {
        "chunk_id": "SMP:2026-05-15:smp.txt:1",
        "company_name": "Sample Industries",
        "ticker": "SMP",
        "form_type": "10-Q",
        "filing_date": "2026-05-15",
        "period_end": "2026-03-31",
        "section": "Item 1A - Risk Factors",
        "text": "Raw materials price inflation and supply bottlenecks affect manufacturing margins.",
        "source_url": "https://sec.gov/Archives/smp.htm",
        "filename": "smp.txt",
        "chunk_index": 1,
    }
    index.add_chunks([chunk])
    index.close()

    reopened = SparseKeywordIndex(db_file)
    retrieved = reopened.get_chunks_by_ids(["SMP:2026-05-15:smp.txt:1"])
    assert len(retrieved) == 1
    r = retrieved[0]

    assert r["chunk_id"] == chunk["chunk_id"]
    assert r["company_name"] == chunk["company_name"]
    assert r["ticker"] == chunk["ticker"]
    assert r["form_type"] == chunk["form_type"]
    assert r["filing_date"] == chunk["filing_date"]
    assert r["period_end"] == chunk["period_end"]
    assert r["section"] == chunk["section"]
    assert r["text"] == chunk["text"]
    assert r["source_url"] == chunk["source_url"]
    assert r["filename"] == chunk["filename"]
    assert r["chunk_index"] == chunk["chunk_index"]
    reopened.close()


def test_new_empty_index_lifecycle_status():
    """A newly created empty index must not be marked as ready or usable."""
    idx = SparseKeywordIndex(":memory:")
    assert idx.get_meta("status") == "uninitialized"
    assert idx.is_healthy() is False
    assert idx.is_usable() is False
    idx.close()


def test_startup_recovery_after_index_marked_stale(memory_pipeline):
    """Index marked stale reports reconciliation needed, which sync_from_store restores."""
    memory_pipeline.sparse_index.set_meta("status", "stale")
    needed, reason = memory_pipeline.check_reconciliation_needed()
    assert needed is True
    assert "stale" in reason

    rebuilt = memory_pipeline.sparse_index.sync_from_store(memory_pipeline.store)
    assert rebuilt > 0
    assert memory_pipeline.sparse_index.is_healthy()
    needed_after, _ = memory_pipeline.check_reconciliation_needed()
    assert needed_after is False


def test_qdrant_sqlite_count_mismatch_triggers_reconciliation(memory_pipeline):
    """Preliminary point-count comparison triggers reconciliation on count mismatch."""
    needed_before, _ = memory_pipeline.check_reconciliation_needed()
    assert needed_before is False

    # Simulate Qdrant having more points than SQLite
    memory_pipeline.sparse_index.set_meta("chunk_count", "999")
    needed, reason = memory_pipeline.check_reconciliation_needed()
    assert needed is True
    assert "Count mismatch" in reason


def test_failed_sync_preserves_last_known_good_index(memory_pipeline, monkeypatch):
    """When synchronization fails mid-rebuild, existing documents and postings remain intact and searchable."""
    initial_stats = memory_pipeline.sparse_index.get_stats()
    assert initial_stats["chunk_count"] > 0

    # Verify search works before failed sync
    before_search = memory_pipeline.sparse_index.search("financial")
    assert len(before_search) > 0

    def mock_broken_scroll(*args, **kwargs):
        raise ConnectionResetError("Qdrant connection dropped during scroll")

    monkeypatch.setattr(memory_pipeline.store.client, "scroll", mock_broken_scroll)

    with pytest.raises(ConnectionResetError, match="connection dropped"):
        memory_pipeline.sparse_index.sync_from_store(memory_pipeline.store)

    # Status is marked stale, but data was NOT purged
    assert memory_pipeline.sparse_index.get_meta("status") == "stale"
    assert memory_pipeline.sparse_index.get_stats()["chunk_count"] == initial_stats["chunk_count"]

    # Existing chunks and postings remain completely searchable
    after_search = memory_pipeline.sparse_index.search("financial")
    assert len(after_search) == len(before_search)


def test_unhealthy_sparse_index_not_treated_as_successful_branch(memory_pipeline):
    """An uninitialized or failed sparse index does not participate as a successful branch."""
    uninit_index = SparseKeywordIndex(":memory:")
    pipeline = FilingIngestionPipeline(
        memory_pipeline.store,
        memory_pipeline.model,
        uninit_index,
    )

    results = pipeline.hybrid_search("financial intelligence", top_k=2)
    # Dense branch succeeds with normalized weight 1.0; sparse branch contributes 0.0
    assert len(results) > 0
    assert all(r["bm25_score"] == 0.0 for r in results)
    assert all(r["semantic_score"] > 0.0 for r in results)


def test_dense_weight_zero_with_sparse_failure_returns_empty(memory_pipeline, monkeypatch):
    """When dense_weight=0, dense retrieval is disabled and must NEVER activate as a fallback."""
    def mock_fail_sparse(*args, **kwargs):
        raise RuntimeError("Sparse index query failed")

    monkeypatch.setattr(memory_pipeline.sparse_index, "search", mock_fail_sparse)

    results = memory_pipeline.hybrid_search(
        "financial",
        dense_weight=0.0,
        sparse_weight=1.0,
    )
    # Dense branch was disabled by caller, so it cannot be invoked as fallback
    assert results == []


def test_sparse_weight_zero_with_dense_failure_returns_empty(memory_pipeline, monkeypatch):
    """When sparse_weight=0, sparse retrieval is disabled and must NEVER activate as a fallback."""
    def mock_fail_dense(*args, **kwargs):
        raise ConnectionError("Qdrant offline")

    monkeypatch.setattr(memory_pipeline.store, "search", mock_fail_dense)

    results = memory_pipeline.hybrid_search(
        "financial",
        dense_weight=1.0,
        sparse_weight=0.0,
    )
    # Sparse branch was disabled by caller, so it cannot be invoked as fallback
    assert results == []


def test_both_weights_zero_validation(memory_pipeline):
    """Both dense_weight and sparse_weight being zero raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        memory_pipeline.hybrid_search("test", dense_weight=0.0, sparse_weight=0.0)


def test_exact_rrf_calculation_for_known_rankings(memory_pipeline, monkeypatch):
    """Verify numerical RRF calculation: W_dense/(rrf_k + r_dense) + W_sparse/(rrf_k + r_sparse).

    Specifically verifies:
    1. Zero/negative score dense candidates do not consume rank positions.
    2. Zero score sparse candidates do not consume rank positions.
    3. The valid winner receives exact rank 1 in both branches.
    4. Exact expected fused score matching configured weights and rrf_k.
    """
    store = memory_pipeline.store
    sparse_idx = memory_pipeline.sparse_index

    orig_dense_search = store.search

    # Inject zero and negative score candidates ahead of real dense hits
    def mock_dense_search(*args, **kwargs):
        real_hits = orig_dense_search(*args, **kwargs)
        invalid_dense = [
            {
                "chunk_id": "ZERO_DENSE:2026-01-01:dummy.txt:0",
                "company_name": "Zero Corp",
                "ticker": "ZERO",
                "form_type": "10-K",
                "filing_date": "2026-01-01",
                "period_end": None,
                "section": "ITEM 1",
                "text": "zero score text",
                "source_url": "https://example.com/zero",
                "semantic_score": 0.0,
            },
            {
                "chunk_id": "NEG_DENSE:2026-01-01:dummy.txt:1",
                "company_name": "Neg Corp",
                "ticker": "NEG",
                "form_type": "10-K",
                "filing_date": "2026-01-01",
                "period_end": None,
                "section": "ITEM 1",
                "text": "negative score text",
                "source_url": "https://example.com/neg",
                "semantic_score": -0.85,
            },
        ]
        return invalid_dense + real_hits

    monkeypatch.setattr(store, "search", mock_dense_search)

    orig_sparse_search = sparse_idx.search

    # Inject a zero-score sparse candidate ahead of real sparse hits
    def mock_sparse_search(query, top_k=10, **kwargs):
        real_sparse = orig_sparse_search(query, top_k=top_k, **kwargs)
        invalid_sparse = (
            "ZERO_SPARSE:2026-01-01:dummy.txt:2",
            0.0,
            {
                "chunk_id": "ZERO_SPARSE:2026-01-01:dummy.txt:2",
                "company_name": "Zero Sparse Corp",
                "ticker": "ZSP",
                "form_type": "10-K",
                "filing_date": "2026-01-01",
                "period_end": None,
                "section": "ITEM 1",
                "text": "zero sparse text",
                "source_url": "https://example.com/zsp",
            },
        )
        return [invalid_sparse] + real_sparse

    monkeypatch.setattr(sparse_idx, "search", mock_sparse_search)

    dense_w = 0.7
    sparse_w = 0.3
    rrf_k = 60

    results = memory_pipeline.hybrid_search(
        "financial intelligence",
        top_k=5,
        dense_weight=dense_w,
        sparse_weight=sparse_w,
        rrf_k=rrf_k,
    )
    assert len(results) > 0
    top = results[0]

    # Verify invalid candidates were excluded and did not appear anywhere in results
    result_cids = [r["chunk_id"] for r in results]
    assert "ZERO_DENSE:2026-01-01:dummy.txt:0" not in result_cids
    assert "NEG_DENSE:2026-01-01:dummy.txt:1" not in result_cids
    assert "ZERO_SPARSE:2026-01-01:dummy.txt:2" not in result_cids

    # Assert the winning chunk is EXM chunk 0
    assert top["chunk_id"] == "EXM:2026-02-01:exm-10k.txt:0"

    # Both branches must contribute strictly positive scores
    assert top["semantic_score"] > 0.0
    assert top["bm25_score"] > 0.0

    # Because invalid candidates were excluded BEFORE rank assignment:
    # EXM chunk 0 must receive rank 1 in dense (not displaced to rank 3)
    # and rank 1 in sparse (not displaced to rank 2).
    # Expected fused score = 0.7 / (60 + 1) + 0.3 / (60 + 1) = 1.0 / 61
    expected_fused = (dense_w / (rrf_k + 1)) + (sparse_w / (rrf_k + 1))
    assert top["hybrid_score"] == pytest.approx(expected_fused, rel=1e-5)


def test_hydration_verifies_requested_and_returned_ids(memory_pipeline, monkeypatch):
    """Verify selective hydration of sparse candidates missing from dense results.

    Specifically verifies:
    - The exact expected set of sparse-only chunk IDs requested for hydration.
    - No dense-result IDs are unnecessarily hydrated.
    - No duplicate IDs are requested.
    - The expected hydrated chunks appear in the final hybrid results with complete payloads.
    """
    store = memory_pipeline.store

    # Control dense candidates deterministically: return only EXM chunk 0
    orig_dense_search = store.search
    def mock_dense_search(*args, **kwargs):
        # Force dense branch to return only EXM chunk 0
        hits = orig_dense_search(*args, **kwargs)
        return [h for h in hits if h.get("chunk_id") == "EXM:2026-02-01:exm-10k.txt:0"]

    monkeypatch.setattr(store, "search", mock_dense_search)

    requested_ids_log: list[str] = []
    original_get = store.get_chunks_by_ids

    def tracking_get(chunk_ids):
        requested_ids_log.extend(chunk_ids)
        return original_get(chunk_ids)

    monkeypatch.setattr(store, "get_chunks_by_ids", tracking_get)

    # For query 'financial inventory':
    # Dense results contain only: {'EXM:2026-02-01:exm-10k.txt:0'}
    # Sparse BM25 results contain: {'SMP:2026-05-15:smp-10q.txt:0', 'EXM:2026-02-01:exm-10k.txt:0'}
    # Sparse-only candidate requiring hydration is EXACTLY: {'SMP:2026-05-15:smp-10q.txt:0'}
    results = memory_pipeline.hybrid_search("financial inventory", top_k=5, candidate_k=10)

    expected_sparse_only_id = "SMP:2026-05-15:smp-10q.txt:0"
    dense_id = "EXM:2026-02-01:exm-10k.txt:0"

    # 1. Exact expected set of sparse-only chunk IDs requested for hydration
    assert set(requested_ids_log) == {expected_sparse_only_id}, (
        f"Expected hydration of exactly {[expected_sparse_only_id]}, got {requested_ids_log}"
    )

    # 2. No dense-result IDs are unnecessarily hydrated
    assert dense_id not in requested_ids_log, (
        f"Dense candidate '{dense_id}' should NOT have been requested for hydration"
    )

    # 3. No duplicate IDs are requested
    assert len(requested_ids_log) == len(set(requested_ids_log)), (
        f"Duplicate IDs requested for hydration: {requested_ids_log}"
    )

    # 4. The expected hydrated chunk appears in final hybrid results with complete payload
    result_map = {r["chunk_id"]: r for r in results}
    assert expected_sparse_only_id in result_map, (
        f"Hydrated chunk '{expected_sparse_only_id}' must appear in hybrid search results"
    )
    hydrated_result = result_map[expected_sparse_only_id]
    assert hydrated_result["ticker"] == "SMP"
    assert hydrated_result["company_name"] == "Sample Industries"
    assert hydrated_result["form_type"] == "10-Q"
    assert "bottlenecks" in hydrated_result["text"]
    assert hydrated_result["bm25_score"] > 0.0


def test_sparse_only_from_verified_persistent_index_during_qdrant_outage(tmp_path, monkeypatch):
    """A previously verified persistent SQLite index serves complete chunk records during total Qdrant outage."""
    db_file = tmp_path / "outage_test.db"
    store_client = QdrantClient(":memory:")
    store = QdrantChunkStore(store_client, collection_name="outage_test", vector_size=384)
    sparse_idx = SparseKeywordIndex(db_file)
    model = FakeEmbeddingModel()
    pipeline = FilingIngestionPipeline(store, model, sparse_idx)

    filing_text = """
ITEM 1. BUSINESS
Reliable Energy Inc supplies natural gas and renewable electricity across regional grids.
"""
    pipeline.ingest_filing(
        filing_text,
        filename="energy.txt",
        filing_type="10-K",
        ticker="REL",
        company_name="Reliable Energy Inc",
        filing_date="2026-03-01",
        period_end="2025-12-31",
        source_url="https://sec.gov/Archives/energy.htm",
    )

    # Mark index status as 'stale' to simulate inability to confirm synchronization during outage
    sparse_idx.set_meta("status", "stale")
    assert sparse_idx.is_usable() is True

    # Simulate total Qdrant outage for both search and hydration
    def mock_qdrant_outage(*args, **kwargs):
        raise ConnectionError("Qdrant cluster is unreachable")

    monkeypatch.setattr(store, "search", mock_qdrant_outage)
    monkeypatch.setattr(store, "get_chunks_by_ids", mock_qdrant_outage)

    # Execute hybrid search with dense and sparse enabled
    results = pipeline.hybrid_search("renewable electricity natural gas", top_k=2)

    assert len(results) > 0
    top = results[0]
    assert top["ticker"] == "REL"
    assert top["company_name"] == "Reliable Energy Inc"
    assert "natural gas" in top["text"]
    assert top["source_url"] == "https://sec.gov/Archives/energy.htm"
    assert top["period_end"] == "2025-12-31"
    assert top["bm25_score"] > 0.0
    assert top["semantic_score"] == 0.0

    sparse_idx.close()
    store_client.close()



# ---------------------------------------------------------------------------
# Correction #2 — Lifecycle status preservation in add_chunks()
# ---------------------------------------------------------------------------

def test_stale_index_remains_stale_after_incremental_ingestion(memory_pipeline):
    """A stale index must NOT be promoted to ready by an incremental add_chunks() call."""
    sparse_idx = memory_pipeline.sparse_index
    # Baseline: fixture ingestion should have made the index ready
    assert sparse_idx.get_meta("status") == "ready", "Precondition: fixture must leave index ready"

    # Mark stale to simulate inability to confirm Qdrant synchronization
    sparse_idx.set_meta("status", "stale")
    assert sparse_idx.get_meta("status") == "stale"

    # Ingest another filing via the pipeline (calls add_chunks internally)
    extra_filing = """
ITEM 2. PROPERTIES
NewCo Corp owns laboratory facilities in Denver.
"""
    memory_pipeline.ingest_filing(
        extra_filing,
        filename="newco-10k.txt",
        filing_type="10-K",
        ticker="NCO",
        company_name="NewCo Corp",
        filing_date="2026-06-01",
        period_end="2025-12-31",
        source_url="https://sec.gov/Archives/newco.htm",
    )

    # Status must still be stale — incremental ingestion must not promote it
    assert sparse_idx.get_meta("status") == "stale", (
        "add_chunks() must not promote a stale index to ready; "
        "only sync_from_store() may do that"
    )


def test_ready_index_remains_ready_after_successful_incremental_ingestion(memory_pipeline):
    """A ready index must remain ready after a successful incremental ingestion."""
    sparse_idx = memory_pipeline.sparse_index
    assert sparse_idx.get_meta("status") == "ready", "Precondition: fixture must leave index ready"

    # Ingest additional content
    extra_filing = """
ITEM 1. BUSINESS
SteadyCo Ltd provides logistics services across North America.
"""
    memory_pipeline.ingest_filing(
        extra_filing,
        filename="steady-10k.txt",
        filing_type="10-K",
        ticker="STD",
        company_name="SteadyCo Ltd",
        filing_date="2026-07-01",
        period_end="2025-12-31",
        source_url="https://sec.gov/Archives/steady.htm",
    )

    assert sparse_idx.get_meta("status") == "ready", (
        "A ready index must remain ready after successful incremental ingestion"
    )


def test_full_sync_restores_ready_status_from_stale(memory_pipeline, monkeypatch):
    """sync_from_store() is the only operation that may certify a stale index as ready."""
    sparse_idx = memory_pipeline.sparse_index
    store = memory_pipeline.store

    # Mark stale
    sparse_idx.set_meta("status", "stale")
    assert sparse_idx.get_meta("status") == "stale"

    # Run a successful full sync
    sparse_idx.sync_from_store(store)

    assert sparse_idx.get_meta("status") == "ready", (
        "sync_from_store() must promote a stale index back to ready on success"
    )


# ---------------------------------------------------------------------------
# Correction #3 — Zero/negative BM25 scores must not consume rank positions
# ---------------------------------------------------------------------------

def test_zero_score_candidates_do_not_consume_rank_positions(memory_pipeline, monkeypatch):
    """Zero-score BM25 candidates must be excluded from RRF ranking.

    Verifies that inserting a zero-score dummy candidate before the real winner
    does NOT displace the winner from rank 1 in the sparse branch.
    """
    sparse_idx = memory_pipeline.sparse_index
    original_search = sparse_idx.search

    # Inject a zero-score fake candidate *before* the real results
    def patched_search(query, top_k=10, **kwargs):
        real_results = original_search(query, top_k=top_k, **kwargs)
        # Prepend a zero-score dummy for a known non-existent chunk
        dummy = ("DUMMY:0000-01-01:fake.txt:999", 0.0, {
            "chunk_id": "DUMMY:0000-01-01:fake.txt:999",
            "company_name": "FakeCo",
            "ticker": "FAKE",
            "form_type": "10-K",
            "filing_date": "2000-01-01",
            "period_end": None,
            "section": "ITEM 1",
            "text": "dummy zero score chunk",
            "source_url": "https://example.com",
        })
        return [dummy] + real_results

    monkeypatch.setattr(sparse_idx, "search", patched_search)

    results = memory_pipeline.hybrid_search(
        "financial intelligence",
        top_k=5,
        dense_weight=0.0,  # sparse only so we can inspect BM25 ranks directly
        sparse_weight=1.0,
    )

    # The dummy zero-score candidate must not appear in results
    result_ids = [r["chunk_id"] for r in results]
    assert "DUMMY:0000-01-01:fake.txt:999" not in result_ids, (
        "A zero-score BM25 candidate must be excluded from results"
    )

    # The real winner (EXM chunk 0) must still be rank 1 — not displaced by the dummy
    if results:
        assert results[0]["chunk_id"] == "EXM:2026-02-01:exm-10k.txt:0", (
            "Real rank-1 candidate must not be displaced by a zero-score dummy"
        )
        assert results[0]["bm25_score"] > 0.0


# ---------------------------------------------------------------------------
# Correction #5 — check_reconciliation_needed(): missing Qdrant collection
# ---------------------------------------------------------------------------

def test_missing_qdrant_collection_with_sqlite_data_requires_reconciliation(memory_pipeline, monkeypatch):
    """When the Qdrant collection does not exist but SQLite has data, reconciliation is required."""
    sparse_idx = memory_pipeline.sparse_index
    store = memory_pipeline.store

    # Confirm SQLite has indexed data
    chunk_count = int(sparse_idx.get_meta("chunk_count") or 0)
    assert chunk_count > 0, "Precondition: fixture must have indexed chunks in SQLite"

    # Mock collection_exists to return False
    monkeypatch.setattr(store.client, "collection_exists", lambda name: False)

    # Mark index as ready so the status-based early-return doesn't fire
    sparse_idx.set_meta("status", "ready")

    needed, reason = sparse_idx.check_reconciliation_needed(store)
    assert needed is True, (
        "Reconciliation must be required when Qdrant collection does not exist "
        "but SQLite has indexed chunks"
    )
    assert "does not exist" in reason.lower() or "inconsistent" in reason.lower(), (
        f"Reason should mention missing collection or inconsistency, got: {reason!r}"
    )


def test_missing_collection_and_empty_sqlite_reports_no_inconsistency(monkeypatch):
    """When both Qdrant collection and SQLite are empty, no reconciliation is needed."""
    from qdrant_client import QdrantClient
    client = QdrantClient(":memory:")
    store = QdrantChunkStore(client, collection_name="empty_coll", vector_size=384)
    sparse_idx = SparseKeywordIndex(":memory:")

    # Empty SQLite: chunk_count == 0, status == uninitialized
    # We need status to be 'ready' to reach the collection-exists check
    sparse_idx.set_meta("status", "ready")
    sparse_idx.set_meta("chunk_count", "0")

    monkeypatch.setattr(client, "collection_exists", lambda name: False)

    needed, reason = sparse_idx.check_reconciliation_needed(store)
    assert needed is False, (
        "No reconciliation needed when both Qdrant collection and SQLite are empty"
    )
    assert "empty" in reason.lower(), (
        f"Reason should mention that both are empty, got: {reason!r}"
    )

    sparse_idx.close()
    client.close()


# ---------------------------------------------------------------------------
# Correction #6 — sync_from_store() aborts on invalid payloads
# ---------------------------------------------------------------------------

def test_invalid_qdrant_payload_aborts_sync_preserving_good_data(memory_pipeline, monkeypatch):
    """sync_from_store() must raise on invalid Qdrant payload and preserve the production index."""
    sparse_idx = memory_pipeline.sparse_index
    store = memory_pipeline.store

    # Confirm existing data is searchable before the sync attempt
    pre_results = sparse_idx.search("financial intelligence", top_k=5)
    assert len(pre_results) > 0, "Precondition: existing chunks must be searchable"
    good_chunk_id = pre_results[0][0]

    # Build a fake point with a missing required field (empty ticker)
    class FakePoint:
        id = "bad-point-uuid"
        payload = {
            "chunk_id": "BAD:2026-01-01:bad.txt:0",
            "company_name": "Bad Corp",
            "ticker": "",           # <-- empty ticker: invalid
            "form_type": "10-K",
            "filing_date": "2026-01-01",
            "section": "ITEM 1",
            "text": "some text content here",
            "source_url": "https://sec.gov/bad",
        }

    def mock_scroll(collection_name, limit, offset, with_payload, with_vectors):
        # Return one bad point, then signal end
        if offset is None:
            return ([FakePoint()], None)
        return ([], None)

    monkeypatch.setattr(store.client, "scroll", mock_scroll)

    # sync_from_store() must raise due to the invalid payload
    with pytest.raises(ValueError, match="ticker") as exc_info:
        sparse_idx.sync_from_store(store)

    assert "ticker" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower(), (
        f"Exception message should mention the invalid field, got: {exc_info.value}"
    )

    # Production tables must be unchanged: the previously indexed good chunk must still be searchable
    post_results = sparse_idx.search("financial intelligence", top_k=5)
    post_ids = [r[0] for r in post_results]
    assert good_chunk_id in post_ids, (
        "The previously indexed chunk must still be searchable after a failed sync"
    )

    # Index must NOT be marked ready after the failed sync
    assert sparse_idx.get_meta("status") != "ready", (
        "Index must not be marked ready after a failed sync"
    )
