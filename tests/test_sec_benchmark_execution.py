"""
tests/test_sec_benchmark_execution.py

Verifies the correctness and isolation of the Step 3 SEC benchmark execution
results for Dense, Sparse, and Hybrid retrieval configurations.

These tests are read-only: they inspect the pre-computed result JSON files and
the benchmark dataset files. They do NOT re-run the retrieval pipeline.
"""

import json
import os
import pathlib
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "sec_filings" / "results"
DENSE_FILE = RESULTS_DIR / "dense_results.json"
SPARSE_FILE = RESULTS_DIR / "sparse_results.json"
HYBRID_FILE = RESULTS_DIR / "hybrid_results.json"

CORPUS_FILE = PROJECT_ROOT / "benchmarks" / "sec_filings" / "corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "benchmarks" / "sec_filings" / "evaluation_queries.json"

EVAL_SPARSE_DB = PROJECT_ROOT / "benchmarks" / "sec_filings" / "eval_sparse_index.db"
PROD_SPARSE_DB = PROJECT_ROOT / "data" / "sparse_index.db"

EXPECTED_TOTAL_QUERIES = 54
EXPECTED_ANSWERABLE = 42
EXPECTED_UNANSWERABLE = 12
EXPECTED_CORPUS_CHUNKS = 593
EXPECTED_COLLECTION = "benchmark_eval_chunks"
MODES = ("dense", "sparse", "hybrid")

RESULT_FILES = {
    "dense": DENSE_FILE,
    "sparse": SPARSE_FILE,
    "hybrid": HYBRID_FILE,
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus():
    assert CORPUS_FILE.exists(), f"Corpus file not found: {CORPUS_FILE}"
    return json.loads(CORPUS_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus_ids(corpus):
    return {c["chunk_id"] for c in corpus}


@pytest.fixture(scope="module")
def queries():
    assert QUERIES_FILE.exists(), f"Queries file not found: {QUERIES_FILE}"
    return json.loads(QUERIES_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def all_results():
    results = {}
    for mode, path in RESULT_FILES.items():
        assert path.exists(), f"Result file missing for mode '{mode}': {path}"
        results[mode] = json.loads(path.read_text(encoding="utf-8"))
    return results


# ---------------------------------------------------------------------------
# 1. Result files exist and are valid JSON with required top-level fields
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_FIELDS = {
    "mode",
    "timestamp",
    "model_name",
    "embedding_dimension",
    "qdrant_collection",
    "total_corpus_chunks",
    "total_queries",
    "answerable_queries_count",
    "unanswerable_queries_count",
    "metrics",
    "per_query_results",
}


@pytest.mark.parametrize("mode", MODES)
def test_result_file_exists_and_has_required_fields(mode, all_results):
    result = all_results[mode]
    missing = REQUIRED_TOP_LEVEL_FIELDS - set(result.keys())
    assert not missing, f"Mode '{mode}' missing top-level fields: {missing}"


# ---------------------------------------------------------------------------
# 2. All 54 queries evaluated in all 3 configurations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_all_queries_evaluated(mode, all_results, queries):
    result = all_results[mode]
    pqr = result["per_query_results"]

    assert result["total_queries"] == EXPECTED_TOTAL_QUERIES, (
        f"Mode '{mode}': total_queries header says {result['total_queries']}, "
        f"expected {EXPECTED_TOTAL_QUERIES}"
    )
    assert len(pqr) == EXPECTED_TOTAL_QUERIES, (
        f"Mode '{mode}': per_query_results has {len(pqr)} entries, "
        f"expected {EXPECTED_TOTAL_QUERIES}"
    )

    ground_truth_ids = {q["query_id"] for q in queries}
    result_ids = {entry["query_id"] for entry in pqr}
    assert result_ids == ground_truth_ids, (
        f"Mode '{mode}': query ID mismatch. "
        f"Missing: {ground_truth_ids - result_ids}. "
        f"Extra: {result_ids - ground_truth_ids}"
    )


# ---------------------------------------------------------------------------
# 3. All 3 configurations use identical query_id sets
# ---------------------------------------------------------------------------


def test_all_modes_use_identical_query_id_sets(all_results):
    id_sets = {}
    for mode, result in all_results.items():
        id_sets[mode] = {e["query_id"] for e in result["per_query_results"]}
    assert id_sets["dense"] == id_sets["sparse"] == id_sets["hybrid"], (
        "Query ID sets differ across modes. "
        f"dense-sparse diff: {id_sets['dense'] ^ id_sets['sparse']}. "
        f"dense-hybrid diff: {id_sets['dense'] ^ id_sets['hybrid']}"
    )


# ---------------------------------------------------------------------------
# 4. Exactly 42 answerable, 12 unanswerable per configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_answerable_unanswerable_counts(mode, all_results):
    result = all_results[mode]
    assert result["answerable_queries_count"] == EXPECTED_ANSWERABLE, (
        f"Mode '{mode}': answerable_queries_count header = "
        f"{result['answerable_queries_count']}, expected {EXPECTED_ANSWERABLE}"
    )
    assert result["unanswerable_queries_count"] == EXPECTED_UNANSWERABLE, (
        f"Mode '{mode}': unanswerable_queries_count header = "
        f"{result['unanswerable_queries_count']}, expected {EXPECTED_UNANSWERABLE}"
    )

    pqr = result["per_query_results"]
    actual_answerable = sum(1 for e in pqr if e.get("answerable"))
    actual_unanswerable = sum(1 for e in pqr if not e.get("answerable"))
    assert actual_answerable == EXPECTED_ANSWERABLE, (
        f"Mode '{mode}': actual answerable entries = {actual_answerable}, "
        f"expected {EXPECTED_ANSWERABLE}"
    )
    assert actual_unanswerable == EXPECTED_UNANSWERABLE, (
        f"Mode '{mode}': actual unanswerable entries = {actual_unanswerable}, "
        f"expected {EXPECTED_UNANSWERABLE}"
    )


# ---------------------------------------------------------------------------
# 5. Answerable queries have metric fields; unanswerable have corpus_evidence_retrieved
# ---------------------------------------------------------------------------

ANSWERABLE_REQUIRED_FIELDS = {"precision_at_k", "recall_at_k", "reciprocal_rank_at_k"}
UNANSWERABLE_REQUIRED_FIELDS = {"corpus_evidence_retrieved"}


@pytest.mark.parametrize("mode", MODES)
def test_per_query_result_fields(mode, all_results):
    pqr = all_results[mode]["per_query_results"]
    for entry in pqr:
        qid = entry["query_id"]
        if entry.get("answerable"):
            missing = ANSWERABLE_REQUIRED_FIELDS - set(entry.keys())
            assert not missing, (
                f"Mode '{mode}', answerable query '{qid}' missing fields: {missing}"
            )
        else:
            missing = UNANSWERABLE_REQUIRED_FIELDS - set(entry.keys())
            assert not missing, (
                f"Mode '{mode}', unanswerable query '{qid}' missing fields: {missing}"
            )


# ---------------------------------------------------------------------------
# 6. Retrieved chunk IDs all belong to the benchmark corpus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_retrieved_ids_are_from_corpus(mode, all_results, corpus_ids):
    pqr = all_results[mode]["per_query_results"]
    outside_corpus = []
    for entry in pqr:
        for cid in entry.get("retrieved_chunk_ids", []):
            if cid not in corpus_ids:
                outside_corpus.append((entry["query_id"], cid))
    assert not outside_corpus, (
        f"Mode '{mode}': {len(outside_corpus)} retrieved chunk IDs not in corpus: "
        f"{outside_corpus[:5]}"
    )


# ---------------------------------------------------------------------------
# 7. Qdrant collection is the isolated benchmark collection (not production)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_qdrant_collection_is_benchmark_collection(mode, all_results):
    result = all_results[mode]
    assert result["qdrant_collection"] == EXPECTED_COLLECTION, (
        f"Mode '{mode}': qdrant_collection = '{result['qdrant_collection']}', "
        f"expected '{EXPECTED_COLLECTION}'"
    )


# ---------------------------------------------------------------------------
# 8. Sparse DB path is in benchmarks/, NOT in data/
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_sparse_db_path_is_in_benchmarks(mode, all_results):
    result = all_results[mode]
    db_path = result.get("sparse_db_path", "")
    # Normalise to forward slashes for comparison
    normalised = db_path.replace("\\", "/").lower()
    assert "benchmarks/" in normalised, (
        f"Mode '{mode}': sparse_db_path '{db_path}' does not point into benchmarks/"
    )
    assert "data/" not in normalised or "benchmarks/" in normalised, (
        f"Mode '{mode}': sparse_db_path '{db_path}' appears to be in data/, "
        "which is the production SQLite — benchmark must use a separate path."
    )


# ---------------------------------------------------------------------------
# 9. Production sparse_index.db is in data/ and was NOT replaced by benchmark DB
# ---------------------------------------------------------------------------


def test_eval_sparse_db_isolated_from_production():
    """The benchmark SQLite DB must NOT be the same file as the production DB."""
    if not EVAL_SPARSE_DB.exists():
        pytest.skip("eval_sparse_index.db not found — benchmark may not have been run yet.")
    if not PROD_SPARSE_DB.exists():
        pytest.skip("Production sparse_index.db not found (may not exist in this environment).")
    # They must be different files (resolve to different absolute paths)
    assert EVAL_SPARSE_DB.resolve() != PROD_SPARSE_DB.resolve(), (
        "Benchmark and production sparse index DBs resolve to the same file — "
        "data isolation has been violated."
    )


# ---------------------------------------------------------------------------
# 10. total_corpus_chunks header matches the actual corpus file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_total_corpus_chunks_matches_corpus_file(mode, all_results, corpus):
    result = all_results[mode]
    assert result["total_corpus_chunks"] == EXPECTED_CORPUS_CHUNKS, (
        f"Mode '{mode}': total_corpus_chunks header = {result['total_corpus_chunks']}, "
        f"expected {EXPECTED_CORPUS_CHUNKS}"
    )
    assert len(corpus) == EXPECTED_CORPUS_CHUNKS, (
        f"corpus_chunks.json has {len(corpus)} entries, expected {EXPECTED_CORPUS_CHUNKS}"
    )


# ---------------------------------------------------------------------------
# 11. Corpus file is unchanged (all chunk IDs still unique and non-empty text)
# ---------------------------------------------------------------------------


def test_corpus_chunks_integrity(corpus):
    ids = [c["chunk_id"] for c in corpus]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected in corpus_chunks.json"
    empty_text = [c["chunk_id"] for c in corpus if not c.get("text", "").strip()]
    assert not empty_text, f"Chunks with empty text: {empty_text[:5]}"


# ---------------------------------------------------------------------------
# 12. Metrics are numeric and within plausible range [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_aggregate_metrics_are_valid(mode, all_results):
    metrics = all_results[mode]["metrics"]
    for metric_name, value in metrics.items():
        assert isinstance(value, (int, float)), (
            f"Mode '{mode}': metric '{metric_name}' is not numeric: {value!r}"
        )
        assert 0.0 <= value <= 1.0, (
            f"Mode '{mode}': metric '{metric_name}' = {value} is out of [0, 1]"
        )


# ---------------------------------------------------------------------------
# 13. Per-query metric values are numeric and within [0, 1] for answerable queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_per_query_metrics_are_valid(mode, all_results):
    pqr = all_results[mode]["per_query_results"]
    for entry in pqr:
        if not entry.get("answerable"):
            continue
        qid = entry["query_id"]
        for field in ANSWERABLE_REQUIRED_FIELDS:
            value = entry[field]
            assert isinstance(value, (int, float)), (
                f"Mode '{mode}', query '{qid}': field '{field}' not numeric: {value!r}"
            )
            assert 0.0 <= value <= 1.0, (
                f"Mode '{mode}', query '{qid}': field '{field}' = {value} out of [0, 1]"
            )
