
import json
from pathlib import Path

import pytest

from src.retrieval.evaluation import (
    answer_grounding_metrics,
    citation_validity_rate,
    evaluate_all_queries,
    evaluate_all_retrievers,
    evaluate_answer_generator,
    evaluate_query,
    evidence_coverage,
    macro_average_metrics,
    mean_reciprocal_rank_at_k,
    paired_bootstrap_confidence_interval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


@pytest.fixture
def evaluation_data():
    path = Path(__file__).parent / "fixtures" / "retrieval_eval.json"
    with path.open(encoding="utf-8") as file:
        return json.load(file)


class FakeEmbeddingModel:
    """Small deterministic embedding model for evaluation tests.

    Encodes each text as a binary bag-of-words vector over a fixed 12-word
    vocabulary.  Results are fully deterministic and require no download.
    """

    def encode(self, texts):
        vocabulary = [
            "supply", "chain", "manufacturing", "costs",
            "apple", "revenue", "growth", "services",
            "cloud", "cybersecurity", "confidential", "information",
        ]

        vectors = []
        for text in texts:
            tokens = set(text.lower().replace(",", "").split())
            vectors.append([
                float(word in tokens) for word in vocabulary
            ])

        return vectors


# ---------------------------------------------------------------------------
# Unit tests — precision_at_k
# ---------------------------------------------------------------------------

def test_precision_at_k():
    assert precision_at_k(["A", "B", "C"], {"A", "C"}, 3) == pytest.approx(2 / 3)


def test_precision_at_k_perfect_score():
    assert precision_at_k(["A", "B"], {"A", "B"}, 2) == pytest.approx(1.0)


def test_precision_at_k_zero_score():
    assert precision_at_k(["X", "Y"], {"A", "B"}, 2) == pytest.approx(0.0)


def test_empty_results_have_zero_precision():
    assert precision_at_k([], {"A"}, 3) == 0.0


def test_precision_at_k_truncates_at_k():
    # Only the first k=1 result is examined; "A" is relevant, so P@1 = 1.0
    assert precision_at_k(["A", "B", "C"], {"A"}, 1) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [0, -1])
def test_precision_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, k)


# ---------------------------------------------------------------------------
# Unit tests — recall_at_k
# ---------------------------------------------------------------------------

def test_recall_at_k():
    assert recall_at_k(["A", "B"], {"A", "C"}, 2) == pytest.approx(0.5)


def test_recall_at_k_perfect_score():
    assert recall_at_k(["A", "B"], {"A", "B"}, 2) == pytest.approx(1.0)


def test_no_relevant_ids_have_zero_recall():
    assert recall_at_k(["A"], set(), 3) == 0.0


def test_recall_at_k_truncates_at_k():
    # Only the first k=1 result is examined; "A" is relevant, so R@1 = 0.5
    assert recall_at_k(["A", "B"], {"A", "B"}, 1) == pytest.approx(0.5)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, k)


# ---------------------------------------------------------------------------
# Unit tests — macro_average_metrics
# ---------------------------------------------------------------------------

def test_macro_average_single_query():
    result = macro_average_metrics([{"precision_at_k": 0.6, "recall_at_k": 1.0}])
    assert result["precision_at_k"] == pytest.approx(0.6)
    assert result["recall_at_k"] == pytest.approx(1.0)


def test_macro_average_multiple_queries():
    results = [
        {"precision_at_k": 1.0, "recall_at_k": 1.0},
        {"precision_at_k": 0.0, "recall_at_k": 0.0},
    ]
    avg = macro_average_metrics(results)
    assert avg["precision_at_k"] == pytest.approx(0.5)
    assert avg["recall_at_k"] == pytest.approx(0.5)


def test_macro_average_empty_list_returns_zeros():
    result = macro_average_metrics([])
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0


def test_macro_average_weights_queries_equally():
    """Each query gets equal weight regardless of relevant-set size."""
    results = [
        {"precision_at_k": 0.8, "recall_at_k": 0.9},
        {"precision_at_k": 0.4, "recall_at_k": 0.5},
        {"precision_at_k": 0.6, "recall_at_k": 0.7},
    ]
    avg = macro_average_metrics(results)
    assert avg["precision_at_k"] == pytest.approx((0.8 + 0.4 + 0.6) / 3)
    assert avg["recall_at_k"] == pytest.approx((0.9 + 0.5 + 0.7) / 3)


# ---------------------------------------------------------------------------
# evaluate_query — keyword-only baseline
# ---------------------------------------------------------------------------

def test_evaluates_keyword_retrieval(evaluation_data):
    case = evaluation_data["queries"][0]

    result = evaluate_query(
        case["query"],
        evaluation_data["chunks"],
        case["relevant_chunk_ids"],
        k=3,
    )

    assert result["retrieved_chunk_ids"][0] == "AAPL_RISK_001"
    assert result["precision_at_k"] == pytest.approx(1 / 3)
    assert result["recall_at_k"] == 1.0


def test_evaluates_all_sample_queries_keyword(evaluation_data):
    results = [
        evaluate_query(
            case["query"],
            evaluation_data["chunks"],
            case["relevant_chunk_ids"],
            k=3,
        )
        for case in evaluation_data["queries"]
    ]

    assert len(results) == 4
    assert all(result["recall_at_k"] == 1.0 for result in results)


def test_evaluate_query_rejects_invalid_k(evaluation_data):
    with pytest.raises(ValueError):
        evaluate_query("test", evaluation_data["chunks"], ["A"], k=0)


# ---------------------------------------------------------------------------
# evaluate_all_retrievers — single-query, three-method comparison
# ---------------------------------------------------------------------------

def test_evaluates_keyword_semantic_and_hybrid(evaluation_data):
    case = evaluation_data["queries"][0]

    result = evaluate_all_retrievers(
        query=case["query"],
        chunks=evaluation_data["chunks"],
        relevant_ids=case["relevant_chunk_ids"],
        model=FakeEmbeddingModel(),
        k=3,
    )

    assert set(result) == {"query", "keyword", "semantic", "hybrid"}

    for method in ("keyword", "semantic", "hybrid"):
        assert result[method]["query"] == case["query"]
        assert "retrieved_chunk_ids" in result[method]
        assert "precision_at_k" in result[method]
        assert "recall_at_k" in result[method]
        assert 0.0 <= result[method]["precision_at_k"] <= 1.0
        assert 0.0 <= result[method]["recall_at_k"] <= 1.0


def test_all_retrievers_find_relevant_chunk(evaluation_data):
    case = evaluation_data["queries"][0]

    result = evaluate_all_retrievers(
        query=case["query"],
        chunks=evaluation_data["chunks"],
        relevant_ids=case["relevant_chunk_ids"],
        model=FakeEmbeddingModel(),
        k=3,
    )

    for method in ("keyword", "semantic", "hybrid"):
        assert result[method]["recall_at_k"] == 1.0


def test_evaluate_all_retrievers_rejects_invalid_k(evaluation_data):
    with pytest.raises(ValueError):
        evaluate_all_retrievers(
            query="test",
            chunks=evaluation_data["chunks"],
            relevant_ids=["A"],
            model=FakeEmbeddingModel(),
            k=0,
        )


def test_evaluate_all_retrievers_rejects_invalid_candidate_k(evaluation_data):
    with pytest.raises(ValueError):
        evaluate_all_retrievers(
            query="test",
            chunks=evaluation_data["chunks"],
            relevant_ids=["A"],
            model=FakeEmbeddingModel(),
            k=3,
            candidate_k=0,
        )


def test_evaluate_all_retrievers_empty_relevant_ids(evaluation_data):
    """Empty relevant set must not crash; recall must be 0.0 for all methods."""
    case = evaluation_data["queries"][0]
    result = evaluate_all_retrievers(
        query=case["query"],
        chunks=evaluation_data["chunks"],
        relevant_ids=[],
        model=FakeEmbeddingModel(),
        k=3,
    )
    for method in ("keyword", "semantic", "hybrid"):
        assert result[method]["recall_at_k"] == 0.0


def test_evaluate_all_retrievers_empty_chunks():
    """Empty chunk list must not crash; all metrics must be 0.0."""
    result = evaluate_all_retrievers(
        query="supply chain costs",
        chunks=[],
        relevant_ids=["AAPL_RISK_001"],
        model=FakeEmbeddingModel(),
        k=3,
    )
    for method in ("keyword", "semantic", "hybrid"):
        assert result[method]["precision_at_k"] == 0.0
        assert result[method]["recall_at_k"] == 0.0


# ---------------------------------------------------------------------------
# evaluate_all_queries — full fixture comparison across all queries
# ---------------------------------------------------------------------------

def test_evaluate_all_queries_returns_correct_structure(evaluation_data):
    output = evaluate_all_queries(
        queries=evaluation_data["queries"],
        chunks=evaluation_data["chunks"],
        model=FakeEmbeddingModel(),
        k=3,
    )

    assert output["k"] == 3
    assert len(output["per_query"]) == len(evaluation_data["queries"])
    assert set(output["macro_average"]) == {"keyword", "semantic", "hybrid"}


def test_evaluate_all_queries_per_query_has_three_methods(evaluation_data):
    output = evaluate_all_queries(
        queries=evaluation_data["queries"],
        chunks=evaluation_data["chunks"],
        model=FakeEmbeddingModel(),
        k=3,
    )

    for entry in output["per_query"]:
        assert set(entry) == {"query", "keyword", "semantic", "hybrid"}


def test_evaluate_all_queries_macro_average_in_range(evaluation_data):
    output = evaluate_all_queries(
        queries=evaluation_data["queries"],
        chunks=evaluation_data["chunks"],
        model=FakeEmbeddingModel(),
        k=3,
    )

    for method in ("keyword", "semantic", "hybrid"):
        macro = output["macro_average"][method]
        assert 0.0 <= macro["precision_at_k"] <= 1.0
        assert 0.0 <= macro["recall_at_k"] <= 1.0


def test_evaluate_all_queries_keyword_recall_across_all_queries(evaluation_data):
    """Keyword retrieval should retrieve the relevant chunk for every fixture query."""
    output = evaluate_all_queries(
        queries=evaluation_data["queries"],
        chunks=evaluation_data["chunks"],
        model=FakeEmbeddingModel(),
        k=3,
    )

    for entry in output["per_query"]:
        assert entry["keyword"]["recall_at_k"] == 1.0, (
            f"Keyword recall failed for query: {entry['query']!r}"
        )


def test_evaluate_all_queries_macro_average_consistent_with_per_query(evaluation_data):
    """Macro-average must equal the mean of per-query metrics."""
    output = evaluate_all_queries(
        queries=evaluation_data["queries"],
        chunks=evaluation_data["chunks"],
        model=FakeEmbeddingModel(),
        k=3,
    )

    for method in ("keyword", "semantic", "hybrid"):
        per_query_metrics = [r[method] for r in output["per_query"]]
        expected = macro_average_metrics(per_query_metrics)
        actual = output["macro_average"][method]
        assert actual["precision_at_k"] == pytest.approx(expected["precision_at_k"])
        assert actual["recall_at_k"] == pytest.approx(expected["recall_at_k"])


def test_evaluate_all_queries_covers_all_configured_queries(evaluation_data):
    """The number of per-query results must match the fixture query count."""
    output = evaluate_all_queries(
        queries=evaluation_data["queries"],
        chunks=evaluation_data["chunks"],
        model=FakeEmbeddingModel(),
        k=3,
    )

    fixture_queries = [q["query"] for q in evaluation_data["queries"]]
    result_queries = [r["query"] for r in output["per_query"]]
    assert result_queries == fixture_queries


def test_evaluate_all_queries_empty_query_list(evaluation_data):
    output = evaluate_all_queries(
        queries=[],
        chunks=evaluation_data["chunks"],
        model=FakeEmbeddingModel(),
        k=3,
    )

    assert output["per_query"] == []
    for method in ("keyword", "semantic", "hybrid"):
        assert output["macro_average"][method]["precision_at_k"] == 0.0
        assert output["macro_average"][method]["recall_at_k"] == 0.0


def test_evaluate_all_queries_rejects_invalid_k(evaluation_data):
    with pytest.raises(ValueError):
        evaluate_all_queries(
            queries=evaluation_data["queries"],
            chunks=evaluation_data["chunks"],
            model=FakeEmbeddingModel(),
            k=0,
        )


def test_evaluate_all_queries_rejects_invalid_candidate_k(evaluation_data):
    with pytest.raises(ValueError):
        evaluate_all_queries(
            queries=evaluation_data["queries"],
            chunks=evaluation_data["chunks"],
            model=FakeEmbeddingModel(),
            k=3,
            candidate_k=0,
        )


# ---------------------------------------------------------------------------
# Unit tests — evidence_coverage & citation_validity_rate
# ---------------------------------------------------------------------------

def test_evidence_coverage_full():
    assert evidence_coverage(["c1", "c2"], ["c1", "c2"]) == pytest.approx(1.0)


def test_evidence_coverage_partial():
    assert evidence_coverage(["c1"], ["c1", "c2"]) == pytest.approx(0.5)


def test_evidence_coverage_empty_relevant():
    assert evidence_coverage(["c1"], []) == 0.0


def test_citation_validity_rate_all_valid():
    assert citation_validity_rate(["S1", "S2"], {"S1", "S2", "S3"}) == pytest.approx(1.0)


def test_citation_validity_rate_partial_invalid():
    assert citation_validity_rate(["S1", "S99"], {"S1", "S2"}) == pytest.approx(0.5)


def test_citation_validity_rate_empty_citations():
    """Undefined validity rate when no citations were made returns None."""
    assert citation_validity_rate([], {"S1", "S2"}) is None


def test_citation_validity_rate_mixed():
    """Test mixed valid and invalid citation labels."""
    labels = ["S1", "S2", "S99"]
    valid = {"S1", "S2", "S3"}
    assert citation_validity_rate(labels, valid) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Unit tests — answer_grounding_metrics & evaluate_answer_generator
# ---------------------------------------------------------------------------

def test_answer_grounding_metrics_grounded_with_verification():
    """When verified by claim verifier, is_grounded is True."""
    answer_res = {
        "question": "What is revenue?",
        "answer": "Revenue grew [S1].",
        "citations": [{"label": "S1", "chunk_id": "c1"}],
        "insufficient_evidence": False,
        "verification": {"verified": True, "verification_available": True},
    }
    metrics = answer_grounding_metrics(answer_res, relevant_chunk_ids=["c1", "c2"], expected_insufficient=False)
    assert metrics["is_grounded"] is True
    assert metrics["has_citations"] is True
    assert metrics["citation_count"] == 1
    assert metrics["cited_chunk_ids"] == ["c1"]
    assert metrics["evidence_coverage"] == pytest.approx(0.5)
    assert metrics["insufficient_evidence_correct"] is True
    assert metrics["claim_verification_status"] == "verified"
    assert metrics["claim_verified"] is True


def test_answer_grounding_metrics_without_verification_is_unknown():
    """Without claim verification, semantic grounding cannot be established from citations alone."""
    answer_res = {
        "question": "What is revenue?",
        "answer": "Revenue grew [S1].",
        "citations": [{"label": "S1", "chunk_id": "c1"}],
        "insufficient_evidence": False,
    }
    metrics = answer_grounding_metrics(answer_res, relevant_chunk_ids=["c1"])
    assert metrics["has_citations"] is True
    assert metrics["is_grounded"] is None  # Unknown/undefined without claim verifier
    assert metrics["claim_verification_status"] == "not_performed"
    assert metrics["claim_verified"] is None


def test_answer_grounding_metrics_unverified():
    """When claim verifier finds unsupported claims, is_grounded is False."""
    answer_res = {
        "question": "What is revenue?",
        "answer": "Revenue grew [S1].",
        "citations": [{"label": "S1", "chunk_id": "c1"}],
        "insufficient_evidence": False,
        "verification": {"verified": False, "verification_available": True, "unsupported_claims": ["grew"]},
    }
    metrics = answer_grounding_metrics(answer_res)
    assert metrics["is_grounded"] is False
    assert metrics["claim_verification_status"] == "unverified"
    assert metrics["claim_verified"] is False


def test_answer_grounding_metrics_insufficient():
    answer_res = {
        "question": "What is revenue?",
        "answer": "I could not find sufficient evidence...",
        "citations": [],
        "insufficient_evidence": True,
    }
    metrics = answer_grounding_metrics(answer_res, expected_insufficient=True)
    assert metrics["is_grounded"] is False
    assert metrics["has_citations"] is False
    assert metrics["citation_count"] == 0
    assert metrics["insufficient_evidence"] is True
    assert metrics["insufficient_evidence_correct"] is True


def test_answer_grounding_metrics_missing_valid_labels_is_none():
    """When valid_labels is not supplied, citation_validity_rate must be None."""
    answer_res = {
        "question": "Q?",
        "answer": "Answer [S1].",
        "citations": [{"label": "S1", "chunk_id": "c1"}],
        "insufficient_evidence": False,
    }
    metrics = answer_grounding_metrics(answer_res)
    assert metrics["citation_validity_rate"] is None


def test_answer_grounding_metrics_fabricated_citation_labels():
    """Fabricated citation labels evaluated against valid_labels report correct rate."""
    answer_res = {
        "question": "Q?",
        "answer": "Answer [S99].",
        "citations": [{"label": "S99", "chunk_id": "c99"}],
        "insufficient_evidence": False,
    }
    metrics = answer_grounding_metrics(answer_res, valid_labels={"S1", "S2"})
    assert metrics["citation_validity_rate"] == 0.0

    # Mixed valid and fabricated
    answer_res_mixed = {
        "question": "Q?",
        "answer": "Answer [S1] and [S99].",
        "citations": [{"label": "S1", "chunk_id": "c1"}, {"label": "S99", "chunk_id": "c99"}],
        "insufficient_evidence": False,
    }
    metrics_mixed = answer_grounding_metrics(answer_res_mixed, valid_labels={"S1", "S2"})
    assert metrics_mixed["citation_validity_rate"] == pytest.approx(0.5)


def test_answer_grounding_metrics_malformed_citation_entries():
    """Malformed entries (non-dict, missing label, missing/empty chunk_id) must not inflate metrics."""
    answer_res = {
        "question": "Q?",
        "answer": "Answer.",
        "citations": [
            None,
            "not-a-dict",
            {},
            {"label": "", "chunk_id": "c1"},
            {"label": "S1", "chunk_id": ""},
            {"label": "S1", "chunk_id": "c1"},  # exactly one valid citation
        ],
        "insufficient_evidence": False,
    }
    metrics = answer_grounding_metrics(answer_res)
    assert metrics["citation_count"] == 1
    assert metrics["has_citations"] is True
    assert metrics["cited_chunk_ids"] == ["c1"]


def test_answer_grounding_metrics_empty_citation_list():
    """Empty citation list must report zero count, False has_citations, and None validity."""
    answer_res = {
        "question": "Q?",
        "answer": "Answer without citations.",
        "citations": [],
        "insufficient_evidence": False,
    }
    metrics = answer_grounding_metrics(answer_res, valid_labels={"S1"})
    assert metrics["citation_count"] == 0
    assert metrics["has_citations"] is False
    assert metrics["cited_chunk_ids"] == []
    assert metrics["citation_validity_rate"] is None


def test_evaluate_answer_generator_batch():
    class FakeGenerator:
        def answer(self, question, top_k=5, filters=None):
            if "unknown" in question:
                return {
                    "question": question,
                    "answer": "Insufficient evidence",
                    "citations": [],
                    "insufficient_evidence": True,
                }
            return {
                "question": question,
                "answer": f"Answer for {question} [S1]",
                "citations": [{"label": "S1", "chunk_id": "c1"}],
                "insufficient_evidence": False,
                "verification": {"verified": True, "verification_available": True},
            }

    test_cases = [
        {
            "question": "What is Apple revenue?",
            "relevant_chunk_ids": ["c1"],
            "expected_insufficient": False,
        },
        {
            "question": "What is an unknown metric?",
            "relevant_chunk_ids": [],
            "expected_insufficient": True,
        },
    ]

    report = evaluate_answer_generator(test_cases, FakeGenerator())
    assert report["total_cases"] == 2
    assert report["grounded_answers_rate"] == pytest.approx(0.5)
    assert report["citation_presence_rate"] == pytest.approx(0.5)
    assert report["insufficient_evidence_accuracy"] == pytest.approx(1.0)
    assert report["mean_citations_per_answer"] == pytest.approx(0.5)
    assert len(report["per_case_results"]) == 2


def test_evaluate_answer_generator_unevaluated_accuracy_returns_none():
    """When cases omit expected_insufficient, accuracy must report None, not 1.0."""
    class FakeGenerator:
        def answer(self, question, top_k=5, filters=None):
            return {
                "question": question,
                "answer": "Answer [S1]",
                "citations": [{"label": "S1", "chunk_id": "c1"}],
                "insufficient_evidence": False,
            }

    test_cases = [{"question": "What is revenue?"}]  # no expected_insufficient
    report = evaluate_answer_generator(test_cases, FakeGenerator())
    assert report["insufficient_evidence_accuracy"] is None


def test_evaluate_answer_generator_empty_cases():
    report = evaluate_answer_generator([], None)
    assert report["total_cases"] == 0
    assert report["insufficient_evidence_accuracy"] is None
    assert report["grounded_answers_rate"] is None
    assert report["per_case_results"] == []


# ---------------------------------------------------------------------------
# Unit tests — reciprocal_rank_at_k, mean_reciprocal_rank_at_k, paired_bootstrap
# ---------------------------------------------------------------------------

def test_reciprocal_rank_at_k_rank_one():
    assert reciprocal_rank_at_k(["A", "B", "C"], {"A"}, 5) == 1.0


def test_reciprocal_rank_at_k_rank_two():
    assert reciprocal_rank_at_k(["X", "A", "C"], {"A"}, 5) == 0.5


def test_reciprocal_rank_at_k_rank_five():
    assert reciprocal_rank_at_k(["X", "Y", "Z", "W", "A"], {"A"}, 5) == 0.2


def test_reciprocal_rank_at_k_beyond_k_returns_zero():
    # Relevant item "A" is at rank 6, beyond k=5
    assert reciprocal_rank_at_k(["X", "Y", "Z", "W", "V", "A"], {"A"}, 5) == 0.0


def test_reciprocal_rank_at_k_not_found_returns_zero():
    assert reciprocal_rank_at_k(["X", "Y", "Z"], {"A"}, 5) == 0.0


def test_reciprocal_rank_at_k_empty_retrieved_returns_zero():
    assert reciprocal_rank_at_k([], {"A"}, 5) == 0.0


def test_reciprocal_rank_at_k_empty_relevant_returns_zero():
    assert reciprocal_rank_at_k(["A", "B"], set(), 5) == 0.0


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_reciprocal_rank_at_k_rejects_invalid_k(invalid_k):
    with pytest.raises(ValueError):
        reciprocal_rank_at_k(["A"], {"A"}, invalid_k)


def test_reciprocal_rank_at_k_uses_first_relevant_hit():
    """When multiple relevant docs exist, RR uses the first (minimum rank)."""
    assert reciprocal_rank_at_k(["A", "B", "C"], {"A", "B"}, 5) == 1.0


def test_mean_reciprocal_rank_at_k():
    rr_scores = [1.0, 0.5, 0.0, 0.2]
    assert mean_reciprocal_rank_at_k(rr_scores) == pytest.approx(1.7 / 4.0)


def test_mean_reciprocal_rank_at_k_empty_returns_zero():
    assert mean_reciprocal_rank_at_k([]) == 0.0


def test_macro_average_metrics_includes_mrr():
    results = [
        {"precision_at_k": 0.5, "recall_at_k": 1.0, "reciprocal_rank_at_k": 1.0},
        {"precision_at_k": 0.25, "recall_at_k": 0.5, "reciprocal_rank_at_k": 0.5},
    ]
    macro = macro_average_metrics(results)
    assert macro["precision_at_k"] == pytest.approx(0.375)
    assert macro["recall_at_k"] == pytest.approx(0.75)
    assert macro["mrr_at_k"] == pytest.approx(0.75)


def test_paired_bootstrap_confidence_interval_identical_scores():
    # Exactly zero difference across all queries
    baseline = [0.8, 0.6, 0.4, 0.2]
    treatment = [0.8, 0.6, 0.4, 0.2]
    res = paired_bootstrap_confidence_interval(baseline, treatment, random_seed=42)
    assert res["mean_difference"] == pytest.approx(0.0)
    assert res["ci_lower"] == pytest.approx(0.0)
    assert res["ci_upper"] == pytest.approx(0.0)
    assert res["wilcoxon_zero_diff_count"] == 4
    assert res["wilcoxon_p_value"] == 1.0
    assert res["statistically_significant"] is False


def test_paired_bootstrap_confidence_interval_significant_improvement():
    # Treatment consistently outperforms baseline
    baseline = [0.1, 0.2, 0.1, 0.2, 0.15, 0.1, 0.2, 0.1, 0.15, 0.1]
    treatment = [0.8, 0.9, 0.85, 0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9]
    res = paired_bootstrap_confidence_interval(
        baseline, treatment, num_bootstrap=5000, random_seed=42
    )
    assert res["mean_difference"] > 0.6
    assert res["ci_lower"] > 0.5
    assert res["ci_upper"] > res["ci_lower"]
    assert res["wilcoxon_p_value"] is not None
    assert res["wilcoxon_p_value"] < 0.01
    assert res["statistically_significant"] is True


def test_paired_bootstrap_confidence_interval_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_confidence_interval([0.5, 0.6], [0.5])


def test_paired_bootstrap_confidence_interval_empty_lists():
    res = paired_bootstrap_confidence_interval([], [])
    assert res["mean_difference"] == 0.0
    assert res["ci_lower"] == 0.0
    assert res["ci_upper"] == 0.0
    assert res["wilcoxon_p_value"] is None
    assert res["statistically_significant"] is False
