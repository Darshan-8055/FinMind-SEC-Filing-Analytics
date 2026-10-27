
import json
from pathlib import Path

import pytest

from src.retrieval.evaluation import (
    evaluate_all_queries,
    evaluate_all_retrievers,
    evaluate_query,
    macro_average_metrics,
    precision_at_k,
    recall_at_k,
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
