
import json
from pathlib import Path

import pytest

from src.retrieval.evaluation import (
    evaluate_query,
    precision_at_k,
    recall_at_k,
)


@pytest.fixture
def evaluation_data():
    path = (
        Path(__file__).parent
        / "fixtures"
        / "retrieval_eval.json"
    )
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def test_precision_at_k():
    assert precision_at_k(
        ["A", "B", "C"], {"A", "C"}, 3
    ) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(
        ["A", "B"], {"A", "C"}, 2
    ) == pytest.approx(0.5)


def test_empty_results_have_zero_precision():
    assert precision_at_k([], {"A"}, 3) == 0.0


def test_no_relevant_ids_have_zero_recall():
    assert recall_at_k(["A"], set(), 3) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_precision_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, k)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, k)


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


def test_evaluates_all_sample_queries(evaluation_data):
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
