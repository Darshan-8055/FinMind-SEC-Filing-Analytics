
"""Evaluation metrics for information retrieval and answer generation.

Metric conventions and limitations
----------------------------------
- Precision@K: (relevant hits in top-K) / K.
  Returns 0.0 when the retrieved list is empty; denominator K is always
  a positive integer so division by zero cannot occur.
- Recall@K: (relevant hits in top-K) / |relevant set|.
  Returns 0.0 when the relevant set is empty; the denominator is the size
  of the relevant set which is checked before division.
- Macro-average: each query is weighted equally.
  Returns zeros when the query list is empty.
- Reciprocal Rank (RR@K): 1 / (rank of first relevant document in top-K), or 0.0 if none.
  Returns 0.0 when retrieved list or relevant set is empty.
- Mean Reciprocal Rank (MRR@K): macro-average of RR@K across queries.
- Evidence Coverage: fraction of known relevant chunks cited in the generated answer.
  Returns 0.0 when relevant set is empty.
- Citation Validity Rate: fraction of citations referencing legitimate retrieved sources.
  Returns None when no citations are cited (undefined; avoids false 100% claims).
- Answer Grounding: evaluates citation integrity, coverage, and claim verification status.
  Semantic grounding cannot be established from citation presence alone; without
  claim verification, overall grounding is reported as None (unknown).
"""

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.keyword_search import keyword_search
from src.retrieval.semantic_search import semantic_search


def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: Iterable[str],
    k: int = 5,
) -> float:
    """Reciprocal rank of the first relevant result in the top-k results.

    Returns 1.0 / rank (1-based) for the first relevant document found within
    the first *k* results. Returns 0.0 if no relevant document appears in the
    top *k*, or if *retrieved_ids* or *relevant_ids* is empty.
    Raises ValueError when k < 1.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    relevant = set(relevant_ids)
    if not relevant or not retrieved_ids:
        return 0.0

    for rank, item in enumerate(retrieved_ids[:k], start=1):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank_at_k(
    per_query_reciprocal_ranks: Sequence[float],
) -> float:
    """Compute the Mean Reciprocal Rank (MRR) across a sequence of reciprocal ranks.

    Returns 0.0 when *per_query_reciprocal_ranks* is empty.
    """
    if not per_query_reciprocal_ranks:
        return 0.0
    return float(sum(per_query_reciprocal_ranks) / len(per_query_reciprocal_ranks))


def paired_bootstrap_confidence_interval(
    baseline_scores: Sequence[float],
    treatment_scores: Sequence[float],
    *,
    num_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Calculate paired bootstrap confidence interval and Wilcoxon p-value for score differences.

    Parameters
    ----------
    baseline_scores:
        Per-query scores for the baseline system (e.g. dense-only or sparse-only).
    treatment_scores:
        Per-query scores for the treatment system (e.g. hybrid).
    num_bootstrap:
        Number of bootstrap resamples (default: 10,000).
    confidence_level:
        Desired confidence level (default: 0.95 for a 95% CI).
    random_seed:
        Fixed random seed for deterministic reproducibility (default: 42).

    Returns
    -------
    dict with keys:
        - "mean_difference": float, treatment mean minus baseline mean
        - "ci_lower": float, lower bound of the bootstrap CI
        - "ci_upper": float, upper bound of the bootstrap CI
        - "confidence_level": float
        - "num_bootstrap": int
        - "wilcoxon_p_value": float | None, two-sided p-value from Wilcoxon signed-rank test
        - "wilcoxon_zero_diff_count": int, count of queries with identical scores
        - "statistically_significant": bool, True if 0 is outside CI and p_value < (1 - confidence_level)
    """
    if len(baseline_scores) != len(treatment_scores):
        raise ValueError(
            f"baseline_scores (len {len(baseline_scores)}) and treatment_scores "
            f"(len {len(treatment_scores)}) must have identical lengths"
        )

    n = len(baseline_scores)
    if n == 0:
        return {
            "mean_difference": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "confidence_level": confidence_level,
            "num_bootstrap": num_bootstrap,
            "wilcoxon_p_value": None,
            "wilcoxon_zero_diff_count": 0,
            "statistically_significant": False,
        }

    b_list = [float(x) for x in baseline_scores]
    t_list = [float(x) for x in treatment_scores]
    diffs = [t - b for t, b in zip(t_list, b_list)]
    mean_diff = sum(diffs) / n

    # Reproducible paired bootstrap resampling using the standard library
    rng = random.Random(random_seed)
    bootstrap_means: list[float] = []
    for _ in range(num_bootstrap):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()

    alpha = 1.0 - confidence_level
    lower_idx = max(0, int(math.floor(alpha / 2.0 * num_bootstrap)))
    upper_idx = min(num_bootstrap - 1, int(math.ceil((1.0 - alpha / 2.0) * num_bootstrap)) - 1)
    ci_lower = bootstrap_means[lower_idx]
    ci_upper = bootstrap_means[upper_idx]

    # Wilcoxon signed-rank test: lazy SciPy import with graceful fallback
    zero_diff_count = sum(1 for d in diffs if d == 0.0)
    wilcoxon_p_value: float | None = None

    if zero_diff_count == n:
        # All differences are exactly 0 — Wilcoxon is undefined; p-value is 1.0
        wilcoxon_p_value = 1.0
    else:
        try:
            from scipy.stats import wilcoxon  # noqa: PLC0415
        except ImportError:
            wilcoxon_p_value = None
        else:
            # zero_method='pratt' includes zero differences in ranking (handles ties correctly)
            res = wilcoxon(t_list, b_list, zero_method="pratt", alternative="two-sided")
            pval = float(res.pvalue)
            # math.isnan replaces np.isnan — no numpy required
            wilcoxon_p_value = 1.0 if math.isnan(pval) else pval

    if wilcoxon_p_value is None:
        is_significant = False
    else:
        is_significant = (ci_lower > 0.0 or ci_upper < 0.0) and (wilcoxon_p_value < alpha)

    return {
        "mean_difference": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "num_bootstrap": num_bootstrap,
        "wilcoxon_p_value": wilcoxon_p_value,
        "wilcoxon_zero_diff_count": zero_diff_count,
        "statistically_significant": is_significant,
    }


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Fraction of the first k retrieved results that are relevant.

    Returns 0.0 when *retrieved_ids* is empty.
    Raises ValueError when k < 1.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    relevant = set(relevant_ids)
    if not retrieved_ids:
        return 0.0

    top_results = retrieved_ids[:k]
    return sum(item in relevant for item in top_results) / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Fraction of all known relevant results found in the first k.

    Returns 0.0 when *relevant_ids* is empty (recall is undefined; the
    convention is to treat it as 0.0 to avoid division by zero).
    Raises ValueError when k < 1.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    top_results = set(retrieved_ids[:k])
    return len(top_results & relevant) / len(relevant)


def _retrieved_ids(results: list[dict[str, Any]]) -> list[str]:
    """Extract stable chunk IDs from retrieval results."""
    return [
        str(result["chunk_id"])
        for result in results
        if result.get("chunk_id") is not None
    ]


def _score_results(
    query: str,
    results: list[dict[str, Any]],
    relevant_ids: set[str],
    k: int,
) -> dict[str, Any]:
    """Calculate evaluation metrics for one ranked result list."""
    retrieved_ids = _retrieved_ids(results)

    return {
        "query": query,
        "retrieved_chunk_ids": retrieved_ids,
        "relevant_chunk_ids": sorted(relevant_ids),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "reciprocal_rank_at_k": reciprocal_rank_at_k(retrieved_ids, relevant_ids, k),
    }


def evaluate_query(
    query: str,
    chunks: Iterable[Mapping[str, Any]],
    relevant_ids: Iterable[str],
    k: int = 3,
) -> dict[str, Any]:
    """Run keyword retrieval and calculate precision and recall.

    This is the baseline keyword-only evaluation function.  Use
    :func:`evaluate_all_retrievers` to compare all three methods at once.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    chunk_list = list(chunks)
    results = keyword_search(query, chunk_list, top_k=k)
    return _score_results(query, results, set(relevant_ids), k)


def evaluate_all_retrievers(
    query: str,
    chunks: Iterable[Mapping[str, Any]],
    relevant_ids: Iterable[str],
    model: Any,
    k: int = 3,
    candidate_k: int = 30,
) -> dict[str, Any]:
    """Evaluate keyword, semantic, and hybrid retrieval for one query.

    All three methods use the same relevance judgments and the same
    evaluation cut-off *k* so results are directly comparable.

    Returns a dict with keys ``"query"``, ``"keyword"``, ``"semantic"``,
    and ``"hybrid"``.  Each method value is the output of
    :func:`_score_results`.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")

    chunk_list = list(chunks)
    relevant = set(relevant_ids)

    keyword_results = keyword_search(query, chunk_list, top_k=k)
    semantic_results = semantic_search(
        query, chunk_list, model, top_k=k
    )
    hybrid_results = hybrid_search(
        query,
        chunk_list,
        model,
        top_k=k,
        candidate_k=candidate_k,
    )

    return {
        "query": query,
        "keyword": _score_results(query, keyword_results, relevant, k),
        "semantic": _score_results(query, semantic_results, relevant, k),
        "hybrid": _score_results(query, hybrid_results, relevant, k),
    }


def macro_average_metrics(
    per_query_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Return macro-average Precision@K, Recall@K, and MRR@K across queries.

    Each element of *per_query_results* must contain ``precision_at_k`` and
    ``recall_at_k`` keys (as produced by :func:`_score_results`). If
    ``reciprocal_rank_at_k`` is present, computes ``mrr_at_k``.

    The macro-average weights every query equally regardless of the size of
    its relevant set.  Returns zeros when the input list is empty so callers
    never encounter a division by zero.

    Returns
    -------
    dict with keys ``precision_at_k``, ``recall_at_k``, and ``mrr_at_k``.
    """
    if not per_query_results:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0}

    n = len(per_query_results)
    avg_precision = sum(r["precision_at_k"] for r in per_query_results) / n
    avg_recall = sum(r["recall_at_k"] for r in per_query_results) / n
    avg_mrr = sum(r.get("reciprocal_rank_at_k", 0.0) for r in per_query_results) / n
    return {
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "mrr_at_k": avg_mrr,
    }


def evaluate_all_queries(
    queries: Iterable[Mapping[str, Any]],
    chunks: Iterable[Mapping[str, Any]],
    model: Any,
    k: int = 3,
    candidate_k: int = 30,
) -> dict[str, Any]:
    """Evaluate all three retrievers across every query in *queries*.

    Each element of *queries* must have:

    - ``query`` (str): the natural-language query text.
    - ``relevant_chunk_ids`` (list[str]): ground-truth relevant chunk IDs.

    Parameters
    ----------
    queries:
        Iterable of query dicts (as stored in the retrieval eval fixture).
    chunks:
        All candidate chunks to search through.
    model:
        Embedding model with an ``encode(texts)`` method.
    k:
        Evaluation cut-off for Precision@K and Recall@K.
    candidate_k:
        Maximum candidates passed to the dense and sparse retrievers inside
        hybrid search.

    Returns
    -------
    dict with:

    - ``k``: the evaluation cut-off used.
    - ``per_query``: list of per-query dicts from
      :func:`evaluate_all_retrievers`.
    - ``macro_average``: dict keyed by method name (``"keyword"``,
      ``"semantic"``, ``"hybrid"``), each holding the macro-average
      Precision@K and Recall@K from :func:`macro_average_metrics`.

    Notes
    -----
    Results produced from the synthetic fixture in
    ``tests/fixtures/retrieval_eval.json`` reflect a small, hand-crafted
    chunk set.  They should not be reported as production retrieval quality
    figures.  Real-data validation requires a live Qdrant service and actual
    SEC filing embeddings.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")

    chunk_list = list(chunks)
    per_query: list[dict[str, Any]] = []

    for query_item in queries:
        result = evaluate_all_retrievers(
            query=query_item["query"],
            chunks=chunk_list,
            relevant_ids=query_item["relevant_chunk_ids"],
            model=model,
            k=k,
            candidate_k=candidate_k,
        )
        per_query.append(result)

    macro: dict[str, dict[str, float]] = {}
    for method in ("keyword", "semantic", "hybrid"):
        method_results = [r[method] for r in per_query]
        macro[method] = macro_average_metrics(method_results)

    return {
        "k": k,
        "per_query": per_query,
        "macro_average": macro,
    }


# ---------------------------------------------------------------------------
# Answer Generation and Citation Evaluation Metrics
# ---------------------------------------------------------------------------

def evidence_coverage(
    cited_chunk_ids: Iterable[str],
    relevant_chunk_ids: Iterable[str],
) -> float:
    """Fraction of relevant evidence chunks that were cited in an answer.

    Returns 0.0 if *relevant_chunk_ids* is empty.
    """
    rel_set = set(relevant_chunk_ids)
    if not rel_set:
        return 0.0

    cited_set = set(cited_chunk_ids)
    return len(cited_set & rel_set) / len(rel_set)


def citation_validity_rate(
    cited_labels: Sequence[str],
    valid_labels: Iterable[str],
) -> float | None:
    """Fraction of cited source labels that actually exist in the retrieved evidence.

    Returns None when *cited_labels* is empty, indicating that citation validity
    is undefined because no citations were made.

    Parameters
    ----------
    cited_labels:
        Sequence of citation labels extracted from the answer (e.g. ``["S1", "S2"]``).
    valid_labels:
        Iterable of valid source labels present in the evidence prompt (e.g. ``{"S1", "S2"}``).

    Returns
    -------
    float | None:
        Rate between 0.0 and 1.0, or None if *cited_labels* is empty.
    """
    if not cited_labels:
        return None

    valid_set = set(valid_labels)
    valid_count = sum(1 for label in cited_labels if label in valid_set)
    return valid_count / len(cited_labels)


def answer_grounding_metrics(
    answer_result: Mapping[str, Any],
    relevant_chunk_ids: Iterable[str] | None = None,
    expected_insufficient: bool | None = None,
    valid_labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Calculate answer citation integrity, evidence coverage, and claim verification status.

    Assumptions and Limitations
    ---------------------------
    - **Citation Presence vs. Grounding**: Merely citing sources does NOT establish
      that the generated claims are semantically entailed or factually accurate.
      The LLM may hallucinate claims while citing authentic source labels.
    - **Claim Verification Requirement**: True semantic grounding can only be
      confirmed when an explicit claim-level verifier evaluates the text. If no
      verification was performed, ``is_grounded`` is reported as ``None`` (unknown)
      rather than claiming success.
    - **Citation Validity**: Reflects whether citation labels match trusted sources;
      it does not evaluate semantic entailment of the cited passage.
    - **Evidence Coverage**: Measures whether known relevant chunks were cited; does
      not measure whether the citations were used accurately in context.

    Parameters
    ----------
    answer_result:
        Output dict from :meth:`AnswerGenerator.answer`.
    relevant_chunk_ids:
        Optional known relevant chunk IDs for the question.
    expected_insufficient:
        Optional expected boolean flag indicating whether the question
        should trigger insufficient-evidence handling.
    valid_labels:
        Optional set of known valid source labels. If omitted, uses the labels
        from ``answer_result.get("citations", [])``.

    Returns
    -------
    dict with:
    - ``citation_count``: Total citations included.
    - ``cited_chunk_ids``: List of chunk_ids cited.
    - ``has_citations``: True if at least one citation is present.
    - ``citation_validity_rate``: Rate of valid citations, or None if no citations.
    - ``insufficient_evidence``: Boolean flag from answer_result.
    - ``insufficient_evidence_correct``: True/False if expected_insufficient provided, else None.
    - ``evidence_coverage``: Coverage ratio if relevant_chunk_ids provided, else None.
    - ``claim_verification_status``: 'verified', 'unverified', 'unavailable', or 'not_performed'.
    - ``claim_verified``: True if verified, False if unverified, None if not performed/unavailable.
    - ``is_grounded``: True if confirmed by claim verifier. False if insufficient evidence or unverified.
      None (unknown) if verification was not performed.
    """
    raw_citations = answer_result.get("citations", [])
    valid_citations: list[Mapping[str, Any]] = []
    cited_chunk_ids: list[str] = []
    cited_labels: list[str] = []

    if isinstance(raw_citations, Sequence) and not isinstance(raw_citations, (str, bytes)):
        for c in raw_citations:
            if not isinstance(c, Mapping):
                continue
            lbl = c.get("label")
            cid = c.get("chunk_id")
            if not isinstance(lbl, str) or not lbl.strip():
                continue
            if isinstance(cid, bool) or not isinstance(cid, (str, int)) or not str(cid).strip():
                continue
            valid_citations.append(c)
            cited_labels.append(lbl.strip())
            cited_chunk_ids.append(str(cid).strip())

    is_insufficient = bool(answer_result.get("insufficient_evidence", False))
    has_citations = bool(valid_citations)
    citation_count = len(valid_citations)

    # Citation validity rate: return None when valid_labels is not supplied.
    # Do not assume all cited labels are valid.
    if valid_labels is not None:
        cit_validity = citation_validity_rate(cited_labels, set(valid_labels))
    else:
        cit_validity = None

    coverage = None
    if relevant_chunk_ids is not None:
        coverage = evidence_coverage(cited_chunk_ids, relevant_chunk_ids)

    insufficient_correct = None
    if expected_insufficient is not None:
        insufficient_correct = (is_insufficient == expected_insufficient)

    # Check claim verification status
    verification = answer_result.get("verification")
    if isinstance(verification, Mapping):
        v_avail = bool(verification.get("verification_available", False))
        v_verified = bool(verification.get("verified", False))
        if not v_avail:
            claim_status = "unavailable"
            claim_verified = None
        elif v_verified:
            claim_status = "verified"
            claim_verified = True
        else:
            claim_status = "unverified"
            claim_verified = False
    else:
        claim_status = "not_performed"
        claim_verified = None

    # Overall grounding assessment:
    # Cannot infer semantic claim support from citation labels alone.
    if is_insufficient:
        is_grounded: bool | None = False
    elif not has_citations:
        is_grounded = False
    elif claim_status == "verified":
        is_grounded = True
    elif claim_status == "unverified":
        is_grounded = False
    else:
        # Verification was not performed or was unavailable; semantic grounding is unknown.
        is_grounded = None

    return {
        "citation_count": citation_count,
        "cited_chunk_ids": cited_chunk_ids,
        "has_citations": has_citations,
        "citation_validity_rate": cit_validity,
        "insufficient_evidence": is_insufficient,
        "insufficient_evidence_correct": insufficient_correct,
        "evidence_coverage": coverage,
        "claim_verification_status": claim_status,
        "claim_verified": claim_verified,
        "is_grounded": is_grounded,
    }


def evaluate_answer_generator(
    test_cases: Iterable[Mapping[str, Any]],
    answer_generator: Any,
) -> dict[str, Any]:
    """Evaluate an AnswerGenerator across multiple test cases.

    Each test case must provide:
    - ``question`` (str)
    Optional keys:
    - ``relevant_chunk_ids`` (list[str])
    - ``expected_insufficient`` (bool)
    - ``top_k`` (int)
    - ``filters`` (dict)

    Returns a summary dict with accuracy, grounding rate, and per-case results.
    """
    cases = list(test_cases)
    if not cases:
        return {
            "total_cases": 0,
            "grounded_answers_rate": None,
            "citation_presence_rate": 0.0,
            "insufficient_evidence_accuracy": None,
            "mean_citations_per_answer": 0.0,
            "per_case_results": [],
        }

    per_case: list[dict[str, Any]] = []
    grounded_eval_count = 0
    grounded_true_count = 0
    citation_present_count = 0
    insufficient_correct_count = 0
    insufficient_evaluated_count = 0
    total_citations = 0

    for case in cases:
        question = case["question"]
        top_k = case.get("top_k", 5)
        filters = case.get("filters")
        relevant_ids = case.get("relevant_chunk_ids")
        expected_insufficient = case.get("expected_insufficient")
        valid_labels = case.get("valid_labels")

        result = answer_generator.answer(question, top_k=top_k, filters=filters)
        metrics = answer_grounding_metrics(
            result,
            relevant_chunk_ids=relevant_ids,
            expected_insufficient=expected_insufficient,
            valid_labels=valid_labels,
        )

        if metrics["has_citations"] and not metrics["insufficient_evidence"]:
            citation_present_count += 1

        if metrics["is_grounded"] is not None:
            grounded_eval_count += 1
            if metrics["is_grounded"] is True:
                grounded_true_count += 1

        if metrics["insufficient_evidence_correct"] is not None:
            insufficient_evaluated_count += 1
            if metrics["insufficient_evidence_correct"]:
                insufficient_correct_count += 1

        total_citations += metrics["citation_count"]

        per_case.append({
            "question": question,
            "answer_result": result,
            "metrics": metrics,
        })

    n = len(cases)
    insufficient_acc = (
        insufficient_correct_count / insufficient_evaluated_count
        if insufficient_evaluated_count > 0
        else None
    )
    grounded_rate = (
        grounded_true_count / grounded_eval_count
        if grounded_eval_count > 0
        else None
    )

    return {
        "total_cases": n,
        "grounded_answers_rate": grounded_rate,
        "citation_presence_rate": citation_present_count / n,
        "insufficient_evidence_accuracy": insufficient_acc,
        "mean_citations_per_answer": total_citations / n,
        "per_case_results": per_case,
    }
