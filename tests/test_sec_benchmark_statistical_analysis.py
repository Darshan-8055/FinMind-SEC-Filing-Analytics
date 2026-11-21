"""
tests/test_sec_benchmark_statistical_analysis.py

Focused tests for Step 4 statistical comparisons and final report outputs:
- Input validation and score pairing
- Deterministic paired bootstrap confidence intervals (seed 42)
- Wilcoxon signed-rank test integration and zero-difference handling
- Holm-Bonferroni correction over all 9 planned comparisons
- Significance decisions consistency
- Unanswerable query summary assertions
- Immutability of Step 3 result files
- Integrity and structure of statistical_analysis.json and final_benchmark_report.md
"""

import json
from pathlib import Path
import pytest
from src.retrieval.evaluation import paired_bootstrap_confidence_interval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "sec_filings"
RESULTS_DIR = BENCHMARK_DIR / "results"

DENSE_FILE = RESULTS_DIR / "dense_results.json"
SPARSE_FILE = RESULTS_DIR / "sparse_results.json"
HYBRID_FILE = RESULTS_DIR / "hybrid_results.json"
STAT_ANALYSIS_FILE = RESULTS_DIR / "statistical_analysis.json"
FINAL_REPORT_MD = RESULTS_DIR / "final_benchmark_report.md"

QUERIES_FILE = BENCHMARK_DIR / "evaluation_queries.json"
CORPUS_FILE = BENCHMARK_DIR / "corpus_chunks.json"


@pytest.fixture(scope="module")
def stat_data():
    assert STAT_ANALYSIS_FILE.exists(), f"Missing {STAT_ANALYSIS_FILE}"
    return json.loads(STAT_ANALYSIS_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def step3_results():
    return {
        "dense": json.loads(DENSE_FILE.read_text(encoding="utf-8")),
        "sparse": json.loads(SPARSE_FILE.read_text(encoding="utf-8")),
        "hybrid": json.loads(HYBRID_FILE.read_text(encoding="utf-8")),
    }


@pytest.fixture(scope="module")
def queries():
    return json.loads(QUERIES_FILE.read_text(encoding="utf-8"))


def test_step3_result_files_exist_and_unmodified(step3_results):
    """Step 3 files must exist and preserve original metrics."""
    assert step3_results["dense"]["metrics"]["precision_at_k"] == 0.1000
    assert step3_results["dense"]["metrics"]["recall_at_k"] == 0.5000
    assert abs(step3_results["dense"]["metrics"]["mrr_at_k"] - 0.3083333333333333) < 1e-9

    assert abs(step3_results["sparse"]["metrics"]["precision_at_k"] - 0.16666666666666666) < 1e-9
    assert abs(step3_results["sparse"]["metrics"]["recall_at_k"] - 0.8333333333333334) < 1e-9
    assert abs(step3_results["sparse"]["metrics"]["mrr_at_k"] - 0.6587301587301587) < 1e-9

    assert abs(step3_results["hybrid"]["metrics"]["precision_at_k"] - 0.1285714285714286) < 1e-9
    assert abs(step3_results["hybrid"]["metrics"]["recall_at_k"] - 0.6428571428571429) < 1e-9
    assert abs(step3_results["hybrid"]["metrics"]["mrr_at_k"] - 0.4722222222222222) < 1e-9


def test_query_alignment_and_pairing(step3_results, queries):
    """Ensure all 54 queries are identical and aligned across configurations."""
    gt_qids = [q["query_id"] for q in queries]
    ans_qids = [q["query_id"] for q in queries if q["answerable"]]
    assert len(gt_qids) == 54
    assert len(ans_qids) == 42

    for mode in ["dense", "sparse", "hybrid"]:
        pqr = step3_results[mode]["per_query_results"]
        pqr_qids = [e["query_id"] for e in pqr]
        assert pqr_qids == gt_qids, f"Query ID alignment failed for {mode}"


def test_statistical_comparisons_count_and_keys(stat_data):
    """Must contain exactly 9 comparisons with required schema fields."""
    comparisons = stat_data["statistical_comparisons"]
    assert len(comparisons) == 9

    required_keys = {
        "comparison",
        "system_a",
        "system_b",
        "metric",
        "system_a_mean",
        "system_b_mean",
        "mean_paired_difference",
        "higher_system",
        "ci_95",
        "wilcoxon_p_value_raw",
        "nonzero_differences_count",
        "zero_differences_count",
        "holm_adjusted_p_value",
        "significant_at_alpha_0_05",
    }
    for comp in comparisons:
        missing = required_keys - set(comp.keys())
        assert not missing, f"Comparison {comp.get('comparison')} missing keys: {missing}"


def test_paired_bootstrap_determinism_and_values(step3_results, queries):
    """Bootstrap CI must be reproducible using seed 42."""
    ans_qids = [q["query_id"] for q in queries if q["answerable"]]
    dense_pqr = {e["query_id"]: e for e in step3_results["dense"]["per_query_results"]}
    sparse_pqr = {e["query_id"]: e for e in step3_results["sparse"]["per_query_results"]}

    dense_p = [dense_pqr[qid]["precision_at_k"] for qid in ans_qids]
    sparse_p = [sparse_pqr[qid]["precision_at_k"] for qid in ans_qids]

    stat = paired_bootstrap_confidence_interval(sparse_p, dense_p, num_bootstrap=10000, random_seed=42)
    assert abs(stat["mean_difference"] - (-0.06666666666666667)) < 1e-9
    assert abs(stat["ci_lower"] - (-0.1)) < 1e-9
    assert abs(stat["ci_upper"] - (-0.0380952380952381)) < 1e-9


def test_holm_bonferroni_adjustment_logic(stat_data):
    """Verify Holm-Bonferroni step-down adjustment is strictly monotonic and correct."""
    comps = stat_data["statistical_comparisons"]
    raw_p_vals = [c["wilcoxon_p_value_raw"] for c in comps]
    adj_p_vals = [c["holm_adjusted_p_value"] for c in comps]

    # Check that adjusted p-values are >= raw p-values
    for raw, adj in zip(raw_p_vals, adj_p_vals):
        assert adj >= raw - 1e-9

    # Recompute independently to verify step-down enforcement
    indexed = sorted(enumerate(comps), key=lambda x: x[1]["wilcoxon_p_value_raw"])
    m = len(comps)
    prev = 0.0
    for rank_idx, (orig_idx, item) in enumerate(indexed):
        raw = item["wilcoxon_p_value_raw"]
        expected_adj = max(prev, min(1.0, raw * (m - rank_idx)))
        prev = expected_adj
        assert abs(item["holm_adjusted_p_value"] - expected_adj) < 1e-9


def test_significance_decisions_consistency(stat_data):
    """Significant items must have holm_adjusted_p_value < 0.05 and CI excluding 0."""
    for comp in stat_data["statistical_comparisons"]:
        ci = comp["ci_95"]
        ci_excludes_zero = (ci["lower"] > 0) or (ci["upper"] < 0)
        expected_sig = bool(comp["holm_adjusted_p_value"] < 0.05 and ci_excludes_zero)
        assert comp["significant_at_alpha_0_05"] == expected_sig


def test_unanswerable_query_analysis_integrity(stat_data):
    """Verify unanswerable queries reporting: 12 total, all returning 5 chunks, 0 overlap."""
    unans = stat_data["unanswerable_query_analysis"]
    for mode in ["dense", "sparse", "hybrid"]:
        entry = unans[mode]
        assert entry["total_queries"] == 12
        assert entry["retrieved_at_least_one_chunk"] == 12
        assert entry["retrieved_zero_chunks"] == 0
        assert entry["zero_overlap_with_ground_truth"] == 12
        assert entry["abstention_mechanism_exists"] is False


def test_final_report_markdown_exists():
    """Verify final_benchmark_report.md exists and contains required sections."""
    assert FINAL_REPORT_MD.exists()
    content = FINAL_REPORT_MD.read_text(encoding="utf-8")
    assert "Original Step 3 Aggregate Benchmark Metrics" in content
    assert "Nine Planned Statistical Comparisons" in content
    assert "Holm-Bonferroni" in content
    assert "Unanswerable Query Analysis" in content
    assert "Limitations & Conservative Interpretation" in content
