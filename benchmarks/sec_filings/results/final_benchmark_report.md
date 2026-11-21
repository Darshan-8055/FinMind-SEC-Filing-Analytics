# FinMind Phase 2: SEC Filing Retrieval Benchmark – Final Statistical Analysis Report

## 1. Executive Summary

This report provides the paired statistical evaluation and comparative analysis for **FinMind Phase 2 (Step 4)**. Three retrieval configurations were evaluated across 54 genuine SEC filing queries (42 answerable, 12 unanswerable) on a corpus of 593 HTML chunks extracted from 8 official SEC EDGAR filings (AAPL, MSFT, AMZN, NVDA).

All queries were executed through the production `FilingIngestionPipeline` under identical parameters ($K=5$, candidate pool $K_{\text{cand}}=30$, RRF constant $K_{\text{rrf}}=60$). Statistical tests comprise paired bootstrap confidence intervals (10,000 resamples, seed 42) and two-sided paired Wilcoxon signed-rank tests with family-wise Holm-Bonferroni correction across all nine comparisons.

---

## 2. Experimental Setup & Dataset Provenance

- **Corpus**: 593 unique chunks derived from 8 primary SEC 10-K and 10-Q HTML filings across Apple, Microsoft, Amazon, and NVIDIA.
- **Queries**: 54 total queries:
  - **42 Answerable Queries**: Target specific financial line items, revenue breakdowns, executive compensations, segment reporting, and operational disclosures. Each possesses verifiable ground-truth evidence quotes extracted verbatim from the corpus.
  - **12 Unanswerable Queries**: Plausible financial inquiries addressing disclosures absent from the ingested filings (e.g., non-existent metrics, foreign reporting concepts, or cross-company acquisitions).
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (dimension: 384), initialized in strict offline mode (`HF_HUB_OFFLINE=1`).
- **Vector Store**: Qdrant in-memory client (`:memory:`), collection `benchmark_eval_chunks`.
- **Sparse Index**: SQLite-backed BM25 Okapi index at `benchmarks/sec_filings/eval_sparse_index.db`.
- **Isolation**: Production vector databases and `data/sparse_index.db` were strictly segregated and untouched.

---

## 3. Original Step 3 Aggregate Benchmark Metrics

Evaluated across the 42 answerable queries at retrieval cutoff $K=5$:

| Metric | Dense-Only (`dense=1.0, sparse=0.0`) | Sparse-Only (`dense=0.0, sparse=1.0`) | Hybrid (`dense=0.7, sparse=0.3`) |
| :--- | :---: | :---: | :---: |
| **Precision@5** | 0.1000 | **0.1667** | 0.1286 |
| **Recall@5** | 0.5000 | **0.8333** | 0.6429 |
| **MRR@5** | 0.3083 | **0.6587** | 0.4722 |

---

## 4. Nine Planned Statistical Comparisons

For each comparison, the paired difference is defined as:
$$\Delta = \text{Score}(\text{System A}) - \text{Score}(\text{System B})$$

### Multiple-Comparison Correction Rationale
When conducting multiple statistical hypothesis tests simultaneously, the likelihood of false positives (Type I error) compounds. With 9 simultaneous tests at nominal $\alpha = 0.05$, the uncorrected family-wise error rate could reach up to $1 - (1 - 0.05)^9 \approx 37\%$. To maintain rigorous control over the Family-Wise Error Rate (FWER $\le 0.05$) across the entire set of planned comparisons, the **Holm-Bonferroni** step-down procedure was applied to all 9 tests treated as a single family.

### Comprehensive Statistical Results Table

| Metric | System Comparison (A vs B) | System A Mean | System B Mean | Mean Paired Diff ($\Delta$) | Higher System | 95% Paired Bootstrap CI | Raw Wilcoxon $p$-value | Holm-Adjusted $p$-value | Significant ($\alpha_{\text{FWER}} = 0.05$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Precision@5** | Dense-only vs Sparse-only | 0.1000 | 0.1667 | -0.0667 | **Sparse-only** | [-0.1000, -0.0381] | 4.6526e-04 | **3.2568e-03** | **Yes** |
| **Recall@5** | Dense-only vs Sparse-only | 0.5000 | 0.8333 | -0.3333 | **Sparse-only** | [-0.5000, -0.1905] | 4.6526e-04 | **3.2568e-03** | **Yes** |
| **MRR@5** | Dense-only vs Sparse-only | 0.3083 | 0.6587 | -0.3504 | **Sparse-only** | [-0.4933, -0.2016] | 4.7041e-05 | **4.2337e-04** | **Yes** |
| **Precision@5** | Hybrid vs Dense-only | 0.1286 | 0.1000 | +0.0286 | **Hybrid** | [+0.0048, +0.0524] | 3.3895e-02 | 6.7790e-02 | No |
| **Recall@5** | Hybrid vs Dense-only | 0.6429 | 0.5000 | +0.1429 | **Hybrid** | [+0.0238, +0.2619] | 3.3895e-02 | 6.7790e-02 | No |
| **MRR@5** | Hybrid vs Dense-only | 0.4722 | 0.3083 | +0.1639 | **Hybrid** | [+0.0619, +0.2611] | 1.9004e-04 | **1.5203e-03** | **Yes** |
| **Precision@5** | Hybrid vs Sparse-only | 0.1286 | 0.1667 | -0.0381 | **Sparse-only** | [-0.0667, -0.0143] | 1.1412e-02 | 5.0831e-02 | No |
| **Recall@5** | Hybrid vs Sparse-only | 0.6429 | 0.8333 | -0.1905 | **Sparse-only** | [-0.3333, -0.0714] | 1.1412e-02 | 5.0831e-02 | No |
| **MRR@5** | Hybrid vs Sparse-only | 0.4722 | 0.6587 | -0.1865 | **Sparse-only** | [-0.3155, -0.0655] | 1.0166e-02 | 5.0831e-02 | No |

---

## 5. Summary of Significant & Non-Significant Findings

### Confirmed Statistically Significant Differences ($\alpha = 0.05$ after Holm correction)
1. **Sparse-only significantly outperforms Dense-only on all metrics**:
   - Precision@5: Sparse is $+0.0667$ higher ($p_{\text{adj}} = 0.0033$).
   - Recall@5: Sparse is $+0.3333$ higher ($p_{\text{adj}} = 0.0033$).
   - MRR@5: Sparse is $+0.3504$ higher ($p_{\text{adj}} = 0.00042$).
2. **Hybrid significantly outperforms Dense-only on Mean Reciprocal Rank (MRR@5)**:
   - MRR@5: Hybrid is $+0.1639$ higher ($p_{\text{adj}} = 0.00152$). The 95% bootstrap CI [+0.0619, +0.2611] strictly excludes zero.

### Non-Significant Differences After Correction
1. **Hybrid vs Dense-only (Precision@5 and Recall@5)**:
   - While Hybrid showed higher sample means ($\Delta P = +0.0286$, $\Delta R = +0.1429$, raw $p = 0.0339$), these do not survive family-wise Holm correction ($p_{\text{adj}} = 0.0678 > 0.05$).
2. **Hybrid vs Sparse-only (Precision@5, Recall@5, MRR@5)**:
   - Sparse-only exhibited higher sample means across all three metrics. Before correction, raw Wilcoxon tests yielded $p \approx 0.0102 - 0.0114$. After Holm correction across the 9 tests, the adjusted $p$-values are $p_{\text{adj}} = 0.05083$. While hovering right at the boundary of significance ($\approx 0.051$), under formal $\alpha = 0.05$ thresholding, the difference cannot be declared statistically significant.

---

## 6. Unanswerable Query Analysis

The 12 unanswerable queries were analyzed independently to determine system behavior on out-of-domain / non-disclosed topics:

| Configuration | Total Unanswerable Queries | Queries Returning $\ge 1$ Chunk | Queries Returning 0 Chunks | Overlap with Ground-Truth Relevant Chunks | Abstention Mechanism Present? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense-only** | 12 | 12 (100%) | 0 (0%) | 0 (0%) | No |
| **Sparse-only** | 12 | 12 (100%) | 0 (0%) | 0 (0%) | No |
| **Hybrid** | 12 | 12 (100%) | 0 (0%) | 0 (0%) | No |

### Important Architectural Interpretation:
- **Retrieval vs. Abstention**: Because the production retrieval pipeline performs standard top-$K$ nearest-neighbor / BM25 search without score-threshold filtering, all 12 unanswerable queries returned 5 chunks.
- **Zero Evidence Overlap**: None of the retrieved chunks matched ground-truth relevant IDs because the target disclosure does not exist in the filings.
- **No Hallucination Prevention Claim**: Retrieving zero ground-truth chunks is a property of the ground-truth definition, not an active abstention by the system. The production pipeline currently lacks an autonomous abstention decision rule.

---

## 7. Limitations & Conservative Interpretation

1. **Benchmark Scope**: The dataset encompasses 42 answerable queries over 8 filings across 4 mega-cap US technology firms. These results cannot be generalized indiscriminately to other SEC forms (e.g., 8-K, S-1), international filings, or non-tech sectors.
2. **Lexical Bias of Financial Questions**: BM25 performance is elevated because financial queries typically feature exact entity names, numeric line items, or specific disclosure terminology that align verbatim with filing text.
3. **Statistical Significance vs Practical Equivalence**: Non-significant comparisons (such as Hybrid vs Sparse-only post-Holm correction) do not establish equivalence between systems; they indicate that on this sample size ($N=42$), differences could not be conclusively distinguished from sampling variation at $\alpha=0.05$.
4. **Architectural Stability**: No retrieval weights or pipeline modifications were altered or tuned to fit these benchmark results.
