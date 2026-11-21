# FinMind Phase 2 — Real SEC Filing Retrieval Benchmark

## 1. Overview

This dataset establishes an authentic, evidence-verified retrieval benchmark for evaluating SEC filing analytics in the FinMind platform. It isolates benchmarking data completely from production Qdrant collections and production SQLite stores.

## 2. Benchmark Corpus Summary

- **Total Authentic SEC Filings**: 8
- **Companies**: 4 (Apple Inc., Microsoft Corp., Amazon.com Inc., NVIDIA Corp.)
- **Filing Types**: 10-K (Annual Reports) and 10-Q (Quarterly Reports)
- **Total Valid Chunks**: 593 unique, non-empty text chunks
- **Chunk Format**: Standard FinMind chunk schema (`chunk_id`, `ticker`, `company_name`, `form_type`, `filing_date`, `section`, `text`, `source_url`, `filename`, `chunk_index`, `cik`, `accession`)
- **Chunk ID Convention**: `{TICKER}:{FILING_DATE}:{FILENAME}:{CHUNK_INDEX}` (deterministic and stable across rebuilds)

## 3. Source Filing Manifest

| Ticker | Company Name | CIK | Form | Filing Date | Accession Number | Document | SHA-256 Hash | SEC EDGAR Source URL |
|---|---|---|---|---|---|---|---|---|
| AAPL | Apple Inc. | 0000320193 | 10-K | 2025-10-31 | 0000320193-25-000079 | aapl-20250927.htm | 95506f43d12b... | https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm |
| AAPL | Apple Inc. | 0000320193 | 10-Q | 2026-07-31 | 0000320193-26-000020 | aapl-20260627.htm | 3606c4c9c10c... | https://www.sec.gov/Archives/edgar/data/320193/000032019326000020/aapl-20260627.htm |
| MSFT | Microsoft Corporation | 0000789019 | 10-K | 2026-07-29 | 0001193125-26-323660 | msft-20260630.htm | 97c55cf25c5e... | https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/msft-20260630.htm |
| MSFT | Microsoft Corporation | 0000789019 | 10-Q | 2026-10-28 | 0001193125-26-436133 | msft-20260930.htm | 501c90ebf1de... | https://www.sec.gov/Archives/edgar/data/789019/000119312526436133/msft-20260930.htm |
| AMZN | Amazon.com, Inc. | 0001018724 | 10-K | 2026-02-06 | 0001018724-26-000004 | amzn-20251231.htm | 919916439612... | https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm |
| AMZN | Amazon.com, Inc. | 0001018724 | 10-Q | 2026-10-30 | 0001018724-26-000040 | amzn-20260930.htm | 93cd44cbf2fe... | https://www.sec.gov/Archives/edgar/data/1018724/000101872426000040/amzn-20260930.htm |
| NVDA | NVIDIA Corporation | 0001045810 | 10-K | 2026-02-25 | 0001045810-26-000021 | nvda-20260125.htm | be43a76d54d9... | https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm |
| NVDA | NVIDIA Corporation | 0001045810 | 10-Q | 2026-08-26 | 0001045810-26-000075 | nvda-20260726.htm | e728694fe788... | https://www.sec.gov/Archives/edgar/data/1045810/000104581026000075/nvda-20260726.htm |

## 4. Query Benchmark Summary

- **Total Queries**: 54
- **Answerable Queries**: 42 (all linked to exact supporting evidence passages and validated chunk IDs)
- **Unanswerable Queries**: 12 (covering out-of-corpus companies, non-disclosed subsegment/operational figures, and non-financial metrics)
- **Manual Verification Protocol**:
  1. Relevant chunk IDs are verified to exist in `corpus_chunks.json`.
  2. Each answerable query contains an exact supporting `evidence_quote` extracted directly from the filing text.
  3. A specific `rationale` documents the financial/operational relevance of the retrieved chunk.
  4. Unanswerable queries explicitly specify the lack of evidence in the benchmark corpus.

## 5. Directory Layout

```
benchmarks/sec_filings/
├── README.md                  # This documentation
├── source_filings.json        # Raw filing manifest with CIKs, URLs, hashes
├── corpus_chunks.json         # 593 parsed and adapted SEC filing chunks
├── evaluation_queries.json    # 54 benchmark queries with ground-truth evidence
└── raw/                       # Original authentic SEC HTML filing documents
```
