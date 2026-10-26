# FinMind Shared Interfaces

## 1. Purpose

This document defines the document chunk format used by the
FinMind retrieval and answer-generation pipeline.

Status: DRAFT - pending confirmation from Member 2.

## 2. Document Chunk Schema

Each chunk represents a passage extracted from an SEC filing.

| Field | Expected type | Purpose |
|---|---|---|
| chunk_id | string | Stable, unique identifier for the chunk |
| company_name | string | Company legal or display name |
| ticker | string | Company ticker symbol |
| form_type | string | SEC filing type, such as 10-K or 10-Q |
| filing_date | string | Filing date in YYYY-MM-DD format |
| period_end | string | Reporting period end in YYYY-MM-DD format |
| section | string | Filing section containing the passage |
| text | string | Extracted passage text |
| source_url | string | URL of the original SEC filing |

The field names, types, and required/optional status must be
confirmed with Member 2 before integration.

## 3. Illustrative Example

{
  "chunk_id": "AAPL_10K_2025_RISK_001",
  "company_name": "Apple Inc.",
  "ticker": "AAPL",
  "form_type": "10-K",
  "filing_date": "2025-10-31",
  "period_end": "2025-09-27",
  "section": "Risk Factors",
  "text": "Illustrative filing passage goes here.",
  "source_url": "https://www.sec.gov/Archives/example"
}

This is an illustrative example, not a verified filing record.
Replace example values with values from an actual SEC filing
before using the record in the application.

## 4. Contract Rules

1. Chunk IDs must be stable and unique.
2. Preserve the original filing source URL.
3. Keep filing dates and reporting period dates distinct.
4. Preserve the exact extracted passage text.
5. Do not fabricate citations or source references.
6. Document and test schema changes before integration.

## 5. Member 1 Retrieval Requirements

The retrieval module needs:
- A searchable text field.
- A stable chunk identifier.
- Company and filing metadata for filtering.
- A source reference for citation generation.

The retrieval module must return the matched passage together
with its original metadata.

## 6. Approval

- Schema owner: Member 2
- Consumer: Member 1
- Status: Pending team agreement
- Last updated: 2026-10-26
