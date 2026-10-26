# FinMind Shared Interfaces

## 1. Purpose

This document defines the document chunk format used by the FinMind ingestion, retrieval, and answer-generation pipeline.

The Python chunk schema is the shared interface between Member 1's ingestion and retrieval modules and the rest of the application. Database field mappings follow the proposed PostgreSQL schema shared by Member 2.

**Status:** Draft — implementation may proceed; database integration details remain subject to team agreement.

## 2. Document Chunk Schema

Each chunk represents a passage extracted from an SEC filing.

| Field | Expected type | Purpose |
|---|---|---|
| `chunk_id` | string | Stable, unique identifier for the chunk |
| `company_name` | string | Company legal or display name |
| `ticker` | string | Company ticker symbol |
| `form_type` | string | SEC filing type, such as 10-K or 10-Q |
| `filing_date` | string | Filing date in YYYY-MM-DD format |
| `period_end` | string | Reporting period end in YYYY-MM-DD format, when available |
| `section` | string | Filing section containing the passage |
| `text` | string | Extracted passage text |
| `source_url` | string | URL of the original SEC filing |

The fields above describe the shared Python chunk representation. Fields may be populated from the filing record or company record rather than extracted directly from passage text.

## 3. Database Mapping

The proposed PostgreSQL schema uses the following mapping:

| Python interface | Proposed database field | Table |
|---|---|---|
| `ticker` | `ticker` | `companies` |
| `company_name` | `company_name` | `companies` |
| `form_type` | `form_type` | `filings` |
| `filing_date` | `filing_date` | `filings` |
| `source_url` | `document_url` | `filings` |
| `chunk_id` | `chunk_id` | `filing_chunks` |
| `section` | `section` | `filing_chunks` |
| `text` | `chunk_text` | `filing_chunks` |
| `period_end` | To be confirmed | To be confirmed |
| Qdrant point reference | `qdrant_point_id` | `filing_chunks` |

The database relationships use `company_id` to associate companies with filings and `filing_id` to associate filings with chunks.

The database mapping is a proposed integration contract, not a claim that database persistence has already been implemented.

## 4. Illustrative Example

```json
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
```

This is an illustrative example, not a verified filing record. Replace example values with values from an actual SEC filing before using the record in the application.

## 5. Contract Rules

1. Chunk IDs must be stable and unique within the agreed identifier scope.
2. Preserve the original filing source URL.
3. Keep filing dates and reporting period dates distinct.
4. Preserve the exact extracted passage text.
5. Do not fabricate citations or source references.
6. Keep Python interface field names separate from database field names when an explicit mapping exists.
7. Document and test schema changes before integration.
8. Preserve relevant company and filing metadata when returning retrieved chunks.

## 6. Member 1 Retrieval Requirements

The retrieval module needs:

- A searchable text field.
- A stable chunk identifier.
- Company and filing metadata for filtering.
- A source reference for citation generation.

Keyword and semantic retrieval must return the matched passage together with its original metadata.

Semantic retrieval currently supports an embedding model and in-memory chunks. Integration with Qdrant is a separate step.

## 7. Approval

- Schema owner: Member 2
- Consumer: Member 1
- Status: Draft — proposed mapping documented; database details to be confirmed during integration
- Last updated: 2026-10-26