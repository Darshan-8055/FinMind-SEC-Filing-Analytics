# FinMind Shared Interfaces

## 1. Purpose

This document defines the document chunk format used by the FinMind ingestion, retrieval, and answer-generation pipeline.

The Python chunk schema is the shared interface between Member 1's ingestion and retrieval modules and the rest of the application. Database field mappings follow the proposed PostgreSQL schema owned and designed by Member 2.

**Status:** Draft — implementation may proceed; database integration details remain subject to team agreement. Member 2 owns the database schema.

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

## 3. Database Mapping (Owned by Member 2)

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

The database mapping is a proposed integration contract owned by Member 2, not a claim that database persistence has already been implemented.

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

Semantic retrieval supports dense vector search via Qdrant (`QdrantChunkStore`) and in-memory cosine similarity, alongside sparse BM25 retrieval and weighted reciprocal-rank fusion hybrid search.

## 7. Approval

- Schema owner: Member 2
- Consumer: Member 1
- Status: Draft — proposed mapping documented; database details to be confirmed during integration
- Last updated: 2026-11-07

---

## 8. Member 1 Ingestion and Retrieval Pipeline

### Pipeline Architecture

```
Filing HTML / Text
       │
       ▼
[narrative_parser]  ──► Section-aware element extraction & token-aware chunking
       │
       ▼
[chunk_adapter]     ──► Validates & shapes chunks to Shared Schema (Section 2)
       │
       ▼
[FilingIngestionPipeline]
       ├──► Model encodes chunks into 384-d vectors (all-MiniLM-L6-v2)
       └──► QdrantChunkStore stores payload & embeddings
       │
       ▼
Search API (`pipeline.search(query, top_k=5, filters={"ticker": "AAPL"})`)
       └──► Cosine vector similarity search over Qdrant
```

### Retrieval Methods Provided

1. **Keyword Retrieval** (`keyword_search`): BM25 term-matching over chunk text.
2. **Semantic Retrieval** (`semantic_search` / `pipeline.search`): Dense embeddings with cosine similarity.
3. **Hybrid Retrieval** (`hybrid_search`): Weighted Reciprocal Rank Fusion (RRF) combining dense and BM25 candidate pools.

---

## 9. Evidence-Grounded Answer Generator Interface

### Class: `AnswerGenerator` (`src/agents/answer_generator.py`)

Takes an injected retrieval callable and an LLM generation callable.

```python
from src.agents.answer_generator import AnswerGenerator

generator = AnswerGenerator(
    retrieve=pipeline.search,
    generate=llm_generate_callable,
    verifier=claim_verifier,              # Optional ClaimVerifier
    require_verified_claims=False,        # When True, unverified claims trigger insufficient_evidence
)
```

### Input Parameters for `answer()`

- `question` (`str`): Non-empty user question.
- `top_k` (`int`, optional, default 5): Number of chunks to retrieve.
- `filters` (`Mapping[str, str]`, optional): Metadata filters passed to the retriever (e.g. `{"ticker": "AAPL"}`).

### Output Schema for `answer()`

The return dictionary contains:

| `question` | `str` | The original question text |
| `answer` | `str` | Cleaned answer prose with unknown citation markers safely stripped |
| `citations` | `list[dict]` | Trusted metadata for every valid cited source label |
| `insufficient_evidence` | `bool` | True if evidence was missing, unanswerable, or ungrounded |
| `potential_conflicts` | `list[dict]` | Detected temporal or reporting-period variances across evidence chunks |
| `verification` | `dict` (optional) | Present when a `ClaimVerifier` was provided or when `require_verified_claims=True` |

### Citation Metadata Structure

Citations are built strictly from the trusted source registry, never from LLM text:

```json
{
  "label": "S1",
  "chunk_id": "AAPL:2025-10-31:aapl-10k.htm:0",
  "company_name": "Apple Inc.",
  "ticker": "AAPL",
  "form_type": "10-K",
  "filing_date": "2025-10-31",
  "period_end": "2025-09-27",
  "section": "Item 1A. Risk Factors",
  "source_url": "https://www.sec.gov/Archives/example"
}
```

---

## 10. Insufficient and Conflicting Evidence Policies

### Insufficient Evidence Policy

The answer generator returns `insufficient_evidence=True` and clear messaging in the following cases:

1. **No retrieved chunks**: Returns `INSUFFICIENT_EVIDENCE_MESSAGE` immediately without calling the LLM.
2. **Empty or invalid LLM response**: Returns `INSUFFICIENT_EVIDENCE_MESSAGE` safely rather than crashing.
3. **Explicit LLM admission**: When the LLM indicates available passages do not answer the question.
4. **No valid citations**: If the answer text cites no recognized source labels, it is marked as ungrounded.
5. **Only unknown labels cited**: If fabricated citation markers (e.g. `[S99]`) are stripped leaving no valid citations.
6. **Unverified or unsupported claims**: When `require_verified_claims=True`, any unverified, unavailable, or malformed verification result triggers `insufficient_evidence=True`.

### Conflicting Evidence Policy

1. **Prompt Isolation**: Passage text is enclosed in `<<< ... >>>` delimiters and treated as untrusted data.
2. **Explicit Explanations**: The prompt instructs the LLM that if passages conflict (differing values across filings, updated dates, or contrasting sections), it must explain the discrepancy and cite all relevant sources.
3. **Context Sensitivity**: The model is instructed not to blindly assume the newest filing is correct without considering the question's specific context (e.g. questions regarding prior fiscal years or initial guidance).
4. **Potential Conflict Detection (`detect_potential_conflicts`)**: Identifies temporal variances (multiple filing dates) and reporting period differences across chunks grouped by entity identity.
   - *Limitation*: This detects temporal/period variances; it does not evaluate semantic contradictions or verify factual accuracy.

---

## 11. Citation Integrity vs. Semantic Claim Verification

There is a critical architectural distinction between syntactic citation validation and semantic claim verification:

1. **Citation Integrity (Syntactic Validation)**:
   - Validates that label markers (e.g. `[S1]`) match real retrieved chunks.
   - Replaces or strips unknown markers (e.g. `[S99]`).
   - Retrieves metadata from trusted source dictionaries, never from generated text.
   - *Limitation*: A citation label only proves the model referenced a source; it does not prove the claim is factually entailed by that passage.

2. **Claim Verification (Semantic Verification)**:
   - Evaluated by `ClaimVerifier` via an injected verification callable.
   - Checks whether individual propositions in the answer are factually supported by cited passages.
   - Identifies specific unsupported claims or substantive contradictions.
   - When `require_verified_claims=True`, an unverified answer cannot be accepted as grounded.

---

## 12. Downstream Integration Guide (Member 4 / Agents)

Other members building agentic workflows (e.g. LangGraph research agents) can integrate directly using:

```python
from src.ingestion.pipeline import FilingIngestionPipeline
from src.retrieval.qdrant_store import QdrantChunkStore
from src.agents.answer_generator import AnswerGenerator, ClaimVerifier

# 1. Connect to pipeline search
answer_generator = AnswerGenerator(
    retrieve=pipeline.search,
    generate=agent_llm_callable,
    verifier=ClaimVerifier(agent_verifier_callable),
    require_verified_claims=True,  # Conservative mode
)

# 2. Invoke for query
result = answer_generator.answer("What were Apple's primary supply chain risks in 2025?")

# 3. Consume structured output
if result["insufficient_evidence"]:
    print("Action: Fall back to broader search or notify user")
else:
    print("Answer:", result["answer"])
    for citation in result["citations"]:
        print(f"[{citation['label']}] {citation['ticker']} {citation['section']} ({citation['filing_date']}) - {citation['source_url']}")
```