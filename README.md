# FinMind — SEC Filing Analytics

Agentic Financial Intelligence and SEC Filing Analytics Platform.

## Member 1: Retrieval & Evidence Generation Architect

Responsibilities:

- Narrative parsing and token-aware chunking of SEC 10-K / 10-Q filings
- Shared chunk schema and ingestion pipeline
- Qdrant vector storage
- Keyword (BM25) retrieval
- Semantic (dense) retrieval using sentence-transformers
- Hybrid retrieval using weighted reciprocal-rank fusion
- Evidence-grounded answer generation with trusted citation metadata
- Robust insufficient-evidence and conflicting-evidence handling
- Claim-level verification interface (`ClaimVerifier`)
- Retrieval and answer generation evaluation — Precision@K, Recall@K, Evidence Coverage, Citation Validity Rate, Grounding metrics

---

## Project structure

```
src/
  agents/
    answer_generator.py   # Grounded answer generation, citations & claim verification
  ingestion/
    narrative_parser.py   # SEC filing text → section-aware chunks
    chunk_adapter.py      # Parser output → shared chunk interface schema
    pipeline.py           # End-to-end ingestion: parse → embed → store
  retrieval/
    keyword_search.py     # BM25-based keyword retrieval
    semantic_search.py    # Cosine-similarity semantic retrieval
    hybrid_search.py      # Weighted RRF hybrid (dense + sparse)
    sparse_index.py       # SQLite-backed persistent BM25 sparse index
    qdrant_store.py       # Qdrant vector store wrapper
    evaluation.py         # Precision@K, Recall@K, grounding & coverage metrics

tests/
  fixtures/
    retrieval_eval.json   # Synthetic evaluation queries and relevance labels
  test_*.py               # Automated unit and integration tests

docs/
  INTERFACES.md           # Shared chunk schema, database mapping, and integration contract

run_retrieval_evaluation.py   # CLI script: compare all three retrievers

```

---

## Setup

### Prerequisites

- Python 3.10 or later
- Docker (for running a local Qdrant service)

### 1. Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -e ".[dev]"
```

Or using `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Core dependencies:

| Package | Purpose |
|---|---|
| `sentence-transformers>=3.0` | Embedding model for semantic retrieval |
| `qdrant-client>=1.9` | Qdrant vector store client |
| `rank-bm25>=0.2.2` | BM25 keyword scoring |
| `tiktoken>=0.5` | Token-aware chunking |
| `unstructured>=0.10` | SEC filing element extraction (optional fallback) |

### 3. Start the local Qdrant service

```powershell
docker run -p 6333:6333 qdrant/qdrant
```

The integration tests that require Qdrant connect to `localhost:6333` by default.

---

## Running tests

Run the full test suite (all retrieval, ingestion, answer generation, and integration tests):

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
```

Run only the answer generator tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_answer_generator.py -v
```

Run only the retrieval evaluation tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_retrieval_evaluation.py -v
```

Run the Qdrant hybrid retrieval tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_qdrant_hybrid_search.py -v
```


The tests use a `FakeEmbeddingModel` (a deterministic 12-word bag-of-words
encoder) and mock callables so they do **not** require downloading a
sentence-transformer model or connecting to external LLM APIs. Integration tests
against Qdrant use in-memory instances (`:memory:`).

---

## Minimal usage examples

### Ingestion pipeline

```python
from qdrant_client import QdrantClient
from src.retrieval.qdrant_store import QdrantChunkStore
from src.retrieval.semantic_search import load_embedding_model
from src.ingestion.pipeline import FilingIngestionPipeline

client = QdrantClient(host="localhost", port=6333)
model = load_embedding_model()  # downloads all-MiniLM-L6-v2 on first run

store = QdrantChunkStore(client, vector_size=384)
pipeline = FilingIngestionPipeline(store, model)

with open("aapl_10k.txt", encoding="utf-8") as fh:
    text = fh.read()

count = pipeline.ingest_filing(
    text,
    filename="aapl_10k.txt",
    filing_type="10-K",
    ticker="AAPL",
    company_name="Apple Inc.",
    filing_date="2025-10-31",
    source_url="https://www.sec.gov/Archives/example",
    period_end="2025-09-27",
)
print(f"Stored {count} chunks")
```

### In-memory retrieval (no Qdrant required)

```python
from src.retrieval.keyword_search import keyword_search
from src.retrieval.semantic_search import semantic_search, load_embedding_model
from src.retrieval.hybrid_search import hybrid_search

chunks = [
    {"chunk_id": "c1", "text": "Supply chain disruptions may increase costs."},
    {"chunk_id": "c2", "text": "Revenue grew due to strong services demand."},
]

model = load_embedding_model()
query = "supply chain manufacturing costs"

keyword_results = keyword_search(query, chunks, top_k=3)
semantic_results = semantic_search(query, chunks, model, top_k=3)
hybrid_results  = hybrid_search(query, chunks, model, top_k=3)
```

### Qdrant-backed scalable hybrid retrieval

```python
from src.retrieval.qdrant_store import QdrantChunkStore
from src.retrieval.sparse_index import SparseKeywordIndex
from src.ingestion.pipeline import FilingIngestionPipeline

# Default pipeline automatically connects to persistent SQLite at data/sparse_index.db:
pipeline = FilingIngestionPipeline(store, model)

# Or pass custom path / in-memory index for tests:
# pipeline = FilingIngestionPipeline(store, model, SparseKeywordIndex(":memory:"))

# Ingest filings via pipeline (synchronously populates Qdrant and SQLite sparse index)
pipeline.ingest_filing(text, filename="aapl-10k.txt", ...)

# Check index synchronization status (preliminary count check against Qdrant):
needed, reason = pipeline.check_reconciliation_needed()
if needed:
    print(f"Reconciliation recommended: {reason}")
    pipeline.sparse_index.sync_from_store(pipeline.store)

# Scalable hybrid search combining dense vectors and BM25 via Reciprocal Rank Fusion:
results = pipeline.hybrid_search(
    "supply chain disruptions and tariffs",
    top_k=5,
    filters={"ticker": "AAPL", "form_type": "10-K"},
    dense_weight=0.7,
    sparse_weight=0.3,
)

for r in results:
    print(f"[{r['ticker']}] {r['section']} (Hybrid: {r['hybrid_score']:.4f}, Dense: {r['semantic_score']:.4f}, BM25: {r['bm25_score']:.4f})")
```

**Key Architectural Features:**
- **Persistent Storage:** Default SQLite database anchored to project root at `data/sparse_index.db`.
- **Lifecycle States:** `uninitialized` (fresh, empty), `ready` (synchronized), `building` (staging rebuild in progress), `stale` (reconciliation needed or interrupted), `failed` (fatal write error).
- **Outage Resilience:** SQLite stores full chunk payloads so sparse-only retrieval can return complete chunk records even during total Qdrant service downtime.
- **Reconciliation:** Preliminary $O(1)$ point-count check; full reconciliation via `sync_from_store()` uses staging tables and preserves the last-known-good index on error. Note that point-count equality is only a sanity check and does not guarantee content equality.
- **Strict Branch Control:** Setting `dense_weight=0` or `sparse_weight=0` disables that branch completely; disabled branches are never activated as fallbacks.


### Evidence-grounded answer generation


```python
from src.agents.answer_generator import AnswerGenerator, ClaimVerifier

answer_gen = AnswerGenerator(
    retrieve=pipeline.search,
    generate=my_llm_callable,
    verifier=ClaimVerifier(my_verifier_callable),
    require_verified_claims=True,
)

response = answer_gen.answer(
    "What risks are discussed in Apple's latest filing?",
    top_k=5,
    filters={"ticker": "AAPL"},
)

print("Answer:", response["answer"])
for citation in response["citations"]:
    print(f"[{citation['label']}] {citation['ticker']} {citation['section']} ({citation['filing_date']})")
    print(f"URL: {citation['source_url']}")
```

---

## Retrieval & Answer Evaluation

### Running the evaluation script

```powershell
.venv\Scripts\python.exe run_retrieval_evaluation.py
```

> **Warning — Synthetic Data Disclaimer:**
> The evaluation script uses a **synthetic 4-chunk fixture** (`tests/fixtures/retrieval_eval.json`).
> These synthetic fixture results are unit test verification artifacts and **must not be reported as
> production retrieval quality or real-world accuracy figures**. Production readiness requires validation
> on real SEC filings.

### Evaluation API

```python
import json
from src.retrieval.evaluation import evaluate_all_queries
from src.retrieval.semantic_search import load_embedding_model

with open("tests/fixtures/retrieval_eval.json") as fh:
    dataset = json.load(fh)

model = load_embedding_model()

output = evaluate_all_queries(
    queries=dataset["queries"],
    chunks=dataset["chunks"],
    model=model,
    k=3,
)

# Per-query results
for entry in output["per_query"]:
    print(entry["query"], entry["keyword"]["recall_at_k"])

# Macro-average across all queries
print(output["macro_average"])
# {'keyword': {'precision_at_k': ..., 'recall_at_k': ...},
#  'semantic': {'precision_at_k': ..., 'recall_at_k': ...},
#  'hybrid':   {'precision_at_k': ..., 'recall_at_k': ...}}
```

### Interpreting the metrics

| Metric | Definition | Convention / Undefined Handling |
|---|---|---|
| **Precision@K** | Fraction of top-K results that are relevant | 0.0 when result list is empty |
| **Recall@K** | Fraction of all relevant docs found in top-K | 0.0 when relevant set is empty |
| **Macro-average** | Mean of per-query metric, equal weight per query | 0.0 when query list is empty |
| **Evidence Coverage** | Fraction of known relevant chunks cited in answer | 0.0 when relevant set is empty |
| **Citation Validity Rate** | Fraction of citations that match valid retrieved sources | `None` when no citations are made (undefined) |
| **Answer Grounding** | Confirms claims are supported by verified evidence | `True` only if claim-verified; `None` if unverified; `False` if insufficient |

### Citation Integrity vs. Semantic Claim Verification

A key architectural distinction is maintained in FinMind:
- **Citation Integrity (Syntactic)**: Confirms that citation markers (e.g. `[S1]`) refer to genuine retrieved chunks, removes unknown labels, and builds metadata strictly from trusted source registries. However, citing a source does *not* prove that the generated text is factually entailed by that source.
- **Claim Verification (Semantic)**: A separate verification layer (`ClaimVerifier`) that evaluates whether individual factual claims in the answer are supported by the cited passages. When `require_verified_claims=True`, an answer whose claims cannot be verified is treated conservatively as insufficient evidence.

### Potential Conflict Detection Limitations

`detect_potential_conflicts()` identifies temporal and reporting-period variances (e.g. chunks spanning multiple filing dates or different fiscal periods for the same company).
- It does **not** evaluate semantic contradictions or verify factual accuracy.
- Downstream systems must not treat different filing dates as proven errors or automatically assume the newest filing is correct without examining the user's specific context.

---

## Environment requirements

| Requirement | Notes |
|---|---|
| Local Qdrant service | Required for ingestion and Qdrant integration tests. Default: `localhost:6333`. |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` — downloaded automatically (384-dim vectors). |
| Qdrant collection vector size | Must match the model dimension (384 for `all-MiniLM-L6-v2`). |

---

## Real-data validation

Integration tests that run against a live Qdrant instance are in
`tests/test_qdrant_store.py`, `tests/test_ingestion_retrieval_integration.py`,
and `tests/test_semantic_ingestion_integration.py`.

### Phase 2 Real SEC Filing Retrieval Benchmark

An authentic SEC filing retrieval benchmark was constructed and evaluated in Phase 2 to validate retrieval quality on real financial disclosures:

- **8 Authentic SEC Filings**: 10-K and 10-Q reports from 4 major companies (Apple, Microsoft, Amazon, and NVIDIA).
- **593 Parsed Chunks**: Extracted and indexed using the standard chunk schema and deterministic chunk IDs.
- **54 Evaluation Queries**: 42 answerable queries (each verified with verbatim supporting quotes and relevant chunk IDs) and 12 unanswerable queries (assessing handling of non-disclosed metrics).
- **Retrieval Configurations**: Dense-only (`dense=1.0, sparse=0.0`), Sparse-only (`dense=0.0, sparse=1.0`), and Hybrid (`dense=0.7, sparse=0.3` with RRF fusion, $k_{\text{rrf}}=60$).
- **Statistical Comparisons**: Nine paired statistical comparisons using 10,000-replicate paired bootstrap confidence intervals (seed 42) and two-sided paired Wilcoxon signed-rank tests with family-wise Holm-Bonferroni correction across all nine tests.

For detailed dataset documentation and provenance, see [`benchmarks/sec_filings/README.md`](benchmarks/sec_filings/README.md).
For the complete statistical evaluation, paired comparisons, and domain findings, see [`benchmarks/sec_filings/results/final_benchmark_report.md`](benchmarks/sec_filings/results/final_benchmark_report.md).

---

## SEC Benchmark Execution

To execute or reproduce the benchmark:

### 1. Environment & Dependencies

- Python 3.10 or later in a virtual environment (`.venv`).
- Core dependencies declared in `pyproject.toml` / `requirements.txt`: `sentence-transformers>=3.0`, `qdrant-client>=1.9`, `rank-bm25>=0.2.2`, `tiktoken>=0.5`, `unstructured>=0.10`.
- SciPy enables Wilcoxon signed-rank statistical testing (the evaluation pipeline includes a graceful fallback if SciPy is unavailable):

```powershell
python -m pip install scipy
```

- The embedding model `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors) must be available in the local HuggingFace cache.

### 2. Offline Execution

To enforce offline execution without external network calls to HuggingFace:

```powershell
$env:HF_HUB_OFFLINE = "1"
```

### 3. Run Benchmark Command

Execute the benchmark runner script:

```powershell
.venv\Scripts\python.exe run_sec_benchmark.py
```

### 4. Benchmark Artifact Locations

- **Result JSON files**: `benchmarks/sec_filings/results/dense_results.json`, `sparse_results.json`, `hybrid_results.json`.
- **Statistical Analysis Payload**: `benchmarks/sec_filings/results/statistical_analysis.json`.
- **Final Markdown Report**: `benchmarks/sec_filings/results/final_benchmark_report.md`.
- **Isolated SQLite Benchmark Store**: `benchmarks/sec_filings/eval_sparse_index.db` (segregated from production `data/sparse_index.db`).


---

## Shared interface contract

The chunk schema and integration contract shared between Member 1 and downstream
members is documented in [`docs/INTERFACES.md`](docs/INTERFACES.md).

Member 2 owns the PostgreSQL database schema and mapping. Required chunk fields:
`chunk_id`, `company_name`, `ticker`, `form_type`, `filing_date`, `section`, `text`, `source_url`.
Optional field: `period_end` (string or `None`).
All fields are preserved intact in citations and Qdrant vector storage.
