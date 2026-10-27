# FinMind — SEC Filing Analytics

Agentic Financial Intelligence and SEC Filing Analytics Platform.

## Member 1: Retrieval Architect

Responsibilities:

- Narrative parsing and token-aware chunking of SEC 10-K / 10-Q filings
- Shared chunk schema and ingestion pipeline
- Qdrant vector storage
- Keyword (BM25) retrieval
- Semantic (dense) retrieval using sentence-transformers
- Hybrid retrieval using weighted reciprocal-rank fusion
- Retrieval evaluation — Precision@K, Recall@K, macro-average comparison

---

## Project structure

```
src/
  ingestion/
    narrative_parser.py   # SEC filing text → section-aware chunks
    chunk_adapter.py      # Parser output → shared chunk interface schema
    pipeline.py           # End-to-end ingestion: parse → embed → store
  retrieval/
    keyword_search.py     # BM25-based keyword retrieval
    semantic_search.py    # Cosine-similarity semantic retrieval
    hybrid_search.py      # Weighted RRF hybrid (dense + sparse)
    qdrant_store.py       # Qdrant vector store wrapper
    evaluation.py         # Precision@K, Recall@K, macro-average metrics

tests/
  fixtures/
    retrieval_eval.json   # Synthetic evaluation queries and relevance labels
  test_*.py               # Automated unit and integration tests

docs/
  INTERFACES.md           # Shared chunk schema and database mapping contract

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

Run the full test suite (all retrieval, ingestion, and integration tests):

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
```

Run only the retrieval evaluation tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_retrieval_evaluation.py -v
```

The tests use a `FakeEmbeddingModel` (a deterministic 12-word bag-of-words
encoder) so they do **not** require downloading a sentence-transformer model or
connecting to Qdrant.  Integration tests that connect to Qdrant require the
local Docker service to be running.

---

## Minimal usage example

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

---

## Retrieval evaluation

### Running the evaluation script

```powershell
.venv\Scripts\python.exe run_retrieval_evaluation.py
```

This evaluates keyword, semantic, and hybrid retrieval against every query in
`tests/fixtures/retrieval_eval.json` and prints per-query and macro-average
Precision@K and Recall@K.

> **Important:** The evaluation script downloads the `all-MiniLM-L6-v2` model
> on first run.  Results come from a **synthetic 4-chunk fixture** and should
> not be reported as production retrieval quality figures.

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

| Metric | Definition | Convention |
|---|---|---|
| **Precision@K** | Fraction of top-K results that are relevant | 0.0 when result list is empty |
| **Recall@K** | Fraction of all relevant docs found in top-K | 0.0 when relevant set is empty |
| **Macro-average** | Mean of per-query metric, equal weight per query | 0.0 when query list is empty |

Hybrid retrieval combines semantic (dense) and BM25 (sparse) rankings using
weighted reciprocal-rank fusion (default: `dense_weight=0.7`,
`sparse_weight=0.3`).  These defaults were not tuned on real data; validate
before changing them.

---

## Environment requirements

| Requirement | Notes |
|---|---|
| Local Qdrant service | Required for ingestion and Qdrant integration tests. Default: `localhost:6333`. |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` — downloaded automatically (384-dim vectors). |
| Qdrant collection vector size | Must match the model dimension (384 for `all-MiniLM-L6-v2`). |

---

## Shared interface contract

The chunk schema shared between Member 1 (retrieval) and other team members is
documented in [`docs/INTERFACES.md`](docs/INTERFACES.md).

Required chunk fields: `chunk_id`, `company_name`, `ticker`, `form_type`,
`filing_date`, `section`, `text`, `source_url`.

Optional field: `period_end` (string or `None`).

---

## Real-data validation

Integration tests that run against a live Qdrant instance are in
`tests/test_qdrant_store.py`, `tests/test_ingestion_retrieval_integration.py`,
and `tests/test_semantic_ingestion_integration.py`.

**Outstanding:** End-to-end validation against real SEC filing documents has not
been performed.  The evaluation fixture uses four synthetic chunks.  To validate
with real data:

1. Obtain SEC 10-K filings (e.g., from SEC EDGAR).
2. Run `FilingIngestionPipeline.ingest_filing()` to parse and store them.
3. Build a relevance judgement file in the same format as
   `tests/fixtures/retrieval_eval.json`.
4. Run `evaluate_all_queries()` and inspect the macro-average results.
