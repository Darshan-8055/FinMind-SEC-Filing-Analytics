"""Persistent SQLite-backed sparse keyword index using BM25 Okapi for FinMind.

Design and Architecture
-----------------------
- Built entirely on Python's standard library `sqlite3` and `math` (zero new dependencies).
- Operates in either persistent mode (disk file, e.g. `data/sparse_index.db`) or in-memory
  mode (`:memory:`) for test isolation.
- Stores complete chunk records (text + metadata) in a `documents` table, enabling fully
  operational sparse retrieval and fallback even when Qdrant is completely offline.
- Indexes term occurrences in a `postings` table with B-tree index on `(term, chunk_id)`.
- Tracks index lifecycle and health via an `index_meta` table (status: 'ready', 'building',
  'stale', 'failed').
- Uses staging tables during full synchronization (`sync_from_store`) to ensure interrupted
  rebuilds never corrupt or expose a partially-indexed state.
- Idempotent: re-ingesting existing `chunk_id`s replaces old postings cleanly without
  duplicate accumulation.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Project root and persistent database defaults
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "sparse_index.db"

# Standard BM25 parameters
DEFAULT_K1: float = 1.5
DEFAULT_B: float = 0.75

REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "company_name",
    "ticker",
    "form_type",
    "filing_date",
    "section",
    "text",
    "source_url",
)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _now_iso() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class SparseKeywordIndex:
    """Persistent SQLite-backed BM25 sparse keyword index."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        cache_size_kb: int = 20000,
    ) -> None:
        self.db_path = str(DEFAULT_DB_PATH if db_path is None else db_path)
        self.k1 = k1
        self.b = b
        self.cache_size_kb = cache_size_kb

        if self.db_path != ":memory:":
            Path(self.db_path).resolve().parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode; explicit transactions via BEGIN/COMMIT
        )
        self.conn.row_factory = sqlite3.Row

        # Performance and resilience pragmas
        self._configure_pragmas()
        self._init_schema()

    def _configure_pragmas(self) -> None:
        """Apply performance pragmas to the SQLite connection."""
        cursor = self.conn.cursor()
        if self.db_path != ":memory:":
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = NORMAL;")
        # Cache size target (negative value = kilobytes in SQLite)
        cursor.execute(f"PRAGMA cache_size = -{abs(self.cache_size_kb)};")
        cursor.execute("PRAGMA foreign_keys = ON;")


    def _init_schema(self) -> None:
        """Initialize schema tables and indexes if not present."""
        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        try:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS index_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    chunk_id TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    form_type TEXT NOT NULL,
                    filing_date TEXT NOT NULL,
                    period_end TEXT,
                    section TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    filename TEXT,
                    chunk_index INTEGER,
                    doc_len INTEGER NOT NULL
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS postings (
                    term TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    term_frequency INTEGER NOT NULL,
                    PRIMARY KEY (term, chunk_id),
                    FOREIGN KEY (chunk_id) REFERENCES documents(chunk_id) ON DELETE CASCADE
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_docs_ticker ON documents(ticker);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_docs_form_type ON documents(form_type);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_docs_filing_date ON documents(filing_date);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_docs_section ON documents(section);"
            )

            # Initialize meta defaults if missing
            meta_defaults = {
                "schema_version": "1.0",
                "status": "uninitialized",
                "chunk_count": "0",
                "avg_doc_len": "0.0",
                "last_sync_timestamp": _now_iso(),
            }
            for k, v in meta_defaults.items():
                cursor.execute(
                    "INSERT OR IGNORE INTO index_meta (key, value) VALUES (?, ?);",
                    (k, v),
                )
            cursor.execute("COMMIT;")
        except Exception:
            cursor.execute("ROLLBACK;")
            raise

    def close(self) -> None:
        """Close database connection."""
        try:
            self.conn.close()
        except Exception:
            pass

    def get_meta(self, key: str) -> str | None:
        """Get a metadata value by key."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM index_meta WHERE key = ?;", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata value by key."""
        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?);",
                (key, str(value)),
            )
            cursor.execute("COMMIT;")
        except Exception:
            cursor.execute("ROLLBACK;")
            raise

    def is_healthy(self) -> bool:
        """Return True only if index status is explicitly 'ready'."""
        return self.get_meta("status") == "ready"

    def is_usable(self) -> bool:
        """Return True if index has verified data that can be queried.

        Permits querying when status is 'ready', or 'stale' (e.g. during a Qdrant outage
        where synchronization cannot currently be confirmed, but existing data is intact).
        Rejects 'uninitialized', 'building', or 'failed' states.
        """
        status = self.get_meta("status")
        chunk_count = int(self.get_meta("chunk_count") or 0)
        return status in ("ready", "stale") and chunk_count > 0

    def check_reconciliation_needed(self, store: Any) -> tuple[bool, str]:
        """Determine whether reconciliation with Qdrant is recommended.

        Returns (needed: bool, reason: str).
        Note: Point count equality is only a preliminary sanity check and does not
        prove content equality. Full reconciliation is performed explicitly via
        sync_from_store().
        """
        status = self.get_meta("status")
        if status == "uninitialized":
            return True, "Index is uninitialized; initial synchronization required"
        if status == "building":
            return True, "Index build was previously interrupted; rebuild required"
        if status == "stale":
            return True, "Index is marked stale; reconciliation required"
        if status == "failed":
            return True, "Index previously encountered an error; rebuild required"
        if status != "ready":
            return True, f"Index status is '{status}'; reconciliation required"

        # Preliminary count check against Qdrant
        try:
            if not store.client.collection_exists(store.collection_name):
                sqlite_count = int(self.get_meta("chunk_count") or 0)
                if sqlite_count > 0:
                    return True, (
                        "Qdrant collection does not exist but SQLite has "
                        f"{sqlite_count} indexed chunks; data may be inconsistent"
                    )
                return False, "Both Qdrant collection and SQLite index are empty; no synchronization needed"
            qdrant_info = store.client.get_collection(store.collection_name)
            qdrant_count = getattr(qdrant_info, "points_count", None)
            sqlite_count = int(self.get_meta("chunk_count") or 0)
            if qdrant_count is not None and qdrant_count != sqlite_count:
                return True, (
                    f"Count mismatch: Qdrant has {qdrant_count} points, "
                    f"SQLite has {sqlite_count} chunks"
                )
        except Exception as exc:
            # Qdrant connection failure does NOT imply SQLite index is corrupt
            logger.debug("Qdrant unavailable during reconciliation check: %s", exc)
            return False, "Qdrant connection unavailable; SQLite index status remains unchanged"

        return False, "Index is ready and preliminary point count matches (full verification requires sync_from_store)"


    def get_stats(self) -> dict[str, Any]:
        """Return dictionary of index statistics and status."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM index_meta;")
        meta = {row["key"]: row["value"] for row in cursor.fetchall()}
        return {
            "status": meta.get("status", "unknown"),
            "chunk_count": int(meta.get("chunk_count", 0)),
            "avg_doc_len": float(meta.get("avg_doc_len", 0.0)),
            "last_sync_timestamp": meta.get("last_sync_timestamp"),
            "db_path": self.db_path,
        }

    def _recompute_corpus_stats(self, cursor: sqlite3.Cursor) -> tuple[int, float]:
        """Recalculate total document count and average document length."""
        cursor.execute("SELECT COUNT(*), COALESCE(SUM(doc_len), 0) FROM documents;")
        row = cursor.fetchone()
        total_docs = row[0]
        total_tokens = row[1]
        avg_len = (total_tokens / total_docs) if total_docs > 0 else 0.0

        cursor.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('chunk_count', ?);",
            (str(total_docs),),
        )
        cursor.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('avg_doc_len', ?);",
            (str(avg_len),),
        )
        cursor.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('last_sync_timestamp', ?);",
            (_now_iso(),),
        )
        return total_docs, avg_len

    def add_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> int:
        """Index a batch of chunk records into SQLite with full payload storage.

        Replaces any existing entries for matching chunk IDs to prevent duplicate
        postings upon re-ingestion.
        """
        if not chunks:
            return 0

        # Validate incoming chunks before acquiring write lock
        prepared_docs: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                raise TypeError(f"Chunk {index} must be a mapping")
            for field in REQUIRED_CHUNK_FIELDS:
                val = chunk.get(field)
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(f"Chunk {index} missing required field: {field}")

            tokens = _tokenize(chunk["text"])
            doc_len = len(tokens)
            term_counts = Counter(tokens)

            period_end = chunk.get("period_end")
            if period_end is not None and not isinstance(period_end, str):
                raise TypeError("period_end must be a string or None")

            prepared_docs.append(
                {
                    "chunk_id": str(chunk["chunk_id"]).strip(),
                    "company_name": str(chunk["company_name"]).strip(),
                    "ticker": str(chunk["ticker"]).strip(),
                    "form_type": str(chunk["form_type"]).strip(),
                    "filing_date": str(chunk["filing_date"]).strip(),
                    "period_end": period_end.strip() if period_end else None,
                    "section": str(chunk["section"]).strip(),
                    "text": chunk["text"],
                    "source_url": str(chunk["source_url"]).strip(),
                    "filename": str(chunk.get("filename") or "") if "filename" in chunk else None,
                    "chunk_index": int(chunk["chunk_index"]) if "chunk_index" in chunk and chunk["chunk_index"] is not None else None,
                    "doc_len": max(1, doc_len),
                    "term_counts": term_counts,
                }
            )

        # Read existing status before entering the write transaction.
        # An incremental batch must NOT promote a 'stale', 'building', or 'failed'
        # index to 'ready' — only a successful full sync_from_store() may do that.
        # The only valid first-ingestion promotion is 'uninitialized' → 'ready'.
        existing_status = self.get_meta("status")

        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        try:
            for doc in prepared_docs:
                cid = doc["chunk_id"]
                # 1. Clean existing postings for idempotency
                cursor.execute("DELETE FROM postings WHERE chunk_id = ?;", (cid,))

                # 2. Insert or replace document record
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO documents (
                        chunk_id, company_name, ticker, form_type, filing_date,
                        period_end, section, text, source_url, filename, chunk_index, doc_len
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        cid,
                        doc["company_name"],
                        doc["ticker"],
                        doc["form_type"],
                        doc["filing_date"],
                        doc["period_end"],
                        doc["section"],
                        doc["text"],
                        doc["source_url"],
                        doc["filename"],
                        doc["chunk_index"],
                        doc["doc_len"],
                    ),
                )

                # 3. Insert posting tuples
                if doc["term_counts"]:
                    cursor.executemany(
                        """
                        INSERT INTO postings (term, chunk_id, term_frequency)
                        VALUES (?, ?, ?);
                        """,
                        [
                            (term, cid, freq)
                            for term, freq in doc["term_counts"].items()
                        ],
                    )

            # Recompute summary statistics and set status.
            # Only promote 'uninitialized' (first ingestion) or preserve 'ready'.
            # 'stale', 'building', and 'failed' are only cleared by sync_from_store().
            self._recompute_corpus_stats(cursor)
            new_status = "ready" if existing_status in ("uninitialized", "ready") else existing_status
            cursor.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('status', ?);",
                (new_status,),
            )
            cursor.execute("COMMIT;")
            return len(prepared_docs)
        except Exception as exc:
            cursor.execute("ROLLBACK;")
            logger.error("Failed to add chunks to sparse index: %s", exc)
            raise

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Retrieve full chunk records directly from SQLite by chunk_id."""
        if not chunk_ids:
            return []

        cursor = self.conn.cursor()
        results: list[dict[str, Any]] = []

        # Batch lookup in chunks of 500
        batch_size = 500
        for i in range(0, len(chunk_ids), batch_size):
            batch = list(chunk_ids[i : i + batch_size])
            placeholders = ",".join("?" for _ in batch)
            cursor.execute(
                f"""
                SELECT chunk_id, company_name, ticker, form_type, filing_date,
                       period_end, section, text, source_url, filename, chunk_index
                FROM documents
                WHERE chunk_id IN ({placeholders});
                """,
                batch,
            )
            for row in cursor.fetchall():
                item = dict(row)
                if item.get("filename") is None:
                    item.pop("filename", None)
                if item.get("chunk_index") is None:
                    item.pop("chunk_index", None)
                results.append(item)

        return results

    def search(
        self,
        query: str,
        *,
        top_k: int = 30,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Perform BM25 sparse search and return top ranked (chunk_id, bm25_score, chunk_dict).

        Returns:
            list of tuples: (chunk_id, bm25_score, full_chunk_payload)
            Ordered by bm25_score descending, tie-broken by chunk_id ascending.
        """
        if not isinstance(query, str) or not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        stats = self.get_stats()
        total_docs = stats["chunk_count"]
        avg_doc_len = stats["avg_doc_len"]

        if total_docs == 0 or avg_doc_len <= 0.0:
            return []

        cursor = self.conn.cursor()
        unique_terms = sorted(set(query_tokens))

        # 1. Fetch document frequencies (DF_t) for query terms
        df_map: dict[str, int] = {}
        term_placeholders = ",".join("?" for _ in unique_terms)
        cursor.execute(
            f"""
            SELECT term, COUNT(chunk_id) as df
            FROM postings
            WHERE term IN ({term_placeholders})
            GROUP BY term;
            """,
            unique_terms,
        )
        for row in cursor.fetchall():
            df_map[row["term"]] = int(row["df"])

        # Terms not present in postings have DF=0
        for t in unique_terms:
            df_map.setdefault(t, 0)

        # 2. Compute IDF using standard Lucene/Okapi positive floor formula
        idf_map: dict[str, float] = {}
        for t, df in df_map.items():
            if df == 0:
                idf_map[t] = 0.0
            else:
                idf_map[t] = math.log(((total_docs - df + 0.5) / (df + 0.5)) + 1.0)

        # 3. Construct candidate query with metadata filtering
        filter_clauses: list[str] = []
        filter_params: list[Any] = []

        if filters:
            allowed_filters = ("ticker", "form_type", "filing_date", "section", "company_name")
            for field, val in filters.items():
                if field in allowed_filters:
                    filter_clauses.append(f"d.{field} = ?")
                    filter_params.append(str(val))
                else:
                    logger.warning("Ignoring unsupported filter field: %s", field)

        where_sql = f"p.term IN ({term_placeholders})"
        if filter_clauses:
            where_sql += " AND " + " AND ".join(filter_clauses)

        query_params = list(unique_terms) + filter_params

        # Retrieve matching postings joined with document metadata
        sql = f"""
            SELECT p.chunk_id, p.term, p.term_frequency, d.doc_len,
                   d.company_name, d.ticker, d.form_type, d.filing_date,
                   d.period_end, d.section, d.text, d.source_url,
                   d.filename, d.chunk_index
            FROM postings p
            JOIN documents d ON p.chunk_id = d.chunk_id
            WHERE {where_sql};
        """

        cursor.execute(sql, query_params)
        rows = cursor.fetchall()

        if not rows:
            return []

        # 4. Compute BM25 scores in memory for retrieved candidates only
        scores: dict[str, float] = {}
        doc_records: dict[str, dict[str, Any]] = {}

        for row in rows:
            cid = row["chunk_id"]
            term = row["term"]
            tf = row["term_frequency"]
            doc_len = row["doc_len"]

            if cid not in doc_records:
                record = {
                    "chunk_id": cid,
                    "company_name": row["company_name"],
                    "ticker": row["ticker"],
                    "form_type": row["form_type"],
                    "filing_date": row["filing_date"],
                    "period_end": row["period_end"],
                    "section": row["section"],
                    "text": row["text"],
                    "source_url": row["source_url"],
                }
                if row["filename"] is not None:
                    record["filename"] = row["filename"]
                if row["chunk_index"] is not None:
                    record["chunk_index"] = row["chunk_index"]
                doc_records[cid] = record
                scores[cid] = 0.0

            idf = idf_map.get(term, 0.0)
            if idf <= 0.0:
                continue

            num = tf * (self.k1 + 1.0)
            denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len))
            scores[cid] += idf * (num / denom)

        # Filter out zero or negative scores
        valid_cids = [cid for cid, sc in scores.items() if sc > 0.0]
        # Sort descending by score, tie-break by chunk_id ascending
        valid_cids.sort(key=lambda cid: (-scores[cid], cid))

        results: list[tuple[str, float, dict[str, Any]]] = []
        for cid in valid_cids[:top_k]:
            results.append((cid, scores[cid], doc_records[cid]))

        return results

    def sync_from_store(self, store: Any, batch_size: int = 500) -> int:
        """Fully synchronize and rebuild index directly from Qdrant payloads.

        Uses staging tables to ensure an interrupted sync never corrupts or
        exposes an incomplete index.
        """
        cursor = self.conn.cursor()
        # Mark building state in meta
        self.set_meta("status", "building")

        try:
            # 1. Create staging tables
            cursor.execute("BEGIN IMMEDIATE;")
            cursor.execute("DROP TABLE IF EXISTS documents_staging;")
            cursor.execute("DROP TABLE IF EXISTS postings_staging;")
            cursor.execute(
                """
                CREATE TABLE documents_staging (
                    chunk_id TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    form_type TEXT NOT NULL,
                    filing_date TEXT NOT NULL,
                    period_end TEXT,
                    section TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    filename TEXT,
                    chunk_index INTEGER,
                    doc_len INTEGER NOT NULL
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE postings_staging (
                    term TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    term_frequency INTEGER NOT NULL,
                    PRIMARY KEY (term, chunk_id)
                );
                """
            )
            cursor.execute("COMMIT;")

            # 2. Scroll all points from Qdrant in batches
            offset: Any = None
            total_rebuilt = 0

            while True:
                response = store.client.scroll(
                    collection_name=store.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = response
                if not points:
                    break

                cursor.execute("BEGIN IMMEDIATE;")
                for pt in points:
                    payload = pt.payload or {}

                    # Strict validation: all 8 required chunk fields must be non-empty strings.
                    # An invalid payload aborts the entire sync, preserving the production tables.
                    for field in REQUIRED_CHUNK_FIELDS:
                        val = payload.get(field)
                        if not isinstance(val, str) or not val.strip():
                            raise ValueError(
                                f"Invalid Qdrant payload: field '{field}' is missing or empty "
                                f"(point id={pt.id}). Synchronization aborted to preserve "
                                f"index integrity."
                            )

                    cid = payload["chunk_id"]
                    text = payload["text"]
                    tokens = _tokenize(text)
                    doc_len = max(1, len(tokens))
                    term_counts = Counter(tokens)

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO documents_staging (
                            chunk_id, company_name, ticker, form_type, filing_date,
                            period_end, section, text, source_url, filename, chunk_index, doc_len
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            cid,
                            payload["company_name"],
                            payload["ticker"],
                            payload["form_type"],
                            payload["filing_date"],
                            payload.get("period_end"),
                            payload["section"],
                            text,
                            payload["source_url"],
                            payload.get("filename"),
                            payload.get("chunk_index"),
                            doc_len,
                        ),
                    )

                    if term_counts:
                        cursor.executemany(
                            """
                            INSERT OR REPLACE INTO postings_staging (term, chunk_id, term_frequency)
                            VALUES (?, ?, ?);
                            """,
                            [(t, cid, f) for t, f in term_counts.items()],
                        )
                    total_rebuilt += 1

                cursor.execute("COMMIT;")
                offset = next_offset
                if offset is None:
                    break

            # 3. Swap staging tables atomically into production
            cursor.execute("BEGIN IMMEDIATE;")
            cursor.execute("DELETE FROM postings;")
            cursor.execute("DELETE FROM documents;")
            cursor.execute("INSERT INTO documents SELECT * FROM documents_staging;")
            cursor.execute("INSERT INTO postings SELECT * FROM postings_staging;")
            cursor.execute("DROP TABLE documents_staging;")
            cursor.execute("DROP TABLE postings_staging;")

            self._recompute_corpus_stats(cursor)
            cursor.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('status', 'ready');"
            )
            cursor.execute("COMMIT;")
            logger.info("Successfully synchronized %d chunks from Qdrant", total_rebuilt)
            return total_rebuilt

        except Exception as exc:
            if self.conn.in_transaction:
                try:
                    cursor.execute("ROLLBACK;")
                except Exception:
                    pass
            logger.error("Interrupted or failed synchronization from Qdrant: %s", exc)
            try:
                self.set_meta("status", "stale")
            except Exception:
                pass
            try:
                cursor.execute("DROP TABLE IF EXISTS documents_staging;")
                cursor.execute("DROP TABLE IF EXISTS postings_staging;")
            except Exception:
                pass
            raise
