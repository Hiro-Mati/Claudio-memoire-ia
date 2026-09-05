# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Local lexical index (SQLite FTS5, BM25) fused with dense retrieval.

Dense embeddings from small local models miss exact tokens: function names,
identifiers, error codes, product names. The local vector collection has no
keyword search, and sparse/hybrid embeddings are only available from a few
hosted providers. This module keeps a lightweight BM25 index next to the
vector index so ``find``/``search`` can boost or recover exact-match hits.

Design constraints:

* The vector index stays the source of truth. Lexical hits are validated
  against it at query time; orphans are purged, so write-path mirroring may
  stay best-effort.
* One SQLite file per workspace (``{storage.workspace}/_system/lexical/lexical.db``),
  or in-memory when no workspace is configured (tests).
* Identifiers are indexed twice: verbatim (``tokenchars '_'`` keeps
  ``snake_case`` whole) and split (``snake case``, ``camel Case``) so both
  ``my_function`` and ``function`` match.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

LEXICAL_INDEX_PATH_ENV = "OPENVIKING_LEXICAL_INDEX_PATH"
MAX_BODY_CHARS = 60_000
_TOKEN_RE = re.compile(r"[0-9A-Za-z_À-ɏЀ-ӿ一-鿿]+")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_SEP_RE = re.compile(r"[_\-./\\:]+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    doc_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    uri TEXT NOT NULL,
    parent_uri TEXT NOT NULL DEFAULT '',
    context_type TEXT NOT NULL DEFAULT 'resource',
    level INTEGER NOT NULL DEFAULT 2
);
CREATE INDEX IF NOT EXISTS entries_account_uri ON entries(account_id, uri);
CREATE INDEX IF NOT EXISTS entries_account_parent ON entries(account_id, parent_uri);
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    body, loose,
    tokenize = "unicode61 remove_diacritics 2 tokenchars '_'"
);
"""


@dataclass(frozen=True)
class LexicalHit:
    doc_id: str
    uri: str
    parent_uri: str
    context_type: str
    level: int
    score: float
    """BM25 relevance normalized to [0, 1] within one result set."""


def split_identifiers(text: str) -> str:
    """Return ``text`` with snake_case, kebab-case, dotted and camelCase split."""
    if not text:
        return ""
    loose = _SEP_RE.sub(" ", text)
    loose = _CAMEL_RE.sub(r"\1 \2", loose)
    return loose


def query_terms(query: str) -> List[str]:
    """Tokens worth matching: alphanumerics and identifiers, lowercased, deduplicated."""
    seen: Dict[str, None] = {}
    for token in _TOKEN_RE.findall(query or ""):
        lowered = token.lower()
        if len(lowered) < 2 and not lowered.isdigit():
            continue
        seen.setdefault(lowered, None)
        for part in split_identifiers(token).split():
            part = part.lower()
            if len(part) >= 3:
                seen.setdefault(part, None)
    return list(seen)


def build_match_expression(query: str) -> Optional[str]:
    """FTS5 MATCH expression: any term may match, BM25 ranks denser matches higher."""
    terms = query_terms(query)
    if not terms:
        return None
    quoted = ['"' + term.replace('"', '""') + '"' for term in terms[:32]]
    return "{body loose} : (" + " OR ".join(quoted) + ")"


def _record_body(record: Mapping[str, Any]) -> str:
    parts = []
    for key in ("name", "abstract", "content"):
        value = record.get(key)
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="ignore")
            except Exception:
                value = ""
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    uri = record.get("uri")
    if isinstance(uri, str):
        parts.append(uri.rsplit("/", 1)[-1])
    body = "\n".join(parts)
    return body[:MAX_BODY_CHARS]


class LexicalIndex:
    """Thread-safe BM25 index over context records."""

    def __init__(self, db_path: str | os.PathLike[str] = ":memory:"):
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL") if self._db_path != ":memory:" else None
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def db_path(self) -> str:
        return self._db_path

    # ---- writes ---------------------------------------------------------

    def upsert_many(self, records: Iterable[Mapping[str, Any]]) -> int:
        rows = []
        for record in records:
            doc_id = record.get("id")
            uri = record.get("uri")
            if not doc_id or not uri:
                continue
            body = _record_body(record)
            rows.append(
                (
                    str(doc_id),
                    str(record.get("account_id") or "default"),
                    str(uri).rstrip("/"),
                    str(record.get("parent_uri") or "").rstrip("/"),
                    str(record.get("context_type") or "resource"),
                    int(record.get("level", 2) or 0),
                    body,
                    split_identifiers(body),
                )
            )
        if not rows:
            return 0
        with self._lock:
            cur = self._conn.cursor()
            for doc_id, account_id, uri, parent_uri, context_type, level, body, loose in rows:
                existing = cur.execute(
                    "SELECT rowid FROM entries WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if existing:
                    rowid = existing[0]
                    cur.execute("DELETE FROM entries_fts WHERE rowid = ?", (rowid,))
                    cur.execute(
                        "UPDATE entries SET account_id=?, uri=?, parent_uri=?, context_type=?, "
                        "level=? WHERE rowid=?",
                        (account_id, uri, parent_uri, context_type, level, rowid),
                    )
                else:
                    cur.execute(
                        "INSERT INTO entries(doc_id, account_id, uri, parent_uri, context_type, "
                        "level) VALUES (?,?,?,?,?,?)",
                        (doc_id, account_id, uri, parent_uri, context_type, level),
                    )
                    rowid = cur.lastrowid
                cur.execute(
                    "INSERT INTO entries_fts(rowid, body, loose) VALUES (?,?,?)",
                    (rowid, body, loose),
                )
            self._conn.commit()
        return len(rows)

    def upsert(self, record: Mapping[str, Any]) -> int:
        return self.upsert_many([record])

    def _delete_rowids(self, cur: sqlite3.Cursor, rowids: Sequence[int]) -> int:
        for rowid in rowids:
            cur.execute("DELETE FROM entries_fts WHERE rowid = ?", (rowid,))
            cur.execute("DELETE FROM entries WHERE rowid = ?", (rowid,))
        return len(rowids)

    def delete_ids(self, doc_ids: Iterable[str]) -> int:
        ids = [str(i) for i in doc_ids if i]
        if not ids:
            return 0
        with self._lock:
            cur = self._conn.cursor()
            rowids = [
                row[0]
                for chunk in (ids[i : i + 500] for i in range(0, len(ids), 500))
                for row in cur.execute(
                    f"SELECT rowid FROM entries WHERE doc_id IN ({','.join('?' * len(chunk))})",
                    chunk,
                ).fetchall()
            ]
            deleted = self._delete_rowids(cur, rowids)
            self._conn.commit()
        return deleted

    def delete_uris(self, account_id: str, uris: Iterable[str], *, recursive: bool = True) -> int:
        with self._lock:
            cur = self._conn.cursor()
            rowids: List[int] = []
            for uri in uris:
                uri = str(uri).rstrip("/")
                if not uri:
                    continue
                rows = cur.execute(
                    "SELECT rowid FROM entries WHERE account_id = ? AND (uri = ? OR uri LIKE ?)",
                    (account_id, uri, uri + "/%" if recursive else uri),
                ).fetchall()
                rowids.extend(row[0] for row in rows)
            deleted = self._delete_rowids(cur, rowids)
            self._conn.commit()
        return deleted

    def move_uri(self, account_id: str, source_uri: str, target_uri: str) -> int:
        source_uri = source_uri.rstrip("/")
        target_uri = target_uri.rstrip("/")
        if not source_uri or not target_uri:
            return 0
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE entries SET uri = ? || substr(uri, ?) WHERE account_id = ? "
                "AND (uri = ? OR uri LIKE ?)",
                (target_uri, len(source_uri) + 1, account_id, source_uri, source_uri + "/%"),
            )
            moved = cur.rowcount
            cur.execute(
                "UPDATE entries SET parent_uri = ? || substr(parent_uri, ?) WHERE account_id = ? "
                "AND (parent_uri = ? OR parent_uri LIKE ?)",
                (target_uri, len(source_uri) + 1, account_id, source_uri, source_uri + "/%"),
            )
            self._conn.commit()
        return moved

    def delete_account(self, account_id: str) -> int:
        with self._lock:
            cur = self._conn.cursor()
            rowids = [
                row[0]
                for row in cur.execute(
                    "SELECT rowid FROM entries WHERE account_id = ?", (account_id,)
                ).fetchall()
            ]
            deleted = self._delete_rowids(cur, rowids)
            self._conn.commit()
        return deleted

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM entries_fts")
            self._conn.execute("DELETE FROM entries")
            self._conn.commit()

    def count(self, account_id: Optional[str] = None) -> int:
        with self._lock:
            if account_id is None:
                row = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM entries WHERE account_id = ?", (account_id,)
                ).fetchone()
        return int(row[0]) if row else 0

    # ---- reads ----------------------------------------------------------

    def search(
        self,
        query: str,
        account_id: str,
        *,
        parent_uri: Optional[str] = None,
        target_directories: Optional[Sequence[str]] = None,
        context_type: Optional[str] = None,
        level: Optional[Sequence[int]] = None,
        visible_user_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[LexicalHit]:
        """BM25 search scoped like the vector search.

        ``visible_user_id`` restricts ``viking://user/...`` hits to that user's
        own tree (mirrors the tenant filter for non-root callers).
        """
        match = build_match_expression(query)
        if match is None or limit <= 0:
            return []
        where = ["e.account_id = ?"]
        params: List[Any] = [account_id]
        if parent_uri is not None:
            where.append("e.parent_uri = ?")
            params.append(parent_uri.rstrip("/"))
        dirs = [d.rstrip("/") for d in (target_directories or []) if d]
        if dirs:
            clauses = []
            for d in dirs:
                clauses.append("(e.uri = ? OR e.uri LIKE ?)")
                params.extend([d, d + "/%"])
            where.append("(" + " OR ".join(clauses) + ")")
        if context_type:
            where.append("e.context_type = ?")
            params.append(context_type)
        if level:
            where.append(f"e.level IN ({','.join('?' * len(level))})")
            params.extend(int(x) for x in level)
        if visible_user_id is not None:
            where.append("(e.uri NOT LIKE 'viking://user/%' OR e.uri LIKE ?)")
            params.append(f"viking://user/{visible_user_id}/%")
        sql = (
            "SELECT e.doc_id, e.uri, e.parent_uri, e.context_type, e.level, "
            "bm25(entries_fts, 1.0, 0.6) AS rank "
            "FROM entries_fts JOIN entries e ON e.rowid = entries_fts.rowid "
            f"WHERE entries_fts MATCH ? AND {' AND '.join(where)} "
            "ORDER BY rank LIMIT ?"
        )
        with self._lock:
            try:
                rows = self._conn.execute(sql, [match, *params, int(limit)]).fetchall()
            except sqlite3.OperationalError as exc:
                logger.debug("lexical search failed for %r: %s", query, exc)
                return []
        if not rows:
            return []
        # bm25() returns negative values, more negative = better match.
        raw = [-float(row[5]) for row in rows]
        best, worst = max(raw), min(raw)
        span = best - worst
        hits = []
        for row, value in zip(rows, raw, strict=True):
            score = 1.0 if span <= 1e-9 else 0.5 + 0.5 * (value - worst) / span
            hits.append(
                LexicalHit(
                    doc_id=str(row[0]),
                    uri=str(row[1]),
                    parent_uri=str(row[2]),
                    context_type=str(row[3]),
                    level=int(row[4]),
                    score=score,
                )
            )
        return hits

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---- process-wide instance ---------------------------------------------------

_INDEX: Optional[LexicalIndex] = None
_INDEX_LOCK = threading.Lock()


def resolve_lexical_index_path() -> str:
    """Workspace-relative database path, env override, or in-memory fallback."""
    override = os.environ.get(LEXICAL_INDEX_PATH_ENV)
    if override:
        return override
    try:
        from openviking_cli.utils.config import get_openviking_config

        workspace = get_openviking_config().storage.workspace
        if workspace:
            return str(
                Path(workspace).expanduser().resolve() / "_system" / "lexical" / "lexical.db"
            )
    except Exception as exc:  # pragma: no cover - config not initialized
        logger.debug("lexical index falls back to memory: %s", exc)
    return ":memory:"


def get_lexical_index() -> LexicalIndex:
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = LexicalIndex(resolve_lexical_index_path())
            logger.info("Lexical index opened at %s", _INDEX.db_path)
        return _INDEX


def set_lexical_index(index: Optional[LexicalIndex]) -> None:
    """Replace the process-wide index (tests)."""
    global _INDEX
    with _INDEX_LOCK:
        _INDEX = index


def lexical_index_enabled() -> bool:
    try:
        from openviking_cli.utils.config import get_openviking_config

        retrieval = get_openviking_config().retrieval
        return bool(getattr(retrieval, "lexical_index_enabled", False))
    except Exception:
        return False
