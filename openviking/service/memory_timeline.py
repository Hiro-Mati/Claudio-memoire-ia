# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Memory provenance, revert and point-in-time reads.

Every session commit already writes ``memory_diff.json`` next to its archive
(``viking://user/{uid}/sessions/{sid}/history/archive_NNN/``) with the memory
files it added, updated (before/after) or deleted. Snapshots (gitoxide inside
RAGFS) can record the memory tree at each commit. This service turns those two
existing records into three user-facing operations:

* ``provenance(uri)``: which session archives created, changed or deleted a
  memory, with the content before and after each change;
* ``revert(uri, archive_uri)``: undo one recorded change through the regular
  write/delete paths so vectors and sidecars stay consistent;
* ``as_of(uri, at)``: the memory file as it was at a given instant, read from
  the latest snapshot committed before that instant.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openviking.server.identity import RequestContext
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

MEMORY_DIFF_FILENAME = "memory_diff.json"
MEMORY_REVERT_FILENAME = "memory_reverts.json"


def _norm(uri: str) -> str:
    return (uri or "").strip().rstrip("/")


def expand_home(uri: str, ctx: RequestContext) -> str:
    """Expand ``viking://~/...`` to the caller's canonical user root."""
    uri = _norm(uri)
    if uri == "viking://~" or uri.startswith("viking://~/"):
        return f"viking://user/{ctx.user.user_id}" + uri[len("viking://~") :]
    return uri


def parse_instant(value: str | datetime) -> datetime:
    """Parse an ISO-8601 instant (naive values are treated as UTC)."""
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip()
        if not text:
            raise InvalidArgumentError("at must be an ISO-8601 timestamp")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise InvalidArgumentError(f"invalid timestamp: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class MemoryTimelineService:
    """Provenance, revert and as-of reads over memory files."""

    def __init__(self, fs: Any):
        self._fs = fs

    # ---- helpers --------------------------------------------------------

    async def _ls(self, uri: str, ctx: RequestContext) -> List[Dict[str, Any]]:
        try:
            entries = await self._fs.ls(uri, ctx)
        except Exception as exc:
            if _is_not_found(exc):
                return []
            raise
        return [e for e in entries if isinstance(e, dict)]

    async def _read_json(self, uri: str, ctx: RequestContext) -> Optional[Dict[str, Any]]:
        try:
            raw = await self._fs.read(uri, ctx)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    async def _iter_archives(self, ctx: RequestContext, *, max_sessions: int) -> List[str]:
        """Archive directory URIs of the caller's sessions, newest session first."""
        sessions_root = f"viking://user/{ctx.user.user_id}/sessions"
        sessions = [e for e in await self._ls(sessions_root, ctx) if e.get("isDir")]
        sessions.sort(key=lambda e: str(e.get("modTime") or ""), reverse=True)
        archives: List[str] = []
        for session in sessions[:max_sessions]:
            history_uri = f"{_norm(session['uri'])}/history"
            for entry in await self._ls(history_uri, ctx):
                if entry.get("isDir") and "archive" in str(entry.get("uri", "")):
                    archives.append(_norm(entry["uri"]))
        return archives

    @staticmethod
    def _ops_for_uri(diff: Dict[str, Any], uri: str) -> List[Dict[str, Any]]:
        operations = diff.get("operations") or {}
        found: List[Dict[str, Any]] = []
        for kind, key in (("add", "adds"), ("update", "updates"), ("delete", "deletes")):
            for item in operations.get(key) or []:
                if isinstance(item, dict) and _norm(str(item.get("uri", ""))) == uri:
                    found.append({"op": kind, **item})
        return found

    # ---- public API -----------------------------------------------------

    async def provenance(
        self,
        uri: str,
        ctx: RequestContext,
        *,
        limit: int = 50,
        max_sessions: int = 200,
    ) -> List[Dict[str, Any]]:
        """Recorded changes to ``uri``, newest first."""
        uri = expand_home(uri, ctx)
        events: List[Dict[str, Any]] = []
        for archive_uri in await self._iter_archives(ctx, max_sessions=max_sessions):
            diff = await self._read_json(f"{archive_uri}/{MEMORY_DIFF_FILENAME}", ctx)
            if not diff:
                continue
            for op in self._ops_for_uri(diff, uri):
                events.append(
                    {
                        "uri": uri,
                        "op": op["op"],
                        "memory_type": op.get("memory_type", "unknown"),
                        "archive_uri": _norm(str(diff.get("archive_uri") or archive_uri)),
                        "session_id": _session_id_from_archive(archive_uri),
                        "extracted_at": diff.get("extracted_at"),
                        "trace_id": diff.get("trace_id"),
                        "before": op.get("before") if op["op"] == "update" else None,
                        "after": op.get("after") if op["op"] in {"add", "update"} else None,
                        "deleted_content": op.get("deleted_content")
                        if op["op"] == "delete"
                        else None,
                    }
                )
            if len(events) >= limit:
                break
        events.sort(key=lambda e: str(e.get("extracted_at") or ""), reverse=True)
        return events[:limit]

    async def revert(self, uri: str, archive_uri: str, ctx: RequestContext) -> Dict[str, Any]:
        """Undo the change recorded for ``uri`` in ``archive_uri``.

        add -> delete the file; update -> restore the ``before`` body;
        delete -> recreate the file from ``deleted_content``. Uses the regular
        FSService write/rm paths so vectors and sidecars are refreshed.
        """
        uri = expand_home(uri, ctx)
        archive_uri = expand_home(archive_uri, ctx)
        diff = await self._read_json(f"{archive_uri}/{MEMORY_DIFF_FILENAME}", ctx)
        if not diff:
            raise NotFoundError(archive_uri, "memory_diff")
        ops = self._ops_for_uri(diff, uri)
        if not ops:
            raise NotFoundError(uri, "memory_change")
        op = ops[-1]
        if op["op"] == "add":
            await self._fs.rm(uri, ctx)
            action = "deleted"
        elif op["op"] == "update":
            before = op.get("before") or ""
            if not before:
                raise InvalidArgumentError(
                    "recorded update has no 'before' body to restore", {"uri": uri}
                )
            await self._fs.write(uri, before, ctx, mode="replace")
            action = "restored_previous"
        else:
            content = op.get("deleted_content") or ""
            if not content:
                raise InvalidArgumentError(
                    "recorded delete has no content to recreate", {"uri": uri}
                )
            await self._fs.write(uri, content, ctx, mode="upsert")
            action = "recreated"
        record = {
            "uri": uri,
            "archive_uri": archive_uri,
            "reverted_op": op["op"],
            "action": action,
            "reverted_at": datetime.now(timezone.utc).isoformat(),
            "by_user": ctx.user.user_id,
        }
        await self._append_revert_record(archive_uri, record, ctx)
        return record

    async def _append_revert_record(
        self, archive_uri: str, record: Dict[str, Any], ctx: RequestContext
    ) -> None:
        """Keep an audit list next to the diff (raw write, no semantic processing)."""
        try:
            viking_fs = self._fs._ensure_initialized()
            audit_uri = f"{archive_uri}/{MEMORY_REVERT_FILENAME}"
            existing = await self._read_json(audit_uri, ctx) or {"reverts": []}
            reverts = list(existing.get("reverts") or [])
            reverts.append(record)
            await viking_fs.write_file(
                uri=audit_uri,
                content=json.dumps({"reverts": reverts}, ensure_ascii=False, indent=2),
                ctx=ctx,
            )
        except Exception as exc:  # pragma: no cover - audit must not break the revert
            logger.warning("memory revert audit not written for %s: %s", archive_uri, exc)

    async def as_of(
        self,
        uri: str,
        at: str | datetime,
        ctx: RequestContext,
        *,
        branch: str = "main",
        max_commits: int = 500,
    ) -> Dict[str, Any]:
        """Content of ``uri`` in the latest snapshot committed at or before ``at``."""
        uri = expand_home(uri, ctx)
        instant = parse_instant(at)
        commits = await self._fs.log(ctx, branch=branch, limit=max_commits, paths=[uri])
        cutoff = instant.timestamp()
        chosen: Optional[Dict[str, Any]] = None
        for commit in commits:  # newest first
            committer = commit.get("committer") or commit.get("author") or {}
            seconds = committer.get("time_seconds")
            if seconds is None:
                continue
            if float(seconds) <= cutoff:
                chosen = commit
                break
        if chosen is None:
            raise NotFoundError(uri, "snapshot_before_instant")
        blob = await self._fs.show(str(chosen["oid"]), ctx, path=uri)
        if isinstance(blob, dict) and "bytes" in blob:
            blob = blob["bytes"]
        content = blob.decode("utf-8", errors="replace") if isinstance(blob, bytes) else str(blob)
        committer = chosen.get("committer") or chosen.get("author") or {}
        committed_at = datetime.fromtimestamp(
            float(committer.get("time_seconds", 0)), tz=timezone.utc
        ).isoformat()
        return {
            "uri": uri,
            "at": instant.isoformat(),
            "commit": chosen.get("oid"),
            "committed_at": committed_at,
            "message": chosen.get("message"),
            "content": content,
        }


def _session_id_from_archive(archive_uri: str) -> Optional[str]:
    marker = "/sessions/"
    if marker not in archive_uri:
        return None
    rest = archive_uri.split(marker, 1)[1]
    return rest.split("/", 1)[0] or None


def _is_not_found(exc: Exception) -> bool:
    if isinstance(exc, NotFoundError):
        return True
    name = type(exc).__name__
    return "NotFound" in name or "not found" in str(exc).lower()
