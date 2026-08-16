# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Memory-specific semantic task."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple

from openviking.server.identity import RequestContext
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.queuefs.semantic_queue import is_semantic_msg_stale
from openviking.storage.queuefs.semantic_sidecar import write_semantic_sidecars
from openviking.storage.viking_fs import LS_ALL_NODES, get_viking_fs
from openviking_cli.utils import VikingURI
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class MemorySemanticTask:
    """Generate one memory directory while preserving its separate lifecycle."""

    def __init__(
        self,
        semantic_service: Any,
        max_concurrent_llm: int,
        default_ctx: RequestContext,
    ) -> None:
        self._semantic_service = semantic_service
        self._max_concurrent_llm = max_concurrent_llm
        self._default_ctx = default_ctx

    async def run(
        self,
        msg: SemanticMsg,
        ctx: Optional[RequestContext] = None,
        lock: Optional[Dict[str, Any]] = None,
    ) -> None:
        viking_fs = get_viking_fs()
        dir_uri = msg.uri
        ctx = ctx or self._default_ctx
        llm_sem = asyncio.Semaphore(self._max_concurrent_llm)
        try:
            entries = await viking_fs.ls(dir_uri, node_limit=LS_ALL_NODES, ctx=ctx)
        except Exception as error:
            raise RuntimeError(f"Failed to list memory directory {dir_uri}: {error}") from error

        file_paths = [
            VikingURI(dir_uri).join(entry["name"]).uri
            for entry in entries
            if entry.get("name")
            and not entry["name"].startswith(".")
            and entry["name"] not in {".", ".."}
            and not entry.get("isDir", False)
        ]
        if not file_paths:
            logger.info("No memory files found in %s", dir_uri)
            return

        existing_summaries: Dict[str, str] = {}
        if msg.changes:
            try:
                old_overview = await viking_fs.read_file(f"{dir_uri}/.overview.md", ctx=ctx)
                if old_overview:
                    existing_summaries = self._semantic_service.parse_overview(old_overview)
            except Exception as error:
                logger.debug("No existing overview.md found for %s: %s", dir_uri, error)

        changed_files: Set[str] = set()
        if msg.changes:
            changed_files = set(msg.changes.get("added", []) + msg.changes.get("modified", []))

        pending: List[Tuple[int, str]] = []
        file_summaries: List[Optional[Dict[str, str]]] = [None] * len(file_paths)
        paths_to_vectorize = changed_files if msg.changes else set(file_paths)
        for index, file_path in enumerate(file_paths):
            file_name = file_path.rsplit("/", 1)[-1]
            if file_path not in changed_files and file_name in existing_summaries:
                file_summaries[index] = {
                    "name": file_name,
                    "summary": existing_summaries[file_name],
                }
            else:
                pending.append((index, file_path))

        if file_paths and not pending:
            self._record_cache(hit=True, miss=False)
        elif pending:
            self._record_cache(hit=len(file_paths) > len(pending), miss=True)

        async def generate(index: int, file_path: str) -> None:
            file_name = file_path.rsplit("/", 1)[-1]
            try:
                generated = await self._semantic_service.generate_file_summary(
                    file_path,
                    llm_sem=llm_sem,
                    ctx=ctx,
                )
            except Exception as error:
                logger.warning("Failed to generate summary for %s: %s", file_path, error)
                generated = {"name": file_name, "summary": ""}

            if file_path in paths_to_vectorize and not msg.skip_vectorization:
                await self._semantic_service.vectorize_file(
                    parent_uri=dir_uri,
                    context_type="memory",
                    file_path=file_path,
                    summary_dict=generated,
                    ctx=ctx,
                    preserve_existing_created_at=True,
                )
            file_summaries[index] = {
                "name": str(generated.get("name") or file_name),
                "summary": str(generated.get("summary") or ""),
            }

        batch_size = max(1, min(self._max_concurrent_llm, 10))
        for start in range(0, len(pending), batch_size):
            await asyncio.gather(
                *(generate(index, path) for index, path in pending[start : start + batch_size])
            )

        completed = [summary for summary in file_summaries if summary is not None]
        generated = await self._semantic_service.generate_overview(
            dir_uri,
            completed,
            [],
            llm_sem=llm_sem,
        )
        overview, abstract = self._semantic_service.normalize_overview(generated)
        try:
            wrote = await self._write_directory_semantics(
                msg=msg,
                viking_fs=viking_fs,
                dir_uri=dir_uri,
                overview=overview,
                abstract=abstract,
                ctx=ctx,
                lock=lock,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to write abstract/overview for {dir_uri}: {error}"
            ) from error
        if not wrote or msg.skip_vectorization:
            return
        await self._semantic_service.vectorize_directory(
            uri=dir_uri,
            context_type="memory",
            abstract=abstract,
            overview=overview,
            ctx=ctx,
        )

    @staticmethod
    def _record_cache(*, hit: bool, miss: bool) -> None:
        try:
            from openviking.metrics.datasources.cache import CacheEventDataSource

            if hit:
                CacheEventDataSource.record_hit("L1")
            if miss:
                CacheEventDataSource.record_miss("L1")
        except Exception:
            pass

    @staticmethod
    async def _write_directory_semantics(
        *,
        msg: SemanticMsg,
        viking_fs: Any,
        dir_uri: str,
        overview: str,
        abstract: str,
        ctx: Optional[RequestContext],
        lock: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return await write_semantic_sidecars(
            viking_fs=viking_fs,
            dir_uri=dir_uri,
            overview=overview,
            abstract=abstract,
            ctx=ctx,
            is_stale=lambda: is_semantic_msg_stale(msg),
            lock=lock,
            log_prefix="[MemorySemantic]",
        )


__all__ = ["MemorySemanticTask"]
