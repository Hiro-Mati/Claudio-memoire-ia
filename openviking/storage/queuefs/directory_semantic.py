# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Directory L0/L1 generation and in-flight refresh ownership."""

from __future__ import annotations

import asyncio
import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openviking.parse.parsers.media import get_media_type
from openviking.server.identity import RequestContext
from openviking.service.task_work_index import (
    TaskExecutionContext,
    bind_task_context,
    get_task_context,
)
from openviking.storage.queuefs.semantic_sidecar import write_semantic_sidecars
from openviking.telemetry import OperationTelemetry, bind_telemetry, get_current_telemetry
from openviking.utils.ingest_options import IngestOptions


@dataclass(frozen=True)
class DirectorySemanticRequest:
    event_id: str
    track_inflight: bool
    uri: str
    context_type: str
    ctx: RequestContext
    file_paths: tuple[str, ...]
    file_summaries: Dict[str, Dict[str, str]]
    file_summary_updates: Dict[str, Dict[str, str]]
    children_dirs: tuple[str, ...]
    children_abstracts: List[Dict[str, str]]
    changed: bool
    skip_vectorization: bool
    ingest_options: IngestOptions
    llm_sem: asyncio.Semaphore
    viking_fs: Any
    lock: Optional[Dict[str, Any]] = None
    task_context: Optional[TaskExecutionContext] = field(default_factory=get_task_context)
    telemetry: OperationTelemetry = field(default_factory=get_current_telemetry)


@dataclass(frozen=True)
class DirectorySemanticResult:
    abstract: str
    changed: bool
    committed: bool


@dataclass
class _DirectoryState:
    latest_request: Optional[DirectorySemanticRequest] = None
    latest_order: int = -1
    revision: int = 0
    event_orders: Dict[str, int] = field(default_factory=dict)
    pending_events: set[str] = field(default_factory=set)
    changed: bool = False
    summary_updates: Dict[str, tuple[int, Dict[str, str]]] = field(default_factory=dict)
    waiters: list[tuple[str, asyncio.Future[DirectorySemanticResult]]] = field(default_factory=list)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    running: bool = False


class DirectorySemanticTask:
    """Own directory snapshots, dirty revisions, L0/L1, and directory vectors."""

    def __init__(self, semantic_service: Any) -> None:
        self._semantic_service = semantic_service
        self._states: dict[str, _DirectoryState] = {}
        self._submission_order = 0

    def mark_dirty(self, directory_uri: str, event_id: str) -> None:
        """Register directory work before its file phase starts."""
        directory_uri = directory_uri.rstrip("/")
        if not directory_uri or not event_id:
            return
        state = self._states.setdefault(directory_uri, _DirectoryState())
        if event_id in state.event_orders:
            return
        self._submission_order += 1
        state.event_orders[event_id] = self._submission_order
        state.pending_events.add(event_id)
        state.revision += 1
        state.ready.clear()

    def discard_dirty(self, directory_uri: str, event_id: str) -> None:
        """Release a registration whose file phase failed before submission."""
        directory_uri = directory_uri.rstrip("/")
        state = self._states.get(directory_uri)
        if state is None or event_id not in state.pending_events:
            return
        state.pending_events.remove(event_id)
        state.event_orders.pop(event_id, None)
        if not state.pending_events:
            state.ready.set()
        if not state.running and not state.waiters and not state.pending_events:
            self._states.pop(directory_uri, None)

    async def read_file_summaries(
        self,
        viking_fs: Any,
        dir_uri: str,
        ctx: RequestContext,
    ) -> Dict[str, str]:
        try:
            overview = await viking_fs.read_file(f"{dir_uri}/.overview.md", ctx=ctx)
        except Exception:
            return {}
        return self._semantic_service.parse_overview(overview) if overview else {}

    async def refresh(self, request: DirectorySemanticRequest) -> DirectorySemanticResult:
        if not request.track_inflight or not request.event_id:
            return await self._run_bound(
                request,
                request.file_summary_updates,
                request.changed,
                lambda: True,
            )

        directory_uri = request.uri.rstrip("/")
        self.mark_dirty(directory_uri, request.event_id)
        state = self._states[directory_uri]
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[DirectorySemanticResult] = loop.create_future()
        order = state.event_orders[request.event_id]
        state.pending_events.discard(request.event_id)
        state.waiters.append((request.event_id, waiter))
        state.changed = state.changed or request.changed
        self._merge_summaries(state, request.file_summary_updates, order)
        if order >= state.latest_order:
            state.latest_request = request
            state.latest_order = order
        if not state.pending_events:
            state.ready.set()

        if state.running:
            return await asyncio.shield(waiter)
        state.running = True

        try:
            while True:
                await state.ready.wait()
                await asyncio.sleep(0)
                if state.pending_events:
                    continue
                revision = state.revision
                latest = state.latest_request
                assert latest is not None
                updates = {
                    path: dict(ordered_summary[1])
                    for path, ordered_summary in state.summary_updates.items()
                }
                result = await self._run_bound(
                    latest,
                    updates,
                    state.changed,
                    lambda expected=revision: expected == state.revision,
                )
                if revision != state.revision or state.pending_events:
                    continue

                self._states.pop(directory_uri, None)
                for event_id, pending in state.waiters:
                    if pending.done():
                        continue
                    pending.set_result(
                        DirectorySemanticResult(
                            abstract=result.abstract,
                            changed=result.changed,
                            committed=result.committed and event_id == latest.event_id,
                        )
                    )
                return waiter.result()
        except asyncio.CancelledError:
            self._states.pop(directory_uri, None)
            for _, pending in state.waiters:
                if not pending.done():
                    pending.cancel()
            raise
        except Exception as error:
            self._states.pop(directory_uri, None)
            for _, pending in state.waiters:
                if pending is waiter:
                    pending.cancel()
                elif not pending.done():
                    pending.set_exception(error)
            raise

    @staticmethod
    def _merge_summaries(
        state: _DirectoryState,
        updates: Dict[str, Dict[str, str]],
        order: int,
    ) -> None:
        for path, summary in updates.items():
            current = state.summary_updates.get(path)
            if current is None or order >= current[0]:
                state.summary_updates[path] = (order, dict(summary))

    async def _run_bound(
        self,
        request: DirectorySemanticRequest,
        summary_updates: Dict[str, Dict[str, str]],
        changed: bool,
        is_current: Any,
    ) -> DirectorySemanticResult:
        task_context = (
            bind_task_context(
                request.task_context.task_id,
                request.task_context.account_id,
                request.task_context.user_id,
            )
            if request.task_context is not None
            else nullcontext()
        )
        with bind_telemetry(request.telemetry), task_context:
            return await self._generate_and_commit(
                request,
                summary_updates,
                changed,
                is_current,
            )

    async def _generate_and_commit(
        self,
        request: DirectorySemanticRequest,
        summary_updates: Dict[str, Dict[str, str]],
        changed: bool,
        is_current: Any,
    ) -> DirectorySemanticResult:
        if not changed and not summary_updates:
            try:
                abstract = await request.viking_fs.read_file(
                    f"{request.uri}/.abstract.md", ctx=request.ctx
                )
                overview = await request.viking_fs.read_file(
                    f"{request.uri}/.overview.md", ctx=request.ctx
                )
                if overview is not None and abstract is not None:
                    return DirectorySemanticResult(
                        abstract=abstract, changed=False, committed=False
                    )
            except Exception:
                pass

        file_summaries = [
            summary_updates.get(path)
            or request.file_summaries.get(path)
            or {"name": path.rsplit("/", 1)[-1], "summary": ""}
            for path in request.file_paths
        ]
        overview = self._select_direct_media_overview(
            request.file_paths,
            request.children_dirs,
            file_summaries,
        )
        if overview is None:
            async with request.llm_sem:
                overview = await self._semantic_service.generate_overview(
                    request.uri,
                    file_summaries,
                    request.children_abstracts,
                )
        overview, abstract = self._semantic_service.normalize_overview(overview)

        wrote = await write_semantic_sidecars(
            viking_fs=request.viking_fs,
            dir_uri=request.uri,
            overview=overview,
            abstract=abstract,
            ctx=request.ctx,
            is_stale=lambda: not is_current(),
            lock=request.lock,
            log_prefix="[DirectorySemanticTask]",
        )
        if not wrote or not is_current():
            return DirectorySemanticResult(abstract=abstract, changed=True, committed=False)

        if not request.skip_vectorization:
            await self._semantic_service.vectorize_directory(
                request.uri,
                context_type=request.context_type,
                abstract=abstract,
                overview=overview,
                ctx=request.ctx,
                ingest_options=request.ingest_options,
            )
        return DirectorySemanticResult(abstract=abstract, changed=True, committed=True)

    @staticmethod
    def _select_direct_media_overview(
        file_paths: tuple[str, ...],
        children_dirs: tuple[str, ...],
        file_summaries: List[Dict[str, str]],
    ) -> Optional[str]:
        if len(file_paths) != 1 or children_dirs or len(file_summaries) != 1:
            return None
        file_path = file_paths[0]
        filename = file_path.rsplit("/", 1)[-1]
        if get_media_type(file_path, None) not in {"audio", "video"}:
            return None

        summary = str(file_summaries[0].get("summary") or "").strip()
        if not summary.startswith("# "):
            return None
        lines = summary.splitlines()
        brief_start = next((idx for idx in range(1, len(lines)) if lines[idx].strip()), None)
        if brief_start is None or lines[brief_start].lstrip().startswith("#"):
            return None
        brief_end = brief_start
        while brief_end < len(lines) and lines[brief_end].strip():
            if lines[brief_end].lstrip().startswith("#"):
                return None
            brief_end += 1
        heading = re.compile(rf"^###\s+{re.escape(filename)}\s*$", re.MULTILINE)
        return summary if any(heading.fullmatch(line) for line in lines[brief_end:]) else None


__all__ = ["DirectorySemanticRequest", "DirectorySemanticResult", "DirectorySemanticTask"]
