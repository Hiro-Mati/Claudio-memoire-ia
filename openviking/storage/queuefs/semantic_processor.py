# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""SemanticProcessor: Processes messages from SemanticQueue, generates .abstract.md and .overview.md."""

import threading
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

from openviking.observability.context import (
    bind_root_observability_context,
    reset_root_observability_context,
)
from openviking.server.identity import RequestContext, Role
from openviking.service.task_work_index import detach_task_context
from openviking.storage.errors import LockAcquisitionError
from openviking.storage.queuefs.directory_semantic import DirectorySemanticTask
from openviking.storage.queuefs.memory_semantic import MemorySemanticTask
from openviking.storage.queuefs.named_queue import DequeueHandlerBase
from openviking.storage.queuefs.semantic_dag import DagStats, SemanticDagExecutor
from openviking.storage.queuefs.semantic_lock import SemanticLockScope
from openviking.storage.queuefs.semantic_msg import SemanticMsg, build_semantic_coalesce_key
from openviking.storage.queuefs.semantic_queue import is_semantic_msg_stale
from openviking.storage.queuefs.semantic_service import SemanticService
from openviking.storage.queuefs.semantic_sync import sync_semantic_tree
from openviking.storage.viking_fs import get_viking_fs
from openviking.telemetry import bind_telemetry, resolve_telemetry
from openviking.telemetry.request_wait_tracker import get_request_wait_tracker
from openviking.telemetry.span_models import create_root_span_attributes
from openviking.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    classify_api_error,
)
from openviking.utils.model_retry import ERROR_CLASS_INPUT_TOO_LARGE, ERROR_CLASS_PERMANENT
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils import VikingURI
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class RequestQueueStats:
    processed: int = 0
    requeue_count: int = 0
    error_count: int = 0


class SemanticProcessor(DequeueHandlerBase):
    """
    Semantic processor, generates .abstract.md and .overview.md bottom-up.

    Processing flow:
    1. Concurrently generate summaries for files in directory
    2. Collect .abstract.md from subdirectories
    3. Generate .abstract.md and .overview.md for this directory
    4. Enqueue to EmbeddingQueue for vectorization
    """

    _stats_lock = threading.Lock()
    _dag_stats_by_telemetry_id: Dict[str, DagStats] = {}
    _dag_stats_by_uri: Dict[str, DagStats] = {}
    _dag_stats_order: List[Tuple[str, str]] = []
    _request_stats_by_telemetry_id: Dict[str, RequestQueueStats] = {}
    _request_stats_order: List[str] = []
    _max_cached_stats = 256

    def __init__(self, max_concurrent_llm: int = 32):
        """
        Initialize SemanticProcessor.

        Args:
            max_concurrent_llm: Maximum concurrent LLM calls
        """
        self.max_concurrent_llm = max_concurrent_llm
        self._default_ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
        self._semantic_service = SemanticService(max_concurrent_llm, self._default_ctx)
        self._circuit_breaker = CircuitBreaker()
        self._directory_semantic_task = DirectorySemanticTask(self._semantic_service)
        self._memory_semantic_task = MemorySemanticTask(
            self._semantic_service,
            max_concurrent_llm,
            self._default_ctx,
        )

    @classmethod
    def _cache_dag_stats(cls, telemetry_id: str, uri: str, stats: DagStats) -> None:
        with cls._stats_lock:
            if telemetry_id:
                cls._dag_stats_by_telemetry_id[telemetry_id] = stats
            cls._dag_stats_by_uri[uri] = stats
            cls._dag_stats_order.append((telemetry_id, uri))
            if len(cls._dag_stats_order) > cls._max_cached_stats:
                old_telemetry_id, old_uri = cls._dag_stats_order.pop(0)
                if old_telemetry_id:
                    cls._dag_stats_by_telemetry_id.pop(old_telemetry_id, None)
                cls._dag_stats_by_uri.pop(old_uri, None)

    @classmethod
    def consume_dag_stats(
        cls,
        telemetry_id: str = "",
        uri: Optional[str] = None,
    ) -> Optional[DagStats]:
        with cls._stats_lock:
            if telemetry_id and telemetry_id in cls._dag_stats_by_telemetry_id:
                stats = cls._dag_stats_by_telemetry_id.pop(telemetry_id, None)
                if uri:
                    cls._dag_stats_by_uri.pop(uri, None)
                return stats
            if uri and uri in cls._dag_stats_by_uri:
                return cls._dag_stats_by_uri.pop(uri, None)
        return None

    @classmethod
    def _merge_request_stats(
        cls,
        telemetry_id: str,
        processed: int = 0,
        requeue_count: int = 0,
        error_count: int = 0,
    ) -> None:
        if not telemetry_id:
            return
        with cls._stats_lock:
            stats = cls._request_stats_by_telemetry_id.setdefault(telemetry_id, RequestQueueStats())
            stats.processed += processed
            stats.requeue_count += requeue_count
            stats.error_count += error_count
            cls._request_stats_order.append(telemetry_id)
            if len(cls._request_stats_order) > cls._max_cached_stats:
                old_telemetry_id = cls._request_stats_order.pop(0)
                if old_telemetry_id != telemetry_id:
                    cls._request_stats_by_telemetry_id.pop(old_telemetry_id, None)

    @classmethod
    def consume_request_stats(cls, telemetry_id: str) -> Optional[RequestQueueStats]:
        if not telemetry_id:
            return None
        with cls._stats_lock:
            return cls._request_stats_by_telemetry_id.pop(telemetry_id, None)

    @staticmethod
    def _ctx_from_semantic_msg(msg: SemanticMsg) -> RequestContext:
        role = Role(msg.role or Role.ROOT)
        return RequestContext(
            user=UserIdentifier(msg.account_id, msg.user_id),
            role=role,
        )

    async def _reenqueue_semantic_msg(self, msg: SemanticMsg) -> None:
        """Re-enqueue a semantic message for later processing.

        Throttles with a sleep when the circuit breaker is open to prevent
        re-enqueue storms (messages cycling at 5/sec during OPEN window).
        """
        import asyncio

        from openviking.storage.queuefs import get_queue_manager

        # Throttle to prevent re-enqueue storm during OPEN window
        wait = self._circuit_breaker.retry_after
        if wait > 0:
            await asyncio.sleep(wait)

        queue_manager = get_queue_manager()
        if queue_manager is not None:
            semantic_queue = queue_manager.get_queue(queue_manager.SEMANTIC)
            await semantic_queue.enqueue(msg)
            logger.info(f"Re-enqueued semantic message: {msg.uri}")
        else:
            logger.warning(f"No queue manager available, cannot re-enqueue: {msg.uri}")

    async def _requeue_semantic_msg_after_error(
        self,
        msg: SemanticMsg,
        data: Optional[Dict[str, Any]],
        error: Exception,
    ) -> None:
        try:
            await self._reenqueue_semantic_msg(msg)
            self._merge_request_stats(msg.telemetry_id, requeue_count=1)
            get_request_wait_tracker().record_semantic_requeue(msg.telemetry_id)
            self.report_requeue()
        except Exception as requeue_err:
            logger.error(f"Failed to re-enqueue semantic message: {requeue_err}")
            self._merge_request_stats(msg.telemetry_id, error_count=1)
            get_request_wait_tracker().mark_semantic_failed(msg.telemetry_id, msg.id, str(error))
            self.report_error(str(error), data)
            return
        self.report_success()

    async def _enqueue_parent_refresh(self, msg: SemanticMsg, uri: str) -> None:
        if msg.context_type not in {"resource", "skill"}:
            return
        parent = VikingURI(uri).parent
        if parent is None:
            return
        parent_uri = parent.uri.rstrip("/")
        if (
            not parent_uri
            or parent_uri in {"viking://", "viking:"}
            or parent_uri == uri.rstrip("/")
        ):
            return

        from openviking.storage.queuefs import get_queue_manager

        queue_manager = get_queue_manager()
        if queue_manager is None:
            return
        semantic_queue = queue_manager.get_queue(queue_manager.SEMANTIC, allow_create=True)
        parent_msg = SemanticMsg(
            uri=parent_uri,
            context_type=msg.context_type,
            recursive=False,
            account_id=msg.account_id,
            user_id=msg.user_id,
            peer_id=msg.peer_id,
            role=msg.role,
            skip_vectorization=msg.skip_vectorization,
            changes={"modified": [uri]},
            coalesce_key=build_semantic_coalesce_key(
                context_type=msg.context_type,
                uri=parent_uri,
                account_id=msg.account_id,
                user_id=msg.user_id,
                peer_id=msg.peer_id,
            ),
        )
        with detach_task_context():
            await semantic_queue.enqueue(parent_msg)
        logger.info("Enqueued parent semantic refresh: %s", parent_uri)

    async def on_dequeue(
        self,
        data: Optional[Dict[str, Any]],
        lock: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process dequeued SemanticMsg, recursively process all subdirectories."""
        msg: Optional[SemanticMsg] = None
        collector = None
        try:
            import json

            if not data:
                return None

            if "data" in data and isinstance(data["data"], str):
                data = json.loads(data["data"])

            assert data is not None
            msg = SemanticMsg.from_dict(data)
            if VikingURI(msg.uri).parent is None:
                logger.warning("Skipping semantic generation for root URI: %s", msg.uri)
                if msg.telemetry_id and msg.id:
                    get_request_wait_tracker().mark_semantic_done(msg.telemetry_id, msg.id)
                self.report_success()
                return None
            if msg.context_type == "memory" and is_semantic_msg_stale(msg):
                logger.info(
                    "Skipping stale semantic message: uri=%s version=%s",
                    msg.uri,
                    msg.coalesce_version,
                )
                if msg.telemetry_id and msg.id:
                    get_request_wait_tracker().mark_semantic_done(msg.telemetry_id, msg.id)
                self.report_success()
                return None
            # Circuit breaker: if API is known-broken, re-enqueue and wait
            try:
                self._circuit_breaker.check()
            except CircuitBreakerOpen:
                logger.warning(
                    f"Circuit breaker is open, re-enqueueing semantic message: {msg.uri}"
                )
                await self._reenqueue_semantic_msg(msg)
                self._merge_request_stats(msg.telemetry_id, requeue_count=1)
                get_request_wait_tracker().record_semantic_requeue(msg.telemetry_id)
                self.report_requeue()
                self.report_success()
                return None
            collector = resolve_telemetry(msg.telemetry_id)
            telemetry_ctx = bind_telemetry(collector) if collector is not None else nullcontext()
            with telemetry_ctx:
                root_attrs = create_root_span_attributes(
                    http_method="QUEUE",
                    http_route=msg.context_type or "/queuefs/semantic",
                    request_id=msg.telemetry_id or msg.id,
                    url_path=msg.uri,
                )
                root_attrs.account_id = msg.account_id
                root_attrs.user_id = msg.user_id
                root_context_token = bind_root_observability_context(root_attrs)
                dirty_owned = False
                try:
                    current_ctx = self._ctx_from_semantic_msg(msg)
                    logger.info(
                        f"Processing semantic generation for: {msg.uri} (recursive={msg.recursive})"
                    )

                    logger.info(f"Processing semantic generation for: {msg})")

                    semantic_lock = await SemanticLockScope.resolve(
                        msg.lock_handoff,
                        caller_lock=lock,
                        fallback_path_factory=lambda: get_viking_fs()._uri_to_path(
                            msg.uri, ctx=current_ctx
                        ),
                    )
                    try:
                        if msg.context_type != "memory":
                            self._directory_semantic_task.mark_dirty(
                                msg.coalesce_key,
                                msg.id,
                            )
                            dirty_owned = True
                        if msg.context_type == "memory":
                            await self._memory_semantic_task.run(
                                msg,
                                ctx=current_ctx,
                                lock=semantic_lock.lock,
                            )
                        else:
                            is_incremental = False
                            target_uri = msg.target_uri
                            run_uri = msg.uri
                            changes = msg.changes
                            viking_fs = get_viking_fs()
                            if msg.target_uri:
                                target_exists = await viking_fs.exists(
                                    msg.target_uri, ctx=current_ctx
                                )
                                if msg.uri != msg.target_uri:
                                    logger.info(
                                        "Syncing semantic source into target before processing: "
                                        f"{msg.uri} -> {msg.target_uri}"
                                    )
                                    diff = await sync_semantic_tree(
                                        msg.uri,
                                        msg.target_uri,
                                        ctx=current_ctx,
                                        lock=semantic_lock.lock,
                                    )
                                    logger.info(
                                        "[SyncDiff] Diff computed: "
                                        f"added_files={len(diff.added_files)}, "
                                        f"deleted_files={len(diff.deleted_files)}, "
                                        f"updated_files={len(diff.updated_files)}, "
                                        f"added_dirs={len(diff.added_dirs)}, "
                                        f"deleted_dirs={len(diff.deleted_dirs)}"
                                    )
                                    changes = diff.to_changes()
                                    is_incremental = True
                                    target_uri = msg.target_uri
                                    run_uri = msg.target_uri
                                elif target_exists and msg.changes and msg.uri == msg.target_uri:
                                    is_incremental = True
                                    logger.info(
                                        f"Using direct incremental semantic update for: {msg.uri}"
                                    )
                            elif msg.changes:
                                is_incremental = True
                                target_uri = msg.uri
                                logger.info(
                                    f"Using direct incremental semantic update for: {msg.uri}"
                                )

                            executor = SemanticDagExecutor(
                                semantic_service=self._semantic_service,
                                context_type=msg.context_type,
                                max_concurrent_llm=self.max_concurrent_llm,
                                ctx=current_ctx,
                                incremental_update=is_incremental,
                                target_uri=target_uri,
                                recursive=msg.recursive,
                                lock=semantic_lock.lock,
                                is_code_repo=msg.is_code_repo,
                                changes=changes,
                                skip_vectorization=msg.skip_vectorization,
                                ingest_options=msg.ingest_options,
                                coalesce_key=msg.coalesce_key,
                                coalesce_event_id=msg.id,
                                directory_task=self._directory_semantic_task,
                            )
                            dirty_owned = False
                            await executor.run(run_uri)
                            self._cache_dag_stats(
                                msg.telemetry_id,
                                run_uri,
                                executor.get_stats(),
                            )
                            if executor.root_directory_committed:
                                await self._enqueue_parent_refresh(msg, target_uri or msg.uri)
                    finally:
                        await semantic_lock.close()
                    get_request_wait_tracker().mark_semantic_done(msg.telemetry_id, msg.id)
                    self._merge_request_stats(msg.telemetry_id, processed=1)
                    logger.info(f"Completed semantic generation for: {msg.uri}")
                    self.report_success()
                    self._circuit_breaker.record_success()
                    return None
                finally:
                    if dirty_owned:
                        self._directory_semantic_task.discard_dirty(
                            msg.coalesce_key,
                            msg.id,
                        )
                    reset_root_observability_context(root_context_token)

        except Exception as e:
            if isinstance(e, LockAcquisitionError):
                logger.warning(
                    "Lock error processing semantic message, re-enqueueing without "
                    "tripping API circuit breaker: %s",
                    e,
                    exc_info=True,
                )
                if msg is not None:
                    await self._requeue_semantic_msg_after_error(msg, data, e)
                else:
                    self.report_error(str(e), data)
                return None

            error_class = classify_api_error(e)
            if error_class == ERROR_CLASS_INPUT_TOO_LARGE:
                logger.error(
                    f"Input too large processing semantic message, dropping: {e}",
                    exc_info=True,
                )
                if msg is not None:
                    self._merge_request_stats(msg.telemetry_id, error_count=1)
                    get_request_wait_tracker().mark_semantic_failed(
                        msg.telemetry_id, msg.id, str(e)
                    )
                self.report_error(str(e), data)
            elif error_class == ERROR_CLASS_PERMANENT:
                logger.critical(
                    f"Permanent API error processing semantic message, dropping: {e}",
                    exc_info=True,
                )
                self._circuit_breaker.record_failure(e)
                if msg is not None:
                    self._merge_request_stats(msg.telemetry_id, error_count=1)
                    get_request_wait_tracker().mark_semantic_failed(
                        msg.telemetry_id, msg.id, str(e)
                    )
                self.report_error(str(e), data)
            else:
                # Transient or unknown — re-enqueue for retry
                logger.warning(
                    f"Transient API error processing semantic message, re-enqueueing: {e}",
                    exc_info=True,
                )
                self._circuit_breaker.record_failure(e)
                if msg is not None:
                    await self._requeue_semantic_msg_after_error(msg, data, e)
                else:
                    self.report_error(str(e), data)
            return None

    async def on_cancelled(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Release a queued semantic lock before cancelled work is ACKed."""
        try:
            import json

            payload = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(payload, str):
                payload = json.loads(payload)
            msg = SemanticMsg.from_dict(payload)
        except (TypeError, ValueError) as exc:
            self.report_error(str(exc), data)
            return None

        if msg.telemetry_id and msg.id:
            get_request_wait_tracker().mark_semantic_done(msg.telemetry_id, msg.id)
        if msg.lock_handoff is not None:
            try:
                viking_fs = get_viking_fs()
                lock = await viking_fs._async_agfs.pathlock_adopt(msg.lock_handoff)
                await viking_fs._async_agfs.pathlock_release(lock)
            except Exception as exc:
                logger.warning("Failed to release cancelled semantic lock: %s", exc)
        self.report_success()
        return None

    def get_dag_stats(self) -> Optional["DagStats"]:
        return SemanticDagExecutor.get_active_stats()
