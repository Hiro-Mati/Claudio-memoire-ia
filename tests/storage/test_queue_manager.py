# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Focused tests for QueueManager concurrency selection."""

import asyncio
import os
import subprocess
import sys
import threading
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

import openviking.storage.queuefs.named_queue as named_queue_module
import openviking.storage.queuefs.queue_manager as queue_manager_module
from openviking.storage.queuefs.named_queue import DequeueHandlerBase, NamedQueue
from openviking.storage.queuefs.queue_manager import QueueManager
from openviking.storage.queuefs.semantic_msg import SemanticMsg


def _legacy_semantic_message(*, coalesce_version: int) -> SemanticMsg:
    return SemanticMsg(
        uri="viking://resources/bootstrap",
        context_type="resource",
        recursive=True,
        coalesce_key="resource|account|user|peer|viking://resources/bootstrap",
        coalesce_version=coalesce_version,
    )


class _SuccessHandler(DequeueHandlerBase):
    async def on_dequeue(self, data):
        self.report_success()
        return data


class _RetryHandler(DequeueHandlerBase):
    async def on_dequeue(self, data):
        del data
        self.report_requeue()
        self.report_success()
        raise named_queue_module.QueueMessageRetry(delay_seconds=0)


def test_queuefs_package_imports_in_a_clean_process(tmp_path) -> None:
    env = os.environ.copy()
    env["OPENVIKING_CONFIG_FILE"] = str(tmp_path / "missing-ov.conf")

    subprocess.run(
        [sys.executable, "-c", "from openviking.storage.queuefs import QueueManager"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_queue_concurrency_uses_separate_configured_values() -> None:
    manager = QueueManager(
        agfs=object(),
        max_concurrent_external_parse=9,
        max_concurrent_add_resource=7,
        max_concurrent_session_commit=5,
    )

    assert manager._max_concurrent_for_queue(manager.EXTERNAL_PARSE) == 9
    assert manager._max_concurrent_for_queue(manager.ADD_RESOURCE) == 7
    assert manager._max_concurrent_for_queue(manager.SESSION_COMMIT) == 5


@pytest.mark.asyncio
async def test_prepare_task_tracking_bootstraps_semantic_before_start(monkeypatch):
    manager = QueueManager(agfs=MagicMock())
    semantic = manager.get_queue(manager.SEMANTIC, allow_create=True)
    embedding = manager.get_queue(manager.EMBEDDING, allow_create=True)
    semantic_snapshot = [
        {"id": "physical", "data": _legacy_semantic_message(coalesce_version=9).to_json()}
    ]
    semantic.snapshot = AsyncMock(return_value=semantic_snapshot)
    embedding.snapshot = AsyncMock(return_value=[])
    semantic.bootstrap_legacy_coalesce = Mock()
    tracker = MagicMock()
    tracker.restore_work_tasks = AsyncMock()

    await manager.prepare_task_tracking(tracker)

    semantic.bootstrap_legacy_coalesce.assert_called_once_with(semantic_snapshot)
    assert manager.is_running() is False


@pytest.mark.asyncio
async def test_concurrent_worker_retries_same_id_without_ack() -> None:
    manager = QueueManager(agfs=MagicMock())
    queue = NamedQueue(MagicMock(), "/queue", "Semantic")
    queue._initialized = True
    queue._async_agfs = AsyncMock()
    queue.set_dequeue_handler(MagicMock())
    queue.size = AsyncMock(side_effect=[1, 0, 0])
    queue.dequeue_raw = AsyncMock(side_effect=[{"id": "physical-1", "data": "{}"}, None])
    queue.process_dequeued = AsyncMock(
        side_effect=named_queue_module.QueueMessageRetry(delay_seconds=0)
    )
    queue.requeue = AsyncMock()
    queue.ack = AsyncMock()
    stop = threading.Event()

    async def requeue_and_stop(msg_id: str) -> None:
        assert msg_id == "physical-1"
        stop.set()

    queue.requeue.side_effect = requeue_and_stop

    await asyncio.wait_for(
        manager._worker_async_concurrent(queue, stop, max_concurrent=2),
        timeout=1,
    )

    queue.requeue.assert_awaited_once_with("physical-1")
    queue.ack.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_success_accounting_waits_for_physical_ack() -> None:
    manager = QueueManager(agfs=MagicMock())
    queue = NamedQueue(MagicMock(), "/queue", "Semantic")
    queue._initialized = True
    queue._async_agfs = AsyncMock()
    queue.set_dequeue_handler(_SuccessHandler())
    queue.size = AsyncMock(side_effect=[1, 0])
    queue.dequeue_raw = AsyncMock(return_value={"id": "physical-1", "data": "{}"})
    stop = threading.Event()
    observed_before_ack = []

    async def ack_and_stop(msg_id: str, data) -> None:
        observed_before_ack.append((queue._processed, queue._in_progress))
        assert msg_id == "physical-1"
        assert data["id"] == "physical-1"
        stop.set()

    queue.ack = AsyncMock(side_effect=ack_and_stop)

    await asyncio.wait_for(
        manager._worker_async_concurrent(queue, stop, max_concurrent=2),
        timeout=1,
    )

    assert observed_before_ack == [(0, 1)]
    assert (queue._processed, queue._in_progress, queue._error_count) == (1, 0, 0)


@pytest.mark.asyncio
async def test_concurrent_retry_accounting_waits_for_physical_requeue() -> None:
    manager = QueueManager(agfs=MagicMock())
    queue = NamedQueue(MagicMock(), "/queue", "Semantic")
    queue._initialized = True
    queue._async_agfs = AsyncMock()
    queue.set_dequeue_handler(_RetryHandler())
    queue.size = AsyncMock(side_effect=[1, 0])
    queue.dequeue_raw = AsyncMock(return_value={"id": "physical-1", "data": "{}"})
    queue.ack = AsyncMock()
    stop = threading.Event()
    observed_before_requeue = []

    async def requeue_and_stop(msg_id: str) -> None:
        observed_before_requeue.append((queue._processed, queue._requeue_count, queue._in_progress))
        assert msg_id == "physical-1"
        stop.set()

    queue.requeue = AsyncMock(side_effect=requeue_and_stop)

    await asyncio.wait_for(
        manager._worker_async_concurrent(queue, stop, max_concurrent=2),
        timeout=1,
    )

    assert observed_before_requeue == [(0, 0, 1)]
    assert (queue._processed, queue._requeue_count, queue._in_progress) == (1, 1, 0)
    queue.ack.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_ack_failure_is_consumed_logged_and_not_counted_successfully(
    caplog,
) -> None:
    manager = QueueManager(agfs=MagicMock())
    queue = NamedQueue(MagicMock(), "/queue", "Semantic")
    queue._initialized = True
    queue._async_agfs = AsyncMock()
    queue.set_dequeue_handler(_SuccessHandler())
    queue.size = AsyncMock(side_effect=[1, 0])
    queue.dequeue_raw = AsyncMock(return_value={"id": "physical-1", "data": "{}"})
    stop = threading.Event()

    async def fail_ack(msg_id: str, data) -> None:
        del msg_id, data
        stop.set()
        raise RuntimeError("QueueFS ack failed")

    queue.ack = AsyncMock(side_effect=fail_ack)

    queue_manager_module.logger.addHandler(caplog.handler)
    try:
        await asyncio.wait_for(
            manager._worker_async_concurrent(queue, stop, max_concurrent=2),
            timeout=1,
        )
    finally:
        queue_manager_module.logger.removeHandler(caplog.handler)

    assert (queue._processed, queue._in_progress, queue._error_count) == (0, 1, 1)
    assert queue._errors[-1].message == "QueueFS ack failed"
    assert "Concurrent settlement failure" in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_concurrent_requeue_failure_is_consumed_logged_and_not_counted_successfully(
    caplog,
) -> None:
    manager = QueueManager(agfs=MagicMock())
    queue = NamedQueue(MagicMock(), "/queue", "Semantic")
    queue._initialized = True
    queue._async_agfs = AsyncMock()
    queue.set_dequeue_handler(_RetryHandler())
    queue.size = AsyncMock(side_effect=[1, 0])
    queue.dequeue_raw = AsyncMock(return_value={"id": "physical-1", "data": "{}"})
    queue.ack = AsyncMock()
    stop = threading.Event()

    async def fail_requeue(msg_id: str) -> None:
        assert msg_id == "physical-1"
        stop.set()
        raise RuntimeError("QueueFS requeue failed")

    queue.requeue = AsyncMock(side_effect=fail_requeue)

    queue_manager_module.logger.addHandler(caplog.handler)
    try:
        await asyncio.wait_for(
            manager._worker_async_concurrent(queue, stop, max_concurrent=2),
            timeout=1,
        )
    finally:
        queue_manager_module.logger.removeHandler(caplog.handler)

    assert (queue._processed, queue._requeue_count, queue._in_progress) == (0, 0, 1)
    assert queue._error_count == 1
    assert queue._errors[-1].message == "QueueFS requeue failed"
    assert "Concurrent settlement failure" in caplog.text
    assert "RuntimeError" in caplog.text
    queue.requeue.assert_awaited_once_with("physical-1")
    queue.ack.assert_not_awaited()
