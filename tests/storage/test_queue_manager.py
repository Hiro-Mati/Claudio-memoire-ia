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
from openviking.storage.queuefs.named_queue import NamedQueue
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
async def test_concurrent_retry_task_propagates_requeue_failure_without_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = QueueManager(agfs=MagicMock())
    queue = NamedQueue(MagicMock(), "/queue", "Semantic")
    queue._initialized = True
    queue._async_agfs = AsyncMock()
    queue.set_dequeue_handler(MagicMock())
    queue.size = AsyncMock(side_effect=[1, 0])
    queue.dequeue_raw = AsyncMock(return_value={"id": "physical-1", "data": "{}"})
    queue.process_dequeued = AsyncMock(
        side_effect=named_queue_module.QueueMessageRetry(delay_seconds=0)
    )
    queue.ack = AsyncMock()
    stop = threading.Event()
    tasks = []
    original_create_task = asyncio.create_task

    async def fail_requeue(msg_id: str) -> None:
        assert msg_id == "physical-1"
        stop.set()
        raise RuntimeError("QueueFS requeue failed")

    def record_task(coro):
        task = original_create_task(coro)
        tasks.append(task)
        return task

    queue.requeue = AsyncMock(side_effect=fail_requeue)
    monkeypatch.setattr(asyncio, "create_task", record_task)

    await asyncio.wait_for(
        manager._worker_async_concurrent(queue, stop, max_concurrent=2),
        timeout=1,
    )

    assert len(tasks) == 1
    with pytest.raises(RuntimeError, match="QueueFS requeue failed"):
        tasks[0].result()
    queue.requeue.assert_awaited_once_with("physical-1")
    queue.ack.assert_not_awaited()
