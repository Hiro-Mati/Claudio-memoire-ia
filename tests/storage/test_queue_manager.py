# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Focused tests for QueueManager concurrency selection."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

import openviking.storage.queuefs.named_queue as named_queue_module
from openviking.storage.queuefs.named_queue import NamedQueue
from openviking.storage.queuefs.queue_manager import QueueManager


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
