# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""QueueFS keyed enqueue and physical retry lifecycle tests."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import openviking.storage.queuefs.named_queue as named_queue_module
from openviking.storage.queuefs.named_queue import DequeueHandlerBase, NamedQueue


@pytest.fixture
def async_agfs() -> AsyncMock:
    client = AsyncMock()
    client.write = AsyncMock(return_value=10)
    return client


@pytest.fixture
def queue(async_agfs: AsyncMock) -> NamedQueue:
    result = NamedQueue(MagicMock(), "/queue", "Semantic")
    result._async_agfs = async_agfs
    result._initialized = True
    return result


@pytest.mark.asyncio
async def test_enqueue_keyed_writes_atomic_request_without_task_metadata(
    queue: NamedQueue, async_agfs: AsyncMock
) -> None:
    message_id = await queue.enqueue_keyed(
        {"id": "semantic-1", "uri": "viking://resources/docs"},
        dispatch_key="semantic-v1:key",
        merge_signature="sha256:sig",
    )

    assert message_id == "10"
    path, raw = async_agfs.write.await_args.args
    assert path.endswith("/Semantic/enqueue_keyed")
    async_agfs.write.assert_awaited_once()
    async_agfs.read.assert_not_awaited()
    assert json.loads(raw) == {
        "dispatch_key": "semantic-v1:key",
        "merge_signature": "sha256:sig",
        "data": json.dumps(
            {"id": "semantic-1", "uri": "viking://resources/docs"},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


@pytest.mark.asyncio
async def test_enqueue_keyed_rejects_task_owned_work(
    queue: NamedQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(named_queue_module, "get_task_context", lambda: object())

    with pytest.raises(ValueError, match="task-owned"):
        await queue.enqueue_keyed({}, dispatch_key="k", merge_signature="s")


class _RetryHandler(DequeueHandlerBase):
    async def on_dequeue(self, data):
        self.report_requeue()
        self.report_success()
        raise named_queue_module.QueueMessageRetry(delay_seconds=0)


@pytest.mark.asyncio
async def test_serial_dequeue_requeues_same_physical_id_and_skips_ack(
    queue: NamedQueue, async_agfs: AsyncMock
) -> None:
    queue.set_dequeue_handler(_RetryHandler())
    async_agfs.read.return_value = json.dumps({"id": "physical-1", "data": "{}"}).encode()

    await queue.dequeue()

    written_paths = [call.args[0] for call in async_agfs.write.await_args_list]
    assert queue.path + "/requeue" in written_paths
    assert queue.path + "/ack" not in written_paths
    assert async_agfs.write.await_args.args == (queue.path + "/requeue", b"physical-1")
