# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""QueueFS keyed enqueue and physical retry lifecycle tests."""

import hashlib
import json
import logging
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


class _SuccessHandler(DequeueHandlerBase):
    async def on_dequeue(self, data):
        self.report_success()
        return data


def _keyed_envelope(*, physical_id: str = "physical-1") -> dict:
    return {
        "id": physical_id,
        "data": json.dumps(
            {
                "_queuefs_keyed_batch": {
                    "schema_version": 1,
                    "dispatch_key": "sensitive-dispatch-key",
                    "merge_signature": "sensitive-merge-signature",
                    "contributions": [
                        '{"uri":"viking://resources/sensitive-a.md"}',
                        '{"telemetry_id":"sensitive-telemetry"}',
                    ],
                }
            }
        ),
    }


def test_keyed_processing_error_status_retains_only_bounded_diagnostics(
    queue: NamedQueue,
) -> None:
    envelope = _keyed_envelope(physical_id="physical-" + "x" * 300)
    queue._on_dequeue_start()

    queue._on_process_error("RuntimeError", envelope)

    error = queue._errors[-1]
    assert error.message == "RuntimeError"
    assert error.data == {
        "physical_id": ("physical-" + "x" * 300)[:128],
        "dispatch_hash_prefix": hashlib.sha256(b"sensitive-dispatch-key").hexdigest()[:12],
        "contribution_count": 2,
        "error_class": "RuntimeError",
    }
    rendered = repr(error)
    for sensitive in (
        "sensitive-dispatch-key",
        "sensitive-merge-signature",
        "sensitive-a.md",
        "sensitive-telemetry",
    ):
        assert sensitive not in rendered


def test_ordinary_processing_error_status_keeps_useful_legacy_details(
    queue: NamedQueue,
) -> None:
    data = {"id": "ordinary-1", "data": '{"kind":"ordinary"}'}
    queue._on_dequeue_start()

    queue._on_process_error("ordinary failure detail", data)

    assert queue._errors[-1].message == "ordinary failure detail"
    assert queue._errors[-1].data == data


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


@pytest.mark.asyncio
async def test_serial_dequeue_propagates_requeue_failure_without_ack(
    queue: NamedQueue, async_agfs: AsyncMock
) -> None:
    queue.set_dequeue_handler(_RetryHandler())
    async_agfs.read.return_value = json.dumps({"id": "physical-1", "data": "{}"}).encode()
    async_agfs.write.side_effect = RuntimeError("QueueFS requeue failed")

    with pytest.raises(RuntimeError, match="QueueFS requeue failed"):
        await queue.dequeue()

    async_agfs.write.assert_awaited_once_with(queue.path + "/requeue", b"physical-1")
    assert queue.path + "/ack" not in [call.args[0] for call in async_agfs.write.await_args_list]
    assert (queue._processed, queue._requeue_count, queue._in_progress) == (0, 0, 1)
    assert queue._error_count == 1
    assert queue._errors[-1].message == "QueueFS requeue failed"


@pytest.mark.asyncio
async def test_serial_ack_failure_is_observable_and_not_counted_successfully(
    queue: NamedQueue, async_agfs: AsyncMock
) -> None:
    queue.set_dequeue_handler(_SuccessHandler())
    async_agfs.read.return_value = json.dumps({"id": "physical-1", "data": "{}"}).encode()
    async_agfs.write.side_effect = RuntimeError("QueueFS ack failed")

    result = await queue.dequeue()

    assert result is None
    async_agfs.write.assert_awaited_once_with(queue.path + "/ack", b"physical-1")
    assert (queue._processed, queue._in_progress, queue._error_count) == (0, 1, 1)
    assert queue._errors[-1].message == "QueueFS ack failed"


@pytest.mark.asyncio
async def test_keyed_ack_failure_status_is_sanitized(
    queue: NamedQueue, async_agfs: AsyncMock, caplog
) -> None:
    queue.set_dequeue_handler(_SuccessHandler())
    envelope = _keyed_envelope()
    async_agfs.read.return_value = json.dumps(envelope).encode()
    async_agfs.write.side_effect = RuntimeError(
        "QueueFS ack failed for viking://resources/sensitive-a.md"
    )

    named_queue_module.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger=named_queue_module.logger.name):
            await queue.dequeue()
    finally:
        named_queue_module.logger.removeHandler(caplog.handler)

    error = queue._errors[-1]
    assert error.message == "RuntimeError"
    assert error.data == {
        "physical_id": "physical-1",
        "dispatch_hash_prefix": hashlib.sha256(b"sensitive-dispatch-key").hexdigest()[:12],
        "contribution_count": 2,
        "error_class": "RuntimeError",
    }
    assert "sensitive-a.md" not in repr(error)
    rendered_logs = "\n".join(
        f"{record.getMessage()} {record.__dict__!r}" for record in caplog.records
    )
    assert "sensitive-a.md" not in rendered_logs
    assert "QueueFS ack failed for" not in rendered_logs
    assert any(getattr(record, "physical_id", "") == "physical-1" for record in caplog.records)
