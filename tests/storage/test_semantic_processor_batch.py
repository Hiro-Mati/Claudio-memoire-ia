# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Batch lifecycle tests for semantic queue processing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import openviking.storage.queuefs.semantic_processor as semantic_processor_module
from openviking.storage.errors import LockAcquisitionError
from openviking.storage.queuefs.named_queue import QueueMessageRetry
from openviking.storage.queuefs.semantic_batch import semantic_batch_route
from openviking.storage.queuefs.semantic_dag import DagStats
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.queuefs.semantic_processor import SemanticProcessor
from openviking.utils.circuit_breaker import CircuitBreakerOpen
from openviking.utils.model_retry import ERROR_CLASS_INPUT_TOO_LARGE, ERROR_CLASS_PERMANENT

ROOT_URI = "viking://resources/docs"
A_URI = f"{ROOT_URI}/a.md"
B_URI = f"{ROOT_URI}/b.md"


def eligible_msg(**overrides) -> SemanticMsg:
    values = {
        "uri": ROOT_URI,
        "context_type": "resource",
        "recursive": False,
        "account_id": "account",
        "user_id": "user",
        "peer_id": "peer",
        "coalesce_key": f"resource|account|user|peer|{ROOT_URI}",
        "changes": {"modified": [A_URI]},
        "aggregate_directory": True,
    }
    values.update(overrides)
    return SemanticMsg(**values)


def keyed_payload(*messages: SemanticMsg) -> dict:
    route = semantic_batch_route(messages[0], task_owned=False)
    assert route is not None
    return {
        "_queuefs_keyed_batch": {
            "schema_version": 1,
            "dispatch_key": route.dispatch_key,
            "merge_signature": route.merge_signature,
            "contributions": [message.to_json() for message in messages],
        }
    }


def queue_envelope(message_id: str, payload: dict) -> dict:
    return {"id": message_id, "data": json.dumps(payload)}


class _FakeVikingFS:
    def _uri_to_path(self, uri, ctx=None):
        del ctx
        return uri


class _FakeExecutor:
    construct_count = 0
    kwargs = None

    def __init__(self, **kwargs):
        type(self).construct_count += 1
        type(self).kwargs = kwargs
        self.stale = False

    async def run(self, root_uri):
        self.root_uri = root_uri

    def get_stats(self):
        return DagStats(total_nodes=2, done_nodes=2)


class _WaitTracker:
    def __init__(self):
        self.semantic_done_calls = set()
        self.requeue_telemetry_ids = []
        self.failed_calls = []

    def mark_semantic_done(self, telemetry_id, root_id, processed_delta=1):
        assert processed_delta == 1
        self.semantic_done_calls.add((telemetry_id, root_id))

    def record_semantic_requeue(self, telemetry_id, delta=1):
        assert delta == 1
        self.requeue_telemetry_ids.append(telemetry_id)

    def mark_semantic_failed(self, telemetry_id, root_id, message):
        self.failed_calls.append(SimpleNamespace(telemetry_id=telemetry_id, root_id=root_id))


class _FailingExecutor(_FakeExecutor):
    error = RuntimeError("executor failure")

    async def run(self, root_uri):
        del root_uri
        raise self.error


class _OpenCircuitBreaker:
    retry_after = 3.5

    def check(self):
        raise CircuitBreakerOpen("open")


@pytest.mark.asyncio
async def test_keyed_batch_runs_one_dag_and_completes_every_semantic_root(monkeypatch):
    first = eligible_msg(telemetry_id="tm-1", changes={"modified": [A_URI]})
    second = eligible_msg(telemetry_id="tm-2", changes={"modified": [B_URI]})
    original_changes = (first.changes.copy(), second.changes.copy())
    tracker = _WaitTracker()
    _FakeExecutor.construct_count = 0
    _FakeExecutor.kwargs = None
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs",
        lambda: _FakeVikingFS(),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticDagExecutor",
        _FakeExecutor,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticLockScope.resolve",
        AsyncMock(return_value=SimpleNamespace(lock=None, close=AsyncMock())),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_request_wait_tracker",
        lambda: tracker,
    )
    processor = SemanticProcessor(max_concurrent_llm=2)
    processor._enqueue_parent_refresh = AsyncMock()

    await processor.on_dequeue(queue_envelope("physical-1", keyed_payload(first, second)))

    assert _FakeExecutor.construct_count == 1
    assert _FakeExecutor.kwargs["changes"] == {"modified": [A_URI, B_URI]}
    assert _FakeExecutor.kwargs["file_contributions"] == {
        A_URI: (first,),
        B_URI: (second,),
    }
    assert _FakeExecutor.kwargs["shared_directory_embedding"] is True
    assert tracker.semantic_done_calls == {
        ("tm-1", first.id),
        ("tm-2", second.id),
    }
    assert first.changes == original_changes[0]
    assert second.changes == original_changes[1]
    processor._enqueue_parent_refresh.assert_awaited_once()
    first_dag_stats = SemanticProcessor.consume_dag_stats("tm-1", ROOT_URI)
    second_dag_stats = SemanticProcessor.consume_dag_stats("tm-2")
    assert first_dag_stats is second_dag_stats
    assert first_dag_stats == DagStats(total_nodes=2, done_nodes=2)
    assert SemanticProcessor.consume_request_stats("tm-1").processed == 1
    assert SemanticProcessor.consume_request_stats("tm-2").processed == 1


@pytest.mark.asyncio
async def test_transient_batch_error_requests_physical_retry_without_new_enqueue(monkeypatch):
    first = eligible_msg(telemetry_id="tm-1", changes={"modified": [A_URI]})
    second = eligible_msg(telemetry_id="tm-2", changes={"modified": [B_URI]})
    tracker = _WaitTracker()
    replacement_enqueue = AsyncMock()
    queue_manager = SimpleNamespace(
        SEMANTIC="semantic",
        get_queue=lambda queue_name: SimpleNamespace(enqueue=replacement_enqueue),
    )
    _FailingExecutor.error = TimeoutError("vlm timeout")
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs",
        lambda: _FakeVikingFS(),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticDagExecutor",
        _FailingExecutor,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticLockScope.resolve",
        AsyncMock(return_value=SimpleNamespace(lock=None, close=AsyncMock())),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_request_wait_tracker",
        lambda: tracker,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.get_queue_manager",
        lambda: queue_manager,
    )

    with pytest.raises(QueueMessageRetry):
        await SemanticProcessor().on_dequeue(
            queue_envelope("physical-1", keyed_payload(first, second))
        )

    replacement_enqueue.assert_not_awaited()
    assert tracker.requeue_telemetry_ids == ["tm-1", "tm-2"]
    assert tracker.failed_calls == []
    assert SemanticProcessor.consume_request_stats("tm-1").requeue_count == 1
    assert SemanticProcessor.consume_request_stats("tm-2").requeue_count == 1


@pytest.mark.asyncio
async def test_permanent_batch_error_fails_all_contributions_and_returns_for_ack(monkeypatch):
    first = eligible_msg(telemetry_id="tm-1", changes={"modified": [A_URI]})
    second = eligible_msg(telemetry_id="tm-2", changes={"modified": [B_URI]})
    tracker = _WaitTracker()
    _FailingExecutor.error = RuntimeError("invalid model input")
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs",
        lambda: _FakeVikingFS(),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticDagExecutor",
        _FailingExecutor,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticLockScope.resolve",
        AsyncMock(return_value=SimpleNamespace(lock=None, close=AsyncMock())),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_request_wait_tracker",
        lambda: tracker,
    )
    monkeypatch.setattr(
        semantic_processor_module,
        "classify_api_error",
        lambda error: ERROR_CLASS_PERMANENT,
    )

    result = await SemanticProcessor().on_dequeue(
        queue_envelope("physical-1", keyed_payload(first, second))
    )

    assert result is None
    assert {(call.telemetry_id, call.root_id) for call in tracker.failed_calls} == {
        ("tm-1", first.id),
        ("tm-2", second.id),
    }
    assert tracker.semantic_done_calls == set()
    assert tracker.requeue_telemetry_ids == []
    assert SemanticProcessor.consume_request_stats("tm-1").error_count == 1
    assert SemanticProcessor.consume_request_stats("tm-2").error_count == 1


@pytest.mark.asyncio
async def test_open_circuit_requests_retry_after_breaker_delay(monkeypatch):
    message = eligible_msg(telemetry_id="tm-1")
    tracker = _WaitTracker()
    replacement_enqueue = AsyncMock()
    queue_manager = SimpleNamespace(
        SEMANTIC="semantic",
        get_queue=lambda queue_name: SimpleNamespace(enqueue=replacement_enqueue),
    )
    _FakeExecutor.construct_count = 0
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticDagExecutor",
        _FakeExecutor,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_request_wait_tracker",
        lambda: tracker,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.get_queue_manager",
        lambda: queue_manager,
    )
    processor = SemanticProcessor()
    processor._circuit_breaker = _OpenCircuitBreaker()

    with pytest.raises(QueueMessageRetry) as caught:
        await processor.on_dequeue(queue_envelope("physical-1", keyed_payload(message)))

    assert caught.value.delay_seconds == 3.5
    assert _FakeExecutor.construct_count == 0
    replacement_enqueue.assert_not_awaited()
    assert tracker.requeue_telemetry_ids == ["tm-1"]
    assert SemanticProcessor.consume_request_stats("tm-1").requeue_count == 1


@pytest.mark.asyncio
async def test_cancelled_batch_completes_every_semantic_root(monkeypatch):
    first = eligible_msg(telemetry_id="tm-1", changes={"modified": [A_URI]})
    second = eligible_msg(telemetry_id="tm-2", changes={"modified": [B_URI]})
    tracker = _WaitTracker()
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_request_wait_tracker",
        lambda: tracker,
    )

    await SemanticProcessor().on_cancelled(
        queue_envelope("physical-1", keyed_payload(first, second))
    )

    assert tracker.semantic_done_calls == {
        ("tm-1", first.id),
        ("tm-2", second.id),
    }
    assert SemanticProcessor.consume_request_stats("tm-1").processed == 1
    assert SemanticProcessor.consume_request_stats("tm-2").processed == 1


@pytest.mark.asyncio
async def test_lock_contention_retries_batch_without_constructing_dag(monkeypatch):
    first = eligible_msg(telemetry_id="tm-1", changes={"modified": [A_URI]})
    second = eligible_msg(telemetry_id="tm-2", changes={"modified": [B_URI]})
    tracker = _WaitTracker()
    replacement_enqueue = AsyncMock()
    queue_manager = SimpleNamespace(
        SEMANTIC="semantic",
        get_queue=lambda queue_name: SimpleNamespace(enqueue=replacement_enqueue),
    )
    _FakeExecutor.construct_count = 0
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs",
        lambda: _FakeVikingFS(),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticDagExecutor",
        _FakeExecutor,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticLockScope.resolve",
        AsyncMock(side_effect=LockAcquisitionError("lock conflict")),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_request_wait_tracker",
        lambda: tracker,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.get_queue_manager",
        lambda: queue_manager,
    )

    with pytest.raises(QueueMessageRetry):
        await SemanticProcessor().on_dequeue(
            queue_envelope("physical-1", keyed_payload(first, second))
        )

    assert _FakeExecutor.construct_count == 0
    replacement_enqueue.assert_not_awaited()
    assert tracker.requeue_telemetry_ids == ["tm-1", "tm-2"]
    assert SemanticProcessor.consume_request_stats("tm-1").requeue_count == 1
    assert SemanticProcessor.consume_request_stats("tm-2").requeue_count == 1


@pytest.mark.asyncio
async def test_input_too_large_fails_every_contribution_once(monkeypatch):
    first = eligible_msg(telemetry_id="tm-1", changes={"modified": [A_URI]})
    second = eligible_msg(telemetry_id="tm-2", changes={"modified": [B_URI]})
    tracker = _WaitTracker()
    _FailingExecutor.error = RuntimeError("oversized prompt")
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs",
        lambda: _FakeVikingFS(),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticDagExecutor",
        _FailingExecutor,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticLockScope.resolve",
        AsyncMock(return_value=SimpleNamespace(lock=None, close=AsyncMock())),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_request_wait_tracker",
        lambda: tracker,
    )
    monkeypatch.setattr(
        semantic_processor_module,
        "classify_api_error",
        lambda error: ERROR_CLASS_INPUT_TOO_LARGE,
    )

    result = await SemanticProcessor().on_dequeue(
        queue_envelope("physical-1", keyed_payload(first, second))
    )

    assert result is None
    assert [(call.telemetry_id, call.root_id) for call in tracker.failed_calls] == [
        ("tm-1", first.id),
        ("tm-2", second.id),
    ]
    assert tracker.semantic_done_calls == set()
    assert tracker.requeue_telemetry_ids == []
    assert SemanticProcessor.consume_request_stats("tm-1").error_count == 1
    assert SemanticProcessor.consume_request_stats("tm-2").error_count == 1


@pytest.mark.asyncio
async def test_malformed_batch_is_reported_for_ack_without_retry(monkeypatch):
    tracker = _WaitTracker()
    errors = []
    processor = SemanticProcessor()
    processor.set_callbacks(lambda: None, lambda: None, lambda error, data: errors.append(error))
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_request_wait_tracker",
        lambda: tracker,
    )

    result = await processor.on_dequeue(
        queue_envelope(
            "physical-1",
            {
                "_queuefs_keyed_batch": {
                    "schema_version": 1,
                    "dispatch_key": "route",
                    "merge_signature": "signature",
                    "contributions": ["not-json"],
                }
            },
        )
    )

    assert result is None
    assert len(errors) == 1
    assert tracker.semantic_done_calls == set()
    assert tracker.failed_calls == []
    assert tracker.requeue_telemetry_ids == []
