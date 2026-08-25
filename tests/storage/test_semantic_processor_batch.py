# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Batch lifecycle tests for semantic queue processing."""

import hashlib
import json
import logging
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


class _RealDagVikingFS:
    def __init__(self, *, write_error=None):
        self._async_agfs = self
        self._write_error = write_error
        self.contents = {A_URI: "content"}

    def _uri_to_path(self, uri, ctx=None):
        del ctx
        return uri

    async def ls(self, uri, node_limit=None, ctx=None):
        del node_limit, ctx
        return [{"name": "a.md", "isDir": False}] if uri == ROOT_URI else []

    async def stat(self, uri, ctx=None):
        del ctx
        return {"size": len(self.contents.get(uri, ""))}

    async def read_file(self, uri, ctx=None):
        del ctx
        return self.contents.get(uri, "")

    async def write_file(self, uri, content, ctx=None, lease_ref=None):
        del ctx, lease_ref
        if self._write_error is not None:
            raise self._write_error
        self.contents[uri] = content

    async def pathlock_acquire_exact_batch(self, paths):
        return {"paths": paths}

    async def pathlock_release(self, lease):
        del lease


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
        self.failed_calls.append(
            SimpleNamespace(telemetry_id=telemetry_id, root_id=root_id, message=message)
        )


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
async def test_same_path_add_then_modify_reaches_processor_with_both_embedding_owners(
    monkeypatch,
):
    first = eligible_msg(telemetry_id="tm-add", changes={"added": [A_URI]})
    second = eligible_msg(telemetry_id="tm-modify", changes={"modified": [A_URI]})
    tracker = _WaitTracker()
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

    await processor.on_dequeue(queue_envelope("physical-fold", keyed_payload(first, second)))

    assert _FakeExecutor.kwargs["changes"] == {"added": [A_URI]}
    assert _FakeExecutor.kwargs["file_contributions"] == {A_URI: (first, second)}
    assert tracker.semantic_done_calls == {
        ("tm-add", first.id),
        ("tm-modify", second.id),
    }


@pytest.mark.asyncio
async def test_keyed_batch_logs_bounded_start_and_end_fields_without_sensitive_data(
    monkeypatch, caplog
):
    first = eligible_msg(
        telemetry_id="sensitive-telemetry-1",
        coalesce_key="sensitive-coalesce-key",
        changes={"modified": [f"{ROOT_URI}/sensitive-a.md"]},
    )
    second = eligible_msg(
        telemetry_id="sensitive-telemetry-2",
        coalesce_key="sensitive-coalesce-key",
        changes={"modified": [f"{ROOT_URI}/sensitive-b.md"]},
    )
    payload = keyed_payload(first, second)
    dispatch_key = payload["_queuefs_keyed_batch"]["dispatch_key"]
    tracker = _WaitTracker()
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
    semantic_processor_module.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger=semantic_processor_module.logger.name):
            await processor.on_dequeue(queue_envelope("physical-1", payload))
    finally:
        semantic_processor_module.logger.removeHandler(caplog.handler)

    records = [
        record
        for record in caplog.records
        if getattr(record, "event", "").startswith("semantic.keyed_batch_")
    ]
    assert [record.event for record in records] == [
        "semantic.keyed_batch_started",
        "semantic.keyed_batch_completed",
    ]
    expected_hash = hashlib.sha256(dispatch_key.encode("utf-8")).hexdigest()[:12]
    for record in records:
        assert record.physical_id == "physical-1"
        assert record.dispatch_hash_prefix == expected_hash
        assert record.contribution_count == 2
        assert record.merged_path_count == 2
        rendered = f"{record.getMessage()} {record.__dict__!r}"
        for sensitive in [
            dispatch_key,
            "sensitive-coalesce-key",
            "sensitive-telemetry-1",
            "sensitive-telemetry-2",
            "sensitive-a.md",
            "sensitive-b.md",
        ]:
            assert sensitive not in rendered


@pytest.mark.asyncio
async def test_keyed_root_skip_logs_completed_once_without_running_dag(monkeypatch, caplog):
    root = eligible_msg(
        uri="viking://",
        telemetry_id="tm-root",
        coalesce_key="resource|account|user|peer|viking://",
        changes={"modified": ["viking://root.md"]},
    )
    payload = keyed_payload(root)
    tracker = _WaitTracker()
    _FakeExecutor.construct_count = 0
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticDagExecutor",
        _FakeExecutor,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_request_wait_tracker",
        lambda: tracker,
    )
    processor = SemanticProcessor()
    semantic_processor_module.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger=semantic_processor_module.logger.name):
            await processor.on_dequeue(queue_envelope("physical-root", payload))
    finally:
        semantic_processor_module.logger.removeHandler(caplog.handler)

    records = [
        record
        for record in caplog.records
        if getattr(record, "event", "").startswith("semantic.keyed_batch_")
    ]
    assert [record.event for record in records] == [
        "semantic.keyed_batch_started",
        "semantic.keyed_batch_completed",
    ]
    assert [record.physical_id for record in records] == [
        "physical-root",
        "physical-root",
    ]
    assert _FakeExecutor.construct_count == 0
    assert tracker.semantic_done_calls == {("tm-root", root.id)}


@pytest.mark.asyncio
async def test_normal_keyed_success_preserves_settlement_and_completion_order(monkeypatch):
    message = eligible_msg(telemetry_id="tm-order")
    tracker = _WaitTracker()
    events = []

    class _RecordingCircuitBreaker:
        def check(self):
            return None

        def record_success(self):
            events.append("circuit_breaker.record_success")

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
    original_reset = semantic_processor_module.reset_root_observability_context

    def recording_reset(token):
        events.append("reset_root_observability_context")
        original_reset(token)

    monkeypatch.setattr(
        semantic_processor_module,
        "reset_root_observability_context",
        recording_reset,
    )
    monkeypatch.setattr(
        semantic_processor_module,
        "_log_keyed_batch_event",
        lambda event, data, work: events.append(event),
    )
    processor = SemanticProcessor()
    processor._circuit_breaker = _RecordingCircuitBreaker()
    processor._enqueue_parent_refresh = AsyncMock()
    processor.set_callbacks(
        lambda: events.append("report_success"),
        lambda: events.append("report_requeue"),
        lambda error, data: events.append("report_error"),
    )

    await processor.on_dequeue(queue_envelope("physical-order", keyed_payload(message)))

    assert events == [
        "semantic.keyed_batch_started",
        "report_success",
        "circuit_breaker.record_success",
        "reset_root_observability_context",
        "semantic.keyed_batch_completed",
    ]
    assert tracker.semantic_done_calls == {("tm-order", message.id)}


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
async def test_permanent_batch_error_fails_all_contributions_and_returns_for_ack(
    monkeypatch, caplog
):
    first = eligible_msg(
        telemetry_id="sensitive-telemetry-1",
        coalesce_key="sensitive-coalesce-key",
        changes={"modified": [f"{ROOT_URI}/sensitive-a.md"]},
    )
    second = eligible_msg(
        telemetry_id="sensitive-telemetry-2",
        coalesce_key="sensitive-coalesce-key",
        changes={"modified": [f"{ROOT_URI}/sensitive-b.md"]},
    )
    tracker = _WaitTracker()
    _FailingExecutor.error = PermissionError(
        "invalid model input for viking://resources/docs/sensitive-a.md"
    )
    reported_errors = []
    root_attrs = []
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
    monkeypatch.setattr(
        semantic_processor_module,
        "bind_root_observability_context",
        lambda attrs: root_attrs.append(attrs) or object(),
    )
    monkeypatch.setattr(
        semantic_processor_module, "reset_root_observability_context", lambda _: None
    )
    processor = SemanticProcessor()
    processor.set_callbacks(
        lambda: None,
        lambda: None,
        lambda error, data: reported_errors.append((error, data)),
    )
    physical_id = "physical-" + "p" * 300

    semantic_processor_module.logger.addHandler(caplog.handler)
    try:
        result = await processor.on_dequeue(
            queue_envelope(physical_id, keyed_payload(first, second))
        )
    finally:
        semantic_processor_module.logger.removeHandler(caplog.handler)

    assert result is None
    assert {(call.telemetry_id, call.root_id) for call in tracker.failed_calls} == {
        ("sensitive-telemetry-1", first.id),
        ("sensitive-telemetry-2", second.id),
    }
    assert {call.message for call in tracker.failed_calls} == {"PermissionError"}
    assert reported_errors == [
        (
            "PermissionError",
            {
                "physical_id": physical_id[:128],
                "dispatch_hash_prefix": hashlib.sha256(
                    keyed_payload(first, second)["_queuefs_keyed_batch"]["dispatch_key"].encode()
                ).hexdigest()[:12],
                "contribution_count": 2,
                "error_class": "PermissionError",
            },
        )
    ]
    assert len(root_attrs) == 1
    assert root_attrs[0].request_id == physical_id[:128]
    assert root_attrs[0].url_path == ""
    assert root_attrs[0].account_id == ""
    assert root_attrs[0].user_id == ""
    rendered_logs = "\n".join(
        f"{record.getMessage()} {record.__dict__!r}" for record in caplog.records
    )
    for sensitive in (
        "sensitive-coalesce-key",
        "sensitive-telemetry-1",
        "sensitive-telemetry-2",
        "sensitive-a.md",
        "sensitive-b.md",
        physical_id,
    ):
        assert sensitive not in rendered_logs
    assert tracker.semantic_done_calls == set()
    assert tracker.requeue_telemetry_ids == []
    assert SemanticProcessor.consume_request_stats("sensitive-telemetry-1").error_count == 1
    assert SemanticProcessor.consume_request_stats("sensitive-telemetry-2").error_count == 1


@pytest.mark.asyncio
async def test_real_dag_summary_timeout_reaches_transient_batch_classifier(monkeypatch):
    first = eligible_msg(telemetry_id="tm-real-1", skip_vectorization=True)
    second = eligible_msg(telemetry_id="tm-real-2", skip_vectorization=True)
    tracker = _WaitTracker()
    viking_fs = _RealDagVikingFS()
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs", lambda: viking_fs
    )
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: viking_fs)
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_dag.get_openviking_config",
        lambda: SimpleNamespace(semantic=SimpleNamespace(overview_sample_limit=32)),
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
    processor._generate_single_file_summary = AsyncMock(
        side_effect=TimeoutError("summary provider timeout")
    )
    processor._enqueue_parent_refresh = AsyncMock()

    with pytest.raises(QueueMessageRetry):
        await processor.on_dequeue(
            queue_envelope("physical-real-transient", keyed_payload(first, second))
        )

    assert tracker.requeue_telemetry_ids == ["tm-real-1", "tm-real-2"]
    assert tracker.failed_calls == []
    assert tracker.semantic_done_calls == set()


@pytest.mark.asyncio
async def test_real_dag_write_permission_error_reaches_permanent_batch_classifier(monkeypatch):
    first = eligible_msg(telemetry_id="tm-real-1", skip_vectorization=True)
    second = eligible_msg(telemetry_id="tm-real-2", skip_vectorization=True)
    tracker = _WaitTracker()
    error = PermissionError("directory sidecar denied")
    viking_fs = _RealDagVikingFS(write_error=error)
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs", lambda: viking_fs
    )
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: viking_fs)
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_dag.get_openviking_config",
        lambda: SimpleNamespace(semantic=SimpleNamespace(overview_sample_limit=32)),
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
    processor._generate_single_file_summary = AsyncMock(
        return_value={"name": "a.md", "summary": "summary"}
    )
    processor._generate_overview = AsyncMock(return_value="overview")
    processor._normalize_overview_generation = lambda generated: (generated, "abstract")
    processor._enqueue_parent_refresh = AsyncMock()

    result = await processor.on_dequeue(
        queue_envelope("physical-real-permanent", keyed_payload(first, second))
    )

    assert result is None
    assert {(call.telemetry_id, call.root_id) for call in tracker.failed_calls} == {
        ("tm-real-1", first.id),
        ("tm-real-2", second.id),
    }
    assert tracker.requeue_telemetry_ids == []
    assert tracker.semantic_done_calls == set()


@pytest.mark.asyncio
async def test_open_circuit_requests_retry_after_breaker_delay(monkeypatch, caplog):
    message = eligible_msg(
        telemetry_id="sensitive-circuit-telemetry",
        coalesce_key="sensitive-circuit-coalesce-key",
        changes={"modified": [f"{ROOT_URI}/sensitive-circuit.md"]},
    )
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

    semantic_processor_module.logger.addHandler(caplog.handler)
    try:
        with pytest.raises(QueueMessageRetry) as caught:
            await processor.on_dequeue(queue_envelope("physical-1", keyed_payload(message)))
    finally:
        semantic_processor_module.logger.removeHandler(caplog.handler)

    assert caught.value.delay_seconds == 3.5
    assert _FakeExecutor.construct_count == 0
    replacement_enqueue.assert_not_awaited()
    assert tracker.requeue_telemetry_ids == ["sensitive-circuit-telemetry"]
    rendered_logs = "\n".join(
        f"{record.getMessage()} {record.__dict__!r}" for record in caplog.records
    )
    for sensitive in (
        "sensitive-circuit-telemetry",
        "sensitive-circuit-coalesce-key",
        "sensitive-circuit.md",
    ):
        assert sensitive not in rendered_logs
    assert SemanticProcessor.consume_request_stats("sensitive-circuit-telemetry").requeue_count == 1


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
