# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Core keyed-batch processing regressions."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.storage.queuefs.semantic_batch import semantic_batch_route
from openviking.storage.queuefs.semantic_dag import DagStats
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.queuefs.semantic_processor import SemanticProcessor

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

    def mark_semantic_done(self, telemetry_id, root_id, processed_delta=1):
        assert processed_delta == 1
        self.semantic_done_calls.add((telemetry_id, root_id))


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
