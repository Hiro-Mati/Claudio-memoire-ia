# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for SemanticQueue keyed-batch routing and legacy bootstrap."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.storage.queuefs.named_queue import NamedQueue
from openviking.storage.queuefs.semantic_batch import semantic_batch_route
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.queuefs.semantic_queue import (
    SemanticQueue,
    is_semantic_coalesce_stale,
)


def semantic_queue_fixture() -> SemanticQueue:
    return SemanticQueue(MagicMock(), "/queue", "Semantic")


def eligible_msg(**overrides) -> SemanticMsg:
    values = {
        "uri": "viking://resources/docs",
        "context_type": "resource",
        "recursive": False,
        "account_id": "account",
        "user_id": "user",
        "peer_id": "peer",
        "coalesce_key": "resource|account|user|peer|viking://resources/docs",
        "changes": {"modified": ["viking://resources/docs/a.md"]},
        "aggregate_directory": True,
    }
    values.update(overrides)
    return SemanticMsg(**values)


def encode_queuefs_batch(messages: list[SemanticMsg]) -> str:
    route = semantic_batch_route(messages[0], task_owned=False)
    assert route is not None
    return json.dumps(
        {
            "_queuefs_keyed_batch": {
                "schema_version": 1,
                "dispatch_key": route.dispatch_key,
                "merge_signature": route.merge_signature,
                "contributions": [message.to_json() for message in messages],
            }
        }
    )


@pytest.mark.asyncio
async def test_eligible_semantic_message_uses_keyed_enqueue(monkeypatch):
    queue = semantic_queue_fixture()
    keyed = AsyncMock(return_value="stored")
    normal = AsyncMock(return_value="normal")
    monkeypatch.setattr(NamedQueue, "enqueue_keyed", keyed)
    monkeypatch.setattr(NamedQueue, "enqueue", normal)
    msg = eligible_msg(coalesce_version=42)
    route = semantic_batch_route(msg, task_owned=False)
    assert route is not None

    assert await queue.enqueue(msg) == "stored"
    assert msg.coalesce_version == 0
    keyed.assert_awaited_once_with(
        msg.to_dict(),
        dispatch_key=route.dispatch_key,
        merge_signature=route.merge_signature,
    )
    normal.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"lock_handoff": {"token": "x"}},
        {"context_type": "memory"},
        {"recursive": True},
        {"target_uri": "viking://resources/target"},
        {"aggregate_directory": False},
        {"use_hierarchical_aggregation": True},
        {"changes": {}},
    ],
)
async def test_ineligible_semantic_message_keeps_normal_enqueue(monkeypatch, overrides):
    queue = semantic_queue_fixture()
    keyed = AsyncMock(return_value="stored")
    normal = AsyncMock(return_value="normal")
    monkeypatch.setattr(NamedQueue, "enqueue_keyed", keyed)
    monkeypatch.setattr(NamedQueue, "enqueue", normal)

    assert await queue.enqueue(eligible_msg(**overrides)) == "normal"
    normal.assert_awaited_once()
    keyed.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_owned_semantic_message_keeps_normal_enqueue(monkeypatch):
    queue = semantic_queue_fixture()
    keyed = AsyncMock(return_value="stored")
    normal = AsyncMock(return_value="normal")
    monkeypatch.setattr(NamedQueue, "enqueue_keyed", keyed)
    monkeypatch.setattr(NamedQueue, "enqueue", normal)
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_queue.get_task_context",
        lambda: object(),
    )

    assert await queue.enqueue(eligible_msg()) == "normal"
    normal.assert_awaited_once()
    keyed.assert_not_awaited()


def test_bootstrap_legacy_versions_reads_single_and_batch_contributions():
    queue = semantic_queue_fixture()
    coalesce_key = "resource|account|user|peer|viking://resources/bootstrap"
    old = eligible_msg(coalesce_key=coalesce_key, coalesce_version=7)
    newer = eligible_msg(coalesce_key=coalesce_key, coalesce_version=11)
    snapshot = [
        {"id": "p1", "data": old.to_json()},
        {"id": "p2", "data": encode_queuefs_batch([newer])},
        {"id": "bad", "data": "not-json"},
    ]

    queue.bootstrap_legacy_coalesce(snapshot)

    assert is_semantic_coalesce_stale(coalesce_key, 10)
    assert not is_semantic_coalesce_stale(coalesce_key, 11)
