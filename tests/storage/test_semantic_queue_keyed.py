# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for SemanticQueue keyed-batch routing."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.storage.queuefs.named_queue import NamedQueue
from openviking.storage.queuefs.semantic_batch import SemanticBatch, semantic_batch_route
from openviking.storage.queuefs.semantic_msg import SemanticMsg, build_semantic_coalesce_key
from openviking.storage.queuefs.semantic_queue import SemanticQueue


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


def test_different_peer_identity_cannot_share_keyed_batch_route():
    first = eligible_msg(
        peer_id="peer-a",
        coalesce_key=build_semantic_coalesce_key(
            context_type="resource",
            uri="viking://resources/docs",
            account_id="account",
            user_id="user",
            peer_id="peer-a",
        ),
    )
    second = eligible_msg(
        peer_id="peer-b",
        coalesce_key=build_semantic_coalesce_key(
            context_type="resource",
            uri="viking://resources/docs",
            account_id="account",
            user_id="user",
            peer_id="peer-b",
        ),
    )

    first_route = semantic_batch_route(first, task_owned=False)
    second_route = semantic_batch_route(second, task_owned=False)

    assert first_route is not None
    assert second_route is not None
    assert first_route.dispatch_key != second_route.dispatch_key
    with pytest.raises(ValueError, match="semantic batch route mismatch"):
        SemanticBatch.from_contributions([first, second])
