# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for the pure semantic keyed-batch model."""

import copy
import json

import pytest

from openviking.storage.queuefs.semantic_batch import (
    SemanticBatch,
    decode_keyed_batch_payload,
    iter_semantic_messages_from_queue_envelope,
    semantic_batch_route,
)
from openviking.storage.queuefs.semantic_msg import SemanticMsg


def eligible_msg(**overrides):
    values = {
        "uri": "viking://resources/docs",
        "context_type": "resource",
        "recursive": False,
        "account_id": "a",
        "user_id": "u",
        "peer_id": "u",
        "coalesce_key": "resource|a|u|u|viking://resources/docs",
        "changes": {"modified": ["viking://resources/docs/a.md"]},
        "aggregate_directory": True,
    }
    values.update(overrides)
    return SemanticMsg(**values)


def encode_test_batch(route, contributions):
    return {
        "_queuefs_keyed_batch": {
            "schema_version": 1,
            "dispatch_key": route.dispatch_key,
            "merge_signature": route.merge_signature,
            "contributions": contributions,
        }
    }


@pytest.mark.parametrize("context_type", ["resource", "skill"])
def test_route_accepts_supported_contexts_with_normalized_matching_target(context_type):
    msg = eligible_msg(
        context_type=context_type,
        uri="viking://resources/docs///",
        target_uri="viking://resources/docs",
    )

    assert semantic_batch_route(msg, task_owned=False) is not None


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_type": "memory"},
        {"context_type": "session"},
        {"recursive": True},
        {"changes": {}},
        {"changes": {"modified": []}},
        {"aggregate_directory": False},
        {"use_hierarchical_aggregation": True},
    ],
)
def test_route_rejects_non_batchable_semantic_messages(overrides):
    assert semantic_batch_route(eligible_msg(**overrides), task_owned=False) is None


@pytest.mark.parametrize(
    ("kinds", "expected_changes", "expected_live"),
    [
        (
            ("added", "modified"),
            {"added": ["viking://resources/docs/a.md"]},
            ["tm-0", "tm-1"],
        ),
        (
            ("modified", "deleted"),
            {"deleted": ["viking://resources/docs/a.md"]},
            [],
        ),
        (
            ("deleted", "modified"),
            {"modified": ["viking://resources/docs/a.md"]},
            ["tm-1"],
        ),
        (
            ("deleted", "added", "modified"),
            {"added": ["viking://resources/docs/a.md"]},
            ["tm-1", "tm-2"],
        ),
    ],
)
def test_change_fold_applies_transitions_and_tracks_every_live_owner(
    kinds, expected_changes, expected_live
):
    path = "viking://resources/docs/a.md"
    contributions = [
        eligible_msg(telemetry_id=f"tm-{index}", changes={kind: [path]})
        for index, kind in enumerate(kinds)
    ]

    batch = SemanticBatch.from_contributions(contributions)

    assert batch.changes == expected_changes
    assert [
        contribution.telemetry_id for contribution in batch.live_contributions.get(path, ())
    ] == expected_live


def test_change_fold_unions_distinct_paths_and_preserves_original_changes():
    first = eligible_msg(
        telemetry_id="tm-a",
        changes={"modified": ["viking://resources/docs/a.md"]},
    )
    second = eligible_msg(
        telemetry_id="tm-b",
        changes={"modified": ["viking://resources/docs/b.md"]},
    )
    first_changes = copy.deepcopy(first.changes)
    second_changes = copy.deepcopy(second.changes)

    batch = SemanticBatch.from_contributions([first, second])

    assert batch.changes == {
        "modified": [
            "viking://resources/docs/a.md",
            "viking://resources/docs/b.md",
        ]
    }
    assert batch.contributions == (first, second)
    assert first.changes == first_changes
    assert second.changes == second_changes
    assert batch.live_contributions == {
        "viking://resources/docs/a.md": (first,),
        "viking://resources/docs/b.md": (second,),
    }


def test_keyed_payload_rejects_signature_mismatch():
    msg = eligible_msg()
    route = semantic_batch_route(msg, task_owned=False)
    assert route is not None
    payload = encode_test_batch(route, [msg.to_json()])
    payload["_queuefs_keyed_batch"]["merge_signature"] = "sha256:wrong"
    with pytest.raises(ValueError, match="merge signature"):
        decode_keyed_batch_payload(payload)


def test_keyed_payload_decodes_contributions_and_queue_envelope():
    first = eligible_msg(telemetry_id="tm-a")
    second = eligible_msg(telemetry_id="tm-b")
    route = semantic_batch_route(first, task_owned=False)
    assert route is not None
    payload = encode_test_batch(route, [first.to_json(), second.to_json()])
    decoded = decode_keyed_batch_payload(payload)
    assert decoded.contributions == (first, second)

    envelope = {"data": json.dumps(payload)}
    assert iter_semantic_messages_from_queue_envelope(envelope) == (first, second)


def test_queue_envelope_decodes_one_ordinary_message():
    msg = eligible_msg()
    assert iter_semantic_messages_from_queue_envelope({"data": msg.to_json()}) == (msg,)


@pytest.mark.parametrize(
    "payload",
    [
        {"_queuefs_keyed_batch": {"schema_version": 2, "contributions": []}},
        {
            "_queuefs_keyed_batch": {
                "schema_version": 1,
                "dispatch_key": "x",
                "merge_signature": "y",
                "contributions": [],
            }
        },
        {
            "_queuefs_keyed_batch": {
                "schema_version": 1,
                "dispatch_key": "x",
                "merge_signature": "y",
                "contributions": [123],
            }
        },
    ],
)
def test_keyed_payload_rejects_invalid_wrapper(payload):
    with pytest.raises(ValueError):
        decode_keyed_batch_payload(payload)
