# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for the pure semantic keyed-batch model."""

import copy
import json
from dataclasses import fields

import pytest

from openviking.storage.queuefs.semantic_batch import (
    ALL_CLASSIFIED_FIELDS,
    CONTRIBUTION_FIELDS,
    GUARD_FIELDS,
    IDENTITY_FIELDS,
    MAX_DISPATCH_KEY_BYTES,
    MAX_KEYED_BATCH_CONTRIBUTIONS,
    MAX_MERGE_SIGNATURE_BYTES,
    SIGNATURE_FIELDS,
    SemanticBatch,
    SemanticBatchRoute,
    build_semantic_dispatch_key,
    build_semantic_execution_signature,
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


def test_every_semantic_field_has_an_explicit_batch_classification():
    assert ALL_CLASSIFIED_FIELDS == {field.name for field in fields(SemanticMsg)}
    assert sum(
        map(len, (IDENTITY_FIELDS, SIGNATURE_FIELDS, CONTRIBUTION_FIELDS, GUARD_FIELDS))
    ) == len(ALL_CLASSIFIED_FIELDS)


def test_route_is_stable_and_excludes_unsafe_messages():
    first = eligible_msg(telemetry_id="tm-1")
    second = eligible_msg(telemetry_id="tm-2", generation_trigger="content_delete")
    assert build_semantic_dispatch_key(first) == build_semantic_dispatch_key(second)
    assert build_semantic_execution_signature(first) == build_semantic_execution_signature(second)
    assert semantic_batch_route(first, task_owned=False) is not None
    assert semantic_batch_route(eligible_msg(lock_handoff={"token": "x"}), task_owned=False) is None
    assert semantic_batch_route(eligible_msg(), task_owned=True) is None
    assert (
        semantic_batch_route(eligible_msg(target_uri="viking://resources/other"), task_owned=False)
        is None
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_type": "memory"},
        {"recursive": True},
        {"changes": {}},
        {"changes": {"modified": []}},
        {"aggregate_directory": False},
        {"use_hierarchical_aggregation": True},
    ],
)
def test_route_rejects_non_batchable_semantic_messages(overrides):
    assert semantic_batch_route(eligible_msg(**overrides), task_owned=False) is None


def test_route_normalizes_uri_fields_but_keeps_empty_target_empty():
    msg = eligible_msg(uri="viking://resources/docs///", target_uri="")
    other = eligible_msg(uri="viking://resources/docs", target_uri="")
    assert build_semantic_execution_signature(msg) == build_semantic_execution_signature(other)


def test_change_fold_uses_last_state_and_keeps_live_contributions():
    a = eligible_msg(telemetry_id="tm-a", changes={"added": ["viking://resources/docs/a.md"]})
    b = eligible_msg(telemetry_id="tm-b", changes={"modified": ["viking://resources/docs/a.md"]})
    c = eligible_msg(telemetry_id="tm-c", changes={"deleted": ["viking://resources/docs/a.md"]})
    d = eligible_msg(telemetry_id="tm-d", changes={"added": ["viking://resources/docs/a.md"]})
    batch = SemanticBatch.from_contributions([a, b, c, d])
    assert batch.changes == {"added": ["viking://resources/docs/a.md"]}
    assert [
        item.telemetry_id for item in batch.live_contributions["viking://resources/docs/a.md"]
    ] == ["tm-d"]


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


def test_protocol_limits_are_named_and_contribution_limit_is_enforced():
    assert MAX_KEYED_BATCH_CONTRIBUTIONS == 1024
    assert MAX_DISPATCH_KEY_BYTES > 0
    assert MAX_MERGE_SIGNATURE_BYTES > 0

    msg = eligible_msg()
    route = semantic_batch_route(msg, task_owned=False)
    assert route is not None
    payload = encode_test_batch(route, [msg.to_json()] * (MAX_KEYED_BATCH_CONTRIBUTIONS + 1))
    with pytest.raises(ValueError, match="1..1024"):
        decode_keyed_batch_payload(payload)


def test_protocol_identifiers_use_non_empty_utf8_byte_caps():
    over_dispatch = "é" * (MAX_DISPATCH_KEY_BYTES // 2 + 1)
    over_signature = "é" * (MAX_MERGE_SIGNATURE_BYTES // 2 + 1)

    with pytest.raises(ValueError, match="dispatch key"):
        SemanticBatchRoute(over_dispatch, "signature")
    with pytest.raises(ValueError, match="merge signature"):
        SemanticBatchRoute("dispatch", over_signature)
    with pytest.raises(ValueError, match="dispatch key"):
        SemanticBatchRoute("", "signature")
    with pytest.raises(ValueError, match="merge signature"):
        SemanticBatchRoute("dispatch", "")


def test_keyed_payload_rejects_oversized_utf8_wrapper_identifiers():
    msg = eligible_msg()
    route = semantic_batch_route(msg, task_owned=False)
    assert route is not None

    dispatch_payload = encode_test_batch(route, [msg.to_json()])
    dispatch_payload["_queuefs_keyed_batch"]["dispatch_key"] = "é" * (
        MAX_DISPATCH_KEY_BYTES // 2 + 1
    )
    with pytest.raises(ValueError, match="dispatch key"):
        decode_keyed_batch_payload(dispatch_payload)

    signature_payload = encode_test_batch(route, [msg.to_json()])
    signature_payload["_queuefs_keyed_batch"]["merge_signature"] = "é" * (
        MAX_MERGE_SIGNATURE_BYTES // 2 + 1
    )
    with pytest.raises(ValueError, match="merge signature"):
        decode_keyed_batch_payload(signature_payload)


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
