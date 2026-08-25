# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Pure data models and codecs for semantic keyed batches.

This module deliberately contains no QueueFS or other I/O.  Queue consumers can
use it to decide whether a message is safe to batch, fold file changes, and
validate a keyed-batch payload before handing it to the processor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .semantic_msg import SemanticMsg

IDENTITY_FIELDS = frozenset({"account_id", "user_id", "peer_id"})
SIGNATURE_FIELDS = frozenset(
    {
        "context_type",
        "uri",
        "target_uri",
        "recursive",
        "role",
        "skip_vectorization",
        "is_code_repo",
        "target_preexisting",
        "source",
        "aggregate_directory",
        "use_hierarchical_aggregation",
        "propagate_to_parent",
    }
)
CONTRIBUTION_FIELDS = frozenset(
    {
        "id",
        "status",
        "timestamp",
        "telemetry_id",
        "ingest_options",
        "coalesce_key",
        "coalesce_version",
        "changes",
        "generation_trigger",
    }
)
GUARD_FIELDS = frozenset({"lock_handoff"})
ALL_CLASSIFIED_FIELDS = IDENTITY_FIELDS | SIGNATURE_FIELDS | CONTRIBUTION_FIELDS | GUARD_FIELDS


@dataclass(frozen=True)
class SemanticBatchRoute:
    """Stable dispatch and execution identifiers for one batchable message."""

    dispatch_key: str
    merge_signature: str


def _canonical_field_value(msg: SemanticMsg, name: str) -> Any:
    value = getattr(msg, name)
    if name in {"uri", "target_uri"}:
        # An empty target remains empty; rstrip() would also preserve that, but
        # spelling this out documents the distinction and avoids surprises for
        # non-string malformed values.
        return value.rstrip("/") if value else ""
    if name == "ingest_options":
        return msg.ingest_options.to_dict()
    return value


def build_semantic_dispatch_key(msg: SemanticMsg) -> str:
    """Build the opaque keyed-dispatch identifier from the coalesce key."""

    digest = hashlib.sha256(msg.coalesce_key.encode("utf-8")).hexdigest()
    return f"semantic-v1:{digest}"


def build_semantic_execution_signature(msg: SemanticMsg) -> str:
    """Hash fields that must agree before contributions share execution."""

    values = {name: _canonical_field_value(msg, name) for name in sorted(SIGNATURE_FIELDS)}
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _has_changes(msg: SemanticMsg) -> bool:
    changes = msg.changes
    if not isinstance(changes, Mapping):
        return False
    return any(changes.get(kind) for kind in ("added", "modified", "deleted"))


def semantic_batch_route(msg: SemanticMsg, *, task_owned: bool) -> Optional[SemanticBatchRoute]:
    """Return a route only for messages safe to execute as keyed batches."""

    if task_owned:
        return None
    if msg.lock_handoff is not None:
        return None
    if msg.context_type == "memory":
        return None
    if msg.recursive:
        return None
    if not _has_changes(msg):
        return None
    if msg.target_uri:
        return None
    if not msg.aggregate_directory:
        return None
    if msg.use_hierarchical_aggregation:
        return None
    if not msg.coalesce_key:
        return None
    return SemanticBatchRoute(
        dispatch_key=build_semantic_dispatch_key(msg),
        merge_signature=build_semantic_execution_signature(msg),
    )


def merge_semantic_changes(
    contributions: Sequence[SemanticMsg],
) -> tuple[dict[str, list[str]], dict[str, tuple[SemanticMsg, ...]]]:
    """Fold changes by path while retaining the contribution that is current.

    A path's last observed state wins.  Path order follows first observation so
    folding does not make otherwise identical batches fluctuate in ordering.
    """

    state: dict[str, tuple[str, SemanticMsg]] = {}
    path_order: list[str] = []
    for msg in contributions:
        changes = msg.changes
        if not isinstance(changes, Mapping):
            continue
        for kind in ("added", "modified", "deleted"):
            paths = changes.get(kind) or ()
            for path in paths:
                if path not in state:
                    path_order.append(path)
                state[path] = (kind, msg)

    merged: dict[str, list[str]] = {}
    live: dict[str, tuple[SemanticMsg, ...]] = {}
    for path in path_order:
        kind, msg = state[path]
        merged.setdefault(kind, []).append(path)
        live[path] = (msg,)
    return merged, live


@dataclass(frozen=True)
class SemanticBatch:
    """Validated same-route semantic contributions and their folded changes."""

    dispatch_key: str
    merge_signature: str
    contributions: tuple[SemanticMsg, ...]
    changes: dict[str, list[str]]
    live_contributions: dict[str, tuple[SemanticMsg, ...]]

    @classmethod
    def from_contributions(cls, contributions: Sequence[SemanticMsg]) -> "SemanticBatch":
        if not contributions:
            raise ValueError("semantic batch has no contributions")
        routes = [semantic_batch_route(msg, task_owned=False) for msg in contributions]
        if any(route is None for route in routes):
            raise ValueError("semantic batch contains an ineligible contribution")
        route_pairs = {(route.dispatch_key, route.merge_signature) for route in routes if route}
        if len(route_pairs) != 1:
            raise ValueError("semantic batch route mismatch")
        changes, live = merge_semantic_changes(contributions)
        route = routes[0]
        assert route is not None
        return cls(route.dispatch_key, route.merge_signature, tuple(contributions), changes, live)


def decode_keyed_batch_payload(payload: Mapping[str, Any]) -> SemanticBatch:
    """Strictly parse and validate a QueueFS keyed-batch payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("semantic keyed batch payload must be an object")
    wrapper = payload.get("_queuefs_keyed_batch")
    if not isinstance(wrapper, Mapping):
        raise ValueError("semantic keyed batch wrapper must be an object")
    schema_version = wrapper.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("unsupported semantic keyed batch schema version")
    encoded_contributions = wrapper.get("contributions")
    if not isinstance(encoded_contributions, list) or not 1 <= len(encoded_contributions) <= 1024:
        raise ValueError("semantic keyed batch contributions must contain 1..1024 items")
    if any(not isinstance(item, str) for item in encoded_contributions):
        raise ValueError("semantic keyed batch contributions must be strings")

    messages: list[SemanticMsg] = []
    for encoded in encoded_contributions:
        try:
            messages.append(SemanticMsg.from_json(encoded))
        except Exception as exc:
            raise ValueError("invalid semantic keyed batch contribution") from exc
    batch = SemanticBatch.from_contributions(messages)

    if wrapper.get("dispatch_key") != batch.dispatch_key:
        raise ValueError("semantic keyed batch dispatch key mismatch")
    if wrapper.get("merge_signature") != batch.merge_signature:
        raise ValueError("semantic keyed batch merge signature mismatch")
    return batch


def iter_semantic_messages_from_queue_envelope(
    envelope: Mapping[str, Any],
) -> tuple[SemanticMsg, ...]:
    """Decode a QueueFS envelope into one ordinary message or batch members."""

    if not isinstance(envelope, Mapping) or not isinstance(envelope.get("data"), str):
        raise ValueError("semantic QueueFS envelope data must be JSON text")
    try:
        payload = json.loads(envelope["data"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid semantic QueueFS envelope JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("semantic QueueFS payload must be an object")
    if "_queuefs_keyed_batch" in payload:
        return decode_keyed_batch_payload(payload).contributions
    try:
        return (SemanticMsg.from_dict(dict(payload)),)
    except Exception as exc:
        raise ValueError("invalid semantic QueueFS message") from exc
