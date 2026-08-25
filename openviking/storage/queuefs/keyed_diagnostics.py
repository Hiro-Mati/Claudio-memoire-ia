# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Bounded diagnostics for opaque QueueFS keyed semantic batches."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

MAX_PHYSICAL_ID_BYTES = 128
MAX_ERROR_CLASS_BYTES = 64
MAX_DIAGNOSTIC_CONTRIBUTIONS = 1024
MAX_DIAGNOSTIC_PATHS = 1_048_576
DISPATCH_HASH_PREFIX_LENGTH = 12

_ERROR_CLASS_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_current_diagnostic: ContextVar[Optional["KeyedBatchDiagnostic"]] = ContextVar(
    "semantic_keyed_batch_diagnostic", default=None
)


def _bounded_utf8(value: Any, max_bytes: int) -> str:
    encoded = str(value or "").encode("utf-8", errors="replace")[:max_bytes]
    return encoded.decode("utf-8", errors="ignore")


def bounded_error_class(error: BaseException | str) -> str:
    """Return a bounded class label, never an exception message."""

    if isinstance(error, BaseException):
        candidate = type(error).__name__
    else:
        candidate = str(error).split(":", 1)[0].strip()
    if not _ERROR_CLASS_PATTERN.fullmatch(candidate):
        candidate = "Error"
    return _bounded_utf8(candidate, MAX_ERROR_CLASS_BYTES) or "Error"


@dataclass(frozen=True)
class KeyedBatchDiagnostic:
    """The only message-specific fields allowed in keyed status and log records."""

    physical_id: str
    dispatch_hash_prefix: str
    contribution_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "physical_id",
            _bounded_utf8(self.physical_id, MAX_PHYSICAL_ID_BYTES),
        )
        object.__setattr__(
            self,
            "dispatch_hash_prefix",
            _bounded_utf8(self.dispatch_hash_prefix, DISPATCH_HASH_PREFIX_LENGTH),
        )
        object.__setattr__(
            self,
            "contribution_count",
            max(0, min(int(self.contribution_count), MAX_DIAGNOSTIC_CONTRIBUTIONS)),
        )

    def as_dict(self, error: BaseException | str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "physical_id": self.physical_id,
            "dispatch_hash_prefix": self.dispatch_hash_prefix,
            "contribution_count": self.contribution_count,
        }
        if error is not None:
            result["error_class"] = bounded_error_class(error)
        return result


def diagnostic_from_queue_data(data: Any) -> Optional[KeyedBatchDiagnostic]:
    """Extract bounded fields without retaining or decoding contributions."""

    if not isinstance(data, Mapping):
        return None
    if {
        "physical_id",
        "dispatch_hash_prefix",
        "contribution_count",
    }.issubset(data):
        return KeyedBatchDiagnostic(
            physical_id=str(data.get("physical_id", "")),
            dispatch_hash_prefix=str(data.get("dispatch_hash_prefix", "")),
            contribution_count=int(data.get("contribution_count", 0)),
        )

    raw_payload = data.get("data")
    if isinstance(raw_payload, str):
        try:
            payload = json.loads(raw_payload)
        except (TypeError, json.JSONDecodeError):
            return None
    elif isinstance(raw_payload, Mapping):
        payload = raw_payload
    else:
        payload = data
    if not isinstance(payload, Mapping):
        return None
    wrapper = payload.get("_queuefs_keyed_batch")
    if not isinstance(wrapper, Mapping):
        return None

    dispatch_key = wrapper.get("dispatch_key")
    dispatch_hash_prefix = ""
    if isinstance(dispatch_key, str):
        dispatch_hash_prefix = hashlib.sha256(dispatch_key.encode("utf-8")).hexdigest()[
            :DISPATCH_HASH_PREFIX_LENGTH
        ]
    contributions = wrapper.get("contributions")
    contribution_count = len(contributions) if isinstance(contributions, list) else 0
    return KeyedBatchDiagnostic(
        physical_id=str(data.get("id", "")),
        dispatch_hash_prefix=dispatch_hash_prefix,
        contribution_count=contribution_count,
    )


def get_keyed_batch_diagnostic() -> Optional[KeyedBatchDiagnostic]:
    return _current_diagnostic.get()


def set_keyed_batch_diagnostic(
    diagnostic: KeyedBatchDiagnostic,
) -> Token[Optional[KeyedBatchDiagnostic]]:
    return _current_diagnostic.set(diagnostic)


def reset_keyed_batch_diagnostic(token: Token[Optional[KeyedBatchDiagnostic]]) -> None:
    _current_diagnostic.reset(token)


@contextmanager
def bind_keyed_batch_diagnostic(
    diagnostic: KeyedBatchDiagnostic,
) -> Iterator[None]:
    token = set_keyed_batch_diagnostic(diagnostic)
    try:
        yield
    finally:
        reset_keyed_batch_diagnostic(token)


class KeyedBatchLogFilter(logging.Filter):
    """Rewrite every log emitted during keyed work to a bounded diagnostic."""

    _SENSITIVE_FIELDS = (
        "uri",
        "url_path",
        "telemetry_id",
        "coalesce_key",
        "dispatch_key",
        "merge_signature",
        "payload",
        "contributions",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        diagnostic = get_keyed_batch_diagnostic()
        if diagnostic is None:
            return True
        error_class = ""
        if record.exc_info and record.exc_info[1] is not None:
            error_class = bounded_error_class(record.exc_info[1])
        elif getattr(record, "error_class", ""):
            error_class = bounded_error_class(record.error_class)

        lifecycle = bool(getattr(record, "keyed_batch_lifecycle", False))
        record.msg = (
            "Semantic keyed batch lifecycle" if lifecycle else "Semantic keyed batch diagnostic"
        )
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        record.physical_id = diagnostic.physical_id
        record.dispatch_hash_prefix = diagnostic.dispatch_hash_prefix
        record.contribution_count = diagnostic.contribution_count
        record.error_class = error_class
        record.request_id = diagnostic.physical_id
        record.account_id = ""
        record.user_id = ""
        if not lifecycle:
            record.event = "semantic.keyed_diagnostic"
        for field in self._SENSITIVE_FIELDS:
            record.__dict__.pop(field, None)
        return True


def install_keyed_batch_log_filter(target: logging.Logger) -> None:
    if not any(isinstance(item, KeyedBatchLogFilter) for item in target.filters):
        target.addFilter(KeyedBatchLogFilter())
