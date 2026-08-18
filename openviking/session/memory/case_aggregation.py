# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic identity matching and aggregation policy for Case memories."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations

CASE_MEMORY_TYPE = "cases"
CASE_IDENTITY_FIELD = "case_identity"
PROPOSED_CASE_IDENTITY_FIELD = "_proposed_case_identity"
CASE_SOURCE_IDS_FIELD = "_case_source_ids"
CASE_PENDING_SOURCES_FIELD = "_case_pending_sources"
CASE_DYNAMIC_FIELDS = (
    "task_signature",
    "input",
    "situation",
    "rubric",
    "evidence",
)
CASE_SYSTEM_FIELDS = (
    "case_status",
    "source_count",
    "last_compacted_source_count",
    "last_compacted_version",
)
CASE_IDENTITY_DIMENSIONS = (
    "goal",
    "subject",
    "action_pattern",
    "success_boundary",
    "context_constraints",
)
CASE_KEY_DIMENSIONS = ("goal", "subject", "success_boundary")
CASE_MATCH_WEIGHTS = {
    "goal": 0.30,
    "subject": 0.25,
    "success_boundary": 0.25,
    "action_pattern": 0.15,
    "context_constraints": 0.05,
}
CASE_MATCH_VALUES = {
    "MATCH": 1.0,
    "COMPATIBLE": 0.7,
    "UNKNOWN": 0.0,
}
CASE_MATCH_THRESHOLD = 0.80
CASE_MATCH_MARGIN = 0.10
CASE_RECOMPACT_SOURCE_DELTA = 3
CASE_BODY_LENGTH_LIMIT = 8000
CASE_STATUS_VALUES = frozenset({"draft", "promoted", "degraded", "archived"})

_CASE_EXACT_LITERAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "URL or absolute path",
        re.compile(
            r"(?:https?://|tos://|s3://|viking://|file://|/Users/|/tmp/|/var/)"
            r"[^\s\"'，。；;]+",
            re.IGNORECASE,
        ),
    ),
    (
        "UUID or run identifier",
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "exact date",
        re.compile(r"\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b"),
    ),
    (
        "exact percentage",
        re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?\s*%"),
    ),
    (
        "exact currency amount",
        re.compile(
            r"(?:(?:USD|CNY|RMB|EUR|GBP)\s*[-+]?\d+(?:[.,]\d+)*|"
            r"[$¥€£]\s*[-+]?\d+(?:[.,]\d+)*)",
            re.IGNORECASE,
        ),
    ),
    (
        "exact year",
        re.compile(r"(?<![\w])(?:19|20)\d{2}(?![\w])"),
    ),
    (
        "exact measured count",
        re.compile(
            r"(?<![\w.])[-+]?\d+(?:\.\d+)?\s*"
            r"(?:pages?|slides?|rows?|columns?|items?|units?|records?|worksheets?|"
            r"sheets?|teams?|minutes?|hours?|days?|months?|years?|decimal\s+places?|"
            r"页|张|行|列|个|件|条|份|次|分钟|小时|天|月|年|位小数)",
            re.IGNORECASE,
        ),
    ),
    (
        "exact file name",
        re.compile(
            r"(?<![./\w-])[\w.-]{2,}\.(?:xlsx|xls|csv|pdf|pptx|ppt|docx|doc|json|zip)\b",
            re.IGNORECASE,
        ),
    ),
)

_CASE_EXACT_YEAR_PATTERN = re.compile(r"(?<![\w])(?:19|20)\d{2}(?![\w])")

CaseMatchLabel = Literal["MATCH", "COMPATIBLE", "UNKNOWN", "CONFLICT"]


class CaseIdentity(BaseModel):
    """Stable task-family identity stored as a compact JSON string."""

    model_config = ConfigDict(extra="ignore")

    goal: str = ""
    subject: str = ""
    action_pattern: str = ""
    success_boundary: str = ""
    context_constraints: list[str] = Field(default_factory=list)
    facets: dict[str, Any] = Field(default_factory=dict)

    @field_validator("context_constraints", mode="before")
    @classmethod
    def _normalize_constraints(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (tuple, set)):
            return [str(item) for item in value if str(item).strip()]
        return value

    @field_validator("facets", mode="before")
    @classmethod
    def _normalize_facets(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, list):
            return {str(index): item for index, item in enumerate(value)}
        if isinstance(value, str):
            return {"value": value} if value.strip() else {}
        return value

    def compact_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class CaseIdentityComparison(BaseModel):
    """LLM classification for one current Case proposal and one candidate."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    candidate_id: str
    goal: CaseMatchLabel
    subject: CaseMatchLabel
    action_pattern: CaseMatchLabel
    success_boundary: CaseMatchLabel
    context_constraints: CaseMatchLabel


def parse_case_identity(value: Any) -> CaseIdentity | None:
    if isinstance(value, CaseIdentity):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(value, dict):
        return None
    try:
        identity = CaseIdentity.model_validate(value)
    except Exception:
        return None
    if not identity.goal or not identity.subject or not identity.success_boundary:
        return None
    return identity


def case_identity_generalization_violations(value: Any) -> list[str]:
    """Validate that a Case identity contains task-family semantics, not run values."""

    identity = parse_case_identity(value)
    if identity is None:
        return ["case_identity must be a valid compact JSON object"]
    return _literal_violations(
        identity.model_dump(mode="json"),
        field_path=CASE_IDENTITY_FIELD,
    )


def case_input_generalization_violations(value: Any) -> list[str]:
    """Validate generalized Case input and its existing variable_types contract."""

    input_payload = _parse_json_object(value)
    if input_payload is None:
        return ["input must be a valid JSON object"]
    violations: list[str] = []
    variable_types = input_payload.get("variable_types")
    if not isinstance(variable_types, list):
        violations.append(
            "input.variable_types must be a list, empty when the task has no variable parameters"
        )
    if "parameter_slots" in input_payload:
        violations.append("input must use variable_types instead of introducing parameter_slots")
    violations.extend(
        _literal_violations(
            input_payload,
            field_path="input",
        )
    )
    return violations


def generalize_case_year_literals(
    identity: CaseIdentity,
    input_value: Any,
) -> tuple[CaseIdentity, str | None]:
    """Replace exact years with a stable role before promoting a draft Case."""

    identity_payload, identity_changed = _replace_exact_year_literals(
        identity.model_dump(mode="json")
    )
    input_payload = _parse_json_object(input_value)
    if input_payload is None:
        return identity, None
    generalized_input, input_changed = _replace_exact_year_literals(input_payload)
    if identity_changed or input_changed:
        variable_types = generalized_input.get("variable_types")
        if isinstance(variable_types, list) and "target year" not in variable_types:
            variable_types.append("target year")
    return (
        CaseIdentity.model_validate(identity_payload),
        json.dumps(
            generalized_input,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def case_generalization_violations(operation: ResolvedOperation) -> list[str]:
    """Return high-confidence one-run literals that must not enter a Case."""

    if operation.memory_type != CASE_MEMORY_TYPE:
        return []
    fields = dict(operation.memory_fields or {})
    violations: list[str] = []

    violations.extend(case_identity_generalization_violations(fields.get(CASE_IDENTITY_FIELD)))

    violations.extend(case_input_generalization_violations(fields.get("input")))

    for field_name in ("task_signature", "situation"):
        violations.extend(_literal_violations(fields.get(field_name), field_path=field_name))
    return list(dict.fromkeys(violations))


def case_generalization_post_validation(
    operations: ResolvedOperations,
    retry_count: int,
    **_: Any,
) -> Any:
    """Retry over-specific Case drafts once, then drop only invalid Case operations."""

    invalid: list[tuple[ResolvedOperation, list[str]]] = []
    for operation in list(operations.upsert_operations or []):
        violations = case_generalization_violations(operation)
        if violations:
            invalid.append((operation, violations))
    if not invalid:
        return None

    if retry_count > 0:
        invalid_ids = {id(operation) for operation, _ in invalid}
        operations.upsert_operations = [
            operation
            for operation in operations.upsert_operations
            if id(operation) not in invalid_ids
        ]
        summary = "; ".join(
            f"{_case_operation_label(operation)}: {', '.join(violations[:4])}"
            for operation, violations in invalid
        )
        operations.errors.append(
            "Dropped Case operations after generalization retry failed: " + summary
        )
        return None

    from openviking.session.memory.extract_loop import PostValidationRetryDecision

    details = "\n".join(
        f"- {_case_operation_label(operation)}: " + "; ".join(violations[:6])
        for operation, violations in invalid
    )
    return PostValidationRetryDecision(
        retry=True,
        include_latest_draft=True,
        compact_retry_context=True,
        instruction=(
            "Case generalization validation failed:\n"
            f"{details}\n\n"
            "Rewrite every invalid Case operation while preserving all valid non-Case operations. "
            "case_identity, task_signature, input preconditions, and situation may contain only "
            "stable task-family semantics. Replace exact IDs, names, dates, amounts, percentages, "
            "paths, filenames, and one-run counts with semantic parameter roles. Record those roles "
            "as concise entries in input.variable_types without retaining their current values. "
            "Do not add a new parameter_slots field."
        ),
    )


def _literal_violations(value: Any, *, field_path: str) -> list[str]:
    violations: list[str] = []
    for path, text in _iter_text_values(value, field_path):
        for label, pattern in _CASE_EXACT_LITERAL_PATTERNS:
            for match in pattern.finditer(text):
                literal = match.group(0).strip()
                if literal:
                    violations.append(f"{path} contains {label}: {literal!r}")
    return violations


def _replace_exact_year_literals(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        replaced, count = _CASE_EXACT_YEAR_PATTERN.subn("target_year", value)
        return replaced, count > 0
    if isinstance(value, dict):
        changed = False
        result: dict[str, Any] = {}
        for key, item in value.items():
            result[key], item_changed = _replace_exact_year_literals(item)
            changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        changed = False
        result: list[Any] = []
        for item in value:
            replaced, item_changed = _replace_exact_year_literals(item)
            result.append(replaced)
            changed = changed or item_changed
        return result, changed
    return value, False


def _iter_text_values(value: Any, path: str):
    if isinstance(value, str):
        yield path, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_text_values(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_text_values(item, f"{path}[{index}]")


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _case_operation_label(operation: ResolvedOperation) -> str:
    fields = dict(operation.memory_fields or {})
    return str(fields.get("case_name") or (operation.uris[0] if operation.uris else "case"))


def fallback_case_identity(fields: dict[str, Any]) -> CaseIdentity:
    """Backfill legacy Cases without inventing benchmark-only identity fields."""

    task_signature = str(fields.get("task_signature") or fields.get("case_name") or "task")
    input_summary = _json_summary(fields.get("input"))
    situation = str(fields.get("situation") or input_summary or task_signature)
    return CaseIdentity(
        goal=task_signature,
        subject=input_summary or task_signature,
        action_pattern=task_signature,
        success_boundary=f"Observable completion of: {task_signature}",
        context_constraints=[situation] if situation else [],
        facets={},
    )


def prepare_case_operation(operation: ResolvedOperation) -> ResolvedOperation:
    """Attach immutable identity and system counters before Case matching."""

    if operation.memory_type != CASE_MEMORY_TYPE:
        return operation
    prepared = operation.model_copy(deep=True)
    fields = dict(prepared.memory_fields or {})
    old_fields = (
        dict(prepared.old_memory_file_content.extra_fields or {})
        if prepared.old_memory_file_content is not None
        else {}
    )

    proposed = (
        parse_case_identity(fields.get(PROPOSED_CASE_IDENTITY_FIELD))
        or parse_case_identity(fields.get(CASE_IDENTITY_FIELD))
        or fallback_case_identity(fields)
    )
    stored = parse_case_identity(old_fields.get(CASE_IDENTITY_FIELD))
    if stored is None and old_fields:
        stored = fallback_case_identity(old_fields)

    fields[PROPOSED_CASE_IDENTITY_FIELD] = proposed.compact_json()
    fields[CASE_IDENTITY_FIELD] = (stored or proposed).compact_json()

    old_source_ids = _normalized_source_ids(old_fields.get(CASE_SOURCE_IDS_FIELD))
    source_key = case_source_key(prepared)
    source_ids = list(old_source_ids)
    if source_key and source_key not in source_ids:
        source_ids.append(source_key)
    if source_ids:
        fields[CASE_SOURCE_IDS_FIELD] = source_ids
    pending_sources = _normalized_pending_sources(old_fields.get(CASE_PENDING_SOURCES_FIELD))
    if not source_key or source_key not in old_source_ids:
        pending_sources = _merge_pending_sources(
            pending_sources,
            [_case_source_summary(prepared, proposed, source_key=source_key)],
        )
    fields[CASE_PENDING_SOURCES_FIELD] = pending_sources

    old_count = _positive_int(old_fields.get("source_count"), default=1 if old_fields else 0)
    if source_key:
        if source_key in old_source_ids:
            source_count = max(old_count, len(source_ids))
        else:
            source_count = max(old_count + 1, len(source_ids))
    else:
        source_count = max(old_count, 1)
    fields["source_count"] = source_count
    fields["case_status"] = normalize_case_status(old_fields.get("case_status"))
    fields["last_compacted_source_count"] = _positive_int(
        old_fields.get("last_compacted_source_count"),
        default=0,
    )
    fields["last_compacted_version"] = _positive_int(
        old_fields.get("last_compacted_version"),
        default=0,
    )
    prepared.memory_fields = fields
    return prepared


def case_source_key(operation: ResolvedOperation) -> str | None:
    source = operation.source
    if source is not None:
        for prefix, value in (
            ("session", source.session_id),
            ("archive", source.archive_uri),
            ("extraction", source.extraction_id),
        ):
            if value:
                return f"{prefix}:{value}"
    fields = dict(operation.memory_fields or {})
    source_id = fields.get("source_extraction_id")
    return f"extraction:{source_id}" if source_id else None


def score_case_comparison(comparison: CaseIdentityComparison) -> float | None:
    labels = comparison.model_dump()
    if any(labels[field] == "CONFLICT" for field in CASE_IDENTITY_DIMENSIONS):
        return None
    if any(labels[field] == "UNKNOWN" for field in CASE_KEY_DIMENSIONS):
        return None
    return sum(
        CASE_MATCH_WEIGHTS[field] * CASE_MATCH_VALUES[labels[field]]
        for field in CASE_IDENTITY_DIMENSIONS
    )


def select_case_primary(
    comparisons: list[CaseIdentityComparison],
) -> tuple[str | None, float | None]:
    ranked = sorted(
        (
            (score, comparison.candidate_id)
            for comparison in comparisons
            if (score := score_case_comparison(comparison)) is not None
            and score >= CASE_MATCH_THRESHOLD
        ),
        reverse=True,
    )
    if not ranked:
        return None, None
    best_score, best_id = ranked[0]
    if len(ranked) > 1 and best_score - ranked[1][0] < CASE_MATCH_MARGIN:
        return None, best_score
    return best_id, best_score


def case_comparison_has_identity_drift(
    comparison: CaseIdentityComparison,
) -> bool:
    """Return whether a compatible match should refresh the aggregated body."""

    return any(
        getattr(comparison, field) in {"COMPATIBLE", "UNKNOWN"}
        for field in CASE_IDENTITY_DIMENSIONS
    )


def normalize_case_status(value: Any, *, default: str = "draft") -> str:
    status = str(value or "").strip().lower()
    if status in CASE_STATUS_VALUES:
        return status
    return default


def should_compact_case(
    *,
    source_count: int,
    last_compacted_source_count: int,
    existing_file: MemoryFile | None,
) -> bool:
    if source_count >= 2 and last_compacted_source_count == 0:
        return True
    if source_count - last_compacted_source_count >= CASE_RECOMPACT_SOURCE_DELTA:
        return True
    if existing_file is None or source_count <= last_compacted_source_count:
        return False
    return _case_body_length(existing_file) > CASE_BODY_LENGTH_LIMIT


def merged_case_source_state(
    *,
    grouped_operations: list[ResolvedOperation],
    existing_file: MemoryFile | None,
) -> tuple[list[str], int]:
    old_fields = dict(existing_file.extra_fields or {}) if existing_file is not None else {}
    source_ids = _normalized_source_ids(old_fields.get(CASE_SOURCE_IDS_FIELD))
    old_count = _positive_int(old_fields.get("source_count"), default=1 if old_fields else 0)
    new_source_count = 0
    for operation in grouped_operations:
        key = case_source_key(operation)
        if key:
            if key not in source_ids:
                source_ids.append(key)
                new_source_count += 1
        elif existing_file is None:
            new_source_count += 1
    source_count = max(old_count + new_source_count, len(source_ids), 1)
    return source_ids, source_count


def merged_case_pending_sources(
    *,
    grouped_operations: list[ResolvedOperation],
    existing_file: MemoryFile | None,
) -> list[dict[str, Any]]:
    old_fields = dict(existing_file.extra_fields or {}) if existing_file is not None else {}
    pending = _normalized_pending_sources(old_fields.get(CASE_PENDING_SOURCES_FIELD))
    for operation in grouped_operations:
        pending = _merge_pending_sources(
            pending,
            _normalized_pending_sources(
                dict(operation.memory_fields or {}).get(CASE_PENDING_SOURCES_FIELD)
            ),
        )
    return pending


def _json_summary(value: Any) -> str:
    if not isinstance(value, str):
        return str(value or "")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value
    if isinstance(parsed, dict):
        return str(parsed.get("summary") or parsed.get("task") or value)
    return value


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _normalized_source_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item) for item in value if str(item).strip()))


def _case_source_summary(
    operation: ResolvedOperation,
    identity: CaseIdentity,
    *,
    source_key: str | None,
) -> dict[str, Any]:
    fields = dict(operation.memory_fields or {})
    summary = {
        "source_id": source_key or "",
        "identity": identity.model_dump(mode="json"),
        "task_signature": _compact_source_text(fields.get("task_signature"), 1000),
        "input": _compact_source_text(fields.get("input"), 2000),
        "situation": _compact_source_text(fields.get("situation"), 1000),
        "evidence": _compact_source_text(fields.get("evidence"), 2000),
    }
    if not summary["source_id"]:
        digest_payload = json.dumps(
            {key: value for key, value in summary.items() if key != "source_id"},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        summary["source_id"] = (
            "content:" + hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()[:16]
        )
    return summary


def _compact_source_text(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "...[truncated]"


def _normalized_pending_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or "").strip()
        if not source_id:
            continue
        result.append(
            {
                "source_id": source_id,
                "identity": dict(item.get("identity") or {}),
                "task_signature": _compact_source_text(item.get("task_signature"), 1000),
                "input": _compact_source_text(item.get("input"), 2000),
                "situation": _compact_source_text(item.get("situation"), 1000),
                "evidence": _compact_source_text(item.get("evidence"), 2000),
            }
        )
    return _merge_pending_sources([], result)


def _merge_pending_sources(
    current: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in [*current, *incoming]:
        source_id = str(item.get("source_id") or "").strip()
        if not source_id:
            continue
        if source_id not in merged:
            order.append(source_id)
        merged[source_id] = item
    return [merged[source_id] for source_id in order]


def _case_body_length(memory_file: MemoryFile) -> int:
    fields = dict(memory_file.extra_fields or {})
    return sum(len(str(fields.get(name) or "")) for name in CASE_DYNAMIC_FIELDS)
