# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic validation shared by normal and training Experience extraction."""

from __future__ import annotations

import re
from typing import Any, Callable

from openviking.server.identity import RequestContext
from openviking.session.memory.case_aggregation import case_generalization_violations
from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
from openviking.session.memory.memory_updater import render_operation_after_file

REQUIRED_EXPERIENCE_SECTIONS = (
    "Situation",
    "Reminder",
    "Procedure",
    "Verification",
    "Fallback",
    "Anti-pattern",
)

_SECTION_RE = re.compile(r"^##\s+([^\n]+?)\s*$\n?(.*?)(?=^##\s+|\Z)", re.MULTILINE | re.DOTALL)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+", re.MULTILINE)
_GENERIC_REMINDER_RE = re.compile(
    r"(?:check|verify|validate|review)\s+(?:all|every|everything)\s+"
    r"(?:requirements?|items?|outputs?|fields?|data|calculations?|formatting)"
    r"|ensure\s+(?:full|complete)\s+(?:compliance|coverage)"
    r"|review\s+carefully"
    r"|检查(?:全部|所有)要求"
    r"|全面检查"
    r"|仔细检查"
    r"|确保(?:完全|全面)?(?:合规|覆盖)",
    re.IGNORECASE,
)
_BROAD_REQUIREMENT_CHECKLIST_RE = re.compile(
    r"\b(?:extract|list|map|check|verify|cross[- ]?reference)\s+"
    r"(?:all|every|100%\s+of)\s+"
    r"(?:(?:explicit(?:ly)?|mandatory|required|listed|user|content)\s+){0,5}"
    r"(?:requirements?|constraints?|elements?|items?|fields?|formulas?|rules?)\b"
    r"|(?:提取|列出|映射|检查|核对)(?:全部|所有|每一项)(?:明确)?(?:要求|约束|内容)",
    re.IGNORECASE,
)
_TRAJECTORY_CONTROL_COUNT_RE = re.compile(
    r"\b(?:after\s+)?\d+(?:\s*[-–]\s*\d+)?\s+(?:[a-z-]+\s+){0,2}"
    r"(?:attempts?|retries?|retry\s+attempts?)\b"
    r"|\bspot[- ]?(?:check|verify|inspect)\s+\d+(?:\s*[-–]\s*\d+)?\b"
    r"|\b(?:every|within|after)\s+\d+\s*(?:seconds?|minutes?|hours?)\b"
    r"|\b\d+\s*(?:seconds?|minutes?|hours?)[- ](?:interval|window|timeout)\b"
    r"|\bafter\s+\d+\s+consecutive\s+(?:polls?|checks?)\b"
    r"|\b(?:re[- ]?run|check|verify|sample|inspect)\s+\d+(?:\.\d+)?%"
    r"|\b(?:random\s+)?samples?\s+(?:of\s+)?\d+(?:\s*[-–]\s*\d+)?\s+"
    r"(?:entities|rows|records|items|values|results|outputs)\b"
    r"|\b(?:at\s+least\s+)?\d+(?:\s*[-–]\s*\d+)?\s+"
    r"(?:sample|sampled|key)\s+(?:entities|rows|records|items|values|results|outputs)\b"
    r"|(?:尝试|重试|抽查)\s*\d+(?:\s*[-–至]\s*\d+)?\s*(?:次|项|条)?",
    re.IGNORECASE,
)
_INTERNAL_INFRASTRUCTURE_RE = re.compile(
    r"\b(?:sandbox\s+(?:output\s+)?trace|trace\s+(?:retrieval|generation|readiness)|"
    r"post[- ]run\s+trace|output[- ]file\s+trace)\b"
    r"|沙箱(?:输出)?轨迹|轨迹(?:下载|生成|就绪|抓取)",
    re.IGNORECASE,
)
_INTERNAL_EVALUATION_RE = re.compile(
    r"\b(?:rubric|criterion|criteria|evaluator|benchmark|evaluation\s+(?:criterion|criteria)|"
    r"rollout[_ -]?evaluation|request[_ -]?id|execution[_ -]?id|case[_ -]?id)\b"
    r"|评测题|评测器|评分项|评分标准|隐藏答案|基准测试",
    re.IGNORECASE,
)
_RAW_IDENTIFIER_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|(?:^|[\s`])/(?:Users|home|tmp|var)/\S+"
    r"|viking://\S+",
    re.IGNORECASE | re.MULTILINE,
)
_SOURCE_INSTANCE_LITERAL_RE = re.compile(
    r"\b[^\s/]+\.(?:xlsx?|csv|docx?|pptx?|pdf|png|jpe?g)\b"
    r"|\b(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?\b"
    r"|[$€£¥]\s*\d+(?:[,.]\d+)*"
    r"|\b\d+(?:\.\d+)?\s*%",
    re.IGNORECASE,
)
_INDEPENDENT_CHECK_RE = re.compile(
    r"\b(?:authoritative|source|input|reopen|render|recalculate|recompute|reconcile|"
    r"compare|cross[- ]?check|assert|parse|persisted|saved artifact|downloaded artifact)\b"
    r"|权威|源数据|原始输入|重新打开|重开|渲染|重算|复算|对账|交叉检查|独立检查|"
    r"比较|断言|解析|实际文件|落盘文件",
    re.IGNORECASE,
)
_FALLBACK_CONDITION_RE = re.compile(
    r"\b(?:if|when|unless|unavailable|fails?|cannot|missing)\b"
    r"|如果|当|无法|不可用|失败|缺失|不能",
    re.IGNORECASE,
)


def experience_structure_issues(
    content: str,
    *,
    max_chars: int = 4200,
    max_english_words: int = 360,
) -> list[str]:
    """Return deterministic defects in one runtime-visible Experience body."""

    sections = _parse_sections(content)
    issues: list[str] = []
    missing = [name for name in REQUIRED_EXPERIENCE_SECTIONS if not sections.get(name, "").strip()]
    if missing:
        issues.append("missing or empty sections: " + ", ".join(missing))

    duplicates = [
        name
        for name in REQUIRED_EXPERIENCE_SECTIONS
        if len(re.findall(rf"^##\s+{re.escape(name)}\s*$", content, re.MULTILINE)) > 1
    ]
    if duplicates:
        issues.append("duplicate sections: " + ", ".join(duplicates))

    if len(content) > max_chars:
        issues.append(f"content is too long ({len(content)} chars; max {max_chars})")
    english_words = re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", content)
    if len(english_words) > max_english_words:
        issues.append(
            f"content is too long ({len(english_words)} English words; max {max_english_words})"
        )

    situation = sections.get("Situation", "")
    if situation and _list_item_count(situation) < 4:
        issues.append(
            "Situation must contain applicability, non-applicability, evidence, and boundary"
        )

    reminder = sections.get("Reminder", "")
    if reminder and _GENERIC_REMINDER_RE.search(reminder):
        issues.append("Reminder is a generic check-all slogan instead of one decision rule")
    if _BROAD_REQUIREMENT_CHECKLIST_RE.search(content):
        issues.append(
            "content collapses heterogeneous requirements into a broad checklist instead of "
            "repairing one root decision"
        )
    if _TRAJECTORY_CONTROL_COUNT_RE.search(content):
        issues.append(
            "content hard-codes a retry, polling, timeout, or sampling count from one trajectory "
            "instead of binding a future runtime policy or stopping condition"
        )
    if _INTERNAL_INFRASTRUCTURE_RE.search(content):
        issues.append(
            "content turns an internal sandbox/trace infrastructure incident into user-task Experience"
        )

    procedure = sections.get("Procedure", "")
    procedure_count = _list_item_count(procedure)
    if procedure and not 2 <= procedure_count <= 5:
        issues.append(f"Procedure must contain 2-5 actions, found {procedure_count}")

    verification = sections.get("Verification", "")
    if verification and _list_item_count(verification) < 2:
        issues.append("Verification must contain at least two concrete checks")
    if verification and not _INDEPENDENT_CHECK_RE.search(verification):
        issues.append("Verification lacks an authoritative or independently observable check")

    fallback = sections.get("Fallback", "")
    if fallback and not _FALLBACK_CONDITION_RE.search(fallback):
        issues.append("Fallback lacks a concrete unavailable-or-failed condition")

    if _INTERNAL_EVALUATION_RE.search(content):
        issues.append("content exposes evaluation-only terminology")
    if _RAW_IDENTIFIER_RE.search(content):
        issues.append("content exposes a raw identifier or local/internal path")
    if _SOURCE_INSTANCE_LITERAL_RE.search(content):
        issues.append("content exposes a source-task filename, date, amount, or exact percentage")

    return list(dict.fromkeys(issues))


def build_memory_extraction_post_validation(
    *,
    context_provider: Any,
    ctx: RequestContext | None,
) -> Callable[..., Any]:
    """Build one hook that validates both Case generalization and Experience quality."""

    async def validate(
        operations: ResolvedOperations,
        retry_count: int,
        **_: Any,
    ) -> Any:
        schemas = context_provider.get_memory_schemas(ctx)
        experience_schema = next(
            (schema for schema in schemas if getattr(schema, "memory_type", "") == "experiences"),
            None,
        )
        extract_context = context_provider.get_extract_context()

        invalid: list[tuple[ResolvedOperation, list[str]]] = []
        for operation in list(operations.upsert_operations or []):
            if operation.memory_type == "cases":
                issues = case_generalization_violations(operation)
            elif operation.memory_type == "experiences" and experience_schema is not None:
                try:
                    content = (await render_operation_after_file(
                        operation,
                        schema=experience_schema,
                        extract_context=extract_context,
                    )).plain_content()
                    issues = experience_structure_issues(content)
                except Exception as exc:
                    issues = [
                        f"failed to render Experience for validation: {type(exc).__name__}: {exc}"
                    ]
            else:
                issues = []
            if issues:
                invalid.append((operation, issues))

        if not invalid:
            return None

        if retry_count > 0:
            invalid_ids = {id(operation) for operation, _ in invalid}
            operations.upsert_operations = [
                operation
                for operation in operations.upsert_operations
                if id(operation) not in invalid_ids
            ]
            operations.errors.append(
                "Dropped invalid memory operations after extraction-quality retry: "
                + "; ".join(
                    f"{_operation_label(operation)}: {', '.join(issues[:4])}"
                    for operation, issues in invalid
                )
            )
            return None

        from openviking.session.memory.extract_loop import PostValidationRetryDecision

        details = "\n".join(
            f"- {_operation_label(operation)}: " + "; ".join(issues[:6])
            for operation, issues in invalid
        )
        return PostValidationRetryDecision(
            retry=True,
            include_latest_draft=True,
            compact_retry_context=True,
            instruction=(
                "Memory extraction quality validation failed:\n"
                f"{details}\n\n"
                "Rewrite every invalid operation while preserving valid operations. For Case, keep "
                "only stable task-family semantics and replace exact instance values with semantic "
                "variable types. For Experience, keep one agent-controllable root decision in the "
                "six required sections; remove source-task filenames, entities, dates, amounts, exact "
                "percentages, IDs, evaluation internals, and fixed retry or sampling counts. Bind "
                "actions to future runtime inputs and observable stop conditions. If the lesson fails "
                "after substituting different valid instance values from the same task family, omit "
                "that Experience instead of paraphrasing the source task."
            ),
        )

    return validate


def _operation_label(operation: ResolvedOperation) -> str:
    fields = dict(operation.memory_fields or {})
    name = fields.get("experience_name") or fields.get("case_name")
    if name:
        return f"{operation.memory_type}:{name}"
    if operation.uris:
        return f"{operation.memory_type}:{operation.uris[0].rsplit('/', 1)[-1]}"
    return operation.memory_type


def _parse_sections(content: str) -> dict[str, str]:
    return {name.strip(): body.strip() for name, body in _SECTION_RE.findall(content)}


def _list_item_count(content: str) -> int:
    return len(_LIST_ITEM_RE.findall(content or ""))
