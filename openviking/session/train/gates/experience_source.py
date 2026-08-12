# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic provenance checks for Experience extraction sources."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openviking.session.train.domain import Trajectory

from .models import GateDecision, GateMode, GateTarget

_TERMINAL_INTERNAL_FAILURE_RE = re.compile(
    r"\b(?:MAIN_AGENT_STD_FAILED|EVALUATOR_FAILED|TOOL_SERVER_TOOL_EXEC_ERROR|"
    r"SandboxTraceError|trace_failed)\b",
    re.IGNORECASE,
)
_INTERNAL_TRACE_RE = re.compile(
    r"\b(?:sandbox\s+(?:output[- ]file\s+)?trace|output[- ]file\s+trace|"
    r"trace\s+(?:retrieval|collection|readiness|generation)|"
    r"fetch_sandbox_output_trace|get_sandbox_trace|retrieve_trace|wait_for_trace)\b",
    re.IGNORECASE,
)
_INTERNAL_ORCHESTRATION_RE = re.compile(
    r"\b(?:rollout\s+(?:execution|evaluation)|bench_rollout|execute_eval|home_test|"
    r"request[_ -]?ids?|execution[_ -]?id)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ExperienceSourceEligibilityGate:
    """Reject Experience updates derived from internal platform incidents.

    The visible draft can euphemize an internal trace failure as a seemingly
    reusable workflow. Source provenance is therefore authoritative: platform
    orchestration failures are not user-task decisions and cannot be repaired by
    rewriting the Experience body.
    """

    mode: GateMode = "enforce"
    name: str = "experience_source_eligibility"

    def applies_to(self, target: GateTarget) -> bool:
        return (
            target.memory_type == "experiences"
            and target.target_kind in {"gradient", "plan_item"}
            and target.trajectory is not None
        )

    async def evaluate(self, target: GateTarget) -> GateDecision | None:
        assert target.trajectory is not None
        reason = experience_source_rejection_reason(target.trajectory)
        if not reason:
            return None
        return GateDecision(
            gate_name=self.name,
            action="reject",
            reason=reason,
            evidence={
                "target_name": target.target_name,
                "source_trajectory": target.trajectory.uri,
            },
            retriable=False,
        )


def experience_source_rejection_reason(trajectory: Trajectory) -> str:
    """Return a stable rejection reason when the source is not user-task learnable."""

    text = _trajectory_core_evidence(trajectory)
    if _TERMINAL_INTERNAL_FAILURE_RE.search(text):
        return "source trajectory is an internal platform failure, not a user-task decision"
    if _INTERNAL_TRACE_RE.search(text) and _INTERNAL_ORCHESTRATION_RE.search(text):
        return "source trajectory concerns internal rollout trace orchestration, not user work"
    return ""


def _trajectory_core_evidence(trajectory: Trajectory) -> str:
    """Return outcome attribution while excluding incidental execution-log errors."""

    metadata = dict(getattr(trajectory, "metadata", {}) or {})
    try:
        metadata_text = json.dumps(metadata, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        metadata_text = str(metadata)
    content = str(getattr(trajectory, "content", "") or "")
    outcome_sections = ""
    evaluation_match = re.search(
        r"(?ms)^##\s+Evaluation\s*$.*?(?=^##\s+Execution\s*$|\Z)",
        content,
    )
    if evaluation_match:
        outcome_sections = evaluation_match.group(0)
    return "\n".join(
        (
            str(getattr(trajectory, "name", "") or ""),
            str(getattr(trajectory, "retrieval_anchor", "") or ""),
            outcome_sections,
            metadata_text,
        )
    )
