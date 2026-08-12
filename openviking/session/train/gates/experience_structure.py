# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic structure and leakage gate for injectable Experiences."""

from __future__ import annotations

from dataclasses import dataclass

from openviking.session.memory.experience_validation import experience_structure_issues

from .models import GateDecision, GateMode, GateTarget


@dataclass(slots=True)
class ExperienceStructureGate:
    """Reject malformed, generic, or evaluation-leaking Experience drafts."""

    mode: GateMode = "enforce"
    name: str = "experience_structure"
    max_chars: int = 4200
    max_english_words: int = 360

    def applies_to(self, target: GateTarget) -> bool:
        return (
            target.memory_type == "experiences"
            and target.target_kind in {"gradient", "plan_item"}
            and target.after_content.strip() != ""
        )

    async def evaluate(self, target: GateTarget) -> GateDecision | None:
        issues = experience_structure_issues(
            target.after_content.strip(),
            max_chars=self.max_chars,
            max_english_words=self.max_english_words,
        )
        if not issues:
            return None
        return GateDecision(
            gate_name=self.name,
            action="reject",
            reason="; ".join(issues),
            evidence={"target_name": target.target_name, "issues": issues},
            retriable=True,
            repair_prompt=(
                "Rewrite only this Experience into the six required sections: Situation, "
                "Reminder, Procedure, Verification, Fallback, and Anti-pattern. Keep one "
                "agent-controllable root failure, bind it to runtime-visible evidence, state "
                "2-5 concrete actions, add an independent check of the actual result, and give "
                "a safe fallback. Remove evaluation internals, raw identifiers, source-case "
                "literals, internal sandbox/trace incidents, fixed trajectory control numbers, "
                "repeated checklist items, and generic 'check everything' wording. "
                "Specific defects: " + "; ".join(issues)
            ),
        )
