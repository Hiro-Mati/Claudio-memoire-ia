# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace

from openviking.session.memory.experience_lifecycle import (
    experience_is_agent_visible,
    experience_lifecycle_fields,
    normalize_experience_status,
)


def _link(name: str) -> dict[str, str]:
    return {
        "from_uri": "viking://user/u/memories/experiences/recovery.md",
        "to_uri": f"viking://user/u/memories/trajectories/{name}.md",
        "link_type": "derived_from",
    }


def _gradient(
    *,
    outcome: str = "failure",
    recovery_status: str = "not_observed",
    passed: bool = False,
    score: float = 0.5,
):
    return SimpleNamespace(
        metadata={
            "trajectory_outcome": outcome,
            "recovery_status": recovery_status,
            "rubric_passed": passed,
            "rubric_score": score,
        }
    )


def test_normalizes_legacy_experience_states():
    assert normalize_experience_status("staging") == "draft"
    assert normalize_experience_status("production") == "promoted"
    assert normalize_experience_status("deprecated") == "degraded"
    assert experience_is_agent_visible("production") is True
    assert experience_is_agent_visible("draft") is False


def test_single_non_full_trajectory_stays_draft():
    fields = experience_lifecycle_fields(
        existing_policy=None,
        links=[_link("one")],
        gradients=[_gradient()],
    )

    assert fields == {
        "status": "draft",
        "source_count": 1,
        "promotion_reason": "single_trajectory_only",
    }


def test_complete_observed_recovery_promotes_immediately():
    fields = experience_lifecycle_fields(
        existing_policy=None,
        links=[_link("one")],
        gradients=[
            _gradient(
                outcome="success",
                recovery_status="observed_recovered",
                passed=True,
                score=1.0,
            )
        ],
    )

    assert fields["status"] == "promoted"
    assert fields["source_count"] == 1
    assert fields["promotion_reason"] == "complete_observed_recovery"


def test_second_independent_trajectory_promotes_existing_draft():
    existing = SimpleNamespace(
        status="draft",
        metadata={"status": "draft", "source_count": 1},
    )

    fields = experience_lifecycle_fields(
        existing_policy=existing,
        links=[_link("one"), _link("two")],
        gradients=[_gradient()],
    )

    assert fields["status"] == "promoted"
    assert fields["source_count"] == 2
    assert fields["promotion_reason"] == "multi_trajectory_confirmation"


def test_degraded_experience_is_not_revived_by_training_update():
    existing = SimpleNamespace(
        status="degraded",
        metadata={"status": "degraded", "source_count": 2},
    )

    fields = experience_lifecycle_fields(
        existing_policy=existing,
        links=[_link("one"), _link("two"), _link("three")],
        gradients=[
            _gradient(
                outcome="success",
                recovery_status="observed_recovered",
                passed=True,
                score=1.0,
            )
        ],
    )

    assert fields["status"] == "degraded"
    assert fields["source_count"] == 3
    assert fields["promotion_reason"] == "awaiting_reconfirmation"
