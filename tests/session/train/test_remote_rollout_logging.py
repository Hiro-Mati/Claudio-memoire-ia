from __future__ import annotations

from pathlib import Path

import pytest

from openviking.session.train import Case, ExecutionContext, ExperienceSet, Rubric
from openviking.session.train.components.remote import RemoteRolloutExecutor


@pytest.mark.asyncio
async def test_failed_remote_rollout_retains_deterministic_vikingbot_log_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def fail_execution(self, client, case, policy_set, context):
        del self, client, case, policy_set, context
        raise RuntimeError("rollout service failed")

    monkeypatch.setattr(
        RemoteRolloutExecutor,
        "_execute_with_retry",
        fail_execution,
    )
    executor = RemoteRolloutExecutor(
        service_url="http://127.0.0.1:1944",
        continue_on_rollout_failure=True,
        options={"session_log_root": str(tmp_path)},
        show_progress=False,
    )
    case = Case(
        name="tau2 case/16:t3",
        task_signature="tau2:airline:test:16:trial:3",
        input={},
        rubric=Rubric(name="rubric", description="", criteria=[]),
    )

    rollouts = await executor.execute(
        [case],
        ExperienceSet(root_uri="viking://user/memories/experiences", policies=[]),
        ExecutionContext(
            policy_snapshot_id="snapshot",
            metadata={"epoch": 4, "rollout_stage": "final_test_rollout"},
        ),
    )

    expected = tmp_path / "final_test_rollout" / "epoch_4" / "tau2_case_16_t3.log"
    assert rollouts[0].metadata["rollout_failed"] is True
    assert rollouts[0].metadata["vikingbot_log_path"] == str(expected.resolve())
