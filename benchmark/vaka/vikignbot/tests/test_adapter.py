from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.vaka.vikignbot.artifacts import _first_tsv_identity
from benchmark.vaka.vikignbot.chat_runtime import (
    ChatRunResult,
    _collect_artifacts,
    _new_session_id,
    _workspace_snapshot,
)
from benchmark.vaka.vikignbot.config import VakaVikingBotConfig
from benchmark.vaka.vikignbot.rollout_executor import VikingBotVakaRolloutExecutor
from benchmark.vaka.vikignbot.service_app import _config_for_request
from openviking.message import Message, TextPart
from openviking.session.train import (
    Case,
    ExecutionContext,
    ExperienceSet,
    Rollout,
    Rubric,
    RubricEvaluation,
)


def _case(*, deliverable_exts: list[str] | None = None) -> Case:
    return Case(
        name="case-1",
        task_signature="create workbook",
        input={
            "prompt": "Create the requested workbook.",
            "deliverable_exts": deliverable_exts or [".xlsx"],
        },
        rubric=Rubric(name="rubric", description="", criteria=[]),
        metadata={"dataset": "gdpval"},
    )


def test_session_ids_are_unique_and_include_case_name() -> None:
    first = _new_session_id("vaka", "case name")
    second = _new_session_id("vaka", "case name")

    assert first.startswith("vaka_case_name_")
    assert second.startswith("vaka_case_name_")
    assert first != second


def test_collect_artifacts_ignores_inputs_and_unrequested_formats(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    reference = input_dir / "source.xlsx"
    reference.write_bytes(b"input")
    before = _workspace_snapshot(tmp_path)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    deliverable = output_dir / "answer.xlsx"
    deliverable.write_bytes(b"answer")
    (output_dir / "notes.txt").write_text("not requested", encoding="utf-8")
    reference.write_bytes(b"modified input")

    assert _collect_artifacts(
        _case(),
        workspace_path=tmp_path,
        before=before,
    ) == [deliverable]


def test_first_tsv_identity_skips_incomplete_rows(tmp_path: Path) -> None:
    users_file = tmp_path / "users.tsv"
    users_file.write_text(
        "user_id\tapi_key\nmissing-key\t\nuser-2\tkey-2\n",
        encoding="utf-8",
    )

    identity = _first_tsv_identity(users_file)

    assert identity.user_id == "user-2"
    assert identity.api_key == "key-2"


def test_request_options_override_chat_config() -> None:
    base = VakaVikingBotConfig(skip_scoring=True)

    configured = _config_for_request(
        base,
        {"max_iterations": 7, "config_path": "~/custom-ov.conf"},
    )

    assert configured.max_iterations == 7
    assert configured.vikingbot_config_path == Path("~/custom-ov.conf").expanduser()
    assert base.max_iterations == 30


@pytest.mark.asyncio
async def test_executor_returns_standard_rollout_and_reused_evaluation(
    tmp_path: Path,
) -> None:
    case = _case()
    workspace = tmp_path / "session"
    workspace.mkdir()
    artifact = workspace / "answer.xlsx"
    artifact.write_bytes(b"answer")
    evaluation = RubricEvaluation(
        passed=False,
        score=0.75,
        criterion_results=[],
        metadata={"judge": "home_test"},
    )

    class FakeRuntime:
        async def run(self, received_case: Case) -> ChatRunResult:
            assert received_case is case
            return ChatRunResult(
                session_id="vaka-session-1",
                final_answer="Done",
                messages=[
                    Message(
                        id="assistant-1",
                        role="assistant",
                        parts=[TextPart(text="Done")],
                    )
                ],
                workspace_path=workspace,
                artifact_paths=[artifact],
                duration_seconds=12.5,
                token_usage={"total_tokens": 123},
            )

    class FakeEvaluator:
        async def evaluate(self, **kwargs):
            assert kwargs["execution_id"] == "vaka-session-1"
            assert kwargs["artifact_paths"] == [artifact]
            return (
                evaluation,
                [{"artifact_id": "artifact-1", "file_name": "answer.xlsx"}],
                "completed",
            )

    executor = VikingBotVakaRolloutExecutor(
        config=VakaVikingBotConfig(
            skip_scoring=True,
            keep_session_workspaces=False,
        ),
        chat_runtime=FakeRuntime(),
        evaluator=FakeEvaluator(),
    )
    rollouts = await executor.execute(
        [case],
        ExperienceSet(root_uri="viking://user/default/memories", policies=[]),
        ExecutionContext(policy_snapshot_id="snapshot-1", metadata={"stage": "test"}),
    )

    assert len(rollouts) == 1
    rollout: Rollout = rollouts[0]
    assert rollout.evaluation is evaluation
    assert rollout.policy_snapshot_id == "snapshot-1"
    assert rollout.metadata["rollout_backend"] == "vikingbot_chat"
    assert rollout.metadata["vikingbot_session_id"] == "vaka-session-1"
    assert rollout.metadata["artifact_count"] == 1
    assert not workspace.exists()
