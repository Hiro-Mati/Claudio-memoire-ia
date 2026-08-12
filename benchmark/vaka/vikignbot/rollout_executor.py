"""OpenViking RolloutExecutor that runs Vaka cases through Vikingbot chat."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from openviking.message import Message, TextPart
from openviking.session.train import Case, ExecutionContext, ExperienceSet, Rollout

from .chat_runtime import ChatRunResult, VikingBotChatRuntime
from .config import VakaVikingBotConfig
from .evaluator import ReusedHomeTestEvaluator


@dataclass(slots=True)
class VikingBotVakaRolloutExecutor:
    config: VakaVikingBotConfig
    chat_runtime: Any | None = None
    evaluator: Any | None = None
    concurrency: int = 1

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        if self.chat_runtime is None:
            self.chat_runtime = VikingBotChatRuntime(self.config)
        if self.evaluator is None:
            self.evaluator = ReusedHomeTestEvaluator(self.config)

    async def execute(
        self,
        cases: list[Case],
        policy_set: ExperienceSet,
        context: ExecutionContext,
    ) -> list[Rollout]:
        del policy_set
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(case: Case) -> Rollout:
            async with semaphore:
                return await self._execute_one(case, context)

        return list(await asyncio.gather(*(run_one(case) for case in cases)))

    async def _execute_one(self, case: Case, context: ExecutionContext) -> Rollout:
        run: ChatRunResult = await self.chat_runtime.run(case)
        prompt = str(case.input.get("prompt") or "")
        dataset = str(case.metadata.get("dataset") or "")
        evaluation, uploaded_artifacts, evaluator_status = await self.evaluator.evaluate(
            execution_id=run.session_id,
            prompt=prompt,
            dataset=dataset,
            final_answer=run.final_answer,
            artifact_paths=run.artifact_paths,
        )
        messages = run.messages or _fallback_messages(prompt, run.final_answer)
        rollout = Rollout(
            case=case,
            messages=messages,
            policy_snapshot_id=context.policy_snapshot_id,
            evaluation=evaluation,
            metadata={
                "rollout_backend": "vikingbot_chat",
                "vikingbot_session_id": run.session_id,
                "vikingbot_workspace": str(run.workspace_path),
                "rollout_duration_seconds": round(run.duration_seconds, 6),
                "token_usage": dict(run.token_usage),
                "artifact_count": len(uploaded_artifacts),
                "artifacts": uploaded_artifacts,
                "evaluator_backend": "home_test",
                "evaluator_status": evaluator_status,
                "execution_metadata": dict(context.metadata),
            },
        )
        if not self.config.keep_session_workspaces:
            shutil.rmtree(run.workspace_path, ignore_errors=True)
        return rollout


def _fallback_messages(prompt: str, final_answer: str) -> list[Message]:
    return [
        Message(
            id=f"vaka-user-{uuid4().hex}",
            role="user",
            parts=[TextPart(text=prompt)],
        ),
        Message(
            id=f"vaka-assistant-{uuid4().hex}",
            role="assistant",
            parts=[TextPart(text=final_answer)],
        ),
    ]
