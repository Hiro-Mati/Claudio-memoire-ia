from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from service_app import ArkAdapterServiceConfig, create_app

from openviking.session.train import Case, ExperienceSet, Rubric, RubricCriterion
from openviking.session.train.components.dataset_service import case_to_dict, policy_set_to_dict


def make_case() -> Case:
    return Case(
        name="adapter case",
        task_signature="adapter-case-signature",
        input={"task_id": "case-1", "user_query": "hello"},
        rubric=Rubric(
            name="platform",
            description="",
            criteria=[
                RubricCriterion(
                    name="platform",
                    description="",
                    required=True,
                    weight=1.0,
                )
            ],
        ),
        metadata={"platform_case_id": "case-1"},
    )


class StaticRepository:
    async def cases_for_split(self, split: str) -> list[Case]:
        return [make_case()] if split == "train" else []


class CompletedPlatformClient:
    def __init__(self) -> None:
        self.completed = False

    async def get_training_task(self, task_id: str) -> dict[str, Any]:
        assert task_id == "task-1"
        return {"task_id": task_id, "status": "running", "current_step": "OV_WAIT"}

    async def complete_external_training(
        self,
        task_id: str,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        assert task_id == "task-1"
        assert idempotency_key.endswith(":task-1:complete")
        self.completed = True
        return {"task_id": task_id, "status": "succeeded"}

    async def submit_rollout_eval(
        self,
        task_id: str,
        *,
        body: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        assert task_id == "task-1"
        assert body["case_ids"] == ["case-1"]
        assert idempotency_key
        return {
            "batch_rollout_id": "batch-1",
            "case_rollouts": [{"case_id": "case-1", "case_rollout_id": "case-rollout-1"}],
        }

    async def get_case_rollout(self, task_id: str, case_rollout_id: str) -> dict[str, Any]:
        return {
            "status": "completed",
            "result": {
                "final_answer": "done",
                "messages": [
                    {"id": "m1", "role": "user", "content": "hello"},
                    {"id": "m2", "role": "assistant", "content": "done"},
                ],
                "evaluation": {"passed": True, "score": 1.0},
                "evaluator_status": "succeeded",
            },
        }


@pytest.mark.asyncio
async def test_generic_service_contract_executes_platform_rollout() -> None:
    platform_client = CompletedPlatformClient()
    app = create_app(
        client=platform_client,  # type: ignore[arg-type]
        repository=StaticRepository(),  # type: ignore[arg-type]
        config=ArkAdapterServiceConfig(
            dataset="ark4-0",
            domain="ark",
            platform_task_id="task-1",
            rollout_concurrency=2,
            rollout_poll_interval_seconds=0.001,
            rollout_timeout_seconds=1,
            admin_token="admin-secret",
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://adapter.test",
    ) as client:
        case_response = await client.post(
            "/v1/cases/query",
            json={
                "dataset": "ark4-0",
                "domain": "ark",
                "split": "train",
                "limit": 10,
                "filters": {},
            },
        )
        assert case_response.status_code == 200
        assert len(case_response.json()["cases"]) == 1

        execute_response = await client.post(
            "/v1/rollouts/execute",
            json={
                "case": case_to_dict(make_case()),
                "policy_set": policy_set_to_dict(
                    ExperienceSet(
                        root_uri="viking://user/memories/experiences",
                        policies=[],
                    )
                ),
                "execution_context": {
                    "policy_snapshot_id": "snapshot-1",
                    "metadata": {"training": True, "epoch": 0},
                },
                "options": {},
            },
        )
        assert execute_response.status_code == 200
        execution_id = execute_response.json()["execution_id"]

        for _ in range(100):
            poll_response = await client.get(f"/v1/rollouts/executions/{execution_id}")
            payload = poll_response.json()
            if payload["status"] == "completed":
                break
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("generic rollout execution did not complete")

        assert payload["rollout"]["evaluation"]["passed"] is True
        assert [message["role"] for message in payload["rollout"]["messages"]] == [
            "user",
            "assistant",
        ]

        unauthorized = await client.get("/admin/platform-task")
        assert unauthorized.status_code == 401

        admin_headers = {"X-Ark4-Admin-Token": "admin-secret"}
        task_response = await client.get("/admin/platform-task", headers=admin_headers)
        assert task_response.status_code == 200
        assert task_response.json()["task_id"] == "task-1"

        complete_response = await client.post(
            "/admin/external-training-completed",
            headers=admin_headers,
        )
        assert complete_response.status_code == 200
        assert complete_response.json()["completion"]["status"] == "succeeded"
        assert platform_client.completed is True
