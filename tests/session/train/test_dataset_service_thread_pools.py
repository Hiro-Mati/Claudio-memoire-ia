from __future__ import annotations

import asyncio
import threading

import pytest

from openviking.session.train import Case, Rubric
from openviking.session.train.components.dataset_service import (
    RolloutExecuteRequest,
    _execute_rollout_request_hosted,
    create_dataset_service_app,
)


def _case() -> Case:
    return Case(
        name="thread-pool-case",
        task_signature="thread-pool-case",
        input={},
        rubric=Rubric(name="rubric", description="", criteria=[]),
    )


def _request() -> RolloutExecuteRequest:
    return RolloutExecuteRequest(
        case={},
        policy_set={
            "root_uri": "viking://user/memories/experiences",
            "policies": [],
        },
        execution_context={"policy_snapshot_id": "snapshot", "metadata": {}},
    )


async def _run_shutdown_handlers(app) -> None:
    for handler in app.router.on_shutdown:
        await handler()


@pytest.mark.asyncio
async def test_hosted_rollout_workers_share_one_bounded_inner_executor() -> None:
    observed_executors = []
    inner_barrier = threading.Barrier(2)

    class Executor:
        async def execute(self, cases, policy_set, context):
            del cases, policy_set, context
            await asyncio.to_thread(inner_barrier.wait)
            observed_executors.append(asyncio.get_running_loop()._default_executor)
            return [threading.get_ident()]

    app = create_dataset_service_app(
        service_name="test",
        make_case_loader=lambda *args, **kwargs: None,
        make_rollout_executor=lambda options: Executor(),
        rollout_thread_workers=2,
    )

    try:
        await asyncio.gather(
            _execute_rollout_request_hosted(app, _request(), _case()),
            _execute_rollout_request_hosted(app, _request(), _case()),
        )

        assert len({id(executor) for executor in observed_executors}) == 1
        assert observed_executors[0] is app.state.rollout_inner_thread_pool
        assert app.state.rollout_inner_thread_pool is not app.state.rollout_thread_pool
        assert app.state.rollout_inner_thread_pool._max_workers == 2
    finally:
        await _run_shutdown_handlers(app)


@pytest.mark.asyncio
async def test_dataset_service_shutdown_closes_both_rollout_thread_pools() -> None:
    app = create_dataset_service_app(
        service_name="test",
        make_case_loader=lambda *args, **kwargs: None,
        make_rollout_executor=lambda options: None,
        rollout_thread_workers=2,
    )
    outer_pool = app.state.rollout_thread_pool
    inner_pool = app.state.rollout_inner_thread_pool

    await _run_shutdown_handlers(app)

    assert outer_pool._shutdown is True
    assert inner_pool._shutdown is True
