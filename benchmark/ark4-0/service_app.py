#!/usr/bin/env python3
"""Generic OpenViking dataset-service facade over the Ark platform APIs."""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from typing import Any

from case_loader import ArkCaseLoader, ArkCaseRepository, task_indices_from_filters
from fastapi import FastAPI, HTTPException, Request
from memory_proxy import MemoryProxyConfig, install_memory_proxy
from platform_client import TrainingPlatformClient
from rollout_executor import ArkRolloutExecutor

from openviking.session.train.components.dataset_service import create_dataset_service_app


@dataclass(slots=True)
class ArkAdapterServiceConfig:
    dataset: str
    domain: str
    platform_task_id: str
    rollout_concurrency: int = 16
    rollout_poll_interval_seconds: float = 2.0
    rollout_timeout_seconds: float = 3600.0
    connector_config: dict[str, Any] = field(default_factory=dict)
    runtime_params: dict[str, Any] = field(default_factory=dict)
    extra_header: dict[str, Any] = field(default_factory=dict)
    idempotency_namespace: str = "openviking-ark4"
    require_messages_for_training: bool = True
    admin_token: str = ""
    memory_proxy: MemoryProxyConfig = field(default_factory=MemoryProxyConfig)

    def __post_init__(self) -> None:
        if not str(self.dataset or "").strip():
            raise ValueError("dataset is required")
        if not str(self.domain or "").strip():
            raise ValueError("domain is required")
        if not str(self.platform_task_id or "").strip():
            raise ValueError("platform_task_id is required")
        if self.rollout_concurrency <= 0:
            raise ValueError("rollout_concurrency must be > 0")


def create_app(
    *,
    client: TrainingPlatformClient,
    repository: ArkCaseRepository,
    config: ArkAdapterServiceConfig,
) -> FastAPI:
    """Create the localhost compatibility service consumed by run_batch_train_eval."""

    def make_case_loader(
        dataset: str,
        domain: str,
        split: str,
        filters: dict[str, Any],
    ) -> ArkCaseLoader:
        if dataset != config.dataset:
            raise ValueError(f"Unsupported dataset: {dataset}")
        if domain != config.domain:
            raise ValueError(f"Unsupported domain: {domain}")
        return ArkCaseLoader(
            repository=repository,
            split=split,
            task_indices=task_indices_from_filters(filters),
        )

    def make_rollout_executor(options: dict[str, Any]) -> ArkRolloutExecutor:
        del options
        return ArkRolloutExecutor(
            client=client,
            platform_task_id=config.platform_task_id,
            connector_config=config.connector_config,
            runtime_params=config.runtime_params,
            extra_header=config.extra_header,
            poll_interval_seconds=config.rollout_poll_interval_seconds,
            timeout_seconds=config.rollout_timeout_seconds,
            idempotency_namespace=config.idempotency_namespace,
            require_messages_for_training=config.require_messages_for_training,
            concurrency=1,
        )

    app = create_dataset_service_app(
        service_name="ark4-platform-adapter",
        make_case_loader=make_case_loader,
        make_rollout_executor=make_rollout_executor,
        max_rollout_concurrency=config.rollout_concurrency,
        rollout_thread_workers=None,
    )
    app.state.platform_task_id = config.platform_task_id
    app.state.dataset = config.dataset
    app.state.domain = config.domain
    memory_proxy = install_memory_proxy(app, config.memory_proxy)

    def authorize_admin(request: Request) -> None:
        if not config.admin_token:
            return
        supplied = request.headers.get("X-Ark4-Admin-Token", "")
        if not hmac.compare_digest(supplied, config.admin_token):
            raise HTTPException(status_code=401, detail="invalid Ark4 admin token")

    @app.get("/admin/platform-task")
    async def get_platform_task(request: Request) -> dict[str, Any]:
        authorize_admin(request)
        task = await client.get_training_task(config.platform_task_id)
        return {"task_id": config.platform_task_id, "task": task}

    @app.post("/admin/external-training-completed")
    async def complete_external_training(request: Request) -> dict[str, Any]:
        authorize_admin(request)
        completion = await client.complete_external_training(
            config.platform_task_id,
            idempotency_key=(f"{config.idempotency_namespace}:{config.platform_task_id}:complete"),
        )
        return {"task_id": config.platform_task_id, "completion": completion}

    @app.get("/admin/memory-proxy")
    async def get_memory_proxy(request: Request) -> dict[str, Any]:
        authorize_admin(request)
        return {
            "enabled": memory_proxy is not None,
            "openviking_target": config.memory_proxy.openviking_target,
            "openviking_url": config.memory_proxy.openviking_url,
            "event_log_file": (
                str(config.memory_proxy.event_log_file)
                if config.memory_proxy.event_log_file is not None
                else None
            ),
            "call_count": memory_proxy.call_count if memory_proxy is not None else 0,
        }

    return app
