# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Compile API models and external task provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from openviking.server.identity import RequestContext
from openviking.service.external_task_service import (
    ExternalTaskError,
    ExternalTaskService,
    ExternalTaskSnapshot,
)
from openviking.service.task_tracker import TaskRecord, TaskStatus, get_task_tracker
from openviking_cli.exceptions import UnauthenticatedError, UnavailableError
from openviking_cli.utils.config.open_viking_config import CompileApiConfig

_EXTERNAL_ACTIVE_STATUSES = frozenset({"accepted", "pending", "running", "committing"})


class CompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: list[str] = Field(alias="from", min_length=1)
    to: str = Field(min_length=1)
    skill: str = Field(min_length=1)
    reason: str | None = None
    runtime_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _normalize(self) -> "CompileRequest":
        sources: list[str] = []
        for source in self.from_:
            normalized = source.strip()
            if not normalized:
                raise ValueError("from must not contain empty values")
            if normalized not in sources:
                sources.append(normalized)
        self.from_ = sources
        self.to = self.to.strip()
        self.skill = self.skill.strip()
        self.reason = self.reason.strip() if self.reason and self.reason.strip() else None
        if not self.to:
            raise ValueError("to must not be empty")
        if not self.skill:
            raise ValueError("skill must not be empty")
        return self


class CompileAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: str = "accepted"
    to: str


class CompileErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class CompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: list[str] = Field(alias="from")
    to: str
    skill: str
    okf_version: str = "0.1"
    created: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    page_count: int = 0
    link_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class ExternalCompileTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    status: str
    stage: str | None = None
    result: CompileResult | None = None
    error: CompileErrorInfo | None = None


@dataclass(frozen=True)
class _CompileEndpoint:
    host: str
    api_key: str
    http_timeout_seconds: float
    poll_interval_ms: int
    gateway_auth: bool = False


class CompileAPIClient:
    """HTTP client for the external `/bot/v1/compile` API family."""

    def __init__(self, endpoint: _CompileEndpoint) -> None:
        self._endpoint = endpoint

    def _headers(
        self,
        connection: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._endpoint.api_key:
            header = "X-Gateway-Token" if self._endpoint.gateway_auth else "Authorization"
            value = (
                self._endpoint.api_key
                if self._endpoint.gateway_auth
                else f"Bearer {self._endpoint.api_key}"
            )
            headers[header] = value
        for field, header in {
            "api_key": "X-API-Key",
            "account_id": "X-OpenViking-Account",
            "user_id": "X-OpenViking-User",
            "actor_peer_id": "X-OpenViking-Actor-Peer",
        }.items():
            value = str(connection.get(field) or "").strip()
            if value:
                headers[header] = value
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    async def create(
        self,
        request: CompileRequest,
        *,
        connection: Mapping[str, Any],
        idempotency_key: str,
    ) -> CompileAccepted:
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        payload["openviking_connection"] = dict(connection)
        body = await self._request(
            "POST",
            "/bot/v1/compile",
            json=payload,
            headers=self._headers(connection, idempotency_key=idempotency_key),
        )
        return self._validate(CompileAccepted, body)

    async def get(
        self,
        task_id: str,
        *,
        connection: Mapping[str, Any],
    ) -> ExternalCompileTask:
        body = await self._request(
            "GET",
            f"/bot/v1/compile/{task_id}",
            headers=self._headers(connection),
        )
        return self._validate(ExternalCompileTask, body)

    async def cancel(
        self,
        task_id: str,
        *,
        connection: Mapping[str, Any],
    ) -> ExternalCompileTask:
        body = await self._request(
            "POST",
            f"/bot/v1/compile/{task_id}/cancel",
            json={},
            headers=self._headers(connection),
        )
        return self._validate(ExternalCompileTask, body)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any] | None = None,
    ) -> Any:
        try:
            async with httpx.AsyncClient(
                timeout=self._endpoint.http_timeout_seconds,
                trust_env=not self._endpoint.gateway_auth,
            ) as client:
                response = await client.request(
                    method,
                    f"{self._endpoint.host}{path}",
                    headers=dict(headers),
                    json=dict(json) if json is not None else None,
                )
        except httpx.RequestError as exc:
            raise ExternalTaskError("UNAVAILABLE", str(exc), transient=True) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalTaskError(
                "INVALID_RESPONSE",
                "Compile API returned a non-JSON response",
                transient=False,
            ) from exc
        if response.is_success:
            return body

        detail = body.get("detail") if isinstance(body, dict) else None
        error = body.get("error") if isinstance(body, dict) else None
        source = detail if isinstance(detail, dict) else error if isinstance(error, dict) else {}
        default_code = "UNAVAILABLE" if response.status_code >= 500 else "INVALID_ARGUMENT"
        code = str(source.get("code") or default_code)
        message = str(source.get("message") or detail or "Compile API request failed")
        status_code = response.status_code
        raise ExternalTaskError(
            code,
            message,
            transient=status_code in {408, 425, 429} or status_code >= 500,
        )

    @staticmethod
    def _validate(model: type[BaseModel], body: Any) -> Any:
        try:
            return model.model_validate(body)
        except ValidationError as exc:
            raise ExternalTaskError(
                "INVALID_RESPONSE",
                f"Compile API returned an invalid response: {exc}",
                transient=False,
            ) from exc


class CompileService:
    """Compile API facade and provider for the generic external task owner."""

    task_type = "compile"
    task_id_prefix = "cmp_"

    def __init__(self, config: CompileApiConfig, tasks: ExternalTaskService) -> None:
        self._config = config
        self._tasks = tasks
        self._local_endpoint: _CompileEndpoint | None = None

    @property
    def poll_interval_seconds(self) -> float:
        return self._endpoint().poll_interval_ms / 1000.0

    def configure_local_backend(self, host: str, gateway_token: str) -> None:
        if self._config.enable:
            return
        self._local_endpoint = _CompileEndpoint(
            host=host.rstrip("/"),
            api_key=gateway_token,
            http_timeout_seconds=10.0,
            poll_interval_ms=3000,
            gateway_auth=True,
        )

    def _endpoint(self) -> _CompileEndpoint:
        if self._config.enable:
            return _CompileEndpoint(
                host=self._config.host,
                api_key=self._config.api_key,
                http_timeout_seconds=self._config.http_timeout_seconds,
                poll_interval_ms=self._config.poll_interval_ms,
            )
        if self._local_endpoint is not None:
            return self._local_endpoint
        raise UnavailableError("compile API", "compile_api is not enabled")

    def _client(self) -> CompileAPIClient:
        return CompileAPIClient(self._endpoint())

    async def create(
        self,
        request: CompileRequest,
        *,
        connection: Mapping[str, Any],
        ctx: RequestContext,
    ) -> CompileAccepted:
        endpoint = self._endpoint()
        if not endpoint.gateway_auth and not str(connection.get("api_key") or "").strip():
            raise UnauthenticatedError("Compile requires a forwardable OpenViking API key")
        task = await self._tasks.create(
            self.task_type,
            resource_id=request.to,
            payload=request.model_dump(mode="json", by_alias=True, exclude_none=True),
            connection=connection,
            ctx=ctx,
        )
        return CompileAccepted(task_id=task.task_id, to=request.to)

    async def submit(
        self,
        ov_task_id: str,
        payload: Mapping[str, Any],
        connection: Mapping[str, Any],
    ) -> str:
        accepted = await self._client().create(
            self._validate_request(payload),
            connection=connection,
            idempotency_key=ov_task_id,
        )
        return accepted.task_id

    async def get(
        self,
        external_task_id: str,
        connection: Mapping[str, Any],
    ) -> ExternalTaskSnapshot:
        task = await self._client().get(external_task_id, connection=connection)
        return self._snapshot(task)

    async def cancel(
        self,
        external_task_id: str,
        connection: Mapping[str, Any],
    ) -> ExternalTaskSnapshot:
        task = await self._client().cancel(external_task_id, connection=connection)
        return self._snapshot(task)

    async def get_owned_task(self, task_id: str, ctx: RequestContext) -> TaskRecord | None:
        task = await get_task_tracker().get(
            task_id,
            account_id=ctx.account_id,
            user_id=ctx.user.user_id,
        )
        return task if task is not None and task.task_type == self.task_type else None

    async def cancel_owned_task(self, task_id: str, ctx: RequestContext) -> TaskRecord | None:
        task = await self.get_owned_task(task_id, ctx)
        if task is None:
            return None
        return await get_task_tracker().cancel(
            task_id,
            account_id=ctx.account_id,
            user_id=ctx.user.user_id,
        )

    @staticmethod
    def _validate_request(payload: Mapping[str, Any]) -> CompileRequest:
        try:
            return CompileRequest.model_validate(payload)
        except ValidationError as exc:
            raise ExternalTaskError(
                "INVALID_ARGUMENT",
                f"Persisted Compile request is invalid: {exc}",
                transient=False,
            ) from exc

    @staticmethod
    def _snapshot(task: ExternalCompileTask) -> ExternalTaskSnapshot:
        if task.status in _EXTERNAL_ACTIVE_STATUSES:
            status = "running"
        elif task.status == "cancelling":
            status = "cancelling"
        elif task.status in {"completed", "failed", "cancelled"}:
            status = task.status
        else:
            raise ExternalTaskError(
                "INVALID_RESPONSE",
                f"Unknown Compile task status: {task.status}",
                transient=False,
            )
        return ExternalTaskSnapshot(
            status=status,
            stage=task.stage or task.status,
            result=(
                task.result.model_dump(mode="json", by_alias=True)
                if task.result is not None
                else None
            ),
            error_code=task.error.code if task.error else None,
            error_message=task.error.message if task.error else None,
        )

    @staticmethod
    def compatibility_status(task: TaskRecord) -> dict[str, Any]:
        data = task.to_dict()
        status = "accepted" if task.status is TaskStatus.PENDING else task.status.value
        result: dict[str, Any] = {
            "task_id": task.task_id,
            "status": status,
            "stage": task.stage or status,
            "created_at": data["created_at_iso"],
            "updated_at": data["updated_at_iso"],
        }
        if task.result is not None:
            result["result"] = task.result
        if task.error:
            code, separator, message = task.error.partition(": ")
            result["error"] = {
                "code": code if separator else "UNKNOWN",
                "message": message if separator else task.error,
            }
        return result


__all__ = [
    "CompileAPIClient",
    "CompileAccepted",
    "CompileRequest",
    "CompileResult",
    "CompileService",
    "ExternalCompileTask",
]
