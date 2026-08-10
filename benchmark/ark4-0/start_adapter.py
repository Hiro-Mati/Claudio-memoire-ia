#!/usr/bin/env python3
"""Start the standalone Ark platform adapter for OpenViking batch training."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_loader import ArkCaseRepository  # noqa: E402
from memory_proxy import MemoryProxyConfig  # noqa: E402
from platform_client import PlatformClientConfig, TrainingPlatformClient  # noqa: E402
from service_app import ArkAdapterServiceConfig, create_app  # noqa: E402

_PROTECTED_ROLLOUT_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "host",
        "content-length",
        "content-type",
        "connection",
        "proxy-connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        "x-admin-token",
        "x-homepage-api-key",
        "x-user-id",
        "x-tt-backend",
        "x-tt-env",
    }
)
_PROJECT_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    host: str = "127.0.0.1"
    port: int = 1944
    dataset: str = "ark4-0"
    domain: str = "ark"
    rollout_concurrency: int = 16
    admin_token: str = ""
    log_level: str = "info"


@dataclass(frozen=True, slots=True)
class PlatformSettings:
    gateway_base_url: str
    api_key: str = ""
    project_id: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    request_timeout_seconds: float = 120.0


@dataclass(frozen=True, slots=True)
class TrainingTaskSettings:
    task_id: str = ""
    task_name: str = "openviking_ark4_external_training"
    workflow_id: str = "ov_external_training"
    agent_id: str = "ark"
    evaluator_id: str = "rollout_builtin@v1"
    ready_poll_interval_seconds: float = 2.0
    ready_timeout_seconds: float = 900.0


@dataclass(frozen=True, slots=True)
class CaseHubSettings:
    dataset_ids: list[str]
    case_ids: list[str] = field(default_factory=list)
    caseset_id: str = ""
    split_field: str = ""
    default_split: str = "train"
    page_size: int = 1000
    detail_concurrency: int = 20


@dataclass(frozen=True, slots=True)
class RolloutSettings:
    poll_interval_seconds: float = 2.0
    timeout_seconds: float = 3600.0
    connector_config: dict[str, Any] = field(default_factory=dict)
    runtime_params: dict[str, Any] = field(default_factory=dict)
    extra_header: dict[str, Any] = field(default_factory=dict)
    idempotency_namespace: str = "openviking-ark4"
    require_messages_for_training: bool = True


@dataclass(frozen=True, slots=True)
class AdapterFileConfig:
    path: Path
    state_file: Path
    service: ServiceSettings
    platform: PlatformSettings
    training_task: TrainingTaskSettings
    casehub: CaseHubSettings
    rollout: RolloutSettings
    memory_proxy: MemoryProxyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start the standalone Ark adapter consumed by OpenViking's native "
            "run_batch_train_eval command."
        )
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "adapter_config.local.json"),
        help=(
            "Path to adapter JSON configuration "
            "(default: adapter_config.local.json next to this script)"
        ),
    )
    return parser.parse_args()


def load_config(path: str | Path) -> AdapterFileConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"adapter config does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"adapter config is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("adapter config root must be a JSON object")

    service_raw = _object(raw, "service")
    platform_raw = _object(raw, "platform")
    task_raw = _object(raw, "training_task")
    casehub_raw = _object(raw, "casehub")
    rollout_raw = _object(raw, "rollout")
    memory_raw = _object(raw, "memory_proxy")

    state_value = str(raw.get("state_file") or "adapter_state.local.json").strip()
    state_file = Path(state_value).expanduser()
    if not state_file.is_absolute():
        state_file = config_path.parent / state_file

    service = ServiceSettings(
        host=_text(service_raw.get("host"), "service.host", default="127.0.0.1"),
        port=_integer(service_raw.get("port"), "service.port", default=1944, minimum=1),
        dataset=_text(service_raw.get("dataset"), "service.dataset", default="ark4-0"),
        domain=_text(service_raw.get("domain"), "service.domain", default="ark"),
        rollout_concurrency=_integer(
            service_raw.get("rollout_concurrency"),
            "service.rollout_concurrency",
            default=16,
            minimum=1,
        ),
        admin_token=_optional_text(service_raw.get("admin_token")),
        log_level=_text(service_raw.get("log_level"), "service.log_level", default="info"),
    )
    if service.port > 65535:
        raise ValueError("service.port must be <= 65535")

    project_id = _optional_text(platform_raw.get("project_id"))
    _validate_project_id(project_id)
    platform = PlatformSettings(
        gateway_base_url=_text(
            platform_raw.get("gateway_base_url"),
            "platform.gateway_base_url",
        ),
        api_key=_optional_text(platform_raw.get("api_key")),
        project_id=project_id,
        headers=_string_dict(
            platform_raw.get("headers"),
            "platform.headers",
        ),
        request_timeout_seconds=_number(
            platform_raw.get("request_timeout_seconds"),
            "platform.request_timeout_seconds",
            default=120.0,
            minimum=0.001,
        ),
    )
    training_task = TrainingTaskSettings(
        task_id=_optional_text(task_raw.get("task_id")),
        task_name=_text(
            task_raw.get("task_name"),
            "training_task.task_name",
            default="openviking_ark4_external_training",
        ),
        workflow_id=_text(
            task_raw.get("workflow_id"),
            "training_task.workflow_id",
            default="ov_external_training",
        ),
        agent_id=_text(task_raw.get("agent_id"), "training_task.agent_id", default="ark"),
        evaluator_id=_text(
            task_raw.get("evaluator_id"),
            "training_task.evaluator_id",
            default="rollout_builtin@v1",
        ),
        ready_poll_interval_seconds=_number(
            task_raw.get("ready_poll_interval_seconds"),
            "training_task.ready_poll_interval_seconds",
            default=2.0,
            minimum=0.001,
        ),
        ready_timeout_seconds=_number(
            task_raw.get("ready_timeout_seconds"),
            "training_task.ready_timeout_seconds",
            default=900.0,
            minimum=0.001,
        ),
    )
    dataset_ids = _string_list(casehub_raw.get("dataset_ids"), "casehub.dataset_ids")
    caseset_id = _optional_text(casehub_raw.get("caseset_id"))
    if not dataset_ids and not caseset_id:
        raise ValueError("casehub.dataset_ids or casehub.caseset_id is required")
    casehub = CaseHubSettings(
        dataset_ids=dataset_ids,
        case_ids=_string_list(casehub_raw.get("case_ids"), "casehub.case_ids"),
        caseset_id=caseset_id,
        split_field=_optional_text(casehub_raw.get("split_field")),
        default_split=_text(
            casehub_raw.get("default_split"),
            "casehub.default_split",
            default="train",
        ),
        page_size=_integer(
            casehub_raw.get("page_size"),
            "casehub.page_size",
            default=1000,
            minimum=1,
        ),
        detail_concurrency=_integer(
            casehub_raw.get("detail_concurrency"),
            "casehub.detail_concurrency",
            default=20,
            minimum=1,
        ),
    )
    extra_header = _any_dict(rollout_raw.get("extra_header"), "rollout.extra_header")
    _validate_extra_header(extra_header)
    rollout = RolloutSettings(
        poll_interval_seconds=_number(
            rollout_raw.get("poll_interval_seconds"),
            "rollout.poll_interval_seconds",
            default=2.0,
            minimum=0.001,
        ),
        timeout_seconds=_number(
            rollout_raw.get("timeout_seconds"),
            "rollout.timeout_seconds",
            default=3600.0,
            minimum=0.001,
        ),
        connector_config=_any_dict(
            rollout_raw.get("connector_config"),
            "rollout.connector_config",
        ),
        runtime_params=_any_dict(rollout_raw.get("runtime_params"), "rollout.runtime_params"),
        extra_header=extra_header,
        idempotency_namespace=_text(
            rollout_raw.get("idempotency_namespace"),
            "rollout.idempotency_namespace",
            default="openviking-ark4",
        ),
        require_messages_for_training=_boolean(
            rollout_raw.get("require_messages_for_training"),
            "rollout.require_messages_for_training",
            default=True,
        ),
    )
    memory_enabled = _boolean(
        memory_raw.get("enabled"),
        "memory_proxy.enabled",
        default=False,
    )
    memory_target = _optional_text(memory_raw.get("openviking_target"))
    event_log_value = _optional_text(memory_raw.get("event_log_file"))
    event_log_file = _relative_path(
        event_log_value or "memory_proxy_events.local.jsonl",
        base=config_path.parent,
    )
    openviking_api_key = _optional_text(memory_raw.get("openviking_api_key"))
    key_file_value = _optional_text(memory_raw.get("openviking_config_file"))
    if not openviking_api_key and key_file_value:
        key_file = _relative_path(key_file_value, base=config_path.parent)
        key_path = _text(
            memory_raw.get("openviking_api_key_json_path"),
            "memory_proxy.openviking_api_key_json_path",
            default="ov_server.api_key",
        )
        openviking_api_key = _read_json_text(key_file, key_path)
    memory_proxy = MemoryProxyConfig(
        enabled=memory_enabled,
        openviking_target=memory_target,
        openviking_url=_text(
            memory_raw.get("openviking_url"),
            "memory_proxy.openviking_url",
            default="http://127.0.0.1:1933",
        ),
        openviking_api_key=openviking_api_key,
        timeout_seconds=_number(
            memory_raw.get("timeout_seconds"),
            "memory_proxy.timeout_seconds",
            default=180.0,
            minimum=0.001,
        ),
        event_log_file=event_log_file,
    )
    _validate_runtime_memory_target(rollout.runtime_params, memory_proxy)
    return AdapterFileConfig(
        path=config_path,
        state_file=state_file.resolve(),
        service=service,
        platform=platform,
        training_task=training_task,
        casehub=casehub,
        rollout=rollout,
        memory_proxy=memory_proxy,
    )


async def run(config: AdapterFileConfig) -> None:
    client_config = PlatformClientConfig(
        gateway_base_url=config.platform.gateway_base_url,
        api_key=config.platform.api_key or None,
        project_id=config.platform.project_id or None,
        headers=config.platform.headers,
        timeout_seconds=config.platform.request_timeout_seconds,
    )
    async with TrainingPlatformClient(client_config) as client:
        task_id = await prepare_platform_task(config, client)
        repository = ArkCaseRepository(
            client=client,
            dataset_ids=config.casehub.dataset_ids,
            case_ids=config.casehub.case_ids,
            caseset_id=config.casehub.caseset_id or None,
            split_field=config.casehub.split_field or None,
            page_size=config.casehub.page_size,
            detail_concurrency=config.casehub.detail_concurrency,
            default_split=config.casehub.default_split,
        )
        cases = await repository.all_cases()
        print(f"[ark4-adapter] loaded {len(cases)} CaseHub case(s)", flush=True)

        app = create_app(
            client=client,
            repository=repository,
            config=ArkAdapterServiceConfig(
                dataset=config.service.dataset,
                domain=config.service.domain,
                platform_task_id=task_id,
                rollout_concurrency=config.service.rollout_concurrency,
                rollout_poll_interval_seconds=config.rollout.poll_interval_seconds,
                rollout_timeout_seconds=config.rollout.timeout_seconds,
                connector_config=config.rollout.connector_config,
                runtime_params=config.rollout.runtime_params,
                extra_header=config.rollout.extra_header,
                idempotency_namespace=config.rollout.idempotency_namespace,
                require_messages_for_training=config.rollout.require_messages_for_training,
                admin_token=config.service.admin_token,
                memory_proxy=config.memory_proxy,
            ),
        )
        server_url = f"http://{config.service.host}:{config.service.port}"
        _merge_state(
            config.state_file,
            {
                "platform_task_id": task_id,
                "status": "serving",
                "service_url": server_url,
                "config_file": str(config.path),
            },
        )
        print(f"[ark4-adapter] platform task: {task_id}", flush=True)
        print(f"[ark4-adapter] listening at {server_url}", flush=True)
        print(
            f"[ark4-adapter] OpenViking argument: --benchmark-service-url {server_url}",
            flush=True,
        )
        if config.memory_proxy.enabled:
            print(
                "[ark4-adapter] memory callback ready: "
                f"target={config.memory_proxy.openviking_target} "
                f"local={server_url}/api/v1/search/find",
                flush=True,
            )
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=config.service.host,
                port=config.service.port,
                log_level=config.service.log_level,
            )
        )
        try:
            await server.serve()
        finally:
            _merge_state(config.state_file, {"status": "stopped"})


async def prepare_platform_task(
    config: AdapterFileConfig,
    client: TrainingPlatformClient,
) -> str:
    state = _load_state(config.state_file)
    task_id = config.training_task.task_id or _optional_text(state.get("platform_task_id"))
    if not task_id:
        if not config.casehub.dataset_ids:
            raise ValueError(
                "casehub.dataset_ids is required when creating a new platform training task"
            )
        body: dict[str, Any] = {
            "task_name": config.training_task.task_name,
            "workflow_id": config.training_task.workflow_id,
            "agent_id": config.training_task.agent_id,
            "casehub_dataset_ids": config.casehub.dataset_ids,
            "evaluator_id": config.training_task.evaluator_id,
        }
        if config.casehub.caseset_id:
            body["casehub_caseset_id"] = config.casehub.caseset_id
        created = await client.create_training_task(body)
        task_id = str(created["task_id"])
        _merge_state(
            config.state_file,
            {"platform_task_id": task_id, "platform_task": created, "status": "created"},
        )
        print(f"[ark4-adapter] created platform task {task_id}", flush=True)
    else:
        print(f"[ark4-adapter] resuming platform task {task_id}", flush=True)

    ready = await client.wait_for_ov_wait(
        task_id,
        poll_interval_seconds=config.training_task.ready_poll_interval_seconds,
        timeout_seconds=config.training_task.ready_timeout_seconds,
    )
    _merge_state(
        config.state_file,
        {"platform_task_id": task_id, "platform_task": ready, "status": "ov_wait"},
    )
    print(f"[ark4-adapter] platform task {task_id} is ready at OV_WAIT", flush=True)
    return task_id


def _object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return dict(value)


def _text(value: Any, label: str, *, default: str | None = None) -> str:
    if value is None and default is not None:
        value = default
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _optional_text(value: Any) -> str:
    return str(value or "").strip()


def _integer(value: Any, label: str, *, default: int, minimum: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return parsed


def _number(value: Any, label: str, *, default: float, minimum: float) -> float:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if parsed < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return parsed


def _boolean(value: Any, label: str, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{label} must be a boolean")


def _string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    result = [_optional_text(item) for item in value]
    return list(dict.fromkeys(item for item in result if item))


def _any_dict(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _string_dict(value: Any, label: str) -> dict[str, str]:
    raw = _any_dict(value, label)
    return {str(key): str(item) for key, item in raw.items()}


def _validate_extra_header(headers: dict[str, Any]) -> None:
    for raw_key, raw_value in headers.items():
        key = str(raw_key).strip()
        value = str(raw_value)
        if not key:
            raise ValueError("rollout.extra_header keys must be non-empty")
        if "\r" in key or "\n" in key or "\r" in value or "\n" in value:
            raise ValueError("rollout.extra_header must not contain CR/LF")
        if key.lower() in _PROTECTED_ROLLOUT_HEADERS:
            raise ValueError(
                f"rollout.extra_header cannot override protected header {key!r}; "
                "use the platform Task binding for lane headers"
            )


def _validate_project_id(project_id: str) -> None:
    if not project_id:
        return
    if not _PROJECT_ID_PATTERN.fullmatch(project_id):
        raise ValueError(
            "platform.project_id must be the 32-character project_id returned by "
            "GET /api/projects, not the project display name"
        )


def _validate_runtime_memory_target(
    runtime_params: dict[str, Any],
    memory_proxy: MemoryProxyConfig,
) -> None:
    if not memory_proxy.enabled:
        return
    memory = runtime_params.get("memory")
    if not isinstance(memory, dict):
        raise ValueError(
            "rollout.runtime_params.memory must be configured when memory_proxy is enabled"
        )
    if memory.get("enabled") is not True:
        raise ValueError(
            "rollout.runtime_params.memory.enabled must be true when memory_proxy is enabled"
        )
    runtime_target = _optional_text(memory.get("openviking_target"))
    if runtime_target != memory_proxy.openviking_target:
        raise ValueError(
            "rollout.runtime_params.memory.openviking_target must equal "
            "memory_proxy.openviking_target"
        )


def _relative_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _read_json_text(path: Path, dotted_path: str) -> str:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"memory proxy config file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read memory proxy config file {path}: {exc}") from exc
    for segment in dotted_path.split("."):
        if not isinstance(value, dict) or segment not in value:
            raise ValueError(f"memory proxy config file {path} has no JSON path {dotted_path!r}")
        value = value[segment]
    result = _optional_text(value)
    if not result:
        raise ValueError(f"memory proxy config file {path} has an empty JSON path {dotted_path!r}")
    return result


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _merge_state(path: Path, updates: dict[str, Any]) -> None:
    value = {**_load_state(path), **updates}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(load_config(args.config)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
