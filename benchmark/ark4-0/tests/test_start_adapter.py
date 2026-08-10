from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from start_adapter import SCRIPT_DIR, load_config, parse_args, prepare_platform_task


def write_config(path: Path, *, task_id: str = "") -> None:
    path.write_text(
        json.dumps(
            {
                "state_file": "state.local.json",
                "service": {"port": 1944, "admin_token": "secret"},
                "platform": {
                    "gateway_base_url": "https://gateway.test",
                    "api_key": "platform-secret",
                    "project_id": "00000000000000000000000000000001",
                },
                "training_task": {"task_id": task_id},
                "casehub": {"dataset_ids": ["dataset-1"], "case_ids": ["case-1"]},
                "rollout": {"require_messages_for_training": True},
            }
        ),
        encoding="utf-8",
    )


def test_load_config_uses_only_file_and_resolves_state_path(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)

    config = load_config(path)

    assert config.platform.gateway_base_url == "https://gateway.test"
    assert config.platform.api_key == "platform-secret"
    assert config.casehub.dataset_ids == ["dataset-1"]
    assert config.casehub.case_ids == ["case-1"]
    assert config.service.dataset == "ark4-0"
    assert config.state_file == tmp_path / "state.local.json"


def test_config_argument_defaults_next_to_start_script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["start_adapter.py"])

    args = parse_args()

    assert Path(args.config) == SCRIPT_DIR / "adapter_config.local.json"


def test_load_config_rejects_protected_rollout_header(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rollout"]["extra_header"] = {"x-tt-backend": "evolution"}
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="protected header"):
        load_config(path)


def test_load_config_rejects_project_display_name(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["platform"]["project_id"] = "ov-ark-test"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="not the project display name"):
        load_config(path)


def test_load_config_resolves_memory_key_and_matching_target(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    ov_config = tmp_path / "openviking.conf"
    ov_config.write_text(
        json.dumps({"bot": {"ov_server": {"api_key": "local-user-key"}}}),
        encoding="utf-8",
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rollout"]["runtime_params"] = {
        "memory": {
            "enabled": True,
            "mode": "read_only",
            "openviking_target": "ov-ark-test",
        }
    }
    raw["memory_proxy"] = {
        "enabled": True,
        "openviking_target": "ov-ark-test",
        "openviking_config_file": "openviking.conf",
        "openviking_api_key_json_path": "bot.ov_server.api_key",
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_config(path)

    assert config.memory_proxy.enabled is True
    assert config.memory_proxy.openviking_api_key == "local-user-key"
    assert config.memory_proxy.event_log_file == tmp_path / "memory_proxy_events.local.jsonl"


def test_load_config_rejects_mismatched_memory_target(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rollout"]["runtime_params"] = {
        "memory": {"enabled": True, "openviking_target": "target-a"}
    }
    raw["memory_proxy"] = {
        "enabled": True,
        "openviking_target": "target-b",
        "openviking_api_key": "local-user-key",
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must equal"):
        load_config(path)


class FakeTaskClient:
    def __init__(self) -> None:
        self.created_body: dict[str, Any] | None = None

    async def create_training_task(self, body: dict[str, Any]) -> dict[str, Any]:
        self.created_body = body
        return {"task_id": "task-created", "status": "pending"}

    async def wait_for_ov_wait(
        self,
        task_id: str,
        *,
        poll_interval_seconds: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        assert task_id == "task-created"
        assert poll_interval_seconds > 0
        assert timeout_seconds > 0
        return {"task_id": task_id, "status": "running", "current_step": "OV_WAIT"}


@pytest.mark.asyncio
async def test_prepare_platform_task_creates_and_persists_task(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    config = load_config(path)
    client = FakeTaskClient()

    task_id = await prepare_platform_task(config, client)  # type: ignore[arg-type]

    assert task_id == "task-created"
    assert client.created_body is not None
    assert client.created_body["casehub_dataset_ids"] == ["dataset-1"]
    state = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert state["platform_task_id"] == "task-created"
    assert state["status"] == "ov_wait"
