"""Run one Vaka case through an isolated Vikingbot chat session."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from openviking.message import Message, TextPart, ToolPart
from openviking.session.train import Case

from .config import VakaVikingBotConfig

_IGNORED_WORKSPACE_PARTS = {
    ".git",
    ".openviking",
    "input",
    "skills",
}


@dataclass(slots=True)
class ChatRunResult:
    session_id: str
    final_answer: str
    messages: list[Message]
    workspace_path: Path
    artifact_paths: list[Path]
    duration_seconds: float
    token_usage: dict[str, int]


class VikingBotChatRuntime:
    """Create a fresh Vikingbot AgentLoop and session for every rollout."""

    def __init__(self, config: VakaVikingBotConfig) -> None:
        self.config = config

    async def run(self, case: Case) -> ChatRunResult:
        session_id = _new_session_id(self.config.session_prefix, case.name)
        agent, session_key, workspace_path = await asyncio.to_thread(
            self._build_agent,
            session_id,
        )
        await agent.sandbox_manager.get_sandbox(session_key)
        workspace_path.mkdir(parents=True, exist_ok=True)
        _mount_generated_images(workspace_path)
        input_files = _stage_reference_files(case, workspace_path)
        before = _workspace_snapshot(workspace_path)
        effective_prompt = _effective_prompt(case, input_files=input_files)

        try:
            started_at = time.perf_counter()
            try:
                final_answer = await asyncio.wait_for(
                    agent.process_direct(effective_prompt, session_key=session_key),
                    timeout=self.config.rollout_timeout_seconds,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    "Vikingbot chat timed out after "
                    f"{self.config.rollout_timeout_seconds}s for session {session_id}"
                ) from exc
            duration_seconds = time.perf_counter() - started_at
            session = agent.sessions.get_or_create(session_key, skip_heartbeat=True)
            runtime_messages = list(session.messages)
            messages = _messages_from_session(runtime_messages)
            artifact_paths = _collect_artifacts(
                case,
                workspace_path=workspace_path,
                before=before,
            )
            token_usage = _aggregate_token_usage(runtime_messages)

            return ChatRunResult(
                session_id=session_id,
                final_answer=str(final_answer or ""),
                messages=messages,
                workspace_path=workspace_path,
                artifact_paths=artifact_paths,
                duration_seconds=duration_seconds,
                token_usage=token_usage,
            )
        finally:
            await agent.close_mcp()
            await agent.sandbox_manager.cleanup_session(session_key)

    def _build_agent(self, session_id: str) -> tuple[Any, Any, Path]:
        import litellm
        from vikingbot.agent.loop import AgentLoop
        from vikingbot.bus.queue import MessageBus
        from vikingbot.cli.commands import _init_bot_data, _make_provider
        from vikingbot.config.loader import ensure_config
        from vikingbot.config.schema import SandboxMode, SessionKey
        from vikingbot.sandbox.manager import SandboxManager
        from vikingbot.session.manager import SessionManager
        from vikingbot.utils.helpers import get_source_workspace_path

        config_path = self.config.vikingbot_config_path
        config = ensure_config(config_path.expanduser() if config_path else None)
        # The configured image provider may expose only a subset of the
        # OpenAI-compatible image parameters (for example, Seedream does not
        # accept DALL-E's ``style`` argument). This dedicated benchmark process
        # should omit unsupported optional arguments instead of failing the
        # whole rollout before image generation starts.
        litellm.drop_params = True
        # A rollout must never see files or chat history from another case.
        config.sandbox.mode = SandboxMode.PER_SESSION
        _init_bot_data(config)
        bus = MessageBus()
        sandbox_manager = SandboxManager(
            config,
            config.workspace_path,
            get_source_workspace_path(),
        )
        session_manager = SessionManager(
            config.bot_data_path,
            sandbox_manager=sandbox_manager,
        )
        provider = _make_provider(config)
        agent = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            model=config.agents.model,
            temperature=config.agents.temperature,
            max_iterations=self.config.max_iterations,
            memory_window=config.agents.memory_window,
            brave_api_key=config.tools.web.search.api_key or None,
            exa_api_key=None,
            gen_image_model=config.agents.gen_image_model,
            exec_config=config.tools.exec,
            cron_service=None,
            session_manager=session_manager,
            sandbox_manager=sandbox_manager,
            config=config,
            eval=True,
            mcp_servers=None,
        )
        session_key = SessionKey(
            type="cli",
            channel_id="vaka",
            chat_id=session_id,
        )
        return agent, session_key, sandbox_manager.get_workspace_path(session_key)


def _new_session_id(prefix: str, case_name: str) -> str:
    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name).strip("_.-")[:48]
    return f"{prefix}_{safe_case}_{uuid4().hex}"


def _stage_reference_files(case: Case, workspace_path: Path) -> list[Path]:
    input_dir = workspace_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    used_names: set[str] = set()
    for index, item in enumerate(case.input.get("reference_files") or []):
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("local_path") or "").strip()
        if not raw_path:
            continue
        source = Path(raw_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Vaka reference file does not exist: {source}")
        name = _unique_file_name(
            str(item.get("file_name") or source.name),
            used_names,
            fallback=f"reference_{index}{source.suffix}",
        )
        target = input_dir / name
        shutil.copy2(source, target)
        staged.append(target)
    return staged


def _unique_file_name(name: str, used: set[str], *, fallback: str) -> str:
    candidate = Path(name).name or fallback
    stem = Path(candidate).stem
    suffix = Path(candidate).suffix
    count = 1
    while candidate.casefold() in used:
        candidate = f"{stem}_{count}{suffix}"
        count += 1
    used.add(candidate.casefold())
    return candidate


def _effective_prompt(case: Case, *, input_files: list[Path]) -> str:
    prompt = str(case.input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"case {case.name} has no input.prompt")
    file_lines = [f"- `input/{path.name}`" for path in input_files]
    deliverable_exts = [str(value) for value in case.input.get("deliverable_exts") or []]
    output_hint = ", ".join(deliverable_exts) if deliverable_exts else "the requested format"
    workspace_note = [
        "",
        "## Workspace files",
        "Reference files are already available in this session workspace:",
        *(file_lines or ["- No reference files were supplied."]),
        "",
        f"Save final deliverables under `output/` using {output_hint}.",
        "Do not modify files under `input/`.",
        (
            "When `generate_image` returns `send://<filename>`, the generated "
            "file is available at `generated_images/<filename>` in this workspace. "
            "Use that local path when composing final deliverables."
        ),
    ]
    return prompt + "\n" + "\n".join(workspace_note)


def _mount_generated_images(workspace_path: Path) -> None:
    """Expose channel-oriented ``send://`` images inside the rollout workspace."""

    from vikingbot.utils import get_data_path

    image_store = get_data_path() / "images"
    image_store.mkdir(parents=True, exist_ok=True)
    mount_path = workspace_path / "generated_images"
    if mount_path.exists() or mount_path.is_symlink():
        return
    mount_path.symlink_to(image_store, target_is_directory=True)


def _workspace_snapshot(root: Path) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot[path.relative_to(root)] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _collect_artifacts(
    case: Case,
    *,
    workspace_path: Path,
    before: dict[Path, tuple[int, int]],
) -> list[Path]:
    expected_exts = {
        str(value).lower() for value in case.input.get("deliverable_exts") or [] if value
    }
    artifacts: list[Path] = []
    for path in workspace_path.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace_path)
        if any(part in _IGNORED_WORKSPACE_PARTS for part in relative.parts):
            continue
        if expected_exts and path.suffix.lower() not in expected_exts:
            continue
        stat = path.stat()
        current = (stat.st_size, stat.st_mtime_ns)
        if relative not in before or before[relative] != current:
            artifacts.append(path)
    return sorted(artifacts, key=lambda value: str(value.relative_to(workspace_path)))


def _messages_from_session(runtime_messages: list[dict[str, Any]]) -> list[Message]:
    messages: list[Message] = []
    for index, raw in enumerate(runtime_messages):
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        if role not in {"user", "assistant"}:
            continue
        parts: list[Any] = []
        content = _content_text(raw.get("content"))
        if content:
            parts.append(TextPart(text=content))
        if role == "assistant":
            for tool_index, tool in enumerate(raw.get("tools_used") or []):
                part = _tool_part(tool, fallback_id=f"tool-{index}-{tool_index}")
                if part is not None:
                    parts.append(part)
        if not parts:
            continue
        messages.append(
            Message(
                id=str(raw.get("response_id") or f"vaka-vikingbot-{index}-{uuid4().hex}"),
                role=role,  # type: ignore[arg-type]
                parts=parts,
                created_at=raw.get("timestamp"),
            )
        )
    return messages


def _tool_part(raw: Any, *, fallback_id: str) -> ToolPart | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("tool_name") or raw.get("name") or "unknown")
    tool_input = raw.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = _json_object(raw.get("args"))
    output = raw.get("result")
    if output is None:
        output = raw.get("tool_output")
    success = raw.get("execute_success")
    return ToolPart(
        tool_id=str(raw.get("tool_id") or fallback_id),
        tool_name=name,
        tool_input=tool_input,
        tool_output=_content_text(output),
        tool_status="completed" if success is not False else "error",
        duration_ms=_optional_float(raw.get("duration") or raw.get("duration_ms")),
        prompt_tokens=_optional_int(raw.get("input_token") or raw.get("prompt_tokens")),
        completion_tokens=_optional_int(raw.get("output_token") or raw.get("completion_tokens")),
    )


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(_content_text(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _aggregate_token_usage(messages: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for message in messages:
        usage = message.get("token_usage") if isinstance(message, dict) else None
        if not isinstance(usage, dict):
            continue
        for key in totals:
            totals[key] += _optional_int(usage.get(key)) or 0
    if not totals["total_tokens"]:
        totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return totals


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
