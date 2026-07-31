"""Deterministic run-local log paths for remote rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def rollout_session_log_path(
    root: str | None,
    *,
    case_name: str,
    metadata: Mapping[str, Any],
) -> Path | None:
    root_text = str(root or "").strip()
    if not root_text:
        return None
    stage = str(metadata.get("rollout_stage") or metadata.get("stage") or "rollout")
    stage = _safe_fragment(stage.split(maxsplit=1)[0])
    try:
        epoch = int(metadata.get("epoch", 0) or 0)
    except (TypeError, ValueError):
        epoch = 0
    safe_case_name = _safe_fragment(case_name)
    return (
        Path(root_text).expanduser().resolve() / stage / f"epoch_{epoch}" / f"{safe_case_name}.log"
    )


def _safe_fragment(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in text)[:80] or "rollout"
