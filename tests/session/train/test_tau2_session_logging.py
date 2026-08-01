from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from loguru import logger as loguru_logger

from benchmark.tau2.train.session_logging import (
    current_session_log_path,
    install_session_log_routing,
    rollout_log_session,
)
from openviking.session.train.components.rollout_log_path import rollout_session_log_path


@pytest.mark.asyncio
async def test_rollout_log_session_isolates_python_and_loguru_records(tmp_path: Path) -> None:
    python_logger = logging.getLogger("test.tau2.session_logging.isolation")
    python_logger.handlers.clear()
    python_logger.propagate = False
    python_logger.setLevel(logging.DEBUG)
    install_session_log_routing(python_logger)

    first_path = tmp_path / "first.log"
    second_path = tmp_path / "second.log"
    both_started = asyncio.Event()
    started = 0
    started_lock = asyncio.Lock()

    async def emit(path: Path, marker: str) -> None:
        nonlocal started
        with rollout_log_session(path):
            assert current_session_log_path() == str(path.resolve())
            python_logger.warning("python-%s", marker)
            async with started_lock:
                started += 1
                if started == 2:
                    both_started.set()
            await both_started.wait()
            loguru_logger.warning("loguru-{}", marker)

    await asyncio.gather(emit(first_path, "first"), emit(second_path, "second"))

    first = first_path.read_text(encoding="utf-8")
    second = second_path.read_text(encoding="utf-8")
    assert "python-first" in first
    assert "loguru-first" in first
    assert "python-second" not in first
    assert "loguru-second" not in first
    assert "python-second" in second
    assert "loguru-second" in second
    assert "python-first" not in second
    assert "loguru-first" not in second
    assert current_session_log_path() is None


def test_rollout_log_session_without_path_preserves_disabled_behavior(tmp_path: Path) -> None:
    with rollout_log_session(None) as active_path:
        assert active_path is None
        assert current_session_log_path() is None

    assert list(tmp_path.iterdir()) == []


def test_rollout_session_log_path_uses_stage_epoch_and_safe_case_name(tmp_path: Path) -> None:
    path = rollout_session_log_path(
        str(tmp_path),
        case_name="tau2 case/16:t3",
        metadata={"epoch": "4", "rollout_stage": "final_test_rollout"},
    )

    assert path == tmp_path / "final_test_rollout" / "epoch_4" / "tau2_case_16_t3.log"
    assert rollout_session_log_path(None, case_name="case", metadata={}) is None


def test_rollout_log_session_close_failure_does_not_fail_rollout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FailingStream:
        def write(self, message: str) -> int:
            return len(message)

        def flush(self) -> None:
            raise OSError("flush failed")

        def close(self) -> None:
            return None

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: FailingStream())

    with rollout_log_session(tmp_path / "session.log") as active_path:
        assert active_path is not None


def test_rollout_log_session_flushes_on_close_instead_of_every_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class RecordingStream:
        def __init__(self) -> None:
            self.flush_count = 0

        def write(self, message: str) -> int:
            return len(message)

        def flush(self) -> None:
            self.flush_count += 1

        def close(self) -> None:
            return None

    stream = RecordingStream()
    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: stream)
    install_session_log_routing()

    with rollout_log_session(tmp_path / "session.log"):
        loguru_logger.info("buffer this record")
        assert stream.flush_count == 0

    assert stream.flush_count == 1
