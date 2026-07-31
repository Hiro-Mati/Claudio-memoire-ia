"""Session-aware log routing for concurrent Tau2 VikingBot rollouts."""

from __future__ import annotations

import logging
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator

from loguru import logger as loguru_logger

_current_log_path: ContextVar[str | None] = ContextVar(
    "tau2_rollout_session_log_path",
    default=None,
)


@dataclass(slots=True)
class _OpenSessionLog:
    stream: IO[str]
    references: int = 1


class _SessionLogRouter:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._open_logs: dict[str, _OpenSessionLog] = {}

    def acquire(self, path: Path) -> str:
        resolved = str(path.expanduser().resolve())
        with self._lock:
            opened = self._open_logs.get(resolved)
            if opened is not None:
                opened.references += 1
                return resolved
            target = Path(resolved)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._open_logs[resolved] = _OpenSessionLog(
                stream=target.open("a", encoding="utf-8"),
            )
        return resolved

    def release(self, path: str) -> None:
        with self._lock:
            opened = self._open_logs.get(path)
            if opened is None:
                return
            opened.references -= 1
            if opened.references > 0:
                return
            self._open_logs.pop(path, None)
            try:
                opened.stream.flush()
            except OSError as exc:
                _report_router_error(path, exc)
            try:
                opened.stream.close()
            except OSError as exc:
                _report_router_error(path, exc)

    def write(self, path: str, message: str) -> None:
        with self._lock:
            opened = self._open_logs.get(path)
            if opened is None:
                return
            try:
                opened.stream.write(message)
                if message and not message.endswith("\n"):
                    opened.stream.write("\n")
                opened.stream.flush()
            except OSError as exc:
                _report_router_error(path, exc)


class _SessionFileHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        path = _current_log_path.get()
        if path is None:
            return
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        _router.write(path, message)


_router = _SessionLogRouter()
_install_lock = threading.Lock()
_python_handler = _SessionFileHandler(level=logging.DEBUG)
_python_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_loguru_sink_id: int | None = None


def install_session_log_routing(*python_loggers: logging.Logger) -> None:
    """Install one shared session sink and attach it to the supplied Python loggers."""

    global _loguru_sink_id
    with _install_lock:
        for python_logger in python_loggers:
            if _python_handler not in python_logger.handlers:
                python_logger.addHandler(_python_handler)
        if _loguru_sink_id is None:
            _loguru_sink_id = loguru_logger.add(
                _write_loguru_message,
                level="DEBUG",
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                    "{name}:{function}:{line} - {message}"
                ),
                catch=True,
            )


@contextmanager
def rollout_log_session(path: Path | None) -> Iterator[str | None]:
    """Route logs emitted in this context to ``path`` without changing aggregate sinks."""

    if path is None:
        yield None
        return
    try:
        resolved = _router.acquire(path)
    except OSError as exc:
        _report_router_error(str(path), exc)
        yield None
        return

    token = _current_log_path.set(resolved)
    try:
        yield resolved
    finally:
        _current_log_path.reset(token)
        _router.release(resolved)


def current_session_log_path() -> str | None:
    """Return the active rollout log path for the current async/thread context."""

    return _current_log_path.get()


def _write_loguru_message(message) -> None:
    path = _current_log_path.get()
    if path is not None:
        _router.write(path, str(message))


def _report_router_error(path: str, exc: OSError) -> None:
    try:
        sys.stderr.write(
            f"[tau2-session-log] failed path={path} error_type={type(exc).__name__} error={exc}\n"
        )
        sys.stderr.flush()
    except Exception:
        pass
