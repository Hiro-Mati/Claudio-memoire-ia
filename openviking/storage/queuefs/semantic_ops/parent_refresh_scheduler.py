# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Coalescing scheduler for parent directory semantic refreshes.

Upstream behaviour ("eager" mode) enqueues one parent refresh every time a
child directory finishes its semantic task. In deep or busy trees this makes
each ancestor re-summarize once per child, which multiplies VLM calls on a
CPU-only machine.

This scheduler implements the two cheaper modes selected by
``semantic.parent_refresh_mode``:

* ``debounced``: refreshes for the same parent are merged and enqueued once,
  after a quiet window of ``semantic.parent_refresh_debounce_s`` seconds with
  no new child completion. The window restarts on every new change.
* ``lazy``: the parent is only marked pending (its sidecar counters are
  already persisted by :func:`plan_abstract_overview_refresh`). The refresh is
  enqueued the first time somebody reads that directory's abstract or overview.

The scheduler is process-local. A pending debounce lost to a crash is not a
correctness problem: the parent's ``pending_child_changes`` counter stays in
its sidecar metadata and the next child change or lazy read reschedules it.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

PARENT_REFRESH_MODES = ("eager", "debounced", "lazy")

EnqueueFn = Callable[[List[str]], Awaitable[None]]


@dataclass
class _PendingRefresh:
    """One coalesced parent refresh waiting for its quiet window."""

    enqueue: EnqueueFn
    modified: List[str] = field(default_factory=list)
    task: Optional[asyncio.Task] = None
    generation: int = 0


class ParentRefreshScheduler:
    """Merge parent refresh requests per coalesce key and enqueue them once."""

    def __init__(self) -> None:
        self._pending: Dict[str, _PendingRefresh] = {}
        self._lock = threading.Lock()

    # ---- inspection -----------------------------------------------------

    def pending_keys(self) -> List[str]:
        with self._lock:
            return list(self._pending)

    def pending_modified(self, key: str) -> List[str]:
        with self._lock:
            entry = self._pending.get(key)
            return list(entry.modified) if entry else []

    # ---- scheduling -----------------------------------------------------

    def schedule(
        self,
        key: str,
        modified_uri: str,
        enqueue: EnqueueFn,
        delay_s: float,
    ) -> None:
        """Record ``modified_uri`` for ``key`` and (re)start its quiet window.

        ``enqueue`` receives the merged list of modified child URIs when the
        window elapses. The most recent callable wins, so the message is built
        from the latest child's identity context.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            entry = self._pending.get(key)
            if entry is None:
                entry = _PendingRefresh(enqueue=enqueue)
                self._pending[key] = entry
            else:
                entry.enqueue = enqueue
                if entry.task is not None and not entry.task.done():
                    entry.task.cancel()
            if modified_uri not in entry.modified:
                entry.modified.append(modified_uri)
            entry.generation += 1
            generation = entry.generation
            entry.task = loop.create_task(self._fire_after(key, generation, delay_s))

    async def _fire_after(self, key: str, generation: int, delay_s: float) -> None:
        try:
            await asyncio.sleep(max(delay_s, 0.0))
        except asyncio.CancelledError:
            return
        await self._fire(key, generation)

    async def _fire(self, key: str, generation: Optional[int]) -> None:
        with self._lock:
            entry = self._pending.get(key)
            if entry is None:
                return
            if generation is not None and entry.generation != generation:
                # A newer change restarted the window; that task will fire.
                return
            del self._pending[key]
        try:
            await entry.enqueue(list(entry.modified))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Coalesced parent refresh failed for %s: %s", key, exc)

    async def flush(self, key: Optional[str] = None) -> int:
        """Enqueue pending refreshes immediately (tests, shutdown). Returns count."""
        with self._lock:
            keys = [key] if key is not None else list(self._pending)
            for k in keys:
                entry = self._pending.get(k)
                if entry is not None and entry.task is not None and not entry.task.done():
                    entry.task.cancel()
        fired = 0
        for k in keys:
            before = k in self._pending
            await self._fire(k, None)
            fired += int(before)
        return fired

    def clear(self) -> None:
        """Drop pending work without enqueueing (tests)."""
        with self._lock:
            for entry in self._pending.values():
                if entry.task is not None and not entry.task.done():
                    entry.task.cancel()
            self._pending.clear()


_SCHEDULER: Optional[ParentRefreshScheduler] = None
_SCHEDULER_LOCK = threading.Lock()


def get_parent_refresh_scheduler() -> ParentRefreshScheduler:
    global _SCHEDULER
    with _SCHEDULER_LOCK:
        if _SCHEDULER is None:
            _SCHEDULER = ParentRefreshScheduler()
        return _SCHEDULER


def resolve_parent_refresh_mode(semantic_config: object) -> str:
    """Return a valid mode from a config object, defaulting to ``eager``."""
    mode = str(getattr(semantic_config, "parent_refresh_mode", "eager") or "eager").lower()
    return mode if mode in PARENT_REFRESH_MODES else "eager"


def resolve_parent_refresh_delay(semantic_config: object) -> float:
    try:
        return float(getattr(semantic_config, "parent_refresh_debounce_s", 30.0))
    except (TypeError, ValueError):
        return 30.0
