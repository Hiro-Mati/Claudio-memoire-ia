# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared async count/time window batcher for streaming session updates."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from openviking.telemetry import tracer
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass(slots=True)
class StreamingBatcherConfig:
    """Count/time window configuration shared by streaming updaters."""

    max_items_per_batch: int = 8
    max_wait_seconds: float = 10.0
    timer_check_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.max_items_per_batch <= 0:
            raise ValueError("max_items_per_batch must be > 0")
        if self.max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be > 0")
        if self.timer_check_interval_seconds <= 0:
            raise ValueError("timer_check_interval_seconds must be > 0")


@dataclass(slots=True)
class StreamingBatchItemOutcome(Generic[R]):
    """Result or exception for one item in an item-aware batch process call."""

    result: R | None = None
    exception: Exception | None = None

    @classmethod
    def success(cls, result: R) -> StreamingBatchItemOutcome[R]:
        return cls(result=result)

    @classmethod
    def failure(cls, exception: Exception) -> StreamingBatchItemOutcome[R]:
        return cls(exception=exception)


@dataclass(slots=True)
class StreamingBatchResults(Generic[R]):
    """Per-item outcomes plus an optional aggregate result for ``flush`` callers."""

    item_outcomes: list[StreamingBatchItemOutcome[R]]
    aggregate_result: R | None = None


@dataclass(slots=True)
class StreamingBatcher(Generic[T, R]):
    """A reusable async batcher whose submit waits for its batch result.

    Items are buffered until either the buffered size reaches
    ``max_items_per_batch`` or the oldest item waits for ``max_wait_seconds``.
    Flush is performed by background tasks/timer; each ``submit`` awaits the
    Future attached to its own batch item, so callers only return after the
    batch containing their item has been processed.
    """

    name: str
    process_batch: Callable[[list[T], str], Awaitable[R]] | None = None
    config: StreamingBatcherConfig = field(default_factory=StreamingBatcherConfig)
    item_size: Callable[[T], int] | None = None
    result_metadata: Callable[[R], dict[str, Any] | None] | None = None
    process_batch_items: Callable[[list[T], str], Awaitable[StreamingBatchResults[R]]] | None = None
    _buffer: list[_PendingBatchItem[T, R]] = field(init=False, repr=False)
    _buffer_lock: asyncio.Lock = field(init=False, repr=False)
    _flush_lock: asyncio.Lock = field(init=False, repr=False)
    _timer_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _last_result: R | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if (self.process_batch is None) == (self.process_batch_items is None):
            raise ValueError("exactly one of process_batch or process_batch_items is required")
        self._buffer = []
        self._buffer_lock = asyncio.Lock()
        self._flush_lock = asyncio.Lock()
        self._timer_task = None
        self._closed = False
        self._last_result = None

    @property
    def closed(self) -> bool:
        return self._closed

    async def get_buffered_size(self) -> int:
        async with self._buffer_lock:
            return sum(self._item_size(item.payload) for item in self._buffer)

    async def get_buffered_item_count(self) -> int:
        async with self._buffer_lock:
            return len(self._buffer)

    async def submit(self, payload: T) -> R:
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")

        payload_size = self._item_size(payload)
        if payload_size > self.config.max_items_per_batch:
            raise ValueError(
                f"{self.name} item size {payload_size} exceeds hard batch limit "
                f"{self.config.max_items_per_batch}"
            )

        self._ensure_timer_task()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[R] = loop.create_future()
        should_flush = False
        async with self._buffer_lock:
            self._buffer.append(
                _PendingBatchItem(
                    payload=payload,
                    submitted_at=time.monotonic(),
                    future=future,
                )
            )
            should_flush = self._buffered_size_unlocked() >= self.config.max_items_per_batch

        if should_flush:
            self._trigger_background_flush("count")

        return await future

    async def close(self) -> R | None:
        if self._closed:
            return None
        self._closed = True
        await self._stop_timer_task()
        return await self.flush("close")

    async def flush(self, reason: str) -> R | None:
        async with self._flush_lock:
            async with self._buffer_lock:
                if not self._buffer:
                    return None
                items = self._buffer
                self._buffer = []

            batches = self._partition_items(items)
            last_result: R | None = None
            for batch_index, batch_items in enumerate(batches):
                batch_id = uuid.uuid4().hex
                batch_trace_id = uuid.uuid4().hex
                try:
                    with tracer.start_as_current_span(
                        name=f"{self.name}.flush",
                        trace_id=batch_trace_id,
                    ):
                        tracer.set("batch_id", batch_id)
                        tracer.set("flush_reason", reason)
                        tracer.set("batch_index", batch_index)
                        tracer.set("request_count", len(batch_items))
                        tracer.set(
                            "input_size",
                            sum(self._item_size(item.payload) for item in batch_items),
                        )
                        if self.process_batch_items is not None:
                            batch_results = await self.process_batch_items(
                                [item.payload for item in batch_items],
                                reason,
                            )
                            if len(batch_results.item_outcomes) != len(batch_items):
                                raise ValueError(
                                    f"{self.name} returned "
                                    f"{len(batch_results.item_outcomes)} item outcomes for "
                                    f"{len(batch_items)} items"
                                )
                            self._annotate_item_results(
                                batch_results,
                                batch_id=batch_id,
                                batch_trace_id=batch_trace_id,
                            )
                        else:
                            assert self.process_batch is not None
                            result = await self.process_batch(
                                [item.payload for item in batch_items],
                                reason,
                            )
                            self._annotate_result(
                                result,
                                batch_id=batch_id,
                                batch_trace_id=batch_trace_id,
                            )
                except Exception as exc:
                    for pending_batch in batches[batch_index:]:
                        for item in pending_batch:
                            if not item.future.done():
                                item.future.set_exception(exc)
                    raise

                if self.process_batch_items is not None:
                    successful_results: list[R] = []
                    failure_count = 0
                    for item, outcome in zip(
                        batch_items,
                        batch_results.item_outcomes,
                        strict=True,
                    ):
                        if item.future.done():
                            continue
                        if outcome.exception is not None:
                            failure_count += 1
                            item.future.set_exception(outcome.exception)
                        else:
                            item_result = cast(R, outcome.result)
                            item.future.set_result(item_result)
                            successful_results.append(item_result)

                    aggregate_result = batch_results.aggregate_result
                    if aggregate_result is None and successful_results:
                        aggregate_result = successful_results[-1]
                    if aggregate_result is not None:
                        self._last_result = aggregate_result
                        last_result = aggregate_result
                    if failure_count:
                        logger.warning(
                            "[%s] batch process call completed with %s isolated item failure(s)",
                            self.name,
                            failure_count,
                        )
                    continue

                self._last_result = result
                last_result = result
                for item in batch_items:
                    if not item.future.done():
                        item.future.set_result(result)
            return last_result

    def _ensure_timer_task(self) -> None:
        if self._timer_task is not None and not self._timer_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[%s] timer loop not started: no running event loop", self.name)
            self._timer_task = None
            return
        self._timer_task = loop.create_task(
            self._run_timer_loop(),
            name=f"{self.name}-flush-loop",
        )

    async def _stop_timer_task(self) -> None:
        task = self._timer_task
        if task is None:
            return
        self._timer_task = None
        if task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run_timer_loop(self) -> None:
        while True:
            flush_task: asyncio.Task[R | None] | None = None
            try:
                await asyncio.sleep(self.config.timer_check_interval_seconds)
                if await self._should_flush_by_time_or_count():
                    flush_task = asyncio.create_task(
                        self.flush("time"),
                        name=f"{self.name}-flush-time",
                    )
                    await asyncio.shield(flush_task)
            except asyncio.CancelledError:
                if flush_task is not None:
                    try:
                        await flush_task
                    except Exception as exc:
                        logger.warning(
                            "[%s] timer flush failed while stopping: %s",
                            self.name,
                            exc,
                        )
                raise
            except Exception as exc:
                logger.warning("[%s] timer flush iteration failed: %s", self.name, exc)

    async def _should_flush_by_time_or_count(self) -> bool:
        async with self._buffer_lock:
            if not self._buffer:
                return False
            if self._buffered_size_unlocked() >= self.config.max_items_per_batch:
                return True
            oldest = min(item.submitted_at for item in self._buffer)
            return (time.monotonic() - oldest) >= self.config.max_wait_seconds

    def _trigger_background_flush(self, reason: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _runner() -> None:
            try:
                await self.flush(reason)
            except Exception as exc:
                logger.warning("[%s] background flush failed: %s", self.name, exc)

        loop.create_task(_runner(), name=f"{self.name}-flush-{reason}")

    def _buffered_size_unlocked(self) -> int:
        return sum(self._item_size(item.payload) for item in self._buffer)

    def _partition_items(
        self,
        items: list[_PendingBatchItem[T, R]],
    ) -> list[list[_PendingBatchItem[T, R]]]:
        """Partition one flush snapshot without exceeding the configured hard limit."""

        batches: list[list[_PendingBatchItem[T, R]]] = []
        current: list[_PendingBatchItem[T, R]] = []
        current_size = 0
        limit = self.config.max_items_per_batch
        for item in items:
            item_size = self._item_size(item.payload)
            if item_size > limit:
                raise ValueError(
                    f"{self.name} item size {item_size} exceeds hard batch limit {limit}"
                )
            if current and current_size + item_size > limit:
                batches.append(current)
                current = []
                current_size = 0
            current.append(item)
            current_size += item_size
        if current:
            batches.append(current)
        return batches

    def _item_size(self, payload: T) -> int:
        if self.item_size is None:
            return 1
        return max(0, int(self.item_size(payload)))

    def _get_result_metadata(self, result: R) -> dict[str, Any] | None:
        if self.result_metadata is not None:
            return self.result_metadata(result)
        metadata = getattr(result, "metadata", None)
        return metadata if isinstance(metadata, dict) else None

    def _annotate_result(self, result: R, *, batch_id: str, batch_trace_id: str) -> None:
        metadata = self._get_result_metadata(result)
        if metadata is not None:
            metadata.setdefault("batch_id", batch_id)
            metadata.setdefault("batch_trace_id", batch_trace_id)

    def _annotate_item_results(
        self,
        batch_results: StreamingBatchResults[R],
        *,
        batch_id: str,
        batch_trace_id: str,
    ) -> None:
        seen: set[int] = set()
        results = [
            outcome.result
            for outcome in batch_results.item_outcomes
            if outcome.exception is None and outcome.result is not None
        ]
        if batch_results.aggregate_result is not None:
            results.append(batch_results.aggregate_result)
        for result in results:
            result_id = id(result)
            if result_id in seen:
                continue
            seen.add(result_id)
            self._annotate_result(
                result,
                batch_id=batch_id,
                batch_trace_id=batch_trace_id,
            )


@dataclass(slots=True)
class _PendingBatchItem(Generic[T, R]):
    payload: T
    submitted_at: float
    future: asyncio.Future[R]
