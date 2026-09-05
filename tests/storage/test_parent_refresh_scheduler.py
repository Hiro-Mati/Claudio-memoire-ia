# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Parent refresh modes: eager (upstream), debounced (coalesced), lazy (on read)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.core.context import ContextLevel
from openviking.server.identity import RequestContext, Role
from openviking.storage.abstract_overview import freshness_metadata, render_abstract_overview
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.queuefs.semantic_ops.freshness_policy import (
    FreshnessAction,
    FreshnessDecision,
)
from openviking.storage.queuefs.semantic_ops.parent_refresh_scheduler import (
    ParentRefreshScheduler,
    get_parent_refresh_scheduler,
    resolve_parent_refresh_mode,
)
from openviking.storage.queuefs.semantic_processor import SemanticProcessor
from openviking.storage.viking_fs._semantic import _SemanticMixin
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.parser_config import SemanticConfig


@pytest.fixture(autouse=True)
def _clear_scheduler():
    get_parent_refresh_scheduler().clear()
    yield
    get_parent_refresh_scheduler().clear()


def _semantic_config(mode: str, debounce: float = 0.05):
    return SimpleNamespace(
        semantic=SimpleNamespace(
            overview_sample_limit=32,
            freshness_refresh_ratio=0.10,
            parent_refresh_mode=mode,
            parent_refresh_debounce_s=debounce,
        )
    )


def _patch_processor(monkeypatch, mode: str, debounce: float = 0.05):
    plan = AsyncMock(
        return_value=FreshnessDecision(
            FreshnessAction.REFRESH_NOW, pending_after=1, total_entries=4
        )
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.plan_abstract_overview_refresh", plan
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_openviking_config",
        lambda: _semantic_config(mode, debounce),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs",
        lambda: SimpleNamespace(),
    )
    semantic_queue = SimpleNamespace(enqueue=AsyncMock())
    queue_manager = SimpleNamespace(SEMANTIC="semantic", get_queue=lambda *_a, **_k: semantic_queue)
    monkeypatch.setattr("openviking.storage.queuefs.get_queue_manager", lambda: queue_manager)
    return semantic_queue


def _child_msg(name: str) -> SemanticMsg:
    return SemanticMsg(
        uri=f"viking://resources/root/{name}",
        context_type="resource",
        role=str(Role.USER),
        generation_trigger="resource_ingest",
    )


# ---- scheduler unit -------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_coalesces_changes_for_one_key():
    scheduler = ParentRefreshScheduler()
    enqueue = AsyncMock()
    for child in ("a", "b", "a", "c"):
        scheduler.schedule("k", f"viking://r/{child}", enqueue, delay_s=0.05)
    assert scheduler.pending_keys() == ["k"]
    await asyncio.sleep(0.25)
    enqueue.assert_awaited_once_with(["viking://r/a", "viking://r/b", "viking://r/c"])
    assert scheduler.pending_keys() == []


@pytest.mark.asyncio
async def test_scheduler_flush_enqueues_immediately():
    scheduler = ParentRefreshScheduler()
    enqueue = AsyncMock()
    scheduler.schedule("k", "viking://r/a", enqueue, delay_s=60)
    fired = await scheduler.flush()
    assert fired == 1
    enqueue.assert_awaited_once_with(["viking://r/a"])


def test_resolve_mode_falls_back_to_eager():
    assert resolve_parent_refresh_mode(SimpleNamespace()) == "eager"
    assert resolve_parent_refresh_mode(SimpleNamespace(parent_refresh_mode="bogus")) == "eager"
    assert resolve_parent_refresh_mode(SimpleNamespace(parent_refresh_mode="LAZY")) == "lazy"


def test_semantic_config_validates_mode():
    assert SemanticConfig().parent_refresh_mode == "eager"
    assert SemanticConfig(parent_refresh_mode="debounced").parent_refresh_debounce_s == 30.0
    with pytest.raises(ValueError):
        SemanticConfig(parent_refresh_mode="bogus")
    with pytest.raises(ValueError):
        SemanticConfig(parent_refresh_debounce_s=-1)


# ---- processor integration -------------------------------------------------


@pytest.mark.asyncio
async def test_eager_mode_enqueues_parent_per_child(monkeypatch):
    semantic_queue = _patch_processor(monkeypatch, "eager")
    processor = SemanticProcessor()
    for name in ("a", "b"):
        msg = _child_msg(name)
        await processor._enqueue_parent_refresh(msg, msg.uri, l0_body_changed=True)
    assert semantic_queue.enqueue.await_count == 2


@pytest.mark.asyncio
async def test_debounced_mode_enqueues_parent_once(monkeypatch):
    semantic_queue = _patch_processor(monkeypatch, "debounced", debounce=0.05)
    processor = SemanticProcessor()
    for name in ("a", "b", "c"):
        msg = _child_msg(name)
        await processor._enqueue_parent_refresh(msg, msg.uri, l0_body_changed=True)
    semantic_queue.enqueue.assert_not_awaited()
    await asyncio.sleep(0.25)
    semantic_queue.enqueue.assert_awaited_once()
    parent_msg = semantic_queue.enqueue.await_args.args[0]
    assert parent_msg.uri == "viking://resources/root"
    assert parent_msg.generation_trigger == "parent_refresh"
    assert parent_msg.changes == {
        "modified": [
            "viking://resources/root/a",
            "viking://resources/root/b",
            "viking://resources/root/c",
        ]
    }


@pytest.mark.asyncio
async def test_lazy_mode_marks_pending_without_enqueue(monkeypatch):
    semantic_queue = _patch_processor(monkeypatch, "lazy")
    processor = SemanticProcessor()
    msg = _child_msg("a")
    await processor._enqueue_parent_refresh(msg, msg.uri, l0_body_changed=True)
    await asyncio.sleep(0.1)
    semantic_queue.enqueue.assert_not_awaited()
    assert get_parent_refresh_scheduler().pending_keys() == []


# ---- lazy read path --------------------------------------------------------


def test_lazy_refresh_context_type_mapping():
    f = _SemanticMixin._lazy_refresh_context_type
    assert f("viking://resources/proj") == "resource"
    assert f("viking://user/alice/resources/x") == "resource"
    assert f("viking://user/alice/skills/demo") == "skill"
    assert f("viking://agent/skills/demo") == "skill"
    assert f("viking://user/alice/memories/preferences") is None
    assert f("viking://user/alice/sessions/s1") is None
    assert f("viking://") is None


class _FakeFS(_SemanticMixin):
    def __init__(self, sidecar: bytes | None):
        self._sidecar = sidecar
        self._async_agfs = SimpleNamespace(read=self._read)

    async def _read(self, path: str):
        if self._sidecar is None:
            raise FileNotFoundError(path)
        return self._sidecar

    @staticmethod
    def _handle_agfs_read(value):
        return value

    @staticmethod
    def _decode_bytes(value: bytes) -> str:
        return value.decode("utf-8")


def _sidecar(pending: int) -> bytes:
    metadata = {
        "directory": "viking://resources/proj/",
        "freshness": freshness_metadata(total_entries=3, sampled_entries=3, pending=pending),
    }
    return render_abstract_overview(
        ContextLevel.ABSTRACT, "viking://resources/proj/", "A summary.", metadata
    ).encode("utf-8")


def _ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier("acme", "alice"), role=Role.USER)


def _patch_lazy_read(monkeypatch, mode: str):
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config", lambda: _semantic_config(mode)
    )
    monkeypatch.setattr(
        "openviking.storage.viking_fs._semantic.is_not_found_error",
        lambda exc: isinstance(exc, FileNotFoundError),
    )
    semantic_queue = SimpleNamespace(enqueue=AsyncMock())
    queue_manager = SimpleNamespace(SEMANTIC="semantic", get_queue=lambda *_a, **_k: semantic_queue)
    monkeypatch.setattr("openviking.storage.queuefs.get_queue_manager", lambda: queue_manager)
    return semantic_queue


@pytest.mark.asyncio
async def test_lazy_read_schedules_refresh_when_pending(monkeypatch):
    semantic_queue = _patch_lazy_read(monkeypatch, "lazy")
    fs = _FakeFS(_sidecar(pending=2))
    await fs._maybe_schedule_lazy_parent_refresh("viking://resources/proj/", "/x/proj", _ctx())
    await fs._maybe_schedule_lazy_parent_refresh("viking://resources/proj/", "/x/proj", _ctx())
    assert get_parent_refresh_scheduler().pending_keys() == [
        "resource|acme|alice|default|viking://resources/proj"
    ]
    await get_parent_refresh_scheduler().flush()
    semantic_queue.enqueue.assert_awaited_once()
    msg = semantic_queue.enqueue.await_args.args[0]
    assert msg.uri == "viking://resources/proj"
    assert msg.context_type == "resource"
    assert msg.account_id == "acme" and msg.user_id == "alice"
    assert msg.recursive is False
    assert msg.generation_trigger == "parent_refresh"


@pytest.mark.asyncio
async def test_lazy_read_ignores_fresh_directory(monkeypatch):
    _patch_lazy_read(monkeypatch, "lazy")
    fs = _FakeFS(_sidecar(pending=0))
    await fs._maybe_schedule_lazy_parent_refresh("viking://resources/proj/", "/x/proj", _ctx())
    assert get_parent_refresh_scheduler().pending_keys() == []


@pytest.mark.asyncio
async def test_lazy_read_refreshes_missing_sidecar(monkeypatch):
    _patch_lazy_read(monkeypatch, "lazy")
    fs = _FakeFS(None)
    await fs._maybe_schedule_lazy_parent_refresh("viking://resources/proj/", "/x/proj", _ctx())
    assert get_parent_refresh_scheduler().pending_keys() == [
        "resource|acme|alice|default|viking://resources/proj"
    ]


@pytest.mark.asyncio
async def test_lazy_read_is_noop_in_other_modes(monkeypatch):
    _patch_lazy_read(monkeypatch, "eager")
    fs = _FakeFS(_sidecar(pending=5))
    fs._async_agfs.read = MagicMock(side_effect=AssertionError("must not read sidecar"))
    await fs._maybe_schedule_lazy_parent_refresh("viking://resources/proj/", "/x/proj", _ctx())
    assert get_parent_refresh_scheduler().pending_keys() == []
