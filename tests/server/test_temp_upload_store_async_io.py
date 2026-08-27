# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Event-loop safety tests for temporary HTTP uploads."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.server import temp_upload_store


class _UploadFile:
    filename = "upload.md"

    def __init__(self, chunks: list[bytes]):
        self._chunks = iter(chunks)

    async def read(self, _size: int) -> bytes:
        return next(self._chunks, b"")


@pytest.mark.asyncio
async def test_save_local_offloads_cleanup_writes_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[str] = []
    real_to_thread = temp_upload_store.asyncio.to_thread

    async def recording_to_thread(func, /, *args, **kwargs):
        calls.append(getattr(func, "__name__", type(func).__name__))
        return await real_to_thread(func, *args, **kwargs)

    config = SimpleNamespace(
        storage=SimpleNamespace(get_upload_temp_dir=lambda: tmp_path),
        temp_upload=SimpleNamespace(ttl_seconds=3600, shared_max_size_bytes=1024),
    )
    monkeypatch.setattr(temp_upload_store, "get_openviking_config", lambda: config)
    monkeypatch.setattr(temp_upload_store.asyncio, "to_thread", recording_to_thread)

    store = temp_upload_store.TempUploadStore(config)
    temp_file_id = await store.save_upload(_UploadFile([b"hello ", b"world"]), "local", object())

    assert (tmp_path / temp_file_id).read_bytes() == b"hello world"
    assert (tmp_path / f"{temp_file_id}.ov_upload.meta").is_file()
    assert "_cleanup_local_temp_files" in calls
    assert "_create_temp_file" in calls
    assert "_open_binary_for_write" in calls
    assert "write" in calls
    assert "_write_json" in calls
    assert "_close_file" in calls


@pytest.mark.asyncio
async def test_resolve_local_offloads_filesystem_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[str] = []
    real_to_thread = temp_upload_store.asyncio.to_thread

    async def recording_to_thread(func, /, *args, **kwargs):
        calls.append(getattr(func, "__name__", type(func).__name__))
        return await real_to_thread(func, *args, **kwargs)

    config = SimpleNamespace(storage=SimpleNamespace(get_upload_temp_dir=lambda: tmp_path))
    uploaded = tmp_path / "upload.md"
    uploaded.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(temp_upload_store, "get_openviking_config", lambda: config)
    monkeypatch.setattr(temp_upload_store.asyncio, "to_thread", recording_to_thread)

    store = temp_upload_store.TempUploadStore(SimpleNamespace(temp_upload=SimpleNamespace()))
    resolved = await store.resolve_for_consume("upload.md", object())

    assert resolved.local_path == str(uploaded)
    assert "_resolve_local" in calls
