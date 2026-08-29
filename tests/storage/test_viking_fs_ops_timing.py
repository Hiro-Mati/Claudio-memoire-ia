from unittest.mock import AsyncMock, Mock

import pytest

from openviking.storage.viking_fs import _ops


class _TimingProbe:
    @_ops._timed_operation
    async def succeed(self, uri: str) -> str:
        await _ops._timed_stage("remote", _return_value("ok"))
        return "ok"

    @_ops._timed_operation
    async def fail(self, uri: str) -> None:
        await _ops._timed_stage("remote", _raise_error())


async def _return_value(value):
    return value


async def _raise_error():
    raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_timed_operation_logs_success_and_stages(monkeypatch):
    info = Mock()
    monkeypatch.setattr(_ops.logger, "info", info)

    assert await _TimingProbe().succeed("viking://resources/a.txt") == "ok"

    info.assert_called_once()
    args = info.call_args.args
    assert args[0] == "[VikingFS][%s-timing] uri=%s status=%s %s total=%.1fms | %s"
    assert args[1:4] == ("succeed", "viking://resources/a.txt", "ok")
    assert "remote=" in args[6]


@pytest.mark.asyncio
async def test_timed_operation_logs_error_without_swallowing(monkeypatch):
    info = Mock()
    monkeypatch.setattr(_ops.logger, "info", info)

    with pytest.raises(RuntimeError, match="boom"):
        await _TimingProbe().fail("viking://resources/a.txt")

    assert info.call_args.args[1:4] == (
        "fail",
        "viking://resources/a.txt",
        "error",
    )
    assert "remote=" in info.call_args.args[6]


@pytest.mark.asyncio
async def test_remove_files_only_maps_uri_and_calls_agfs_rm():
    probe = object.__new__(_ops._OpsMixin)
    probe._uri_to_path = Mock(return_value="/local/account/upload/raw")
    probe._pathlock_fs_ctx = Mock(return_value={"lease": "outer"})
    probe._async_agfs = Mock()
    probe._async_agfs.rm = AsyncMock(return_value={"deleted": 1})
    probe._ensure_access = AsyncMock(side_effect=AssertionError("ACL must not run"))
    probe._delete_from_vector_store = AsyncMock(
        side_effect=AssertionError("vector delete must not run")
    )

    result = await probe.remove_files(
        "viking://upload/raw",
        recursive=True,
        lease_ref={"id": "outer"},
        auto_pathlock=False,
    )

    assert result == {"deleted": 1}
    probe._async_agfs.rm.assert_awaited_once_with(
        "/local/account/upload/raw",
        recursive=True,
        fs_ctx={"lease": "outer"},
        auto_pathlock=False,
    )
    probe._ensure_access.assert_not_awaited()
    probe._delete_from_vector_store.assert_not_awaited()
