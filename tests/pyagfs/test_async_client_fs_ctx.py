from __future__ import annotations

from typing import Any

import pytest

from openviking.pyagfs.async_client import AsyncAGFSClient
from openviking.pyagfs.helpers import cp


class _RecordingClient:
    """Record ctx values received by the sync AGFS surface."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str] | None]] = []

    def write(self, path: str, data: bytes, *, ctx: dict[str, str] | None = None) -> str:
        """Record write ctx and return a stable fake backend id."""
        self.calls.append(("write", path, ctx))
        return "written"

    def read(self, path: str, *, ctx: dict[str, str] | None = None, **_: Any) -> bytes:
        """Record read ctx and return a stable fake payload."""
        self.calls.append(("read", path, ctx))
        return b"payload"


@pytest.mark.asyncio
async def test_async_client_derives_account_ctx_from_local_agfs_path() -> None:
    client = _RecordingClient()
    agfs = AsyncAGFSClient(client)

    await agfs.write("/local/acct-1/data/file.txt", b"x")

    assert client.calls == [("write", "/local/acct-1/data/file.txt", {"account_id": "acct-1"})]


@pytest.mark.asyncio
async def test_async_client_uses_system_ctx_for_non_local_agfs_path() -> None:
    client = _RecordingClient()
    agfs = AsyncAGFSClient(client)

    await agfs.read("/queue/semantic/dequeue")

    assert client.calls == [("read", "/queue/semantic/dequeue", {"account_id": "_system"})]


@pytest.mark.asyncio
async def test_async_client_preserves_explicit_fs_ctx() -> None:
    client = _RecordingClient()
    agfs = AsyncAGFSClient(client)

    await agfs.write(
        "/local/path-account/data/file.txt", b"x", fs_ctx={"account_id": "ctx-account"}
    )

    assert client.calls == [
        ("write", "/local/path-account/data/file.txt", {"account_id": "ctx-account"})
    ]


@pytest.mark.asyncio
async def test_async_client_rejects_cross_account_mv() -> None:
    client = _RecordingClient()
    agfs = AsyncAGFSClient(client)

    with pytest.raises(ValueError, match="cross-account"):
        await agfs.mv("/local/a/data/file.txt", "/local/b/data/file.txt")


def test_cp_rejects_cross_account_raw_copy() -> None:
    class _CpClient:
        """Minimal cp client; methods must not be called after account guard fails."""

        def stat(self, path: str) -> dict[str, Any]:
            """Fail if cp checks storage before validating account boundaries."""
            raise AssertionError(f"unexpected stat call: {path}")

    with pytest.raises(ValueError, match="cross-account"):
        cp(_CpClient(), "/local/a/data/file.txt", "/local/b/data/file.txt")


def test_cp_rejects_cross_encryption_domain_raw_copy() -> None:
    class _CpClient:
        """Minimal cp client; methods must not be called after account guard fails."""

        def stat(self, path: str) -> dict[str, Any]:
            """Fail if cp checks storage before validating account boundaries."""
            raise AssertionError(f"unexpected stat call: {path}")

    with pytest.raises(ValueError, match="cross-account"):
        cp(_CpClient(), "/local/a/data/file.txt", "/mem/data/file.txt")


def test_sync_cp_preserves_none_return_contract() -> None:
    class _NativeCopyClient:
        def cp(
            self,
            source: str,
            target: str,
            recursive: bool = False,
            *,
            ctx: dict[str, str] | None = None,
        ) -> dict[str, int]:
            del source, target, recursive, ctx
            return {"entries_copied": 1}

    result = cp(
        _NativeCopyClient(),
        "/local/a/data/file.txt",
        "/local/a/data/file-copy.txt",
    )

    assert result is None


@pytest.mark.asyncio
async def test_async_cp_calls_native_binding_once_with_recursive_and_ctx() -> None:
    class _NativeCopyClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, bool, dict[str, str] | None]] = []

        def cp(
            self,
            source: str,
            target: str,
            recursive: bool = False,
            *,
            ctx: dict[str, str] | None = None,
        ) -> dict[str, int]:
            self.calls.append((source, target, recursive, ctx))
            return {"entries_copied": 4}

    client = _NativeCopyClient()
    agfs = AsyncAGFSClient(client)

    result = await agfs.cp(
        "/local/acct/data/source",
        "/local/acct/data/target",
        recursive=True,
        fs_ctx={"account_id": "acct", "lease_ref": "lease"},
    )

    assert result == {"entries_copied": 4}
    assert client.calls == [
        (
            "/local/acct/data/source",
            "/local/acct/data/target",
            True,
            {"account_id": "acct", "lease_ref": "lease"},
        )
    ]
