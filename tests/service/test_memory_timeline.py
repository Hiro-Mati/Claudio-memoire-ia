# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Memory provenance, revert and as-of reads over memory_diff.json and snapshots."""

import json
from datetime import datetime, timezone

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service.memory_timeline import MemoryTimelineService, expand_home, parse_instant
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier

USER = "viking://user/alice"
MEM = f"{USER}/memories/preferences/editor.md"
ARCHIVE_1 = f"{USER}/sessions/s1/history/archive_001"
ARCHIVE_2 = f"{USER}/sessions/s2/history/archive_001"


def _diff(archive, *, adds=(), updates=(), deletes=(), at="2026-09-01T10:00:00Z"):
    return {
        "archive_uri": archive,
        "trace_id": "t",
        "extracted_at": at,
        "operations": {"adds": list(adds), "updates": list(updates), "deletes": list(deletes)},
        "skipped_operations": [],
    }


class FakeFS:
    """Minimal FSService double: directory listing, raw reads, write/rm, snapshot log/show."""

    def __init__(self):
        self.files = {}
        self.dirs = {}
        self.writes = []
        self.removed = []
        self.commits = []
        self.blobs = {}
        self.raw_writes = {}

    # directory structure
    def add_dir(self, parent, uri, mod="2026-09-01T00:00:00Z"):
        self.dirs.setdefault(parent, []).append({"uri": uri, "isDir": True, "modTime": mod})

    async def ls(self, uri, ctx, **kwargs):
        if uri not in self.dirs:
            raise NotFoundError(uri, "directory")
        return list(self.dirs[uri])

    async def read(self, uri, ctx, offset=0, limit=-1):
        if uri not in self.files:
            raise NotFoundError(uri, "file")
        return self.files[uri]

    async def write(self, uri, content, ctx, mode="replace", **kwargs):
        self.writes.append((uri, content, mode))
        self.files[uri] = content
        return {"uri": uri}

    async def rm(self, uri, ctx, recursive=False, **kwargs):
        self.removed.append(uri)
        self.files.pop(uri, None)
        return None

    async def log(self, ctx, *, branch="main", limit=20, paths=None):
        return list(self.commits)

    async def show(self, target_ref, ctx, *, path=None):
        return self.blobs[(target_ref, path)]

    def _ensure_initialized(self):
        fs = self

        class _VFS:
            async def write_file(self, uri, content, ctx):
                fs.raw_writes[uri] = content
                fs.files[uri] = content

        return _VFS()


def _ctx():
    return RequestContext(user=UserIdentifier("acme", "alice"), role=Role.USER)


@pytest.fixture
def fs():
    fake = FakeFS()
    fake.add_dir(f"{USER}/sessions", f"{USER}/sessions/s1", mod="2026-09-01T00:00:00Z")
    fake.add_dir(f"{USER}/sessions", f"{USER}/sessions/s2", mod="2026-09-02T00:00:00Z")
    fake.add_dir(f"{USER}/sessions/s1/history", ARCHIVE_1)
    fake.add_dir(f"{USER}/sessions/s2/history", ARCHIVE_2)
    fake.files[f"{ARCHIVE_1}/memory_diff.json"] = json.dumps(
        _diff(ARCHIVE_1, adds=[{"uri": MEM, "memory_type": "preferences", "after": "v1"}])
    )
    fake.files[f"{ARCHIVE_2}/memory_diff.json"] = json.dumps(
        _diff(
            ARCHIVE_2,
            updates=[{"uri": MEM, "memory_type": "preferences", "before": "v1", "after": "v2"}],
            at="2026-09-02T10:00:00Z",
        )
    )
    fake.files[MEM] = "v2"
    return fake


def test_expand_home_and_parse_instant():
    ctx = _ctx()
    assert expand_home("viking://~/memories/x.md", ctx) == f"{USER}/memories/x.md"
    assert expand_home("viking://resources/a/", ctx) == "viking://resources/a"
    assert parse_instant("2026-09-01T10:00:00Z").tzinfo is not None
    assert parse_instant("2026-09-01T10:00:00").isoformat() == "2026-09-01T10:00:00+00:00"
    with pytest.raises(InvalidArgumentError):
        parse_instant("not a date")


@pytest.mark.asyncio
async def test_provenance_lists_changes_newest_first(fs):
    events = await MemoryTimelineService(fs).provenance(
        "viking://~/memories/preferences/editor.md", _ctx()
    )
    assert [e["op"] for e in events] == ["update", "add"]
    assert events[0]["archive_uri"] == ARCHIVE_2
    assert events[0]["session_id"] == "s2"
    assert events[0]["before"] == "v1" and events[0]["after"] == "v2"
    assert events[1]["after"] == "v1" and events[1]["before"] is None


@pytest.mark.asyncio
async def test_provenance_ignores_other_memories_and_missing_history(fs):
    fs.add_dir(f"{USER}/sessions", f"{USER}/sessions/s3")  # no history dir
    events = await MemoryTimelineService(fs).provenance(f"{USER}/memories/other.md", _ctx())
    assert events == []


@pytest.mark.asyncio
async def test_revert_update_restores_previous_body(fs):
    result = await MemoryTimelineService(fs).revert(MEM, ARCHIVE_2, _ctx())
    assert result["action"] == "restored_previous"
    assert fs.writes == [(MEM, "v1", "replace")]
    audit = json.loads(fs.raw_writes[f"{ARCHIVE_2}/memory_reverts.json"])
    assert audit["reverts"][0]["reverted_op"] == "update"


@pytest.mark.asyncio
async def test_revert_add_deletes_and_revert_delete_recreates(fs):
    service = MemoryTimelineService(fs)
    result = await service.revert(MEM, ARCHIVE_1, _ctx())
    assert result["action"] == "deleted" and fs.removed == [MEM]

    archive_3 = f"{USER}/sessions/s2/history/archive_002"
    fs.add_dir(f"{USER}/sessions/s2/history", archive_3)
    fs.files[f"{archive_3}/memory_diff.json"] = json.dumps(
        _diff(
            archive_3,
            deletes=[{"uri": MEM, "memory_type": "preferences", "deleted_content": "gone"}],
        )
    )
    result = await service.revert(MEM, archive_3, _ctx())
    assert result["action"] == "recreated"
    assert fs.writes[-1] == (MEM, "gone", "upsert")


@pytest.mark.asyncio
async def test_revert_rejects_unknown_archive_or_uri(fs):
    service = MemoryTimelineService(fs)
    with pytest.raises(NotFoundError):
        await service.revert(MEM, f"{USER}/sessions/s9/history/archive_001", _ctx())
    with pytest.raises(NotFoundError):
        await service.revert(f"{USER}/memories/none.md", ARCHIVE_1, _ctx())


@pytest.mark.asyncio
async def test_as_of_picks_latest_snapshot_before_instant(fs):
    t1 = int(datetime(2026, 9, 1, 9, tzinfo=timezone.utc).timestamp())
    t2 = int(datetime(2026, 9, 2, 9, tzinfo=timezone.utc).timestamp())
    fs.commits = [
        {"oid": "c2", "committer": {"time_seconds": t2}, "message": "second"},
        {"oid": "c1", "committer": {"time_seconds": t1}, "message": "first"},
    ]
    fs.blobs[("c1", MEM)] = b"v1"
    fs.blobs[("c2", MEM)] = {"oid": "b", "size": 2, "bytes": b"v2"}
    service = MemoryTimelineService(fs)

    result = await service.as_of(MEM, "2026-09-01T12:00:00Z", _ctx())
    assert result["commit"] == "c1" and result["content"] == "v1"
    result = await service.as_of(MEM, "2026-09-03T00:00:00Z", _ctx())
    assert result["commit"] == "c2" and result["content"] == "v2"
    with pytest.raises(NotFoundError):
        await service.as_of(MEM, "2026-08-01T00:00:00Z", _ctx())
