# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Tests for request-scoped wait behavior on write APIs."""

from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service.resource_service import ResourceService
from openviking.storage.content_write import ContentWriteCoordinator
from openviking.storage.queuefs.semantic_dag import SemanticDagExecutor
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.telemetry.context import bind_telemetry
from openviking.telemetry.operation import OperationTelemetry
from openviking.telemetry.request_wait_tracker import RequestWaitTracker
from openviking.utils import embedding_utils
from openviking_cli.session.user_id import UserIdentifier


class _FakeRequestWaitTracker:
    def __init__(self, queue_status):
        self.queue_status = queue_status
        self.registered_requests = []
        self.wait_calls = []
        self.build_calls = []
        self.cleaned = []

    def register_request(self, telemetry_id: str) -> None:
        self.registered_requests.append(telemetry_id)

    async def wait_for_request(self, telemetry_id: str, timeout, poll_interval=None):
        del poll_interval
        self.wait_calls.append((telemetry_id, timeout))

    def build_queue_status(self, telemetry_id: str):
        self.build_calls.append(telemetry_id)
        return self.queue_status

    def cleanup(self, telemetry_id: str) -> None:
        self.cleaned.append(telemetry_id)


class _ExplodingQueueManager:
    async def wait_complete(self, *args, **kwargs):
        raise AssertionError("global queue wait should not be used")


class _RecordingEmbeddingQueue:
    def __init__(self):
        self.messages = []

    async def enqueue(self, message):
        self.messages.append(message)
        return message.id


class _BatchWaitVikingFS:
    root_uri = "viking://resources/docs"

    def __init__(self):
        self.contents = {
            f"{self.root_uri}/0.md": "zero",
            f"{self.root_uri}/1.md": "one",
            f"{self.root_uri}/2.md": "two",
        }
        self._async_agfs = self

    async def ls(self, uri, node_limit=None, ctx=None):
        del node_limit, ctx
        if uri == self.root_uri:
            return [
                {"name": "0.md", "isDir": False},
                {"name": "1.md", "isDir": False},
                {"name": "2.md", "isDir": False},
            ]
        return []

    async def stat(self, uri, ctx=None):
        del ctx
        return {"size": len(self.contents.get(uri, ""))}

    async def read_file(self, uri, ctx=None):
        del ctx
        return self.contents.get(uri, "")

    async def write_file(self, uri, content, ctx=None, lease_ref=None):
        del ctx, lease_ref
        self.contents[uri] = content

    async def pathlock_acquire_exact_batch(self, paths):
        return {"paths": paths}

    async def pathlock_release(self, lease):
        del lease

    def _uri_to_path(self, uri, ctx=None):
        del ctx
        return uri


class _BatchWaitProcessor:
    def __init__(self, tracker):
        self.tracker = tracker
        self.file_embeddings = []
        self.directory_embeddings = []

    async def _generate_single_file_summary(self, file_path, llm_sem=None, ctx=None):
        del llm_sem, ctx
        return {"name": file_path.rsplit("/", 1)[-1], "summary": "summary"}

    async def _generate_overview(self, dir_uri, file_summaries, children_abstracts, **kwargs):
        del dir_uri, file_summaries, children_abstracts, kwargs
        return "overview"

    def _normalize_overview_generation(self, overview):
        return overview, "abstract"

    async def _vectorize_single_file(
        self,
        parent_uri,
        context_type,
        file_path,
        summary_dict,
        ctx=None,
        use_summary=False,
        preserve_existing_created_at=False,
        ingest_options=None,
        *,
        telemetry_id=None,
        track_wait=True,
    ):
        del (
            parent_uri,
            context_type,
            summary_dict,
            ctx,
            use_summary,
            preserve_existing_created_at,
            ingest_options,
        )
        embedding = SimpleNamespace(
            id=f"file-{telemetry_id}-{file_path.rsplit('/', 1)[-1]}",
            telemetry_id=telemetry_id,
            track_wait=track_wait,
        )
        self.file_embeddings.append(embedding)
        if track_wait:
            self.tracker.register_embedding_root(embedding.telemetry_id, embedding.id)

    async def _vectorize_directory(
        self,
        uri,
        context_type,
        abstract,
        overview,
        ctx=None,
        ingest_options=None,
        *,
        telemetry_id=None,
        track_wait=True,
    ):
        del uri, context_type, abstract, overview, ctx, ingest_options
        self.directory_embeddings.extend(
            [
                SimpleNamespace(telemetry_id=telemetry_id, track_wait=track_wait),
                SimpleNamespace(telemetry_id=telemetry_id, track_wait=track_wait),
            ]
        )


class _FakeVikingFS:
    def __init__(self, file_uri: str, root_uri: str):
        self._file_uri = file_uri
        self._root_uri = root_uri
        self.content = {file_uri: "original"}
        self._async_agfs = self

    def _ensure_mutable_access(self, uri: str, ctx):
        del uri, ctx

    async def pathlock_acquire_exact(self, lock_path):
        del lock_path
        return SimpleNamespace(id="lock-1")

    async def pathlock_release(self, lease):
        del lease

    async def stat(self, uri: str, ctx=None):
        del ctx
        if uri == self._file_uri:
            return {"isDir": False}
        if uri == self._root_uri:
            return {"isDir": True}
        raise AssertionError(f"unexpected stat uri: {uri}")

    def _uri_to_path(self, uri: str, ctx=None):
        del ctx
        return f"/fake/{uri.replace('://', '/').strip('/')}"

    async def delete_temp(self, temp_uri: str, ctx=None):
        del temp_uri, ctx
        return None

    async def read_file(self, uri: str, ctx=None):
        del ctx
        return self.content[uri]

    async def write_file(self, uri: str, content: str, ctx=None, lease_ref=None):
        del ctx, lease_ref
        self.content[uri] = content

    async def rm(self, uri: str, ctx=None, lock_handle=None, lease_ref=None):
        del ctx, lock_handle, lease_ref
        self.content.pop(uri, None)


@pytest.mark.asyncio
async def test_add_skill_wait_uses_request_tracker(monkeypatch):
    tracker = _FakeRequestWaitTracker(
        {
            "Semantic": {"processed": 0, "error_count": 0, "errors": []},
            "Embedding": {"processed": 1, "error_count": 0, "errors": []},
        }
    )
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
    telemetry = OperationTelemetry(operation="resources.add_skill", enabled=True)

    async def _fake_process_skill(**kwargs):
        del kwargs
        return {"status": "success", "uri": "viking://user/default/skills/demo", "name": "demo"}

    resource_service = ResourceService(
        viking_fs=object(),
        resource_processor=object(),
        skill_processor=SimpleNamespace(process_skill=_fake_process_skill),
    )
    monkeypatch.setattr(
        "openviking.service.resource_service.get_queue_manager",
        lambda: _ExplodingQueueManager(),
    )
    monkeypatch.setattr(
        "openviking.service.resource_service.get_request_wait_tracker",
        lambda: tracker,
        raising=False,
    )

    with bind_telemetry(telemetry):
        result = await resource_service.add_skill(
            data={"name": "demo", "content": "# Demo"},
            ctx=ctx,
            wait=True,
            timeout=9.0,
            target_uri="viking://user/default/skills",
        )

    assert result["queue_status"] == tracker.queue_status
    assert tracker.registered_requests == [telemetry.telemetry_id]
    assert tracker.wait_calls == [(telemetry.telemetry_id, 9.0)]
    assert tracker.build_calls == [telemetry.telemetry_id]
    assert tracker.cleaned == [telemetry.telemetry_id]


@pytest.mark.asyncio
async def test_add_skill_wait_uses_request_tracker_when_telemetry_disabled(monkeypatch):
    tracker = _FakeRequestWaitTracker(
        {
            "Semantic": {"processed": 0, "error_count": 0, "errors": []},
            "Embedding": {"processed": 1, "error_count": 0, "errors": []},
        }
    )
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
    telemetry = OperationTelemetry(operation="resources.add_skill", enabled=False)

    async def _fake_process_skill(**kwargs):
        del kwargs
        return {"status": "success", "uri": "viking://user/default/skills/demo", "name": "demo"}

    resource_service = ResourceService(
        viking_fs=object(),
        resource_processor=object(),
        skill_processor=SimpleNamespace(process_skill=_fake_process_skill),
    )
    monkeypatch.setattr(
        "openviking.service.resource_service.get_queue_manager",
        lambda: _ExplodingQueueManager(),
    )
    monkeypatch.setattr(
        "openviking.service.resource_service.get_request_wait_tracker",
        lambda: tracker,
        raising=False,
    )

    with bind_telemetry(telemetry):
        result = await resource_service.add_skill(
            data={"name": "demo", "content": "# Demo"},
            ctx=ctx,
            wait=True,
            timeout=9.0,
            target_uri="viking://user/default/skills",
        )

    assert result["root_uri"] == "viking://user/default/skills/demo"
    assert result["queue_status"] == tracker.queue_status
    assert tracker.registered_requests == [telemetry.telemetry_id]
    assert tracker.wait_calls == [(telemetry.telemetry_id, 9.0)]
    assert tracker.build_calls == [telemetry.telemetry_id]
    assert tracker.cleaned == [telemetry.telemetry_id]


@pytest.mark.asyncio
async def test_content_write_wait_uses_request_tracker(monkeypatch):
    file_uri = "viking://resources/demo/doc.md"
    root_uri = "viking://resources/demo"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    telemetry = OperationTelemetry(operation="content.write", enabled=True)
    tracker = _FakeRequestWaitTracker(
        {
            "Semantic": {"processed": 1, "error_count": 0, "errors": []},
            "Embedding": {"processed": 0, "error_count": 0, "errors": []},
        }
    )
    coordinator = ContentWriteCoordinator(
        viking_fs=_FakeVikingFS(file_uri=file_uri, root_uri=root_uri)
    )

    monkeypatch.setattr(
        "openviking.storage.content_write.get_request_wait_tracker",
        lambda: tracker,
        raising=False,
    )

    async def _fake_enqueue_semantic_refresh(**kwargs):
        del kwargs
        return None

    async def _explode_wait_for_queues(*, timeout):
        del timeout
        raise AssertionError("global queue wait should not be used")

    monkeypatch.setattr(coordinator, "_enqueue_semantic_refresh", _fake_enqueue_semantic_refresh)
    monkeypatch.setattr(coordinator, "_wait_for_queues", _explode_wait_for_queues)

    with bind_telemetry(telemetry):
        result = await coordinator.write(
            uri=file_uri,
            content="updated",
            ctx=ctx,
            wait=True,
            timeout=5.0,
        )

    assert result["queue_status"] == tracker.queue_status
    assert tracker.registered_requests == [telemetry.telemetry_id]
    assert tracker.wait_calls == [(telemetry.telemetry_id, 5.0)]
    assert tracker.build_calls == [telemetry.telemetry_id]
    assert tracker.cleaned == [telemetry.telemetry_id]
    assert result["semantic_status"] == "complete"
    assert result["vector_status"] == "complete"


@pytest.mark.asyncio
async def test_content_write_wait_uses_request_tracker_when_telemetry_disabled(monkeypatch):
    file_uri = "viking://resources/demo/doc.md"
    root_uri = "viking://resources/demo"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    telemetry = OperationTelemetry(operation="content.write", enabled=False)
    tracker = _FakeRequestWaitTracker(
        {
            "Semantic": {"processed": 1, "error_count": 0, "errors": []},
            "Embedding": {"processed": 0, "error_count": 0, "errors": []},
        }
    )
    coordinator = ContentWriteCoordinator(
        viking_fs=_FakeVikingFS(file_uri=file_uri, root_uri=root_uri)
    )

    monkeypatch.setattr(
        "openviking.storage.content_write.get_request_wait_tracker",
        lambda: tracker,
        raising=False,
    )

    async def _fake_enqueue_semantic_refresh(**kwargs):
        del kwargs
        return None

    async def _explode_wait_for_queues(*, timeout):
        del timeout
        raise AssertionError("global queue wait should not be used")

    monkeypatch.setattr(coordinator, "_enqueue_semantic_refresh", _fake_enqueue_semantic_refresh)
    monkeypatch.setattr(coordinator, "_wait_for_queues", _explode_wait_for_queues)

    with bind_telemetry(telemetry):
        result = await coordinator.write(
            uri=file_uri,
            content="updated",
            ctx=ctx,
            wait=True,
            timeout=5.0,
        )

    assert result["queue_status"] == tracker.queue_status
    assert tracker.registered_requests == [telemetry.telemetry_id]
    assert tracker.wait_calls == [(telemetry.telemetry_id, 5.0)]
    assert tracker.build_calls == [telemetry.telemetry_id]
    assert tracker.cleaned == [telemetry.telemetry_id]
    assert result["semantic_status"] == "complete"
    assert result["vector_status"] == "complete"


@pytest.mark.asyncio
async def test_keyed_write_wait_excludes_detached_directory_embeddings(monkeypatch):
    tracker = RequestWaitTracker()
    telemetry_id = "tm-keyed-file-wait"
    semantic_id = "semantic-batch"
    file_embedding = SimpleNamespace(id="file-vector", telemetry_id=telemetry_id)
    directory_embedding = SimpleNamespace(id="directory-vector", telemetry_id="")
    queue = _RecordingEmbeddingQueue()
    tracker.cleanup(telemetry_id)
    tracker.register_request(telemetry_id)
    tracker.register_semantic_root(telemetry_id, semantic_id)
    monkeypatch.setattr(embedding_utils, "get_request_wait_tracker", lambda: tracker)

    try:
        await embedding_utils._enqueue_embedding_message(
            queue,
            directory_embedding,
            failure_message="directory embedding failed",
            track_wait=False,
        )
        await embedding_utils._enqueue_embedding_message(
            queue,
            file_embedding,
            failure_message="file embedding failed",
            track_wait=True,
        )

        tracker.mark_semantic_done(telemetry_id, semantic_id)
        assert tracker.is_complete(telemetry_id) is False

        tracker.mark_embedding_done(telemetry_id, file_embedding.id)
        await tracker.wait_for_request(telemetry_id, timeout=0.01, poll_interval=0.001)

        assert tracker.is_complete(telemetry_id) is True
        assert [message.id for message in queue.messages] == [
            directory_embedding.id,
            file_embedding.id,
        ]
    finally:
        tracker.cleanup(telemetry_id)


@pytest.mark.asyncio
async def test_three_coalesced_requests_wait_for_their_own_file_embeddings(monkeypatch):
    root_uri = _BatchWaitVikingFS.root_uri
    tracker = RequestWaitTracker()
    messages = [
        SemanticMsg(
            uri=root_uri,
            context_type="resource",
            recursive=False,
            account_id="account",
            user_id="user",
            peer_id="user",
            changes={"modified": [f"{root_uri}/{index}.md"]},
            aggregate_directory=True,
            telemetry_id=f"tm-{index}",
        )
        for index in range(3)
    ]
    for message in messages:
        tracker.cleanup(message.telemetry_id)
        tracker.register_request(message.telemetry_id)
        tracker.register_semantic_root(message.telemetry_id, message.id)

    processor = _BatchWaitProcessor(tracker)
    viking_fs = _BatchWaitVikingFS()
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: viking_fs)
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_dag.get_openviking_config",
        lambda: SimpleNamespace(semantic=SimpleNamespace(overview_sample_limit=32)),
    )
    executor = SemanticDagExecutor(
        processor=processor,
        context_type="resource",
        max_concurrent_llm=2,
        ctx=RequestContext(user=UserIdentifier("account", "user"), role=Role.USER),
        incremental_update=True,
        target_uri=root_uri,
        recursive=False,
        changes={"modified": [f"{root_uri}/{index}.md" for index in range(3)]},
        file_contributions={
            f"{root_uri}/{index}.md": (message,) for index, message in enumerate(messages)
        },
        shared_directory_embedding=True,
    )

    try:
        await executor.run(root_uri)

        assert len(processor.directory_embeddings) == 2
        assert all(embedding.telemetry_id == "" for embedding in processor.directory_embeddings)
        assert all(embedding.track_wait is False for embedding in processor.directory_embeddings)
        assert [embedding.telemetry_id for embedding in processor.file_embeddings] == [
            "tm-0",
            "tm-1",
            "tm-2",
        ]

        for message in messages:
            tracker.mark_semantic_done(message.telemetry_id, message.id)
        assert all(not tracker.is_complete(message.telemetry_id) for message in messages)

        for embedding in processor.file_embeddings:
            tracker.mark_embedding_done(
                embedding.telemetry_id,
                embedding.id,
                vector_written=True,
            )
        assert all(tracker.is_complete(message.telemetry_id) for message in messages)
    finally:
        for message in messages:
            tracker.cleanup(message.telemetry_id)


async def _return_true(handle, path):
    del handle, path
    return True


async def _return_none(handle):
    del handle
    return None
