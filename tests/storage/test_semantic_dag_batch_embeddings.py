# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Batch embedding ownership tests for semantic DAG execution."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs.semantic_dag import SemanticDagExecutor
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.utils import embedding_utils
from openviking_cli.session.user_id import UserIdentifier

ROOT_URI = "viking://resources/docs"
FILE_URI = f"{ROOT_URI}/a.md"


def request_ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier("account", "user"), role=Role.USER)


def eligible_msg(**overrides) -> SemanticMsg:
    values = {
        "uri": ROOT_URI,
        "context_type": "resource",
        "recursive": False,
        "account_id": "account",
        "user_id": "user",
        "peer_id": "peer",
        "coalesce_key": "resource|account|user|peer|viking://resources/docs",
        "changes": {"modified": [FILE_URI]},
        "aggregate_directory": True,
    }
    values.update(overrides)
    return SemanticMsg(**values)


class FakeEmbeddingQueue:
    def __init__(self):
        self.messages = []

    async def enqueue(self, message):
        self.messages.append(message)
        return message.id


def fake_manager(queue):
    return SimpleNamespace(
        EMBEDDING="embedding",
        get_queue=lambda queue_name: queue,
    )


class FakeVikingFS:
    def __init__(self):
        self._async_agfs = self
        self.contents = {FILE_URI: "content"}

    async def ls(self, uri, node_limit=None, ctx=None):
        del node_limit, ctx
        if uri == ROOT_URI:
            return [{"name": "a.md", "isDir": False}]
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
        return uri.replace("viking://", "/local/account/")


class FakeSemanticProcessor:
    def __init__(self):
        self.summary_calls = []
        self.file_vector_calls = []
        self.dir_vector_calls = []

    async def _generate_single_file_summary(self, file_path, llm_sem=None, ctx=None):
        del llm_sem, ctx
        self.summary_calls.append(file_path)
        return {"name": "a.md", "summary": "summary"}

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
            file_path,
            summary_dict,
            ctx,
            use_summary,
            preserve_existing_created_at,
        )
        self.file_vector_calls.append(
            SimpleNamespace(
                telemetry_id=telemetry_id,
                tags=list(ingest_options.search_tags or []),
                track_wait=track_wait,
            )
        )

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
        del uri, context_type, abstract, overview, ctx
        self.dir_vector_calls.append(
            SimpleNamespace(
                ingest_options=ingest_options,
                telemetry_id=telemetry_id,
                track_wait=track_wait,
            )
        )


@pytest.fixture
def embedding_dependencies(monkeypatch):
    queue = FakeEmbeddingQueue()
    tracker = Mock()
    fs = FakeVikingFS()
    monkeypatch.setattr(embedding_utils, "get_request_wait_tracker", lambda: tracker)
    monkeypatch.setattr(embedding_utils, "get_queue_manager", lambda: fake_manager(queue))
    monkeypatch.setattr(embedding_utils, "get_viking_fs", lambda: fs)
    monkeypatch.setattr(
        embedding_utils,
        "get_openviking_config",
        lambda: SimpleNamespace(
            embedding=SimpleNamespace(text_source="summary_only", max_input_tokens=1000)
        ),
    )
    return queue, tracker


@pytest.mark.asyncio
async def test_vectorize_file_uses_explicit_telemetry_and_registers_that_waiter(
    embedding_dependencies,
):
    queue, tracker = embedding_dependencies

    await embedding_utils.vectorize_file(
        file_path=FILE_URI,
        summary_dict={"name": "a.md", "summary": "summary"},
        parent_uri=ROOT_URI,
        context_type="resource",
        ctx=request_ctx(),
        telemetry_id="tm-original",
        track_wait=True,
    )

    assert queue.messages[0].telemetry_id == "tm-original"
    tracker.register_embedding_root.assert_called_once_with("tm-original", queue.messages[0].id)


@pytest.mark.asyncio
async def test_shared_directory_embedding_has_no_request_wait_root(embedding_dependencies):
    queue, tracker = embedding_dependencies

    await embedding_utils.vectorize_directory_meta(
        uri=ROOT_URI,
        abstract="abstract",
        overview="overview",
        context_type="resource",
        ctx=request_ctx(),
        telemetry_id="",
        track_wait=False,
    )

    assert len(queue.messages) == 2
    assert all(message.telemetry_id == "" for message in queue.messages)
    tracker.register_embedding_root.assert_not_called()


@pytest.mark.asyncio
async def test_batch_dag_summarizes_file_once_and_fans_out_embeddings(monkeypatch):
    first = eligible_msg(telemetry_id="tm-1", ingest_options={"search_tags": ["one"]})
    second = eligible_msg(telemetry_id="tm-2", ingest_options={"search_tags": ["two"]})
    processor = FakeSemanticProcessor()
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: FakeVikingFS()
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_dag.get_openviking_config",
        lambda: SimpleNamespace(semantic=SimpleNamespace(overview_sample_limit=32)),
    )
    executor = SemanticDagExecutor(
        processor=processor,
        context_type="resource",
        max_concurrent_llm=2,
        ctx=request_ctx(),
        incremental_update=True,
        target_uri=first.uri,
        recursive=False,
        changes={"modified": [FILE_URI]},
        file_contributions={FILE_URI: (first, second)},
        shared_directory_embedding=True,
    )

    await executor.run(first.uri)

    assert processor.summary_calls == [FILE_URI]
    assert [(call.telemetry_id, call.tags) for call in processor.file_vector_calls] == [
        ("tm-1", ["one"]),
        ("tm-2", ["two"]),
    ]
    assert all(call.track_wait is True for call in processor.file_vector_calls)
    assert len(processor.dir_vector_calls) == 1
    assert all(
        call.ingest_options is None and call.telemetry_id == "" and call.track_wait is False
        for call in processor.dir_vector_calls
    )
