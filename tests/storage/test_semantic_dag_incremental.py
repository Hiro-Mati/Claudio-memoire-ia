# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
import re

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs.directory_semantic import DirectorySemanticTask
from openviking.storage.queuefs.semantic_dag import SemanticDagExecutor
from openviking.utils.ingest_options import IngestOptions
from openviking_cli.session.user_id import UserIdentifier


class _FakeVikingFS:
    def __init__(self, tree, file_contents):
        self._tree = tree
        self._file_contents = file_contents
        self.writes = []
        self._async_agfs = self

    async def ls(self, uri, node_limit=None, ctx=None):
        return self._tree.get(uri, [])

    async def read_file(self, path, ctx=None):
        return self._file_contents.get(path, "")

    async def write_file(self, path, content, ctx=None, lease_ref=None):
        self._file_contents[path] = content
        self.writes.append((path, content))

    async def pathlock_acquire_exact_batch(self, paths):
        return {"paths": paths}

    async def pathlock_release(self, lease):
        return None

    def _uri_to_path(self, uri, ctx=None):
        return uri.replace("viking://", "/local/acc1/")


class _FakeProcessor:
    def __init__(self):
        self.generated_overviews = []
        self.file_vector_options = {}
        self.directory_vector_options = []
        self.first_generation_started = asyncio.Event()
        self.first_generation_returned = asyncio.Event()
        self.release_first_generation = asyncio.Event()
        self.second_file_started = asyncio.Event()
        self.release_second_file = asyncio.Event()

    def parse_overview(self, overview_content):
        return dict(re.findall(r"^-\s*([^:]+):\s*(.*)$", overview_content, re.MULTILINE))

    async def generate_file_summary(self, file_path, llm_sem=None, ctx=None):
        del llm_sem, ctx
        name = file_path.rsplit("/", 1)[-1]
        if name == "b.txt":
            self.second_file_started.set()
            await self.release_second_file.wait()
        return {"name": name, "summary": f"summary-{name}"}

    async def generate_overview(self, dir_uri, file_summaries, children_abstracts):
        del dir_uri, children_abstracts
        overview = "FILES:\n" + "\n".join(
            f"- {item['name']}: {item['summary']}" for item in file_summaries
        )
        self.generated_overviews.append(overview)
        if len(self.generated_overviews) == 1:
            self.first_generation_started.set()
            await self.release_first_generation.wait()
            self.first_generation_returned.set()
        return overview

    def normalize_overview(self, overview):
        return overview, "abstract"

    async def vectorize_file(self, file_path, ingest_options=None, **kwargs):
        self.file_vector_options[file_path] = ingest_options

    async def vectorize_directory(self, uri, ingest_options=None, **kwargs):
        self.directory_vector_options.append(ingest_options)


@pytest.mark.asyncio
async def test_direct_incremental_updates_coalesce_only_directory_generation(monkeypatch):
    root_uri = "viking://resources/docs"
    tree = {
        root_uri: [
            {"name": "a.txt", "isDir": False},
            {"name": "b.txt", "isDir": False},
        ],
    }
    fake_fs = _FakeVikingFS(
        tree=tree,
        file_contents={
            f"{root_uri}/a.txt": "new a",
            f"{root_uri}/b.txt": "new b",
            f"{root_uri}/.overview.md": "FILES:\n- a.txt: old-a\n- b.txt: old-b",
            f"{root_uri}/.abstract.md": "old-abstract",
        },
    )
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: fake_fs)

    class _DirectoryTask(DirectorySemanticTask):
        def __init__(self, semantic_service):
            super().__init__(semantic_service)
            self.second_submitted = asyncio.Event()

        async def refresh(self, request):
            if request.event_id == "event-b.txt":
                self.second_submitted.set()
            return await super().refresh(request)

    processor = _FakeProcessor()
    directory_task = _DirectoryTask(processor)
    ctx_a = RequestContext(user=UserIdentifier("acc1", "user-a"), role=Role.USER)
    ctx_b = RequestContext(user=UserIdentifier("acc1", "user-b"), role=Role.USER)
    options_a = IngestOptions.from_search_tags(["team=a"])
    options_b = IngestOptions.from_search_tags(["team=b"])

    def make_executor(file_name, ingest_options, ctx):
        return SemanticDagExecutor(
            semantic_service=processor,
            context_type="resource",
            max_concurrent_llm=2,
            ctx=ctx,
            incremental_update=True,
            target_uri=root_uri,
            changes={"modified": [f"{root_uri}/{file_name}"]},
            ingest_options=ingest_options,
            event_id=f"event-{file_name}",
            directory_task=directory_task,
        )

    executor_a = make_executor("a.txt", options_a, ctx_a)
    executor_b = make_executor("b.txt", options_b, ctx_b)

    task_a = asyncio.create_task(executor_a.run(root_uri))
    await processor.first_generation_started.wait()
    task_b = asyncio.create_task(executor_b.run(root_uri))
    await processor.second_file_started.wait()
    processor.release_first_generation.set()
    await processor.first_generation_returned.wait()
    assert fake_fs.writes == []
    processor.release_second_file.set()
    await directory_task.second_submitted.wait()
    await asyncio.gather(task_a, task_b)

    assert processor.file_vector_options == {
        f"{root_uri}/a.txt": options_a,
        f"{root_uri}/b.txt": options_b,
    }
    assert len(processor.generated_overviews) == 2
    assert fake_fs.writes == [
        (
            f"{root_uri}/.overview.md",
            "FILES:\n- a.txt: summary-a.txt\n- b.txt: summary-b.txt",
        ),
        (f"{root_uri}/.abstract.md", "abstract"),
    ]
    assert processor.directory_vector_options == [options_b]
    assert not executor_a.root_directory_committed
    assert executor_b.root_directory_committed


if __name__ == "__main__":
    pytest.main([__file__])
