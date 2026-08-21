# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json
from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service.search_service import SearchService
from openviking_cli.retrieve import (
    ContextType,
    FindResult,
    MatchedContext,
    QueryResult,
    TypedQuery,
)
from openviking_cli.session.user_id import UserIdentifier


def _memory_file(status: str | None) -> str:
    fields = {} if status is None else {"case_status": status}
    return "# Memory\n\n<!-- MEMORY_FIELDS\n" + json.dumps(fields) + "\n-->"


def _context(uri: str, *, relations=None) -> MatchedContext:
    context = MatchedContext(
        uri=uri,
        context_type=ContextType.MEMORY,
        score=0.9,
    )
    if relations is not None:
        # Current retrieval results no longer expose relations. Keep a dynamic
        # attribute here to cover compatibility with older result objects.
        context.relations = list(relations)
    return context


class _FakeVikingFS:
    def __init__(self, result: FindResult, files: dict[str, str]):
        self.result = result
        self.files = files
        self.read_uris: list[str] = []

    async def find(self, **kwargs):
        return self.result

    async def search(self, **kwargs):
        return self.result

    async def read_file(self, uri: str, ctx=None):
        del ctx
        self.read_uris.append(uri)
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return self.files[uri]


@pytest.fixture
def request_ctx() -> RequestContext:
    return RequestContext(
        user=UserIdentifier(account_id="default", user_id="default"),
        role=Role.USER,
    )


@pytest.mark.asyncio
async def test_agent_recall_only_returns_promoted_cases_and_filters_provenance(request_ctx):
    promoted_uri = "viking://user/default/default/memories/cases/promoted.md"
    draft_uri = "viking://user/default/default/memories/cases/draft.md"
    experience_uri = "viking://user/default/default/memories/experiences/check.md"

    promoted = _context(promoted_uri)
    draft = _context(draft_uri)
    experience = _context(
        experience_uri,
        relations=[
            SimpleNamespace(uri=promoted_uri, abstract="promoted"),
            SimpleNamespace(uri=draft_uri, abstract="draft"),
        ],
    )
    query_result = QueryResult(
        query=TypedQuery(query="report", context_type=None, intent=""),
        matched_contexts=[promoted, draft, experience],
        searched_directories=["viking://user/default/default/memories"],
    )
    result = FindResult(
        memories=[promoted, draft, experience],
        resources=[],
        skills=[],
        query_results=[query_result],
    )
    viking_fs = _FakeVikingFS(
        result,
        {
            promoted_uri: _memory_file("promoted"),
            draft_uri: _memory_file("draft"),
        },
    )

    filtered = await SearchService(viking_fs).find(
        query="report",
        ctx=request_ctx,
        retrieval_purpose="agent_recall",
    )

    assert [item.uri for item in filtered.memories] == [promoted_uri, experience_uri]
    assert [item.uri for item in filtered.query_results[0].matched_contexts] == [
        promoted_uri,
        experience_uri,
    ]
    assert [item.uri for item in experience.relations] == [promoted_uri]
    assert filtered.total == 2
    assert sorted(viking_fs.read_uris) == sorted([promoted_uri, draft_uri])


@pytest.mark.asyncio
async def test_agent_recall_excludes_case_when_status_is_missing_or_unreadable(request_ctx):
    missing_status_uri = "viking://user/default/default/memories/cases/missing-status.md"
    unreadable_uri = "viking://user/default/default/memories/cases/unreadable.md"
    result = FindResult(
        memories=[_context(missing_status_uri), _context(unreadable_uri)],
        resources=[],
        skills=[],
    )
    viking_fs = _FakeVikingFS(result, {missing_status_uri: _memory_file(None)})

    filtered = await SearchService(viking_fs).search(
        query="report",
        ctx=request_ctx,
        retrieval_purpose="agent_recall",
    )

    assert filtered.memories == []
    assert filtered.total == 0


@pytest.mark.asyncio
async def test_agent_recall_excludes_case_directory_summaries(request_ctx):
    overview_uri = "viking://user/default/default/memories/cases/.overview.md"
    abstract_uri = "viking://user/default/default/memories/cases/reports/.abstract.md"
    experience_uri = "viking://user/default/default/memories/experiences/check.md"
    result = FindResult(
        memories=[_context(overview_uri), _context(abstract_uri), _context(experience_uri)],
        resources=[],
        skills=[],
    )
    viking_fs = _FakeVikingFS(result, {})

    filtered = await SearchService(viking_fs).find(
        query="report",
        ctx=request_ctx,
        retrieval_purpose="agent_recall",
    )

    assert [item.uri for item in filtered.memories] == [experience_uri]
    assert filtered.total == 1
    assert viking_fs.read_uris == []


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", ["general", "case_merge", "case_governance"])
async def test_non_agent_recall_purposes_keep_draft_cases(request_ctx, purpose):
    draft_uri = "viking://user/default/default/memories/cases/draft.md"
    result = FindResult(memories=[_context(draft_uri)], resources=[], skills=[])
    viking_fs = _FakeVikingFS(result, {draft_uri: _memory_file("draft")})

    unfiltered = await SearchService(viking_fs).find(
        query="report",
        ctx=request_ctx,
        retrieval_purpose=purpose,
    )

    assert [item.uri for item in unfiltered.memories] == [draft_uri]
    assert viking_fs.read_uris == []
