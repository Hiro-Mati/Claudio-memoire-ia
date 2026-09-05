# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lexical (BM25) fusion inside HierarchicalRetriever."""

import pytest

from openviking.retrieve.hierarchical_retriever import HierarchicalRetriever, RetrieverMode
from openviking.retrieve.lexical_index import LexicalIndex, set_lexical_index
from openviking.server.identity import RequestContext, Role
from openviking_cli.retrieve.types import TypedQuery
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config import RetrievalConfig

ROOT = "viking://resources/root"


def _result(uri, score, level=2, abstract=None):
    return {
        "uri": uri,
        "abstract": abstract if abstract is not None else uri.rsplit("/", 1)[-1],
        "_score": score,
        "level": level,
        "context_type": "resource",
        "parent_uri": uri.rsplit("/", 1)[0],
    }


def _lexical_record(doc_id, uri, content, level=2):
    return {
        "id": doc_id,
        "account_id": "acme",
        "uri": uri,
        "parent_uri": uri.rsplit("/", 1)[0],
        "context_type": "resource",
        "level": level,
        "name": uri.rsplit("/", 1)[-1],
        "abstract": content,
        "content": content,
    }


class DummyEmbedResult:
    dense_vector = [1.0]
    sparse_vector = None


class DummyEmbedder:
    def prepare_embedding_input(self, text):
        return text

    async def embed_async(self, text, is_query=False):
        return DummyEmbedResult()

    def embed(self, text, is_query=False):
        return DummyEmbedResult()


class DummyStorage:
    """Vector store double: dense hits plus a record lookup used by lexical recovery."""

    def __init__(self, dense, records):
        self.collection_name = "context"
        self.acl_manager = None
        self._dense = dense
        self._records = {(r["uri"], r["level"]): r for r in records}
        self.filter_calls = []

    def _acl_enabled(self, ctx):
        return False

    async def collection_exists_bound(self):
        return True

    async def search_in_tenant(self, ctx, **kwargs):
        return [dict(r) for r in self._dense]

    async def search_children_in_tenant(self, ctx, parent_uri, **kwargs):
        return [dict(r) for r in self._dense if r["parent_uri"] == parent_uri]

    async def filter(
        self,
        filter,
        limit=10,
        offset=0,
        output_fields=None,
        order_by=None,
        order_desc=False,
        *,
        ctx,
    ):
        # Recover uri/level from the And([Eq(account), PathScope(uri), Eq(level)]) expression.
        uri = filter.conds[1].path
        level = filter.conds[2].value
        self.filter_calls.append((uri, level))
        record = self._records.get((uri, level))
        return [dict(record)] if record else []

    async def scroll(self, filter=None, limit=100, cursor=None, output_fields=None, *, ctx):
        return list(self._records.values()), None


@pytest.fixture
def lexical_index():
    index = LexicalIndex(":memory:")
    set_lexical_index(index)
    HierarchicalRetriever._lexical_rebuild_scheduled.clear()
    yield index
    set_lexical_index(None)
    HierarchicalRetriever._lexical_rebuild_scheduled.clear()
    index.close()


def _ctx():
    return RequestContext(user=UserIdentifier("acme", "alice"), role=Role.USER)


def _retriever(storage, boost, monkeypatch):
    monkeypatch.setattr(
        "openviking.retrieve.hierarchical_retriever.lexical_index_enabled", lambda: True
    )
    return HierarchicalRetriever(
        storage=storage,
        embedder=DummyEmbedder(),
        rerank_config=None,
        retrieval_config=RetrievalConfig(
            lexical_index_enabled=True, lexical_boost=boost, lexical_limit=10
        ),
    )


@pytest.mark.asyncio
async def test_quick_find_boosts_and_recovers_exact_tokens(lexical_index, monkeypatch):
    dense = [
        _result(f"{ROOT}/generic.md", 0.80, abstract="general prose"),
        _result(f"{ROOT}/handler.py", 0.30, abstract="def parse_abstract_overview"),
    ]
    records = [
        _lexical_record("g", f"{ROOT}/generic.md", "general prose"),
        _lexical_record("h", f"{ROOT}/handler.py", "def parse_abstract_overview(raw)"),
        _lexical_record("o", f"{ROOT}/other.py", "parse_abstract_overview is called here too"),
    ]
    lexical_index.upsert_many(records)
    lexical_index.upsert(_lexical_record("z", f"{ROOT}/deleted.py", "parse_abstract_overview gone"))
    storage = DummyStorage(dense, records)
    retriever = _retriever(storage, 0.5, monkeypatch)

    result = await retriever.retrieve(
        TypedQuery(
            query="parse_abstract_overview", context_type=None, intent="", target_directories=[ROOT]
        ),
        ctx=_ctx(),
        limit=5,
        mode=RetrieverMode.QUICK,
    )
    by_uri = {m.uri: m.score for m in result.matched_contexts}

    # exact identifier: handler.py (dense 0.30) is boosted by up to 0.5 * bm25
    assert 0.65 <= by_uri[f"{ROOT}/handler.py"] <= 0.80
    assert by_uri[f"{ROOT}/generic.md"] == pytest.approx(0.80)
    # other.py was absent from dense results and is recovered through the vector lookup
    assert f"{ROOT}/other.py" in by_uri
    assert 0 < by_uri[f"{ROOT}/other.py"] <= 0.5
    # a stale lexical entry (no vector record) is purged, never surfaced
    assert f"{ROOT}/deleted.py" not in by_uri
    assert lexical_index.count() == 3
    assert (f"{ROOT}/deleted.py", 2) in storage.filter_calls


@pytest.mark.asyncio
async def test_children_search_is_fused_during_recursion(lexical_index, monkeypatch):
    dense = [
        _result(f"{ROOT}/sub", 0.6, level=1, abstract="sub dir"),
        _result(f"{ROOT}/sub/a.py", 0.2, abstract="alpha"),
    ]
    records = [
        _lexical_record("a", f"{ROOT}/sub/a.py", "alpha"),
        _lexical_record("b", f"{ROOT}/sub/b.py", "needle_token here"),
    ]
    lexical_index.upsert_many(records)
    storage = DummyStorage(dense, records)
    retriever = _retriever(storage, 0.4, monkeypatch)

    result = await retriever.retrieve(
        TypedQuery(query="needle_token", context_type=None, intent="", target_directories=[ROOT]),
        ctx=_ctx(),
        limit=5,
        mode=RetrieverMode.THINKING,
    )
    uris = [m.uri for m in result.matched_contexts]
    assert f"{ROOT}/sub/b.py" in uris


@pytest.mark.asyncio
async def test_zero_boost_leaves_dense_results_untouched(lexical_index, monkeypatch):
    dense = [_result(f"{ROOT}/generic.md", 0.7)]
    lexical_index.upsert(_lexical_record("x", f"{ROOT}/x.py", "needle_token"))
    storage = DummyStorage(dense, [])
    retriever = _retriever(storage, 0.0, monkeypatch)
    result = await retriever.retrieve(
        TypedQuery(query="needle_token", context_type=None, intent="", target_directories=[ROOT]),
        ctx=_ctx(),
        limit=5,
        mode=RetrieverMode.QUICK,
    )
    assert [m.uri for m in result.matched_contexts] == [f"{ROOT}/generic.md"]
    assert storage.filter_calls == []


@pytest.mark.asyncio
async def test_empty_index_is_rebuilt_from_vector_store(lexical_index, monkeypatch):
    records = [_lexical_record("k", f"{ROOT}/k.py", "rebuild_me")]
    storage = DummyStorage([], records)
    retriever = _retriever(storage, 0.5, monkeypatch)
    assert lexical_index.count("acme") == 0
    await retriever.retrieve(
        TypedQuery(query="rebuild_me", context_type=None, intent="", target_directories=[ROOT]),
        ctx=_ctx(),
        limit=5,
        mode=RetrieverMode.QUICK,
    )
    # the rebuild task was scheduled on the running loop; let it run
    import asyncio

    await asyncio.sleep(0.05)
    assert lexical_index.count("acme") == 1
