# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Local BM25 index: indexing, identifier handling, scoping, moves and deletes."""

import pytest

from openviking.retrieve.lexical_index import (
    LexicalIndex,
    build_match_expression,
    query_terms,
    split_identifiers,
)


def _rec(doc_id, uri, content, *, level=2, parent=None, account="acme", ctype="resource"):
    return {
        "id": doc_id,
        "account_id": account,
        "uri": uri,
        "parent_uri": parent or uri.rsplit("/", 1)[0],
        "context_type": ctype,
        "level": level,
        "name": uri.rsplit("/", 1)[-1],
        "abstract": content[:80],
        "content": content,
    }


@pytest.fixture
def index():
    idx = LexicalIndex(":memory:")
    yield idx
    idx.close()


def test_identifier_splitting_and_terms():
    assert split_identifiers("parse_abstract_overview camelCaseName a.b-c") == (
        "parse abstract overview camel Case Name a b c"
    )
    terms = query_terms("Where is parse_abstract_overview defined?")
    assert "parse_abstract_overview" in terms
    assert "abstract" in terms and "overview" in terms
    assert build_match_expression("   ") is None
    assert build_match_expression("x") is None


def test_exact_identifier_is_found_even_split(index):
    index.upsert_many(
        [
            _rec("1", "viking://resources/proj/a.py", "def parse_abstract_overview(raw): ..."),
            _rec("2", "viking://resources/proj/b.md", "Notes about cooking pasta."),
        ]
    )
    hits = index.search("parse_abstract_overview", "acme")
    assert [h.uri for h in hits] == ["viking://resources/proj/a.py"]
    assert hits[0].score == 1.0
    hits = index.search("overview parser", "acme")
    assert hits and hits[0].uri == "viking://resources/proj/a.py"
    assert index.search("pasta", "acme")[0].uri == "viking://resources/proj/b.md"


def test_scoping_by_parent_target_type_level_and_user(index):
    index.upsert_many(
        [
            _rec(
                "1", "viking://resources/proj/a.py", "token alpha", parent="viking://resources/proj"
            ),
            _rec(
                "2",
                "viking://resources/other/b.py",
                "token alpha",
                parent="viking://resources/other",
            ),
            _rec("3", "viking://user/alice/memories/m.md", "token alpha", ctype="memory", level=2),
            _rec("4", "viking://user/bob/memories/m.md", "token alpha", ctype="memory", level=2),
            _rec(
                "5", "viking://resources/proj", "token alpha", level=0, parent="viking://resources"
            ),
            _rec("6", "viking://resources/proj/a.py", "token alpha", account="zeta"),
        ]
    )
    assert {h.uri for h in index.search("alpha", "acme")} == {
        "viking://resources/proj/a.py",
        "viking://resources/other/b.py",
        "viking://user/alice/memories/m.md",
        "viking://user/bob/memories/m.md",
        "viking://resources/proj",
    }
    assert [
        h.uri for h in index.search("alpha", "acme", parent_uri="viking://resources/other")
    ] == ["viking://resources/other/b.py"]
    assert {
        h.uri for h in index.search("alpha", "acme", target_directories=["viking://resources/proj"])
    } == {
        "viking://resources/proj/a.py",
        "viking://resources/proj",
    }
    assert {h.uri for h in index.search("alpha", "acme", context_type="memory")} == {
        "viking://user/alice/memories/m.md",
        "viking://user/bob/memories/m.md",
    }
    assert [h.uri for h in index.search("alpha", "acme", level=[0])] == ["viking://resources/proj"]
    visible = {h.uri for h in index.search("alpha", "acme", visible_user_id="alice")}
    assert "viking://user/bob/memories/m.md" not in visible
    assert "viking://user/alice/memories/m.md" in visible
    assert index.count("zeta") == 1


def test_upsert_replaces_delete_and_move(index):
    index.upsert(_rec("1", "viking://resources/proj/a.py", "old content"))
    index.upsert(_rec("1", "viking://resources/proj/a.py", "brand new content"))
    assert index.count() == 1
    assert index.search("old", "acme") == []
    assert index.search("brand", "acme")[0].uri == "viking://resources/proj/a.py"

    index.upsert(_rec("2", "viking://resources/proj/sub/c.py", "deep file"))
    assert index.move_uri("acme", "viking://resources/proj", "viking://resources/moved") == 2
    hits = index.search("brand", "acme")
    assert hits[0].uri == "viking://resources/moved/a.py"
    assert hits[0].parent_uri == "viking://resources/moved"
    assert index.search("deep", "acme")[0].uri == "viking://resources/moved/sub/c.py"

    assert index.delete_uris("acme", ["viking://resources/moved/sub"]) == 1
    assert index.delete_ids(["1"]) == 1
    assert index.count() == 0


def test_bm25_scores_are_normalized_and_ordered(index):
    index.upsert_many(
        [
            _rec("1", "viking://resources/p/dense.md", "retry retry retry backoff"),
            _rec(
                "2",
                "viking://resources/p/sparse.md",
                "one retry mention among lots of other words here",
            ),
        ]
    )
    hits = index.search("retry backoff", "acme")
    assert [h.uri for h in hits] == [
        "viking://resources/p/dense.md",
        "viking://resources/p/sparse.md",
    ]
    assert hits[0].score == 1.0
    assert 0.5 <= hits[1].score < 1.0
