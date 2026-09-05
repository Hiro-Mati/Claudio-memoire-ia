# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """Configuration for retrieval ranking behavior."""

    hotness_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for blending hotness into final retrieval scores. "
            "0 disables hotness boost; 1 uses only hotness."
        ),
    )
    score_propagation_alpha: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for each child result's own score when blending with its parent score "
            "during hierarchical retrieval. 0 uses only the parent score; "
            "1 uses only the child score."
        ),
    )
    recall_intent_timeout_s: float = Field(
        default=5.0,
        gt=0.0,
        description="Timeout in seconds for optional context query expansion.",
    )
    recall_rewrite_timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout in seconds for optional context digest rewriting.",
    )
    enable_intent: bool = Field(
        default=True,
        description=(
            "Whether search() loads session context and runs LLM intent analysis / query "
            "planning when session_id is present. false skips session load, "
            "get_context_for_search, and IntentAnalyzer — searches with the raw query only "
            "(same path as no-session search)."
        ),
    )

    lexical_index_enabled: bool = Field(
        default=False,
        description=(
            "Maintain a local BM25 (SQLite FTS5) index of context records next to the vector "
            "index. Required for lexical_boost > 0. The index is filled on vector writes and "
            "rebuilt lazily from the vector index when empty."
        ),
    )
    lexical_boost: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Bonus added to a candidate's similarity score from its normalized BM25 match "
            "(score = min(1, dense + lexical_boost * bm25)). Exact-token hits missing from "
            "the dense results are recovered with score lexical_boost * bm25. 0 disables "
            "lexical fusion."
        ),
    )
    lexical_limit: int = Field(
        default=20,
        ge=1,
        description="Maximum lexical hits considered per search scope.",
    )

    model_config = {"extra": "forbid"}
