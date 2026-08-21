# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Search Service for OpenViking.

Provides semantic search operations: search, find.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from openviking.server.identity import RequestContext
from openviking.session.memory.case_aggregation import normalize_case_status
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.storage.viking_fs import VikingFS
from openviking.utils.image_search import (
    image_bytes_to_data_uri,
    is_data_image_uri,
    is_http_url,
    is_viking_uri,
)
from openviking_cli.exceptions import InvalidArgumentError, NotInitializedError
from openviking_cli.utils import get_logger

if TYPE_CHECKING:
    from openviking.session import Session

logger = get_logger(__name__)

RetrievalPurpose = Literal[
    "general",
    "agent_recall",
    "case_merge",
    "case_governance",
]


def _case_memory_file_uri(uri: str) -> Optional[str]:
    source_uri = str(uri or "").split("#", 1)[0].rstrip("/")
    if "/memories/cases/" not in source_uri or not source_uri.endswith(".md"):
        return None
    if source_uri.endswith(("/.abstract.md", "/.overview.md")):
        return None
    return source_uri


def _is_case_directory_summary_uri(uri: str) -> bool:
    source_uri = str(uri or "").split("#", 1)[0].rstrip("/")
    return "/memories/cases/" in source_uri and source_uri.endswith(
        ("/.abstract.md", "/.overview.md")
    )


async def _apply_agent_recall_policy(
    result: Any,
    *,
    viking_fs: VikingFS,
    ctx: RequestContext,
) -> Any:
    """Hide draft, degraded, and archived Cases from Agent-facing recall."""
    memories = getattr(result, "memories", None)
    if not isinstance(memories, list):
        return result

    query_results = getattr(result, "query_results", None) or []
    contexts = list(memories)
    for query_result in query_results:
        contexts.extend(getattr(query_result, "matched_contexts", None) or [])

    case_uris = {
        case_uri
        for context in contexts
        for candidate_uri in [getattr(context, "uri", "")]
        if (case_uri := _case_memory_file_uri(candidate_uri)) is not None
    }
    for context in contexts:
        for relation in getattr(context, "relations", None) or []:
            case_uri = _case_memory_file_uri(getattr(relation, "uri", ""))
            if case_uri is not None:
                case_uris.add(case_uri)

    async def _is_promoted(uri: str) -> tuple[str, bool]:
        try:
            raw = await viking_fs.read_file(uri, ctx=ctx)
            memory_file = MemoryFileUtils.read(raw, uri=uri)
            status = normalize_case_status(memory_file.extra_fields.get("case_status"))
            return uri, status == "promoted"
        except Exception as exc:
            logger.warning(
                "Exclude Case from agent recall because its status could not be verified: "
                "uri=%s error=%s",
                uri,
                exc,
            )
            return uri, False

    promoted_by_uri = dict(await asyncio.gather(*(_is_promoted(uri) for uri in case_uris)))

    def _is_visible(context: Any) -> bool:
        uri = getattr(context, "uri", "")
        if _is_case_directory_summary_uri(uri):
            return False
        case_uri = _case_memory_file_uri(uri)
        return case_uri is None or promoted_by_uri.get(case_uri, False)

    def _filter_relations(context: Any) -> None:
        relations = getattr(context, "relations", None)
        if not isinstance(relations, list):
            return
        context.relations = [
            relation
            for relation in relations
            if (
                not _is_case_directory_summary_uri(getattr(relation, "uri", ""))
                and (
                    (case_uri := _case_memory_file_uri(getattr(relation, "uri", ""))) is None
                    or promoted_by_uri.get(case_uri, False)
                )
            )
        ]

    result.memories = [context for context in memories if _is_visible(context)]
    for context in result.memories:
        _filter_relations(context)
    for query_result in query_results:
        matched_contexts = getattr(query_result, "matched_contexts", None)
        if not isinstance(matched_contexts, list):
            continue
        query_result.matched_contexts = [
            context for context in matched_contexts if _is_visible(context)
        ]
        for context in query_result.matched_contexts:
            _filter_relations(context)

    if hasattr(result, "total"):
        result.total = (
            len(result.memories)
            + len(getattr(result, "resources", None) or [])
            + len(getattr(result, "skills", None) or [])
        )
    return result


def _ensure_non_empty_query(query: str, image_url: Optional[str] = None) -> None:
    if not query.strip() and not image_url:
        raise InvalidArgumentError("Search query or image_url must not be empty.")


class SearchService:
    """Semantic search service."""

    def __init__(self, viking_fs: Optional[VikingFS] = None):
        self._viking_fs = viking_fs

    def set_viking_fs(self, viking_fs: VikingFS) -> None:
        """Set VikingFS instance (for deferred initialization)."""
        self._viking_fs = viking_fs

    def _ensure_initialized(self) -> VikingFS:
        """Ensure VikingFS is initialized."""
        if not self._viking_fs:
            raise NotInitializedError("VikingFS")
        return self._viking_fs

    def is_intent_enabled(self) -> bool:
        """Whether search uses session context for LLM intent analysis.

        When false, callers should skip session.load / get_context_for_search:
        VikingFS.search ignores session_info and searches with the raw query.
        Default is True (matches RetrievalConfig) when config is unset.
        """
        if not self._viking_fs or self._viking_fs.retrieval_config is None:
            return True
        return bool(self._viking_fs.retrieval_config.enable_intent)

    async def _resolve_image_url(
        self,
        image_url: Optional[str],
        ctx: RequestContext,
    ) -> Optional[str]:
        if not image_url:
            return None
        if is_viking_uri(image_url):
            viking_fs = self._ensure_initialized()
            content = await viking_fs.read_file_bytes(image_url, ctx=ctx)
            return image_bytes_to_data_uri(content, image_url)
        if is_data_image_uri(image_url) or is_http_url(image_url):
            return image_url
        raise InvalidArgumentError(
            "image_url must be a data:image base64 URI, http(s) URL, or viking:// URI."
        )

    async def search(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: Union[str, List[str]] = "",
        session: Optional["Session"] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
        level: Optional[List[int]] = None,
        image_url: Optional[str] = None,
        retrieval_purpose: RetrievalPurpose = "general",
    ) -> Any:
        """Complex search with session context.

        Args:
            query: Query string
            target_uri: Target directory URI(s), supports str or List[str]
            session: Session object for context
            limit: Max results
            score_threshold: Score threshold
            filter: Metadata filters
            level: Filter by level (0=abstract, 1=overview, 2=file)

        Returns:
            FindResult
        """
        resolved_image_url = await self._resolve_image_url(image_url, ctx)
        _ensure_non_empty_query(query, resolved_image_url)
        viking_fs = self._ensure_initialized()

        session_info = None
        # Intent off: session_info is unused by VikingFS — skip the archive/message scan.
        if session is not None and self.is_intent_enabled() and not resolved_image_url:
            session_info = await session.get_context_for_search(query)

        result = await viking_fs.search(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            session_info=session_info,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter,
            level=level,
            image_url=resolved_image_url,
        )
        if retrieval_purpose == "agent_recall":
            result = await _apply_agent_recall_policy(result, viking_fs=viking_fs, ctx=ctx)
        return result

    async def find(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: Union[str, List[str]] = "",
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
        level: Optional[List[int]] = None,
        image_url: Optional[str] = None,
        retrieval_purpose: RetrievalPurpose = "general",
    ) -> Any:
        """Semantic search without session context.

        Args:
            query: Query string
            target_uri: Target directory URI(s), supports str or List[str]
            limit: Max results
            score_threshold: Score threshold
            filter: Metadata filters
            level: Filter by level (0=abstract, 1=overview, 2=file)

        Returns:
            FindResult
        """
        resolved_image_url = await self._resolve_image_url(image_url, ctx)
        _ensure_non_empty_query(query, resolved_image_url)
        viking_fs = self._ensure_initialized()
        result = await viking_fs.find(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter,
            level=level,
            image_url=resolved_image_url,
        )
        if retrieval_purpose == "agent_recall":
            result = await _apply_agent_recall_policy(result, viking_fs=viking_fs, ctx=ctx)
        return result
