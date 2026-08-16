# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""File semantic phase for resource and skill ingestion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openviking.server.identity import RequestContext
from openviking.utils.ingest_options import IngestOptions
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FileSemanticResult:
    summary: Dict[str, str]
    directory_changed: bool


class FileSemanticTask:
    """Generate one file summary and update only that file's vector."""

    def __init__(
        self,
        semantic_service: Any,
        *,
        context_type: str,
        ctx: RequestContext,
        llm_sem: asyncio.Semaphore,
        ingest_options: IngestOptions,
        skip_vectorization: bool,
        is_code_repo: bool,
    ) -> None:
        self._semantic_service = semantic_service
        self._context_type = context_type
        self._ctx = ctx
        self._llm_sem = llm_sem
        self._ingest_options = ingest_options
        self._skip_vectorization = skip_vectorization
        self._is_code_repo = is_code_repo

    async def run(
        self,
        *,
        parent_uri: str,
        file_path: str,
        content_changed: bool,
        existing_summary: Optional[Dict[str, str]],
    ) -> FileSemanticResult:
        file_name = file_path.rsplit("/", 1)[-1]
        summary = existing_summary
        vector_summary = existing_summary
        if summary is None:
            try:
                generated = await self._semantic_service.generate_file_summary(
                    file_path,
                    llm_sem=self._llm_sem,
                    ctx=self._ctx,
                )
            except Exception as error:
                logger.warning("Failed to generate summary for %s: %s", file_path, error)
                generated = {"name": file_name, "summary": ""}
            summary = {
                "name": str(generated.get("name") or file_name),
                "summary": str(generated.get("summary") or ""),
            }
            vector_summary = generated

        if content_changed and not self._skip_vectorization:
            assert vector_summary is not None
            await self._semantic_service.vectorize_file(
                parent_uri=parent_uri,
                context_type=self._context_type,
                file_path=file_path,
                summary_dict=vector_summary,
                ctx=self._ctx,
                use_summary=self._is_code_repo and bool(summary["summary"]),
                ingest_options=self._ingest_options,
            )

        return FileSemanticResult(
            summary=summary,
            directory_changed=content_changed or existing_summary is None,
        )


__all__ = ["FileSemanticResult", "FileSemanticTask"]
