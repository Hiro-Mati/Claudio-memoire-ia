# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Semantic generation and vector operations shared by semantic tasks."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from openviking.parse.parsers.constants import (
    CODE_EXTENSIONS,
    DOCUMENTATION_EXTENSIONS,
    FILE_TYPE_CODE,
    FILE_TYPE_DOCUMENTATION,
    FILE_TYPE_OTHER,
)
from openviking.parse.parsers.media.utils import (
    MPEG_TS_PROBE_BYTES,
    generate_audio_summary,
    generate_image_summary,
    generate_video_summary,
    get_media_type,
)
from openviking.prompts import render_prompt
from openviking.server.identity import RequestContext, Role
from openviking.storage.viking_fs import get_viking_fs
from openviking.telemetry import bind_telemetry_stage
from openviking.utils.ingest_options import IngestOptions
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config import get_openviking_config
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticService:
    """Generate file/directory semantics and submit their vector updates."""

    def __init__(
        self,
        max_concurrent_llm: int = 32,
        default_ctx: Optional[RequestContext] = None,
    ) -> None:
        self.max_concurrent_llm = max_concurrent_llm
        self._default_ctx = default_ctx or RequestContext(
            user=UserIdentifier.the_default_user(),
            role=Role.ROOT,
        )

    @staticmethod
    def _detect_file_type(file_name: str) -> str:
        file_name_lower = file_name.lower()
        if any(file_name_lower.endswith(ext) for ext in DOCUMENTATION_EXTENSIONS):
            return FILE_TYPE_DOCUMENTATION

        from openviking.parse.parsers.code.ast.providers import supports_code_skeleton

        if supports_code_skeleton(file_name):
            return FILE_TYPE_CODE
        if any(file_name_lower.endswith(ext) for ext in CODE_EXTENSIONS):
            return FILE_TYPE_CODE
        return FILE_TYPE_OTHER

    async def generate_text_summary(
        self,
        file_path: str,
        file_name: str,
        llm_sem: asyncio.Semaphore,
        ctx: Optional[RequestContext] = None,
    ) -> Dict[str, Any]:
        viking_fs = get_viking_fs()
        config = get_openviking_config()
        vlm = config.vlm
        content = await viking_fs.read_file(file_path, ctx=ctx or self._default_ctx)
        if isinstance(content, bytes):
            from openviking.utils.embedding_utils import _decode_text_bytes

            content = _decode_text_bytes(content)

        def result(summary: str) -> Dict[str, Any]:
            return {"name": file_name, "summary": summary}

        max_chars = config.semantic.max_file_content_chars
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(truncated)"

        file_type = self._detect_file_type(file_name)
        if file_type == FILE_TYPE_CODE:
            from openviking.parse.parsers.code.ast import extract_skeleton_result

            extraction = extract_skeleton_result(file_name, content)
            if extraction.text:
                skeleton = extraction.text[: config.semantic.max_skeleton_chars]
                return result(skeleton)
            if not vlm.is_available():
                logger.warning("VLM not available for code summary fallback: %s", file_path)
                return result("")

            from openviking.session.memory.utils.language import resolve_output_language

            prompt = render_prompt(
                "semantic.code_summary",
                {
                    "file_name": file_name,
                    "content": content,
                    "output_language": resolve_output_language(content, config=config),
                },
            )
            async with llm_sem:
                with bind_telemetry_stage("resource_summarize"):
                    summary = await vlm.get_completion_async(prompt)
            return result(summary.strip())

        if not vlm.is_available():
            logger.warning("VLM not available, using empty summary")
            return result("")

        from openviking.session.memory.utils.language import resolve_output_language

        prompt = render_prompt(
            (
                "semantic.document_summary"
                if file_type == FILE_TYPE_DOCUMENTATION
                else "semantic.file_summary"
            ),
            {
                "file_name": file_name,
                "content": content,
                "output_language": resolve_output_language(content, config=config),
            },
        )
        async with llm_sem:
            with bind_telemetry_stage("resource_summarize"):
                summary = await vlm.get_completion_async(prompt)
        return result(summary.strip())

    async def generate_file_summary(
        self,
        file_path: str,
        llm_sem: Optional[asyncio.Semaphore] = None,
        ctx: Optional[RequestContext] = None,
    ) -> Dict[str, Any]:
        file_name = file_path.rsplit("/", 1)[-1]
        llm_sem = llm_sem or asyncio.Semaphore(self.max_concurrent_llm)
        media_type = get_media_type(file_name, None)
        if file_name.lower().endswith(".ts"):
            try:
                prefix = await get_viking_fs().read(
                    file_path,
                    offset=0,
                    size=MPEG_TS_PROBE_BYTES,
                    ctx=ctx,
                )
            except Exception:
                prefix = None
            media_type = get_media_type(file_name, None, content=prefix)
        if media_type == "image":
            return await generate_image_summary(file_path, file_name, llm_sem, ctx=ctx)
        if media_type == "audio":
            return await generate_audio_summary(file_path, file_name, llm_sem, ctx=ctx)
        if media_type == "video":
            return await generate_video_summary(file_path, file_name, llm_sem, ctx=ctx)
        return await self.generate_text_summary(file_path, file_name, llm_sem, ctx=ctx)

    @staticmethod
    def _replace_index_references(generated_content: str, file_index_map: Dict[int, str]) -> str:
        def replace_index(match: re.Match[str]) -> str:
            return file_index_map.get(int(match.group(1)), match.group(0))

        return re.sub(r"\[(\d+)\]", replace_index, generated_content)

    @staticmethod
    def _truncate_generated_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]

        first_sentence_end = None
        last_sentence_end = None
        for match in re.finditer(r"\.(?!\d)(?=\s|$)|[!?](?=\s|$)|[。？！]", text):
            if first_sentence_end is None:
                first_sentence_end = match.end()
            if match.end() <= max_chars:
                last_sentence_end = match.end()
            elif last_sentence_end is not None:
                break
        if last_sentence_end is not None:
            return text[:last_sentence_end].strip()
        if first_sentence_end is not None:
            return text[:first_sentence_end].strip()

        candidate = text[: max_chars - 3].rstrip()
        word_boundary = candidate.rfind(" ")
        return (
            candidate[:word_boundary].rstrip() + "..." if word_boundary > 0 else candidate + "..."
        )

    @staticmethod
    def _extract_abstract(overview: str) -> str:
        content_lines = []
        in_header = True
        for line in overview.splitlines():
            if in_header and line.startswith("#"):
                continue
            if in_header and line.strip():
                in_header = False
            if not in_header:
                if line.startswith("##"):
                    break
                if line.strip():
                    content_lines.append(line.strip())
        return "\n".join(content_lines).strip()

    def normalize_overview(self, generated_content: str) -> Tuple[str, str]:
        return self.enforce_size_limits(
            generated_content,
            self._extract_abstract(generated_content),
        )

    def enforce_size_limits(self, overview: str, abstract: str) -> Tuple[str, str]:
        semantic = get_openviking_config().semantic
        if len(overview) > semantic.overview_max_chars:
            overview = self._truncate_generated_text(overview, semantic.overview_max_chars)
        if len(abstract) > semantic.abstract_max_chars:
            abstract = self._truncate_generated_text(abstract, semantic.abstract_max_chars)
        return overview, abstract

    @staticmethod
    def parse_overview(overview_content: str) -> Dict[str, str]:
        summaries: Dict[str, str] = {}
        if not overview_content or not overview_content.strip():
            return summaries

        current_file = None
        current_summary_lines: List[str] = []
        for line in overview_content.splitlines():
            header_match = re.match(r"^###\s+(.+?)\s*$", line)
            if header_match:
                if current_file and current_summary_lines:
                    summaries[current_file] = " ".join(current_summary_lines).strip()
                file_name = header_match.group(1).strip()
                parts = file_name.split()
                current_file = parts[0] if len(parts) >= 2 and parts[0] == parts[1] else file_name
                current_summary_lines = []
                continue

            numbered_match = re.match(r"^\[(\d+)\]\s+(.+?):\s*(.+)$", line)
            if numbered_match:
                if current_file and current_summary_lines:
                    summaries[current_file] = " ".join(current_summary_lines).strip()
                current_file = numbered_match.group(2).strip()
                current_summary_lines = [numbered_match.group(3).strip()]
                continue

            if current_file:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    current_summary_lines.append(stripped)
        if current_file and current_summary_lines:
            summaries[current_file] = " ".join(current_summary_lines).strip()
        return summaries

    async def generate_overview(
        self,
        dir_uri: str,
        file_summaries: List[Dict[str, str]],
        children_abstracts: List[Dict[str, str]],
        llm_sem: Optional[asyncio.Semaphore] = None,
    ) -> str:
        config = get_openviking_config()
        semantic = config.semantic
        if not config.vlm.is_available():
            logger.warning("VLM not available, using default overview")
            return f"# {dir_uri.rsplit('/', 1)[-1]}\n\n[Directory overview is not ready]"

        from openviking.session.memory.utils.language import resolve_output_language

        file_index_map = {idx: item["name"] for idx, item in enumerate(file_summaries, 1)}
        file_lines = [
            f"[{idx}] {item['name']}: {item['summary']}"
            for idx, item in enumerate(file_summaries, 1)
        ]
        file_text = "\n".join(file_lines) or "None"
        child_text = (
            "\n".join(f"- {item['name']}/: {item['abstract']}" for item in children_abstracts)
            or "None"
        )
        language_source = (
            "\n".join(
                part
                for part in (
                    file_text if file_summaries else "",
                    child_text if children_abstracts else "",
                )
                if part
            )
            or dir_uri.rsplit("/", 1)[-1]
        )
        output_language = resolve_output_language(language_source, config=config)
        estimated_size = len(file_text) + len(child_text)
        entry_count = len(file_summaries) + len(children_abstracts)

        if (
            estimated_size > semantic.max_overview_prompt_chars
            and entry_count > semantic.overview_batch_size
        ):
            return await self._generate_overview_batched(
                dir_uri,
                file_summaries,
                children_abstracts,
                file_index_map,
                llm_sem or asyncio.Semaphore(self.max_concurrent_llm),
                output_language,
            )

        if estimated_size > semantic.max_overview_prompt_chars:
            budget = semantic.max_overview_prompt_chars - len(child_text)
            per_file = max(100, budget // max(len(file_summaries), 1))
            file_text = "\n".join(
                f"[{idx}] {item['name']}: {item['summary'][:per_file]}"
                for idx, item in enumerate(file_summaries, 1)
            )
        return await self._generate_overview_once(
            dir_uri,
            file_text,
            child_text,
            file_index_map,
            output_language,
        )

    async def _generate_overview_once(
        self,
        dir_uri: str,
        file_summaries: str,
        children_abstracts: str,
        file_index_map: Dict[int, str],
        output_language: str,
    ) -> str:
        try:
            prompt = render_prompt(
                "semantic.overview_generation",
                {
                    "dir_name": dir_uri.rsplit("/", 1)[-1],
                    "file_summaries": file_summaries,
                    "children_abstracts": children_abstracts,
                    "output_language": output_language,
                },
            )
            with bind_telemetry_stage("resource_summarize"):
                overview = await get_openviking_config().vlm.get_completion_async(prompt)
            return self._replace_index_references(overview, file_index_map).strip()
        except Exception as error:
            logger.error("Failed to generate overview for %s: %s", dir_uri, error, exc_info=True)
            return f"# {dir_uri.rsplit('/', 1)[-1]}\n\n[Directory overview is not generated]"

    async def _generate_overview_batched(
        self,
        dir_uri: str,
        file_summaries: List[Dict[str, str]],
        children_abstracts: List[Dict[str, str]],
        file_index_map: Dict[int, str],
        llm_sem: asyncio.Semaphore,
        output_language: str,
    ) -> str:
        config = get_openviking_config()
        vlm = config.vlm
        batch_size = config.semantic.overview_batch_size
        dir_name = dir_uri.rsplit("/", 1)[-1]
        work = [("file", idx, item) for idx, item in enumerate(file_summaries, 1)] + [
            ("child", None, item) for item in children_abstracts
        ]
        batches = [work[idx : idx + batch_size] for idx in range(0, len(work), batch_size)]

        async def run_batch(
            batch: List[tuple[str, Optional[int], Dict[str, str]]],
        ) -> Optional[str]:
            file_lines = []
            child_lines = []
            indexes = {}
            for kind, index, item in batch:
                if kind == "file":
                    assert index is not None
                    indexes[index] = item["name"]
                    file_lines.append(f"[{index}] {item['name']}: {item['summary']}")
                else:
                    child_lines.append(f"- {item['name']}/: {item['abstract']}")
            prompt = render_prompt(
                "semantic.overview_generation",
                {
                    "dir_name": dir_name,
                    "file_summaries": "\n".join(file_lines) or "None",
                    "children_abstracts": "\n".join(child_lines) or "None",
                    "output_language": output_language,
                },
            )
            try:
                async with llm_sem:
                    with bind_telemetry_stage("resource_summarize"):
                        partial = await vlm.get_completion_async(prompt)
                return self._replace_index_references(partial, indexes).strip()
            except Exception as error:
                logger.warning("Failed to generate overview batch for %s: %s", dir_uri, error)
                return None

        partials = [result for result in await asyncio.gather(*map(run_batch, batches)) if result]
        if not partials:
            return f"# {dir_name}\n\n[Directory overview is not generated]"
        if len(partials) == 1:
            return partials[0]
        try:
            prompt = render_prompt(
                "semantic.overview_generation",
                {
                    "dir_name": dir_name,
                    "file_summaries": "\n\n---\n\n".join(partials),
                    "children_abstracts": "None",
                    "output_language": output_language,
                },
            )
            with bind_telemetry_stage("resource_summarize"):
                overview = await vlm.get_completion_async(prompt)
            return self._replace_index_references(overview, file_index_map).strip()
        except Exception as error:
            logger.error("Failed to merge overview batches for %s: %s", dir_uri, error)
            return partials[0]

    async def vectorize_directory(
        self,
        uri: str,
        context_type: str,
        abstract: str,
        overview: str,
        ctx: Optional[RequestContext] = None,
        ingest_options: IngestOptions | None = None,
    ) -> None:
        from openviking.utils.embedding_utils import vectorize_directory_meta

        await vectorize_directory_meta(
            uri=uri,
            abstract=abstract,
            overview=overview,
            context_type=context_type,
            ctx=ctx or self._default_ctx,
            ingest_options=ingest_options,
        )

    async def vectorize_file(
        self,
        parent_uri: str,
        context_type: str,
        file_path: str,
        summary_dict: Dict[str, str],
        ctx: Optional[RequestContext] = None,
        use_summary: bool = False,
        preserve_existing_created_at: bool = False,
        ingest_options: IngestOptions | None = None,
    ) -> None:
        from openviking.utils.embedding_utils import vectorize_file

        await vectorize_file(
            file_path=file_path,
            summary_dict=summary_dict,
            parent_uri=parent_uri,
            context_type=context_type,
            ctx=ctx or self._default_ctx,
            use_summary=use_summary,
            preserve_existing_created_at=preserve_existing_created_at,
            ingest_options=ingest_options,
        )


__all__ = ["SemanticService"]
