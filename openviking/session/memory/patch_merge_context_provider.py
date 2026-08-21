# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Context provider for planning merges of structured memory-file patches."""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass, field
from typing import Any

from openviking.server.identity import RequestContext
from openviking.session.memory.case_aggregation import (
    CASE_PENDING_SOURCES_FIELD,
    PROPOSED_CASE_IDENTITY_FIELD,
)
from openviking.session.memory.dataclass import MemoryFile, MemoryTypeSchema
from openviking.session.memory.session_extract_context_provider import (
    SessionExtractContextProvider,
)
from openviking.session.memory.utils.language import resolve_output_language_from_text

_SYSTEM_HIDDEN_FIELDS = {
    "source_extraction_id",
    "source_extraction_ids",
    "last_update_trace_id",
    "feedback_stats",
    "_case_source_ids",
    CASE_PENDING_SOURCES_FIELD,
    PROPOSED_CASE_IDENTITY_FIELD,
}
_MAX_EXTRA_CANDIDATE_FILES = 10
_MAX_CASE_EXTRA_CANDIDATE_FILES = 3
_PATCH_METADATA_KEYS = ("confidence",)


@dataclass(slots=True)
class PatchMergePatch:
    """A before/after memory-file patch rendered as field-level line diffs."""

    before_file: MemoryFile | None
    after_file: MemoryFile
    metadata: dict[str, Any] = field(default_factory=dict)
    proposal_id: str | None = None

    @property
    def target_uri(self) -> str | None:
        return self.after_file.uri or (
            self.before_file.uri if self.before_file is not None else None
        )

    @property
    def memory_type(self) -> str:
        return str(
            self.after_file.memory_type
            or (self.before_file.memory_type if self.before_file is not None else "")
            or self.after_file.extra_fields.get("memory_type")
            or (
                self.before_file.extra_fields.get("memory_type")
                if self.before_file is not None
                else ""
            )
        )

    @property
    def target_name(self) -> str:
        fields = self.after_file.extra_fields or {}
        memory_type = self.memory_type
        type_specific_key = f"{str(memory_type).rstrip('s')}_name"
        name = (
            fields.get(type_specific_key)
            or fields.get("experience_name")  # backward compat
            or fields.get("name")
        )
        if name:
            return str(name)
        uri = self.target_uri
        if uri:
            # For SKILL.md-style paths, use the directory name.
            if uri.endswith("/SKILL.md"):
                parts = uri.rstrip("/").split("/")
                if len(parts) >= 2:
                    return parts[-2]
            return uri.rstrip("/").split("/")[-1].removesuffix(".md")
        return "unknown"


def _resolve_patch_output_language(patches: list[PatchMergePatch]) -> str:
    return resolve_output_language_from_text(_patch_language_text(patches), fallback_language="en")


def _patch_language_text(patches: list[PatchMergePatch]) -> str:
    parts: list[str] = []
    for patch in patches:
        parts.extend(_memory_file_language_text(patch.after_file))
    return "\n".join(part for part in parts if part)


def _memory_file_language_text(file: MemoryFile) -> list[str]:
    parts: list[str] = []
    for key, value in (file.extra_fields or {}).items():
        if key in _SYSTEM_HIDDEN_FIELDS or key in {"memory_type", "version"}:
            continue
        parts.extend(_string_values(value))
    if file.content:
        parts.append(file.content)
    return parts


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for entry in value for item in _string_values(entry)]
    if isinstance(value, dict):
        return [item for entry in value.values() for item in _string_values(entry)]
    return []


class PatchMergeContextProvider(SessionExtractContextProvider):
    """Provide original memory files and structured field diffs to the merge planner.

    The provider is generic: callers decide which patches to pass in; this class
    only exposes original files as prefetched read results and memory-file field
    diffs as compact merge context.
    """

    def __init__(
        self,
        *,
        memory_type: str,
        patches: list[PatchMergePatch],
        required_file_uris: list[str] | None = None,
        output_language: str | None = None,
    ):
        super().__init__(messages=[])
        self.memory_type = memory_type
        self.required_file_uris = list(required_file_uris or [])
        self.patches = list(patches)
        self._output_language = output_language or _resolve_patch_output_language(self.patches)
        self._candidate_files_by_id: dict[str, MemoryFile] = {}

    def instruction(self) -> str:
        output_language = self._output_language
        schema = self._get_registry().get(self.memory_type)
        content_fields = schema.content_field_names() if schema is not None else ()
        structured_fields = ", ".join(f"`{name}`" for name in content_fields)
        experience_guidance = ""
        if self.memory_type == "experiences":
            experience_guidance = """
When merging `experiences`, every synthesized upsert must keep the skill-loader
format: put the full runtime-facing Markdown in the `constraint` field, with
exactly these top-level sections: `## Situation`, `## Reminder`, `## Procedure`,
`## Verification`, `## Fallback`, and `## Anti-pattern`. Do not output only a production reminder such as
`# name` plus `## 规则`; convert that content into the six sections. Preserve
source-binding, applicability, scope ambiguity, canonical value/source-field
rules, and anti-patterns from the strongest patches.
"""
        case_instruction = ""
        if self.memory_type == "cases":
            case_instruction = """
When merging `cases`, first classify each current proposal against every listed
existing candidate and every earlier current proposal. For each pair, classify
goal, subject, action_pattern, success_boundary, and context_constraints as one
of MATCH, COMPATIBLE, UNKNOWN, or CONFLICT. Put every classification in
case_comparisons. Goal, subject, and success_boundary are hard identity
dimensions: UNKNOWN blocks automatic merge and CONFLICT requires a separate
Case. The server computes the weighted score and validates the grouping; do not
invent a numeric score.

Use exactly one primary Case for each current proposal. Keep ambiguous proposals
as separate draft Cases and never update multiple candidates. For a group that
requires full compaction, return complete replacement values for
task_signature, input, situation, rubric, and evidence. Retain only stable
cross-session task features. Exact one-run IDs, dates, amounts, counts, paths,
and transient tool state belong to trajectories. The rubric field means
internal observable success criteria inferred from the user task and execution
evidence; never copy evaluator rubrics, weights, hidden answers, or benchmark
criteria.

The `Pending Case Source Summaries` section contains every independent source
received since the previous compaction. Use all of them when rewriting the Case,
but do not copy source_id or one-run details into the Case body.

Compaction is semantic consensus, not content union. Keep the shared goal,
subject, action pattern, completion boundary, and compatible stable
preconditions. When sources use different concrete values for the same role,
remove the values and record the role in input.variable_types. A fact supported
only by one source must not become a stable Case constraint. Missing mention in
a later source is not by itself a conflict with an already promoted consensus;
explicit incompatible requirements are a conflict and require a separate Case.
Never concatenate source descriptions, criteria, or evidence.

For an ordinary accepted source update, use an empty field_operations object:
the server will update only provenance, links, counters, and the pending-source
buffer. Rewrite no body field partially. When the second independent source,
the configured source delta, identity drift, content conflict, or body-length
threshold requires full compaction, replace all five dynamic fields together:
task_signature, input, situation, rubric, and evidence. When this first
compaction promotes a draft Case, whether stored or formed in the current
batch, also return generalized_case_identity. Derive it from the semantic
intersection of every pending source, with no one-run values. Do not return
generalized_case_identity for ordinary updates or use it to rewrite an already
promoted Case.
"""
        return f"""You are a memory patch merge planner.

You are given original memory files and structured memory-file field diffs. Return only a compact merge plan that follows the provided JSON schema. Do not repeat complete memory files.

Do not call tools. Output JSON only.

All memory content must be written in {output_language}.

Reconcile independent extraction patch proposals: merge duplicate/overlapping
memories into one canonical group, and keep distinct memories separate. Every
input proposal_id must appear exactly once in either groups[].proposal_ids or
delete_proposal_ids. A candidate_id may be included in at most one group when an
existing stored memory should be the canonical item. canonical_proposal_id must
also be present in that group's proposal_ids. Use field_operations only for
fields that need synthesis; each field uses its existing schema merge_op.
Normalize URI/path variants for directory/filename fields. Treat path segment
fields as stable schema identifiers, not free-form labels. Reuse existing
equivalent directories across singular/plural, synonym, or language/script
variants. For new segments, use singular snake_case for English and one concise
canonical term for Chinese; e.g. book not books, 书籍 not 书/图书. If a loser URI
is an existing file, the server will delete it after validating the group. Use
delete_proposal_ids only for explicit deletion proposals that should remain deletes.

When synthesizing an upsert, preserve the `{self.memory_type}` schema's structured
content fields: {structured_fields}. Put only content bodies in those fields; the
storage template adds the Markdown structure. A provenance-only ordinary Case update
may use the explicitly allowed empty field_operations described below.
{experience_guidance}
{case_instruction}
"""

    def get_tools(self) -> list[str]:
        return []

    def get_memory_schemas(self, ctx: RequestContext) -> list[MemoryTypeSchema]:
        del ctx
        schema = self._get_registry().get(self.memory_type)
        if schema is None or not schema.enabled:
            raise ValueError(f"Memory schema not found or disabled: {self.memory_type}")
        return [schema]

    async def prefetch(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        call_id = 0
        file_uris = await self._resolve_prefetch_file_uris()
        for uri in file_uris:
            call_id = await self._append_structured_read_result(
                messages,
                call_id,
                uri,
            )

        self._candidate_files_by_id = self._build_candidate_files_by_id()
        messages.append(
            {
                "role": "user",
                "content": _render_field_diff_patches(
                    self.patches,
                    schema=self._get_registry().get(self.memory_type),
                    candidate_files_by_id=self._candidate_files_by_id,
                ),
            }
        )
        return messages

    @property
    def candidate_files_by_id(self) -> dict[str, MemoryFile]:
        return dict(self._candidate_files_by_id)

    def _build_candidate_files_by_id(self) -> dict[str, MemoryFile]:
        if self.memory_type == "cases":
            return {
                candidate_id_for_uri(uri): memory_file
                for uri, memory_file in sorted(self.read_file_contents.items())
                if uri
            }
        patch_uris = {
            uri
            for patch in self.patches
            for uri in (
                patch.target_uri,
                patch.before_file.uri if patch.before_file is not None else None,
            )
            if uri
        }
        return {
            candidate_id_for_uri(uri): memory_file
            for uri, memory_file in sorted(self.read_file_contents.items())
            if uri and uri not in patch_uris
        }

    async def _resolve_prefetch_file_uris(self) -> list[str]:
        """Resolve required files plus semantic-search candidates for this merge."""

        required_uris = _dedupe_uris(self.required_file_uris)
        if not any(patch.before_file is None for patch in self.patches):
            return required_uris

        if self.memory_type == "cases":
            max_extra_candidate_files = _MAX_CASE_EXTRA_CANDIDATE_FILES
        else:
            max_extra_candidate_files = min(
                _MAX_EXTRA_CANDIDATE_FILES,
                max(5, len(required_uris)),
            )
        search_limit = max_extra_candidate_files * 2
        candidate_uris = await self._search_candidate_file_uris(limit=search_limit)
        extra_uris: list[str] = []
        required_set = set(required_uris)
        for uri in candidate_uris:
            if not uri or uri in required_set or uri in extra_uris:
                continue
            extra_uris.append(uri)
            if len(extra_uris) >= max_extra_candidate_files:
                break
        return [*required_uris, *extra_uris]

    async def _search_candidate_file_uris(self, *, limit: int) -> list[str]:
        schema = self._get_registry().get(self.memory_type)
        if schema is None or not schema.directory:
            return []
        search_dirs = self._render_search_directories(schema)
        if not search_dirs:
            return []
        query = _build_patch_search_query(self.patches)
        if not query:
            return []
        return await self.search_files(query=query, search_uris=search_dirs, limit=limit)

    def _render_search_directories(self, schema: MemoryTypeSchema) -> list[str]:
        if self._isolation_handler:
            return list(dict.fromkeys(self._isolation_handler.render_schema_directories(schema)))

        ctx = self._ctx
        user = getattr(ctx, "user", None)
        user_id = (
            getattr(ctx, "user_id", None)
            or getattr(user, "user_id", None)
            or _infer_user_space_from_uris(self.required_file_uris)
            or _infer_user_space_from_uris([patch.target_uri for patch in self.patches])
        )
        if not user_id:
            return []

        from openviking.session.memory.utils.uri import render_template

        return [render_template(schema.directory, {"user_space": user_id})]


def _render_field_diff_patches(
    patches: list[PatchMergePatch],
    *,
    schema: MemoryTypeSchema | None = None,
    candidate_files_by_id: dict[str, MemoryFile] | None = None,
) -> str:
    if not patches:
        return "# Memory File Patches\n\nNo patches provided."
    rendered = [
        _render_one_field_diff_patch(index, patch, schema=schema)
        for index, patch in enumerate(patches, start=1)
    ]
    content = "# Memory File Patches\n\n" + "\n\n".join(rendered).rstrip()
    candidates = dict(candidate_files_by_id or {})
    if candidates:
        candidate_lines = [
            f"- candidate_id={candidate_id} uri={memory_file.uri}"
            for candidate_id, memory_file in candidates.items()
        ]
        content += "\n\n# Existing Candidate IDs\n\n" + "\n".join(candidate_lines)
    pending_sources = _collect_case_pending_sources(patches, candidates)
    if pending_sources:
        content += "\n\n# Pending Case Source Summaries\n\n"
        content += "\n".join(f"- {_compact_value(item)}" for item in pending_sources)
    return content


def _collect_case_pending_sources(
    patches: list[PatchMergePatch],
    candidate_files_by_id: dict[str, MemoryFile],
) -> list[dict[str, Any]]:
    if not any(patch.memory_type == "cases" for patch in patches):
        return []
    files = [
        file
        for patch in patches
        for file in (patch.before_file, patch.after_file)
        if file is not None
    ]
    files.extend(candidate_files_by_id.values())
    result: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for file in files:
        raw_sources = dict(file.extra_fields or {}).get(CASE_PENDING_SOURCES_FIELD)
        if not isinstance(raw_sources, list):
            continue
        for item in raw_sources:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id") or "").strip()
            if not source_id:
                continue
            if source_id not in result:
                order.append(source_id)
            result[source_id] = dict(item)
    return [result[source_id] for source_id in order]


def _render_one_field_diff_patch(
    index: int,
    patch: PatchMergePatch,
    *,
    schema: MemoryTypeSchema | None = None,
) -> str:
    proposal_suffix = f" [proposal_id={patch.proposal_id}]" if patch.proposal_id else ""
    lines = [f"Patch {index}{proposal_suffix}"]
    if patch.metadata:
        compact_metadata = _compact_patch_metadata(patch.metadata)
        if compact_metadata:
            lines.append(f"  meta: {_compact_value(compact_metadata)}")
    field_diffs = _field_diffs(patch.before_file, patch.after_file, schema=schema)
    if not field_diffs:
        lines.append("  (no changes)")
        return "\n".join(lines)
    for field_name, diff in field_diffs:
        lines.append(f"  {field_name}:")
        # Strip unified diff headers (---, +++) but keep @@ hunk markers and content
        for diff_line in diff.splitlines():
            if diff_line.startswith("---") or diff_line.startswith("+++"):
                continue
            lines.append(f"    {diff_line}")
    return "\n".join(lines)


def _field_diffs(
    before_file: MemoryFile | None,
    after_file: MemoryFile,
    *,
    schema: MemoryTypeSchema | None = None,
) -> list[tuple[str, str]]:
    before_fields = (
        _memory_file_fields(before_file, schema=schema) if before_file is not None else {}
    )
    after_fields = _memory_file_fields(after_file, schema=schema)
    diffs: list[tuple[str, str]] = []
    for field_name in sorted(set(before_fields) | set(after_fields)):
        before_value = before_fields.get(field_name)
        after_value = after_fields.get(field_name)
        if before_value == after_value:
            continue
        diff = _value_unified_diff(
            field_name=field_name,
            before_value=before_value,
            after_value=after_value,
        )
        if diff.strip():
            diffs.append((field_name, diff))
    return diffs


def _memory_file_fields(
    file: MemoryFile,
    *,
    schema: MemoryTypeSchema | None = None,
) -> dict[str, Any]:
    fields = dict(file.extra_fields or {})
    for hidden_field in _SYSTEM_HIDDEN_FIELDS:
        fields.pop(hidden_field, None)
    if file.memory_type is not None:
        fields["memory_type"] = file.memory_type
    has_structured_content = bool(schema) and any(
        fields.get(name) for name in schema.content_field_names() if name != "content"
    )
    if file.content and not has_structured_content:
        fields["content"] = file.content
    if file.links:
        fields["links"] = file.links
    if file.backlinks:
        fields["backlinks"] = file.backlinks
    return fields


def _value_unified_diff(*, field_name: str, before_value: Any, after_value: Any) -> str:
    before_lines = _value_lines(before_value)
    after_lines = _value_lines(after_value)
    diff_lines = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"{field_name}.before",
        tofile=f"{field_name}.after",
        n=1,
        lineterm="",
    )
    return "\n".join(diff_lines)


def _value_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.splitlines()
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).splitlines()


def _compact_value(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _hide_system_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _hide_system_fields(item)
            for key, item in value.items()
            if key not in _SYSTEM_HIDDEN_FIELDS
        }
    if isinstance(value, list):
        return [_hide_system_fields(item) for item in value]
    return value


def _compact_patch_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only metadata that helps reconcile patch proposals.

    Full gradient metadata can contain large duplicated fields (links, uris, and
    memory_fields). The patch body already renders the target URI and field
    changes, while source links are merged outside the LLM response. Keep only
    decision signals that help the merge model rank or reconcile proposals.
    """

    cleaned = _hide_system_fields(dict(metadata or {}))
    result = {
        key: cleaned[key]
        for key in _PATCH_METADATA_KEYS
        if key in cleaned and _metadata_value_is_useful(cleaned[key])
    }

    return result


def _metadata_value_is_useful(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if isinstance(value, (list, dict, tuple, set)) and not value:
        return False
    return True


def _dedupe_uris(uris: list[str] | None) -> list[str]:
    return list(dict.fromkeys(uri for uri in (uris or []) if uri))


def _build_patch_search_query(patches: list[PatchMergePatch]) -> str:
    parts: list[str] = []
    for patch in patches:
        if patch.target_name:
            parts.append(str(patch.target_name))
        if patch.target_uri:
            parts.append(str(patch.target_uri).rstrip("/").split("/")[-1].removesuffix(".md"))
        content = str(patch.after_file.content or "")
        if content:
            parts.append(_truncate_query_text(content, 1200))
    return _truncate_query_text("\n\n".join(parts), 5000)


def _truncate_query_text(text: Any, max_chars: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _infer_user_space_from_uris(uris: list[str | None]) -> str | None:
    for uri in uris:
        if not uri:
            continue
        prefix = "viking://user/"
        if not uri.startswith(prefix):
            continue
        rest = uri.removeprefix(prefix)
        user_space = rest.split("/", 1)[0]
        if user_space and user_space != "memories":
            return user_space
    return None


def candidate_id_for_uri(uri: str) -> str:
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:12]
    return f"candidate:{digest}"
