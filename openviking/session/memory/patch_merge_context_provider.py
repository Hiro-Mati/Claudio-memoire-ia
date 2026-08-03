# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Context provider for merging structured memory-file patches via ExtractLoop."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from openviking.server.identity import RequestContext
from openviking.session.memory.dataclass import MemoryFile, MemoryTypeSchema
from openviking.session.memory.session_extract_context_provider import (
    SessionExtractContextProvider,
)
from openviking.session.memory.utils.language import resolve_output_language_from_text
from openviking.telemetry import tracer

_SYSTEM_HIDDEN_FIELDS = {
    "source_extraction_id",
    "source_extraction_ids",
    "last_update_trace_id",
    "feedback_stats",
}
_MAX_EXTRA_CANDIDATE_FILES = 10
_PATCH_METADATA_KEYS = ("confidence",)


@dataclass(slots=True)
class PatchMergePatch:
    """A before/after memory-file patch rendered as field-level line diffs."""

    before_file: MemoryFile | None
    after_file: MemoryFile
    metadata: dict[str, Any] = field(default_factory=dict)
    patch_id: int | None = None

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


class PatchMergeSourceBinding(BaseModel):
    """Invocation-local lineage from one merged operation to its input patches."""

    operation_kind: Literal["upsert", "delete"]
    operation_page_id: int
    source_patch_ids: list[int]


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
    """Provide original memory files and structured field diffs to ExtractLoop.

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
        seen_patch_ids: set[int] = set()
        for index, patch in enumerate(self.patches, start=1):
            patch.patch_id = patch.patch_id or index
            if patch.patch_id in seen_patch_ids:
                raise ValueError(f"Duplicate PatchMerge patch_id: {patch.patch_id}")
            seen_patch_ids.add(patch.patch_id)
        self._output_language = output_language or _resolve_patch_output_language(self.patches)

    def create_operations_model(self, schema_model_generator: Any, role_scope: Any = None) -> Any:
        base_model = super().create_operations_model(schema_model_generator, role_scope)
        return create_model(
            f"PatchMerge{base_model.__name__}",
            __base__=base_model,
            source_bindings=(
                list[PatchMergeSourceBinding],
                Field(
                    ...,
                    description=(
                        "Required source lineage for every PatchMerge output operation. "
                        "Reference output page_ids and the contributing input patch_ids."
                    ),
                ),
            ),
        )

    def validate_and_attach_operation_metadata(
        self,
        raw_operations: Any,
        resolved_operations: Any,
    ) -> list[str]:
        valid_patch_ids = {
            int(patch.patch_id) for patch in self.patches if patch.patch_id is not None
        }
        upserts_by_page_id = {
            int(operation.page_id): operation
            for operation in list(resolved_operations.upsert_operations or [])
            if operation.page_id is not None
        }
        resolved_delete_uris = {
            str(memory_file.uri)
            for memory_file in list(resolved_operations.delete_file_contents or [])
            if memory_file.uri
        }
        page_id_map = getattr(self._extract_context, "page_id_map", None)
        deletes_by_page_id: dict[int, str] = {}
        for delete_id in list(getattr(raw_operations, "delete_ids", []) or []):
            page_id = getattr(delete_id, "delete_page_id", None)
            if page_id is None or page_id_map is None:
                continue
            uri = page_id_map.resolve(page_id)
            if uri and str(uri) in resolved_delete_uris:
                deletes_by_page_id[int(page_id)] = str(uri)
        bindings_by_key: dict[tuple[str, int], list[int]] = {}
        errors: list[str] = []

        for binding in list(getattr(raw_operations, "source_bindings", []) or []):
            key = (str(binding.operation_kind), int(binding.operation_page_id))
            if key in bindings_by_key:
                errors.append(
                    "duplicate source binding for "
                    f"operation_kind={key[0]} operation_page_id={key[1]}"
                )
                continue
            source_patch_ids = list(dict.fromkeys(int(item) for item in binding.source_patch_ids))
            if not source_patch_ids:
                errors.append(
                    f"empty source_patch_ids for operation_kind={key[0]} operation_page_id={key[1]}"
                )
                continue
            unknown_patch_ids = sorted(set(source_patch_ids) - valid_patch_ids)
            if unknown_patch_ids:
                errors.append(
                    f"unknown source_patch_ids={unknown_patch_ids} for "
                    f"operation_kind={key[0]} operation_page_id={key[1]}"
                )
                continue
            if key[0] == "upsert" and key[1] not in upserts_by_page_id:
                errors.append(f"unknown upsert operation_page_id={key[1]}")
                continue
            if key[0] == "delete" and key[1] not in deletes_by_page_id:
                errors.append(f"unknown delete operation_page_id={key[1]}")
                continue
            bindings_by_key[key] = source_patch_ids

        for page_id in sorted(upserts_by_page_id):
            if ("upsert", page_id) not in bindings_by_key:
                errors.append(f"missing source binding for upsert operation_page_id={page_id}")
        for page_id in sorted(deletes_by_page_id):
            if ("delete", page_id) not in bindings_by_key:
                errors.append(f"missing source binding for delete operation_page_id={page_id}")

        if errors:
            tracer.info(
                "[patch_merge] source binding validation failed "
                f"memory_type={self.memory_type} errors={errors}"
            )
            return errors

        for page_id, operation in upserts_by_page_id.items():
            operation.source_patch_ids = list(bindings_by_key[("upsert", page_id)])
        resolved_operations.delete_source_patch_ids = {
            uri: list(bindings_by_key[("delete", page_id)])
            for page_id, uri in deletes_by_page_id.items()
        }
        tracer.info(
            "[patch_merge] source bindings resolved "
            f"memory_type={self.memory_type} "
            f"upserts={[(page_id, operation.source_patch_ids) for page_id, operation in sorted(upserts_by_page_id.items())]} "
            f"deletes={resolved_operations.delete_source_patch_ids}"
        )
        return []

    def instruction(self) -> str:
        output_language = self._output_language
        schema = self._get_registry().get(self.memory_type)
        content_fields = schema.content_field_names() if schema is not None else ()
        structured_fields = ", ".join(f"`{name}`" for name in content_fields)
        experience_guidance = ""
        if self.memory_type == "experiences":
            experience_guidance = """Preserve source-binding, applicability, scope ambiguity,
canonical value/source-field rules, and anti-patterns from the strongest patches.
"""
        return f"""You are a memory patch merge agent.

You are given original memory files and structured memory-file field diffs. Merge them by producing final memory operations that follow the provided JSON schema.

Do not call tools. Output JSON only.

All memory content must be written in {output_language}.

Reconcile independent extraction patch proposals: merge duplicate/overlapping
memories into one canonical file patch, and keep distinct memories separate.
Normalize URI/path variants for directory/filename fields. Treat path segment
fields as stable schema identifiers, not free-form labels. Reuse existing
equivalent directories across singular/plural, synonym, or language/script
variants. For new segments, use singular snake_case for English and one concise
canonical term for Chinese; e.g. book not books, 书籍 not 书/图书. If a loser URI
is an existing file, put it in delete_ids; if it is only a new proposal, omit it.

Every upsert must preserve the `{self.memory_type}` schema's structured content fields:
{structured_fields}. Put only content bodies in those fields; the storage template adds the
Markdown structure.
{experience_guidance}
For every output upsert or delete, emit exactly one source binding in `source_bindings`.
Set `operation_page_id` to that output operation's page_id and list every contributing input
`patch_id` in `source_patch_ids`. Do not omit a binding, invent a patch_id, or bind an output to
an unrelated patch.
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

        messages.append(
            {
                "role": "user",
                "content": _render_field_diff_patches(
                    self.patches,
                    schema=self._get_registry().get(self.memory_type),
                ),
            }
        )
        return messages

    async def _resolve_prefetch_file_uris(self) -> list[str]:
        """Resolve required files plus semantic-search candidates for this merge."""

        required_uris = _dedupe_uris(self.required_file_uris)
        max_extra_candidate_files = min(_MAX_EXTRA_CANDIDATE_FILES, max(5, len(required_uris)))
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
) -> str:
    if not patches:
        return "# Memory File Patches\n\nNo patches provided."
    rendered = [
        _render_one_field_diff_patch(index, patch, schema=schema)
        for index, patch in enumerate(patches, start=1)
    ]
    return "# Memory File Patches\n\n" + "\n\n".join(rendered).rstrip()


def _render_one_field_diff_patch(
    index: int,
    patch: PatchMergePatch,
    *,
    schema: MemoryTypeSchema | None = None,
) -> str:
    patch_id = patch.patch_id or index
    lines = [f"Patch {index} [patch_id={patch_id}]"]
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
