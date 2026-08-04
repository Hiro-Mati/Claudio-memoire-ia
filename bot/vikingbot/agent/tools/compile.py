"""Request-local tools used by the compile structured task."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from fnmatch import fnmatchcase
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from openviking.core.namespace import context_type_for_uri, relative_uri_path
from openviking.core.skill_loader import SkillLoader, validate_skill_format
from openviking.session.memory.utils.link_renderer import LinkRenderer
from openviking.utils.path_safety import (
    safe_join_viking_uri,
    sanitize_relative_viking_path,
    validate_safe_viking_uri_path,
)
from openviking.utils.skill_processor import validate_skill_name
from openviking_cli.exceptions import OpenVikingError
from vikingbot.agent.tools.base import Tool, ToolContext
from vikingbot.compile.models import (
    COMPILE_STAGING_ROOT,
    COMPILE_WIKI_PAGE_ROOT,
    COMPILE_WORK_ROOT,
    CompileBundleDraft,
    CompileContract,
    CompileLimits,
    CompileOutputPlan,
    CompileOutputReceipt,
    CompileSkillChecklist,
    CompileSkillChecklistSelection,
    CompileSkillRule,
    CompileValidationIssue,
    CompileValidationReport,
    CompileWorkItem,
    CompileWorkReceipt,
)
from vikingbot.compile.renderer import (
    is_reserved_output_file_uri,
    is_reserved_wiki_page_uri,
    validate_declared_okf_markdown,
    validate_relative_file_path,
    validate_relative_page_path,
    wiki_page_path_from_title,
)

_LINK_FIELDS = frozenset({"f", "t", "link_type", "weight", "match_text", "description"})


def _normalize_workspace_path(path: str) -> str:
    normalized = sanitize_relative_viking_path(path)
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _path_is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _validate_public_output_path(path: str) -> str:
    normalized = _normalize_workspace_path(path)
    if _path_is_within(normalized, COMPILE_STAGING_ROOT) or _path_is_within(normalized, ".compile"):
        raise ValueError(f"reserved internal Compile path: {normalized}")
    return normalized


def _uri_in_roots(uri: str, roots: tuple[str, ...]) -> bool:
    normalized = str(uri or "").strip().rstrip("/")
    if not normalized.startswith("viking://"):
        return False
    try:
        normalized = validate_safe_viking_uri_path(normalized)
    except ValueError:
        return False
    return any(
        normalized == root.rstrip("/") or bool(relative_uri_path(root, normalized))
        for root in roots
    )


def _skill_workspace_read_hint(uri: str) -> str | None:
    value = str(uri or "").strip()
    if value.startswith("viking://skills/"):
        value = value[len("viking://") :]
    if not value.startswith("skills/"):
        return None
    try:
        return _normalize_workspace_path(value)
    except ValueError:
        return None


def _successful_multi_read_uris(rendered: str, uris: list[str]) -> set[str]:
    successful: set[str] = set()
    for uri in uris:
        start = f"--- START OF {uri} ---\n"
        end = f"\n--- END OF {uri} ---"
        _, found, remainder = rendered.partition(start)
        if not found:
            continue
        content, found, _ = remainder.partition(end)
        if found and not content.lstrip().startswith("ERROR:"):
            successful.add(uri)
    return successful


class CompileScopedTool(Tool):
    """Guard an existing OpenViking read tool without changing its implementation."""

    def __init__(
        self,
        tool: Tool,
        *,
        roots: tuple[str, ...],
        limits: CompileLimits,
        result_budget: dict[str, int],
        budget_lock: asyncio.Lock,
        observed_content_uris: set[str] | None = None,
    ):
        self._tool = tool
        self._roots = roots
        self._limits = limits
        self._result_budget = result_budget
        self._budget_lock = budget_lock
        self._observed_content_uris = observed_content_uris

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._tool.parameters

    async def execute(self, tool_context: ToolContext, **kwargs: Any) -> str:
        uris: list[str] = []
        if self.name == "openviking_search":
            value = kwargs.get("target_uri")
            if not value:
                return "Error: Compile search requires target_uri within the task scope."
            uris.append(str(value))
        elif self.name in {"openviking_list", "openviking_grep", "openviking_glob"}:
            value = kwargs.get("uri")
            if not value or str(value).rstrip("/") in {"viking:", "viking://"}:
                return f"Error: Compile {self.name} requires uri within the task scope."
            uris.append(str(value))
            if self.name == "openviking_list" and kwargs.get("recursive"):
                kwargs["node_limit"] = min(
                    int(kwargs.get("node_limit") or self._limits.target_inventory_entries),
                    self._limits.target_inventory_entries,
                )
        elif self.name == "openviking_multi_read":
            values = kwargs.get("uris")
            if not isinstance(values, list) or not values:
                return "Error: Compile multi-read requires at least one URI."
            if len(values) > self._limits.tool_uri_count:
                return "Error: Compile multi-read URI limit exceeded."
            uris.extend(str(value) for value in values)

        if len(uris) > self._limits.tool_uri_count:
            return "Error: Compile tool URI limit exceeded."
        for uri in uris:
            if not _uri_in_roots(uri, self._roots):
                workspace_path = _skill_workspace_read_hint(uri)
                if workspace_path:
                    return (
                        "Error: Skill workspace files must be read with read_file using path "
                        f'"{workspace_path}", not with an openviking_* tool.'
                    )
                return f"Error: URI is outside the Compile task scope: {uri}"

        result = await self._tool.execute(tool_context, **kwargs)
        if (
            isinstance(result, str)
            and result.startswith("Error")
            and not result.startswith("Error:")
        ):
            result = "Error: " + result[len("Error") :].lstrip(" :")
        rendered = str(result)
        size = len(rendered.encode("utf-8"))
        if size > self._limits.tool_result_bytes:
            return "Error: Compile tool result exceeds the per-call size limit."
        async with self._budget_lock:
            total = self._result_budget.get("bytes", 0) + size
            if total > self._limits.tool_total_result_bytes:
                return "Error: Compile task tool-result budget exceeded."
            self._result_budget["bytes"] = total
        if (
            self.name == "openviking_multi_read"
            and self._observed_content_uris is not None
            and not rendered.lstrip().startswith("Error:")
        ):
            self._observed_content_uris.update(_successful_multi_read_uris(rendered, uris))
        return rendered


class CompileReadTrackingTool(Tool):
    """Track successful workspace reads while preserving the existing file tool."""

    def __init__(self, tool: Tool, observed_workspace_paths: set[str]):
        self._tool = tool
        self._observed_workspace_paths = observed_workspace_paths

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._tool.parameters

    async def execute(self, tool_context: ToolContext, **kwargs: Any) -> str:
        result = str(await self._tool.execute(tool_context, **kwargs))
        if not result.lstrip().startswith("Error"):
            path = kwargs.get("path")
            if path:
                try:
                    self._observed_workspace_paths.add(_normalize_workspace_path(str(path)))
                except ValueError:
                    pass
        return result


class ReadWorkspaceFilesTool(Tool):
    """Read many workspace files in one call for read-only orchestration phases.

    Read-only phases (planning, plan audit, candidate audit) must read every
    required workspace artifact before submitting. Doing that one ``read_file``
    at a time costs one model turn per file; a compile task with many worker
    reports spends minutes just cycling reads. This tool reads a bounded batch
    in a single turn, records each successful read in ``observed_workspace_paths``
    exactly like :class:`CompileReadTrackingTool`, and charges the same
    per-call/total result budget as :class:`CompileScopedTool` so batching never
    grows the context beyond the existing limits.
    """

    def __init__(
        self,
        read_tool: Tool,
        *,
        limits: CompileLimits,
        result_budget: dict[str, int],
        budget_lock: asyncio.Lock,
        observed_workspace_paths: set[str],
    ):
        self._read_tool = read_tool
        self._limits = limits
        self._result_budget = result_budget
        self._budget_lock = budget_lock
        self._observed_workspace_paths = observed_workspace_paths

    @property
    def name(self) -> str:
        return "read_workspace_files"

    @property
    def description(self) -> str:
        return (
            "Read several workspace files at once. Provide the exact workspace paths "
            "(for example the source manifest, Skill checklist, and every work report) "
            "and receive their contents in one call. Prefer this over reading files one "
            "at a time before submitting."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace file paths to read in this batch.",
                }
            },
            "required": ["paths"],
        }

    async def execute(
        self, tool_context: ToolContext, paths: list[str] | None = None, **kwargs: Any
    ) -> str:
        if not isinstance(paths, list) or not paths:
            return "Error: read_workspace_files requires a non-empty list of paths."
        if len(paths) > self._limits.tool_uri_count:
            return "Error: Compile tool URI limit exceeded."
        sections: list[str] = []
        for path in paths:
            result = str(await self._read_tool.execute(tool_context, path=str(path)))
            if not result.lstrip().startswith("Error"):
                try:
                    self._observed_workspace_paths.add(_normalize_workspace_path(str(path)))
                except ValueError:
                    pass
            sections.append(f"===== {path} =====\n{result}")
        rendered = "\n\n".join(sections)
        size = len(rendered.encode("utf-8"))
        if size > self._limits.tool_result_bytes:
            return "Error: Compile tool result exceeds the per-call size limit."
        async with self._budget_lock:
            total = self._result_budget.get("bytes", 0) + size
            if total > self._limits.tool_total_result_bytes:
                return "Error: Compile task tool-result budget exceeded."
            self._result_budget["bytes"] = total
        return rendered


class SubmitCompileWorkTool(Tool):
    """Accept one source-complete worker report stored in the task workspace."""

    def __init__(
        self,
        *,
        work_item: CompileWorkItem,
        observed_content_uris: set[str],
        limits: CompileLimits,
    ):
        self.work_item = work_item
        self.observed_content_uris = observed_content_uris
        self.limits = limits
        self.receipt: CompileWorkReceipt | None = None

    @property
    def name(self) -> str:
        return "submit_compile_work"

    @property
    def description(self) -> str:
        return (
            "Submit the completed report for the assigned Compile work item after reading every "
            "assigned source. The report must be a non-empty UTF-8 file under the required work "
            "directory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return CompileWorkReceipt.model_json_schema()

    async def execute(
        self,
        tool_context: ToolContext,
        work_item_id: str,
        processed_source_uris: list[str],
        report_workspace_path: str,
        **kwargs: Any,
    ) -> str:
        del kwargs
        self.receipt = None
        try:
            receipt = CompileWorkReceipt.model_validate(
                {
                    "work_item_id": work_item_id,
                    "processed_source_uris": processed_source_uris,
                    "report_workspace_path": report_workspace_path,
                }
            )
            expected = set(self.work_item.source_uris)
            if receipt.work_item_id != self.work_item.work_item_id:
                raise ValueError("work_item_id does not match the assigned work item")
            if set(receipt.processed_source_uris) != expected:
                raise ValueError("processed_source_uris must exactly match the assigned sources")
            unread = sorted(expected - self.observed_content_uris)
            if unread:
                raise ValueError("assigned sources were not read: " + ", ".join(unread))
            path = _normalize_workspace_path(receipt.report_workspace_path)
            required_root = f"{COMPILE_WORK_ROOT}/{receipt.work_item_id}"
            if not _path_is_within(path, required_root):
                raise ValueError(f"report_workspace_path must be under {required_root}/")
            if tool_context.sandbox_manager is None:
                raise ValueError("task sandbox is unavailable")
            sandbox = await tool_context.sandbox_manager.get_sandbox(tool_context.session_key)
            payload = await sandbox.read_file_bytes(path)
            if not payload or len(payload) > self.limits.work_report_bytes:
                raise ValueError("work report is empty or exceeds the per-report size limit")
            payload.decode("utf-8")
        except (UnicodeDecodeError, ValidationError, ValueError) as exc:
            return f"Error: Invalid Compile work receipt: {exc}"
        self.receipt = receipt.model_copy(update={"report_workspace_path": path})
        return f"Compile work item '{receipt.work_item_id}' accepted."


class SubmitCompileChecklistTool(Tool):
    """Accept an ephemeral checklist grounded to trusted Skill evidence blocks."""

    def __init__(
        self,
        *,
        evidence_blocks: Mapping[str, tuple[str, str]],
        required_evidence_ids: set[str] | None = None,
        max_critical_rules: int | None = None,
    ):
        self.evidence_blocks = dict(evidence_blocks)
        self.required_evidence_ids = required_evidence_ids or set()
        self.max_critical_rules = max_critical_rules
        self.structured_result: CompileSkillChecklist | None = None

    @property
    def name(self) -> str:
        return "submit_compile_skill_checklist"

    @property
    def description(self) -> str:
        return (
            "Submit every binding output and quality rule explicitly stated by the selected Skill, "
            "selecting one trusted evidence block and the phase where it can be checked."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return CompileSkillChecklistSelection.model_json_schema()

    async def execute(
        self,
        tool_context: ToolContext,
        rules: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        del tool_context, kwargs
        self.structured_result = None
        try:
            selection = CompileSkillChecklistSelection.model_validate({"rules": rules})
            rule_ids = [rule.rule_id for rule in selection.rules]
            if len(rule_ids) != len(set(rule_ids)):
                raise ValueError("rule_id values must be unique")
            selected_evidence_ids = {rule.evidence_id for rule in selection.rules}
            missing_evidence_ids = sorted(self.required_evidence_ids - selected_evidence_ids)
            if missing_evidence_ids:
                raise ValueError(
                    "strongly binding Skill evidence blocks must each be selected: "
                    + ", ".join(missing_evidence_ids)
                )
            if self.max_critical_rules is not None:
                critical_count = sum(1 for rule in selection.rules if rule.critical)
                if critical_count > self.max_critical_rules:
                    raise ValueError(
                        f"at most {self.max_critical_rules} rules may be critical=true; "
                        f"got {critical_count}. Keep only the few rules whose violation makes "
                        "the output unusable; set the rest critical=false."
                    )
            grounded_rules = []
            for rule in selection.rules:
                evidence = self.evidence_blocks.get(rule.evidence_id)
                if evidence is None:
                    raise ValueError(
                        f"rule {rule.rule_id} has unknown evidence_id {rule.evidence_id}"
                    )
                path, quote = evidence
                grounded_rules.append(
                    CompileSkillRule(
                        rule_id=rule.rule_id,
                        statement=rule.statement,
                        evidence_path=path,
                        evidence_quote=quote,
                        phase=rule.phase,
                        critical=rule.critical,
                    )
                )
            checklist = CompileSkillChecklist(rules=grounded_rules)
        except (ValidationError, ValueError) as exc:
            return f"Error: Invalid Compile Skill checklist: {exc}"
        self.structured_result = checklist
        return f"Compile Skill checklist accepted with {len(checklist.rules)} rule(s)."


class SubmitCompilePlanTool(Tool):
    """Accept the complete, source-mapped output TODO list before materialization."""

    def __init__(
        self,
        *,
        source_ids: set[str],
        source_roots: Mapping[str, str] | None = None,
        source_file_uris: set[str],
        coverage_unit_ids: set[str],
        report_ids: set[str],
        target_uri: str,
        catalog_uris: set[str],
        file_catalog_uris: set[str],
        contract: CompileContract,
        limits: CompileLimits,
        required_workspace_reads: set[str],
        observed_workspace_paths: set[str],
        wiki_uri_resolver: Callable[[str], Awaitable[bool]] | None = None,
    ):
        self.source_ids = source_ids
        self.source_roots = dict(source_roots or {})
        self.source_file_uris = source_file_uris
        self.coverage_unit_ids = coverage_unit_ids
        self.report_ids = report_ids
        self.target_uri = target_uri.rstrip("/")
        self.catalog_uris = catalog_uris
        self.file_catalog_uris = file_catalog_uris
        self.contract = contract
        self.limits = limits
        self.required_workspace_reads = required_workspace_reads
        self.observed_workspace_paths = observed_workspace_paths
        self.wiki_uri_resolver = wiki_uri_resolver
        self.structured_result: CompileOutputPlan | None = None

    @property
    def name(self) -> str:
        return "submit_compile_plan"

    @property
    def description(self) -> str:
        return (
            "Submit the complete deduplicated output plan. This is the Compile TODO list; "
            "do not include bodies or claim that planned files already exist."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        schema = CompileOutputPlan.model_json_schema()
        link_def = schema.get("$defs", {}).get("WikiLink", {})
        match_schema = link_def.get("properties", {}).get("match_text")
        if isinstance(match_schema, dict):
            match_schema["description"] = (
                "Exact non-empty anchor text that will appear in the source page body outside "
                "frontmatter, code, existing Markdown links, and citations."
            )
        return schema

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        # Some providers double-encode a large, otherwise valid tool payload as {"raw": "{...}"}.
        # Decode only a complete JSON object and only when no real schema field is present.
        if set(params) == {"raw"} and isinstance(params["raw"], str):
            try:
                decoded = json.loads(params["raw"])
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                params.clear()
                params.update(decoded)
        if "raw" in params:
            return ["raw must be a complete JSON object encoded with the tool schema fields"]
        return super().validate_params(params)

    async def execute(
        self,
        tool_context: ToolContext,
        pages: list[dict[str, Any]] | None = None,
        files: list[dict[str, Any]] | None = None,
        links: list[dict[str, Any]] | None = None,
        coverage: list[dict[str, Any]] | None = None,
        groups: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        del tool_context, kwargs
        self.structured_result = None
        try:
            unread = sorted(self.required_workspace_reads - self.observed_workspace_paths)
            if unread:
                raise ValueError("required workspace files were not read: " + ", ".join(unread))
            plan = CompileOutputPlan.model_validate(
                {
                    "pages": pages or [],
                    "files": files or [],
                    "links": links or [],
                    "coverage": coverage or [],
                    "groups": groups or [],
                }
            )
            await self._validate_plan(plan)
        except (ValidationError, ValueError) as exc:
            return f"Error: Invalid Compile output plan: {exc}"
        self.structured_result = plan
        return (
            f"Compile output plan accepted with {len(plan.pages)} page(s) and "
            f"{len(plan.files)} file(s)."
        )

    async def _validate_plan(self, plan: CompileOutputPlan) -> None:
        if self.source_roots and set(self.source_roots) != self.source_ids:
            raise ValueError("source_roots must exactly match source_ids")
        if len(plan.pages) > self.limits.output_pages:
            raise ValueError("page limit exceeded")
        if len(plan.files) > self.limits.output_files:
            raise ValueError("file limit exceeded")
        target_type = context_type_for_uri(self.target_uri)
        if target_type == "skill" and (plan.pages or plan.links):
            raise ValueError("Skill targets only accept artifact files")
        if target_type == "memory" and plan.files:
            raise ValueError("Memory targets only accept Wiki pages")
        if self.contract.output == "wiki" and plan.files:
            raise ValueError("Skill contract only permits Wiki page outputs")
        if self.contract.output == "files" and (plan.pages or plan.links):
            raise ValueError("Skill contract only permits file outputs")

        output_ids = [item.output_id for item in plan.pages]
        output_ids.extend(item.output_id for item in plan.files)
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("output_id values must be unique")
        page_ids = [page.page_id for page in plan.pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("page_id values must be unique")
        page_id_set = set(page_ids)

        paths: list[str] = []
        for page in plan.pages:
            if not page.title.strip() or not page.page_type.strip() or not page.summary.strip():
                raise ValueError(f"page {page.page_id} has empty required fields")
            if "\n" in page.summary.strip() or "\r" in page.summary.strip():
                raise ValueError(f"page {page.page_id} summary must be one line")
            invalid_source_ids = sorted(set(page.source_ids) - self.source_ids)
            if invalid_source_ids:
                raise ValueError(
                    f"page {page.page_id} has unknown source_ids: " + ", ".join(invalid_source_ids)
                )
            invalid_source_uris = sorted(set(page.source_uris) - self.source_file_uris)
            if invalid_source_uris:
                displayed = ", ".join(invalid_source_uris[:8])
                if len(invalid_source_uris) > 8:
                    displayed += f", ... ({len(invalid_source_uris)} total)"
                raise ValueError(
                    f"page {page.page_id} has source_uris absent from the exact leaf-source "
                    f"manifest: {displayed}. Replace them with a small representative set of exact "
                    "manifest leaf URIs; source_uris are evidence citations, while exhaustive source "
                    "disposition belongs in coverage"
                )
            if self.source_roots:
                for source_id in page.source_ids:
                    if not any(
                        _uri_in_roots(uri, (self.source_roots[source_id],))
                        for uri in page.source_uris
                    ):
                        raise ValueError(f"page {page.page_id} has no source URI for {source_id}")
                if any(
                    not any(
                        _uri_in_roots(uri, (self.source_roots[source_id],))
                        for source_id in page.source_ids
                    )
                    for uri in page.source_uris
                ):
                    raise ValueError(f"page {page.page_id} has source URIs outside its source_ids")
            invalid_report_ids = sorted(set(page.report_ids) - self.report_ids)
            if invalid_report_ids:
                raise ValueError(
                    f"page {page.page_id} has unknown report_ids: " + ", ".join(invalid_report_ids)
                )
            if page.update_uri is not None:
                update_uri = page.update_uri.rstrip("/")
                if is_reserved_wiki_page_uri(update_uri):
                    raise ValueError(f"page {page.page_id} cannot update a reserved Wiki file")
                if not await self._is_wiki_uri(update_uri):
                    raise ValueError(f"page {page.page_id} update_uri is not a Wiki page")
                relative = relative_uri_path(self.target_uri, update_uri)
                if not relative:
                    raise ValueError(f"page {page.page_id} update_uri is outside the target")
                paths.append(_validate_public_output_path(validate_relative_page_path(relative)))
            else:
                relative = _validate_public_output_path(
                    validate_relative_page_path(page.path_hint or "")
                )
                if safe_join_viking_uri(self.target_uri, relative) in self.file_catalog_uris:
                    raise ValueError(f"page {page.page_id} path already exists; use update_uri")
                paths.append(relative)
        for index, file in enumerate(plan.files):
            invalid_report_ids = sorted(set(file.report_ids) - self.report_ids)
            if invalid_report_ids:
                raise ValueError(
                    f"file {index} has unknown report_ids: " + ", ".join(invalid_report_ids)
                )
            if file.update_uri is not None:
                if target_type == "skill":
                    raise ValueError("Skill plans require relative path entries, not update_uri")
                if file.update_uri.rstrip("/") not in self.file_catalog_uris:
                    raise ValueError(f"file {index} update_uri is not in the target catalog")
                relative = relative_uri_path(self.target_uri, file.update_uri.rstrip("/"))
                if not relative:
                    raise ValueError(f"file {index} update_uri is outside the target")
                paths.append(validate_relative_file_path(relative))
            else:
                relative = validate_relative_file_path(
                    _validate_public_output_path(file.path or "")
                )
                if safe_join_viking_uri(self.target_uri, relative) in self.file_catalog_uris:
                    raise ValueError(f"file {index} path already exists; use update_uri")
                paths.append(relative)
        if len(paths) != len(set(paths)):
            raise ValueError("planned final output paths must be unique")
        if target_type == "skill":
            skill_names = {path.split("/", 1)[0] for path in paths if "/" in path}
            if len(skill_names) != 1 or any("/" not in path for path in paths):
                raise ValueError(
                    "Skill plans must contain exactly one top-level <skill-name>/ directory"
                )
            skill_name = next(iter(skill_names))
            if f"{skill_name}/SKILL.md" not in paths:
                raise ValueError(f"Skill plan must include {skill_name}/SKILL.md")

        for index, link in enumerate(plan.links):
            if link.f is None or link.t is None:
                raise ValueError(f"link {index} endpoints must be non-null")
            if link.f not in page_id_set or link.t not in page_id_set:
                raise ValueError(f"link {index} references an unknown page_id")
            if link.f == link.t:
                raise ValueError(f"link {index} must not be a self-link")
            if not link.match_text:
                raise ValueError(f"link {index} match_text is required")
            if "[" in link.match_text or "]" in link.match_text:
                raise ValueError(f"link {index} match_text must not contain Markdown brackets")

        decisions = {decision.unit_id: decision for decision in plan.coverage}
        if len(decisions) != len(plan.coverage):
            raise ValueError("coverage unit_id values must be unique")
        if not set(decisions) <= self.coverage_unit_ids:
            unknown = sorted(set(decisions) - self.coverage_unit_ids)
            raise ValueError("coverage contains unknown units: " + ", ".join(unknown))
        if self.contract.coverage == "all_sources" and set(decisions) != self.coverage_unit_ids:
            missing = sorted(self.coverage_unit_ids - set(decisions))
            extra = sorted(set(decisions) - self.coverage_unit_ids)
            raise ValueError(
                "coverage decisions must match all units"
                + (f"; missing: {', '.join(missing)}" if missing else "")
                + (f"; unknown: {', '.join(extra)}" if extra else "")
            )
        output_id_set = set(output_ids)
        for decision in plan.coverage:
            if not set(decision.output_ids) <= output_id_set:
                raise ValueError(f"coverage unit {decision.unit_id} references unknown outputs")
            if decision.duplicate_of and decision.duplicate_of not in self.coverage_unit_ids:
                raise ValueError(f"coverage unit {decision.unit_id} has invalid duplicate_of")
            if decision.duplicate_of == decision.unit_id:
                raise ValueError(f"coverage unit {decision.unit_id} cannot duplicate itself")
            if decision.duplicate_of and (
                decisions.get(decision.duplicate_of) is None
                or decisions[decision.duplicate_of].status != "covered"
            ):
                raise ValueError(
                    f"coverage unit {decision.unit_id} must duplicate a covered canonical unit"
                )

        missing = [path for path in self.contract.required_paths if path not in paths]
        missing.extend(
            pattern
            for pattern in self.contract.required_globs
            if not any(fnmatchcase(path, pattern) for path in paths)
        )
        if missing:
            raise ValueError("missing contract output(s): " + ", ".join(missing))

    async def _is_wiki_uri(self, uri: str) -> bool:
        if uri in self.catalog_uris:
            return True
        if uri not in self.file_catalog_uris or self.wiki_uri_resolver is None:
            return False
        if await self.wiki_uri_resolver(uri):
            self.catalog_uris.add(uri)
            return True
        return False


class SubmitCompileOutputsTool(Tool):
    """Accept a bounded output batch only after every planned file exists."""

    def __init__(
        self,
        *,
        expected_paths: Mapping[str, tuple[str, bool]],
        expected_source_uris: Mapping[str, tuple[str, ...]] | None = None,
        required_workspace_reads: set[str],
        observed_workspace_paths: set[str],
        limits: CompileLimits,
    ):
        self.expected_paths = dict(expected_paths)
        self.expected_source_uris = dict(expected_source_uris or {})
        self.required_workspace_reads = required_workspace_reads
        self.observed_workspace_paths = observed_workspace_paths
        self.limits = limits
        self.receipt: CompileOutputReceipt | None = None

    @property
    def name(self) -> str:
        return "submit_compile_outputs"

    @property
    def description(self) -> str:
        return "Mark the assigned Compile outputs complete after writing every exact path."

    @property
    def parameters(self) -> dict[str, Any]:
        return CompileOutputReceipt.model_json_schema()

    async def execute(
        self,
        tool_context: ToolContext,
        output_ids: list[str],
        **kwargs: Any,
    ) -> str:
        del kwargs
        self.receipt = None
        try:
            receipt = CompileOutputReceipt(output_ids=output_ids)
            if len(receipt.output_ids) != len(set(receipt.output_ids)) or set(
                receipt.output_ids
            ) != set(self.expected_paths):
                raise ValueError("output_ids must contain every assigned output exactly once")
            unread = sorted(self.required_workspace_reads - self.observed_workspace_paths)
            if unread:
                raise ValueError("required workspace files were not read: " + ", ".join(unread))
            if tool_context.sandbox_manager is None:
                raise ValueError("task sandbox is unavailable")
            sandbox = await tool_context.sandbox_manager.get_sandbox(tool_context.session_key)
            total = 0
            missing: list[str] = []
            for output_id, (path, require_utf8) in self.expected_paths.items():
                try:
                    payload = await sandbox.read_file_bytes(path)
                except Exception:
                    missing.append(f"{output_id} ({path})")
                    continue
                if not payload:
                    raise ValueError(f"output {output_id} is empty: {path}")
                if require_utf8:
                    try:
                        text = payload.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError(f"output {output_id} must be UTF-8 Markdown") from exc
                    if text.lstrip().startswith("---"):
                        raise ValueError(
                            f"Wiki output {output_id} must not include YAML frontmatter"
                        )
                    source_uris = self.expected_source_uris.get(output_id, ())
                    if source_uris and not any(f"]({uri})" in text for uri in source_uris):
                        raise ValueError(
                            f"Wiki output {output_id} must cite one of its planned source URIs"
                        )
                total += len(payload)
                if total > self.limits.output_total_bytes:
                    raise ValueError("output batch size limit exceeded")
            if missing:
                raise ValueError("planned output files do not exist: " + ", ".join(missing))
        except (ValidationError, ValueError) as exc:
            return f"Error: Invalid Compile output receipt: {exc}"
        self.receipt = receipt
        return f"Compile output batch accepted with {len(receipt.output_ids)} file(s)."


class SubmitCompileValidationTool(Tool):
    """Accept an independent audit of a candidate Compile bundle."""

    def __init__(
        self,
        *,
        required_workspace_reads: set[str] | None = None,
        observed_workspace_paths: set[str] | None = None,
        required_requirement_ids: set[str] | None = None,
        required_rule_ids: set[str] | None = None,
    ):
        self.required_workspace_reads = (
            required_workspace_reads if required_workspace_reads is not None else set()
        )
        self.observed_workspace_paths = (
            observed_workspace_paths if observed_workspace_paths is not None else set()
        )
        self.required_requirement_ids = required_requirement_ids or set()
        self.required_rule_ids = required_rule_ids or set()
        self.structured_result: CompileValidationReport | None = None

    @property
    def name(self) -> str:
        return "submit_compile_validation"

    @property
    def description(self) -> str:
        return "Submit whether the candidate fully satisfies the selected Skill and contract."

    @property
    def parameters(self) -> dict[str, Any]:
        return CompileValidationReport.model_json_schema()

    async def execute(
        self,
        tool_context: ToolContext,
        passed: bool,
        issues: list[dict[str, Any]] | None = None,
        checked_requirement_ids: list[str] | None = None,
        rule_checks: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        del tool_context, kwargs
        self.structured_result = None
        try:
            unread = sorted(self.required_workspace_reads - self.observed_workspace_paths)
            if unread:
                raise ValueError("required workspace files were not read: " + ", ".join(unread))
            report = CompileValidationReport(
                passed=passed,
                issues=[CompileValidationIssue.model_validate(issue) for issue in issues or []],
                checked_requirement_ids=checked_requirement_ids or [],
                rule_checks=rule_checks or [],
            )
            if set(report.checked_requirement_ids) != self.required_requirement_ids or len(
                report.checked_requirement_ids
            ) != len(self.required_requirement_ids):
                raise ValueError(
                    "checked_requirement_ids must contain every contract requirement exactly once"
                )
            checked_rule_ids = [check.rule_id for check in report.rule_checks]
            if set(checked_rule_ids) != self.required_rule_ids or len(checked_rule_ids) != len(
                self.required_rule_ids
            ):
                raise ValueError("rule_checks must contain every required audit rule exactly once")
            if report.passed and any(not check.passed for check in report.rule_checks):
                raise ValueError("passed cannot be true when a Skill rule check failed")
            if report.passed == bool(report.issues):
                raise ValueError("passed must be true exactly when issues is empty")
        except (ValidationError, ValueError) as exc:
            return f"Error: Invalid Compile validation report: {exc}"
        self.structured_result = report
        return "Compile validation accepted."


class SubmitCompileBundleTool(Tool):
    def __init__(
        self,
        *,
        source_ids: set[str],
        catalog_uris: set[str],
        file_catalog_uris: set[str] | None = None,
        target_uri: str,
        limits: CompileLimits,
        require_workspace_files: bool = False,
        require_workspace_pages: bool = False,
        workspace_baseline: set[str] | None = None,
        wiki_uri_resolver: Callable[[str], Awaitable[bool]] | None = None,
        exec_enabled: bool = True,
        contract: CompileContract | None = None,
        source_file_uris: set[str] | None = None,
        required_workspace_reads: set[str] | None = None,
        observed_workspace_paths: set[str] | None = None,
    ):
        self.source_ids = source_ids
        self.catalog_uris = catalog_uris
        self.file_catalog_uris = set(catalog_uris)
        self.file_catalog_uris.update(file_catalog_uris or ())
        self.target_uri = target_uri.rstrip("/")
        self.limits = limits
        self.require_workspace_files = require_workspace_files
        self.require_workspace_pages = require_workspace_pages
        self.workspace_baseline = (
            None
            if workspace_baseline is None
            else {_normalize_workspace_path(path) for path in workspace_baseline}
        )
        self.wiki_uri_resolver = wiki_uri_resolver
        self.exec_enabled = exec_enabled
        self.contract = contract or CompileContract()
        self.source_file_uris = source_file_uris or set()
        self.required_workspace_reads = (
            required_workspace_reads if required_workspace_reads is not None else set()
        )
        self.observed_workspace_paths = (
            observed_workspace_paths if observed_workspace_paths is not None else set()
        )
        self.bundle: CompileBundleDraft | None = None
        self.file_payloads: list[bytes | None] = []
        self.page_workspace_paths: dict[int, str] = {}
        self.workspace_files: set[str] | None = None
        self.skill_name: str | None = None

    @property
    def _is_skill_target(self) -> bool:
        return context_type_for_uri(self.target_uri) == "skill"

    @property
    def name(self) -> str:
        return "submit_compile_bundle"

    @property
    def description(self) -> str:
        artifact_writers = "write_file or exec" if self.exec_enabled else "write_file"
        workspace_notice = (
            f" Generate artifact files with {artifact_writers}, then reference them with "
            "workspace_path; do not inline file content."
            if self.require_workspace_files
            else ""
        )
        if self._is_skill_target:
            return (
                "Submit one complete OpenViking Skill package. Include every file under "
                "<skill-name>/ and include <skill-name>/SKILL.md."
                f"{workspace_notice}"
            )
        return (
            "Submit the final output only after every path and format explicitly required "
            "by the Skill is represented. Treat only actual Wiki content as Wiki pages and "
            f"preserve exact-path Skill outputs as artifact files.{workspace_notice}"
        )

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if "raw" in params:
            message = "use the tool schema directly; do not wrap the payload in a JSON string"
            if self.require_workspace_files:
                artifact_writers = "write_file or exec" if self.exec_enabled else "write_file"
                message += (
                    f"; generate artifact files with {artifact_writers} and submit them using "
                    "workspace_path instead of inline content"
                )
            return [message]
        return super().validate_params(params)

    @property
    def parameters(self) -> dict[str, Any]:
        schema = CompileBundleDraft.model_json_schema()
        required = schema.setdefault("required", [])
        if "files" not in required:
            required.append("files")
        definitions = schema.get("$defs", {})
        if self.require_workspace_files:
            file_schema = definitions.get("CompileFileDraft", {})
            file_properties = file_schema.get("properties", {})
            if isinstance(file_properties, dict):
                file_properties.pop("content", None)
            file_required = file_schema.setdefault("required", [])
            if "content" in file_required:
                file_required.remove("content")
            if "workspace_path" not in file_required:
                file_required.append("workspace_path")
        if self._is_skill_target:
            schema["properties"].pop("pages", None)
            schema["properties"].pop("links", None)
            required[:] = [field for field in required if field not in {"pages", "links"}]
            definitions.pop("WikiPageDraft", None)
            definitions.pop("WikiLink", None)
            file_schema = definitions.get("CompileFileDraft", {})
            file_schema.get("properties", {}).pop("update_uri", None)
            file_required = file_schema.setdefault("required", [])
            if "path" not in file_required:
                file_required.append("path")
            schema.pop("title", None)
            return schema
        file_schema = definitions.get("CompileFileDraft", {})
        file_schema["oneOf"] = [
            {"required": ["path"], "properties": {"path": {"type": "string"}}},
            {
                "required": ["update_uri"],
                "properties": {"update_uri": {"type": "string"}},
            },
        ]
        if self.require_workspace_pages:
            page_def = schema.get("$defs", {}).get("WikiPageDraft", {})
            page_properties = page_def.get("properties", {})
            if isinstance(page_properties, dict):
                page_properties.pop("body_markdown", None)
            page_required = page_def.setdefault("required", [])
            if "body_markdown" in page_required:
                page_required.remove("body_markdown")
            if "body_workspace_path" not in page_required:
                page_required.append("body_workspace_path")
        link_def = schema.get("$defs", {}).get("WikiLink", {})
        match_schema = link_def.get("properties", {}).get("match_text")
        if isinstance(match_schema, dict):
            match_schema["description"] = (
                "Exact anchor text that must appear in the source page draft body outside "
                "frontmatter, code, existing Markdown links, and Citations."
            )
        schema.pop("title", None)
        return schema

    async def execute(
        self,
        tool_context: ToolContext,
        pages: list[dict[str, Any]] | None = None,
        files: list[dict[str, Any]] | None = None,
        links: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        self.bundle = None
        self.file_payloads = []
        self.page_workspace_paths = {}
        self.workspace_files = None
        self.skill_name = None
        raw_links = links or []
        for index, link in enumerate(raw_links):
            if not isinstance(link, Mapping) or set(link) - _LINK_FIELDS:
                return f"Error: links[{index}] contains unknown fields."
        try:
            unread = sorted(self.required_workspace_reads - self.observed_workspace_paths)
            if unread:
                raise ValueError("required workspace files were not read: " + ", ".join(unread))
            bundle = CompileBundleDraft.model_validate(
                {"pages": pages or [], "files": files or [], "links": raw_links}
            )
            await self._validate_workspace_manifest(
                bundle,
                tool_context=tool_context,
            )
            self.page_workspace_paths = {
                page.page_id: _normalize_workspace_path(page.body_workspace_path)
                for page in bundle.pages
                if page.body_workspace_path is not None
            }
            bundle = await self._materialize_page_bodies(bundle, tool_context=tool_context)
            payloads = await self._validate_bundle(bundle, tool_context=tool_context)
            self._validate_contract(bundle)
        except (ValidationError, ValueError) as exc:
            kind = "Skill" if self._is_skill_target else "Compile"
            return f"Error: Invalid {kind} bundle: {exc}"
        self.bundle = bundle
        self.file_payloads = payloads
        if self._is_skill_target:
            return (
                f"Skill bundle accepted for '{self.skill_name}' with {len(bundle.files)} file(s)."
            )
        return (
            f"Compile bundle accepted with {len(bundle.pages)} page(s) and "
            f"{len(bundle.files)} file(s)."
        )

    async def _list_workspace_files(
        self,
        *,
        tool_context: ToolContext,
    ) -> set[str]:
        if tool_context.sandbox_manager is None:
            raise ValueError("task sandbox is unavailable")
        sandbox = await tool_context.sandbox_manager.get_sandbox(tool_context.session_key)
        files: set[str] = set()
        pending = [""]
        visited = 0
        while pending:
            directory = pending.pop()
            try:
                entries = await sandbox.list_dir(directory or ".")
            except Exception as exc:
                raise ValueError("task workspace could not be inspected") from exc
            for name, is_dir in entries:
                relative = _normalize_workspace_path(f"{directory}/{name}" if directory else name)
                visited += 1
                if visited > self.limits.target_inventory_entries:
                    raise ValueError("task workspace inventory limit exceeded")
                if _path_is_within(relative, COMPILE_STAGING_ROOT):
                    continue
                if name in {".git", "__pycache__"}:
                    continue
                if is_dir:
                    pending.append(relative)
                elif not relative.endswith((".pyc", ".pyo")):
                    files.add(relative)
        return files

    async def _validate_workspace_manifest(
        self,
        bundle: CompileBundleDraft,
        *,
        tool_context: ToolContext,
    ) -> None:
        if context_type_for_uri(self.target_uri) != "resource":
            return
        page_paths = {
            _normalize_workspace_path(page.body_workspace_path)
            for page in bundle.pages
            if page.body_workspace_path is not None
        }
        artifact_paths = {
            _normalize_workspace_path(file.workspace_path)
            for file in bundle.files
            if file.workspace_path is not None
        }
        errors: list[str] = []
        invalid_pages = sorted(
            path for path in page_paths if not _path_is_within(path, COMPILE_WIKI_PAGE_ROOT)
        )
        if self.require_workspace_pages and invalid_pages:
            errors.append(
                "Wiki page body workspace paths must be temporary files under "
                f"{COMPILE_WIKI_PAGE_ROOT}/, not Skill artifact paths: " + ", ".join(invalid_pages)
            )
        invalid_artifacts = sorted(
            path for path in artifact_paths if _path_is_within(path, COMPILE_STAGING_ROOT)
        )
        if invalid_artifacts:
            errors.append(
                "Skill artifact workspace paths must remain outside the Compile staging "
                "directory: " + ", ".join(invalid_artifacts)
            )

        if self.workspace_baseline is not None:
            current_files = await self._list_workspace_files(tool_context=tool_context)
            self.workspace_files = current_files
            generated_artifacts = current_files - self.workspace_baseline
            missing_artifacts = sorted(generated_artifacts - artifact_paths)
            if missing_artifacts:
                errors.append(
                    "generated Skill artifacts are missing from files; preserve their "
                    "required paths and submit them unchanged: " + ", ".join(missing_artifacts)
                )
        if errors:
            raise ValueError("; ".join(errors))

    async def _read_workspace_bytes(
        self,
        workspace_path: str,
        *,
        tool_context: ToolContext,
        label: str,
    ) -> bytes:
        try:
            relative = _normalize_workspace_path(workspace_path)
            if tool_context.sandbox_manager is None:
                raise ValueError("task sandbox is unavailable")
            sandbox = await tool_context.sandbox_manager.get_sandbox(tool_context.session_key)
            return await sandbox.read_file_bytes(relative)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{label} workspace path could not be read: {workspace_path}") from exc

    async def _materialize_page_bodies(
        self,
        bundle: CompileBundleDraft,
        *,
        tool_context: ToolContext,
    ) -> CompileBundleDraft:
        artifact_workspace_paths = {
            _normalize_workspace_path(file.workspace_path)
            for file in bundle.files
            if file.workspace_path is not None
        }
        pages = []
        for page in bundle.pages:
            if self.require_workspace_pages and page.body_markdown is not None:
                raise ValueError(
                    f"page {page.page_id} body must be generated with write_file and "
                    "submitted using body_workspace_path instead of inline Markdown"
                )
            if page.body_workspace_path is None:
                pages.append(page)
                continue
            workspace_path = _normalize_workspace_path(page.body_workspace_path)
            if workspace_path in artifact_workspace_paths:
                raise ValueError(
                    f"page {page.page_id} body must be a separate reader-oriented "
                    "workspace file, not an exact artifact file"
                )
            raw = await self._read_workspace_bytes(
                workspace_path,
                tool_context=tool_context,
                label=f"page {page.page_id} body",
            )
            try:
                body = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"page {page.page_id} body_workspace_path must contain UTF-8 Markdown"
                ) from exc
            updates: dict[str, Any] = {
                "body_markdown": body,
                "body_workspace_path": None,
            }
            if (
                page.update_uri is None
                and page.path_hint is None
                and _path_is_within(workspace_path, COMPILE_WIKI_PAGE_ROOT)
            ):
                updates["path_hint"] = workspace_path.removeprefix(COMPILE_WIKI_PAGE_ROOT + "/")
            pages.append(page.model_copy(update=updates))
        return bundle.model_copy(update={"pages": pages})

    async def _validate_bundle(
        self, bundle: CompileBundleDraft, *, tool_context: ToolContext
    ) -> list[bytes | None]:
        target_type = context_type_for_uri(self.target_uri)
        if len(bundle.pages) > self.limits.output_pages:
            raise ValueError("page limit exceeded")
        if len(bundle.files) > self.limits.output_files:
            raise ValueError("file limit exceeded")
        if not bundle.pages and bundle.links:
            raise ValueError("empty bundle must not contain links")
        if target_type == "skill" and (bundle.pages or bundle.links):
            raise ValueError("Skill targets only accept artifact files")
        if bundle.files and target_type not in {"resource", "skill"}:
            raise ValueError(
                "raw artifact files are only supported for Resource targets or exact "
                "Skill namespace targets; re-run ov compile with a supported target"
            )
        if self.require_workspace_files and any(file.content is not None for file in bundle.files):
            artifact_writers = "write_file or exec" if self.exec_enabled else "write_file"
            raise ValueError(
                f"artifact files must be generated with {artifact_writers} and submitted "
                "using workspace_path instead of inline content"
            )
        page_ids: set[int] = set()
        final_uris: set[str] = set()
        total_bytes = 0
        if self.source_file_uris:
            missing_citations = [
                page.page_id
                for page in bundle.pages
                if page.body_markdown is not None
                and not any(f"]({uri})" in page.body_markdown for uri in self.source_file_uris)
            ]
            if missing_citations:
                raise ValueError(
                    "pages "
                    + ", ".join(map(str, missing_citations))
                    + " must each link at least one exact source file URI"
                )
        for page in bundle.pages:
            if page.body_markdown is None:
                raise ValueError(f"page {page.page_id} body was not materialized")
            if page.page_id in page_ids:
                raise ValueError(f"duplicate page_id: {page.page_id}")
            page_ids.add(page.page_id)
            if not page.title.strip() or not page.page_type.strip() or not page.summary.strip():
                raise ValueError(f"page {page.page_id} has empty required fields")
            if "\n" in page.summary.strip() or "\r" in page.summary.strip():
                raise ValueError(f"page {page.page_id} summary must be one line")
            if page.body_markdown.lstrip().startswith("---"):
                raise ValueError(
                    f"page {page.page_id} must not include YAML frontmatter. If this is a "
                    "Skill-prescribed artifact, do not edit or strip its frontmatter; submit "
                    f"it through files and create a separate Wiki body under "
                    f"{COMPILE_WIKI_PAGE_ROOT}/"
                )
            if not page.source_ids or any(
                source_id not in self.source_ids for source_id in page.source_ids
            ):
                raise ValueError(f"page {page.page_id} has invalid source_ids")
            if page.update_uri:
                final_uri = page.update_uri.rstrip("/")
                if is_reserved_wiki_page_uri(final_uri):
                    raise ValueError(f"page {page.page_id} cannot update a reserved Wiki file")
                if not await self._is_wiki_uri(final_uri):
                    raise ValueError(
                        f"page {page.page_id} update_uri is not an existing OKF Wiki page"
                    )
                if page.path_hint:
                    raise ValueError(f"page {page.page_id} cannot rename an update")
                relative = relative_uri_path(self.target_uri, final_uri)
                if relative:
                    _validate_public_output_path(relative)
            else:
                hint = page.path_hint or wiki_page_path_from_title(page.title)
                relative = validate_relative_page_path(hint)
                _validate_public_output_path(relative)
                final_uri = safe_join_viking_uri(self.target_uri, relative).rstrip("/")
                if final_uri in self.file_catalog_uris:
                    raise ValueError(f"page {page.page_id} path exists; use its update_uri")
            if final_uri in final_uris:
                raise ValueError(f"duplicate final Wiki path: {final_uri}")
            final_uris.add(final_uri)
            total_bytes += len(page.body_markdown.encode("utf-8"))

        file_payloads: list[bytes | None] = []
        for index, file in enumerate(bundle.files):
            if target_type == "skill":
                if file.update_uri:
                    raise ValueError("Skill bundles require relative path entries, not update_uri")
                relative = validate_relative_file_path(file.path or "")
                _validate_public_output_path(relative)
                final_uri = safe_join_viking_uri(self.target_uri, relative).rstrip("/")
            elif file.update_uri:
                final_uri = validate_safe_viking_uri_path(file.update_uri).rstrip("/")
                relative = relative_uri_path(self.target_uri, final_uri)
                if relative:
                    _validate_public_output_path(relative)
                if is_reserved_output_file_uri(final_uri):
                    raise ValueError(f"file {index} cannot update a reserved file")
                if final_uri not in self.file_catalog_uris:
                    raise ValueError(f"file {index} update_uri is not in the catalog")
            else:
                relative = validate_relative_file_path(file.path or "")
                _validate_public_output_path(relative)
                final_uri = safe_join_viking_uri(self.target_uri, relative).rstrip("/")
                if final_uri in self.file_catalog_uris:
                    raise ValueError(f"file {index} path exists; use its update_uri")
            if final_uri in final_uris:
                raise ValueError(f"duplicate final output path: {final_uri}")
            final_uris.add(final_uri)

            if file.content is not None:
                payload = None
                content_bytes = file.content.encode("utf-8")
            else:
                payload = await self._read_workspace_bytes(
                    file.workspace_path or "",
                    tool_context=tool_context,
                    label=f"file {index}",
                )
                content_bytes = payload
            total_bytes += len(content_bytes)
            if total_bytes > self.limits.output_total_bytes:
                raise ValueError("draft content size limit exceeded")
            if target_type == "resource":
                page_type = validate_declared_okf_markdown(final_uri, content_bytes)
                existing_wiki = bool(file.update_uri and await self._is_wiki_uri(final_uri))
                if existing_wiki and page_type is None:
                    raise ValueError(
                        f"file {index} updates an existing Wiki page and must retain "
                        "valid OKF frontmatter with a non-empty type"
                    )
            file_payloads.append(payload)

        if total_bytes > self.limits.output_total_bytes:
            raise ValueError("draft content size limit exceeded")
        if target_type == "skill":
            self.skill_name = self._validate_skill_bundle(bundle, file_payloads)
        page_by_id = {page.page_id: page for page in bundle.pages}
        link_errors: list[str] = []
        for index, link in enumerate(bundle.links):
            prefix = f"links[{index}]"
            if link.f is None or link.t is None:
                link_errors.append(f"{prefix} endpoints must be non-null")
                continue
            if link.f == link.t:
                link_errors.append(f"{prefix} must not be a self-link")
                continue
            if link.f not in page_ids or link.t not in page_ids:
                link_errors.append(f"{prefix} endpoints must reference bundle pages")
                continue
            if not link.match_text:
                link_errors.append(f"{prefix} match_text is required")
                continue
            if "[" in link.match_text or "]" in link.match_text:
                link_errors.append(f"{prefix} match_text must not contain Markdown brackets")
                continue
            source_page = page_by_id[link.f]
            if (
                LinkRenderer._find_match_span(
                    source_page.body_markdown,
                    link.match_text,
                    LinkRenderer.protected_markdown_spans(source_page.body_markdown),
                )
                is None
            ):
                link_errors.append(
                    f"{prefix} from page {link.f} has non-linkable anchor "
                    f"{link.match_text!r}; remove the link or use exact unprotected "
                    "text from that page body"
                )
        if link_errors:
            raise ValueError(f"{len(link_errors)} invalid link(s): " + "; ".join(link_errors))
        return file_payloads

    def _validate_contract(self, bundle: CompileBundleDraft) -> None:
        if not bundle.pages and not bundle.files:
            raise ValueError("Compile bundle must contain at least one output")
        if self.contract.output == "wiki" and bundle.files:
            raise ValueError("Skill contract only permits Wiki page outputs")
        if self.contract.output == "files" and (bundle.pages or bundle.links):
            raise ValueError("Skill contract only permits file outputs")

        paths: set[str] = set()
        for page in bundle.pages:
            if page.update_uri:
                relative = relative_uri_path(self.target_uri, page.update_uri.rstrip("/"))
                if relative:
                    paths.add(relative)
            else:
                paths.add(
                    validate_relative_page_path(
                        page.path_hint or wiki_page_path_from_title(page.title)
                    )
                )
        for file in bundle.files:
            if file.update_uri:
                relative = relative_uri_path(self.target_uri, file.update_uri.rstrip("/"))
                if relative:
                    paths.add(relative)
            else:
                paths.add(validate_relative_file_path(file.path or ""))

        missing = [path for path in self.contract.required_paths if path not in paths]
        missing.extend(
            pattern
            for pattern in self.contract.required_globs
            if not any(fnmatchcase(path, pattern) for path in paths)
        )
        if missing:
            raise ValueError("missing contract output(s): " + ", ".join(missing))

    async def _is_wiki_uri(self, uri: str) -> bool:
        if uri in self.catalog_uris:
            return True
        if uri not in self.file_catalog_uris or self.wiki_uri_resolver is None:
            return False
        if await self.wiki_uri_resolver(uri):
            self.catalog_uris.add(uri)
            return True
        return False

    @staticmethod
    def _validate_skill_bundle(
        bundle: CompileBundleDraft, file_payloads: list[bytes | None]
    ) -> str:
        if not bundle.files:
            raise ValueError("Skill bundle must contain files")

        skill_names: set[str] = set()
        contents: dict[str, bytes] = {}
        for index, file in enumerate(bundle.files):
            relative = validate_relative_file_path(file.path or "")
            parts = relative.split("/")
            if len(parts) < 2:
                raise ValueError(f"file {index} must be under <skill-name>/, got: {relative}")
            skill_names.add(parts[0])
            payload = (
                file.content.encode("utf-8") if file.content is not None else file_payloads[index]
            )
            if payload is None:
                raise ValueError(f"file {index} has no materialized content")
            contents[relative] = payload

        if len(skill_names) != 1:
            raise ValueError("Skill bundle must contain exactly one top-level Skill directory")
        skill_name = next(iter(skill_names))
        skill_md_path = f"{skill_name}/SKILL.md"
        skill_md = contents.get(skill_md_path)
        if skill_md is None:
            raise ValueError(f"Skill bundle must include {skill_md_path}")
        try:
            skill_md_text = skill_md.decode("utf-8")
            parsed = SkillLoader.parse(skill_md_text, source_path=skill_md_path)
            parsed_name = validate_skill_name(parsed.get("name"))
        except (UnicodeDecodeError, ValueError, OpenVikingError, yaml.YAMLError) as exc:
            raise ValueError(str(exc)) from exc
        if parsed_name != skill_name:
            raise ValueError(f"Skill name '{parsed_name}' does not match directory '{skill_name}'")
        validation = validate_skill_format(
            skill_md_text,
            strict=True,
            skill_dir_name=skill_name,
            source_path=skill_md_path,
        )
        if not validation["valid"]:
            messages = [
                str(issue.get("message") or issue.get("rule") or "invalid Skill")
                for issue in validation["errors"]
            ]
            raise ValueError("; ".join(messages))
        return skill_name


SubmitWikiBundleTool = SubmitCompileBundleTool


__all__ = [
    "CompileReadTrackingTool",
    "CompileScopedTool",
    "SubmitCompileBundleTool",
    "SubmitCompileOutputsTool",
    "SubmitCompilePlanTool",
    "SubmitCompileValidationTool",
    "SubmitCompileWorkTool",
    "SubmitWikiBundleTool",
]
