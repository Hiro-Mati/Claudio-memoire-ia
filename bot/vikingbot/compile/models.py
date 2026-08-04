"""Models and stable limits for VikingBot compile tasks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from openviking.session.memory.dataclass import WikiLink
from vikingbot.channels.openapi_models import OpenVikingConnection

DEFAULT_COMPILE_REASON = (
    "Follow the loaded Skill's instructions to transform the provided source materials "
    "into the outputs required by the Skill."
)
COMPILE_STAGING_ROOT = "__compile_staging__"
COMPILE_WIKI_PAGE_ROOT = f"{COMPILE_STAGING_ROOT}/wiki_pages"
COMPILE_WORK_ROOT = f"{COMPILE_STAGING_ROOT}/work"
COMPILE_SOURCE_MANIFEST = f"{COMPILE_STAGING_ROOT}/source-manifest.json"
COMPILE_OUTPUT_PLAN = f"{COMPILE_STAGING_ROOT}/output-plan.json"
COMPILE_SKILL_CHECKLIST = f"{COMPILE_STAGING_ROOT}/skill-checklist.json"
COMPILE_PLAN_AUDIT_CONTEXT = f"{COMPILE_STAGING_ROOT}/plan-audit-context.json"
COMPILE_TASK_REASON_AUDIT_CONTEXT = f"{COMPILE_STAGING_ROOT}/task-reason-audit-context.json"
COMPILE_CONTRACT_FILE = "compile-contract.yaml"
OKF_VERSION = "0.1"
TERMINAL_STATUSES = frozenset({"completed", "failed"})


class CompileLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_roots: int = 16
    source_catalog_entries: int = 200
    source_inventory_entries: int = 5000
    skill_files: int = 128
    skill_file_bytes: int = 8 * 1024 * 1024
    skill_total_bytes: int = 32 * 1024 * 1024
    target_inventory_entries: int = 2000
    target_catalog_pages: int = 10
    initial_prompt_chars: int = 200_000
    tool_uri_count: int = 32
    tool_result_bytes: int = 1024 * 1024
    tool_total_result_bytes: int = 8 * 1024 * 1024
    output_pages: int = 64
    output_files: int = 64
    output_total_bytes: int = 4 * 1024 * 1024
    work_items: int = 64
    work_item_sources: int = 24
    work_item_concurrency: int = 3
    plan_audit_concurrency: int = 10
    plan_audit_enabled: bool = True
    max_critical_rules: int = 8
    task_reason_challenge_enabled: bool = False
    work_report_bytes: int = 64 * 1024
    repair_attempts: int = 2
    concurrent_tasks: int = 2
    accepted_tasks: int = 16
    accepted_tasks_per_principal: int = 4
    queue_wait_seconds: float = 5 * 60
    task_runtime_seconds: float = 4 * 60 * 60
    terminal_task_retention_seconds: float = 24 * 60 * 60
    terminal_task_records: int = 1000


class CompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: list[str] = Field(alias="from", min_length=1)
    to: str = Field(min_length=1)
    reason: str | None = None
    skill: str = Field(min_length=1)
    openviking_connection: OpenVikingConnection | None = None
    _principal_scope: str = PrivateAttr(default="local")


class SanitizedCompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: list[str] = Field(alias="from")
    to: str
    reason: str
    skill: str


class WikiPageDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: int
    title: str
    page_type: str
    summary: str
    body_markdown: str | None = Field(
        default=None,
        description=(
            "Inline Markdown body for an actual Wiki page. Link relevant known source "
            "URIs with ordinary Markdown links; never invent link targets."
        ),
    )
    body_workspace_path: str | None = Field(
        default=None,
        description=(
            f"Relative path under {COMPILE_WIKI_PAGE_ROOT}/ for a generated "
            "UTF-8 Markdown Wiki body. When path_hint is omitted, the path below this "
            "directory becomes the final Wiki page path."
        ),
    )
    source_ids: list[str] = Field(
        description="Identifiers of supplied source roots that support this Wiki page."
    )
    tags: list[str] = Field(default_factory=list)
    path_hint: str | None = Field(
        default=None,
        description=(
            "Optional final relative path for a new Wiki page. It overrides the path derived "
            "from body_workspace_path; use it only when the Skill requires a different path."
        ),
    )
    update_uri: str | None = None

    @model_validator(mode="after")
    def validate_body(self) -> "WikiPageDraft":
        if (self.body_markdown is None) == (self.body_workspace_path is None):
            raise ValueError("exactly one of body_markdown or body_workspace_path is required")
        return self


class CompileFileDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(
        default=None,
        description="Relative target path for a new file under the Compile target.",
    )
    update_uri: str | None = Field(
        default=None,
        description="Catalog URI of an existing target file to replace.",
    )
    content: str | None = Field(
        default=None,
        description="Exact UTF-8 text file content, including any required frontmatter.",
    )
    workspace_path: str | None = Field(
        default=None,
        description=(
            "Explicit relative path of a file already generated in the task workspace; "
            "its bytes are preserved exactly."
        ),
    )

    @model_validator(mode="after")
    def validate_shape(self) -> "CompileFileDraft":
        if (self.path is None) == (self.update_uri is None):
            raise ValueError("exactly one of path or update_uri is required")
        if (self.content is None) == (self.workspace_path is None):
            raise ValueError("exactly one of content or workspace_path is required")
        return self


class CompileRequirement(BaseModel):
    """One binding Skill requirement with verifiable source evidence."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    statement: str = Field(min_length=1, max_length=1000)
    evidence_path: str = Field(
        min_length=1,
        description=(
            "Exact path relative to the selected Skill root; use SKILL.md for the main file."
        ),
    )
    evidence_quote: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "A concise verbatim excerpt that proves the requirement; Markdown formatting and "
            "line wrapping are not significant."
        ),
    )
    enforcement: Literal["semantic", "required_path", "required_glob"] = "semantic"
    output_pattern: str | None = Field(
        default=None,
        description=(
            "Exact relative output path or glob for path enforcement; submit null for semantic "
            "requirements."
        ),
    )

    @model_validator(mode="after")
    def validate_enforcement(self) -> "CompileRequirement":
        if self.enforcement == "semantic" and self.output_pattern is None:
            return self
        if self.enforcement != "semantic" and self.output_pattern is not None:
            return self
        raise ValueError("output_pattern is required only for path and glob requirements")


class CompileContract(BaseModel):
    """Skill-owned completion criteria interpreted by the Compile runtime."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    output: Literal["wiki", "files", "mixed"] = "mixed"
    coverage: Literal["all_sources", "selected", "custom"] = "all_sources"
    required_paths: list[str] = Field(default_factory=list)
    required_globs: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    validation_rubrics: list[str] = Field(default_factory=list)
    validation_commands: list[str] = Field(default_factory=list)
    requirements: list[CompileRequirement] = Field(default_factory=list)

    @field_validator("required_paths", "required_globs")
    @classmethod
    def validate_patterns(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            pattern = value.strip().replace("\\", "/")
            if (
                not pattern
                or pattern.startswith("/")
                or pattern == COMPILE_STAGING_ROOT
                or pattern.startswith(COMPILE_STAGING_ROOT + "/")
                or pattern == ".compile"
                or pattern.startswith(".compile/")
                or any(part in {"", ".", ".."} for part in pattern.split("/"))
            ):
                raise ValueError(f"invalid Compile output pattern: {value}")
            if pattern not in normalized:
                normalized.append(pattern)
        return normalized

    @field_validator("capabilities", "validation_commands")
    @classmethod
    def validate_non_empty_values(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if len(normalized) != len(values):
            raise ValueError("Compile contract values must be non-empty and unique")
        return normalized

    @field_validator("validation_rubrics")
    @classmethod
    def validate_rubric_paths(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            path = value.strip().replace("\\", "/")
            if (
                not path
                or path.startswith("/")
                or any(part in {"", ".", ".."} for part in path.split("/"))
            ):
                raise ValueError(f"invalid validation rubric path: {value}")
            if path not in normalized:
                normalized.append(path)
        return normalized


class CompileWorkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_item_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    source_uris: list[str] = Field(min_length=1)


class CompileWorkReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_item_id: str
    processed_source_uris: list[str] = Field(min_length=1)
    report_workspace_path: str


class CompilePlannedPage(BaseModel):
    """One Wiki page in the trusted Compile output TODO list."""

    model_config = ConfigDict(extra="forbid")

    output_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    page_id: int = Field(gt=0)
    title: str = Field(min_length=1)
    page_type: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    source_uris: list[str] = Field(min_length=1)
    report_ids: list[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    path_hint: str | None = None
    update_uri: str | None = None

    @model_validator(mode="after")
    def validate_destination(self) -> "CompilePlannedPage":
        if (self.path_hint is None) == (self.update_uri is None):
            raise ValueError("exactly one of path_hint or update_uri is required")
        return self


class CompilePlannedFile(BaseModel):
    """One exact artifact in the trusted Compile output TODO list."""

    model_config = ConfigDict(extra="forbid")

    output_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1)
    report_ids: list[str] = Field(default_factory=list)
    path: str | None = None
    update_uri: str | None = None

    @model_validator(mode="after")
    def validate_destination(self) -> "CompilePlannedFile":
        if (self.path is None) == (self.update_uri is None):
            raise ValueError("exactly one of path or update_uri is required")
        return self


class CompileCoverageDecision(BaseModel):
    """Final disposition of one complete source coverage unit."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str
    status: Literal["covered", "duplicate", "unsupported"]
    output_ids: list[str] = Field(default_factory=list)
    duplicate_of: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "CompileCoverageDecision":
        if self.status == "covered":
            if not self.output_ids:
                raise ValueError("covered units require at least one output_id")
            if self.duplicate_of is not None:
                raise ValueError("covered units must not declare duplicate_of")
        elif self.status == "duplicate":
            if not self.duplicate_of:
                raise ValueError("duplicate units require duplicate_of")
            if self.output_ids:
                raise ValueError("duplicate units must not declare output_ids")
        else:
            if not self.reason:
                raise ValueError("unsupported units require a reason")
            if self.output_ids or self.duplicate_of is not None:
                raise ValueError("unsupported units must not map to outputs or duplicates")
        return self


class CompileMaterializationGroup(BaseModel):
    """A cohesive output batch and its dependencies in the Compile DAG."""

    model_config = ConfigDict(extra="forbid")

    group_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    output_ids: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)


class CompileOutputPlan(BaseModel):
    """Structured Compile TODO list produced before any final output is written."""

    model_config = ConfigDict(extra="forbid")

    pages: list[CompilePlannedPage] = Field(default_factory=list)
    files: list[CompilePlannedFile] = Field(default_factory=list)
    links: list[WikiLink] = Field(default_factory=list)
    coverage: list[CompileCoverageDecision] = Field(default_factory=list)
    groups: list[CompileMaterializationGroup] = Field(
        min_length=1,
        description=(
            "A complete partition of outputs into cohesive generation groups. Put outputs that "
            "must share identifiers, facts, or formatting in one group; use depends_on when a "
            "group must read already generated outputs."
        ),
    )

    @model_validator(mode="after")
    def validate_outputs(self) -> "CompileOutputPlan":
        if not self.pages and not self.files:
            raise ValueError("Compile output plan must contain at least one output")
        output_ids = [output.output_id for output in self.pages]
        output_ids.extend(output.output_id for output in self.files)
        grouped_ids = [output_id for group in self.groups for output_id in group.output_ids]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise ValueError("materialization group output_ids must be unique")
        if set(grouped_ids) != set(output_ids):
            raise ValueError("materialization groups must contain every output_id exactly once")

        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("materialization group_id values must be unique")
        known_groups = set(group_ids)
        dependencies = {group.group_id: set(group.depends_on) for group in self.groups}
        for group_id, depends_on in dependencies.items():
            if group_id in depends_on:
                raise ValueError(f"materialization group {group_id} cannot depend on itself")
            unknown = depends_on - known_groups
            if unknown:
                raise ValueError(
                    f"materialization group {group_id} has unknown dependencies: "
                    + ", ".join(sorted(unknown))
                )

        remaining = dict(dependencies)
        resolved: set[str] = set()
        while remaining:
            ready = {group_id for group_id, deps in remaining.items() if deps <= resolved}
            if not ready:
                raise ValueError("materialization group dependencies must be acyclic")
            resolved.update(ready)
            for group_id in ready:
                remaining.pop(group_id)
        return self


class CompileOutputReceipt(BaseModel):
    """Proof that one bounded batch of planned outputs was materialized."""

    model_config = ConfigDict(extra="forbid")

    output_ids: list[str] = Field(min_length=1)


class CompileSkillRule(BaseModel):
    """One source-grounded, binding rule extracted from the selected Skill."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    statement: str = Field(min_length=1, max_length=1000)
    evidence_path: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1, max_length=1000)
    phase: Literal["plan", "candidate", "both"]
    critical: bool = False


class CompileSkillChecklist(BaseModel):
    """Ephemeral audit checklist grounded only in selected Skill text."""

    model_config = ConfigDict(extra="forbid")

    rules: list[CompileSkillRule] = Field(default_factory=list)


class CompileSkillRuleSelection(BaseModel):
    """Model-selected rule and the trusted Skill evidence block that supports it."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    statement: str = Field(min_length=1, max_length=1000)
    evidence_id: str = Field(pattern=r"^ev[0-9]{4,}$")
    phase: Literal["plan", "candidate", "both"]
    critical: bool = False


class CompileSkillChecklistSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: list[CompileSkillRuleSelection] = Field(default_factory=list)


class CompileRuleCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    passed: bool
    evidence: str = Field(min_length=1, max_length=2000)


class CompileValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    path: str | None = None


class CompileValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    issues: list[CompileValidationIssue] = Field(default_factory=list)
    checked_requirement_ids: list[str] = Field(default_factory=list)
    rule_checks: list[CompileRuleCheck] = Field(default_factory=list)


class CompileBundleDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pages: list[WikiPageDraft] = Field(
        description=(
            "Actual Wiki pages only; navigation files such as index.md and log.md are "
            "artifacts and must not be placed here."
        )
    )
    files: list[CompileFileDraft] = Field(
        default_factory=list,
        description=(
            "Skill-prescribed exact-path artifacts, including Markdown, YAML, JSON, "
            "and binary files; preserve every required path and format."
        ),
    )
    links: list[WikiLink] = Field(
        default_factory=list,
        description=(
            "Useful non-self relationships between generated Wiki pages only; omit "
            "them when no valid related page exists."
        ),
    )


# Compatibility for callers that adopted the original Wiki-specific name.
WikiBundleDraft = CompileBundleDraft


class CompileErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class CompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: list[str] = Field(alias="from")
    to: str
    skill: str
    okf_version: str = OKF_VERSION
    created: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    page_count: int = 0
    file_count: int = 0
    link_count: int = 0
    source_count: int = 0
    processed_source_count: int = 0
    validation_attempts: int = 0
    contract_source: Literal["declared", "derived", "inferred"] = "inferred"
    warnings: list[str] = Field(default_factory=list)


class CompileTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    principal_scope: str
    sanitized_request: SanitizedCompileRequest
    status: Literal["accepted", "running", "committing", "completed", "failed"]
    stage: str
    created_at: str
    updated_at: str
    result: CompileResult | None = None
    error: CompileErrorInfo | None = None

    def public_dict(self) -> dict[str, Any]:
        data = self.model_dump(exclude={"principal_scope", "sanitized_request"}, exclude_none=True)
        if self.result is not None:
            data["result"] = self.result.model_dump(by_alias=True)
        return data


class CompileAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: Literal["accepted"] = "accepted"
    to: str


class CompileFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, stage: str):
        super().__init__(message)
        self.code = code
        self.stage = stage


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "CompileAccepted",
    "CompileBundleDraft",
    "CompileContract",
    "CompileCoverageDecision",
    "CompileOutputPlan",
    "CompileOutputReceipt",
    "CompilePlannedFile",
    "CompilePlannedPage",
    "CompileRequirement",
    "CompileValidationIssue",
    "CompileValidationReport",
    "CompileWorkItem",
    "CompileWorkReceipt",
    "CompileErrorInfo",
    "CompileFileDraft",
    "CompileFailure",
    "CompileLimits",
    "CompileRequest",
    "CompileResult",
    "CompileTask",
    "DEFAULT_COMPILE_REASON",
    "COMPILE_CONTRACT_FILE",
    "COMPILE_OUTPUT_PLAN",
    "COMPILE_SOURCE_MANIFEST",
    "COMPILE_WORK_ROOT",
    "OKF_VERSION",
    "SanitizedCompileRequest",
    "TERMINAL_STATUSES",
    "WikiBundleDraft",
    "WikiPageDraft",
    "utc_now",
]
