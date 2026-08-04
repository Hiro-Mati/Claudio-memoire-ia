import asyncio
import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from vikingbot.agent.loop import AgentLoop
from vikingbot.agent.tools.base import Tool, ToolContext
from vikingbot.agent.tools.compile import (
    CompileReadTrackingTool,
    CompileScopedTool,
    ReadWorkspaceFilesTool,
    SubmitCompileBundleTool,
    SubmitCompileChecklistTool,
    SubmitCompileOutputsTool,
    SubmitCompilePlanTool,
    SubmitCompileValidationTool,
    SubmitCompileWorkTool,
)
from vikingbot.agent.tools.registry import ToolRegistry
from vikingbot.compile.models import (
    DEFAULT_COMPILE_REASON,
    CompileBundleDraft,
    CompileContract,
    CompileFailure,
    CompileLimits,
    CompileOutputPlan,
    CompileRequest,
    CompileSkillChecklist,
    CompileSkillRule,
    CompileTask,
    CompileValidationReport,
    CompileWorkItem,
    CompileWorkReceipt,
    SanitizedCompileRequest,
    utc_now,
)
from vikingbot.compile.renderer import WikiRenderer, content_hash, wiki_page_path_from_title
from vikingbot.compile.service import BotCompileService, CompileCapabilities
from vikingbot.compile.store import CompileTaskStore
from vikingbot.config.schema import DirectBackendConfig, SandboxBackend, SessionKey

from openviking.core.skill_loader import SkillLoader
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import OpenVikingError


def _page(page_id: int, title: str, **overrides):
    value = {
        "page_id": page_id,
        "title": title,
        "page_type": "concept",
        "summary": f"Summary for {title}",
        "body_markdown": f"Body for {title}",
        "source_ids": ["src_1"],
        "path_hint": f"{title.lower()}.md",
    }
    value.update(overrides)
    return value


def test_default_compile_reason_does_not_create_redundant_reason_audit_rules():
    from vikingbot.compile.service import _task_reason_audit_rules

    assert _task_reason_audit_rules(DEFAULT_COMPILE_REASON) == []


def test_skill_loader_distinguishes_missing_and_explicit_empty_allowed_tools():
    missing = SkillLoader.parse("---\nname: a\ndescription: A\n---\nDo it")
    empty = SkillLoader.parse("---\nname: a\ndescription: A\nallowed-tools: []\n---\nDo it")
    assert missing["allowed_tools_declared"] is False
    assert empty["allowed_tools_declared"] is True
    assert "allowed-tools: ''" in SkillLoader.to_skill_md(empty)


def test_skill_loader_accepts_standard_and_legacy_allowed_tools_forms():
    standard = SkillLoader.parse(
        "---\nname: a\ndescription: A\nallowed-tools: Read Write Bash(git:*)\n---\nDo it"
    )
    ara = SkillLoader.parse(
        "---\nname: a\ndescription: A\n"
        "allowed-tools: Read, Write, Bash(python *|git clone *), Glob\n---\nDo it"
    )
    legacy = SkillLoader.parse(
        "---\nname: a\ndescription: A\nallowed-tools: [Read, Write]\n---\nDo it"
    )

    assert standard["allowed_tools"] == ["Read", "Write", "Bash(git:*)"]
    assert ara["allowed_tools"] == [
        "Read",
        "Write",
        "Bash(python *|git clone *)",
        "Glob",
    ]
    assert legacy["allowed_tools"] == ["Read", "Write"]


def test_skill_loader_rejects_invalid_allowed_tools():
    with pytest.raises(ValueError, match="space-separated string or an array of strings"):
        SkillLoader.parse("---\nname: a\ndescription: A\nallowed-tools: [Read, 3]\n---\nDo it")
    with pytest.raises(ValueError, match="unbalanced parentheses"):
        SkillLoader.parse(
            "---\nname: a\ndescription: A\nallowed-tools: Read Bash(git:*\n---\nDo it"
        )


def test_compile_bundle_schema_distinguishes_wiki_pages_and_artifact_files():
    schema = CompileBundleDraft.model_json_schema()
    properties = schema["properties"]
    page_properties = schema["$defs"]["WikiPageDraft"]["properties"]

    assert "Actual Wiki pages only" in properties["pages"]["description"]
    assert "Markdown, YAML, JSON" in properties["files"]["description"]
    assert "generated Wiki pages only" in properties["links"]["description"]
    assert "known source URIs" in page_properties["body_markdown"]["description"]
    assert (
        "generated UTF-8 Markdown Wiki body"
        in page_properties["body_workspace_path"]["description"]
    )
    assert (
        "__compile_staging__/wiki_pages/" in page_properties["body_workspace_path"]["description"]
    )
    assert "overrides the path derived" in page_properties["path_hint"]["description"]
    assert (
        "becomes the final Wiki page path" in page_properties["body_workspace_path"]["description"]
    )
    assert "supplied source roots" in page_properties["source_ids"]["description"]
    assert "preserve every required path and format" in properties["files"]["description"]


def test_compile_limit_defaults_match_the_resource_envelope():
    limits = CompileLimits()

    assert limits.concurrent_tasks == 2
    assert limits.target_inventory_entries == 2000
    assert limits.target_catalog_pages == 10
    assert limits.task_runtime_seconds == 4 * 60 * 60
    assert DirectBackendConfig().allow_compile_exec is False


def test_compile_contract_loads_nested_portable_schema(tmp_path: Path):
    skill_dir = tmp_path / "compiler"
    skill_dir.mkdir()
    (skill_dir / "rubric.md").write_text("# Validate", encoding="utf-8")
    (skill_dir / "compile-contract.yaml").write_text(
        """version: 1
output:
  kind: files
coverage:
  mode: custom
required_outputs:
  - path: PAPER.md
  - glob: evidence/figures/*.md
capabilities:
  required: [exec]
validators:
  - type: rubric
    path: rubric.md
  - type: command
    run: python3 skills/compiler/validate.py
""",
        encoding="utf-8",
    )

    contract, source = BotCompileService._load_compile_contract(skill_dir)

    assert source == "declared"
    assert contract == CompileContract(
        output="files",
        coverage="custom",
        required_paths=["PAPER.md"],
        required_globs=["evidence/figures/*.md"],
        capabilities=["exec"],
        validation_rubrics=["rubric.md"],
        validation_commands=["python3 skills/compiler/validate.py"],
    )


def test_compile_contract_requirement_path_enforcement_is_deterministic(tmp_path: Path):
    skill_dir = tmp_path / "compiler"
    skill_dir.mkdir()
    (skill_dir / "compile-contract.yaml").write_text(
        """requirements:
  - requirement_id: paper
    statement: Produce the paper index.
    evidence_path: SKILL.md
    evidence_quote: Produce the paper index.
    enforcement: required_path
    output_pattern: PAPER.md
  - requirement_id: evidence
    statement: Preserve evidence files.
    evidence_path: SKILL.md
    evidence_quote: Preserve evidence files.
    enforcement: required_glob
    output_pattern: evidence/*.md
""",
        encoding="utf-8",
    )

    contract, _source = BotCompileService._load_compile_contract(skill_dir)

    assert contract.required_paths == ["PAPER.md"]
    assert contract.required_globs == ["evidence/*.md"]


def test_compile_contract_rejects_internal_output_paths():
    with pytest.raises(ValueError, match="invalid Compile output pattern"):
        CompileContract(required_paths=["__compile_staging__/candidate.json"])
    with pytest.raises(ValueError, match="invalid Compile output pattern"):
        CompileContract(required_globs=[".compile/*.json"])


def test_compile_contract_rejects_unknown_nested_fields(tmp_path: Path):
    skill_dir = tmp_path / "compiler"
    skill_dir.mkdir()
    (skill_dir / "compile-contract.yaml").write_text(
        "output:\n  kind: files\n  format: markdown\n",
        encoding="utf-8",
    )

    with pytest.raises(CompileFailure, match="output must contain exactly one kind field"):
        BotCompileService._load_compile_contract(skill_dir)


def test_compile_contract_capabilities_fail_before_generation():
    with pytest.raises(CompileFailure, match="Missing Compile capabilities: exec"):
        BotCompileService._check_contract_capabilities(
            CompileContract(capabilities=["exec"]),
            CompileCapabilities(exec_enabled=False),
        )


@pytest.mark.parametrize(
    ("target_type", "output"),
    [("resource", "mixed"), ("memory", "wiki"), ("skill", "files")],
)
def test_compile_contract_is_deterministically_inferred_from_target_type(target_type, output):
    contract = BotCompileService._inferred_compile_contract(target_type=target_type)

    assert contract.output == output
    assert contract.coverage == "all_sources"
    assert contract.validation_rubrics == []


@pytest.mark.asyncio
async def test_compile_output_plan_is_a_complete_source_mapped_todo_list():
    observed = {"__compile_staging__/source-manifest.json", "work/report.md"}
    tool = SubmitCompilePlanTool(
        source_ids={"src_1"},
        source_roots={"src_1": "viking://resources/source"},
        source_file_uris={"viking://resources/source/a.md"},
        coverage_unit_ids={"unit-001"},
        report_ids={"item-001"},
        target_uri="viking://resources/wiki",
        catalog_uris=set(),
        file_catalog_uris=set(),
        contract=CompileContract(output="mixed", required_paths=["index.md"]),
        limits=CompileLimits(),
        required_workspace_reads=observed,
        observed_workspace_paths=observed,
    )

    accepted = await tool.execute(
        ToolContext(),
        pages=[
            {
                "output_id": "page-001",
                "page_id": 1,
                "title": "Reading Skills",
                "page_type": "concept",
                "summary": "Reusable reading skills.",
                "source_ids": ["src_1"],
                "source_uris": ["viking://resources/source/a.md"],
                "report_ids": ["item-001"],
                "path_hint": "concepts/reading-skills.md",
            }
        ],
        files=[
            {
                "output_id": "index",
                "description": "Navigation index",
                "report_ids": ["item-001"],
                "path": "index.md",
            }
        ],
        coverage=[
            {
                "unit_id": "unit-001",
                "status": "covered",
                "output_ids": ["page-001"],
            }
        ],
        groups=[
            {
                "group_id": "wiki",
                "output_ids": ["page-001", "index"],
            }
        ],
    )

    assert accepted == "Compile output plan accepted with 1 page(s) and 1 file(s)."
    assert tool.structured_result is not None
    assert tool.structured_result.pages[0].source_uris == ["viking://resources/source/a.md"]

    invalid_source = await tool.execute(
        ToolContext(),
        pages=[
            {
                "output_id": "page-001",
                "page_id": 1,
                "title": "Reading Skills",
                "page_type": "concept",
                "summary": "Reusable reading skills.",
                "source_ids": ["src_1"],
                "source_uris": ["viking://resources/source"],
                "report_ids": ["item-001"],
                "path_hint": "concepts/reading-skills.md",
            }
        ],
        files=[
            {
                "output_id": "index",
                "description": "Navigation index",
                "report_ids": ["item-001"],
                "path": "index.md",
            }
        ],
        coverage=[
            {
                "unit_id": "unit-001",
                "status": "covered",
                "output_ids": ["page-001"],
            }
        ],
        groups=[{"group_id": "wiki", "output_ids": ["page-001", "index"]}],
    )
    assert "absent from the exact leaf-source manifest" in invalid_source
    assert "small representative set" in invalid_source
    assert "viking://resources/source" in invalid_source

    invalid_report = await tool.execute(
        ToolContext(),
        pages=[
            {
                "output_id": "page-001",
                "page_id": 1,
                "title": "Reading Skills",
                "page_type": "concept",
                "summary": "Reusable reading skills.",
                "source_ids": ["src_1"],
                "source_uris": ["viking://resources/source/a.md"],
                "report_ids": ["item-unknown"],
                "path_hint": "concepts/reading-skills.md",
            }
        ],
        files=[
            {
                "output_id": "index",
                "description": "Navigation index",
                "report_ids": ["item-001"],
                "path": "index.md",
            }
        ],
        coverage=[
            {
                "unit_id": "unit-001",
                "status": "covered",
                "output_ids": ["page-001"],
            }
        ],
        groups=[{"group_id": "wiki", "output_ids": ["page-001", "index"]}],
    )
    assert "page 1 has unknown report_ids: item-unknown" in invalid_report

    rejected = await tool.execute(ToolContext(), files=[{"path": "index.md"}])
    assert "Compile output plan must contain at least one output" not in rejected
    assert "validation error" in rejected


@pytest.mark.asyncio
async def test_compile_plan_decodes_complete_raw_json_object():
    tool = SubmitCompilePlanTool(
        source_ids=set(),
        source_file_uris=set(),
        coverage_unit_ids=set(),
        report_ids=set(),
        target_uri="viking://resources/wiki",
        catalog_uris=set(),
        file_catalog_uris=set(),
        contract=CompileContract(),
        limits=CompileLimits(),
        required_workspace_reads=set(),
        observed_workspace_paths=set(),
    )
    params = {
        "raw": json.dumps(
            {
                "files": [
                    {
                        "output_id": "index",
                        "description": "Navigation index",
                        "path": "index.md",
                    }
                ],
                "groups": [{"group_id": "navigation", "output_ids": ["index"]}],
            }
        )
    }

    assert tool.validate_params(params) == []
    assert "raw" not in params
    result = await tool.execute(ToolContext(), **params)

    assert result == "Compile output plan accepted with 0 page(s) and 1 file(s)."


@pytest.mark.asyncio
async def test_compile_plan_resolves_existing_wiki_outside_relevance_catalog():
    source_uri = "viking://resources/source/a.md"
    existing_uri = "viking://resources/wiki/existing.md"
    catalog_uris: set[str] = set()
    resolved: list[str] = []

    async def resolve(uri: str) -> bool:
        resolved.append(uri)
        return uri == existing_uri

    tool = SubmitCompilePlanTool(
        source_ids={"src_1"},
        source_roots={"src_1": "viking://resources/source"},
        source_file_uris={source_uri},
        coverage_unit_ids={"unit-001"},
        report_ids={"item-001"},
        target_uri="viking://resources/wiki",
        catalog_uris=catalog_uris,
        file_catalog_uris={existing_uri},
        contract=CompileContract(output="wiki"),
        limits=CompileLimits(),
        required_workspace_reads=set(),
        observed_workspace_paths=set(),
        wiki_uri_resolver=resolve,
    )

    accepted = await tool.execute(
        ToolContext(),
        pages=[
            {
                "output_id": "existing",
                "page_id": 1,
                "title": "Existing",
                "page_type": "concept",
                "summary": "Update existing knowledge.",
                "source_ids": ["src_1"],
                "source_uris": [source_uri],
                "report_ids": ["item-001"],
                "update_uri": existing_uri,
            }
        ],
        coverage=[{"unit_id": "unit-001", "status": "covered", "output_ids": ["existing"]}],
        groups=[{"group_id": "existing", "output_ids": ["existing"]}],
    )

    assert accepted.startswith("Compile output plan accepted")
    assert resolved == [existing_uri]
    assert existing_uri in catalog_uris


@pytest.mark.asyncio
async def test_compile_plan_rejects_unmaterializable_skill_shapes():
    tool = SubmitCompilePlanTool(
        source_ids=set(),
        source_file_uris=set(),
        coverage_unit_ids=set(),
        report_ids=set(),
        target_uri="viking://agent/skills",
        catalog_uris=set(),
        file_catalog_uris={"viking://agent/skills/weekly-report/SKILL.md"},
        contract=CompileContract(output="files"),
        limits=CompileLimits(),
        required_workspace_reads=set(),
        observed_workspace_paths=set(),
    )

    root_file = await tool.execute(
        ToolContext(),
        files=[{"output_id": "skill", "description": "Skill", "path": "SKILL.md"}],
        groups=[{"group_id": "skill", "output_ids": ["skill"]}],
    )
    assert "one top-level <skill-name>/ directory" in root_file

    update = await tool.execute(
        ToolContext(),
        files=[
            {
                "output_id": "skill",
                "description": "Skill",
                "update_uri": "viking://agent/skills/weekly-report/SKILL.md",
            }
        ],
        groups=[{"group_id": "skill", "output_ids": ["skill"]}],
    )
    assert "relative path entries, not update_uri" in update


@pytest.mark.asyncio
async def test_compile_plan_executes_through_the_shared_tool_registry():
    tool = SubmitCompilePlanTool(
        source_ids=set(),
        source_file_uris=set(),
        coverage_unit_ids=set(),
        report_ids=set(),
        target_uri="viking://resources/output",
        catalog_uris=set(),
        file_catalog_uris=set(),
        contract=CompileContract(output="mixed"),
        limits=CompileLimits(),
        required_workspace_reads=set(),
        observed_workspace_paths=set(),
    )
    registry = ToolRegistry()
    registry.register(tool)

    result = await registry.execute(
        tool.name,
        {
            "files": [{"output_id": "index", "description": "Index", "path": "index.md"}],
            "groups": [{"group_id": "navigation", "output_ids": ["index"]}],
        },
        SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
    )

    assert result == "Compile output plan accepted with 0 page(s) and 1 file(s)."


def test_compile_output_groups_form_a_complete_acyclic_partition():
    file = {"output_id": "a", "description": "A", "path": "a.md"}
    with pytest.raises(ValidationError, match="every output_id exactly once"):
        CompileOutputPlan.model_validate(
            {
                "files": [file],
                "groups": [{"group_id": "empty", "output_ids": ["unknown"]}],
            }
        )

    with pytest.raises(ValidationError, match="must be acyclic"):
        CompileOutputPlan.model_validate(
            {
                "files": [
                    file,
                    {"output_id": "b", "description": "B", "path": "b.md"},
                ],
                "groups": [
                    {"group_id": "one", "output_ids": ["a"], "depends_on": ["two"]},
                    {"group_id": "two", "output_ids": ["b"], "depends_on": ["one"]},
                ],
            }
        )


def test_candidate_audit_reads_every_generated_text_output():
    plan = CompileOutputPlan.model_validate(
        {
            "pages": [
                {
                    "output_id": "page",
                    "page_id": 1,
                    "title": "Page",
                    "page_type": "concept",
                    "summary": "Summary",
                    "source_ids": ["source"],
                    "source_uris": ["viking://resources/source/a.md"],
                    "report_ids": ["report"],
                    "path_hint": "page.md",
                }
            ],
            "files": [
                {"output_id": "text", "description": "Text", "path": "data.json"},
                {"output_id": "image", "description": "Image", "path": "figure.png"},
            ],
            "groups": [{"group_id": "all", "output_ids": ["page", "text", "image"]}],
        }
    )
    service = object.__new__(BotCompileService)

    paths = service._text_output_workspace_paths(
        plan=plan,
        file_payloads=[b'{"ok": true}', b"\x89PNG\r\n\x1a\n\xff"],
        target_uri="viking://resources/wiki",
    )

    assert paths == {"__compile_staging__/wiki_pages/page.md", "data.json"}


@pytest.mark.asyncio
async def test_materialization_groups_run_in_dependency_order_and_read_text_dependencies():
    files = {"work/report.md": b"evidence"}
    calls: list[tuple[list[str], set[str]]] = []

    class Sandbox:
        async def read_file_bytes(self, path):
            return files[path]

    class Manager:
        async def get_sandbox(self, session_key):
            del session_key
            return Sandbox()

    manager = Manager()

    class Loop:
        sandbox_manager = manager

        async def run_structured_task(self, *, tool_registry, session_key, **kwargs):
            del kwargs
            submit = tool_registry.get("submit_compile_outputs")
            assert isinstance(submit, SubmitCompileOutputsTool)
            output_ids = list(submit.expected_paths)
            calls.append((output_ids, set(submit.required_workspace_reads)))
            for output_id, (path, _require_utf8) in submit.expected_paths.items():
                files[path] = f"generated {output_id}".encode()
            submit.observed_workspace_paths.update(submit.required_workspace_reads)
            result = await submit.execute(
                ToolContext(session_key=session_key, sandbox_manager=manager),
                output_ids=output_ids,
            )
            assert result.startswith("Compile output batch accepted")
            return submit.receipt, [], {}, 1

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()

    def build_registry(_request_loop, *, submission_tool, **kwargs):
        del kwargs
        registry = ToolRegistry()
        registry.register(submission_tool)
        return registry, set()

    async def set_state(*args, **kwargs):
        del args, kwargs

    service._build_compile_registry = build_registry
    service._set_state = set_state
    plan = CompileOutputPlan.model_validate(
        {
            "files": [
                {
                    "output_id": "claims",
                    "description": "Claims",
                    "report_ids": ["report"],
                    "path": "logic/claims.md",
                },
                {
                    "output_id": "index",
                    "description": "Index",
                    "report_ids": ["report"],
                    "path": "PAPER.md",
                },
            ],
            "groups": [
                {"group_id": "logic", "output_ids": ["claims"]},
                {
                    "group_id": "navigation",
                    "output_ids": ["index"],
                    "depends_on": ["logic"],
                },
            ],
        }
    )

    await service._run_output_batches(
        task_id="cmp",
        request_loop=Loop(),
        request=SanitizedCompileRequest.model_validate(
            {
                "from": ["viking://resources/source"],
                "to": "viking://resources/output",
                "skill": "viking://agent/skills/compiler",
                "reason": "Compile",
            }
        ),
        skill_name="compiler",
        skill_content="Compile files.",
        skill_references={},
        checklist=CompileSkillChecklist(),
        plan=plan,
        receipts=[
            CompileWorkReceipt(
                work_item_id="report",
                processed_source_uris=["viking://resources/source/a.md"],
                report_workspace_path="work/report.md",
            )
        ],
        capabilities=CompileCapabilities(exec_enabled=False),
        validation_report=None,
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        connection={},
        registry_kwargs={},
    )

    assert calls == [
        (["claims"], {"work/report.md"}),
        (["index"], {"work/report.md", "logic/claims.md"}),
    ]


@pytest.mark.asyncio
async def test_materialization_prompt_prefills_required_source_citations():
    source_uri = "viking://resources/source/a.md"
    files = {"work/report.md": b"evidence"}
    prompts: list[str] = []

    class Sandbox:
        async def read_file_bytes(self, path):
            return files[path]

    class Manager:
        async def get_sandbox(self, session_key):
            del session_key
            return Sandbox()

    manager = Manager()

    class Loop:
        sandbox_manager = manager

        async def run_structured_task(self, *, tool_registry, session_key, user_prompt, **kwargs):
            del kwargs
            prompts.append(user_prompt)
            submit = tool_registry.get("submit_compile_outputs")
            output_ids = list(submit.expected_paths)
            for output_id, (path, require_utf8) in submit.expected_paths.items():
                body = f"# {output_id}\n"
                if require_utf8:
                    body += f"\n[source]({source_uri})"
                files[path] = body.encode()
            submit.observed_workspace_paths.update(submit.required_workspace_reads)
            result = await submit.execute(
                ToolContext(session_key=session_key, sandbox_manager=manager),
                output_ids=output_ids,
            )
            assert result.startswith("Compile output batch accepted")
            return submit.receipt, [], {}, 1

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()

    def build_registry(_request_loop, *, submission_tool, **kwargs):
        del kwargs
        registry = ToolRegistry()
        registry.register(submission_tool)
        return registry, set()

    async def set_state(*args, **kwargs):
        del args, kwargs

    service._build_compile_registry = build_registry
    service._set_state = set_state
    plan = CompileOutputPlan.model_validate(
        {
            "pages": [
                {
                    "output_id": "page-001",
                    "page_id": 1,
                    "title": "Reading",
                    "page_type": "concept",
                    "summary": "Reading concept",
                    "source_ids": ["src_1"],
                    "path_hint": "concepts/reading",
                    "report_ids": ["report"],
                    "source_uris": [source_uri],
                }
            ],
            "groups": [{"group_id": "core", "output_ids": ["page-001"]}],
        }
    )

    await service._run_output_batches(
        task_id="cmp",
        request_loop=Loop(),
        request=SanitizedCompileRequest.model_validate(
            {
                "from": ["viking://resources/source"],
                "to": "viking://resources/output",
                "skill": "viking://agent/skills/compiler",
                "reason": "Compile",
            }
        ),
        skill_name="compiler",
        skill_content="Compile files.",
        skill_references={},
        checklist=CompileSkillChecklist(),
        plan=plan,
        receipts=[
            CompileWorkReceipt(
                work_item_id="report",
                processed_source_uris=[source_uri],
                report_workspace_path="work/report.md",
            )
        ],
        capabilities=CompileCapabilities(exec_enabled=False),
        validation_report=None,
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        connection={},
        registry_kwargs={},
    )

    assert prompts
    prompt = prompts[0]
    assert "required_source_citations" in prompt
    # The exact inline-link form the submit gate checks is prefilled for the model.
    assert f"[source]({source_uri})" in prompt
    assert "read_workspace_files" not in prompt  # guidance lives in the system prompt


@pytest.mark.asyncio
async def test_compile_output_receipt_requires_real_files_and_planned_page_citations():
    files = {"index.md": b"# Index"}

    class Sandbox:
        async def read_file_bytes(self, path):
            return files[path]

    class Manager:
        async def get_sandbox(self, session_key):
            del session_key
            return Sandbox()

    source_uri = "viking://resources/source/a.md"
    tool = SubmitCompileOutputsTool(
        expected_paths={
            "page-001": ("__compile_staging__/wiki_pages/concepts/reading.md", True),
            "index": ("index.md", False),
        },
        expected_source_uris={"page-001": (source_uri,)},
        required_workspace_reads={"__compile_staging__/work/item-001/report.md"},
        observed_workspace_paths={"__compile_staging__/work/item-001/report.md"},
        limits=CompileLimits(),
    )
    context = ToolContext(
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=Manager(),
    )

    missing = await tool.execute(context, output_ids=["page-001", "index"])
    assert "planned output files do not exist" in missing

    files["__compile_staging__/wiki_pages/concepts/reading.md"] = b"# Reading\n\nNo source."
    uncited = await tool.execute(context, output_ids=["page-001", "index"])
    assert "must cite one of its planned source URIs" in uncited

    files["__compile_staging__/wiki_pages/concepts/reading.md"] = (
        f"# Reading\n\n[Source]({source_uri})".encode()
    )
    accepted = await tool.execute(context, output_ids=["page-001", "index"])
    assert accepted == "Compile output batch accepted with 2 file(s)."
    assert tool.receipt is not None


@pytest.mark.asyncio
async def test_trusted_assembly_materializes_every_planned_output_without_internal_files():
    source_uri = "viking://resources/source/a.md"
    plan = CompileOutputPlan.model_validate(
        {
            "pages": [
                {
                    "output_id": "page-001",
                    "page_id": 1,
                    "title": "Reading Skills",
                    "page_type": "concept",
                    "summary": "Reusable reading skills.",
                    "source_ids": ["src_1"],
                    "source_uris": [source_uri],
                    "report_ids": ["item-001"],
                    "path_hint": "concepts/reading-skills.md",
                },
                {
                    "output_id": "page-002",
                    "page_id": 2,
                    "title": "Assessment Methods",
                    "page_type": "concept",
                    "summary": "Reusable assessment methods.",
                    "source_ids": ["src_1"],
                    "source_uris": [source_uri],
                    "report_ids": ["item-001"],
                    "path_hint": "concepts/assessment-methods.md",
                },
            ],
            "files": [
                {
                    "output_id": "index",
                    "description": "Navigation index",
                    "report_ids": ["item-001"],
                    "path": "index.md",
                }
            ],
            "groups": [
                {
                    "group_id": "wiki-pages",
                    "output_ids": ["page-001", "page-002"],
                },
                {
                    "group_id": "navigation",
                    "output_ids": ["index"],
                    "depends_on": ["wiki-pages"],
                },
            ],
        }
    )
    files = {
        "__compile_staging__/wiki_pages/concepts/reading-skills.md": (
            f"# Reading Skills\n\n[Source]({source_uri})".encode()
        ),
        "__compile_staging__/wiki_pages/concepts/assessment-methods.md": (
            f"# Assessment Methods\n\n[Source]({source_uri})".encode()
        ),
        "index.md": b"# Index\n",
    }

    class Sandbox:
        async def read_file_bytes(self, path):
            return files[path]

    class Manager:
        async def get_sandbox(self, session_key):
            del session_key
            return Sandbox()

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()
    submit, bundle, payloads = await service._assemble_output_plan(
        plan=plan,
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=Manager(),
        capabilities=CompileCapabilities(exec_enabled=False),
        contract=CompileContract(output="mixed"),
        registry_kwargs={
            "source_ids": {"src_1"},
            "catalog_uris": set(),
            "file_catalog_uris": set(),
            "source_file_uris": {source_uri},
            "target_uri": "viking://resources/wiki",
            "workspace_baseline": None,
            "wiki_uri_resolver": None,
        },
    )

    assert [page.page_id for page in bundle.pages] == [1, 2]
    assert all(page.body_markdown for page in bundle.pages)
    assert [file.path for file in bundle.files] == ["index.md"]
    assert payloads == [b"# Index\n"]
    assert "__compile_staging__/output-plan.json" not in {file.path for file in bundle.files}
    assert set(submit.page_workspace_paths) == {1, 2}


def test_linked_skill_paths_follows_local_markdown_references_recursively():
    skill_texts = {
        "SKILL.md": (
            "Read [schema](references/schema.md#pages), "
            "`${SKILL_ROOT}/references/plain.md`, and [site](https://example.com)."
        ),
        "references/schema.md": "Then read [naming](naming.md).",
        "references/naming.md": "Use abstract names.",
        "references/plain.md": "Use exact file outputs.",
        "references/unlinked.md": "Ignore me.",
    }

    assert BotCompileService._linked_skill_paths(skill_texts) == [
        "references/schema.md",
        "references/plain.md",
        "references/naming.md",
    ]


def test_coverage_units_are_complete_first_level_entries_not_recursive_file_count():
    sources = [
        {
            "source_id": "src_1",
            "directory_uri": "viking://resources/source",
            "entries": [
                {
                    "uri": "viking://resources/source/A",
                    "name": "A",
                    "is_dir": True,
                },
                {
                    "uri": "viking://resources/source/A/one.md",
                    "name": "one.md",
                    "is_dir": False,
                },
                {
                    "uri": "viking://resources/source/A/two.md",
                    "name": "two.md",
                    "is_dir": False,
                },
                {
                    "uri": "viking://resources/source/B.md",
                    "name": "B.md",
                    "is_dir": False,
                },
            ],
        }
    ]

    units = BotCompileService._build_coverage_units(sources)

    assert [(unit["name"], unit["leaf_source_count"]) for unit in units] == [
        ("A", 2),
        ("B.md", 1),
    ]


@pytest.mark.asyncio
async def test_declared_validator_failure_returns_repairable_issue():
    class Sandbox:
        async def execute(self, command, timeout):
            assert command == "python3 skills/compiler/validate.py"
            assert timeout == 300
            return "missing logic/claims.md\n\nExit code: 1"

    class Manager:
        async def get_sandbox(self, session_key):
            return Sandbox()

    service = object.__new__(BotCompileService)
    report = await service._run_contract_commands(
        contract=CompileContract(validation_commands=["python3 skills/compiler/validate.py"]),
        sandbox_manager=Manager(),
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
    )

    assert not report.passed
    assert report.issues[0].code == "VALIDATOR_FAILED"
    assert "missing logic/claims.md" in report.issues[0].message


def test_wiki_page_requires_exactly_one_body_source():
    body = _page(1, "One")
    body.pop("body_markdown")

    with pytest.raises(ValueError, match="exactly one of body_markdown"):
        CompileBundleDraft.model_validate({"pages": [body]})
    with pytest.raises(ValueError, match="exactly one of body_markdown"):
        CompileBundleDraft.model_validate(
            {
                "pages": [
                    {
                        **body,
                        "body_markdown": "Inline",
                        "body_workspace_path": "pages/one.md",
                    }
                ]
            }
        )


def test_submit_tool_rejects_raw_payload_wrapper_with_actionable_hint():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )

    assert tool.validate_params({"raw": "{}"}) == [
        "use the tool schema directly; do not wrap the payload in a JSON string"
    ]
    assert {"pages", "files"} <= set(tool.parameters["required"])
    assert "missing required files" in tool.validate_params({"pages": []})
    assert tool.validate_params({"pages": [], "files": []}) == []


@pytest.mark.asyncio
async def test_file_contract_accepts_exact_artifact_tree_and_rejects_wiki_or_missing_files():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/ara",
        limits=CompileLimits(),
        contract=CompileContract(
            output="files",
            required_paths=["PAPER.md"],
            required_globs=["logic/*.md"],
        ),
    )

    missing = await tool.execute(
        ToolContext(),
        files=[{"path": "PAPER.md", "content": "# Paper"}],
    )
    assert "missing contract output(s): logic/*.md" in missing

    accepted = await tool.execute(
        ToolContext(),
        files=[
            {"path": "PAPER.md", "content": "# Paper"},
            {"path": "logic/problem.md", "content": "# Problem"},
        ],
    )
    assert accepted == "Compile bundle accepted with 0 page(s) and 2 file(s)."

    rejected_page = await tool.execute(ToolContext(), pages=[_page(1, "Paper")])
    assert "contract only permits file outputs" in rejected_page


@pytest.mark.asyncio
async def test_submit_tool_never_accepts_internal_compile_paths():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/output",
        limits=CompileLimits(),
    )

    result = await tool.execute(
        ToolContext(),
        files=[{"path": "__compile_staging__/source-manifest.json", "content": "{}"}],
    )

    assert "reserved internal Compile path" in result


@pytest.mark.asyncio
async def test_work_receipt_requires_all_assigned_sources_to_be_observed_and_reported():
    item = CompileWorkItem(
        work_item_id="item-001",
        source_uris=["viking://resources/source/a.md", "viking://resources/source/b.md"],
    )
    observed = {"viking://resources/source/a.md"}

    class Sandbox:
        async def read_file_bytes(self, path):
            assert path == "__compile_staging__/work/item-001/report.md"
            return b"# Complete report"

    class Manager:
        async def get_sandbox(self, session_key):
            return Sandbox()

    context = ToolContext(
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=Manager(),
    )
    tool = SubmitCompileWorkTool(
        work_item=item,
        observed_content_uris=observed,
        limits=CompileLimits(),
    )
    params = {
        "work_item_id": item.work_item_id,
        "processed_source_uris": item.source_uris,
        "report_workspace_path": "__compile_staging__/work/item-001/report.md",
    }

    unread = await tool.execute(context, **params)
    assert "assigned sources were not read" in unread

    observed.add("viking://resources/source/b.md")
    accepted = await tool.execute(context, **params)
    assert accepted == "Compile work item 'item-001' accepted."
    assert tool.receipt is not None


@pytest.mark.asyncio
async def test_skill_checklist_requires_strong_and_trusted_skill_evidence():
    tool = SubmitCompileChecklistTool(
        evidence_blocks={
            "ev0001": ("SKILL.md", "1. **Create** an `index.md` at the output root."),
            "ev0002": ("SKILL.md", "Optional background."),
        },
        required_evidence_ids={"ev0001"},
    )
    rule = {
        "rule_id": "root_index",
        "statement": "Create a root index.",
        "evidence_id": "ev0001",
        "phase": "plan",
    }

    missing_required = await tool.execute(ToolContext(), rules=[{**rule, "evidence_id": "ev0002"}])
    assert "strongly binding Skill evidence blocks" in missing_required

    invented = await tool.execute(
        ToolContext(),
        rules=[rule, {**rule, "rule_id": "invented", "evidence_id": "ev9999"}],
    )
    assert "unknown evidence_id" in invented

    accepted = await tool.execute(ToolContext(), rules=[rule])
    assert accepted == "Compile Skill checklist accepted with 1 rule(s)."
    assert tool.structured_result is not None
    assert tool.structured_result.rules[0].evidence_path == "SKILL.md"
    assert tool.structured_result.rules[0].evidence_quote.startswith("1. **Create**")
    assert tool.structured_result.rules[0].phase == "plan"


@pytest.mark.asyncio
async def test_skill_checklist_rejects_too_many_critical_rules():
    tool = SubmitCompileChecklistTool(
        evidence_blocks={
            "ev0001": ("SKILL.md", "1. **Create** an `index.md` at the output root."),
        },
        max_critical_rules=1,
    )
    base = {"statement": "Rule.", "evidence_id": "ev0001", "phase": "plan"}
    two_critical = [
        {**base, "rule_id": "a", "critical": True},
        {**base, "rule_id": "b", "critical": True},
    ]

    rejected = await tool.execute(ToolContext(), rules=two_critical)
    assert "at most 1 rules may be critical" in rejected
    assert tool.structured_result is None

    accepted = await tool.execute(
        ToolContext(),
        rules=[
            {**base, "rule_id": "a", "critical": True},
            {**base, "rule_id": "b", "critical": False},
        ],
    )
    assert accepted == "Compile Skill checklist accepted with 2 rule(s)."


def test_skill_evidence_blocks_preserve_wrapped_directives_and_stable_ids():
    blocks = BotCompileService._skill_evidence_blocks(
        {
            "references/schema.md": "- Secondary rule.\n",
            "SKILL.md": (
                "# Rules\n\n"
                "- Create one canonical page for every meaningful source\n"
                "  group and keep its final path stable.\n"
                "- Create a root index.\n"
            ),
        }
    )

    assert list(blocks) == ["ev0001", "ev0002", "ev0003", "ev0004"]
    assert blocks["ev0002"] == (
        "SKILL.md",
        "- Create one canonical page for every meaningful source group and keep its final path stable.",
    )
    assert blocks["ev0004"] == ("references/schema.md", "- Secondary rule.")
    assert BotCompileService._required_skill_evidence_ids(blocks) == {
        "ev0002",
        "ev0003",
    }


@pytest.mark.asyncio
async def test_validation_submission_requires_consistent_pass_state():
    tool = SubmitCompileValidationTool()

    invalid = await tool.execute(ToolContext(), passed=True, issues=[{"code": "X", "message": "x"}])
    assert invalid.startswith("Error:")

    accepted = await tool.execute(
        ToolContext(), passed=False, issues=[{"code": "X", "message": "x"}]
    )
    assert accepted == "Compile validation accepted."
    assert tool.structured_result is not None and not tool.structured_result.passed


@pytest.mark.asyncio
async def test_validation_submission_requires_declared_workspace_reads():
    observed = set()
    tool = SubmitCompileValidationTool(
        required_workspace_reads={
            "__compile_staging__/candidate.json",
            "__compile_staging__/work/item-001/report.md",
        },
        observed_workspace_paths=observed,
    )

    unread = await tool.execute(ToolContext(), passed=True)
    assert "required workspace files were not read" in unread

    observed.update(tool.required_workspace_reads)
    accepted = await tool.execute(ToolContext(), passed=True)
    assert accepted == "Compile validation accepted."


@pytest.mark.asyncio
async def test_validation_submission_checks_every_contract_requirement_exactly_once():
    tool = SubmitCompileValidationTool(required_requirement_ids={"coverage", "naming"})

    missing = await tool.execute(ToolContext(), passed=True, checked_requirement_ids=["coverage"])
    assert "every contract requirement exactly once" in missing

    duplicate = await tool.execute(
        ToolContext(),
        passed=True,
        checked_requirement_ids=["coverage", "coverage", "naming"],
    )
    assert "every contract requirement exactly once" in duplicate

    accepted = await tool.execute(
        ToolContext(),
        passed=True,
        checked_requirement_ids=["coverage", "naming"],
    )
    assert accepted == "Compile validation accepted."


@pytest.mark.asyncio
async def test_validation_submission_checks_every_applicable_skill_rule_exactly_once():
    tool = SubmitCompileValidationTool(required_rule_ids={"root_index", "source_pages"})
    root_check = {
        "rule_id": "root_index",
        "passed": True,
        "evidence": "index.md is planned at the root",
    }

    missing = await tool.execute(ToolContext(), passed=True, rule_checks=[root_check])
    assert "every required audit rule exactly once" in missing

    failed_without_issue = await tool.execute(
        ToolContext(),
        passed=False,
        rule_checks=[
            root_check,
            {"rule_id": "source_pages", "passed": False, "evidence": "none planned"},
        ],
    )
    assert "passed must be true exactly when issues is empty" in failed_without_issue

    accepted = await tool.execute(
        ToolContext(),
        passed=False,
        issues=[{"code": "SKILL_RULE_FAILED", "message": "Source pages are missing."}],
        rule_checks=[
            root_check,
            {"rule_id": "source_pages", "passed": False, "evidence": "none planned"},
        ],
    )
    assert accepted == "Compile validation accepted."


def test_materialization_checklist_json_drops_evidence():
    rules = [
        CompileSkillRule(
            rule_id=f"rule_{index:02d}",
            statement=f"Rule {index} statement",
            evidence_path=f"SKILL.md#section-{index}",
            evidence_quote=f"Quoted directive {index} that grounds the rule.",
            phase="plan" if index % 2 == 0 else "candidate",
        )
        for index in range(3)
    ]
    checklist = CompileSkillChecklist(rules=rules)

    slim = BotCompileService._materialization_checklist_json(checklist)
    slim_data = json.loads(slim)

    assert [rule["rule_id"] for rule in slim_data["rules"]] == [
        "rule_00",
        "rule_01",
        "rule_02",
    ]
    for rule in slim_data["rules"]:
        assert set(rule) == {"rule_id", "statement", "phase"}
    # The heavy grounding fields must not leak into the batch prompt.
    assert "evidence_path" not in slim
    assert "evidence_quote" not in slim
    assert "SKILL.md#section-" not in slim
    assert "Quoted directive" not in slim
    # But the full checklist the audit phases receive still carries evidence.
    assert "evidence_quote" in checklist.model_dump_json()


@pytest.mark.asyncio
async def test_plan_audit_batches_rules_and_aggregates_failures():
    rules = [
        CompileSkillRule(
            rule_id=f"rule_{index:02d}",
            statement=f"Rule {index}",
            evidence_path="SKILL.md",
            evidence_quote=f"Must satisfy rule {index}.",
            phase="plan",
            critical=True,
        )
        for index in range(13)
    ]
    calls = []

    class RequestLoop:
        async def run_structured_task(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                batch = rules[:12]
            elif len(calls) == 2:
                batch = rules[12:]
            else:
                batch = []
            failed = len(calls) == 2
            task_rule_id = next(
                (
                    rule_id
                    for rule_id in (
                        "compile_task_reason_001",
                        "compile_task_reason_002",
                    )
                    if rule_id in kwargs["user_prompt"]
                ),
                None,
            )
            report = CompileValidationReport(
                passed=not failed,
                issues=(
                    [{"code": "SKILL_RULE_FAILED", "message": "Rule 12 failed."}] if failed else []
                ),
                rule_checks=[
                    {
                        "rule_id": rule.rule_id,
                        "passed": not failed,
                        "evidence": f"Concrete evidence for {rule.rule_id}",
                    }
                    for rule in batch
                ]
                + (
                    [
                        {
                            "rule_id": task_rule_id,
                            "passed": not failed,
                            "evidence": f"Checked {task_rule_id} against plan fields.",
                        },
                    ]
                    if task_rule_id is not None
                    else []
                ),
            )
            return report, [], None, 1

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits(task_reason_challenge_enabled=True)
    service._build_compile_registry = lambda *args, **kwargs: (ToolRegistry(), [])
    report = await service._audit_output_plan(
        request_loop=RequestLoop(),
        task_reason="Inspect every input; Keep names reusable and put instance details in the content.",
        skill_name="generic-skill",
        skill_content="Apply the selected Skill.",
        skill_references={},
        checklist=CompileSkillChecklist(rules=rules),
        contract=CompileContract(),
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        connection={},
        capabilities=CompileCapabilities(exec_enabled=False),
        registry_kwargs={},
    )

    assert len(calls) == 6
    assert "Audit batch 1 of 4." in calls[0]["user_prompt"]
    assert "Audit batch 2 of 4." in calls[1]["user_prompt"]
    assert "Audit batch 3 of 4." in calls[2]["user_prompt"]
    assert "Audit batch 3 of 4." in calls[3]["user_prompt"]
    assert "Audit batch 4 of 4." in calls[4]["user_prompt"]
    assert "Audit batch 4 of 4." in calls[5]["user_prompt"]
    assert "Reject the plan when any applicable Skill rule fails" in calls[0]["system_prompt"]
    assert "task_reason" not in calls[0]["system_prompt"]
    assert "binding user instruction" in calls[2]["system_prompt"]
    assert "compile_task_reason" not in calls[0]["user_prompt"]
    assert "compile_task_reason" not in calls[1]["user_prompt"]
    assert "compile_task_reason_001" in calls[2]["user_prompt"]
    assert "compile_task_reason_002" not in calls[2]["user_prompt"]
    assert "adversarial second pass" in calls[3]["system_prompt"]
    assert "compile_task_reason_002" in calls[4]["user_prompt"]
    assert "compile_task_reason_001" not in calls[4]["user_prompt"]
    assert "Keep names reusable" in calls[4]["user_prompt"]
    assert "adversarial second pass" in calls[5]["system_prompt"]
    assert "audit the one supplied task-reason rule" in calls[2]["system_prompt"]
    assert "task-reason-audit-context.json" in calls[2]["user_prompt"]
    assert "plan-audit-context.json" not in calls[2]["user_prompt"]
    assert not report.passed
    assert [issue.message for issue in report.issues] == ["Rule 12 failed."]
    assert [check.rule_id for check in report.rule_checks] == [rule.rule_id for rule in rules] + [
        "compile_task_reason_001",
        "compile_task_reason_002",
    ]


@pytest.mark.asyncio
async def test_plan_task_reason_challenge_skipped_by_default():
    calls = []

    class RequestLoop:
        async def run_structured_task(self, **kwargs):
            calls.append(kwargs)
            return (
                CompileValidationReport(
                    passed=True,
                    rule_checks=[
                        {
                            "rule_id": "compile_task_reason_001",
                            "passed": True,
                            "evidence": "Checked every exact title and path.",
                        }
                    ],
                ),
                [],
                None,
                1,
            )

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()
    service._build_compile_registry = lambda *args, **kwargs: (ToolRegistry(), [])
    report = await service._audit_output_plan(
        request_loop=RequestLoop(),
        task_reason="Keep names reusable and put instance details in the content.",
        skill_name="generic-skill",
        skill_content="Apply the selected Skill.",
        skill_references={},
        checklist=CompileSkillChecklist(),
        contract=CompileContract(),
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        connection={},
        capabilities=CompileCapabilities(exec_enabled=False),
        registry_kwargs={},
    )

    # With the default flag off, a passing task-reason batch is not re-audited by
    # the adversarial challenger, so exactly one call is made and no challenge runs.
    assert len(calls) == 1
    assert "adversarial second pass" not in calls[0]["system_prompt"]
    assert report.passed


@pytest.mark.asyncio
async def test_plan_task_reason_challenger_can_veto_primary_pass():
    calls = []

    class RequestLoop:
        async def run_structured_task(self, **kwargs):
            calls.append(kwargs)
            passed = len(calls) == 1
            return (
                CompileValidationReport(
                    passed=passed,
                    issues=(
                        []
                        if passed
                        else [{"code": "COUNTEREXAMPLE", "message": "A title violates the rule."}]
                    ),
                    rule_checks=[
                        {
                            "rule_id": "compile_task_reason_001",
                            "passed": passed,
                            "evidence": "Checked every exact title and path.",
                        }
                    ],
                ),
                [],
                None,
                1,
            )

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits(task_reason_challenge_enabled=True)
    service._build_compile_registry = lambda *args, **kwargs: (ToolRegistry(), [])
    report = await service._audit_output_plan(
        request_loop=RequestLoop(),
        task_reason="Keep names reusable and put instance details in the content.",
        skill_name="generic-skill",
        skill_content="Apply the selected Skill.",
        skill_references={},
        checklist=CompileSkillChecklist(),
        contract=CompileContract(),
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        connection={},
        capabilities=CompileCapabilities(exec_enabled=False),
        registry_kwargs={},
    )

    assert len(calls) == 2
    assert "adversarial second pass" not in calls[0]["system_prompt"]
    assert "adversarial second pass" in calls[1]["system_prompt"]
    assert not report.passed
    assert [issue.code for issue in report.issues] == ["COUNTEREXAMPLE"]
    assert [check.rule_id for check in report.rule_checks] == ["compile_task_reason_001"]


@pytest.mark.asyncio
async def test_candidate_audit_requires_explicit_task_reason_check():
    rule = CompileSkillRule(
        rule_id="candidate_rule",
        statement="Check the generated output.",
        evidence_path="SKILL.md",
        evidence_quote="Check the generated output.",
        phase="candidate",
        critical=True,
    )
    calls = []
    submission_tools = []

    class RequestLoop:
        async def run_structured_task(self, **kwargs):
            calls.append(kwargs)
            rule_id = next(
                (
                    rule_id
                    for rule_id in (
                        "compile_task_reason_001",
                        "compile_task_reason_002",
                    )
                    if rule_id in kwargs["user_prompt"]
                ),
                "candidate_rule",
            )
            return (
                CompileValidationReport(
                    passed=True,
                    rule_checks=[
                        {
                            "rule_id": rule_id,
                            "passed": True,
                            "evidence": f"Checked {rule_id} in output.md.",
                        },
                    ],
                ),
                [],
                None,
                1,
            )

    def build_registry(*args, **kwargs):
        del args
        submission_tools.append(kwargs["submission_tool"])
        return ToolRegistry(), []

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits(task_reason_challenge_enabled=True)
    service._build_compile_registry = build_registry
    report = await service._audit_candidate(
        request_loop=RequestLoop(),
        task_reason="Inspect every output; Keep output names reusable.",
        skill_name="generic-skill",
        skill_content="Apply the selected Skill.",
        skill_references={},
        checklist=CompileSkillChecklist(rules=[rule]),
        contract=CompileContract(),
        required_output_reads=set(),
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        connection={},
        capabilities=CompileCapabilities(exec_enabled=False),
        registry_kwargs={},
    )

    assert report.passed
    assert len(calls) == 5
    assert submission_tools[0].required_rule_ids == {"candidate_rule"}
    assert submission_tools[1].required_rule_ids == {"compile_task_reason_001"}
    assert submission_tools[2].required_rule_ids == {"compile_task_reason_001"}
    assert submission_tools[3].required_rule_ids == {"compile_task_reason_002"}
    assert submission_tools[4].required_rule_ids == {"compile_task_reason_002"}
    assert "task-reason-audit-context.json" not in calls[0]["user_prompt"]
    assert "Audit the one supplied task-reason rule" in calls[1]["system_prompt"]
    assert "compile_task_reason_001" in calls[1]["user_prompt"]
    assert "compile_task_reason_002" not in calls[1]["user_prompt"]
    assert "adversarial second pass" in calls[2]["system_prompt"]
    assert "compile_task_reason_002" in calls[3]["user_prompt"]
    assert "compile_task_reason_001" not in calls[3]["user_prompt"]
    assert "Keep output names reusable." in calls[3]["user_prompt"]
    assert "adversarial second pass" in calls[4]["system_prompt"]
    assert [check.rule_id for check in report.rule_checks] == [
        "candidate_rule",
        "compile_task_reason_001",
        "compile_task_reason_002",
    ]


@pytest.mark.asyncio
async def test_plan_audit_context_preserves_binding_task_reason():
    files = {
        "__compile_staging__/source-manifest.json": json.dumps(
            {"coverage_units": [], "work_items": []}
        ).encode()
    }

    class Sandbox:
        async def read_file_bytes(self, path):
            return files[path]

        async def write_file(self, path, content):
            files[path] = content.encode()

    class Manager:
        async def get_sandbox(self, session_key):
            del session_key
            return Sandbox()

    plan = CompileOutputPlan.model_validate(
        {
            "files": [{"output_id": "index", "description": "Index", "path": "index.md"}],
            "groups": [{"group_id": "navigation", "output_ids": ["index"]}],
        }
    )
    service = object.__new__(BotCompileService)
    await service._write_plan_audit_context(
        sandbox_manager=Manager(),
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        task_reason="Keep output names reusable and move instance details into content.",
        plan=plan,
        receipts=[],
    )

    context = json.loads(files["__compile_staging__/plan-audit-context.json"])
    assert context["task_reason"] == (
        "Keep output names reusable and move instance details into content."
    )
    task_context = json.loads(files["__compile_staging__/task-reason-audit-context.json"])
    assert task_context == {
        "task_reason": "Keep output names reusable and move instance details into content.",
        "output_plan": plan.model_dump(mode="json"),
    }


@pytest.mark.asyncio
async def test_wiki_submission_requires_exact_leaf_citation_and_plain_relation_anchor():
    source_uri = "viking://resources/source/chapter.md"
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        source_file_uris={source_uri},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )

    missing_citation = await tool.execute(ToolContext(), pages=[_page(1, "One"), _page(2, "Two")])
    assert "pages 1, 2 must each link at least one exact source file URI" in missing_citation

    invalid_anchor = await tool.execute(
        ToolContext(),
        pages=[
            _page(1, "One", body_markdown=f"Read [Two] next.\n\n[{source_uri}]({source_uri})"),
            _page(2, "Two", body_markdown=f"Two.\n\n[{source_uri}]({source_uri})"),
        ],
        links=[{"f": 1, "t": 2, "match_text": "[Two]"}],
    )
    assert "match_text must not contain Markdown brackets" in invalid_anchor

    accepted = await tool.execute(
        ToolContext(),
        pages=[_page(1, "One", body_markdown=f"One.\n\n[来源]({source_uri})")],
    )
    assert accepted == "Compile bundle accepted with 1 page(s) and 0 file(s)."


def test_submit_tool_schema_requires_workspace_page_bodies_when_available():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        require_workspace_pages=True,
    )

    page_schema = tool.parameters["$defs"]["WikiPageDraft"]
    assert "body_markdown" not in page_schema["properties"]
    assert "body_workspace_path" in page_schema["required"]


@pytest.mark.parametrize(
    "target_uri",
    ["viking://resources/wiki", "viking://agent/skills"],
)
@pytest.mark.parametrize("exec_enabled", [True, False])
def test_submit_tool_schema_requires_workspace_artifacts_when_available(target_uri, exec_enabled):
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri=target_uri,
        limits=CompileLimits(),
        require_workspace_files=True,
        exec_enabled=exec_enabled,
    )

    file_schema = tool.parameters["$defs"]["CompileFileDraft"]
    assert "content" not in file_schema["properties"]
    assert "workspace_path" in file_schema["required"]
    if target_uri.startswith("viking://resources/"):
        assert file_schema["oneOf"] == [
            {"required": ["path"], "properties": {"path": {"type": "string"}}},
            {
                "required": ["update_uri"],
                "properties": {"update_uri": {"type": "string"}},
            },
        ]
    hint = tool.validate_params({"raw": "{}"})[0]
    assert ("write_file or exec" in tool.description) is exec_enabled
    assert ("write_file or exec" in hint) is exec_enabled
    assert "workspace_path instead of inline content" in hint
    if not exec_enabled:
        assert "with write_file," in tool.description
        assert "with write_file and submit" in hint


@pytest.mark.asyncio
async def test_submit_tool_accepts_only_one_complete_skill_package():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://agent/skills",
        limits=CompileLimits(),
    )

    schema = tool.parameters
    assert set(schema["properties"]) == {"files"}
    assert schema["required"] == ["files"]
    assert set(schema["$defs"]) == {"CompileFileDraft"}
    assert "update_uri" not in schema["$defs"]["CompileFileDraft"]["properties"]
    assert "path" in schema["$defs"]["CompileFileDraft"]["required"]

    accepted = await tool.execute(
        ToolContext(),
        files=[
            {
                "path": "weekly-report/SKILL.md",
                "content": (
                    "---\n"
                    "name: weekly-report\n"
                    "description: Generate a concise weekly report.\n"
                    "---\n\n"
                    "Follow the source material and produce the report."
                ),
            },
            {
                "path": "weekly-report/references/format.md",
                "content": "# Weekly report format\n",
            },
        ],
    )

    assert accepted == "Skill bundle accepted for 'weekly-report' with 2 file(s)."
    assert tool.skill_name == "weekly-report"
    assert tool.bundle is not None and tool.bundle.pages == []

    missing_skill_md = await tool.execute(
        ToolContext(),
        files=[{"path": "weekly-report/references/format.md", "content": "# Format"}],
    )
    assert "must include weekly-report/SKILL.md" in missing_skill_md

    multiple_skills = await tool.execute(
        ToolContext(),
        files=[
            {
                "path": "one/SKILL.md",
                "content": "---\nname: one\ndescription: One\n---\nOne",
            },
            {"path": "two/guide.md", "content": "Two"},
        ],
    )
    assert "exactly one top-level Skill directory" in multiple_skills

    derived_file = await tool.execute(
        ToolContext(),
        files=[
            {
                "path": "weekly-report/SKILL.md",
                "content": ("---\nname: weekly-report\ndescription: Weekly report\n---\nWrite it"),
            },
            {"path": "weekly-report/.overview.md", "content": "Generated"},
        ],
    )
    assert "reserved output file path" in derived_file
    assert tool.bundle is None

    invalid_yaml = await tool.execute(
        ToolContext(),
        files=[
            {
                "path": "weekly-report/SKILL.md",
                "content": "---\nname: [\ndescription: Weekly report\n---\nWrite it",
            }
        ],
    )
    assert invalid_yaml.startswith("Error: Invalid Skill bundle:")

    long_description = await tool.execute(
        ToolContext(),
        files=[
            {
                "path": "weekly-report/SKILL.md",
                "content": (
                    "---\nname: weekly-report\ndescription: " + "x" * 1025 + "\n---\nWrite it"
                ),
            }
        ],
    )
    assert "description must not exceed 1024 characters" in long_description


def test_renderer_creates_okf_pages_links_and_citations():
    summary = "Residual building block designs, network variants, shortcut connection types, and design principles aligned with VGG architecture."
    bundle = CompileBundleDraft.model_validate(
        {
            "pages": [
                _page(1, "Alpha", body_markdown="Read Beta next.", summary=summary),
                _page(2, "Beta"),
            ],
            "links": [{"f": 1, "t": 2, "match_text": "Beta"}],
        }
    )
    rendered = WikiRenderer().render(
        bundle=bundle,
        target_uri="viking://resources/wiki",
        source_roots={"src_1": "viking://resources/source"},
        catalog_uris=set(),
        existing_raw={},
    )
    assert rendered.created == [
        "viking://resources/wiki/alpha.md",
        "viking://resources/wiki/beta.md",
    ]
    assert rendered.wiki_uris == [
        "viking://resources/wiki/alpha.md",
        "viking://resources/wiki/beta.md",
    ]
    assert rendered.link_count == 1
    first = rendered.operations[0]
    assert first["precondition"] == {"kind": "create_if_absent"}
    assert "type: concept" in first["content"]
    assert f"description: {summary}\n" in first["content"]
    assert "Read [Beta](./beta.md) next." in first["content"]
    assert "[1] [source](viking://resources/source)" in first["content"]


def test_renderer_localizes_generated_sections_from_body_language():
    source_detail = "viking://resources/source/章节.md"
    bundle = CompileBundleDraft.model_validate(
        {
            "pages": [
                _page(
                    1,
                    "语文课标",
                    path_hint="entities/语文课标.md",
                    body_markdown=(
                        "语文课程通过阅读框架描述第一学段要求。\n\n"
                        "https://example.com/a/very/long/english/source/path\n\n"
                        "# 引用来源\n\n"
                        f"[9] [章节]({source_detail})"
                    ),
                ),
                _page(
                    2,
                    "阅读框架",
                    path_hint="concepts/阅读框架.md",
                    body_markdown="阅读框架描述学生的主要能力层级。",
                ),
            ],
            "links": [{"f": 1, "t": 2, "match_text": "阅读框架"}],
        }
    )

    rendered = WikiRenderer().render(
        bundle=bundle,
        target_uri="viking://resources/wiki",
        source_roots={"src_1": "viking://resources/source"},
        catalog_uris=set(),
        existing_raw={},
    )
    operations = {operation["uri"]: operation["content"] for operation in rendered.operations}
    standard = operations["viking://resources/wiki/entities/语文课标.md"]
    framework = operations["viking://resources/wiki/concepts/阅读框架.md"]

    assert standard.count("# 引用来源") == 1
    assert "# Citations" not in standard
    assert f"[1] [章节]({source_detail})" in standard
    assert "## 相关页面" in framework
    assert "Related pages" not in framework
    assert "# 引用来源" in framework


def test_wiki_page_title_path_normalizes_spaced_dashes_only():
    assert (
        wiki_page_path_from_title("Experimental Designs - Residual Networks")
        == "Experimental_Designs_Residual_Networks"
    )
    assert wiki_page_path_from_title("One-Page Overview") == "One-Page_Overview"


def test_renderer_treats_index_and_log_as_artifacts_not_wiki_pages():
    page_bundle = CompileBundleDraft.model_validate(
        {"pages": [_page(1, "Index", path_hint="index.md")]}
    )
    with pytest.raises(ValueError, match="reserved Wiki page path"):
        WikiRenderer().render(
            bundle=page_bundle,
            target_uri="viking://resources/wiki",
            source_roots={"src_1": "viking://resources/source"},
            catalog_uris=set(),
            existing_raw={},
        )

    artifact_bundle = CompileBundleDraft.model_validate(
        {
            "pages": [],
            "files": [
                {"path": "index.md", "content": "# Wiki\n"},
                {"path": "concepts/log.md", "content": "# Updates\n"},
            ],
        }
    )
    rendered = WikiRenderer().render(
        bundle=artifact_bundle,
        target_uri="viking://resources/wiki",
        source_roots={},
        catalog_uris=set(),
        existing_raw={},
    )
    assert rendered.created == [
        "viking://resources/wiki/index.md",
        "viking://resources/wiki/concepts/log.md",
    ]
    assert rendered.wiki_uris == []


def test_renderer_updates_existing_index_artifact():
    uri = "viking://resources/wiki/index.md"
    bundle = CompileBundleDraft.model_validate(
        {"pages": [], "files": [{"update_uri": uri, "content": "# New index\n"}]}
    )
    rendered = WikiRenderer().render(
        bundle=bundle,
        target_uri="viking://resources/wiki",
        source_roots={},
        catalog_uris=set(),
        file_catalog_uris={uri},
        existing_raw={},
        existing_bytes={uri: b"# Old index\n"},
    )
    assert rendered.updated == [uri]
    assert rendered.operations[0]["uri"] == uri


def test_renderer_linkifies_source_uris_and_adds_resource_backlinks():
    source_detail = "viking://resources/source/chapter_1.md"
    outside = "viking://resources/outside/chapter.md"
    bundle = CompileBundleDraft.model_validate(
        {
            "pages": [
                _page(
                    1,
                    "Overview",
                    body_markdown=(
                        "Read Details next.\n\n"
                        f"Source: {source_detail} section 2\n\n"
                        f"Keep code unchanged: `{source_detail}`\n\n"
                        f"Outside stays plain: {outside}"
                    ),
                ),
                _page(2, "Details"),
            ],
            "links": [{"f": 1, "t": 2, "match_text": "Details"}],
        }
    )

    rendered = WikiRenderer().render(
        bundle=bundle,
        target_uri="viking://resources/wiki",
        source_roots={"src_1": "viking://resources/source"},
        catalog_uris=set(),
        existing_raw={},
    )
    operations = {operation["uri"]: operation["content"] for operation in rendered.operations}
    overview = operations["viking://resources/wiki/overview.md"]
    details = operations["viking://resources/wiki/details.md"]

    assert "[Details](./details.md)" in overview
    assert f"[chapter_1]({source_detail})" in overview
    assert f"`{source_detail}`" in overview
    assert outside in overview and f"]({outside})" not in overview
    assert f"[1] [chapter_1]({source_detail})" in overview
    assert "[2] [source](viking://resources/source)" in overview
    assert f"[1] [chapter_1]({source_detail})  \n[2] [source]" in overview
    assert "## Related pages" in details
    assert "- [Overview](./overview.md)" in details
    assert rendered.link_count == 1


def test_renderer_adds_raw_text_and_workspace_binary_to_same_bundle():
    paper = "---\ntitle: ARA Demo\nauthors: [Ada]\n---\n\n# Layer Index\n"
    image_bytes = b"\x89PNG\r\n\x1a\nfigure"
    bundle = CompileBundleDraft.model_validate(
        {
            "pages": [_page(1, "Overview")],
            "files": [
                {"path": "PAPER.md", "content": paper},
                {
                    "path": "trace/exploration_tree.yaml",
                    "content": "nodes: []\n",
                },
                {
                    "path": "evidence/figures/figure1.png",
                    "workspace_path": "ara-output/figure1.png",
                },
            ],
        }
    )
    rendered = WikiRenderer().render(
        bundle=bundle,
        target_uri="viking://resources/wiki",
        source_roots={"src_1": "viking://resources/source"},
        catalog_uris=set(),
        existing_raw={},
        file_payloads=[None, None, image_bytes],
    )
    operations = {operation["uri"]: operation for operation in rendered.operations}
    assert operations["viking://resources/wiki/PAPER.md"]["content"] == paper
    assert (
        operations["viking://resources/wiki/trace/exploration_tree.yaml"]["content"]
        == "nodes: []\n"
    )
    encoded = operations["viking://resources/wiki/evidence/figures/figure1.png"]["content_base64"]
    assert base64.b64decode(encoded) == image_bytes
    assert rendered.created == [
        "viking://resources/wiki/overview.md",
        "viking://resources/wiki/PAPER.md",
        "viking://resources/wiki/trace/exploration_tree.yaml",
        "viking://resources/wiki/evidence/figures/figure1.png",
    ]
    assert rendered.wiki_uris == ["viking://resources/wiki/overview.md"]


def test_renderer_accepts_minimal_okf_artifact_with_unknown_fields():
    content = (
        "---\n"
        "type: research_artifact\n"
        "ara_version: 1.0\n"
        "custom: anything\n"
        "---\n\n"
        "# Research artifact\n"
    )
    bundle = CompileBundleDraft.model_validate(
        {"pages": [], "files": [{"path": "research.md", "content": content}]}
    )

    rendered = WikiRenderer().render(
        bundle=bundle,
        target_uri="viking://resources/wiki",
        source_roots={},
        catalog_uris=set(),
        existing_raw={},
    )

    assert rendered.operations[0]["content"] == content
    assert rendered.wiki_uris == ["viking://resources/wiki/research.md"]


@pytest.mark.asyncio
async def test_compile_appends_wiki_search_tag_once_per_uri():
    class Client:
        def __init__(self):
            self.calls = []

        async def set_tags(self, uri, tags, mode):
            self.calls.append((uri, tags, mode))
            return {"tags_updated": True}

    raw_client = Client()
    client = SimpleNamespace(client=raw_client)

    await BotCompileService._tag_wiki_files(
        client,
        ["viking://resources/wiki/a.md", "viking://resources/wiki/a.md"],
        target_uri="viking://resources/wiki",
    )

    assert raw_client.calls == [("viking://resources/wiki/a.md", ["ov.kind=wiki"], "append")]


@pytest.mark.asyncio
async def test_compile_collects_all_wiki_search_tag_failures():
    class Client:
        async def set_tags(self, uri, tags, mode):
            del tags, mode
            if uri.endswith("a.md"):
                raise OpenVikingError("tag service unavailable", code="UNAVAILABLE")
            return {"tags_updated": False}

    with pytest.raises(OpenVikingError, match="Content was written successfully") as exc:
        await BotCompileService._tag_wiki_files(
            SimpleNamespace(client=Client()),
            ["viking://resources/wiki/a.md", "viking://resources/wiki/b.md"],
            target_uri="viking://resources/wiki",
        )

    assert exc.value.code == "REFRESH_FAILED"
    assert exc.value.details["failures"] == {
        "viking://resources/wiki/a.md": "tag service unavailable",
        "viking://resources/wiki/b.md": "indexed record was not found",
    }
    assert "ov --sudo reindex viking://resources/wiki --mode vectors_only --wait true" in str(
        exc.value
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("---\ntype:\n---\n", 'field "type" must be a non-empty string'),
        ("---\ntype: [concept]\n---\n", 'field "type" must be a non-empty string'),
        ("---\ntype: concept\nauthors: [\n---\n", "invalid YAML frontmatter"),
        ("---\ntype: concept\n# no closing delimiter", "unterminated YAML frontmatter"),
    ],
)
def test_renderer_rejects_invalid_declared_okf_artifact(content, message):
    bundle = CompileBundleDraft.model_validate(
        {"pages": [], "files": [{"path": "invalid.md", "content": content}]}
    )

    with pytest.raises(ValueError, match=message):
        WikiRenderer().render(
            bundle=bundle,
            target_uri="viking://resources/wiki",
            source_roots={},
            catalog_uris=set(),
            existing_raw={},
        )


def test_renderer_checks_size_before_parsing_okf_artifact():
    content = "---\ntype: [broken\n---\n"
    bundle = CompileBundleDraft.model_validate(
        {"pages": [], "files": [{"path": "large.md", "content": content}]}
    )

    with pytest.raises(ValueError, match="final content size limit"):
        WikiRenderer(CompileLimits(output_total_bytes=8)).render(
            bundle=bundle,
            target_uri="viking://resources/wiki",
            source_roots={},
            catalog_uris=set(),
            existing_raw={},
        )


def test_renderer_rejects_page_path_colliding_with_artifact():
    bundle = CompileBundleDraft.model_validate({"pages": [_page(1, "PAPER", path_hint="PAPER.md")]})

    with pytest.raises(ValueError, match="already exists"):
        WikiRenderer().render(
            bundle=bundle,
            target_uri="viking://resources/wiki",
            source_roots={"src_1": "viking://resources/source"},
            catalog_uris=set(),
            file_catalog_uris={"viking://resources/wiki/PAPER.md"},
            existing_raw={},
        )


def test_renderer_raw_update_cannot_remove_okf_from_existing_wiki_page():
    uri = "viking://resources/wiki/existing.md"
    bundle = CompileBundleDraft.model_validate(
        {"pages": [], "files": [{"update_uri": uri, "content": "# Plain Markdown"}]}
    )

    with pytest.raises(ValueError, match="must retain valid OKF"):
        WikiRenderer().render(
            bundle=bundle,
            target_uri="viking://resources/wiki",
            source_roots={},
            catalog_uris={uri},
            file_catalog_uris={uri},
            existing_raw={},
            existing_bytes={uri: b"---\ntype: concept\n---\n\n# Existing"},
        )


def test_renderer_raw_file_update_uses_byte_hash_and_detects_unchanged():
    uri = "viking://resources/wiki/trace/exploration_tree.yaml"
    old = b"nodes: []\n"
    unchanged_bundle = CompileBundleDraft.model_validate(
        {"pages": [], "files": [{"update_uri": uri, "content": old.decode()}]}
    )
    renderer = WikiRenderer()
    unchanged = renderer.render(
        bundle=unchanged_bundle,
        target_uri="viking://resources/wiki",
        source_roots={},
        catalog_uris=set(),
        existing_raw={},
        file_catalog_uris={uri},
        existing_bytes={uri: old},
    )
    assert unchanged.unchanged == [uri]
    assert unchanged.operations == []

    changed_bundle = CompileBundleDraft.model_validate(
        {"pages": [], "files": [{"update_uri": uri, "content": "nodes: [root]\n"}]}
    )
    changed = renderer.render(
        bundle=changed_bundle,
        target_uri="viking://resources/wiki",
        source_roots={},
        catalog_uris=set(),
        existing_raw={},
        file_catalog_uris={uri},
        existing_bytes={uri: old},
    )
    assert changed.updated == [uri]
    assert changed.operations[0]["precondition"] == {
        "kind": "replace_if_hash",
        "base_hash": content_hash(old),
    }


def test_renderer_empty_existing_update_uses_hash_precondition_and_preserves_uri():
    uri = "viking://resources/wiki/empty.md"
    bundle = CompileBundleDraft.model_validate(
        {
            "pages": [
                _page(1, "Empty", path_hint=None, update_uri=uri),
            ]
        }
    )
    rendered = WikiRenderer().render(
        bundle=bundle,
        target_uri="viking://resources/wiki",
        source_roots={"src_1": "viking://resources/source"},
        catalog_uris={uri},
        existing_raw={uri: ""},
    )
    assert rendered.updated == [uri]
    assert rendered.created == []
    assert rendered.operations[0]["precondition"] == {
        "kind": "replace_if_hash",
        "base_hash": content_hash(""),
    }


def test_renderer_preserves_unknown_frontmatter_and_merges_scoped_citations():
    uri = "viking://resources/wiki/topic.md"
    old = """---
type: legacy
title: Old
description: Old summary
custom: keep-me
---

Old body

# Citations

[1] [Detail](viking://resources/source/detail.md)
[2] [Outside](viking://resources/other/no.md)
"""
    bundle = CompileBundleDraft.model_validate(
        {
            "pages": [
                _page(
                    1,
                    "Topic",
                    update_uri=uri,
                    path_hint=None,
                    tags=[" stable ", "stable", "new"],
                    body_markdown=(
                        "New body\n\n# Citations\n\n"
                        "[9] [Detail](viking://resources/source/detail.md)"
                    ),
                )
            ]
        }
    )
    rendered = WikiRenderer().render(
        bundle=bundle,
        target_uri="viking://resources/wiki",
        source_roots={"src_1": "viking://resources/source"},
        catalog_uris={uri},
        existing_raw={uri: old},
    )
    content = rendered.operations[0]["content"]
    assert "custom: keep-me" in content
    assert "tags: [stable, new]" in content
    assert content.count("viking://resources/source/detail.md") == 1
    assert "viking://resources/other/no.md" not in content
    assert "[2] [source](viking://resources/source)" in content


def test_memory_renderer_round_trips_fields_and_only_bumps_changed_version():
    uri = "viking://user/alice/memories/preferences/wiki/topic.md"
    initial_bundle = CompileBundleDraft.model_validate(
        {"pages": [_page(1, "Topic", path_hint="topic.md")]}
    )
    renderer = WikiRenderer()
    created = renderer.render(
        bundle=initial_bundle,
        target_uri="viking://user/alice/memories/preferences/wiki",
        source_roots={"src_1": "viking://resources/source"},
        catalog_uris=set(),
        existing_raw={},
    )
    raw = created.operations[0]["content"]
    memory = MemoryFileUtils.read(raw, uri=uri)
    assert memory.extra_fields["category"] == "concept"
    assert memory.extra_fields["version"] == 1

    update = CompileBundleDraft.model_validate(
        {"pages": [_page(1, "Topic", path_hint=None, update_uri=uri)]}
    )
    unchanged = renderer.render(
        bundle=update,
        target_uri="viking://user/alice/memories/preferences/wiki",
        source_roots={"src_1": "viking://resources/source"},
        catalog_uris={uri},
        existing_raw={uri: raw},
    )
    assert unchanged.unchanged == [uri]
    assert unchanged.operations == []

    changed_bundle = CompileBundleDraft.model_validate(
        {
            "pages": [
                _page(
                    1,
                    "Topic",
                    path_hint=None,
                    update_uri=uri,
                    body_markdown="Changed body",
                )
            ]
        }
    )
    changed = renderer.render(
        bundle=changed_bundle,
        target_uri="viking://user/alice/memories/preferences/wiki",
        source_roots={"src_1": "viking://resources/source"},
        catalog_uris={uri},
        existing_raw={uri: raw},
    )
    assert MemoryFileUtils.read(changed.operations[0]["content"]).extra_fields["version"] == 2


@pytest.mark.asyncio
async def test_submit_tool_rejects_protected_anchor_and_path_collision():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        file_catalog_uris={"viking://resources/wiki/existing.md"},
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )
    context = ToolContext()
    collision = await tool.execute(
        context,
        pages=[_page(1, "Existing", path_hint="existing.md")],
    )
    assert collision.startswith("Error:")
    protected = await tool.execute(
        context,
        pages=[
            _page(1, "One", body_markdown="`Two`"),
            _page(2, "Two"),
        ],
        links=[{"f": 1, "t": 2, "match_text": "Two"}],
    )
    assert protected.startswith("Error:")
    assert tool.bundle is None

    accepted = await tool.execute(context, pages=[], links=[])
    assert "must contain at least one output" in accepted
    assert tool.bundle is None


@pytest.mark.asyncio
async def test_submit_tool_checks_size_before_parsing_okf_artifact():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(output_total_bytes=8),
    )

    result = await tool.execute(
        ToolContext(),
        pages=[],
        files=[
            {
                "path": "large.md",
                "content": "---\ntype: [broken\n---\n",
            }
        ],
    )

    assert result.startswith("Error: Invalid Compile bundle:")
    assert "draft content size limit exceeded" in result


@pytest.mark.asyncio
async def test_submit_tool_raw_update_cannot_remove_okf_from_existing_wiki_page():
    uri = "viking://resources/wiki/existing.md"
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris={uri},
        file_catalog_uris={uri},
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )

    result = await tool.execute(
        ToolContext(),
        pages=[],
        files=[{"update_uri": uri, "content": "# Plain Markdown"}],
    )

    assert result.startswith("Error: Invalid Compile bundle:")
    assert "must retain valid OKF frontmatter" in result


@pytest.mark.asyncio
async def test_submit_tool_allows_standard_index_artifact_update():
    uri = "viking://resources/wiki/index.md"
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        file_catalog_uris={uri},
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )

    result = await tool.execute(
        ToolContext(),
        pages=[],
        files=[{"update_uri": uri, "content": "# Updated index\n"}],
    )

    assert result == "Compile bundle accepted with 0 page(s) and 1 file(s)."


@pytest.mark.asyncio
async def test_submit_tool_allows_safe_dotfile_artifacts():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/project",
        limits=CompileLimits(),
    )

    result = await tool.execute(
        ToolContext(),
        pages=[],
        files=[{"path": ".gitignore", "content": "build/\n"}],
    )

    assert result == "Compile bundle accepted with 0 page(s) and 1 file(s)."


@pytest.mark.asyncio
async def test_submit_tool_resolves_existing_updates_outside_relevant_catalog():
    wiki_uri = "viking://resources/wiki/existing.md"
    artifact_uri = "viking://resources/wiki/PAPER.md"
    resolved = []

    async def resolve(uri):
        resolved.append(uri)
        return uri == wiki_uri

    catalog_uris = set()
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=catalog_uris,
        file_catalog_uris={wiki_uri, artifact_uri},
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        wiki_uri_resolver=resolve,
    )

    accepted = await tool.execute(
        ToolContext(),
        pages=[_page(1, "Existing", update_uri=wiki_uri, path_hint=None)],
    )
    assert accepted.startswith("Compile bundle accepted")
    assert wiki_uri in catalog_uris

    rejected = await tool.execute(
        ToolContext(),
        pages=[],
        files=[{"update_uri": wiki_uri, "content": "# Plain Markdown"}],
    )
    assert "must retain valid OKF frontmatter" in rejected

    artifact = await tool.execute(
        ToolContext(),
        pages=[],
        files=[{"update_uri": artifact_uri, "content": "# Paper"}],
    )
    assert artifact.startswith("Compile bundle accepted")
    assert resolved == [wiki_uri, artifact_uri]


@pytest.mark.asyncio
async def test_submit_tool_reports_all_invalid_links():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )

    result = await tool.execute(
        ToolContext(),
        pages=[
            _page(1, "One", body_markdown="First page body"),
            _page(2, "Two"),
            _page(3, "Three"),
        ],
        links=[
            {"f": 1, "t": 2, "match_text": "Missing One"},
            {"f": 1, "t": 3, "match_text": "Missing Two"},
        ],
    )

    assert result.startswith("Error: Invalid Compile bundle: 2 invalid link(s):")
    assert "links[0] from page 1 has non-linkable anchor 'Missing One'" in result
    assert "links[1] from page 1 has non-linkable anchor 'Missing Two'" in result
    assert tool.bundle is None


@pytest.mark.asyncio
async def test_submit_tool_requires_workspace_paths_for_artifacts():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        require_workspace_files=True,
    )
    result = await tool.execute(
        ToolContext(),
        pages=[],
        files=[
            {
                "path": "logic/claims.md",
                "content": "# Claims",
            },
            {
                "path": "logic/concepts.md",
                "workspace_path": "ara-output/logic/concepts.md",
            },
        ],
    )

    assert result.startswith("Error: Invalid Compile bundle:")
    assert "must be generated with write_file" in result
    assert tool.bundle is None

    single_inline = await tool.execute(
        ToolContext(),
        pages=[],
        files=[
            {
                "path": "PAPER.md",
                "content": "---\ntitle: ARA Paper\nauthors: [Ada]\n---\n\n# Paper",
            }
        ],
    )
    assert single_inline.startswith("Error: Invalid Compile bundle:")
    assert "submitted using workspace_path instead of inline content" in single_inline

    inline_tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )
    rejected = await inline_tool.execute(
        ToolContext(),
        pages=[],
        files=[{"path": "concept.md", "content": "---\ntype: ''\n---\n"}],
    )
    assert rejected.startswith("Error: Invalid Compile bundle:")
    assert 'field "type" must be a non-empty string' in rejected


@pytest.mark.asyncio
async def test_submit_tool_reads_explicit_workspace_file_and_rejects_memory_files():
    class Sandbox:
        async def read_file_bytes(self, path):
            assert path == "ara-output/figure.png"
            return b"PNG"

    class Manager:
        async def get_sandbox(self, session_key):
            assert session_key is not None
            return Sandbox()

    context = ToolContext(
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=Manager(),
    )
    resource_tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )
    params = {
        "pages": [],
        "files": [
            {
                "path": "evidence/figure.png",
                "workspace_path": "ara-output/figure.png",
            }
        ],
    }
    accepted = await resource_tool.execute(context, **params)
    assert not accepted.startswith("Error:")
    assert resource_tool.file_payloads == [b"PNG"]

    memory_tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://user/memories/preferences/wiki",
        limits=CompileLimits(),
    )
    rejected = await memory_tool.execute(
        context,
        pages=[],
        files=[{"path": "trace/tree.yaml", "content": "nodes: []"}],
    )
    assert rejected.startswith("Error:")
    assert "Resource targets" in rejected


@pytest.mark.asyncio
async def test_submit_tool_rejects_non_utf8_declared_okf_workspace_markdown():
    class Sandbox:
        async def read_file_bytes(self, path):
            assert path == "generated/concept.md"
            return b"---\ntype: concept\n---\n\xff"

    class Manager:
        async def get_sandbox(self, session_key):
            return Sandbox()

    context = ToolContext(
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=Manager(),
    )
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )

    rejected = await tool.execute(
        context,
        pages=[],
        files=[
            {
                "path": "concept.md",
                "workspace_path": "generated/concept.md",
            }
        ],
    )

    assert rejected.startswith("Error: Invalid Compile bundle:")
    assert "must be UTF-8" in rejected


@pytest.mark.asyncio
async def test_submit_tool_materializes_workspace_page_body_before_validation():
    class Sandbox:
        async def read_file_bytes(self, path):
            return {
                "__compile_staging__/wiki_pages/concepts/overview.md": b"Read Details next.",
                "__compile_staging__/wiki_pages/details.md": b"Details body.",
            }[path]

    class Manager:
        async def get_sandbox(self, session_key):
            assert session_key is not None
            return Sandbox()

    context = ToolContext(
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=Manager(),
    )
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        require_workspace_pages=True,
    )
    overview = _page(1, "Overview")
    overview.pop("body_markdown")
    overview["path_hint"] = None
    overview["body_workspace_path"] = "__compile_staging__/wiki_pages/concepts/overview.md"
    details = _page(2, "Details")
    details.pop("body_markdown")
    details["body_workspace_path"] = "__compile_staging__/wiki_pages/details.md"

    accepted = await tool.execute(
        context,
        pages=[overview, details],
        files=[],
        links=[{"f": 1, "t": 2, "match_text": "Details"}],
    )
    assert accepted == "Compile bundle accepted with 2 page(s) and 0 file(s)."
    assert tool.bundle is not None
    assert tool.bundle.pages[0].body_markdown == "Read Details next."
    assert tool.bundle.pages[0].body_workspace_path is None
    assert tool.bundle.pages[0].path_hint == "concepts/overview.md"

    rejected = await tool.execute(
        context,
        pages=[_page(1, "Inline")],
        files=[],
    )
    assert "must be generated with write_file" in rejected
    assert tool.bundle is None


@pytest.mark.asyncio
async def test_submit_tool_workspace_update_does_not_derive_path_hint():
    uri = "viking://resources/wiki/existing.md"

    class Sandbox:
        async def read_file_bytes(self, path):
            assert path == "__compile_staging__/wiki_pages/existing.md"
            return b"Updated body."

    class Manager:
        async def get_sandbox(self, session_key):
            assert session_key is not None
            return Sandbox()

    async def resolve(candidate):
        return candidate == uri

    context = ToolContext(
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=Manager(),
    )
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        file_catalog_uris={uri},
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        require_workspace_pages=True,
        wiki_uri_resolver=resolve,
    )
    page = _page(1, "Existing", update_uri=uri, path_hint=None)
    page.pop("body_markdown")
    page["body_workspace_path"] = "__compile_staging__/wiki_pages/existing.md"

    accepted = await tool.execute(context, pages=[page], files=[])

    assert accepted == "Compile bundle accepted with 1 page(s) and 0 file(s)."
    assert tool.bundle is not None
    assert tool.bundle.pages[0].body_markdown == "Updated body."
    assert tool.bundle.pages[0].body_workspace_path is None
    assert tool.bundle.pages[0].path_hint is None


@pytest.mark.asyncio
async def test_submit_tool_rejects_artifact_reused_as_wiki_body():
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        require_workspace_pages=True,
    )
    page = _page(1, "Overview")
    page.pop("body_markdown")
    page["body_workspace_path"] = "./ara-output/PAPER.md"

    result = await tool.execute(
        ToolContext(),
        pages=[page],
        files=[
            {
                "path": "PAPER.md",
                "workspace_path": "ara-output/PAPER.md",
            }
        ],
    )

    assert result.startswith("Error: Invalid Compile bundle:")
    assert "must be temporary files under __compile_staging__/wiki_pages/" in result
    assert tool.bundle is None


@pytest.mark.asyncio
async def test_submit_tool_preserves_generated_skill_artifacts_alongside_staged_wiki_pages():
    files = {
        "skills/compiler/SKILL.md": b"input skill",
        "ara-output/PAPER.md": b"---\ntitle: ARA\nauthors: [Ada]\nyear: 2026\n---\n\n# ARA",
        "ara-output/logic/problem.md": b"# Problem",
        "__compile_staging__/wiki_pages/overview.md": b"# Reader overview",
        "__compile_staging__/tmp/notes.md": b"scratch",
    }

    class Sandbox:
        async def list_dir(self, path):
            directories = {
                ".": [("ara-output", True), ("__compile_staging__", True), ("skills", True)],
                "ara-output": [("PAPER.md", False), ("logic", True)],
                "ara-output/logic": [("problem.md", False)],
                "skills": [("compiler", True)],
                "skills/compiler": [("SKILL.md", False)],
            }
            return directories[path]

        async def read_file_bytes(self, path):
            return files[path]

    class Manager:
        async def get_sandbox(self, session_key):
            assert session_key is not None
            return Sandbox()

    context = ToolContext(
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=Manager(),
    )
    tool = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        require_workspace_files=True,
        require_workspace_pages=True,
        workspace_baseline={"skills/compiler/SKILL.md"},
    )
    page = _page(1, "Overview")
    page.pop("body_markdown")
    page["body_workspace_path"] = "__compile_staging__/wiki_pages/overview.md"

    rejected = await tool.execute(
        context,
        pages=[page],
        files=[],
    )
    assert "generated Skill artifacts are missing from files" in rejected
    assert "ara-output/PAPER.md" in rejected
    assert "ara-output/logic/problem.md" in rejected

    accepted = await tool.execute(
        context,
        pages=[page],
        files=[
            {
                "path": "PAPER.md",
                "workspace_path": "ara-output/PAPER.md",
            },
            {
                "path": "logic/problem.md",
                "workspace_path": "ara-output/logic/problem.md",
            },
        ],
    )
    assert accepted == "Compile bundle accepted with 1 page(s) and 2 file(s)."
    assert tool.bundle is not None
    assert [file.path for file in tool.bundle.files] == [
        "PAPER.md",
        "logic/problem.md",
    ]
    assert tool.file_payloads == [
        files["ara-output/PAPER.md"],
        files["ara-output/logic/problem.md"],
    ]
    assert tool.workspace_files == {
        "ara-output/PAPER.md",
        "ara-output/logic/problem.md",
        "skills/compiler/SKILL.md",
    }


class _EchoTool(Tool):
    @property
    def name(self):
        return "openviking_list"

    @property
    def description(self):
        return "echo"

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, tool_context, **kwargs):
        del tool_context
        return json.dumps(kwargs)


class _PartialMultiReadTool(Tool):
    @property
    def name(self):
        return "openviking_multi_read"

    @property
    def description(self):
        return "partial multi-read"

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, tool_context, **kwargs):
        del tool_context
        first, second = kwargs["uris"]
        return (
            f"Multi-read results for 2 resources (level: read):\n\n"
            f"--- START OF {first} ---\ncontent\n--- END OF {first} ---\n\n"
            f"--- START OF {second} ---\nERROR: unavailable\n--- END OF {second} ---"
        )


@pytest.mark.asyncio
async def test_scoped_tool_requires_and_bounds_openviking_uri():
    wrapped = CompileScopedTool(
        _EchoTool(),
        roots=("viking://resources/source",),
        limits=CompileLimits(),
        result_budget={"bytes": 0},
        budget_lock=__import__("asyncio").Lock(),
    )
    context = ToolContext()
    assert (await wrapped.execute(context)).startswith("Error:")
    assert (await wrapped.execute(context, uri="viking://resources/other")).startswith("Error:")
    assert (
        await wrapped.execute(
            context,
            uri="viking://resources/source/../../other",
        )
    ).startswith("Error:")
    accepted = await wrapped.execute(context, uri="viking://resources/source/child", recursive=True)
    assert '"node_limit": 2000' in accepted


@pytest.mark.asyncio
async def test_scoped_tool_only_records_successful_multi_reads():
    observed = set()
    wrapped = CompileScopedTool(
        _PartialMultiReadTool(),
        roots=("viking://resources/source",),
        limits=CompileLimits(),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        observed_content_uris=observed,
    )
    first = "viking://resources/source/a.md"
    second = "viking://resources/source/b.md"

    await wrapped.execute(ToolContext(), uris=[first, second])

    assert observed == {first}


@pytest.mark.asyncio
async def test_scoped_tool_enforces_per_call_and_total_result_budgets():
    limits = CompileLimits(tool_result_bytes=8, tool_total_result_bytes=12)
    budget = {"bytes": 0}
    wrapped = CompileScopedTool(
        _EchoTool(),
        roots=("viking://resources/source",),
        limits=limits,
        result_budget=budget,
        budget_lock=__import__("asyncio").Lock(),
    )
    context = ToolContext()
    oversized = await wrapped.execute(context, uri="viking://resources/source/child")
    assert oversized.startswith("Error:")
    assert budget["bytes"] == 0


@pytest.mark.asyncio
async def test_structured_wrapper_delegates_to_only_existing_loop_without_fallback():
    registry = ToolRegistry()
    submit = SubmitCompileBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )
    registry.register(submit)
    expected = CompileBundleDraft.model_validate({"pages": []})

    class FakeLoop:
        async def _run_agent_loop(self, **kwargs):
            assert kwargs["tool_registry"] is registry
            assert kwargs["stop_tool_names"] == ["submit_compile_bundle"]
            assert kwargs["allow_final_fallback"] is False
            assert kwargs["inject_write_experience"] is False
            assert kwargs["messages"] == [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ]
            submit.bundle = expected
            return None, None, [], {"total_tokens": 1}, 1

    bundle, tools, usage, iterations = await AgentLoop.run_structured_task(
        FakeLoop(),
        system_prompt="system",
        user_prompt="user",
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        tool_registry=registry,
        openviking_tool_names=set(),
        stop_tool_names=["submit_compile_bundle"],
        openviking_connection={"api_key": "secret"},
    )
    assert bundle is expected
    assert tools == []
    assert usage == {"total_tokens": 1}
    assert iterations == 1


@pytest.mark.asyncio
async def test_request_normalization_uses_default_reason_and_canonical_skill(monkeypatch):
    class Client:
        created = set()
        skill_content = "---\nname: wiki\ndescription: Wiki\n---\nCompile it"

        async def attrs(self, uri):
            if uri == "viking://resources/wiki" and uri not in self.created:
                raise OpenVikingError("missing", code="NOT_FOUND")
            return {"uri": uri.rstrip("/")}

        async def stat(self, uri):
            return {"uri": uri, "isDir": True}

        async def mkdir(self, uri):
            self.created.add(uri)

        async def get_skill(self, name, *, target_uri):
            assert name == "wiki"
            assert target_uri == "viking://agent/skills"
            return {
                "root_uri": "viking://agent/skills/wiki",
                "content": self.skill_content,
            }

        async def close(self):
            return None

    async def create_client(**kwargs):
        assert kwargs["connection"]["api_key"] == "secret"
        return Client()

    monkeypatch.setattr("vikingbot.compile.service.VikingClient.create", create_client)
    service = object.__new__(BotCompileService)
    service.config = None
    service.limits = CompileLimits()
    normalized = await service._normalize_request(
        CompileRequest.model_validate(
            {
                "from": ["viking://resources/source", "viking://resources/source"],
                "to": "viking://resources/wiki",
                "skill": "viking://agent/skills/wiki/SKILL.md",
                "reason": "   ",
            }
        ),
        connection={"api_key": "secret"},
    )
    assert normalized.from_ == ["viking://resources/source"]
    assert normalized.to == "viking://resources/wiki"
    assert normalized.skill == "viking://agent/skills/wiki"
    assert normalized.reason == DEFAULT_COMPILE_REASON

    Client.created.clear()
    Client.skill_content = "---\nname: wiki\n---\nCompile it"
    with pytest.raises(CompileFailure) as raised:
        await service._normalize_request(
            CompileRequest.model_validate(
                {
                    "from": ["viking://resources/source"],
                    "to": "viking://resources/wiki",
                    "skill": "viking://agent/skills/wiki",
                }
            ),
            connection={"api_key": "secret"},
        )
    assert raised.value.code == "SKILL_INVALID"
    assert Client.created == set()


def test_compile_target_accepts_only_exact_skill_namespaces():
    directory = {"isDir": True}

    BotCompileService._validate_target_directory("viking://agent/skills", directory)
    BotCompileService._validate_target_directory("viking://user/alice/skills", directory)
    BotCompileService._validate_target_directory("viking://user/skills", directory)

    with pytest.raises(CompileFailure, match="supported skills namespace"):
        BotCompileService._validate_target_directory(
            "viking://agent/skills/existing-skill", directory
        )
    with pytest.raises(CompileFailure, match="supported skills namespace"):
        BotCompileService._validate_target_directory(
            "viking://agent/legacy-agent/skills", directory
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "target_uri",
    ["viking://agent/skills", "viking://user/alice/skills"],
)
async def test_write_skill_bundle_reuses_add_and_update_skill(existing, target_uri):
    class Client:
        def __init__(self):
            self.called = ""

        async def stat(self, uri):
            assert uri == f"{target_uri}/weekly-report"
            if not existing:
                raise OpenVikingError("missing", code="NOT_FOUND")
            return {"uri": uri, "isDir": True}

        async def get_skill(self, skill_name, *, target_uri: str):
            assert existing
            assert skill_name == "weekly-report"
            return {
                "content": "---\nname: weekly-report\ndescription: Old\n---\nOld",
                "files": [
                    {
                        "path": "assets/keep.bin",
                        "uri": f"{target_uri}/weekly-report/assets/keep.bin",
                        "is_dir": False,
                    },
                    {
                        "path": ".overview.md",
                        "uri": f"{target_uri}/weekly-report/.overview.md",
                        "is_dir": False,
                    },
                ],
            }

        async def download_bytes(self, uri):
            assert uri == f"{target_uri}/weekly-report/assets/keep.bin"
            return b"keep"

        async def add_skill(self, path, *, target_uri, wait, timeout):
            self.called = "add"
            self._assert_package(path, target_uri, wait, timeout)
            return {"root_uri": f"{target_uri}/weekly-report"}

        async def update_skill(self, skill_name, path, *, target_uri, wait, timeout):
            assert skill_name == "weekly-report"
            self.called = "update"
            self._assert_package(path, target_uri, wait, timeout)
            return {"root_uri": f"{target_uri}/weekly-report"}

        @staticmethod
        def _assert_package(path, target_uri, wait, timeout):
            skill_dir = Path(path)
            assert skill_dir.name == "weekly-report"
            assert wait is True
            assert timeout == 30
            assert "name: weekly-report" in (skill_dir / "SKILL.md").read_text()
            assert (skill_dir / "assets" / "logo.bin").read_bytes() == b"\x00\x01"
            if existing:
                assert (skill_dir / "assets" / "keep.bin").read_bytes() == b"keep"
                assert not (skill_dir / ".overview.md").exists()

    bundle = CompileBundleDraft.model_validate(
        {
            "pages": [],
            "files": [
                {
                    "path": "weekly-report/SKILL.md",
                    "content": (
                        "---\nname: weekly-report\ndescription: Weekly report\n---\nWrite it"
                    ),
                },
                {
                    "path": "weekly-report/assets/logo.bin",
                    "workspace_path": "generated/logo.bin",
                },
            ],
        }
    )
    client = Client()
    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()

    action, root_uri = await service._write_skill_bundle(
        client=client,
        target_uri=target_uri,
        bundle=bundle,
        file_payloads=[None, b"\x00\x01"],
        skill_name="weekly-report",
        timeout=30,
    )

    assert action == ("update" if existing else "create")
    assert client.called == ("update" if existing else "add")
    assert root_uri == f"{target_uri}/weekly-report"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_uri",
    ["viking://agent/skills", "viking://user/alice/skills"],
)
async def test_execute_skill_target_skips_recursive_catalog_and_completes(
    monkeypatch, tmp_path: Path, target_uri: str
):
    class TaskConfig:
        def __init__(self):
            self.bot_data_path = tmp_path
            self.workspace_path = tmp_path / "host-workspace"
            self.skills = []
            self.sandbox = SimpleNamespace(mode=None)

        def model_copy(self, *, deep):
            assert deep is True
            return TaskConfig()

    class FakeSandboxManager:
        def __init__(self, config, workspace_parent, workspace_path):
            del config, workspace_path
            self.workspace = workspace_parent / "workspace"
            self.workspace.mkdir(parents=True)

        def get_workspace_path(self, session_key):
            del session_key
            return self.workspace

        async def get_sandbox(self, session_key):
            del session_key
            workspace = self.workspace

            class Sandbox:
                async def write_file(self, path, content):
                    target = workspace / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")

                async def read_file_bytes(self, path):
                    return (workspace / path).read_bytes()

            return Sandbox()

        async def cleanup_session(self, session_key):
            del session_key

    class FakeSkillsLoader:
        def __init__(self, workspace, *, builtin_skills_dir):
            del workspace, builtin_skills_dir

        def load_skills_for_context(self, names):
            assert names == ["skill-creator"]
            return "Create a standards-compliant Skill."

        def _get_skill_meta(self, name):
            assert name == "skill-creator"
            return {}

    class FakeRequestLoop:
        def __init__(self, **kwargs):
            self.sandbox_manager = kwargs["sandbox_manager"]

        async def close_mcp(self):
            return None

    class Client:
        added = ""

        async def get_skill(self, skill_name, *, target_uri):
            assert skill_name == "skill-creator"
            assert target_uri == "viking://agent/skills"
            return {
                "root_uri": "viking://agent/skills/skill-creator",
                "content": (
                    "---\nname: skill-creator\ndescription: Create Skills\n---\nCreate one."
                ),
                "files": [],
            }

        async def stat(self, uri):
            assert uri == f"{target_uri}/weekly-report"
            raise OpenVikingError("missing", code="NOT_FOUND")

        async def add_skill(self, path, *, target_uri, wait, timeout):
            assert wait is True
            assert timeout == 300
            assert "name: weekly-report" in (Path(path) / "SKILL.md").read_text()
            self.added = f"{target_uri}/weekly-report"
            return {"root_uri": self.added}

        async def tree(self, uri, *, node_limit):
            raise AssertionError(
                f"Skill target must not preload recursive catalog: {uri}, {node_limit}"
            )

        async def close(self):
            return None

    class Store:
        def __init__(self, task):
            self.task = task

        async def update(self, task_id, mutate):
            assert task_id == self.task.task_id
            mutate(self.task)
            return self.task

    async def create_client(**kwargs):
        del kwargs
        return client

    async def no_op(*args, **kwargs):
        del args, kwargs

    async def build_sources(client, roots):
        del client, roots
        return []

    async def build_skill_checklist(**kwargs):
        del kwargs
        return CompileSkillChecklist()

    async def plan_outputs(**kwargs):
        del kwargs
        return CompileOutputPlan.model_validate(
            {
                "files": [
                    {
                        "output_id": "skill-md",
                        "description": "Complete generated Skill",
                        "path": "weekly-report/SKILL.md",
                    }
                ],
                "groups": [
                    {
                        "group_id": "skill-package",
                        "output_ids": ["skill-md"],
                    }
                ],
            }
        )

    async def pass_validation(**kwargs):
        del kwargs
        return CompileValidationReport(passed=True)

    async def materialize_outputs(**kwargs):
        sandbox = await kwargs["request_loop"].sandbox_manager.get_sandbox(kwargs["session_key"])
        await sandbox.write_file(
            "weekly-report/SKILL.md",
            "---\nname: weekly-report\ndescription: Weekly report\n---\nWrite it",
        )

    monkeypatch.setattr("vikingbot.compile.service.SandboxManager", FakeSandboxManager)
    monkeypatch.setattr("vikingbot.compile.service.SkillsLoader", FakeSkillsLoader)
    monkeypatch.setattr("vikingbot.compile.service.AgentLoop", FakeRequestLoop)
    monkeypatch.setattr("vikingbot.compile.service.VikingClient.create", create_client)

    host_loop = SimpleNamespace(
        config=TaskConfig(),
        bus=None,
        provider=None,
        workspace=tmp_path,
        model=None,
        temperature=0,
        max_iterations=1,
        memory_window=1,
        brave_api_key=None,
        exa_api_key=None,
        gen_image_model=None,
        exec_config=None,
        _mcp_servers={},
    )
    service = BotCompileService(agent_loop=host_loop)
    request = SanitizedCompileRequest.model_validate(
        {
            "from": ["viking://resources/weekly"],
            "to": target_uri,
            "skill": "viking://agent/skills/skill-creator",
            "reason": "Create a weekly report Skill",
        }
    )
    task = CompileTask(
        task_id="cmp_skill",
        principal_scope="owner",
        sanitized_request=request,
        status="accepted",
        stage="queued",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    client = Client()
    service.store = Store(task)
    monkeypatch.setattr(service, "_materialize_skill", no_op)
    monkeypatch.setattr(
        service,
        "_load_skill_texts",
        lambda skill_dir: {
            "SKILL.md": "---\nname: skill-creator\ndescription: Create Skills\n---\nCreate one."
        },
    )
    monkeypatch.setattr(service, "_check_requirements", no_op)
    monkeypatch.setattr(service, "_build_sources", build_sources)
    monkeypatch.setattr(service, "_build_skill_checklist", build_skill_checklist)
    monkeypatch.setattr(service, "_plan_outputs", plan_outputs)
    monkeypatch.setattr(service, "_audit_output_plan", pass_validation)
    monkeypatch.setattr(service, "_run_output_batches", materialize_outputs)
    monkeypatch.setattr(service, "_audit_candidate", pass_validation)

    await service._execute_task(task.task_id, request, {"api_key": "secret"})

    assert task.status == "completed"
    assert task.result is not None
    assert task.result.created == [f"{target_uri}/weekly-report"]
    assert task.result.updated == []
    assert task.result.page_count == 0
    assert task.result.file_count == 1
    assert task.result.contract_source == "inferred"
    assert client.added == f"{target_uri}/weekly-report"


@pytest.mark.asyncio
async def test_execute_emits_partial_result_when_a_batch_keeps_failing(
    monkeypatch, tmp_path: Path
):
    target_uri = "viking://resources/wiki"
    source_uri = "viking://resources/source/a.md"

    class TaskConfig:
        def __init__(self):
            self.bot_data_path = tmp_path
            self.workspace_path = tmp_path / "host-workspace"
            self.skills = []
            self.sandbox = SimpleNamespace(mode=None)
            self.agents = SimpleNamespace(compile_temperature=0.0, compile_allow_partial=True)

        def model_copy(self, *, deep):
            assert deep is True
            return TaskConfig()

    class FakeSandboxManager:
        def __init__(self, config, workspace_parent, workspace_path):
            del config, workspace_path
            self.workspace = workspace_parent / "workspace"
            self.workspace.mkdir(parents=True)

        def get_workspace_path(self, session_key):
            del session_key
            return self.workspace

        async def get_sandbox(self, session_key):
            del session_key
            workspace = self.workspace

            class Sandbox:
                async def write_file(self, path, content):
                    target = workspace / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(content, bytes):
                        target.write_bytes(content)
                    else:
                        target.write_text(content, encoding="utf-8")

                async def read_file_bytes(self, path):
                    return (workspace / path).read_bytes()

            return Sandbox()

        async def cleanup_session(self, session_key):
            del session_key

    class FakeSkillsLoader:
        def __init__(self, workspace, *, builtin_skills_dir):
            del workspace, builtin_skills_dir

        def load_skills_for_context(self, names):
            del names
            return "Transform sources into a Wiki."

        def _get_skill_meta(self, name):
            del name
            return {}

    class FakeRequestLoop:
        def __init__(self, **kwargs):
            self.sandbox_manager = kwargs["sandbox_manager"]

        async def close_mcp(self):
            return None

    class Client:
        def __init__(self):
            self.operations = None

        async def get_skill(self, skill_name, *, target_uri):
            del skill_name, target_uri
            return {
                "root_uri": "viking://agent/skills/llm-wiki",
                "content": "---\nname: llm-wiki\ndescription: Wiki\n---\nWrite it.",
                "files": [],
            }

        async def read_raw(self, uri):
            del uri
            return ""

        async def download_bytes(self, uri):
            del uri
            return b""

        async def batch_write(self, *, root_uri, operations, wait, timeout):
            del root_uri, wait, timeout
            self.operations = operations
            created = [op["uri"] for op in operations]
            return {"created": created, "updated": [], "unchanged": []}

        @property
        def client(self):
            return self

        async def set_tags(self, uri, tags, *, mode):
            del uri, tags, mode
            return {"tags_updated": True}

        async def close(self):
            return None

    class Store:
        def __init__(self, task):
            self.task = task

        async def update(self, task_id, mutate):
            assert task_id == self.task.task_id
            mutate(self.task)
            return self.task

    async def create_client(**kwargs):
        del kwargs
        return client

    async def no_op(*args, **kwargs):
        del args, kwargs

    async def build_sources(client, roots):
        del client, roots
        return [
            {
                "source_id": "src_1",
                "directory_uri": "viking://resources/source",
                "overview": "Source overview",
                "entries": [{"uri": source_uri, "is_dir": False, "name": "a.md"}],
                "catalog_truncated": False,
            }
        ]

    async def build_catalog(client, target, *, query):
        del client, target, query
        return [], {}

    async def build_skill_checklist(**kwargs):
        del kwargs
        return CompileSkillChecklist()

    plan = _partial_plan(source_uri)

    async def plan_outputs(**kwargs):
        del kwargs
        return plan

    async def pass_validation(**kwargs):
        del kwargs
        return CompileValidationReport(passed=True)

    async def failing_materialize(**kwargs):
        # Write only the first compliant page, then fail so the batch never
        # submits cleanly. Every retry re-raises, exhausting repair attempts.
        sandbox = await kwargs["request_loop"].sandbox_manager.get_sandbox(kwargs["session_key"])
        await sandbox.write_file(
            "__compile_staging__/wiki_pages/concepts/reading-skills.md",
            f"# Reading Skills\n\n[Source]({source_uri})",
        )
        await sandbox.write_file("index.md", "# Index\n")
        raise CompileFailure(
            "AGENT_OUTPUT_INVALID",
            "assessment-methods batch never submitted",
            stage="materializing_outputs",
        )

    monkeypatch.setattr("vikingbot.compile.service.SandboxManager", FakeSandboxManager)
    monkeypatch.setattr("vikingbot.compile.service.SkillsLoader", FakeSkillsLoader)
    monkeypatch.setattr("vikingbot.compile.service.AgentLoop", FakeRequestLoop)
    monkeypatch.setattr("vikingbot.compile.service.VikingClient.create", create_client)

    host_loop = SimpleNamespace(
        config=TaskConfig(),
        bus=None,
        provider=None,
        workspace=tmp_path,
        model=None,
        temperature=0,
        max_iterations=1,
        memory_window=1,
        brave_api_key=None,
        exa_api_key=None,
        gen_image_model=None,
        exec_config=None,
        _mcp_servers={},
    )
    service = BotCompileService(agent_loop=host_loop)
    request = SanitizedCompileRequest.model_validate(
        {
            "from": ["viking://resources/source"],
            "to": target_uri,
            "skill": "viking://agent/skills/llm-wiki",
            "reason": "Compile the source into a Wiki",
        }
    )
    task = CompileTask(
        task_id="cmp_partial",
        principal_scope="owner",
        sanitized_request=request,
        status="accepted",
        stage="queued",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    client = Client()
    service.store = Store(task)
    monkeypatch.setattr(service, "_materialize_skill", no_op)
    monkeypatch.setattr(
        service,
        "_load_skill_texts",
        lambda skill_dir: {"SKILL.md": "---\nname: llm-wiki\ndescription: Wiki\n---\nWrite it."},
    )
    monkeypatch.setattr(service, "_check_requirements", no_op)
    monkeypatch.setattr(service, "_build_sources", build_sources)
    monkeypatch.setattr(service, "_build_catalog", build_catalog)
    monkeypatch.setattr(service, "_build_skill_checklist", build_skill_checklist)
    monkeypatch.setattr(service, "_run_work_items", lambda **kwargs: _empty_receipts())
    monkeypatch.setattr(service, "_plan_outputs", plan_outputs)
    monkeypatch.setattr(service, "_audit_output_plan", pass_validation)
    monkeypatch.setattr(service, "_run_output_batches", failing_materialize)
    monkeypatch.setattr(service, "_audit_candidate", pass_validation)

    await service._execute_task(task.task_id, request, {"api_key": "secret"})

    assert task.status == "completed"
    assert task.result is not None
    # Only the compliant page plus the index survived; the failing page was dropped.
    assert task.result.page_count == 1
    assert task.result.file_count == 1
    assert task.result.warnings
    assert "2/3" in task.result.warnings[0]
    assert any("page-002" in line for line in task.result.warnings)
    # The task retried materialization up to the limit before degrading.
    assert task.result.validation_attempts == service.limits.repair_attempts + 1


async def _empty_receipts():
    return []



@pytest.mark.asyncio
async def test_source_context_builds_bounded_compact_recursive_catalog():
    class Client:
        client = None

        def __init__(self):
            self.client = self

        async def overview(self, uri):
            assert uri == "viking://resources/source"
            return "Source overview"

        async def list_resources(self, *, path, recursive, node_limit):
            assert path == "viking://resources/source"
            assert recursive is True
            assert node_limit == 4
            return [
                {
                    "name": "guide.md",
                    "title": "Readable Guide",
                    "uri": f"{path}/docs/guide.md",
                    "isDir": False,
                    "abstract": "A" * 600,
                },
                {
                    "uri": f"{path}/docs",
                    "isDir": True,
                    "summary": "Documentation",
                },
                {
                    "name": ".overview.md",
                    "uri": f"{path}/.overview.md",
                    "isDir": False,
                },
            ]

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits(source_catalog_entries=3, source_inventory_entries=3)
    sources = await service._build_sources(Client(), ["viking://resources/source"])

    assert sources == [
        {
            "source_id": "src_1",
            "directory_uri": "viking://resources/source",
            "overview": "Source overview",
            "entries": [
                {
                    "name": "guide.md",
                    "title": "Readable Guide",
                    "uri": "viking://resources/source/docs/guide.md",
                    "is_dir": False,
                    "summary": "A" * 500,
                },
                {
                    "name": "docs",
                    "title": "docs",
                    "uri": "viking://resources/source/docs",
                    "is_dir": True,
                    "summary": "Documentation",
                },
            ],
            "catalog_truncated": False,
        }
    ]


def test_compile_work_items_cover_every_source_once_in_bounded_batches():
    service = object.__new__(BotCompileService)
    service.limits = CompileLimits(work_item_sources=2, tool_uri_count=3, work_items=3)
    uris = [f"viking://resources/source/{name}.md" for name in "abcde"]
    sources = [
        {
            "source_id": "src_1",
            "directory_uri": "viking://resources/source",
            "entries": [{"uri": uri, "is_dir": False} for uri in uris],
        }
    ]

    items = service._build_work_items(sources)

    assert [item.work_item_id for item in items] == ["item-001", "item-002", "item-003"]
    assert [uri for item in items for uri in item.source_uris] == uris


@pytest.mark.asyncio
async def test_target_catalog_includes_raw_files_and_marks_wiki_pages():
    class Client:
        def __init__(self):
            self.reads = []
            self.find_call = None

        async def tree(self, uri, *, node_limit):
            assert uri == "viking://resources/ara"
            assert node_limit == CompileLimits().target_inventory_entries + 1
            return [
                {"uri": f"{uri}/Overview.md", "isDir": False, "abstract": "Overview"},
                {"uri": f"{uri}/PAPER.md", "isDir": False},
                {"uri": f"{uri}/broken.md", "isDir": False},
                {"uri": f"{uri}/Long.md", "isDir": False, "size": 2048},
                {"uri": f"{uri}/trace/tree.yaml", "isDir": False},
                {"uri": f"{uri}/figures/chart.png", "isDir": False},
                {"uri": f"{uri}/.overview.md", "isDir": False},
            ]

        async def find(self, query, **kwargs):
            self.find_call = (query, kwargs)
            return {
                "resources": [
                    {
                        "uri": "viking://resources/ara/Overview.md",
                        "abstract": "Relevant overview",
                    },
                    {"uri": "viking://resources/ara/PAPER.md"},
                    {"uri": "viking://resources/ara/Long.md"},
                    {"uri": "viking://resources/ara/trace/tree.yaml"},
                    {"uri": "viking://resources/outside.md"},
                ]
            }

        async def read_raw(self, uri, *, offset=0, limit=-1):
            assert offset == 0
            self.reads.append((uri, limit))
            content = {
                "viking://resources/ara/Overview.md": (
                    "---\ntype: overview\ncustom: kept\n---\n\n# Overview"
                ),
                "viking://resources/ara/PAPER.md": (
                    "---\ntitle: ARA Paper\nauthors: [Ada]\n---\n\n# Paper"
                ),
                "viking://resources/ara/broken.md": "---\ntype:\n---\n",
                "viking://resources/ara/Long.md": (
                    "---\n"
                    + "".join(f"custom_{index}: value\n" for index in range(130))
                    + "type: long_form\n---\n\n# Long"
                ),
            }[uri]
            if limit == -1:
                return content
            return "".join(content.splitlines(keepends=True)[:limit])

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()
    client = Client()
    catalog, inventory = await service._build_catalog(
        client,
        "viking://resources/ara",
        query="compile ResNet\nsource overview",
    )

    assert catalog == [
        {
            "uri": "viking://resources/ara/Overview.md",
            "kind": "wiki_page",
            "title": "Overview",
            "type": "overview",
            "summary": "Relevant overview",
            "page_id": 1,
        },
        {
            "uri": "viking://resources/ara/PAPER.md",
            "kind": "file",
            "title": "PAPER.md",
            "type": "",
            "summary": "",
        },
        {
            "uri": "viking://resources/ara/Long.md",
            "kind": "wiki_page",
            "title": "Long",
            "type": "long_form",
            "summary": "",
            "page_id": 2,
        },
        {
            "uri": "viking://resources/ara/trace/tree.yaml",
            "kind": "file",
            "title": "tree.yaml",
            "type": "",
            "summary": "",
        },
    ]
    assert set(inventory) == {
        "viking://resources/ara/Overview.md",
        "viking://resources/ara/PAPER.md",
        "viking://resources/ara/broken.md",
        "viking://resources/ara/Long.md",
        "viking://resources/ara/trace/tree.yaml",
        "viking://resources/ara/figures/chart.png",
    }
    assert client.find_call == (
        "compile ResNet\nsource overview",
        {
            "target_uri": "viking://resources/ara",
            "context_type": "resource",
            "limit": CompileLimits().target_catalog_pages,
        },
    )
    assert client.reads.count(("viking://resources/ara/Long.md", 128)) == 1
    assert client.reads.count(("viking://resources/ara/Long.md", -1)) == 1
    assert {uri for uri, _ in client.reads} == {
        "viking://resources/ara/Overview.md",
        "viking://resources/ara/PAPER.md",
        "viking://resources/ara/Long.md",
    }


@pytest.mark.asyncio
async def test_target_catalog_search_failure_keeps_collision_inventory():
    class Client:
        async def tree(self, uri, *, node_limit):
            return [{"uri": f"{uri}/existing.md", "isDir": False, "size": 10}]

        async def find(self, query, **kwargs):
            raise RuntimeError("index unavailable")

        async def read_raw(self, uri, *, offset=0, limit=-1):
            raise AssertionError(f"unexpected content read: {uri}")

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()
    catalog, inventory = await service._build_catalog(
        Client(),
        "viking://resources/wiki",
        query="relevant",
    )

    assert catalog == []
    assert set(inventory) == {"viking://resources/wiki/existing.md"}


@pytest.mark.asyncio
async def test_target_catalog_keeps_standard_index_and_log_artifacts():
    root = "viking://resources/wiki"

    class Client:
        async def tree(self, uri, *, node_limit):
            assert uri == root
            return [
                {"uri": f"{root}/index.md", "isDir": False},
                {"uri": f"{root}/log.md", "isDir": False},
                {"uri": f"{root}/.overview.md", "isDir": False},
            ]

        async def find(self, query, **kwargs):
            return {
                "resources": [
                    {"uri": f"{root}/index.md"},
                    {"uri": f"{root}/log.md"},
                ]
            }

        async def read_raw(self, uri, *, offset=0, limit=-1):
            return "# Navigation\n"

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()
    catalog, inventory = await service._build_catalog(Client(), root, query="wiki")

    assert set(inventory) == {f"{root}/index.md", f"{root}/log.md"}
    assert [(item["uri"], item["kind"]) for item in catalog] == [
        (f"{root}/index.md", "file"),
        (f"{root}/log.md", "file"),
    ]


@pytest.mark.asyncio
async def test_memory_target_catalog_uses_memory_search_results():
    target = "viking://user/alice/memories/preferences/wiki"
    existing = f"{target}/topic.md"

    class Client:
        async def tree(self, uri, *, node_limit):
            assert uri == target
            return [{"uri": existing, "isDir": False, "abstract": "Existing topic"}]

        async def find(self, query, **kwargs):
            assert query == "topic"
            assert kwargs == {
                "target_uri": target,
                "context_type": "memory",
                "limit": CompileLimits().target_catalog_pages,
            }
            return {"memories": [{"uri": existing, "abstract": "Relevant topic"}]}

        async def read_raw(self, uri, *, offset=0, limit=-1):
            assert uri == existing
            return "---\ntype: concept\n---\n\n# Topic"

    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()
    catalog, inventory = await service._build_catalog(
        Client(),
        target,
        query="topic",
    )

    assert set(inventory) == {existing}
    assert catalog == [
        {
            "uri": existing,
            "kind": "wiki_page",
            "title": "topic",
            "type": "concept",
            "summary": "Relevant topic",
            "page_id": 1,
        }
    ]


class _NamedTool(_EchoTool):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("uri", "expected_path"),
    [
        ("skills/wiki/references/guide.md", "skills/wiki/references/guide.md"),
        ("viking://skills/wiki/references/guide.md", "skills/wiki/references/guide.md"),
    ],
)
async def test_scoped_tool_redirects_skill_workspace_reads(uri, expected_path):
    wrapped = CompileScopedTool(
        _NamedTool("openviking_multi_read"),
        roots=("viking://resources/source",),
        limits=CompileLimits(),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
    )

    result = await wrapped.execute(ToolContext(), uris=[uri])

    assert "read_file" in result
    assert expected_path in result


@pytest.mark.asyncio
async def test_scoped_tool_keeps_normal_scope_errors_and_valid_reads():
    wrapped = CompileScopedTool(
        _NamedTool("openviking_multi_read"),
        roots=("viking://resources/source",),
        limits=CompileLimits(),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
    )

    rejected = await wrapped.execute(ToolContext(), uris=["viking://resources/other/file.md"])
    accepted = await wrapped.execute(ToolContext(), uris=["viking://resources/source/file.md"])

    assert "outside the Compile task scope" in rejected
    assert "read_file" not in rejected
    assert "viking://resources/source/file.md" in accepted


@pytest.mark.parametrize("exec_enabled", [True, False])
def test_compile_registry_has_a_fixed_ara_compatible_tool_set(exec_enabled):
    available = ToolRegistry()
    for name in (
        "read_file",
        "write_file",
        "edit_file",
        "exec",
        "web_search",
        "message",
        "cron",
        "spawn",
        "openviking_list",
        "openviking_search",
        "openviking_grep",
        "openviking_glob",
        "openviking_multi_read",
        "openviking_add_resource",
        "openviking_memory_commit",
    ):
        available.register(_NamedTool(name))
    request_loop = SimpleNamespace(tools=available, config=None)
    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()
    common = {
        "request_loop": request_loop,
        "roots": ("viking://resources/source", "viking://resources/wiki"),
        "target_uri": "viking://resources/wiki",
        "source_ids": {"src_1"},
        "catalog_uris": set(),
    }

    registry, ov_names = service._build_compile_registry(
        **common,
        capabilities=CompileCapabilities(exec_enabled=exec_enabled),
    )
    expected_tools = {
        "read_file",
        "write_file",
        "edit_file",
        "openviking_list",
        "openviking_search",
        "openviking_grep",
        "openviking_glob",
        "openviking_multi_read",
        "submit_compile_bundle",
    }
    if exec_enabled:
        expected_tools.add("exec")
    assert set(registry.tool_names) == expected_tools
    assert ov_names == {
        "openviking_list",
        "openviking_search",
        "openviking_grep",
        "openviking_glob",
        "openviking_multi_read",
    }
    assert registry.tool_names[-1] == "submit_compile_bundle"
    assert all(isinstance(registry.get(name), CompileScopedTool) for name in ov_names)
    submit = registry.get("submit_compile_bundle")
    assert submit.require_workspace_files is True
    assert submit.require_workspace_pages is True
    assert submit.exec_enabled is exec_enabled


def test_compile_validation_registry_is_read_only():
    available = ToolRegistry()
    for name in ("read_file", "write_file", "edit_file", "exec", "openviking_multi_read"):
        available.register(_NamedTool(name))
    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()

    observed = set()
    registry, _ = service._build_compile_registry(
        SimpleNamespace(tools=available, config=None),
        roots=("viking://resources/source",),
        target_uri="viking://resources/output",
        source_ids={"src_1"},
        catalog_uris=set(),
        capabilities=CompileCapabilities(exec_enabled=True),
        submission_tool=SubmitCompileValidationTool(),
        observed_workspace_paths=observed,
        read_only=True,
    )

    assert set(registry.tool_names) == {
        "read_file",
        "read_workspace_files",
        "submit_compile_validation",
    }
    assert isinstance(registry.get("read_file"), CompileReadTrackingTool)
    assert isinstance(registry.get("read_workspace_files"), ReadWorkspaceFilesTool)


def test_materialization_registry_includes_batch_read_tool():
    available = ToolRegistry()
    for name in ("read_file", "write_file", "edit_file", "openviking_multi_read"):
        available.register(_NamedTool(name))
    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()

    observed: set[str] = set()
    registry, _ = service._build_compile_registry(
        SimpleNamespace(tools=available, config=None),
        roots=("viking://resources/source",),
        target_uri="viking://resources/output",
        source_ids={"src_1"},
        catalog_uris=set(),
        capabilities=CompileCapabilities(exec_enabled=False),
        submission_tool=_NamedTool("submit_compile_outputs"),
        observed_workspace_paths=observed,
        read_only=False,
    )

    # Materialization can write files yet still batch-read its dependency reports.
    assert "write_file" in registry.tool_names
    assert isinstance(registry.get("read_workspace_files"), ReadWorkspaceFilesTool)
    assert isinstance(registry.get("read_file"), CompileReadTrackingTool)


class _FakeReadFileTool(Tool):
    """Return canned content per path, or an error for missing paths."""

    def __init__(self, contents: dict[str, str]):
        self._contents = contents

    @property
    def name(self):
        return "read_file"

    @property
    def description(self):
        return "fake read"

    @property
    def parameters(self):
        return {"type": "object", "properties": {"path": {"type": "string"}}}

    async def execute(self, tool_context, path=None, **kwargs):
        del tool_context, kwargs
        if path in self._contents:
            return self._contents[path]
        return f"Error: {path} not found"


@pytest.mark.asyncio
async def test_read_workspace_files_batches_reads_and_records_observed():
    observed: set[str] = set()
    tool = ReadWorkspaceFilesTool(
        _FakeReadFileTool(
            {
                "__compile_staging__/source-manifest.json": "{manifest}",
                "__compile_staging__/work/item-001/report.md": "# Report one",
            }
        ),
        limits=CompileLimits(),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        observed_workspace_paths=observed,
    )

    result = await tool.execute(
        ToolContext(),
        paths=[
            "__compile_staging__/source-manifest.json",
            "__compile_staging__/work/item-001/report.md",
            "__compile_staging__/work/item-404/report.md",
        ],
    )

    assert "{manifest}" in result
    assert "# Report one" in result
    assert "Error: __compile_staging__/work/item-404/report.md not found" in result
    # Only the successful reads are recorded so the submit gate is satisfied.
    assert observed == {
        "__compile_staging__/source-manifest.json",
        "__compile_staging__/work/item-001/report.md",
    }


@pytest.mark.asyncio
async def test_read_workspace_files_satisfies_plan_submit_gate():
    required = {
        "__compile_staging__/source-manifest.json",
        "__compile_staging__/work/item-001/report.md",
    }
    observed: set[str] = set()
    reader = ReadWorkspaceFilesTool(
        _FakeReadFileTool({path: "data" for path in required}),
        limits=CompileLimits(),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        observed_workspace_paths=observed,
    )

    await reader.execute(ToolContext(), paths=sorted(required))

    assert not (required - observed)


@pytest.mark.asyncio
async def test_read_workspace_files_rejects_empty_and_over_limit():
    tool = ReadWorkspaceFilesTool(
        _FakeReadFileTool({}),
        limits=CompileLimits(tool_uri_count=2),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        observed_workspace_paths=set(),
    )

    assert (await tool.execute(ToolContext(), paths=[])).startswith("Error:")
    assert (await tool.execute(ToolContext())).startswith("Error:")
    over_limit = await tool.execute(ToolContext(), paths=["a", "b", "c"])
    assert "URI limit exceeded" in over_limit


@pytest.mark.asyncio
async def test_read_workspace_files_enforces_per_call_and_total_budget():
    observed: set[str] = set()
    limits = CompileLimits(tool_result_bytes=8, tool_total_result_bytes=64)
    tool = ReadWorkspaceFilesTool(
        _FakeReadFileTool({"a": "x" * 100}),
        limits=limits,
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        observed_workspace_paths=observed,
    )

    oversized = await tool.execute(ToolContext(), paths=["a"])
    assert "per-call size limit" in oversized
    # Path was still read successfully before the size guard rejected the payload.
    assert observed == {"a"}

    budget = {"bytes": 60}
    total_tool = ReadWorkspaceFilesTool(
        _FakeReadFileTool({"b": "short"}),
        limits=CompileLimits(tool_result_bytes=1024, tool_total_result_bytes=64),
        result_budget=budget,
        budget_lock=asyncio.Lock(),
        observed_workspace_paths=set(),
    )
    exhausted = await total_tool.execute(ToolContext(), paths=["b"])
    assert "tool-result budget exceeded" in exhausted


def _compile_service(
    tmp_path: Path,
    *,
    auth_mode: str,
    backend: SandboxBackend,
    allow_compile_exec: bool | None = None,
    limits: CompileLimits | None = None,
) -> BotCompileService:
    direct_exec_enabled = (
        DirectBackendConfig().allow_compile_exec
        if allow_compile_exec is None
        else allow_compile_exec
    )
    config = SimpleNamespace(
        bot_data_path=tmp_path,
        ov_server=SimpleNamespace(effective_auth_mode=auth_mode),
        sandbox=SimpleNamespace(
            backend=backend,
            backends=SimpleNamespace(
                direct=SimpleNamespace(allow_compile_exec=direct_exec_enabled),
            ),
        ),
    )
    return BotCompileService(
        agent_loop=SimpleNamespace(config=config),
        limits=limits,
    )


def _compile_request(*, connection: bool = False) -> CompileRequest:
    payload = {
        "from": ["viking://resources/source"],
        "to": "viking://resources/wiki",
        "skill": "viking://agent/skills/wiki",
    }
    if connection:
        payload["openviking_connection"] = {"api_key": "secret"}
    return CompileRequest.model_validate(payload)


def _sanitized_compile_request() -> SanitizedCompileRequest:
    return SanitizedCompileRequest.model_validate(
        {
            "from": ["viking://resources/source"],
            "to": "viking://resources/wiki",
            "reason": "Compile",
            "skill": "viking://agent/skills/wiki",
        }
    )


@pytest.mark.parametrize(
    ("backend", "allow_compile_exec", "expected"),
    [
        (SandboxBackend.DIRECT, False, False),
        (SandboxBackend.DIRECT, True, True),
        (SandboxBackend.SRT, False, True),
        (SandboxBackend.DOCKER, False, True),
        (SandboxBackend.OPENSANDBOX, False, True),
        (SandboxBackend.AIOSANDBOX, False, True),
    ],
)
def test_compile_exec_capability_depends_on_backend(
    tmp_path: Path,
    backend: SandboxBackend,
    allow_compile_exec: bool,
    expected: bool,
):
    service = _compile_service(
        tmp_path,
        auth_mode="dev",
        backend=backend,
        allow_compile_exec=allow_compile_exec,
    )

    assert service._compile_capabilities().exec_enabled is expected


def test_compile_exec_capability_fails_closed_for_unknown_backend(tmp_path: Path):
    service = _compile_service(
        tmp_path,
        auth_mode="dev",
        backend=SandboxBackend.DIRECT,
        allow_compile_exec=True,
    )
    service.config.sandbox.backend = "future-backend"

    assert service._compile_capabilities() == CompileCapabilities(exec_enabled=False)


@pytest.mark.asyncio
async def test_compile_without_exec_rejects_cli_skill_before_sandbox_access(
    tmp_path: Path,
):
    service = object.__new__(BotCompileService)

    class SandboxManager:
        async def get_sandbox(self, session_key):
            raise AssertionError(f"sandbox must not be accessed: {session_key}")

    with pytest.raises(CompileFailure) as raised:
        await service._check_requirements(
            {"requires": {"bins": ["python3"], "env": ["API_TOKEN"]}},
            capabilities=CompileCapabilities(exec_enabled=False),
            sandbox_manager=SandboxManager(),
            session_key=SessionKey(type="compile", channel_id="task", chat_id="task"),
            workspace=tmp_path,
            skill_name="cli-skill",
        )

    assert raised.value.code == "SKILL_CAPABILITY_UNAVAILABLE"
    assert "bin:python3" in str(raised.value)
    assert "env:API_TOKEN" in str(raised.value)
    assert "allow_compile_exec=true" in str(raised.value)


@pytest.mark.asyncio
async def test_compile_without_exec_syncs_ordinary_skill_without_running_commands(
    tmp_path: Path,
):
    service = object.__new__(BotCompileService)
    skill_dir = tmp_path / "skills" / "wiki"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("Write Wiki pages.", encoding="utf-8")

    class Sandbox:
        def __init__(self):
            self.files = {}

        async def write_file(self, path, content):
            self.files[path] = content

        async def execute(self, command):
            raise AssertionError(f"command must not run: {command}")

    sandbox = Sandbox()

    class SandboxManager:
        async def get_sandbox(self, session_key):
            assert session_key.type == "compile"
            return sandbox

    await service._check_requirements(
        {},
        capabilities=CompileCapabilities(exec_enabled=False),
        sandbox_manager=SandboxManager(),
        session_key=SessionKey(type="compile", channel_id="task", chat_id="task"),
        workspace=tmp_path,
        skill_name="wiki",
    )

    assert sandbox.files == {"skills/wiki/SKILL.md": "Write Wiki pages."}


@pytest.mark.asyncio
async def test_local_dev_compile_uses_config_backed_connection(monkeypatch, tmp_path: Path):
    service = _compile_service(
        tmp_path,
        auth_mode="dev",
        backend=SandboxBackend.DIRECT,
    )
    started = asyncio.Event()
    release = asyncio.Event()
    observed = {}

    async def normalize(request, *, connection):
        del request
        observed["connection"] = connection
        return _sanitized_compile_request()

    async def run_task(task_id, request, connection):
        del task_id, request
        observed["runner_connection"] = connection
        started.set()
        await release.wait()

    monkeypatch.setattr(service, "_normalize_request", normalize)
    monkeypatch.setattr(service, "_run_task", run_task)

    accepted = await service.create_task(_compile_request(), principal_scope="dev")
    await started.wait()

    assert accepted.status == "accepted"
    assert service._compile_capabilities().exec_enabled is False
    assert observed == {"connection": {}, "runner_connection": {}}
    runners = list(service._tasks)
    release.set()
    await asyncio.gather(*runners)
    assert service._admitted_tasks == 0


@pytest.mark.asyncio
async def test_direct_compile_without_exec_is_accepted_by_default(monkeypatch, tmp_path: Path):
    service = _compile_service(
        tmp_path,
        auth_mode="api_key",
        backend=SandboxBackend.DIRECT,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def normalize(request, *, connection):
        del request
        assert connection == {"api_key": "secret"}
        return _sanitized_compile_request()

    async def run_task(*args):
        del args
        started.set()
        await release.wait()

    monkeypatch.setattr(service, "_normalize_request", normalize)
    monkeypatch.setattr(service, "_run_task", run_task)

    accepted = await service.create_task(
        _compile_request(connection=True),
        principal_scope="owner",
    )
    await started.wait()

    assert accepted.status == "accepted"
    assert service._compile_capabilities().exec_enabled is False
    release.set()
    await asyncio.gather(*service._tasks)


@pytest.mark.asyncio
async def test_compile_admission_is_bounded_per_principal_and_globally(monkeypatch, tmp_path: Path):
    limits = CompileLimits(
        accepted_tasks=2,
        accepted_tasks_per_principal=1,
    )
    service = _compile_service(
        tmp_path,
        auth_mode="api_key",
        backend=SandboxBackend.AIOSANDBOX,
        limits=limits,
    )
    release = asyncio.Event()

    async def normalize(request, *, connection):
        del request, connection
        return _sanitized_compile_request()

    async def run_task(*args):
        del args
        await release.wait()

    monkeypatch.setattr(service, "_normalize_request", normalize)
    monkeypatch.setattr(service, "_run_task", run_task)

    await service.create_task(_compile_request(connection=True), principal_scope="alice")
    with pytest.raises(CompileFailure) as per_principal:
        await service.create_task(_compile_request(connection=True), principal_scope="alice")
    await service.create_task(_compile_request(connection=True), principal_scope="bob")
    with pytest.raises(CompileFailure) as global_limit:
        await service.create_task(_compile_request(connection=True), principal_scope="carol")

    assert per_principal.value.code == "RESOURCE_EXHAUSTED"
    assert global_limit.value.code == "RESOURCE_EXHAUSTED"
    runners = list(service._tasks)
    release.set()
    await asyncio.gather(*runners)
    assert service._admitted_tasks == 0
    assert service._admitted_by_principal == {}


@pytest.mark.asyncio
async def test_compile_queue_wait_has_a_deadline(tmp_path: Path):
    service = _compile_service(
        tmp_path,
        auth_mode="api_key",
        backend=SandboxBackend.AIOSANDBOX,
        limits=CompileLimits(concurrent_tasks=1, queue_wait_seconds=0.01),
    )
    request = _sanitized_compile_request()
    task = CompileTask(
        task_id="cmp_queued",
        principal_scope="owner",
        sanitized_request=request,
        status="accepted",
        stage="queued",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    await service.store.create(task)
    await service._semaphore.acquire()
    try:
        await service._run_task(task.task_id, request, {"api_key": "secret"})
    finally:
        service._semaphore.release()

    failed = await service.store.get(task.task_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.stage == "queued"
    assert failed.error is not None
    assert failed.error.code == "DEADLINE_EXCEEDED"
    assert service._target_locks == {}


@pytest.mark.asyncio
async def test_task_store_restart_marks_nonterminal_without_persisting_connection(tmp_path: Path):
    store = CompileTaskStore(tmp_path)
    now = utc_now()
    task = CompileTask(
        task_id="cmp_test",
        principal_scope="owner",
        sanitized_request=SanitizedCompileRequest.model_validate(
            {
                "from": ["viking://resources/source"],
                "to": "viking://resources/wiki",
                "reason": "Compile",
                "skill": "viking://agent/skills/wiki",
            }
        ),
        status="running",
        stage="agent",
        created_at=now,
        updated_at=now,
    )
    await store.create(task)
    assert "api_key" not in (store.root / "cmp_test.json").read_text()
    assert await store.mark_interrupted_failed() == 1
    failed = await store.get("cmp_test")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error is not None and failed.error.code == "BOT_RESTARTED"


@pytest.mark.asyncio
async def test_task_store_missing_lookups_do_not_retain_locks(tmp_path: Path):
    store = CompileTaskStore(tmp_path)

    for index in range(2_000):
        assert await store.get(f"cmp_missing_{index}") is None
    for invalid in ("missing", "cmp_bad/path", "cmp_bad\\path"):
        with pytest.raises(ValueError, match="invalid compile task id"):
            await store.get(invalid)

    assert store._locks == {}
    assert not list(store.root.iterdir())


@pytest.mark.asyncio
async def test_task_store_releases_lock_after_concurrent_users_finish(tmp_path: Path):
    store = CompileTaskStore(tmp_path)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first_user():
        async with store._task_lock("cmp_shared"):
            first_entered.set()
            await release_first.wait()

    async def second_user():
        await first_entered.wait()
        async with store._task_lock("cmp_shared"):
            second_entered.set()

    first = asyncio.create_task(first_user())
    second = asyncio.create_task(second_user())
    await first_entered.wait()
    await asyncio.sleep(0)

    assert not second_entered.is_set()
    assert store._locks["cmp_shared"][1] == 2

    release_first.set()
    await asyncio.gather(first, second)

    assert second_entered.is_set()
    assert store._locks == {}


@pytest.mark.asyncio
async def test_task_store_prunes_expired_and_excess_terminal_records(tmp_path: Path):
    store = CompileTaskStore(tmp_path)
    request = _sanitized_compile_request()
    now = datetime.now(timezone.utc)
    timestamps = {
        "cmp_expired": now - timedelta(days=2),
        "cmp_older": now - timedelta(minutes=2),
        "cmp_newest": now - timedelta(minutes=1),
    }
    for task_id, updated_at in timestamps.items():
        timestamp = updated_at.isoformat().replace("+00:00", "Z")
        await store.create(
            CompileTask(
                task_id=task_id,
                principal_scope="owner",
                sanitized_request=request,
                status="completed",
                stage="completed",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )

    assert await store.prune_terminal(retention_seconds=24 * 60 * 60, max_records=1) == 2
    assert await store.get("cmp_expired") is None
    assert await store.get("cmp_older") is None
    assert await store.get("cmp_newest") is not None


@pytest.mark.asyncio
async def test_task_owner_isolation_and_skill_snapshot_sync(tmp_path: Path):
    store = CompileTaskStore(tmp_path)
    now = utc_now()
    task = CompileTask(
        task_id="cmp_owner",
        principal_scope="owner",
        sanitized_request=SanitizedCompileRequest.model_validate(
            {
                "from": ["viking://resources/source"],
                "to": "viking://resources/wiki",
                "reason": "Compile",
                "skill": "viking://agent/skills/wiki",
            }
        ),
        status="accepted",
        stage="queued",
        created_at=now,
        updated_at=now,
    )
    await store.create(task)
    service = object.__new__(BotCompileService)
    service.store = store
    service._started = True
    service._start_lock = __import__("asyncio").Lock()
    assert await service.get_task("cmp_owner", principal_scope="other") is None
    assert (await service.get_task("cmp_owner", principal_scope="owner"))["task_id"] == "cmp_owner"

    workspace = tmp_path / "workspace"
    skill_dir = workspace / "skills" / "wiki" / "references"
    skill_dir.mkdir(parents=True)
    (workspace / "skills" / "wiki" / "SKILL.md").write_text("Skill", encoding="utf-8")
    (skill_dir / "guide.md").write_text("Guide", encoding="utf-8")
    (skill_dir / "asset.bin").write_bytes(b"\xff\x00")

    class Sandbox:
        def __init__(self):
            self.files = {}

        async def write_file(self, path, content):
            self.files[path] = content

    sandbox = Sandbox()
    await BotCompileService._sync_skill_snapshot(
        sandbox=sandbox,
        workspace=workspace,
        skill_name="wiki",
    )
    assert sandbox.files == {
        "skills/wiki/SKILL.md": "Skill",
        "skills/wiki/references/guide.md": "Guide",
    }


def _partial_plan(source_uri: str) -> CompileOutputPlan:
    return CompileOutputPlan.model_validate(
        {
            "pages": [
                {
                    "output_id": "page-001",
                    "page_id": 1,
                    "title": "Reading Skills",
                    "page_type": "concept",
                    "summary": "Reusable reading skills.",
                    "source_ids": ["src_1"],
                    "source_uris": [source_uri],
                    "report_ids": ["item-001"],
                    "path_hint": "concepts/reading-skills.md",
                },
                {
                    "output_id": "page-002",
                    "page_id": 2,
                    "title": "Assessment Methods",
                    "page_type": "concept",
                    "summary": "Reusable assessment methods.",
                    "source_ids": ["src_1"],
                    "source_uris": [source_uri],
                    "report_ids": ["item-001"],
                    "path_hint": "concepts/assessment-methods.md",
                },
            ],
            "files": [
                {
                    "output_id": "index",
                    "description": "Navigation index",
                    "report_ids": ["item-001"],
                    "path": "index.md",
                }
            ],
            "links": [
                {
                    "f": 1,
                    "t": 2,
                    "match_text": "Assessment",
                    "link_type": "related_to",
                    "description": "related concepts",
                }
            ],
            "coverage": [
                {
                    "unit_id": "unit-1",
                    "status": "covered",
                    "output_ids": ["page-001", "page-002", "index"],
                }
            ],
            "groups": [
                {
                    "group_id": "wiki-pages",
                    "output_ids": ["page-001", "page-002"],
                },
                {
                    "group_id": "navigation",
                    "output_ids": ["index"],
                    "depends_on": ["wiki-pages"],
                },
            ],
        }
    )


def _partial_service():
    service = object.__new__(BotCompileService)
    service.limits = CompileLimits()
    service.config = SimpleNamespace(agents=SimpleNamespace(compile_allow_partial=True))
    return service


def _partial_manager(files: dict[str, bytes]):
    class Sandbox:
        async def read_file_bytes(self, path):
            return files[path]

    class Manager:
        async def get_sandbox(self, session_key):
            del session_key
            return Sandbox()

    return Manager()


@pytest.mark.asyncio
async def test_collect_written_outputs_skips_missing_uncited_and_frontmatter():
    source_uri = "viking://resources/source/a.md"
    plan = _partial_plan(source_uri)
    files = {
        # Compliant page.
        "__compile_staging__/wiki_pages/concepts/reading-skills.md": (
            f"# Reading Skills\n\n[Source]({source_uri})".encode()
        ),
        # Present but missing the required inline source citation.
        "__compile_staging__/wiki_pages/concepts/assessment-methods.md": (
            b"# Assessment Methods\n\nNo citation."
        ),
        # File output written with YAML frontmatter is fine for a file (no UTF-8 body rules),
        # so a compliant navigation file is retained here.
        "index.md": b"# Index\n",
    }
    service = _partial_service()
    written, skipped = await service._collect_written_outputs(
        plan=plan,
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=_partial_manager(files),
        target_uri="viking://resources/wiki",
    )
    assert written == {"page-001", "index"}
    assert any("page-002" in line and "citation" in line for line in skipped)


@pytest.mark.asyncio
async def test_collect_written_outputs_skips_frontmatter_pages():
    source_uri = "viking://resources/source/a.md"
    plan = _partial_plan(source_uri)
    files = {
        "__compile_staging__/wiki_pages/concepts/reading-skills.md": (
            f"---\ntitle: x\n---\n# Reading Skills\n\n[Source]({source_uri})".encode()
        ),
    }
    service = _partial_service()
    written, skipped = await service._collect_written_outputs(
        plan=plan,
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=_partial_manager(files),
        target_uri="viking://resources/wiki",
    )
    assert "page-001" not in written
    assert any("page-001" in line and "frontmatter" in line for line in skipped)
    assert any("page-002" in line and "not written" in line for line in skipped)


def test_reduce_plan_to_written_prunes_links_coverage_and_groups():
    source_uri = "viking://resources/source/a.md"
    plan = _partial_plan(source_uri)
    service = object.__new__(BotCompileService)
    reduced = service._reduce_plan_to_written(plan, {"page-001", "index"})
    assert {page.output_id for page in reduced.pages} == {"page-001"}
    assert {file.output_id for file in reduced.files} == {"index"}
    # The only link referenced dropped page-002, so it is pruned.
    assert reduced.links == []
    # Coverage keeps only surviving outputs.
    assert reduced.coverage[0].output_ids == ["page-001", "index"]
    # Groups keep only surviving outputs; none becomes empty here.
    grouped = {oid for group in reduced.groups for oid in group.output_ids}
    assert grouped == {"page-001", "index"}


def test_salvage_written_outputs_copies_pages_before_cleanup(tmp_path):
    service = object.__new__(BotCompileService)
    service._salvage_paths = {}
    # Mirror the real nested layout: compile_workspaces/<id>/compile__.../
    workspace_parent = tmp_path / "cmp_test"
    workspace = workspace_parent / "compile__cmp_test__cmp_test"
    wiki_pages = workspace / "__compile_staging__" / "wiki_pages" / "concepts"
    wiki_pages.mkdir(parents=True)
    (wiki_pages / "reading-skills.md").write_text("# Reading Skills\n", encoding="utf-8")
    (workspace / "index.md").write_text("# Index\n", encoding="utf-8")

    service._salvage_written_outputs(
        task_id="cmp_test",
        workspace=workspace,
        workspace_parent=workspace_parent,
    )

    salvage_root = tmp_path / "cmp_test_salvage"
    assert service._salvage_paths["cmp_test"] == salvage_root
    assert (salvage_root / "wiki_pages" / "concepts" / "reading-skills.md").is_file()
    assert (salvage_root / "index.md").is_file()


def test_salvage_written_outputs_noop_when_nothing_written(tmp_path):
    service = object.__new__(BotCompileService)
    service._salvage_paths = {}
    workspace_parent = tmp_path / "cmp_empty"
    workspace = workspace_parent / "compile__cmp_empty__cmp_empty"
    workspace.mkdir(parents=True)

    service._salvage_written_outputs(
        task_id="cmp_empty",
        workspace=workspace,
        workspace_parent=workspace_parent,
    )

    assert "cmp_empty" not in service._salvage_paths
    assert not (tmp_path / "cmp_empty_salvage").exists()


@pytest.mark.asyncio
async def test_try_partial_bundle_assembles_written_outputs_with_warnings():
    source_uri = "viking://resources/source/a.md"
    plan = _partial_plan(source_uri)
    files = {
        "__compile_staging__/wiki_pages/concepts/reading-skills.md": (
            f"# Reading Skills\n\n[Source]({source_uri})".encode()
        ),
        # page-002 never written.
        "index.md": b"# Index\n",
    }
    service = _partial_service()
    partial = await service._try_partial_bundle(
        plan=plan,
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=_partial_manager(files),
        capabilities=CompileCapabilities(exec_enabled=False),
        contract=CompileContract(output="mixed"),
        registry_kwargs={
            "source_ids": {"src_1"},
            "catalog_uris": set(),
            "file_catalog_uris": set(),
            "source_file_uris": {source_uri},
            "target_uri": "viking://resources/wiki",
            "workspace_baseline": None,
            "wiki_uri_resolver": None,
        },
        target_uri="viking://resources/wiki",
    )
    assert partial is not None
    reduced_plan, _submit, bundle, _payloads, warnings = partial
    assert {page.page_id for page in bundle.pages} == {1}
    assert [file.path for file in bundle.files] == ["index.md"]
    assert warnings and "2/3" in warnings[0]
    assert any("page-002" in line for line in warnings)


@pytest.mark.asyncio
async def test_try_partial_bundle_returns_none_when_disabled_or_empty():
    source_uri = "viking://resources/source/a.md"
    plan = _partial_plan(source_uri)
    # Nothing written -> None even when allowed.
    service = _partial_service()
    empty = await service._try_partial_bundle(
        plan=plan,
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=_partial_manager({}),
        capabilities=CompileCapabilities(exec_enabled=False),
        contract=CompileContract(output="mixed"),
        registry_kwargs={
            "source_ids": {"src_1"},
            "catalog_uris": set(),
            "file_catalog_uris": set(),
            "source_file_uris": {source_uri},
            "target_uri": "viking://resources/wiki",
            "workspace_baseline": None,
            "wiki_uri_resolver": None,
        },
        target_uri="viking://resources/wiki",
    )
    assert empty is None

    # Disabled -> None even when something compliant was written.
    disabled = object.__new__(BotCompileService)
    disabled.limits = CompileLimits()
    disabled.config = SimpleNamespace(agents=SimpleNamespace(compile_allow_partial=False))
    files = {
        "__compile_staging__/wiki_pages/concepts/reading-skills.md": (
            f"# Reading Skills\n\n[Source]({source_uri})".encode()
        ),
        "index.md": b"# Index\n",
    }
    result = await disabled._try_partial_bundle(
        plan=plan,
        session_key=SessionKey(type="compile", channel_id="cmp", chat_id="cmp"),
        sandbox_manager=_partial_manager(files),
        capabilities=CompileCapabilities(exec_enabled=False),
        contract=CompileContract(output="mixed"),
        registry_kwargs={
            "source_ids": {"src_1"},
            "catalog_uris": set(),
            "file_catalog_uris": set(),
            "source_file_uris": {source_uri},
            "target_uri": "viking://resources/wiki",
            "workspace_baseline": None,
            "wiki_uri_resolver": None,
        },
        target_uri="viking://resources/wiki",
    )
    assert result is None
