# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import json

import pytest

from openviking.session.memory.case_aggregation import (
    CASE_PENDING_SOURCES_FIELD,
    CASE_SOURCE_IDS_FIELD,
    CaseIdentity,
    CaseIdentityComparison,
    case_comparison_has_identity_drift,
    case_generalization_post_validation,
    case_generalization_violations,
    fallback_case_identity,
    generalize_case_year_literals,
    normalize_case_status,
    prepare_case_operation,
    score_case_comparison,
    select_case_primary,
    should_compact_case,
)
from openviking.session.memory.dataclass import (
    MemoryFile,
    MemoryOperationSource,
    ResolvedOperation,
    ResolvedOperations,
)
from openviking.session.memory.memory_type_registry import create_default_registry
from openviking.session.memory.memory_updater import ExtractContext
from openviking.session.memory.patch_merge_context_provider import (
    PatchMergePatch,
    candidate_id_for_uri,
)
from openviking.session.memory.schema_model_generator import SchemaModelGenerator
from openviking.session.memory.streaming_memory_updater import (
    MemoryMergePlanError,
    MemoryMergeProposal,
    build_candidate_merge_proposals,
    build_memory_merge_proposals,
    create_memory_merge_plan_model,
    finalize_case_merge_operations,
    normalize_case_merge_plan,
    reconstruct_memory_operations_from_plan,
    repair_missing_case_comparisons,
    sanitize_case_merge_plan_payload,
    validate_case_merge_plan,
)


def _identity(name: str) -> str:
    return CaseIdentity(
        goal=f"complete {name}",
        subject=f"{name} document",
        action_pattern=f"prepare {name}",
        success_boundary=f"{name} is complete and usable",
        context_constraints=["source data is available"],
    ).compact_json()


def _generalized_identity() -> CaseIdentity:
    return CaseIdentity(
        goal="prepare a reusable analytical report",
        subject="structured source data",
        action_pattern="analyze source data and produce a report",
        success_boundary="the requested report is complete and usable",
        context_constraints=["source data is available"],
    )


def _case_file(
    *,
    uri: str = "viking://user/u/memories/cases/report.md",
    source_count: int = 1,
    last_compacted_source_count: int = 0,
) -> MemoryFile:
    return MemoryFile(
        uri=uri,
        memory_type="cases",
        extra_fields={
            "case_name": "report",
            "case_identity": _identity("report"),
            "task_signature": "prepare a reusable report",
            "input": (
                '{"preconditions":["source data exists"],"variable_types":["source data location"]}'
            ),
            "situation": "A report must be prepared from source data.",
            "rubric": '{"criteria":[{"name":"usable","required":true,"weight":1.0}]}',
            "evidence": "Independent sessions require the same report workflow.",
            "case_status": "promoted" if source_count >= 2 else "draft",
            "source_count": source_count,
            "last_compacted_source_count": last_compacted_source_count,
            "last_compacted_version": 2 if last_compacted_source_count else 0,
            CASE_SOURCE_IDS_FIELD: [f"session:old-{index}" for index in range(source_count)],
            "version": 2,
        },
    )


def _case_operation(
    *,
    session_id: str,
    old_file: MemoryFile | None = None,
) -> ResolvedOperation:
    return ResolvedOperation(
        old_memory_file_content=old_file,
        memory_type="cases",
        uris=[old_file.uri if old_file else "viking://user/u/memories/cases/report.md"],
        memory_fields={
            "case_name": "report",
            "case_identity": _identity("report"),
            "task_signature": "prepare a reusable report",
            "input": (
                '{"preconditions":["source data exists"],"variable_types":["source data location"]}'
            ),
            "situation": "A report must be prepared from source data.",
            "rubric": '{"criteria":[{"name":"usable","required":true,"weight":1.0}]}',
            "evidence": f"Evidence from {session_id}.",
        },
        source=MemoryOperationSource(session_id=session_id, extraction_id=session_id),
    )


def _comparison(
    *,
    proposal_id: str = "new:0",
    candidate_id: str = "candidate:one",
    goal: str = "MATCH",
    subject: str = "MATCH",
    action_pattern: str = "MATCH",
    success_boundary: str = "MATCH",
    context_constraints: str = "MATCH",
) -> CaseIdentityComparison:
    return CaseIdentityComparison(
        proposal_id=proposal_id,
        candidate_id=candidate_id,
        goal=goal,
        subject=subject,
        action_pattern=action_pattern,
        success_boundary=success_boundary,
        context_constraints=context_constraints,
    )


def test_case_schema_exposes_identity_but_hides_system_fields_from_extractor():
    schema = create_default_registry().get("cases")
    model = SchemaModelGenerator([schema]).create_flat_data_model(schema)
    fields = model.model_fields

    assert fields["case_identity"].is_required()
    assert "case_status" not in fields
    assert "source_count" not in fields
    assert "last_compacted_source_count" not in fields
    merge_ops = {field.name: field.merge_op.value for field in schema.fields}
    assert merge_ops["case_name"] == "immutable"
    assert merge_ops["case_identity"] == "immutable"
    assert merge_ops["task_signature"] == "replace"
    assert merge_ops["rubric"] == "replace"
    assert "## Rubric" not in schema.content_template
    assert "## Evidence" not in schema.content_template


@pytest.mark.asyncio
async def test_missing_case_comparison_repair_uses_only_compact_identity_context():
    operation = prepare_case_operation(_case_operation(session_id="new-source"))
    schema = create_default_registry().get("cases")
    required = await build_memory_merge_proposals(
        operations=[operation],
        delete_files=[],
        schema=schema,
        extract_context=ExtractContext([]),
    )
    required[0].proposal_id = "new:0"
    required[0].patch.proposal_id = "new:0"
    candidate = _case_file()
    candidate.extra_fields["rubric"] = "sensitive and very large rubric"
    candidate.extra_fields["evidence"] = "large historical evidence"
    candidate_id = candidate_id_for_uri(candidate.uri)
    candidates = build_candidate_merge_proposals({candidate_id: candidate})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}

    class FakeVLM:
        calls = []

        async def get_completion_async(self, **kwargs):
            self.calls.append(kwargs)
            return json.dumps(
                {
                    "case_comparisons": [
                        _comparison(
                            proposal_id="new:0",
                            candidate_id=candidate_id,
                        ).model_dump()
                    ]
                }
            )

    vlm = FakeVLM()
    repaired = await repair_missing_case_comparisons(
        vlm=vlm,
        missing_pairs=[("new:0", candidate_id)],
        all_proposals=all_proposals,
        completion_content=lambda value: value,
    )

    assert len(repaired) == 1
    assert repaired[0].proposal_id == "new:0"
    prompt = "\n".join(str(message["content"]) for message in vlm.calls[0]["messages"])
    assert "case_identity" in prompt
    assert "task_signature" in prompt
    assert "sensitive and very large rubric" not in prompt
    assert "large historical evidence" not in prompt


@pytest.mark.asyncio
async def test_missing_case_comparison_repair_retries_only_unresolved_pairs():
    operation = prepare_case_operation(_case_operation(session_id="new-source"))
    schema = create_default_registry().get("cases")
    required = await build_memory_merge_proposals(
        operations=[operation], delete_files=[], schema=schema, extract_context=ExtractContext([])
    )
    required[0].proposal_id = "new:0"
    required[0].patch.proposal_id = "new:0"
    first = _case_file(uri="viking://user/u/memories/cases/first.md")
    second = _case_file(uri="viking://user/u/memories/cases/second.md")
    candidates = build_candidate_merge_proposals(
        {candidate_id_for_uri(item.uri): item for item in (first, second)}
    )
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    pairs = [("new:0", proposal.proposal_id) for proposal in candidates]

    class FakeVLM:
        calls = []

        async def get_completion_async(self, **kwargs):
            self.calls.append(kwargs)
            requested = json.loads(kwargs["messages"][1]["content"])["pairs"]
            selected = requested[:1] if len(self.calls) == 1 else requested
            return json.dumps(
                {
                    "case_comparisons": [
                        _comparison(
                            proposal_id=item["proposal_id"],
                            candidate_id=item["candidate_id"],
                        ).model_dump()
                        for item in selected
                    ]
                }
            )

    vlm = FakeVLM()
    repaired = await repair_missing_case_comparisons(
        vlm=vlm,
        missing_pairs=pairs,
        all_proposals=all_proposals,
        completion_content=lambda value: value,
    )

    assert {(item.proposal_id, item.candidate_id) for item in repaired} == set(pairs)
    assert len(vlm.calls) == 2
    second_request = json.loads(vlm.calls[1]["messages"][1]["content"])["pairs"]
    assert second_request == [
        {"proposal_id": pairs[1][0], "candidate_id": pairs[1][1]}
    ]


def test_case_merge_payload_ignores_system_managed_and_immutable_fields():
    schema = create_default_registry().get("cases")
    payload = {
        "groups": [
            {
                "proposal_ids": ["new:0"],
                "canonical_proposal_id": "new:0",
                "field_operations": {
                    "task_signature": "generalized task",
                    "case_name": "model_authored_name",
                    "case_identity": _identity("wrong"),
                    "case_status": "promoted",
                    "source_count": 2,
                    "last_compacted_source_count": 2,
                },
            },
            {
                "proposal_ids": [],
            },
        ],
        "delete_proposal_ids": [],
        "case_comparisons": [],
    }

    sanitized = sanitize_case_merge_plan_payload(payload, schema=schema)

    assert len(sanitized["groups"]) == 1
    assert sanitized["groups"][0]["field_operations"] == {"task_signature": "generalized task"}


def test_legacy_case_identity_fallback_does_not_use_rubric_text():
    fallback = fallback_case_identity(
        {
            "case_name": "prepare_report",
            "task_signature": "prepare a reusable report",
            "input": '{"summary":"report"}',
            "situation": "A report is requested.",
            "rubric": (
                '{"description":"HIDDEN BENCHMARK ANSWER",'
                '"criteria":[{"weight":2,"description":"secret"}]}'
            ),
        }
    )

    assert fallback.success_boundary == "Observable completion of: prepare a reusable report"
    assert "HIDDEN" not in fallback.compact_json()
    assert "secret" not in fallback.compact_json()


def test_case_match_score_and_margin_are_deterministic():
    comparison = _comparison(context_constraints="COMPATIBLE")
    assert score_case_comparison(comparison) == pytest.approx(0.985)
    assert select_case_primary([comparison]) == ("candidate:one", pytest.approx(0.985))

    close_second = _comparison(
        candidate_id="candidate:two",
        action_pattern="COMPATIBLE",
    )
    primary, best_score = select_case_primary([comparison, close_second])
    assert primary is None
    assert best_score == pytest.approx(0.985)


def test_case_status_and_identity_drift_are_normalized():
    assert normalize_case_status("PROMOTED") == "promoted"
    assert normalize_case_status("unexpected") == "draft"
    assert not case_comparison_has_identity_drift(_comparison())
    assert case_comparison_has_identity_drift(_comparison(context_constraints="COMPATIBLE"))


def test_case_generalization_rejects_run_values_and_requires_variable_types():
    operation = _case_operation(session_id="new")
    operation.memory_fields.update(
        {
            "case_identity": CaseIdentity(
                goal="build a pricing model with a 15% discount",
                subject="2026 product pricing",
                action_pattern="apply a discount above 1000 units",
                success_boundary="write pricing_final.xlsx",
            ).compact_json(),
            "input": (
                '{"summary":"Forecast 2000 units","preconditions":[],'
                '"parameter_slots":{"discount":"15%"}}'
            ),
            "task_signature": "Build pricing_final.xlsx for 2026",
        }
    )

    violations = case_generalization_violations(operation)

    assert any("15%" in item for item in violations)
    assert any("2026" in item for item in violations)
    assert any("1000 units" in item for item in violations)
    assert any("pricing_final.xlsx" in item for item in violations)
    assert any("input.variable_types" in item for item in violations)
    assert any("parameter_slots" in item for item in violations)


def test_case_promotion_generalizes_exact_years_without_another_llm_call():
    identity = CaseIdentity(
        goal="analyze 2024 source records",
        subject="2024 publication data",
        action_pattern="filter records for 2024",
        success_boundary="deliver the 2024 analysis",
        context_constraints=["the requested year is 2024"],
    )
    generalized_identity, generalized_input = generalize_case_year_literals(
        identity,
        '{"summary":"Analyze 2024 records","preconditions":["2024 data exists"],'
        '"variable_types":[]}',
    )

    assert "2024" not in generalized_identity.compact_json()
    assert "target_year" in generalized_identity.compact_json()
    assert generalized_input is not None
    assert "2024" not in generalized_input
    assert json.loads(generalized_input)["variable_types"] == ["target year"]


def test_case_generalization_retry_drops_only_invalid_case_operation():
    invalid_case = _case_operation(session_id="new")
    invalid_case.memory_fields["case_identity"] = CaseIdentity(
        goal="build a 15% discount model",
        subject="pricing data",
        action_pattern="calculate margin",
        success_boundary="spreadsheet is usable",
    ).compact_json()
    valid_profile = ResolvedOperation(
        memory_type="profile",
        uris=["viking://user/u/memories/profile.md"],
        memory_fields={"content": "Prefers concise reports."},
    )
    operations = ResolvedOperations(
        upsert_operations=[invalid_case, valid_profile],
        delete_file_contents=[],
        errors=[],
    )

    retry = case_generalization_post_validation(operations, 0)
    assert retry.retry is True
    assert retry.include_latest_draft is True
    assert "input.variable_types" in retry.instruction
    assert "parameter_slots" in retry.instruction

    final = case_generalization_post_validation(operations, 1)
    assert final is None
    assert operations.upsert_operations == [valid_profile]
    assert "generalization retry failed" in operations.errors[0]


@pytest.mark.parametrize(
    ("field_name", "label"),
    [
        ("goal", "UNKNOWN"),
        ("subject", "CONFLICT"),
        ("success_boundary", "UNKNOWN"),
        ("action_pattern", "CONFLICT"),
    ],
)
def test_case_unknown_key_or_any_conflict_blocks_auto_merge(field_name: str, label: str):
    comparison = _comparison(**{field_name: label})
    assert score_case_comparison(comparison) is None
    assert select_case_primary([comparison]) == (None, None)


def test_prepare_case_operation_deduplicates_independent_session_sources():
    old_file = _case_file(source_count=2, last_compacted_source_count=2)
    operation = _case_operation(session_id="old-1", old_file=old_file)

    prepared = prepare_case_operation(operation)

    assert prepared.memory_fields["source_count"] == 2
    assert prepared.memory_fields[CASE_SOURCE_IDS_FIELD] == [
        "session:old-0",
        "session:old-1",
    ]
    assert prepared.memory_fields[CASE_PENDING_SOURCES_FIELD] == []
    assert json.loads(prepared.memory_fields["case_identity"])["goal"] == "complete report"


def test_prepare_case_operation_buffers_safe_source_summary_without_rubric():
    prepared = prepare_case_operation(_case_operation(session_id="new-1"))

    pending = prepared.memory_fields[CASE_PENDING_SOURCES_FIELD]
    assert len(pending) == 1
    assert pending[0]["source_id"] == "session:new-1"
    assert pending[0]["identity"]["goal"] == "complete report"
    assert pending[0]["task_signature"] == "prepare a reusable report"
    assert pending[0]["evidence"] == "Evidence from new-1."
    assert "rubric" not in pending[0]
    assert "criteria" not in json.dumps(pending[0], ensure_ascii=False)


def test_case_compaction_policy_uses_second_source_and_later_delta():
    assert should_compact_case(
        source_count=2,
        last_compacted_source_count=0,
        existing_file=_case_file(),
    )
    assert not should_compact_case(
        source_count=4,
        last_compacted_source_count=2,
        existing_file=_case_file(source_count=2, last_compacted_source_count=2),
    )
    assert should_compact_case(
        source_count=5,
        last_compacted_source_count=2,
        existing_file=_case_file(source_count=2, last_compacted_source_count=2),
    )


@pytest.mark.asyncio
async def test_case_plan_requires_code_selected_primary():
    schema = create_default_registry().get("cases")
    operation = prepare_case_operation(_case_operation(session_id="new"))
    required = await build_memory_merge_proposals(
        operations=[operation],
        delete_files=[],
        schema=schema,
        extract_context=ExtractContext([]),
    )
    required[0].proposal_id = "new:0"
    required[0].patch.proposal_id = "new:0"
    candidate_file = _case_file()
    candidate_id = candidate_id_for_uri(candidate_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: candidate_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    complete_fields = {
        "task_signature": "prepare a reusable report",
        "input": (
            '{"preconditions":["source data exists"],"variable_types":["source data location"]}'
        ),
        "situation": "A report must be prepared from source data.",
        "rubric": '{"criteria":[{"name":"usable","required":true,"weight":1.0}]}',
        "evidence": "Two independent sessions support this Case.",
    }
    comparison = _comparison(proposal_id="new:0", candidate_id=candidate_id).model_dump()
    valid_plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0", candidate_id],
                    "canonical_proposal_id": candidate_id,
                    "field_operations": complete_fields,
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [comparison],
        }
    )
    validate_case_merge_plan(
        valid_plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )

    invalid_plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0"],
                    "canonical_proposal_id": "new:0",
                    "field_operations": {},
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [comparison],
        }
    )
    with pytest.raises(MemoryMergePlanError, match="deterministic assignment"):
        validate_case_merge_plan(
            invalid_plan,
            required_proposals=required,
            all_proposals=all_proposals,
        )


@pytest.mark.asyncio
async def test_case_plan_ignores_extra_read_only_candidate_comparisons():
    schema = create_default_registry().get("cases")
    operation = prepare_case_operation(_case_operation(session_id="new"))
    required = await build_memory_merge_proposals(
        operations=[operation],
        delete_files=[],
        schema=schema,
        extract_context=ExtractContext([]),
    )
    required[0].proposal_id = "new:0"
    required[0].patch.proposal_id = "new:0"
    candidate_files = [
        _case_file(uri="viking://user/u/memories/cases/first.md"),
        _case_file(uri="viking://user/u/memories/cases/second.md"),
    ]
    candidates = build_candidate_merge_proposals(
        {candidate_id_for_uri(file.uri): file for file in candidate_files}
    )
    candidate_ids = sorted(proposal.proposal_id for proposal in candidates)
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0"],
                    "canonical_proposal_id": "new:0",
                    "field_operations": {},
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(
                    proposal_id="new:0",
                    candidate_id=candidate_id,
                    goal="CONFLICT",
                ).model_dump()
                for candidate_id in candidate_ids
            ]
            + [
                _comparison(
                    proposal_id=candidate_ids[0],
                    candidate_id=candidate_ids[1],
                ).model_dump()
            ],
        }
    )

    validate_case_merge_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )


@pytest.mark.asyncio
async def test_case_plan_ignores_irrelevant_extra_comparisons():
    schema = create_default_registry().get("cases")
    operation = prepare_case_operation(_case_operation(session_id="new"))
    required = await build_memory_merge_proposals(
        operations=[operation],
        delete_files=[],
        schema=schema,
        extract_context=ExtractContext([]),
    )
    required[0].proposal_id = "new:0"
    required[0].patch.proposal_id = "new:0"
    candidate_file = _case_file()
    candidate_id = candidate_id_for_uri(candidate_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: candidate_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0"],
                    "canonical_proposal_id": "new:0",
                    "field_operations": {},
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(
                    proposal_id="new:0",
                    candidate_id=candidate_id,
                    goal="CONFLICT",
                ).model_dump(),
                _comparison(
                    proposal_id="unrelated:proposal",
                    candidate_id=candidate_id,
                ).model_dump(),
            ],
        }
    )

    validate_case_merge_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )


@pytest.mark.asyncio
async def test_case_plan_still_rejects_missing_required_comparisons():
    schema = create_default_registry().get("cases")
    operation = prepare_case_operation(_case_operation(session_id="new"))
    required = await build_memory_merge_proposals(
        operations=[operation],
        delete_files=[],
        schema=schema,
        extract_context=ExtractContext([]),
    )
    required[0].proposal_id = "new:0"
    required[0].patch.proposal_id = "new:0"
    candidate_file = _case_file()
    candidate_id = candidate_id_for_uri(candidate_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: candidate_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0"],
                    "canonical_proposal_id": "new:0",
                    "field_operations": {},
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(
                    proposal_id="unrelated:proposal",
                    candidate_id=candidate_id,
                ).model_dump()
            ],
        }
    )

    with pytest.raises(MemoryMergePlanError, match="coverage mismatch"):
        validate_case_merge_plan(
            plan,
            required_proposals=required,
            all_proposals=all_proposals,
        )


@pytest.mark.asyncio
async def test_case_plan_accepts_reversed_comparison_orientation():
    schema = create_default_registry().get("cases")
    operations = [
        prepare_case_operation(_case_operation(session_id="first")),
        prepare_case_operation(_case_operation(session_id="second")),
    ]
    required = await build_memory_merge_proposals(
        operations=operations,
        delete_files=[],
        schema=schema,
        extract_context=ExtractContext([]),
    )
    for index, proposal in enumerate(required):
        proposal.proposal_id = f"new:{index}"
        proposal.patch.proposal_id = proposal.proposal_id
    all_proposals = {proposal.proposal_id: proposal for proposal in required}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0", "new:1"],
                    "canonical_proposal_id": "new:0",
                    "generalized_case_identity": _generalized_identity().model_dump(mode="json"),
                    "field_operations": {
                        "task_signature": "prepare a generalized reusable report",
                        "input": (
                            '{"preconditions":["source data exists"],'
                            '"variable_types":["source data location"]}'
                        ),
                        "situation": "Independent sessions require the same report workflow.",
                        "rubric": ('{"criteria":[{"name":"usable","required":true,"weight":1.0}]}'),
                        "evidence": "Two independent sessions support this Case.",
                    },
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(
                    proposal_id="new:0",
                    candidate_id="new:1",
                ).model_dump()
            ],
        }
    )

    validate_case_merge_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )
    merged = await reconstruct_memory_operations_from_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
        schema=schema,
    )
    finalized = finalize_case_merge_operations(
        merged,
        plan=plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )
    fields = finalized.upsert_operations[0].memory_fields

    assert fields["source_count"] == 2
    assert fields["case_status"] == "promoted"
    assert fields["case_identity"] == _generalized_identity().compact_json()


@pytest.mark.asyncio
async def test_case_plan_drops_read_only_candidate_groups():
    schema = create_default_registry().get("cases")
    operation = prepare_case_operation(_case_operation(session_id="new"))
    required = await build_memory_merge_proposals(
        operations=[operation],
        delete_files=[],
        schema=schema,
        extract_context=ExtractContext([]),
    )
    required[0].proposal_id = "new:0"
    required[0].patch.proposal_id = "new:0"
    candidate_file = _case_file()
    candidate_id = candidate_id_for_uri(candidate_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: candidate_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0"],
                    "canonical_proposal_id": "new:0",
                    "field_operations": {},
                },
                {
                    "proposal_ids": [candidate_id],
                    "canonical_proposal_id": candidate_id,
                    "field_operations": {},
                },
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(
                    proposal_id="new:0",
                    candidate_id=candidate_id,
                    goal="CONFLICT",
                ).model_dump()
            ],
        }
    )

    plan = normalize_case_merge_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )

    assert [group.proposal_ids for group in plan.groups] == [["new:0"]]
    validate_case_merge_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )


@pytest.mark.asyncio
async def test_case_plan_normalizes_existing_case_as_canonical():
    schema = create_default_registry().get("cases")
    operation = prepare_case_operation(_case_operation(session_id="new"))
    required = await build_memory_merge_proposals(
        operations=[operation],
        delete_files=[],
        schema=schema,
        extract_context=ExtractContext([]),
    )
    required[0].proposal_id = "new:0"
    required[0].patch.proposal_id = "new:0"
    candidate_file = _case_file()
    candidate_id = candidate_id_for_uri(candidate_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: candidate_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0", candidate_id],
                    "canonical_proposal_id": "new:0",
                    "field_operations": {
                        "task_signature": "prepare a reusable report",
                        "input": (
                            '{"preconditions":["source data exists"],'
                            '"variable_types":["source data location"]}'
                        ),
                        "situation": "A report must be prepared from source data.",
                        "rubric": ('{"criteria":[{"name":"usable","required":true,"weight":1.0}]}'),
                        "evidence": "Two independent sessions support this Case.",
                    },
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(proposal_id="new:0", candidate_id=candidate_id).model_dump()
            ],
        }
    )

    normalized = normalize_case_merge_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )

    assert normalized.groups[0].canonical_proposal_id == candidate_id
    validate_case_merge_plan(
        normalized,
        required_proposals=required,
        all_proposals=all_proposals,
    )


@pytest.mark.asyncio
async def test_case_ordinary_update_only_advances_sources_not_body():
    schema = create_default_registry().get("cases")
    old_file = _case_file(source_count=2, last_compacted_source_count=2)
    operation = prepare_case_operation(_case_operation(session_id="new-3", old_file=old_file))
    required = [
        MemoryMergeProposal(
            proposal_id="new:0",
            patch=PatchMergePatch(
                before_file=old_file,
                after_file=old_file.model_copy(deep=True),
                proposal_id="new:0",
            ),
            operation=operation,
        )
    ]
    candidate_id = candidate_id_for_uri(old_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: old_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0", candidate_id],
                    "canonical_proposal_id": candidate_id,
                    "field_operations": {},
                    "generalized_case_identity": CaseIdentity(
                        goal="this attempted rewrite must be ignored",
                        subject="a different task family",
                        action_pattern="replace the promoted identity",
                        success_boundary="the stored identity is overwritten",
                    ).model_dump(mode="json"),
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(proposal_id="new:0", candidate_id=candidate_id).model_dump()
            ],
        }
    )
    merged = await reconstruct_memory_operations_from_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
        schema=schema,
    )
    finalized = finalize_case_merge_operations(
        merged,
        plan=plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )
    fields = finalized.upsert_operations[0].memory_fields

    assert fields["source_count"] == 3
    assert fields["case_status"] == "promoted"
    assert fields["case_identity"] == old_file.extra_fields["case_identity"]
    assert [item["source_id"] for item in fields[CASE_PENDING_SOURCES_FIELD]] == ["session:new-3"]
    assert not set({"task_signature", "input", "situation", "rubric", "evidence"}) & set(fields)


@pytest.mark.asyncio
async def test_case_ordinary_update_ignores_unscheduled_body_rewrite():
    schema = create_default_registry().get("cases")
    old_file = _case_file(source_count=2, last_compacted_source_count=2)
    operation = prepare_case_operation(_case_operation(session_id="new-3", old_file=old_file))
    required = [
        MemoryMergeProposal(
            proposal_id="new:0",
            patch=PatchMergePatch(
                before_file=old_file,
                after_file=old_file.model_copy(deep=True),
                proposal_id="new:0",
            ),
            operation=operation,
        )
    ]
    candidate_id = candidate_id_for_uri(old_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: old_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0", candidate_id],
                    "canonical_proposal_id": candidate_id,
                    "field_operations": {
                        "task_signature": "an unscheduled rewrite",
                        "input": (
                            '{"preconditions":["rewritten"],'
                            '"variable_types":["rewritten variable"]}'
                        ),
                        "situation": "An unscheduled rewrite.",
                        "rubric": (
                            '{"criteria":[{"name":"rewritten","required":true,"weight":1.0}]}'
                        ),
                        "evidence": "An unscheduled rewrite.",
                    },
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(proposal_id="new:0", candidate_id=candidate_id).model_dump()
            ],
        }
    )
    merged = await reconstruct_memory_operations_from_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
        schema=schema,
    )
    finalized = finalize_case_merge_operations(
        merged,
        plan=plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )
    fields = finalized.upsert_operations[0].memory_fields

    assert fields["source_count"] == 3
    assert fields["last_compacted_source_count"] == 2
    assert [item["source_id"] for item in fields[CASE_PENDING_SOURCES_FIELD]] == ["session:new-3"]
    assert not set({"task_signature", "input", "situation", "rubric", "evidence"}) & set(fields)


@pytest.mark.asyncio
async def test_compatible_identity_waits_for_scheduled_compaction():
    schema = create_default_registry().get("cases")
    old_file = _case_file(source_count=2, last_compacted_source_count=2)
    operation = prepare_case_operation(_case_operation(session_id="new-3", old_file=old_file))
    required = [
        MemoryMergeProposal(
            proposal_id="new:0",
            patch=PatchMergePatch(
                before_file=old_file,
                after_file=old_file.model_copy(deep=True),
                proposal_id="new:0",
            ),
            operation=operation,
        )
    ]
    candidate_id = candidate_id_for_uri(old_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: old_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0", candidate_id],
                    "canonical_proposal_id": candidate_id,
                    "field_operations": {},
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(
                    proposal_id="new:0",
                    candidate_id=candidate_id,
                    context_constraints="COMPATIBLE",
                ).model_dump()
            ],
        }
    )
    merged = await reconstruct_memory_operations_from_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
        schema=schema,
    )

    finalized = finalize_case_merge_operations(
        merged,
        plan=plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )
    fields = finalized.upsert_operations[0].memory_fields

    assert fields["source_count"] == 3
    assert fields["last_compacted_source_count"] == 2
    assert [item["source_id"] for item in fields[CASE_PENDING_SOURCES_FIELD]] == ["session:new-3"]


@pytest.mark.asyncio
async def test_second_source_compaction_rejects_partial_body():
    schema = create_default_registry().get("cases")
    old_file = _case_file(source_count=1, last_compacted_source_count=0)
    operation = prepare_case_operation(_case_operation(session_id="new-2", old_file=old_file))
    required = [
        MemoryMergeProposal(
            proposal_id="new:0",
            patch=PatchMergePatch(
                before_file=old_file,
                after_file=old_file.model_copy(deep=True),
                proposal_id="new:0",
            ),
            operation=operation,
        )
    ]
    candidate_id = candidate_id_for_uri(old_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: old_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0", candidate_id],
                    "canonical_proposal_id": candidate_id,
                    "field_operations": {"evidence": "Only one field was returned."},
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(proposal_id="new:0", candidate_id=candidate_id).model_dump()
            ],
        }
    )
    merged = await reconstruct_memory_operations_from_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
        schema=schema,
    )

    with pytest.raises(MemoryMergePlanError, match="complete replacement fields"):
        finalize_case_merge_operations(
            merged,
            plan=plan,
            required_proposals=required,
            all_proposals=all_proposals,
        )


@pytest.mark.asyncio
async def test_second_source_compaction_rewrites_body_and_clears_pending_sources():
    schema = create_default_registry().get("cases")
    old_file = _case_file(source_count=1, last_compacted_source_count=0)
    operation = prepare_case_operation(_case_operation(session_id="new-2", old_file=old_file))
    required = [
        MemoryMergeProposal(
            proposal_id="new:0",
            patch=PatchMergePatch(
                before_file=old_file,
                after_file=old_file.model_copy(deep=True),
                proposal_id="new:0",
            ),
            operation=operation,
        )
    ]
    candidate_id = candidate_id_for_uri(old_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: old_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0", candidate_id],
                    "canonical_proposal_id": candidate_id,
                    "generalized_case_identity": _generalized_identity().model_dump(mode="json"),
                    "field_operations": {
                        "task_signature": "prepare a generalized reusable report",
                        "input": (
                            '{"preconditions":["source data exists"],'
                            '"variable_types":["source data location"]}'
                        ),
                        "situation": "Independent sessions require the same report workflow.",
                        "rubric": ('{"criteria":[{"name":"usable","required":true,"weight":1.0}]}'),
                        "evidence": "Two independent sessions support this Case.",
                    },
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(proposal_id="new:0", candidate_id=candidate_id).model_dump()
            ],
        }
    )
    merged = await reconstruct_memory_operations_from_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
        schema=schema,
    )
    finalized = finalize_case_merge_operations(
        merged,
        plan=plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )
    fields = finalized.upsert_operations[0].memory_fields

    assert fields["source_count"] == 2
    assert fields["last_compacted_source_count"] == 2
    assert fields["case_status"] == "promoted"
    assert fields["task_signature"] == "prepare a generalized reusable report"
    assert fields["case_identity"] == _generalized_identity().compact_json()
    assert fields[CASE_PENDING_SOURCES_FIELD] == []


@pytest.mark.asyncio
async def test_second_source_compaction_requires_generalized_identity():
    schema = create_default_registry().get("cases")
    old_file = _case_file(source_count=1, last_compacted_source_count=0)
    operation = prepare_case_operation(_case_operation(session_id="new-2", old_file=old_file))
    required = [
        MemoryMergeProposal(
            proposal_id="new:0",
            patch=PatchMergePatch(
                before_file=old_file,
                after_file=old_file.model_copy(deep=True),
                proposal_id="new:0",
            ),
            operation=operation,
        )
    ]
    candidate_id = candidate_id_for_uri(old_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: old_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["new:0", candidate_id],
                    "canonical_proposal_id": candidate_id,
                    "field_operations": {
                        "task_signature": "prepare a generalized reusable report",
                        "input": (
                            '{"preconditions":["source data exists"],'
                            '"variable_types":["source data location"]}'
                        ),
                        "situation": "Independent sessions require the same report workflow.",
                        "rubric": ('{"criteria":[{"name":"usable","required":true,"weight":1.0}]}'),
                        "evidence": "Two independent sessions support this Case.",
                    },
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(
                    proposal_id="new:0",
                    candidate_id=candidate_id,
                ).model_dump()
            ],
        }
    )
    merged = await reconstruct_memory_operations_from_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
        schema=schema,
    )

    with pytest.raises(
        MemoryMergePlanError,
        match="requires generalized_case_identity",
    ):
        finalize_case_merge_operations(
            merged,
            plan=plan,
            required_proposals=required,
            all_proposals=all_proposals,
        )


@pytest.mark.asyncio
async def test_conflicting_target_is_split_into_new_draft_without_deleting_original():
    schema = create_default_registry().get("cases")
    old_file = _case_file(source_count=2, last_compacted_source_count=2)
    operation = prepare_case_operation(_case_operation(session_id="conflict", old_file=old_file))
    required = [
        MemoryMergeProposal(
            proposal_id="conflict:0",
            patch=PatchMergePatch(
                before_file=old_file,
                after_file=old_file.model_copy(deep=True),
                proposal_id="conflict:0",
            ),
            operation=operation,
        )
    ]
    candidate_id = candidate_id_for_uri(old_file.uri)
    candidates = build_candidate_merge_proposals({candidate_id: old_file})
    all_proposals = {proposal.proposal_id: proposal for proposal in [*required, *candidates]}
    model = create_memory_merge_plan_model(schema)
    plan = model.model_validate(
        {
            "groups": [
                {
                    "proposal_ids": ["conflict:0"],
                    "canonical_proposal_id": "conflict:0",
                    "field_operations": {},
                }
            ],
            "delete_proposal_ids": [],
            "case_comparisons": [
                _comparison(
                    proposal_id="conflict:0",
                    candidate_id=candidate_id,
                    goal="CONFLICT",
                ).model_dump()
            ],
        }
    )
    validate_case_merge_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )
    merged = await reconstruct_memory_operations_from_plan(
        plan,
        required_proposals=required,
        all_proposals=all_proposals,
        schema=schema,
    )
    finalized = finalize_case_merge_operations(
        merged,
        plan=plan,
        required_proposals=required,
        all_proposals=all_proposals,
    )

    new_operation = finalized.upsert_operations[0]
    assert new_operation.old_memory_file_content is None
    assert new_operation.memory_fields["case_status"] == "draft"
    assert new_operation.memory_fields[CASE_SOURCE_IDS_FIELD] == ["session:conflict"]
    assert [
        item["source_id"] for item in new_operation.memory_fields[CASE_PENDING_SOURCES_FIELD]
    ] == ["session:conflict"]
    assert "_variant_" in new_operation.uris[0]
    assert finalized.delete_file_contents == []
    assert finalized.link_replacements == {old_file.uri: new_operation.uris[0]}
