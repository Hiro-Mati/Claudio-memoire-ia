from types import SimpleNamespace

from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
from openviking.session.memory.experience_validation import (
    build_memory_extraction_post_validation,
    experience_structure_issues,
)


def _experience_body(*, procedure: str, verification: str) -> str:
    return f"""## Situation
- Applies when: a reusable task pattern is present
- Does not apply when: the task does not require this decision
- Evidence binding: authoritative input defines the relationship
- Decision boundary: before generating the persisted result

## Reminder
- Bind the decision to authoritative runtime input.

## Procedure
{procedure}

## Verification
{verification}

## Fallback
- If the source is unavailable, ask for the missing evidence instead of guessing.

## Anti-pattern
- Do not infer values from an unrelated task.
"""


def test_experience_structure_rejects_fixed_sample_counts_and_source_literals():
    content = _experience_body(
        procedure=(
            "1. Test calculations for 2-3 sample entities before generation.\n"
            "2. Save the requested artifact."
        ),
        verification=(
            "- Reopen source.xlsx and compare the persisted output.\n"
            "- Confirm the target remains within 12.5%."
        ),
    )

    issues = experience_structure_issues(content)

    assert any("sampling count" in issue for issue in issues)
    assert any("source-task filename" in issue for issue in issues)


def test_experience_structure_rejects_sample_of_fixed_count():
    content = _experience_body(
        procedure="1. Map runtime fields.\n2. Save the artifact.",
        verification=(
            "- Reopen the persisted result and cross-check a random sample of 3-5 values.\n"
            "- Recompute each selected value from the source."
        ),
    )

    issues = experience_structure_issues(content)

    assert any("sampling count" in issue for issue in issues)


def test_experience_structure_rejects_spot_verify_fixed_count():
    content = _experience_body(
        procedure="1. Map runtime fields.\n2. Save the artifact.",
        verification=(
            "- Spot-verify 2-3 aggregate values directly against the raw source.\n"
            "- Reopen the persisted result and recompute boundary values."
        ),
    )

    issues = experience_structure_issues(content)

    assert any("sampling count" in issue for issue in issues)


def test_normal_extraction_hook_retries_then_drops_invalid_experience(monkeypatch):
    invalid_content = _experience_body(
        procedure="1. Test 3 sample rows.\n2. Save the artifact.",
        verification="- Reopen the source and compare it.\n- Recompute the saved result.",
    )
    monkeypatch.setattr(
        "openviking.session.memory.experience_validation.render_operation_after_file",
        lambda *args, **kwargs: SimpleNamespace(plain_content=lambda: invalid_content),
    )
    provider = SimpleNamespace(
        get_memory_schemas=lambda ctx: [SimpleNamespace(memory_type="experiences")],
        get_extract_context=lambda: None,
    )
    hook = build_memory_extraction_post_validation(context_provider=provider, ctx=None)
    operation = ResolvedOperation(
        memory_type="experiences",
        memory_fields={"experience_name": "reusable_check"},
        uris=["viking://user/default/memories/experiences/reusable_check.md"],
    )
    operations = ResolvedOperations(
        upsert_operations=[operation],
        delete_file_contents=[],
        errors=[],
    )

    retry = hook(operations, 0)
    assert retry is not None and retry.retry is True
    assert "sampling counts" in retry.instruction

    final = hook(operations, 1)
    assert final is None
    assert operations.upsert_operations == []
    assert any("Dropped invalid memory operations" in error for error in operations.errors)
