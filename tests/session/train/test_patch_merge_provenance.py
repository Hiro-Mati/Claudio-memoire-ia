# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from openviking.session.memory.dataclass import (
    MemoryFile,
    ResolvedOperation,
    ResolvedOperations,
    StoredLink,
)
from openviking.session.train import ExperienceSet, PatchMergePolicyOptimizer, PatchSemanticGradient
from openviking.session.train.components.policy_optimizer import _operations_to_plan_items


def _experience_fields(name: str, situation: str) -> dict[str, str]:
    return {
        "experience_name": name,
        "status": "production",
        "trigger_code": "def should_trigger(ctx):\n    return True\n",
        "situation": situation,
        "reminder": "Use the validated procedure.",
        "procedure": "- Follow the validated steps.",
        "anti_pattern": "- Do not skip validation.",
    }


def _gradient(*, index: int, trajectory_uri: str) -> PatchSemanticGradient:
    uri = f"viking://user/u/memories/experiences/source_{index}.md"
    fields = _experience_fields(f"source_{index}", f"source situation {index}")
    return PatchSemanticGradient(
        before_file=None,
        after_file=MemoryFile(
            uri=uri,
            content=f"source content {index}",
            memory_type="experiences",
            extra_fields={"memory_type": "experiences", **fields},
        ),
        base_version=None,
        rationale="test source binding",
        links=[
            StoredLink(
                from_uri=uri,
                to_uri=trajectory_uri,
                link_type="derived_from",
                weight=1.0,
            )
        ],
        confidence=1.0,
    )


def test_rewritten_experiences_use_explicit_source_patch_trajectory_provenance():
    root_uri = "viking://user/u/memories/experiences"
    trajectory_uris = [
        "viking://user/u/memories/trajectories/first.md",
        "viking://user/u/memories/trajectories/second.md",
    ]
    gradients = [
        _gradient(index=index, trajectory_uri=trajectory_uri)
        for index, trajectory_uri in enumerate(trajectory_uris, start=1)
    ]
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                old_memory_file_content=None,
                memory_fields=_experience_fields(
                    "renamed_first",
                    "completely rewritten first output",
                ),
                memory_type="experiences",
                uris=[f"{root_uri}/renamed_first.md"],
                source_patch_ids=[1],
            ),
            ResolvedOperation(
                old_memory_file_content=None,
                memory_fields=_experience_fields(
                    "renamed_second",
                    "completely rewritten second output",
                ),
                memory_type="experiences",
                uris=[f"{root_uri}/renamed_second.md"],
                source_patch_ids=[2],
            ),
        ],
        delete_file_contents=[],
        errors=[],
    )

    items = _operations_to_plan_items(
        operations=operations,
        gradients=gradients,
        policy_set=ExperienceSet(root_uri=root_uri, policies=[]),
        memory_type="experiences",
        schema=PatchMergePolicyOptimizer()._get_schema(),
    )

    assert [[link.to_uri for link in item.links] for item in items] == [
        [trajectory_uris[0]],
        [trajectory_uris[1]],
    ]


def test_synthesized_experience_keeps_all_explicit_source_trajectories():
    root_uri = "viking://user/u/memories/experiences"
    target_uri = f"{root_uri}/synthesized.md"
    trajectory_uris = [
        "viking://user/u/memories/trajectories/first.md",
        "viking://user/u/memories/trajectories/second.md",
    ]
    gradients = [
        _gradient(index=index, trajectory_uri=trajectory_uri)
        for index, trajectory_uri in enumerate(trajectory_uris, start=1)
    ]
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                old_memory_file_content=None,
                memory_fields=_experience_fields(
                    "synthesized",
                    "output synthesized from both source patches",
                ),
                memory_type="experiences",
                uris=[target_uri],
                source_patch_ids=[1, 2],
            )
        ],
        delete_file_contents=[],
        errors=[],
    )

    items = _operations_to_plan_items(
        operations=operations,
        gradients=gradients,
        policy_set=ExperienceSet(root_uri=root_uri, policies=[]),
        memory_type="experiences",
        schema=PatchMergePolicyOptimizer()._get_schema(),
    )

    assert [link.to_uri for link in items[0].links] == trajectory_uris
    assert all(link.from_uri == target_uri for link in items[0].links)


def test_deleted_experience_uses_explicit_source_patch_trajectory_provenance():
    root_uri = "viking://user/u/memories/experiences"
    target_uri = f"{root_uri}/obsolete.md"
    trajectory_uris = [
        "viking://user/u/memories/trajectories/first.md",
        "viking://user/u/memories/trajectories/second.md",
    ]
    gradients = [
        _gradient(index=index, trajectory_uri=trajectory_uri)
        for index, trajectory_uri in enumerate(trajectory_uris, start=1)
    ]
    operations = ResolvedOperations(
        upsert_operations=[],
        delete_file_contents=[
            MemoryFile(
                uri=target_uri,
                content="obsolete content",
                memory_type="experiences",
                extra_fields={"experience_name": "obsolete"},
            )
        ],
        errors=[],
        delete_source_patch_ids={target_uri: [2]},
    )

    items = _operations_to_plan_items(
        operations=operations,
        gradients=gradients,
        policy_set=ExperienceSet(root_uri=root_uri, policies=[]),
        memory_type="experiences",
        schema=PatchMergePolicyOptimizer()._get_schema(),
    )

    assert len(items) == 1
    assert items[0].kind == "delete"
    assert [link.to_uri for link in items[0].links] == [trajectory_uris[1]]
