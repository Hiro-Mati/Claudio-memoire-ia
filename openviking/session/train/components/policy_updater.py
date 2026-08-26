# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""PolicyUpdater component implementations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from openviking.server.error_mapping import is_not_found_error
from openviking.session.memory.dataclass import (
    MemoryFile,
    MemoryTypeSchema,
    ResolvedOperation,
    ResolvedOperations,
    StoredLink,
)
from openviking.session.memory.experience_lifecycle import (
    experience_case_link_uris,
    normalize_experience_status,
)
from openviking.session.memory.memory_type_registry import (
    MemoryTypeRegistry,
    create_default_registry,
)
from openviking.session.memory.memory_updater import (
    MemoryUpdater,
    MemoryVersionConflictError,
    resolve_memory_fields,
)
from openviking.session.memory.utils.memory_file_utils import (
    MemoryFileUtils,
)
from openviking.session.train.domain import (
    Policy,
    PolicyApplyResult,
    PolicyPlanItem,
    PolicySet,
    PolicyUpdatePlan,
)
from openviking.storage.viking_fs import get_viking_fs
from openviking.telemetry import tracer

_EXPERIENCE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_POLICY_APPLY_LOCK_TIMEOUT_SECONDS = 300.0
_POLICY_APPLY_LOCK_MAX_ACQUISITIONS = 3


@dataclass(slots=True)
class DryRunPolicyUpdater:
    """PolicyUpdater that records a plan without writing files.

    Unlike a pure no-op, this updater simulates executable plan items into an
    updated ExperienceSet snapshot, which makes tests and offline review useful
    before enabling a writing updater.
    """

    simulate: bool = True
    registry: MemoryTypeRegistry | None = None

    @tracer("train.policy_updater.dry_run.apply", ignore_result=True, ignore_args=True)
    async def apply(
        self,
        plan: PolicyUpdatePlan,
        policy_set: PolicySet,
        context: Any = None,
        *,
        transaction_handle: Any = None,
    ) -> PolicyApplyResult:
        del transaction_handle
        del context
        updated_policy_set = (
            await _apply_items_to_snapshot(
                plan.items,
                policy_set,
                registry=self.registry or create_default_registry(),
            )
            if self.simulate and plan.items
            else policy_set
        )
        return PolicyApplyResult(
            updated_policy_set=updated_policy_set,
            written_uris=[],
            metadata={
                "dry_run": True,
                "simulated": self.simulate,
                "plan": plan.metadata,
                "item_count": len(plan.items),
            },
        )


@dataclass(slots=True)
class MemoryFilePolicyUpdater:
    """PolicyUpdater that writes policy files via VikingFS.

    It consumes executable ``upsert`` and ``delete`` plan items. The updater
    performs a lightweight base-content guard when ``before_content`` is
    available to avoid blindly overwriting or deleting a diverged policy set
    snapshot.
    """

    viking_fs: Any = None
    vikingdb: Any = None
    registry: MemoryTypeRegistry | None = None

    @tracer("train.policy_updater.memory_file.apply", ignore_result=True, ignore_args=True)
    async def apply(
        self,
        plan: PolicyUpdatePlan,
        policy_set: PolicySet,
        context: Any = None,
        *,
        transaction_handle: Any = None,
    ) -> PolicyApplyResult:
        viking_fs = self.viking_fs or get_viking_fs()
        if viking_fs is None:
            raise RuntimeError("VikingFS is required to apply policy update plans")

        registry = self.registry or create_default_registry()
        updated_policy_set = await _apply_items_to_snapshot(
            plan.items,
            policy_set,
            registry=registry,
        )
        operations, preflight_errors = _plan_to_resolved_operations(
            plan=plan,
            policy_set=policy_set,
            updated_policy_set=updated_policy_set,
            registry=registry,
        )
        if preflight_errors:
            return PolicyApplyResult(
                updated_policy_set=policy_set,
                errors=preflight_errors,
                metadata={
                    "dry_run": False,
                    "item_count": len(plan.items),
                    "preflight_failed": True,
                },
            )

        apply_lease = transaction_handle
        owns_apply_lease = False
        pathlock_client = getattr(viking_fs, "_async_agfs", None)
        allow_unlocked_test_fallback = bool(
            getattr(viking_fs, "_allow_policy_snapshot_fallback", False)
        )
        has_mutations = bool(
            operations.upsert_operations
            or operations.delete_file_contents
            or operations.resolved_links
        )
        if (
            has_mutations
            and not allow_unlocked_test_fallback
            and (pathlock_client is None or getattr(viking_fs, "_uri_to_path", None) is None)
        ):
            raise RuntimeError(
                "MemoryFilePolicyUpdater requires exact-batch path locking for mutations"
            )
        lock_paths = _policy_apply_lock_paths(
            operations,
            viking_fs,
            context,
            registry=registry,
        )
        version_conflicts: list[MemoryVersionConflictError] = []
        try:
            if lock_paths and pathlock_client is not None:
                required_paths = set(lock_paths)
                for acquisition in range(1, _POLICY_APPLY_LOCK_MAX_ACQUISITIONS + 1):
                    apply_lease = await pathlock_client.pathlock_acquire_exact_batch(
                        sorted(required_paths),
                        timeout_secs=_POLICY_APPLY_LOCK_TIMEOUT_SECONDS,
                    )
                    owns_apply_lease = True
                    version_conflicts = await _preflight_operation_versions(
                        operations,
                        viking_fs=viking_fs,
                        ctx=context,
                        allow_snapshot_fallback=allow_unlocked_test_fallback,
                    )
                    if version_conflicts:
                        break
                    expanded_paths = required_paths | set(
                        _policy_preflight_relation_lock_paths(
                            operations,
                            viking_fs=viking_fs,
                            ctx=context,
                        )
                    )
                    if expanded_paths == required_paths:
                        break
                    await pathlock_client.pathlock_release(apply_lease)
                    owns_apply_lease = False
                    apply_lease = transaction_handle
                    required_paths = expanded_paths
                    _clear_operation_precondition_files(operations)
                    if acquisition == _POLICY_APPLY_LOCK_MAX_ACQUISITIONS:
                        raise RuntimeError(
                            "Unable to stabilize policy apply lock coverage after "
                            f"{_POLICY_APPLY_LOCK_MAX_ACQUISITIONS} acquisitions"
                        )
                lock_paths = sorted(required_paths)
            else:
                version_conflicts = await _preflight_operation_versions(
                    operations,
                    viking_fs=viking_fs,
                    ctx=context,
                    allow_snapshot_fallback=allow_unlocked_test_fallback,
                )
            if version_conflicts:
                return PolicyApplyResult(
                    updated_policy_set=policy_set,
                    errors=[str(conflict) for conflict in version_conflicts],
                    metadata={
                        "dry_run": False,
                        "item_count": len(plan.items),
                        "version_conflict": True,
                        "conflicts": [
                            {
                                "uri": conflict.uri,
                                "expected_version": conflict.expected_version,
                                "actual_version": conflict.actual_version,
                                "expected_absent": conflict.expected_absent,
                            }
                            for conflict in version_conflicts
                        ],
                    },
                )

            updater = MemoryUpdater(
                registry=registry,
                vikingdb=self.vikingdb,
                transaction_handle=apply_lease,
                defer_archived_vector_cleanup=True,
            )
            updater._viking_fs = viking_fs
            apply_result = await updater.apply_operations(
                operations,
                context,
                extract_context=None,
                isolation_handler=None,
            )
        finally:
            if owns_apply_lease:
                await pathlock_client.pathlock_release(apply_lease)

        # Index cleanup does not participate in the file/CAS transaction.
        # Run its single batched downstream call after releasing file locks;
        # the archived status and Agent read guard are already authoritative.
        await updater._remove_archived_vectors(apply_result, context)

        errors = [*preflight_errors, *[f"{uri}: {exc}" for uri, exc in apply_result.errors]]
        operation_target_uris = {
            uri for operation in operations.upsert_operations for uri in operation.uris
        } | {memory_file.uri for memory_file in operations.delete_file_contents if memory_file.uri}
        primary_write_errors = [
            (uri, exc) for uri, exc in apply_result.errors if uri in operation_target_uris
        ]
        apply_conflicts = [
            exc for _, exc in apply_result.errors if isinstance(exc, MemoryVersionConflictError)
        ]

        return PolicyApplyResult(
            # A Case backlink cleanup failure must not make callers believe a
            # successfully archived Experience is still promoted.  Only a
            # primary policy-file write failure invalidates the simulated set.
            updated_policy_set=(updated_policy_set if not primary_write_errors else policy_set),
            written_uris=list(apply_result.written_uris + apply_result.edited_uris),
            deleted_uris=list(apply_result.deleted_uris),
            errors=errors,
            metadata={
                "dry_run": False,
                "item_count": len(plan.items),
                "operation_upsert_count": len(operations.upsert_operations),
                "operation_delete_count": len(operations.delete_file_contents),
                "version_conflict": bool(apply_conflicts),
                "apply_lock_path_count": len(lock_paths),
            },
        )


def _policy_apply_lock_paths(
    operations: ResolvedOperations,
    viking_fs: Any,
    ctx: Any,
    *,
    registry: MemoryTypeRegistry,
) -> list[str]:
    """Return one sorted exact-lock batch for all policy side effects."""

    uri_to_path = getattr(viking_fs, "_uri_to_path", None)
    if uri_to_path is None:
        return []
    uris: set[str] = set()
    for op in operations.upsert_operations:
        for uri in op.uris or []:
            if not uri:
                continue
            uris.add(uri)
            schema = registry.get(op.memory_type)
            directory, separator, _ = uri.rstrip("/").rpartition("/")
            if separator and schema is not None and schema.overview_template:
                uris.add(f"{directory}/.overview.md")
            if op.lifecycle_action == "archive":
                old_file = op.old_memory_file_content
                uris.update(_policy_archive_case_uris(old_file, experience_uri=uri))
                if op.archive_replacement_uri:
                    uris.add(op.archive_replacement_uri)
    for memory_file in operations.delete_file_contents:
        if memory_file.uri:
            uris.add(memory_file.uri)
            memory_type = str(
                memory_file.memory_type
                or memory_file.extra_fields.get("memory_type")
                or MemoryUpdater.memory_type_from_uri(memory_file.uri)
                or ""
            )
            schema = registry.get(memory_type)
            directory, separator, _ = memory_file.uri.rstrip("/").rpartition("/")
            if separator and schema is not None and schema.overview_template:
                uris.add(f"{directory}/.overview.md")
    for link in operations.resolved_links:
        if link.from_uri:
            uris.add(link.from_uri)
        if link.to_uri:
            uris.add(link.to_uri)
    return sorted({uri_to_path(uri, ctx=ctx) for uri in uris})


def _policy_preflight_relation_lock_paths(
    operations: ResolvedOperations,
    *,
    viking_fs: Any,
    ctx: Any,
) -> list[str]:
    """Expand archive locks from the files read under the current lease."""

    uri_to_path = getattr(viking_fs, "_uri_to_path", None)
    if uri_to_path is None:
        return []
    case_uris: set[str] = set()
    for op in operations.upsert_operations:
        if op.lifecycle_action != "archive":
            continue
        for experience_uri in op.uris:
            case_uris.update(
                _policy_archive_case_uris(
                    op.precondition_files.get(experience_uri),
                    experience_uri=experience_uri,
                )
            )
    return sorted({uri_to_path(uri, ctx=ctx) for uri in case_uris})


def _policy_archive_case_uris(
    memory_file: MemoryFile | None,
    *,
    experience_uri: str,
) -> set[str]:
    if memory_file is None:
        return set()
    result = experience_case_link_uris(
        memory_file.backlinks,
        experience_uri=experience_uri,
    )
    archived_case_uris = memory_file.extra_fields.get("archived_case_uris", [])
    if not isinstance(archived_case_uris, (list, tuple, set)):
        return result
    result.update(str(case_uri) for case_uri in archived_case_uris if str(case_uri))
    return result


def _clear_operation_precondition_files(operations: ResolvedOperations) -> None:
    for op in operations.upsert_operations:
        op.precondition_files.clear()


async def _preflight_operation_versions(
    operations: ResolvedOperations,
    *,
    viking_fs: Any,
    ctx: Any,
    allow_snapshot_fallback: bool = False,
) -> list[MemoryVersionConflictError]:
    """Validate the whole batch before its first write and cache each read."""

    conflicts: list[MemoryVersionConflictError] = []
    cache: dict[str, MemoryFile | None] = {}
    preflighted_uris: set[str] = set()

    async def _read(uri: str) -> MemoryFile | None:
        if uri in cache:
            return cache[uri]
        try:
            raw = await viking_fs.read_file(uri, ctx=ctx)
        except Exception as exc:
            if not is_not_found_error(exc) and not isinstance(exc, KeyError):
                raise
            cache[uri] = None
            return None
        cache[uri] = MemoryFileUtils.read(raw or "", uri=uri) if raw is not None else None
        return cache[uri]

    for op in operations.upsert_operations:
        for uri in op.uris:
            if uri in preflighted_uris:
                # Preserve the existing ordered-plan behavior for repeated
                # targets: the later operation reads the file produced by the
                # earlier operation instead of being evaluated against the
                # original snapshot a second time.
                op.expected_version = None
                op.expected_absent = False
                op.precondition_files.pop(uri, None)
                continue
            preflighted_uris.add(uri)
            current = await _read(uri)
            if (
                current is None
                and allow_snapshot_fallback
                and op.old_memory_file_content is not None
            ):
                current = op.old_memory_file_content.model_copy(deep=True)
                cache[uri] = current
            op.precondition_files[uri] = current
            try:
                MemoryUpdater._validate_operation_precondition(op, uri, current)
            except MemoryVersionConflictError as conflict:
                conflicts.append(conflict)

    return conflicts


def _policy_body_metadata(
    policy: Policy,
    *,
    schema: MemoryTypeSchema | None = None,
) -> dict[str, Any]:
    if schema is not None and schema.content_template:
        return {
            name: policy.content if name == "content" else policy.metadata.get(name)
            for name in schema.content_field_names()
            if name == "content" or policy.metadata.get(name) is not None
        }
    return {"content": policy.content}


async def _apply_items_to_snapshot(
    items: list[PolicyPlanItem],
    policy_set: PolicySet,
    *,
    registry: MemoryTypeRegistry,
) -> PolicySet:
    policies_by_uri = {policy.uri: policy for policy in policy_set.policies}
    result = list(policy_set.policies)

    for item in items:
        uri = _target_uri(item, policy_set.root_uri)

        if item.kind == "delete":
            existing = policies_by_uri.get(uri) or _find_policy(
                PolicySet(
                    policy_set.root_uri,
                    result,
                    metadata=dict(policy_set.metadata),
                    viking_fs=policy_set.viking_fs,
                    request_context=policy_set.request_context,
                ),
                uri=None,
                name=item.target_name,
            )
            if item.memory_type == "experiences" and existing is not None:
                metadata = dict(existing.metadata)
                metadata.update(
                    {
                        "status": "archived",
                        "promotion_reason": "superseded_or_obsolete",
                    }
                )
                archived = Policy(
                    name=existing.name,
                    uri=existing.uri,
                    version=existing.version + 1,
                    status="archived",
                    content=existing.content,
                    metadata=metadata,
                    links=list(existing.links or []),
                    backlinks=list(existing.backlinks or []),
                )
                result = [archived if policy.uri == existing.uri else policy for policy in result]
                policies_by_uri[existing.uri] = archived
                continue
            remove_uri = existing.uri if existing is not None else uri
            result = [
                policy
                for policy in result
                if policy.uri != remove_uri and policy.name != item.target_name
            ]
            policies_by_uri.pop(remove_uri, None)
            policies_by_uri.pop(uri, None)
            continue

        if item.kind != "upsert" or item.after_content is None:
            continue
        existing = policies_by_uri.get(uri) or _find_policy(
            PolicySet(
                policy_set.root_uri,
                result,
                metadata=dict(policy_set.metadata),
                viking_fs=policy_set.viking_fs,
                request_context=policy_set.request_context,
            ),
            uri=None,
            name=item.target_name,
        )
        metadata = dict(existing.metadata) if existing is not None else {}
        patch_fields = _metadata_patch_fields(item)
        memory_type = item.memory_type or "experiences"
        schema = registry.get(memory_type)
        if schema is not None:
            metadata = await resolve_memory_fields(
                patch_fields,
                schema=schema,
                old_file=MemoryFile(
                    content=existing.content if existing is not None else "",
                    memory_type=memory_type,
                    extra_fields=metadata,
                )
                if existing is not None
                else None,
            )
        else:
            metadata.update(patch_fields)
        metadata.setdefault("memory_type", memory_type)
        metadata["experience_name"] = item.target_name
        if memory_type == "experiences":
            metadata.pop("trigger_code", None)
            status = normalize_experience_status(
                metadata.get("status"),
                default=(
                    normalize_experience_status(existing.status)
                    if existing is not None
                    else "draft"
                ),
            )
            metadata["status"] = status
        else:
            status = existing.status if existing is not None else "draft"
        version = (existing.version + 1) if existing is not None else 1
        updated = Policy(
            name=item.target_name,
            uri=uri,
            version=version,
            status=status,
            content=item.after_content,
            metadata=metadata,
            links=_merge_policy_links(
                list(existing.links or []) if existing is not None else [],
                list(item.links or []),
            ),
            backlinks=list(existing.backlinks or []) if existing is not None else [],
        )
        if existing is None:
            result.append(updated)
        else:
            result = [updated if policy.uri == existing.uri else policy for policy in result]
        policies_by_uri[uri] = updated

    result.sort(key=lambda policy: policy.uri)
    return PolicySet(
        root_uri=policy_set.root_uri,
        policies=result,
        metadata=dict(policy_set.metadata),
        viking_fs=policy_set.viking_fs,
        request_context=policy_set.request_context,
    )


def _metadata_patch_fields(item: PolicyPlanItem) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("patch_metadata", "merge_memory_fields"):
        value = item.metadata.get(key) if isinstance(item.metadata, dict) else None
        if isinstance(value, dict):
            fields.update(
                {
                    field_key: field_value
                    for field_key, field_value in value.items()
                    if field_key not in {"content", "constraint", "trigger_code"}
                }
            )
    return fields


def _merge_policy_links(existing: list[Any], incoming: list[Any]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
    for raw in [*existing, *incoming]:
        if isinstance(raw, StoredLink):
            link = raw.model_dump()
        elif isinstance(raw, dict):
            link = dict(raw)
        else:
            continue
        key = (
            str(link.get("from_uri") or ""),
            str(link.get("to_uri") or ""),
            str(link.get("link_type") or ""),
            link.get("match_text"),
        )
        merged[key] = link
    return list(merged.values())


def _find_policy(
    policy_set: PolicySet,
    *,
    uri: str | None,
    name: str,
) -> Policy | None:
    for policy in policy_set.policies:
        if uri and policy.uri == uri:
            return policy
        if not uri and policy.name == name:
            return policy
    return None


def _target_uri(item: PolicyPlanItem, root_uri: str) -> str:
    if item.target_uri:
        return item.target_uri
    return f"{root_uri.rstrip('/')}/{_safe_experience_filename(item.target_name)}.md"


def _plan_to_resolved_operations(
    *,
    plan: PolicyUpdatePlan,
    policy_set: PolicySet,
    updated_policy_set: PolicySet,
    registry: MemoryTypeRegistry,
) -> tuple[ResolvedOperations, list[str]]:
    upserts: list[ResolvedOperation] = []
    deletes: list[MemoryFile] = []
    links: list[StoredLink] = []
    errors: list[str] = []

    for item in plan.items:
        uri = _target_uri(item, policy_set.root_uri)
        current = _find_policy(policy_set, uri=uri, name=item.target_name)
        if (
            current is not None
            and item.before_content is not None
            and _normalize_guard_content(current.content)
            != _normalize_guard_content(item.before_content)
        ):
            errors.append(
                f"base content mismatch for {item.target_name}: expected gradient before_content"
            )
            continue

        if item.kind == "delete":
            if item.memory_type == "experiences":
                if current is None:
                    errors.append(f"cannot archive missing Experience: {item.target_name}")
                    continue
                updated = _find_policy(updated_policy_set, uri=uri, name=item.target_name)
                if updated is None:
                    errors.append(
                        f"planned archived policy not found after simulation: {item.target_name}"
                    )
                    continue
                upserts.append(
                    _resolved_policy_upsert(
                        item=item,
                        uri=uri,
                        current=current,
                        updated=updated,
                        registry=registry,
                        lifecycle_action="archive",
                    )
                )
                continue
            deletes.append(_policy_or_plan_item_memory_file(item, uri=uri, current=current))
            continue

        if item.kind != "upsert":
            continue
        if item.after_content is None:
            errors.append(f"missing after_content for {item.target_name}")
            continue

        updated = _find_policy(updated_policy_set, uri=uri, name=item.target_name)
        if updated is None:
            errors.append(f"planned policy not found after simulation: {item.target_name}")
            continue

        lifecycle_action = str(item.metadata.get("lifecycle_action") or "") or None
        if (
            item.memory_type == "experiences"
            and updated.status == "archived"
            and (current is None or normalize_experience_status(current.status) != "archived")
        ):
            lifecycle_action = "archive"
        upserts.append(
            _resolved_policy_upsert(
                item=item,
                uri=uri,
                current=current,
                updated=updated,
                registry=registry,
                lifecycle_action=lifecycle_action,
            )
        )
        links.extend(_source_trajectory_links(exp_uri=uri, links=item.links))

    return (
        ResolvedOperations(
            upsert_operations=upserts,
            delete_file_contents=deletes,
            errors=[],
            resolved_links=links,
        ),
        errors,
    )


def _resolved_policy_upsert(
    *,
    item: PolicyPlanItem,
    uri: str,
    current: Policy | None,
    updated: Policy,
    registry: MemoryTypeRegistry,
    lifecycle_action: str | None,
) -> ResolvedOperation:
    superseded_by = list(item.metadata.get("superseded_by") or [])
    replacement_uris = [
        str(candidate)
        for candidate in superseded_by
        if str(candidate).startswith("viking://") and str(candidate) != uri
    ]
    replacement_uri = replacement_uris[0] if len(replacement_uris) == 1 else None
    memory_type = item.memory_type or "experiences"
    return ResolvedOperation(
        old_memory_file_content=(_policy_to_memory_file(current) if current is not None else None),
        memory_fields={
            **dict(updated.metadata),
            **_policy_body_metadata(
                updated,
                schema=registry.get(memory_type),
            ),
            "memory_type": memory_type,
            "experience_name": updated.name,
            "status": updated.status,
        },
        memory_type=memory_type,
        uris=[uri],
        expected_version=(
            item.base_version
            if item.base_version is not None
            else (current.version if current is not None else None)
        ),
        expected_absent=current is None,
        lifecycle_action=lifecycle_action,
        archive_replacement_uri=replacement_uri,
    )


def _policy_or_plan_item_memory_file(
    item: PolicyPlanItem,
    *,
    uri: str,
    current: Policy | None,
) -> MemoryFile:
    if current is not None:
        return _policy_to_memory_file(current)
    return MemoryFile(
        uri=uri,
        content=item.before_content or "",
        memory_type=item.memory_type or "experiences",
        extra_fields={
            "memory_type": item.memory_type or "experiences",
            "experience_name": item.target_name,
            **({"version": item.base_version} if item.base_version is not None else {}),
        },
    )


def _policy_to_memory_file(policy: Policy | None) -> MemoryFile | None:
    if policy is None:
        return None
    return MemoryFile(
        uri=policy.uri,
        content=policy.content,
        links=list(policy.links or []),
        backlinks=list(policy.backlinks or []),
        memory_type="experiences",
        extra_fields={
            **dict(policy.metadata),
            "memory_type": "experiences",
            "experience_name": policy.name,
            "version": policy.version,
            "status": policy.status,
        },
    )


def _source_trajectory_links(
    *,
    exp_uri: str,
    links: list[StoredLink],
) -> list[StoredLink]:
    result: list[StoredLink] = []
    seen: set[tuple[str, str | None]] = set()
    for link in links or []:
        if (
            link.link_type != "derived_from"
            or not link.to_uri
            or "/memories/trajectories/" not in link.to_uri
        ):
            continue
        key = (link.to_uri, link.match_text)
        if key in seen:
            continue
        seen.add(key)
        update = {"from_uri": exp_uri, "match_text": None, "description": ""}
        if not link.created_at:
            update["created_at"] = datetime.now(timezone.utc).isoformat()
        result.append(link.model_copy(update=update))
    return result


def _safe_experience_filename(name: str) -> str:
    filename = _EXPERIENCE_NAME_RE.sub("_", name.strip()).strip("._-")
    return filename or "new_experience"


def _normalize_guard_content(content: str) -> str:
    return content.strip()
