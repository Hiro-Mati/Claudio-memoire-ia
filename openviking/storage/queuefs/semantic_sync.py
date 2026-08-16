# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Temp-to-target tree synchronization for semantic ingestion."""

from __future__ import annotations

from typing import Any, Dict, Optional

from openviking.parse.image_rewrite import IMAGE_MAPPINGS_FILENAME, rewrite_image_uris
from openviking.server.identity import RequestContext
from openviking.storage.viking_fs import SyncDiff, get_viking_fs
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


async def sync_semantic_tree(
    root_uri: str,
    target_uri: str,
    ctx: Optional[RequestContext] = None,
    file_change_status: Optional[Dict[str, bool]] = None,
    lock: Optional[Dict[str, Any]] = None,
) -> SyncDiff:
    """Move parsed content into its target and carry parser sidecars."""
    viking_fs = get_viking_fs()
    if not await viking_fs.exists(root_uri, ctx=ctx):
        raise FileNotFoundError(
            f"Semantic source no longer exists; refusing to sync into {target_uri}: {root_uri}"
        )

    target_exists = await viking_fs.exists(target_uri, ctx=ctx)
    diff = await viking_fs.sync_tree(
        root_uri,
        target_uri,
        ctx=ctx,
        file_change_status=file_change_status,
        lease_ref=lock,
        delete_temp_after=False,
    )
    await _rewrite_synced_image_uris(root_uri, target_uri, ctx=ctx, lock=lock)
    if target_exists:
        try:
            await viking_fs.delete_temp(root_uri, ctx=ctx)
        except Exception as error:
            logger.error("[SyncDiff] Failed to delete root directory %s: %s", root_uri, error)
    return diff


async def _rewrite_synced_image_uris(
    root_uri: str,
    target_uri: str,
    ctx: Optional[RequestContext] = None,
    lock: Optional[Dict[str, Any]] = None,
) -> None:
    viking_fs = get_viking_fs()
    root_prefix = root_uri.rstrip("/")
    target_prefix = target_uri.rstrip("/")
    if root_prefix != target_prefix:
        try:
            matches = (await viking_fs.glob("**/*.md", uri=target_prefix, ctx=ctx)).get(
                "matches", []
            )
        except Exception:
            matches = []

        candidate_dirs = set()
        for markdown_uri in matches:
            directory = markdown_uri.rsplit("/", 1)[0]
            while directory == target_prefix or directory.startswith(target_prefix + "/"):
                candidate_dirs.add(directory[len(target_prefix) :].lstrip("/"))
                if directory == target_prefix:
                    break
                directory = directory.rsplit("/", 1)[0]

        for relative_dir in candidate_dirs:
            source = "/".join(
                part for part in (root_prefix, relative_dir, IMAGE_MAPPINGS_FILENAME) if part
            )
            target = "/".join(
                part for part in (target_prefix, relative_dir, IMAGE_MAPPINGS_FILENAME) if part
            )
            try:
                await viking_fs.stat(target, ctx=ctx)
                continue
            except Exception:
                pass
            try:
                content = await viking_fs.read_file(source, ctx=ctx)
                await viking_fs.write_file(target, content, ctx=ctx, lease_ref=lock)
            except Exception:
                continue

    try:
        await rewrite_image_uris(target_uri, ctx=ctx, lease_ref=lock)
    except Exception as error:
        logger.error("[SyncDiff] Failed to rewrite image URIs for %s: %s", target_uri, error)


__all__ = ["sync_semantic_tree"]
