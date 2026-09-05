# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Memory provenance, revert and as-of endpoints (``/api/v1/memory``)."""

from typing import Optional

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel, ConfigDict

from openviking.core.path_variables import resolve_path_variables
from openviking.core.uri_validation import validate_request_viking_uri
from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.service.memory_timeline import MemoryTimelineService

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


def _timeline() -> MemoryTimelineService:
    return MemoryTimelineService(get_service().fs)


@router.get("/provenance")
async def provenance(
    uri: str = Query(..., description="Memory file URI (viking://~/memories/...)"),
    limit: int = Query(50, ge=1, le=500, description="Maximum recorded changes to return"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Recorded changes to a memory file: which session archive added, updated or deleted it."""
    resolved = validate_request_viking_uri(resolve_path_variables(uri), _ctx)
    events = await _timeline().provenance(resolved, _ctx, limit=limit)
    return Response(status="ok", result={"uri": resolved, "events": events})


class RevertRequest(BaseModel):
    """Body for ``POST /api/v1/memory/revert``."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    archive_uri: str


@router.post("/revert")
async def revert(
    request: RevertRequest = Body(...),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Undo the change recorded for ``uri`` in ``archive_uri`` (see provenance)."""
    resolved = validate_request_viking_uri(resolve_path_variables(request.uri), _ctx)
    archive = validate_request_viking_uri(resolve_path_variables(request.archive_uri), _ctx)
    result = await _timeline().revert(resolved, archive, _ctx)
    return Response(status="ok", result=result)


@router.get("/as-of")
async def as_of(
    uri: str = Query(..., description="Memory file URI"),
    at: str = Query(..., description="ISO-8601 instant, e.g. 2026-09-01T12:00:00Z"),
    branch: Optional[str] = Query("main", description="Snapshot branch"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """The memory file as it was in the latest snapshot committed at or before ``at``."""
    resolved = validate_request_viking_uri(resolve_path_variables(uri), _ctx)
    result = await _timeline().as_of(resolved, at, _ctx, branch=branch or "main")
    return Response(status="ok", result=result)
