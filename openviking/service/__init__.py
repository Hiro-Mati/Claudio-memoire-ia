# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Service layer for OpenViking.

Provides business logic decoupled from transport layer,
enabling reuse across HTTP Server and CLI.
"""

import importlib

_LAZY_IMPORTS = {
    "OpenVikingService": "openviking.service.core",
    "ComponentStatus": "openviking.service.debug_service",
    "DebugService": "openviking.service.debug_service",
    "SystemStatus": "openviking.service.debug_service",
    "FSService": "openviking.service.fs_service",
    "PackService": "openviking.service.pack_service",
    "RelationService": "openviking.service.relation_service",
    "ResourceService": "openviking.service.resource_service",
    "SearchService": "openviking.service.search_service",
    "SessionService": "openviking.service.session_service",
}


def __getattr__(name: str):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value

__all__ = [
    "OpenVikingService",
    "ComponentStatus",
    "DebugService",
    "SystemStatus",
    "FSService",
    "RelationService",
    "PackService",
    "SearchService",
    "ResourceService",
    "SessionService",
]
