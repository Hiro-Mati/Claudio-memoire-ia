# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Process roles: api processes must not run QueueFS consumers."""

from openviking.server.config import ServerConfig
from openviking.service.core import QUEUE_ROLE_ENV, queue_consumers_enabled


def test_queue_consumers_follow_role_env(monkeypatch):
    monkeypatch.delenv(QUEUE_ROLE_ENV, raising=False)
    assert queue_consumers_enabled() is True
    monkeypatch.setenv(QUEUE_ROLE_ENV, "api")
    assert queue_consumers_enabled() is False
    monkeypatch.setenv(QUEUE_ROLE_ENV, "worker")
    assert queue_consumers_enabled() is True
    monkeypatch.setenv(QUEUE_ROLE_ENV, " ALL ")
    assert queue_consumers_enabled() is True


def test_server_config_accepts_queue_role():
    assert ServerConfig().queue_role == "all"
    assert ServerConfig(queue_role="api").queue_role == "api"
