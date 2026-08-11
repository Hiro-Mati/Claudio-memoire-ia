# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Focused tests for QueueManager concurrency selection."""

from openviking.service.task_work_index import QueueTaskMetadata
from openviking.storage.queuefs.queue_manager import QueueManager


def test_queue_concurrency_uses_separate_configured_values() -> None:
    manager = QueueManager(
        agfs=object(),
        max_concurrent_external_parse=9,
        max_concurrent_add_resource=7,
        max_concurrent_session_commit=5,
    )

    assert manager._max_concurrent_for_queue(manager.EXTERNAL_PARSE) == 9
    assert manager._max_concurrent_for_queue(manager.ADD_RESOURCE) == 7
    assert manager._max_concurrent_for_queue(manager.SESSION_COMMIT) == 5


def test_has_task_work_uses_runtime_index_without_queue_io() -> None:
    manager = QueueManager(agfs=object())
    metadata = QueueTaskMetadata(task_id="task-1", work_id="work-1")

    assert not manager.has_task_work("task-1")
    assert manager._task_work_index.register(manager.SESSION_COMMIT, metadata)
    assert manager.has_task_work("task-1")
