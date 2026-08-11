# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Focused tests for QueueManager task work lookup."""

from openviking.service.task_work_index import QueueTaskMetadata
from openviking.storage.queuefs.queue_manager import QueueManager


def test_has_task_work_uses_runtime_index_without_queue_io() -> None:
    manager = QueueManager(agfs=object())
    metadata = QueueTaskMetadata(task_id="task-1", work_id="work-1")

    assert not manager.has_task_work("task-1")
    assert manager._task_work_index.register(manager.SESSION_COMMIT, metadata)
    assert manager.has_task_work("task-1")
