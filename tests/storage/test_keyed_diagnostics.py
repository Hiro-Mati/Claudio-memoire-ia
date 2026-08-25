# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Privacy regressions for keyed semantic logs that cross queue layers."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

import openviking.storage.queuefs.embedding_queue as embedding_queue_module
from openviking.storage.queuefs.embedding_msg import EmbeddingMsg
from openviking.storage.queuefs.embedding_queue import EmbeddingQueue
from openviking.storage.queuefs.keyed_diagnostics import (
    KeyedBatchDiagnostic,
    bind_keyed_batch_diagnostic,
)


@pytest.mark.asyncio
async def test_embedding_queue_hides_uri_and_telemetry_only_for_keyed_work(caplog) -> None:
    queue = EmbeddingQueue(MagicMock(), "/queue", "Embedding")
    queue._async_agfs = AsyncMock()
    queue._async_agfs.write.return_value = "embedding-physical"
    queue._initialized = True
    uri = "viking://resources/private/sensitive.md"
    telemetry_id = "sensitive-telemetry"
    message = EmbeddingMsg(
        message="summary",
        context_data={"uri": uri},
        telemetry_id=telemetry_id,
    )
    diagnostic = KeyedBatchDiagnostic(
        physical_id="semantic-physical",
        dispatch_hash_prefix="abcdef012345",
        contribution_count=2,
    )

    embedding_queue_module.logger.addHandler(caplog.handler)
    try:
        with (
            caplog.at_level(logging.DEBUG, logger=embedding_queue_module.logger.name),
            bind_keyed_batch_diagnostic(diagnostic),
        ):
            await queue.enqueue(message)
    finally:
        embedding_queue_module.logger.removeHandler(caplog.handler)

    keyed_records = [
        record for record in caplog.records if record.name == embedding_queue_module.logger.name
    ]
    assert keyed_records
    for record in keyed_records:
        assert record.physical_id == "semantic-physical"
        assert record.dispatch_hash_prefix == "abcdef012345"
        assert record.contribution_count == 2
        rendered = f"{record.getMessage()} {record.__dict__!r}"
        assert uri not in rendered
        assert telemetry_id not in rendered

    caplog.clear()
    embedding_queue_module.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger=embedding_queue_module.logger.name):
            await queue.enqueue(message)
    finally:
        embedding_queue_module.logger.removeHandler(caplog.handler)

    assert uri in caplog.text
