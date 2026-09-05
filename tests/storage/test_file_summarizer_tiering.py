# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tiered summarization: per-file summaries use file_summarizer, overviews keep vlm."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.storage.queuefs.semantic_processor import SemanticProcessor
from openviking_cli.utils.config.open_viking_config import OpenVikingConfig
from openviking_cli.utils.config.vlm_config import VLMConfig


def test_file_summarizer_falls_back_to_vlm():
    config = OpenVikingConfig.model_validate(
        {"vlm": {"provider": "openai", "model": "big", "api_key": "k"}}
    )
    assert config.get_file_summarizer() is config.vlm

    config = OpenVikingConfig.model_validate(
        {
            "vlm": {"provider": "openai", "model": "big", "api_key": "k"},
            "file_summarizer": {},
        }
    )
    assert config.get_file_summarizer() is config.vlm

    config = OpenVikingConfig.model_validate(
        {
            "vlm": {"provider": "openai", "model": "big", "api_key": "k"},
            "file_summarizer": {"provider": "openai", "model": "tiny", "api_key": "k"},
        }
    )
    assert isinstance(config.get_file_summarizer(), VLMConfig)
    assert config.get_file_summarizer().model == "tiny"
    assert config.vlm.model == "big"


class _FakeVLM:
    def __init__(self, name: str, answer: str = "summary"):
        self.name = name
        self.answer = answer
        self.calls = []

    def is_available(self) -> bool:
        return True

    async def get_completion_async(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.answer


@pytest.mark.asyncio
async def test_text_summary_uses_file_summarizer_not_vlm(monkeypatch):
    big = _FakeVLM("big")
    tiny = _FakeVLM("tiny", answer="  A tiny summary.  ")
    config = SimpleNamespace(
        vlm=big,
        get_file_summarizer=lambda: tiny,
        semantic=SimpleNamespace(max_file_content_chars=30000, max_skeleton_chars=12000),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_openviking_config", lambda: config
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs",
        lambda: SimpleNamespace(read_file=AsyncMock(return_value="# Notes\n\nSome prose.")),
    )
    monkeypatch.setattr(
        "openviking.session.memory.utils.language.resolve_output_language",
        lambda content, config=None: "en",
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.render_prompt",
        lambda prompt_id, params: f"{prompt_id}:{params['file_name']}",
    )

    result = await SemanticProcessor()._generate_text_summary(
        "/x/notes.md", "notes.md", asyncio.Semaphore(1)
    )

    assert result == {"name": "notes.md", "summary": "A tiny summary."}
    assert tiny.calls == ["semantic.document_summary:notes.md"]
    assert big.calls == []
