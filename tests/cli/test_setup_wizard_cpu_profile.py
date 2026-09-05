# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""CPU-only profile of the setup wizard (no GPU accelerator detected)."""

import pytest

from openviking_cli.setup_wizard import (
    EMBEDDING_PRESETS,
    FILE_SUMMARIZER_PRESET,
    VLM_PRESETS,
    _build_cpu_only_extras,
    _get_recommended_indices,
    _has_gpu_accelerator,
)
from openviking_cli.utils.config.open_viking_config import OpenVikingConfig


def test_cpu_only_profile_ignores_ram_tiers():
    for ram in (4, 16, 32, 128):
        emb_idx, vlm_idx = _get_recommended_indices(ram, gpu=False)
        assert EMBEDDING_PRESETS[emb_idx].model == "qwen3-embedding:0.6b"
        assert VLM_PRESETS[vlm_idx].ollama_model == "qwen3.5:4b"
    # default keeps the upstream RAM tiers
    emb_idx, _ = _get_recommended_indices(32)
    assert EMBEDDING_PRESETS[emb_idx].model == "qwen3-embedding:8b"


def test_cpu_only_extras_validate_against_config_schema():
    extras = _build_cpu_only_extras()
    assert extras["file_summarizer"]["model"] == FILE_SUMMARIZER_PRESET.litellm_model
    assert extras["semantic"]["parent_refresh_mode"] == "debounced"
    assert extras["retrieval"]["lexical_index_enabled"] is True
    config = OpenVikingConfig.model_validate(
        {
            "vlm": {"provider": "litellm", "model": "ollama/qwen3.5:4b", "api_key": "no-key"},
            **extras,
        }
    )
    assert config.get_file_summarizer().model == "ollama/qwen3.5:0.8b"
    assert config.semantic.parent_refresh_debounce_s == 60
    assert config.retrieval.lexical_boost == pytest.approx(0.3)


def test_gpu_detection_override_and_fallbacks(monkeypatch):
    monkeypatch.setenv("OPENVIKING_ASSUME_GPU", "1")
    assert _has_gpu_accelerator() is True
    monkeypatch.setenv("OPENVIKING_ASSUME_GPU", "0")
    assert _has_gpu_accelerator() is False
    monkeypatch.delenv("OPENVIKING_ASSUME_GPU")

    monkeypatch.setattr("openviking_cli.setup_wizard.sys.platform", "linux")
    monkeypatch.setattr("openviking_cli.setup_wizard.shutil.which", lambda name: None)
    assert _has_gpu_accelerator() is False

    monkeypatch.setattr("openviking_cli.setup_wizard.sys.platform", "darwin")
    monkeypatch.setattr("openviking_cli.setup_wizard.platform.machine", lambda: "arm64")
    assert _has_gpu_accelerator() is True
