# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Named context contracts: validation, merge semantics, user-config storage."""

import pytest

from openviking.retrieve.context_assembler.contracts import (
    ContextContract,
    apply_contract,
    validate_contract_name,
)
from openviking.server.config import UserConfig


def test_contract_pins_only_set_fields():
    contract = ContextContract(max_tokens=1500, purpose="coding", description="for Claude Code")
    assert contract.pinned() == {"max_tokens": 1500, "purpose": "coding"}
    with pytest.raises(ValueError):
        ContextContract(max_tokens=10)
    with pytest.raises(ValueError):
        ContextContract(unknown_field=1)


def test_contract_names():
    assert validate_contract_name("  claude-code ") == "claude-code"
    for bad in ("", "a/b", "with space", "x" * 65):
        with pytest.raises(ValueError):
            validate_contract_name(bad)


def test_apply_contract_never_overrides_explicit_fields():
    request = {"query": "q", "max_tokens": 8000, "purpose": None, "dedup_turns": 0}
    merged, applied = apply_contract(
        request,
        explicitly_set={"query", "max_tokens"},
        contract={"max_tokens": 1500, "purpose": "coding", "dedup_turns": 5},
    )
    assert merged["max_tokens"] == 8000  # explicit request value wins
    assert merged["purpose"] == "coding" and merged["dedup_turns"] == 5
    assert applied == ["purpose", "dedup_turns"]


def test_user_config_validates_contracts():
    config = UserConfig.model_validate(
        {"context_contracts": {"claude-code": {"max_tokens": 2000, "peer_scope": "actor"}}}
    )
    assert config.context_contracts == {"claude-code": {"max_tokens": 2000, "peer_scope": "actor"}}
    with pytest.raises(ValueError):
        UserConfig.model_validate({"context_contracts": {"bad name": {"max_tokens": 2000}}})
    with pytest.raises(ValueError):
        UserConfig.model_validate({"context_contracts": {"ok": {"max_tokens": 1}}})
    assert UserConfig().context_contracts == {}
