# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Named context contracts.

An agent declares once what it can accept from a recall: token budget, memory
quotas, detail tiers, dedup window, peer scope, digest rewriting. The contract
is stored in the user's settings under a name and applied to every
``mode="context"`` search (or legacy ``/recall``) that names it. Fields set
explicitly on the request still win, so a contract is a set of defaults the
server guarantees, not a cage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_NAME_MAX = 64


class ContextContract(BaseModel):
    """The subset of context-assembly parameters an agent may pin."""

    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = Field(default=None, max_length=500)
    max_tokens: Optional[int] = Field(default=None, ge=64, le=32000)
    quotas: Optional[Dict[str, int]] = None
    purpose: Optional[Literal["chat", "coding"]] = None
    detail: Optional[Any] = None
    dedup_turns: Optional[int] = Field(default=None, ge=0, le=100)
    peer_scope: Optional[Literal["actor", "all"]] = None
    other_peer_penalty: Optional[Union[float, Dict[str, float]]] = None
    rewrite: Optional[Union[bool, Literal["auto"]]] = None
    rewrite_max_bullets: Optional[int] = Field(default=None, ge=1, le=20)
    query_expansion: Optional[Literal["off", "auto"]] = None
    score_threshold: Optional[float] = None
    exclude_uris: Optional[List[str]] = None
    limit: Optional[int] = Field(default=None, ge=1, le=200)

    def pinned(self) -> Dict[str, Any]:
        """Only the fields the contract actually sets (description excluded)."""
        data = self.model_dump(exclude_none=True)
        data.pop("description", None)
        return data


def validate_contract_name(name: str) -> str:
    name = (name or "").strip()
    if not name or len(name) > CONTRACT_NAME_MAX:
        raise ValueError(f"contract name must be 1..{CONTRACT_NAME_MAX} characters")
    if any(ch in name for ch in "/\\ \t\n"):
        raise ValueError("contract name must not contain slashes or whitespace")
    return name


def apply_contract(
    request_values: Mapping[str, Any],
    explicitly_set: Set[str],
    contract: ContextContract | Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Merge a contract under a request.

    Returns the merged values and the list of fields the contract supplied.
    A field the caller set explicitly is never overridden.
    """
    if not isinstance(contract, ContextContract):
        contract = ContextContract.model_validate(dict(contract))
    merged = dict(request_values)
    applied: List[str] = []
    for key, value in contract.pinned().items():
        if key in explicitly_set:
            continue
        merged[key] = value
        applied.append(key)
    return merged, applied
