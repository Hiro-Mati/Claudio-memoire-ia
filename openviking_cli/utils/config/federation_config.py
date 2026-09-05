# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Federation configuration: signed OVPack exchange between OpenViking servers."""

from typing import List, Optional

from pydantic import BaseModel, Field


class FederationConfig(BaseModel):
    """Signing and trust settings for OVPack packages exchanged between servers.

    A server that exports with ``signing_key_file`` set embeds an Ed25519
    signature of the package manifest. A server that imports verifies any
    embedded signature; ``require_signature`` rejects unsigned packages and
    ``trusted_public_keys`` restricts accepted signers.
    """

    signing_key_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to this server's Ed25519 private key (32 raw bytes, hex text, or PEM). "
            "Generate one with `python -m openviking.storage.ovpack.signing --generate PATH`. "
            "When set, every exported .ovpack carries a manifest signature."
        ),
    )
    key_id: Optional[str] = Field(
        default=None,
        description="Optional human-readable identifier recorded next to the signature.",
    )
    trusted_public_keys: List[str] = Field(
        default_factory=list,
        description=(
            "Hex-encoded Ed25519 public keys of servers whose packages may be imported. "
            "Empty means any valid signature is accepted (the signer is still recorded)."
        ),
    )
    require_signature: bool = Field(
        default=False,
        description="Reject .ovpack imports that carry no valid signature.",
    )

    model_config = {"extra": "forbid"}
