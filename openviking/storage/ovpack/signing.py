# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Ed25519 signatures for OVPack manifests.

The manifest already binds every file through per-entry SHA-256 and a
``content_sha256`` digest, so signing the manifest bytes is enough to make
the whole package tamper-evident. The signature lives inside the archive at
``{base}/_ovpack/manifest.sig.json``:

    {"algorithm": "ed25519", "public_key": "<hex>", "signature": "<hex>",
     "signed_at": "...", "key_id": "..."}

Verification needs no network: the public key travels with the package, and
``federation.trusted_public_keys`` decides whether that signer is accepted.
"""

from __future__ import annotations

import argparse
import binascii
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from openviking_cli.exceptions import InvalidArgumentError, PermissionDeniedError

from .format import OVPACK_INTERNAL_DIR

SIGNATURE_FILENAME = "manifest.sig.json"
SIGNATURE_ZIP_LEAF = f"{OVPACK_INTERNAL_DIR}/{SIGNATURE_FILENAME}"
ALGORITHM = "ed25519"


# ---- keys -------------------------------------------------------------------


def generate_signing_key(path: str | os.PathLike[str]) -> str:
    """Write a new private key (hex) to ``path`` and return the public key hex."""
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw.hex() + "\n", encoding="utf-8")
    try:
        os.chmod(target, 0o600)
    except OSError:  # pragma: no cover - Windows
        pass
    return public_key_hex(key)


def load_private_key(path: str | os.PathLike[str]) -> Ed25519PrivateKey:
    """Load a private key stored as 32 raw bytes, hex text, or PEM."""
    data = Path(path).expanduser().read_bytes()
    stripped = data.strip()
    if stripped.startswith(b"-----BEGIN"):
        key = serialization.load_pem_private_key(stripped, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise InvalidArgumentError("signing key is not an Ed25519 key")
        return key
    if len(data) == 32:
        return Ed25519PrivateKey.from_private_bytes(data)
    try:
        raw = binascii.unhexlify(stripped)
    except (binascii.Error, ValueError) as exc:
        raise InvalidArgumentError("signing key must be 32 raw bytes, hex, or PEM") from exc
    if len(raw) != 32:
        raise InvalidArgumentError("signing key must decode to 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(raw)


def public_key_hex(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return public.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    ).hex()


# ---- sign / verify ----------------------------------------------------------


def sign_manifest(
    manifest_bytes: bytes, key: Ed25519PrivateKey, *, key_id: Optional[str] = None
) -> Dict[str, Any]:
    signature = key.sign(manifest_bytes)
    record: Dict[str, Any] = {
        "algorithm": ALGORITHM,
        "public_key": public_key_hex(key),
        "signature": signature.hex(),
        "signed_at": datetime.now(timezone.utc).isoformat(),
    }
    if key_id:
        record["key_id"] = key_id
    return record


def verify_manifest_signature(manifest_bytes: bytes, record: Dict[str, Any]) -> str:
    """Return the signer's public key hex, or raise InvalidArgumentError."""
    if not isinstance(record, dict) or record.get("algorithm") != ALGORITHM:
        raise InvalidArgumentError("unsupported or malformed ovpack signature")
    try:
        public = Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(record["public_key"])))
        signature = bytes.fromhex(str(record["signature"]))
    except (KeyError, ValueError) as exc:
        raise InvalidArgumentError("malformed ovpack signature") from exc
    try:
        public.verify(signature, manifest_bytes)
    except InvalidSignature as exc:
        raise InvalidArgumentError(
            "ovpack signature does not match the manifest (package altered or wrong key)"
        ) from exc
    return public_key_hex(public)


def signature_zip_path(base_name: str) -> str:
    return f"{base_name}/{SIGNATURE_ZIP_LEAF}"


def read_signature(zf: zipfile.ZipFile, base_name: str) -> Optional[Dict[str, Any]]:
    try:
        raw = zf.read(signature_zip_path(base_name))
    except KeyError:
        return None
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidArgumentError("malformed ovpack signature file") from exc
    return record if isinstance(record, dict) else None


def verify_package_signature(
    zf: zipfile.ZipFile,
    base_name: str,
    manifest_bytes: bytes,
    *,
    require_signature: bool = False,
    trusted_public_keys: Optional[list[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Enforce the federation policy on one package.

    Returns ``{"public_key": ..., "key_id": ..., "signed_at": ...}`` when a
    valid signature is present, ``None`` when the package is unsigned and
    unsigned packages are allowed.
    """
    record = read_signature(zf, base_name)
    if record is None:
        if require_signature:
            raise PermissionDeniedError(
                "ovpack import requires a signed package (federation.require_signature)",
                resource=base_name,
            )
        return None
    signer = verify_manifest_signature(manifest_bytes, record)
    trusted = [k.strip().lower() for k in (trusted_public_keys or []) if k and k.strip()]
    if trusted and signer.lower() not in trusted:
        raise PermissionDeniedError(
            f"ovpack signer {signer[:16]}... is not in federation.trusted_public_keys",
            resource=base_name,
        )
    return {
        "public_key": signer,
        "key_id": record.get("key_id"),
        "signed_at": record.get("signed_at"),
    }


def federation_settings() -> Dict[str, Any]:
    """Current federation config as a plain dict (empty when unavailable)."""
    try:
        from openviking_cli.utils.config import get_openviking_config

        federation = getattr(get_openviking_config(), "federation", None)
        return federation.model_dump() if federation is not None else {}
    except Exception:
        return {}


# ---- CLI --------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Manage OVPack signing keys")
    parser.add_argument("--generate", metavar="PATH", help="write a new private key to PATH")
    parser.add_argument("--public-key", metavar="PATH", help="print the public key of PATH")
    args = parser.parse_args(argv)
    if args.generate:
        public = generate_signing_key(args.generate)
        print(f"private key written to {args.generate}")
        print(f"public key (add to federation.trusted_public_keys on peers): {public}")
        return 0
    if args.public_key:
        print(public_key_hex(load_private_key(args.public_key)))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
