# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Ed25519 signatures on OVPack manifests and the federation import policy."""

import io
import json
import zipfile

import pytest

from openviking.storage.ovpack import signing
from openviking_cli.exceptions import InvalidArgumentError, PermissionDeniedError

BASE = "pkg"
MANIFEST = json.dumps({"format_version": 3, "kind": "openviking.ovpack"}).encode()


def _zip(manifest: bytes, record=None) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{BASE}/_ovpack/manifest.json", manifest)
        if record is not None:
            zf.writestr(signing.signature_zip_path(BASE), json.dumps(record))
    buf.seek(0)
    return zipfile.ZipFile(buf, "r")


def test_generate_load_and_roundtrip(tmp_path):
    key_path = tmp_path / "federation.key"
    public = signing.generate_signing_key(key_path)
    key = signing.load_private_key(key_path)
    assert signing.public_key_hex(key) == public
    record = signing.sign_manifest(MANIFEST, key, key_id="laptop")
    assert record["algorithm"] == "ed25519" and record["key_id"] == "laptop"
    assert signing.verify_manifest_signature(MANIFEST, record) == public
    with pytest.raises(InvalidArgumentError):
        signing.verify_manifest_signature(MANIFEST + b" ", record)
    with pytest.raises(InvalidArgumentError):
        signing.verify_manifest_signature(MANIFEST, {**record, "algorithm": "rsa"})


def test_load_private_key_accepts_raw_bytes(tmp_path):
    key_path = tmp_path / "raw.key"
    key_path.write_bytes(bytes(range(32)))
    assert len(signing.public_key_hex(signing.load_private_key(key_path))) == 64
    (tmp_path / "bad.key").write_text("nothex", encoding="utf-8")
    with pytest.raises(InvalidArgumentError):
        signing.load_private_key(tmp_path / "bad.key")


def test_package_policy(tmp_path):
    public = signing.generate_signing_key(tmp_path / "k")
    key = signing.load_private_key(tmp_path / "k")
    record = signing.sign_manifest(MANIFEST, key)

    # unsigned package: allowed unless required
    assert signing.verify_package_signature(_zip(MANIFEST), BASE, MANIFEST) is None
    with pytest.raises(PermissionDeniedError):
        signing.verify_package_signature(_zip(MANIFEST), BASE, MANIFEST, require_signature=True)

    # signed package: accepted, signer reported
    signer = signing.verify_package_signature(_zip(MANIFEST, record), BASE, MANIFEST)
    assert signer["public_key"] == public

    # trusted-keys allowlist
    assert signing.verify_package_signature(
        _zip(MANIFEST, record), BASE, MANIFEST, trusted_public_keys=[public.upper()]
    )
    with pytest.raises(PermissionDeniedError):
        signing.verify_package_signature(
            _zip(MANIFEST, record), BASE, MANIFEST, trusted_public_keys=["00" * 32]
        )

    # tampered manifest
    with pytest.raises(InvalidArgumentError):
        signing.verify_package_signature(_zip(MANIFEST, record), BASE, MANIFEST + b"x")


def test_cli_generates_and_prints_public_key(tmp_path, capsys):
    path = tmp_path / "gen.key"
    assert signing.main(["--generate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "public key" in out
    assert signing.main(["--public-key", str(path)]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed in out
