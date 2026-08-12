"""Upload Vikingbot workspace artifacts for the existing HomeTest evaluator."""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import VakaVikingBotConfig


@dataclass(slots=True, frozen=True)
class HomepageIdentity:
    api_key: str
    user_id: str


class HomepageArtifactUploader:
    """Use Homepage presign + PUT, without invoking the old rollout API."""

    def __init__(self, config: VakaVikingBotConfig) -> None:
        self.base_url = config.homepage_base_url.rstrip("/")
        self.timeout_seconds = config.artifact_upload_timeout_seconds
        self.identity = _resolve_identity(config)

    async def upload(self, paths: list[Path]) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._upload_sync, paths)

    def _upload_sync(self, paths: list[Path]) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"artifact does not exist: {path}")
            presigned = self._presign(path.name)
            self._put(path, presigned)
            artifacts.append(
                {
                    "artifact_id": presigned["doc_id"],
                    "file_name": path.name,
                    "name": path.name,
                    "tos_path": presigned["path"],
                    "local_path": str(path),
                }
            )
        return artifacts

    def _presign(self, file_name: str) -> dict[str, str]:
        response = requests.post(
            f"{self.base_url}/homepage/open/v1/presign",
            headers=self._headers(),
            json={"file_name": file_name, "data_persistence": True},
            timeout=(30, self.timeout_seconds),
        )
        response.raise_for_status()
        body = response.json()
        if body.get("message") != "success":
            raise RuntimeError(f"Homepage artifact presign failed: {body}")
        result = {
            "url": str(body.get("url") or "").strip(),
            "path": str(body.get("path") or "").strip(),
            "doc_id": str(body.get("doc_id") or "").strip(),
        }
        if not all(result.values()):
            raise RuntimeError("Homepage artifact presign returned incomplete data")
        return result

    def _put(self, path: Path, presigned: dict[str, str]) -> None:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "x-tos-meta-doc_id": presigned["doc_id"],
            "Content-Length": str(path.stat().st_size),
        }
        with path.open("rb") as source:
            response = requests.put(
                presigned["url"],
                data=source,
                headers=headers,
                timeout=(60, self.timeout_seconds),
            )
        if response.status_code not in {200, 201}:
            raise RuntimeError(
                "Homepage artifact upload failed: "
                f"status={response.status_code}, body={response.text[:500]}"
            )

    def _headers(self) -> dict[str, str]:
        return {
            "X-HOMEPAGE-API-KEY": self.identity.api_key,
            "x-user-id": self.identity.user_id,
        }


def _resolve_identity(config: VakaVikingBotConfig) -> HomepageIdentity:
    if config.homepage_api_key and config.homepage_user_id:
        return HomepageIdentity(
            api_key=config.homepage_api_key,
            user_id=config.homepage_user_id,
        )
    if config.vaka_users_file is None:
        raise RuntimeError(
            "artifact scoring requires VAKA_VIKINGBOT_HOMEPAGE_API_KEY + "
            "VAKA_VIKINGBOT_HOMEPAGE_USER_ID, or VAKA_USERS_FILE"
        )
    return _first_tsv_identity(config.vaka_users_file)


def _first_tsv_identity(path: Path) -> HomepageIdentity:
    if not path.is_file():
        raise FileNotFoundError(f"Vaka users file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        for row in reader:
            normalized = {
                str(key or "").strip().lower(): str(value or "").strip()
                for key, value in row.items()
            }
            api_key = normalized.get("api_key", "")
            user_id = normalized.get("user_id", "")
            if api_key and user_id:
                return HomepageIdentity(api_key=api_key, user_id=user_id)
    raise RuntimeError(f"Vaka users file contains no usable identity: {path}")
