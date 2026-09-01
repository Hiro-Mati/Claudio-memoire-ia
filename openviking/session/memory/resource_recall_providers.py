"""External resource recall providers used during session memory extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.parse import urlencode

import httpx

from openviking.session.extraction_context_policy import ExternalResourceProviderPolicy
from openviking.telemetry import tracer


@dataclass
class ResourceRecallItem:
    uri: str
    provider: str
    provider_type: str
    title: str = ""
    content: str = ""
    score: float | None = None

    def to_snippet(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in {
                "uri": self.uri,
                "provider": self.provider,
                "provider_type": self.provider_type,
                "title": self.title,
                "content": self.content,
                "score": self.score,
            }.items()
            if value is not None and value != ""
        }

    def to_ref(self) -> Dict[str, Any]:
        ref: Dict[str, Any] = {
            "resource_uri": self.uri,
            "source": "extraction_context_recall",
            "provider": self.provider,
            "provider_type": self.provider_type,
        }
        if self.title:
            ref["title"] = self.title
        if self.score is not None:
            ref["score"] = self.score
        return ref


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalise_base_url(value: str) -> str:
    return value.rstrip("/")


def _normalise_api_prefix(value: str) -> str:
    if not value:
        return ""
    return "/" + value.strip("/")


def _response_result(payload: Any) -> Any:
    if isinstance(payload, dict) and "result" in payload:
        return payload.get("result")
    return payload


def _extract_search_hits(result: Any) -> List[Dict[str, Any]]:
    if isinstance(result, dict):
        for key in ("entries", "items", "results", "matches"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        flattened: List[Dict[str, Any]] = []
        for key in ("memories", "resources", "skills"):
            value = result.get(key)
            if isinstance(value, list):
                flattened.extend(item for item in value if isinstance(item, dict))
        if flattened:
            return flattened
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


def _hit_uri(hit: Dict[str, Any]) -> str:
    for key in ("uri", "resource_uri", "path"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _hit_score(hit: Dict[str, Any]) -> float | None:
    for key in ("score", "similarity", "rank_score"):
        value = hit.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _hit_title(hit: Dict[str, Any], uri: str) -> str:
    for key in ("title", "name", "topic"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return uri.rstrip("/").rsplit("/", 1)[-1] if uri else ""


def _hit_text(hit: Dict[str, Any]) -> str:
    for key in ("text", "content", "abstract", "snippet", "summary"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class OpenVikingHttpResourceProvider:
    """Read-only resource provider backed by another OpenViking HTTP service."""

    def __init__(
        self,
        policy: ExternalResourceProviderPolicy,
        *,
        base_url: str,
        api_prefix: str = "",
        api_key: str = "",
        token: str = "",
        openviking: str = "",
        region: str = "",
        account: str = "",
        user: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.policy = policy
        self.base_url = _normalise_base_url(base_url)
        self.api_prefix = _normalise_api_prefix(api_prefix)
        self.api_key = api_key
        self.token = token
        self.openviking = openviking
        self.region = region
        self.account = account
        self.user = user
        self.timeout = timeout

    @classmethod
    def from_env(cls, policy: ExternalResourceProviderPolicy) -> "OpenVikingHttpResourceProvider | None":
        prefix = "THIRD_OV" if policy.type == "third_ov" else "OPENVIKING_RESOURCE_RECALL"
        base_url = _env_first(
            f"{prefix}_OPENVIKING_URL",
            f"{prefix}_BASE_URL",
            f"{prefix}_URL",
        )
        if not base_url:
            return None
        timeout_raw = _env_first(f"{prefix}_TIMEOUT", f"{prefix}_TIMEOUT_SECONDS")
        try:
            timeout = float(timeout_raw) if timeout_raw else 30.0
        except ValueError:
            timeout = 30.0
        return cls(
            policy,
            base_url=base_url,
            api_prefix=_env_first(f"{prefix}_API_PREFIX"),
            api_key=_env_first(f"{prefix}_API_KEY", f"{prefix}_TOKEN"),
            token=_env_first(f"{prefix}_TOKEN"),
            openviking=_env_first(f"{prefix}_OPENVIKING"),
            region=_env_first(f"{prefix}_REGION"),
            account=_env_first(f"{prefix}_ACCOUNT"),
            user=_env_first(f"{prefix}_USER"),
            timeout=timeout,
        )

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "openviking-extraction-resource-recall/1.0",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.token:
            headers["token"] = self.token
        if self.openviking:
            headers["openviking"] = self.openviking
        if self.region:
            headers["region"] = self.region
        if self.account:
            headers["X-OpenViking-Account"] = self.account
        if self.user:
            headers["X-OpenViking-User"] = self.user
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{self.api_prefix}{path}"

    async def search(self, query: str, *, limit: int) -> List[ResourceRecallItem]:
        if not self.base_url:
            return []
        payload = {
            "query": query,
            "target_uri": self.policy.target_uri,
            "limit": limit,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers()) as client:
                response = await client.post(self._url("/api/v1/search/search"), json=payload)
                response.raise_for_status()
                result = _response_result(response.json())
        except Exception as exc:
            tracer.error(f"External resource recall search failed for {self.policy.name}: {exc}")
            return []
        items: List[ResourceRecallItem] = []
        for hit in _extract_search_hits(result):
            uri = _hit_uri(hit)
            if not uri:
                continue
            items.append(
                ResourceRecallItem(
                    uri=uri,
                    provider=self.policy.name,
                    provider_type=self.policy.type,
                    title=_hit_title(hit, uri),
                    content=_hit_text(hit),
                    score=_hit_score(hit),
                )
            )
        return items

    async def read(self, item: ResourceRecallItem) -> ResourceRecallItem:
        query = urlencode({"uri": item.uri})
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers()) as client:
                response = await client.get(self._url(f"/api/v1/content/read?{query}"))
                response.raise_for_status()
                result = _response_result(response.json())
        except Exception as exc:
            tracer.error(f"External resource recall read failed for {self.policy.name}: {exc}")
            return item
        content = ""
        title = item.title
        if isinstance(result, dict):
            content_value = result.get("content") or result.get("text")
            if isinstance(content_value, str):
                content = content_value
            title_value = result.get("title") or result.get("name")
            if isinstance(title_value, str) and title_value.strip():
                title = title_value.strip()
        elif isinstance(result, str):
            content = result
        return ResourceRecallItem(
            uri=item.uri,
            provider=item.provider,
            provider_type=item.provider_type,
            title=title,
            content=content or item.content,
            score=item.score,
        )


def create_external_resource_provider(
    policy: ExternalResourceProviderPolicy,
) -> OpenVikingHttpResourceProvider | None:
    if policy.type in {"third_ov", "openviking_http"}:
        provider = OpenVikingHttpResourceProvider.from_env(policy)
        if provider is None:
            tracer.info(
                f"External resource provider {policy.name} is configured but no base URL is set"
            )
        return provider
    return None
