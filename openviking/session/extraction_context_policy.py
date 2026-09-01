"""Policy for commit-time memory extraction recall context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from openviking_cli.exceptions import InvalidArgumentError

_TOP_KEYS = {"memory_recall", "event_recall", "resource_recall"}
_RECALL_KEYS = {
    "enabled",
    "mode",
    "max_queries",
    "max_entries",
    "max_tokens",
    "scopes",
    "external_providers",
}
_EXTERNAL_PROVIDER_KEYS = {"name", "type", "target_uri", "max_entries", "max_tokens"}
_MODES = {"off", "selective"}
_EXTERNAL_PROVIDER_TYPES = {"third_ov", "openviking_http"}
_DEFAULT_RESOURCE_SCOPES = ["viking://resources/"]


def _coerce_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise InvalidArgumentError(f"extraction_context_policy.{field} must be a boolean")


def _coerce_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise InvalidArgumentError(f"extraction_context_policy.{field} must be an integer")
    return min(max(parsed, minimum), maximum)


def _coerce_mode(value: Any, *, field: str) -> str:
    mode = str(value or "selective").strip()
    if mode not in _MODES:
        raise InvalidArgumentError(f"extraction_context_policy.{field}.mode must be one of: off, selective")
    return mode


def _coerce_scopes(value: Any, *, field: str) -> List[str]:
    if value is None:
        return list(_DEFAULT_RESOURCE_SCOPES)
    if not isinstance(value, list):
        raise InvalidArgumentError(f"extraction_context_policy.{field}.scopes must be a list")
    scopes = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise InvalidArgumentError(
                f"extraction_context_policy.{field}.scopes must contain non-empty strings"
            )
        scopes.append(item.strip())
    return scopes[:10]


def _coerce_optional_str(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise InvalidArgumentError(f"extraction_context_policy.{field} must be a string")
    return value.strip()


@dataclass
class ExternalResourceProviderPolicy:
    name: str
    type: str
    target_uri: str = "viking://resources/"
    max_entries: int = 0
    max_tokens: int = 0

    @classmethod
    def from_dict(
        cls,
        data: Any,
        *,
        field: str,
        default_max_entries: int,
        default_max_tokens: int,
    ) -> "ExternalResourceProviderPolicy":
        if not isinstance(data, dict):
            raise InvalidArgumentError(f"extraction_context_policy.{field} must contain objects")
        extra = set(data) - _EXTERNAL_PROVIDER_KEYS
        if extra:
            raise InvalidArgumentError(
                f"extraction_context_policy.{field} provider supports only: "
                + ", ".join(sorted(_EXTERNAL_PROVIDER_KEYS))
            )
        provider_type = _coerce_optional_str(data.get("type"), field=f"{field}.type")
        if provider_type not in _EXTERNAL_PROVIDER_TYPES:
            raise InvalidArgumentError(
                f"extraction_context_policy.{field}.type must be one of: "
                + ", ".join(sorted(_EXTERNAL_PROVIDER_TYPES))
            )
        name = _coerce_optional_str(data.get("name") or provider_type, field=f"{field}.name")
        if not name:
            raise InvalidArgumentError(f"extraction_context_policy.{field}.name must be non-empty")
        target_uri = _coerce_optional_str(
            data.get("target_uri") or "viking://resources/",
            field=f"{field}.target_uri",
        )
        if not target_uri:
            raise InvalidArgumentError(
                f"extraction_context_policy.{field}.target_uri must be non-empty"
            )
        return cls(
            name=name,
            type=provider_type,
            target_uri=target_uri,
            max_entries=_coerce_int(
                data.get("max_entries", default_max_entries),
                field=f"{field}.max_entries",
                minimum=0,
                maximum=20,
            ),
            max_tokens=_coerce_int(
                data.get("max_tokens", default_max_tokens),
                field=f"{field}.max_tokens",
                minimum=0,
                maximum=20_000,
            ),
        )

    def active(self) -> bool:
        return self.max_entries > 0 and self.max_tokens > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "target_uri": self.target_uri,
            "max_entries": self.max_entries,
            "max_tokens": self.max_tokens,
        }


def _coerce_external_providers(
    value: Any,
    *,
    field: str,
    default_max_entries: int,
    default_max_tokens: int,
) -> List[ExternalResourceProviderPolicy]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise InvalidArgumentError(
            f"extraction_context_policy.{field}.external_providers must be a list"
        )
    providers: List[ExternalResourceProviderPolicy] = []
    seen_names = set()
    for idx, item in enumerate(value[:5]):
        provider = ExternalResourceProviderPolicy.from_dict(
            item,
            field=f"{field}.external_providers[{idx}]",
            default_max_entries=default_max_entries,
            default_max_tokens=default_max_tokens,
        )
        if provider.name in seen_names:
            continue
        seen_names.add(provider.name)
        providers.append(provider)
    return providers


@dataclass
class RecallPolicy:
    enabled: bool = False
    mode: str = "selective"
    max_queries: int = 0
    max_entries: int = 0
    max_tokens: int = 0
    scopes: List[str] = field(default_factory=list)
    external_providers: List[ExternalResourceProviderPolicy] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: Any,
        *,
        defaults: "RecallPolicy",
        key: str,
        allow_scopes: bool = False,
    ) -> "RecallPolicy":
        if data is None:
            return cls(
                enabled=defaults.enabled,
                mode=defaults.mode,
                max_queries=defaults.max_queries,
                max_entries=defaults.max_entries,
                max_tokens=defaults.max_tokens,
                scopes=list(defaults.scopes),
                external_providers=list(defaults.external_providers),
            )
        if not isinstance(data, dict):
            raise InvalidArgumentError(f"extraction_context_policy.{key} must be an object")
        allowed_keys = _RECALL_KEYS if allow_scopes else _RECALL_KEYS - {"scopes", "external_providers"}
        extra = set(data) - allowed_keys
        if extra:
            raise InvalidArgumentError(
                f"extraction_context_policy.{key} supports only: "
                + ", ".join(sorted(allowed_keys))
            )
        if "scopes" in data and not allow_scopes:
            raise InvalidArgumentError(f"extraction_context_policy.{key}.scopes is not supported")
        if "external_providers" in data and not allow_scopes:
            raise InvalidArgumentError(
                f"extraction_context_policy.{key}.external_providers is not supported"
            )
        max_entries = _coerce_int(
            data.get("max_entries", defaults.max_entries),
            field=f"{key}.max_entries",
            minimum=0,
            maximum=20,
        )
        max_tokens = _coerce_int(
            data.get("max_tokens", defaults.max_tokens),
            field=f"{key}.max_tokens",
            minimum=0,
            maximum=20_000,
        )
        return cls(
            enabled=_coerce_bool(data.get("enabled", defaults.enabled), field=f"{key}.enabled"),
            mode=_coerce_mode(data.get("mode", defaults.mode), field=key),
            max_queries=_coerce_int(
                data.get("max_queries", defaults.max_queries),
                field=f"{key}.max_queries",
                minimum=0,
                maximum=5,
            ),
            max_entries=max_entries,
            max_tokens=max_tokens,
            scopes=(
                _coerce_scopes(data.get("scopes", defaults.scopes), field=key)
                if allow_scopes
                else []
            ),
            external_providers=(
                _coerce_external_providers(
                    data.get("external_providers", defaults.external_providers),
                    field=key,
                    default_max_entries=max_entries,
                    default_max_tokens=max_tokens,
                )
                if allow_scopes
                else []
            ),
        )

    def active(self) -> bool:
        return self.enabled and self.mode != "off" and self.max_entries > 0 and self.max_tokens > 0

    def has_active_external_providers(self) -> bool:
        return self.enabled and self.mode != "off" and any(
            provider.active() for provider in self.external_providers
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "enabled": self.enabled,
            "mode": self.mode,
            "max_queries": self.max_queries,
            "max_entries": self.max_entries,
            "max_tokens": self.max_tokens,
        }
        if self.scopes:
            data["scopes"] = list(self.scopes)
        if self.external_providers:
            data["external_providers"] = [
                provider.to_dict() for provider in self.external_providers
            ]
        return data


@dataclass
class ExtractionContextPolicy:
    memory_recall: RecallPolicy = field(
        default_factory=lambda: RecallPolicy(
            enabled=True,
            mode="selective",
            max_queries=3,
            max_entries=8,
            max_tokens=6_000,
        )
    )
    event_recall: RecallPolicy = field(
        default_factory=lambda: RecallPolicy(
            enabled=True,
            mode="selective",
            max_queries=2,
            max_entries=5,
            max_tokens=4_000,
        )
    )
    resource_recall: RecallPolicy = field(
        default_factory=lambda: RecallPolicy(
            enabled=False,
            mode="selective",
            max_queries=3,
            max_entries=5,
            max_tokens=4_000,
            scopes=list(_DEFAULT_RESOURCE_SCOPES),
        )
    )

    @classmethod
    def default(cls) -> "ExtractionContextPolicy":
        return cls()

    @classmethod
    def from_dict(cls, data: Any) -> "ExtractionContextPolicy":
        defaults = cls.default()
        if data is None:
            return defaults
        if isinstance(data, ExtractionContextPolicy):
            return data
        if not isinstance(data, dict):
            raise InvalidArgumentError("extraction_context_policy must be an object")
        extra = set(data) - _TOP_KEYS
        if extra:
            raise InvalidArgumentError(
                "extraction_context_policy supports only: " + ", ".join(sorted(_TOP_KEYS))
            )
        return cls(
            memory_recall=RecallPolicy.from_dict(
                data.get("memory_recall"),
                defaults=defaults.memory_recall,
                key="memory_recall",
            ),
            event_recall=RecallPolicy.from_dict(
                data.get("event_recall"),
                defaults=defaults.event_recall,
                key="event_recall",
            ),
            resource_recall=RecallPolicy.from_dict(
                data.get("resource_recall"),
                defaults=defaults.resource_recall,
                key="resource_recall",
                allow_scopes=True,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_recall": self.memory_recall.to_dict(),
            "event_recall": self.event_recall.to_dict(),
            "resource_recall": self.resource_recall.to_dict(),
        }
