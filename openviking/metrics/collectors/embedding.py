# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""
Event collector: EmbeddingCollector.

Tracks embedding provider outcomes and token usage:
- Requests and duration histogram by provider/model/status
- Error counter by provider/model/normalized error code
- Successful provider/model call and token counters

It is fed by EmbeddingEventDataSource events emitted from embedding call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from openviking.metrics.core.base import MetricCollector

from .base import EventMetricCollector


@dataclass
class EmbeddingCollector(EventMetricCollector):
    """
    Translate embedding provider outcomes and usage events into metrics.

    The collector treats embedding outcomes as event-driven writes because the interesting facts
    are known at completion time and do not require scrape-time state inspection.
    """

    DOMAIN: ClassVar[str] = "embedding"
    # rule: <METRICS_NAMESPACE>_<DOMAIN>_calls_total
    # e.g.: openviking_embedding_calls_total
    CALLS_TOTAL: ClassVar[str] = MetricCollector.metric_name(DOMAIN, "calls", unit="total")
    # rule: <METRICS_NAMESPACE>_<DOMAIN>_tokens_input_total
    # e.g.: openviking_embedding_tokens_input_total
    TOKENS_INPUT_TOTAL: ClassVar[str] = MetricCollector.metric_name(
        DOMAIN, "tokens_input", unit="total"
    )
    # rule: <METRICS_NAMESPACE>_<DOMAIN>_tokens_output_total
    # e.g.: openviking_embedding_tokens_output_total
    TOKENS_OUTPUT_TOTAL: ClassVar[str] = MetricCollector.metric_name(
        DOMAIN, "tokens_output", unit="total"
    )
    # rule: <METRICS_NAMESPACE>_<DOMAIN>_tokens_total
    # e.g.: openviking_embedding_tokens_total
    TOKENS_TOTAL: ClassVar[str] = MetricCollector.metric_name(DOMAIN, "tokens", unit="total")
    # rule: <METRICS_NAMESPACE>_<DOMAIN>_requests_total
    # e.g.: openviking_embedding_requests_total
    REQUESTS_TOTAL: ClassVar[str] = MetricCollector.metric_name(DOMAIN, "requests", unit="total")
    # rule: <METRICS_NAMESPACE>_<DOMAIN>_request_duration_seconds
    # e.g.: openviking_embedding_request_duration_seconds
    REQUEST_DURATION_SECONDS: ClassVar[str] = MetricCollector.metric_name(
        DOMAIN, "request_duration", unit="seconds"
    )
    # rule: <METRICS_NAMESPACE>_<DOMAIN>_errors_total
    # e.g.: openviking_embedding_errors_total
    ERRORS_TOTAL: ClassVar[str] = MetricCollector.metric_name(DOMAIN, "errors", unit="total")

    SUPPORTED_EVENTS: ClassVar[frozenset[str]] = frozenset(
        {
            "embedding.call",
            "embedding.outcome",
        }
    )

    def collect(self, registry=None) -> None:
        """Implement the unified collector interface as a no-op for this event-driven collector."""
        return None

    def receive_hook(self, event_name: str, payload: dict, registry) -> None:
        """
        Translate one supported embedding event into the corresponding metric writes.
        """
        if event_name == "embedding.call":
            self.record_call(
                registry,
                provider=str(payload["provider"]),
                model_name=str(payload["model_name"]),
                prompt_tokens=int(payload["prompt_tokens"]),
                completion_tokens=int(payload["completion_tokens"]),
                account_id=(
                    None if payload.get("account_id") is None else str(payload.get("account_id"))
                ),
            )
            return
        if event_name == "embedding.outcome":
            self.record_outcome(
                registry,
                provider=str(payload["provider"]),
                model_name=str(payload["model_name"]),
                status=str(payload["status"]),
                duration_seconds=float(payload["duration_seconds"]),
                error_code=str(payload.get("error_code") or "unknown"),
                account_id=(
                    None if payload.get("account_id") is None else str(payload.get("account_id"))
                ),
            )

    def record_call(
        self,
        registry,
        *,
        provider: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        account_id: str | None = None,
    ) -> None:
        """Record one successful embedding provider call and its token usage."""
        labels = {"provider": str(provider), "model_name": str(model_name)}
        registry.inc_counter(
            self.CALLS_TOTAL,
            labels=labels,
            label_names=("provider", "model_name"),
            account_id=account_id,
        )
        if int(prompt_tokens) > 0:
            registry.inc_counter(
                self.TOKENS_INPUT_TOTAL,
                labels=labels,
                label_names=("provider", "model_name"),
                amount=int(prompt_tokens),
                account_id=account_id,
            )
        if int(completion_tokens) > 0:
            registry.inc_counter(
                self.TOKENS_OUTPUT_TOTAL,
                labels=labels,
                label_names=("provider", "model_name"),
                amount=int(completion_tokens),
                account_id=account_id,
            )
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        if total_tokens > 0:
            registry.inc_counter(
                self.TOKENS_TOTAL,
                labels=labels,
                label_names=("provider", "model_name"),
                amount=total_tokens,
                account_id=account_id,
            )

    def record_outcome(
        self,
        registry,
        *,
        provider: str,
        model_name: str,
        status: str,
        duration_seconds: float,
        error_code: str,
        account_id: str | None = None,
    ) -> None:
        """Record one completed embedding provider request, including failures."""
        labels = {
            "provider": str(provider),
            "model_name": str(model_name),
            "status": "error" if status == "error" else "ok",
        }
        registry.inc_counter(
            self.REQUESTS_TOTAL,
            labels=labels,
            label_names=("provider", "model_name", "status"),
            account_id=account_id,
        )
        registry.observe_histogram(
            self.REQUEST_DURATION_SECONDS,
            max(float(duration_seconds), 0.0),
            labels=labels,
            label_names=("provider", "model_name", "status"),
            account_id=account_id,
        )
        if labels["status"] == "error":
            registry.inc_counter(
                self.ERRORS_TOTAL,
                labels={
                    "provider": str(provider),
                    "model_name": str(model_name),
                    "error_code": str(error_code or "unknown"),
                },
                label_names=("provider", "model_name", "error_code"),
                account_id=account_id,
            )
