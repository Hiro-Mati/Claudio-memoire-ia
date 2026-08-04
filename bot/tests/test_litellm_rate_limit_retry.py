# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for LiteLLMProvider rate-limit retry behavior.

Higher ov compile plan-audit concurrency makes transient 429/RPM/TPM throttling
more likely; the provider must retry those errors with backoff while still
failing fast on every other error class.
"""

from types import SimpleNamespace

import pytest

from vikingbot.providers import litellm_provider
from vikingbot.providers.litellm_provider import LiteLLMProvider


def _make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        api_key="test-key",
        default_model="anthropic/claude-opus-4-5",
        langfuse_client=SimpleNamespace(enabled=False, _client=None),
    )


def _fake_completion_response(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant(_seconds):
        return None

    monkeypatch.setattr(litellm_provider.asyncio, "sleep", _instant)
    monkeypatch.setattr(litellm_provider, "rate_limit_retry_delay", lambda attempt: 0.0)


@pytest.mark.asyncio
async def test_chat_retries_rate_limit_then_succeeds(monkeypatch):
    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Error code: 429 - RateLimitExceeded")
        return _fake_completion_response("ok")

    monkeypatch.setattr(litellm_provider, "acompletion", fake_acompletion)

    provider = _make_provider()
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert calls["n"] == 3
    assert response.finish_reason == "stop"
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_chat_does_not_retry_non_rate_limit_error(monkeypatch):
    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        calls["n"] += 1
        raise RuntimeError("Error code: 400 - invalid parameter")

    monkeypatch.setattr(litellm_provider, "acompletion", fake_acompletion)

    provider = _make_provider()
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    # 400 is not retryable: called once, then the broad handler returns an error
    # response instead of retrying.
    assert calls["n"] == 1
    assert response.finish_reason == "error"


@pytest.mark.asyncio
async def test_chat_gives_up_after_max_rate_limit_retries(monkeypatch):
    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        calls["n"] += 1
        raise RuntimeError("429 TooManyRequests")

    monkeypatch.setattr(litellm_provider, "acompletion", fake_acompletion)

    provider = _make_provider()
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    # Initial attempt plus _MAX_RATE_LIMIT_RETRIES retries, then the error is
    # re-raised and converted to an error response by the outer handler.
    assert calls["n"] == provider._MAX_RATE_LIMIT_RETRIES + 1
    assert response.finish_reason == "error"


@pytest.mark.asyncio
async def test_chat_stream_retries_rate_limit_then_streams(monkeypatch):
    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("Error code: 429 - RPM (Requests Per Minute) limit")

        async def _stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(
                            reasoning_content=None, content="hello", tool_calls=[]
                        ),
                    )
                ],
                usage=None,
            )

        return _stream()

    monkeypatch.setattr(litellm_provider, "acompletion", fake_acompletion)

    provider = _make_provider()
    events = [
        event
        async for event in provider.chat_stream(messages=[{"role": "user", "content": "hi"}])
    ]

    assert calls["n"] == 2
    response_event = next(e for e in events if e.type == "response")
    assert response_event.response.finish_reason == "stop"
    assert response_event.response.content == "hello"
