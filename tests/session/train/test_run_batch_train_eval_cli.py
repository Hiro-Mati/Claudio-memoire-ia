from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.session.train import batch_runner
from openviking.session.train.batch_runner import BatchTrainEvalConfig
from openviking.session.train.run_batch_train_eval import (
    _parse_server_header,
    main_async,
    parse_args,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_train_concurrency_defaults_to_200(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_batch_train_eval", "--dataset", "tau2", "--domain", "airline"],
    )

    args = parse_args()
    config = BatchTrainEvalConfig(dataset="tau2", domain="airline")

    assert args.concurrency == 200
    assert args.commit_concurrency == 200
    assert args.commit_timeout_seconds is None
    assert config.concurrency == 200
    assert config.commit_concurrency == 200
    assert config.commit_timeout_seconds is None


def test_train_concurrency_explicit_overrides_are_preserved(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_batch_train_eval",
            "--dataset",
            "tau2",
            "--domain",
            "airline",
            "--concurrency",
            "80",
            "--commit-concurrency",
            "90",
            "--commit-timeout-seconds",
            "900",
        ],
    )

    args = parse_args()
    config = BatchTrainEvalConfig(
        dataset="tau2",
        domain="airline",
        concurrency=80,
        commit_concurrency=90,
        commit_timeout_seconds=900,
    )

    assert args.concurrency == 80
    assert args.commit_concurrency == 90
    assert args.commit_timeout_seconds == 900
    assert config.concurrency == 80
    assert config.commit_concurrency == 90
    assert config.commit_timeout_seconds == 900


def test_server_headers_are_repeatable(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_batch_train_eval",
            "--dataset",
            "tau2",
            "--domain",
            "airline",
            "--server-header",
            "X-Tenant-ID=tenant-a",
            "--server-header",
            "Authorization: Bearer token==",
        ],
    )

    args = parse_args()

    assert dict(args.server_header) == {
        "X-Tenant-ID": "tenant-a",
        "Authorization": "Bearer token==",
    }


@pytest.mark.asyncio
async def test_casehub_selection_is_repeatable_and_reaches_batch_config(monkeypatch) -> None:
    captured: BatchTrainEvalConfig | None = None

    async def capture(config: BatchTrainEvalConfig) -> SimpleNamespace:
        nonlocal captured
        captured = config
        return SimpleNamespace(train_epochs=[])

    monkeypatch.setattr(batch_runner, "run_batch_train_eval", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_batch_train_eval",
            "--dataset",
            "ark4-0",
            "--domain",
            "ark",
            "--casehub-dataset-id",
            "dataset-1",
            "--casehub-case-id",
            "case-1",
            "--casehub-case-id",
            "case-2",
            "--casehub-eval-dataset-id",
            "dataset-eval",
        ],
    )

    assert await main_async() == 0
    assert captured is not None
    assert captured.casehub_dataset_ids == ["dataset-1"]
    assert captured.casehub_case_ids == ["case-1", "case-2"]
    assert captured.casehub_eval_dataset_ids == ["dataset-eval"]


def test_casehub_case_requires_dataset() -> None:
    with pytest.raises(ValueError, match="casehub_dataset_ids is required"):
        BatchTrainEvalConfig(
            dataset="ark4-0",
            domain="ark",
            casehub_case_ids=["case-1"],
        )


@pytest.mark.parametrize(
    "value",
    (
        "missing-separator",
        "Bad Header=value",
        "X-Empty=",
        "X-Newline=value\nnext",
    ),
)
def test_server_header_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_server_header(value)


def test_build_http_client_passes_server_headers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(batch_runner, "AsyncHTTPClient", FakeClient)
    config = BatchTrainEvalConfig(
        dataset="tau2",
        domain="airline",
        server_url="http://127.0.0.1:1933",
        api_key="test-key",
        server_headers={
            "X-Tenant-ID": "tenant-a",
            "Authorization": "Bearer token",
        },
    )

    client = batch_runner._build_http_client(config)

    assert isinstance(client, FakeClient)
    assert captured["extra_headers"] == config.server_headers


def test_build_http_client_preserves_config_headers_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(batch_runner, "AsyncHTTPClient", FakeClient)
    config = BatchTrainEvalConfig(
        dataset="tau2",
        domain="airline",
        server_url="http://127.0.0.1:1933",
        api_key="test-key",
    )

    batch_runner._build_http_client(config)

    assert "extra_headers" not in captured


def test_tau2_launchers_use_concurrency_200() -> None:
    launcher = (REPO_ROOT / "benchmark/tau2/train/run_batch_train_eval.sh").read_text()
    restart_launcher = (
        REPO_ROOT / "benchmark/tau2/train/restart_vikingbot_train_eval.sh"
    ).read_text()

    assert "--concurrency 200" in launcher
    assert "--commit-concurrency 200" in launcher
    assert "--commit-concurrency 200" in restart_launcher


def test_no_eval_each_epoch_overrides_tau2_wrapper_default(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_batch_train_eval",
            "--dataset",
            "tau2",
            "--domain",
            "airline",
            "--eval-each-epoch",
            "--no-eval-each-epoch",
        ],
    )

    args = parse_args()

    assert args.eval_each_epoch is False
    assert args.skip_final_eval is False
