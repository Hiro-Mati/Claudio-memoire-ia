"""HTTP service exposing Vaka cases with Vikingbot-backed rollouts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI

from openviking.session.train.components.dataset_service import create_dataset_service_app
from openviking.session.train.components.remote import RemoteCaseLoader

from .config import VakaVikingBotConfig
from .rollout_executor import VikingBotVakaRolloutExecutor


def create_app(config: VakaVikingBotConfig | None = None) -> FastAPI:
    """Build a service compatible with ``RemoteCaseLoader/RemoteRolloutExecutor``."""

    base_config = config or VakaVikingBotConfig.from_env()
    base_config.validate()

    def make_case_loader(
        dataset: str,
        domain: str,
        split: str,
        filters: dict[str, Any],
    ) -> RemoteCaseLoader:
        # The legacy service is used only as a case catalog. Rollout execution
        # always stays in this process and goes through Vikingbot chat.
        return RemoteCaseLoader(
            service_url=base_config.case_service_url,
            dataset=dataset,
            domain=domain,
            split=split,
            filters=filters,
        )

    def make_rollout_executor(options: dict[str, Any]) -> VikingBotVakaRolloutExecutor:
        rollout_config = _config_for_request(base_config, options)
        return VikingBotVakaRolloutExecutor(
            config=rollout_config,
            concurrency=1,
        )

    app = create_dataset_service_app(
        service_name="vaka-vikingbot",
        make_case_loader=make_case_loader,
        make_rollout_executor=make_rollout_executor,
        max_rollout_concurrency=base_config.max_rollout_concurrency,
        # Each AgentLoop contains blocking provider/tool setup. Running one
        # rollout event loop per worker keeps polling endpoints responsive.
        rollout_thread_workers=base_config.max_rollout_concurrency,
    )
    app.state.vaka_vikingbot_config = base_config
    return app


def _config_for_request(
    base: VakaVikingBotConfig,
    options: dict[str, Any],
) -> VakaVikingBotConfig:
    updates: dict[str, Any] = {}
    max_iterations = options.get("max_iterations")
    if max_iterations is not None:
        parsed_iterations = int(max_iterations)
        if parsed_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        updates["max_iterations"] = parsed_iterations

    config_path = str(options.get("config_path") or "").strip()
    if config_path:
        updates["vikingbot_config_path"] = Path(config_path).expanduser()

    return replace(base, **updates) if updates else base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
