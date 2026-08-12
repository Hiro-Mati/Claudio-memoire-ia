"""Vaka benchmark adapter backed by the Vikingbot chat runtime."""

from typing import Any

from .config import VakaVikingBotConfig
from .rollout_executor import VikingBotVakaRolloutExecutor


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Import the HTTP layer lazily so ``python -m ...service_app`` stays clean."""

    from .service_app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = [
    "VakaVikingBotConfig",
    "VikingBotVakaRolloutExecutor",
    "create_app",
]
