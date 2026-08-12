"""Configuration for the Vaka Vikingbot rollout adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OV_TRAIN_ROOT = REPO_ROOT.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


@dataclass(slots=True, frozen=True)
class VakaVikingBotConfig:
    """Runtime paths and endpoints used by the adapter.

    The old proxy remains the source of Vaka cases and ``HomeTestJudge``. Only
    rollout execution moves to Vikingbot.
    """

    case_service_url: str = "http://127.0.0.1:8765"
    vikingbot_config_path: Path | None = None
    session_prefix: str = "vaka_rollout"
    rollout_timeout_seconds: float = 3600.0
    max_iterations: int = 30
    max_rollout_concurrency: int = 20
    keep_session_workspaces: bool = True

    skip_scoring: bool = False
    evolution_proxy_root: Path = OV_TRAIN_ROOT / "evolution/components/ov-benchmark-proxy"
    home_test_root: Path = OV_TRAIN_ROOT / "home_test"
    home_test_python: Path = OV_TRAIN_ROOT / "home_test/.venv/bin/python"
    home_test_template_train: Path = (
        OV_TRAIN_ROOT
        / "evolution/components/ov-benchmark-proxy/home_test_templates/mem_train_template.xlsx"
    )
    home_test_template_eval: Path = (
        OV_TRAIN_ROOT
        / "evolution/components/ov-benchmark-proxy/home_test_templates/mem_eval_template.xlsx"
    )
    evaluator_timeout_seconds: float = 2400.0

    homepage_base_url: str = "https://aisearch-stg-api.console.volcengine.com"
    homepage_api_key: str = ""
    homepage_user_id: str = ""
    vaka_users_file: Path | None = None
    artifact_upload_timeout_seconds: float = 1200.0

    @classmethod
    def from_env(cls) -> "VakaVikingBotConfig":
        config_value = os.getenv("VAKA_VIKINGBOT_CONFIG") or os.getenv("OPENVIKING_CONFIG_FILE")
        users_value = os.getenv("VAKA_VIKINGBOT_USERS_FILE") or os.getenv("VAKA_USERS_FILE")
        return cls(
            case_service_url=os.getenv(
                "VAKA_VIKINGBOT_CASE_SERVICE_URL", "http://127.0.0.1:8765"
            ).rstrip("/"),
            vikingbot_config_path=(Path(config_value).expanduser() if config_value else None),
            session_prefix=os.getenv("VAKA_VIKINGBOT_SESSION_PREFIX", "vaka_rollout"),
            rollout_timeout_seconds=float(
                os.getenv("VAKA_VIKINGBOT_ROLLOUT_TIMEOUT_SECONDS", "3600")
            ),
            max_iterations=int(os.getenv("VAKA_VIKINGBOT_MAX_ITERATIONS", "30")),
            max_rollout_concurrency=int(os.getenv("VAKA_VIKINGBOT_MAX_CONCURRENCY", "20")),
            keep_session_workspaces=_env_bool("VAKA_VIKINGBOT_KEEP_SESSION_WORKSPACES", True),
            skip_scoring=_env_bool("VAKA_VIKINGBOT_SKIP_SCORING", False),
            evolution_proxy_root=Path(
                os.getenv(
                    "VAKA_VIKINGBOT_EVOLUTION_PROXY_ROOT",
                    str(OV_TRAIN_ROOT / "evolution/components/ov-benchmark-proxy"),
                )
            ).expanduser(),
            home_test_root=Path(
                os.getenv("HOME_TEST_ROOT", str(OV_TRAIN_ROOT / "home_test"))
            ).expanduser(),
            home_test_python=Path(
                os.getenv(
                    "HOME_TEST_PYTHON",
                    str(OV_TRAIN_ROOT / "home_test/.venv/bin/python"),
                )
            ).expanduser(),
            home_test_template_train=Path(
                os.getenv(
                    "HOME_TEST_TEMPLATE_TRAIN",
                    str(
                        OV_TRAIN_ROOT / "evolution/components/ov-benchmark-proxy/"
                        "home_test_templates/mem_train_template.xlsx"
                    ),
                )
            ).expanduser(),
            home_test_template_eval=Path(
                os.getenv(
                    "HOME_TEST_TEMPLATE_EVAL",
                    str(
                        OV_TRAIN_ROOT / "evolution/components/ov-benchmark-proxy/"
                        "home_test_templates/mem_eval_template.xlsx"
                    ),
                )
            ).expanduser(),
            evaluator_timeout_seconds=float(
                os.getenv("VAKA_VIKINGBOT_EVALUATOR_TIMEOUT_SECONDS", "2400")
            ),
            homepage_base_url=os.getenv(
                "VAKA_VIKINGBOT_HOMEPAGE_BASE_URL",
                "https://aisearch-stg-api.console.volcengine.com",
            ).rstrip("/"),
            homepage_api_key=os.getenv("VAKA_VIKINGBOT_HOMEPAGE_API_KEY", ""),
            homepage_user_id=os.getenv("VAKA_VIKINGBOT_HOMEPAGE_USER_ID", ""),
            vaka_users_file=Path(users_value).expanduser() if users_value else None,
            artifact_upload_timeout_seconds=float(
                os.getenv("VAKA_VIKINGBOT_ARTIFACT_UPLOAD_TIMEOUT_SECONDS", "1200")
            ),
        )

    def validate(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        if self.max_rollout_concurrency <= 0:
            raise ValueError("max_rollout_concurrency must be > 0")
        if self.rollout_timeout_seconds <= 0:
            raise ValueError("rollout_timeout_seconds must be > 0")
        if self.skip_scoring:
            return
        required_paths = {
            "evolution_proxy_root": self.evolution_proxy_root,
            "home_test_root": self.home_test_root,
            "home_test_python": self.home_test_python,
            "home_test_template_train": self.home_test_template_train,
            "home_test_template_eval": self.home_test_template_eval,
        }
        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing evaluator dependency paths: " + ", ".join(missing))
