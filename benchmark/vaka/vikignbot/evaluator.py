"""Adapter around the existing evolution HomeTest evaluator implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from openviking.session.train import RubricEvaluation
from openviking.session.train.components.dataset_service import evaluation_from_dict

from .artifacts import HomepageArtifactUploader
from .config import VakaVikingBotConfig


class ReusedHomeTestEvaluator:
    """Upload local artifacts and delegate scoring to the old HomeTestJudge."""

    def __init__(
        self,
        config: VakaVikingBotConfig,
        *,
        uploader: HomepageArtifactUploader | None = None,
        judge: Any | None = None,
    ) -> None:
        self.config = config
        self.uploader = uploader
        self.judge = judge
        if not config.skip_scoring:
            self.uploader = uploader or HomepageArtifactUploader(config)
            self.judge = judge or _build_home_test_judge(config)

    async def evaluate(
        self,
        *,
        execution_id: str,
        prompt: str,
        dataset: str,
        final_answer: str,
        artifact_paths: list[Path],
    ) -> tuple[RubricEvaluation | None, list[dict[str, Any]], str]:
        if self.config.skip_scoring:
            return None, [], "skipped"
        assert self.uploader is not None
        assert self.judge is not None
        artifacts = await self.uploader.upload(artifact_paths)
        evaluation_data, status, error = await self.judge.judge(
            execution_id=execution_id,
            prompt=prompt,
            dataset=dataset,
            final_answer=final_answer,
            artifacts=artifacts,
        )
        if status == "failed":
            raise RuntimeError(f"HomeTest evaluator failed: {error or 'unknown error'}")
        evaluation = evaluation_from_dict(evaluation_data)
        return evaluation, artifacts, status


def _build_home_test_judge(config: VakaVikingBotConfig) -> Any:
    source_root = config.evolution_proxy_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from ov_benchmark_proxy.hometest_judge import HomeTestJudge
    except ImportError as exc:
        raise RuntimeError(f"cannot import HomeTestJudge from {source_root}") from exc

    return HomeTestJudge(
        home_test_root=config.home_test_root,
        python=config.home_test_python,
        template_paths=[
            config.home_test_template_train,
            config.home_test_template_eval,
        ],
        timeout_seconds=config.evaluator_timeout_seconds,
    )
