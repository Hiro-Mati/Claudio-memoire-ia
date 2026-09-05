# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Local memory evaluation on a LoCoMo subset, against the running server.

Before adopting a model, a prompt or a retrieval setting, measure it. This
script runs the existing LoCoMo evaluation flow (benchmark/locomo/openviking)
on a small subset with the *current* ov.conf, judges the answers with a local
Ollama model, and stores a summary tagged with the configuration fingerprint
so two runs can be compared.

Stages (each can be skipped once done):

  1. data     download benchmark/locomo/data/locomo10.json (2.8 MB) if missing
  2. import   ingest N LoCoMo samples into isolated users sample_{i}
  3. eval     answer Q questions per sample with the server's vlm
  4. judge    grade answers with an OpenAI-compatible judge (Ollama by default)
  5. stat     aggregate scores and write result/eval_local_<ts>.json

Expect long runtimes on CPU-only machines: importing one sample means
extracting memories from ~20 conversation sessions with the local VLM.
Start with ``--samples 1 --questions 5`` and ``--check`` to validate the setup.

Usage:
    python scripts/eval_local.py --check
    python scripts/eval_local.py --samples 1 --questions 5
    python scripts/eval_local.py --samples 1 --questions 5 --skip-import
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark" / "locomo"
FLOW = BENCH / "openviking"
DATA = BENCH / "data" / "locomo10.json"
RESULT = FLOW / "result"
DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def log(msg: str) -> None:
    print(f"[eval-local] {msg}", flush=True)


def run(cmd: list[str], cwd: Path) -> None:
    log("+ " + " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], cwd=str(cwd))


def config_fingerprint() -> dict:
    """The parts of ov.conf that change memory quality, for run comparison."""
    try:
        from openviking_cli.utils.config import get_openviking_config

        cfg = get_openviking_config()
        dense = (cfg.embedding.model_dump() or {}).get("dense") or {}
        snapshot = {
            "vlm": cfg.vlm.model,
            "file_summarizer": cfg.get_file_summarizer().model,
            "query_planner": cfg.get_query_planner().model,
            "embedding": dense.get("model"),
            "rerank": getattr(cfg.rerank, "model", None) or getattr(cfg.rerank, "model_name", None),
            "retrieval": cfg.retrieval.model_dump(),
            "semantic.parent_refresh_mode": cfg.semantic.parent_refresh_mode,
        }
    except Exception as exc:  # pragma: no cover - config optional for --check
        snapshot = {"error": str(exc)}
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()
    return {"fingerprint": digest[:12], "config": snapshot}


def ensure_data() -> Path:
    if DATA.exists():
        return DATA
    DATA.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading LoCoMo dataset to {DATA}")
    with urllib.request.urlopen(DATA_URL, timeout=60) as resp, DATA.open("wb") as out:
        out.write(resp.read())
    return DATA


def server_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples", type=int, default=1, help="LoCoMo samples to use (from 0)")
    parser.add_argument("--questions", type=int, default=5, help="questions per sample")
    parser.add_argument("--openviking-url", default="http://127.0.0.1:1933")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--judge-token", default="no-key")
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--check", action="store_true", help="validate setup and exit")
    parser.add_argument("--label", default="", help="free label stored in the summary")
    args = parser.parse_args()

    RESULT.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint()
    log(
        f"config fingerprint {fingerprint['fingerprint']}: {json.dumps(fingerprint['config'], default=str)[:300]}"
    )

    if not server_ok(args.openviking_url):
        log(f"server not reachable at {args.openviking_url}; start openviking-server first")
        return 2
    data = ensure_data()
    log(f"dataset ready: {data} ({data.stat().st_size // 1024} KB)")
    if args.check:
        log("setup looks good")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    qa_csv = RESULT / f"eval_local_{stamp}.csv"
    t0 = time.monotonic()

    if not args.skip_import:
        for idx in range(args.samples):
            run(
                [
                    sys.executable,
                    FLOW / "import_to_ov.py",
                    "--input",
                    data,
                    "--sample",
                    str(idx),
                    "--openviking-url",
                    args.openviking_url,
                    "--success-csv",
                    RESULT / f"eval_local_{stamp}_import.csv",
                    "--error-log",
                    RESULT / f"eval_local_{stamp}_import_errors.log",
                ],
                cwd=FLOW,
            )
    import_s = time.monotonic() - t0

    if not args.skip_eval:
        for idx in range(args.samples):
            run(
                [
                    sys.executable,
                    FLOW / "run_eval.py",
                    data,
                    "--output",
                    qa_csv,
                    "--sample",
                    str(idx),
                    "--count",
                    str(args.questions),
                    "--threads",
                    "1",
                    "--openviking-url",
                    args.openviking_url,
                    "--single-search-context-limit",
                    "30",
                    "--single-search-rerank-limit",
                    "0",
                    "--single-search-max-context-chars",
                    "20000",
                    "--update-mode",
                ],
                cwd=FLOW,
            )
        run(
            [
                sys.executable,
                FLOW / "judge.py",
                "--input",
                qa_csv,
                "--provider",
                "openai",
                "--base-url",
                args.judge_base_url,
                "--token",
                args.judge_token,
                "--model",
                args.judge_model,
                "--parallel",
                "1",
            ],
            cwd=FLOW,
        )
        run([sys.executable, FLOW / "stat_judge_result.py", "--input", qa_csv], cwd=FLOW)

    summary = {
        "timestamp": stamp,
        "label": args.label,
        "samples": args.samples,
        "questions_per_sample": args.questions,
        "judge_model": args.judge_model,
        "qa_csv": str(qa_csv),
        "import_seconds": round(import_s, 1),
        "total_seconds": round(time.monotonic() - t0, 1),
        **fingerprint,
    }
    out = RESULT / f"eval_local_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log(f"summary written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
