# Vaka rollout through Vikingbot chat

This adapter keeps the existing Vaka case catalog and `home_test` evaluator,
but replaces remote Homepage rollout execution with a local Vikingbot chat.

For every rollout it:

1. creates a new UUID-backed Vikingbot session;
2. creates an isolated `per-session` workspace;
3. copies case reference files into `input/` and sends the task through
   `AgentLoop.process_direct`;
4. collects this session's generated deliverables;
5. uploads those deliverables and delegates scoring to the existing
   `HomeTestJudge` implementation;
6. returns the ordinary OpenViking `Rollout` structure, so Train/Eval and
   report generation do not need changes.

## Prerequisites

- The original Vaka proxy is running on `127.0.0.1:8765`. It is queried only
  for `/v1/cases/query`; its rollout endpoint is not used.
- `home_test` and `evolution/components/ov-benchmark-proxy` exist beside this
  OpenViking checkout.
- The proxy `.env` contains a usable `VAKA_USERS_FILE` or this adapter receives
  `VAKA_VIKINGBOT_HOMEPAGE_API_KEY` and
  `VAKA_VIKINGBOT_HOMEPAGE_USER_ID`.
- The OpenViking config passed to `run_batch_train_eval --config` contains the
  Vikingbot `bot`/`vlm` settings needed by the chat runtime.

## Start

```bash
cd /Users/bytedance/work/space/OV-train/OpenViking
./benchmark/vaka/vikignbot/run.bash
```

The adapter listens on `http://127.0.0.1:8766`. Check it with:

```bash
curl -s http://127.0.0.1:8766/health | jq
```

## Run one Eval case

Use the normal runner and change only `--benchmark-service-url`:

```bash
python3 -m openviking.session.train.run_batch_train_eval \
  --dataset vaka_dev_v1 \
  --domain benchmark \
  --config /Users/bytedance/.openviking/ov-eval-vaka.conf \
  --server-url http://127.0.0.1:1933 \
  --api-key "$OV_API_KEY" \
  --benchmark-service-url http://127.0.0.1:8766 \
  --epochs 0 \
  --skip-baseline-eval \
  --eval-split test \
  --eval-index 19 \
  --trials 1 \
  --batch-size 1 \
  --concurrency 1 \
  --output temp/vikingbot_case19.json \
  --events-output temp/vikingbot_case19_events.json \
  --result-dir-name vikingbot_case19
```

The returned rollout metadata includes `vikingbot_session_id`,
`vikingbot_workspace`, artifact details, token usage, execution duration, and
evaluator status. Session workspaces are retained by default for diagnosis.
