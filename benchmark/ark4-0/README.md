# Ark 4.0 standalone training adapter

This directory exposes the Ark external-training platform through the generic remote service
contract consumed by OpenViking's native `run_batch_train_eval` command. The adapter and the
trainer are deliberately separate processes.

The adapter owns the platform-facing lifecycle:

1. Create a platform `ov_external_training` Task, or resume a configured/state-file Task.
2. Wait until the Task reaches `OV_WAIT`.
3. Load CaseHub cases.
4. Serve the generic Case and Rollout endpoints on a fixed address.
5. Translate OpenViking rollout requests to platform rollout-eval requests and poll through the
   same gateway.
6. Expose an explicit local admin endpoint for `external-training-completed`.

The adapter never starts OpenViking training and never sends the completion signal merely because
its process exits. Start OpenViking's native runner separately, inspect its result, and then call
the completion admin endpoint.

## 1. Create a local configuration

Copy the example; local JSON files are git-ignored because they contain credentials:

```bash
cp benchmark/ark4-0/adapter_config.example.json \
  benchmark/ark4-0/adapter_config.local.json
```

Edit `adapter_config.local.json`:

- `platform.gateway_base_url`: gateway origin only; do not append `/inspect`.
- `platform.api_key` and `platform.project_id`: gateway authentication and project selection.
  `project_id` must be the 32-character id returned by `GET /api/projects`, not the project
  display name. For example, `ov-ark-test` is a name and cannot be put in this field.
- `platform.headers`: optional additional gateway headers.
- `casehub.dataset_ids`: dataset ids used for Task creation and CaseHub loading.
- `casehub.case_ids`: optional case id whitelist for one-case/small-batch testing.
- `casehub.caseset_id`: optional caseset restriction.
- `training_task.task_id`: leave empty to create a Task; set it to resume a specific Task.
- `service.host` / `service.port`: fixed address given to OpenViking.
- `service.admin_token`: required in the `X-Ark4-Admin-Token` header when configured.
- `memory_proxy`: Tool Server callback proxy. It adds the local OpenViking user API key and
  exposes the four `/api/v1/search/*`, `/api/v1/content/*`, and `/api/v1/fs/stat` compatibility
  routes on the same adapter port.

All adapter behavior and platform credentials come from this JSON file. The adapter does not read
`ARK4_*` environment variables.

The `state_file` path is resolved relative to the configuration file. A newly created Task id is
written there. On restart, an empty `training_task.task_id` resumes the Task recorded in that state
file. Remove the local state file when intentionally creating a fresh Task.

## 2. Start the adapter

```bash
bash benchmark/ark4-0/start_adapter.sh
```

`--config` is optional. Without it, the launcher reads `adapter_config.local.json` next to
`start_adapter.sh`. To use another file:

```bash
bash benchmark/ark4-0/start_adapter.sh --config /path/to/another-adapter.json
```

Expected startup output:

```text
[ark4-adapter] created platform task task_xxx
[ark4-adapter] platform task task_xxx is ready at OV_WAIT
[ark4-adapter] loaded N CaseHub case(s)
[ark4-adapter] listening at http://127.0.0.1:1944
[ark4-adapter] OpenViking argument: --benchmark-service-url http://127.0.0.1:1944
```

The process stays running while OpenViking executes rollouts.

Check the adapter and platform Task from another terminal:

```bash
curl http://127.0.0.1:1944/health

curl http://127.0.0.1:1944/admin/platform-task \
  -H 'X-Ark4-Admin-Token: replace-with-a-local-admin-token'
```

## 3. Run OpenViking's native trainer separately

Make sure the OpenViking server is already running, then invoke the repository's original runner:

```bash
python3 -m openviking.session.train.run_batch_train_eval \
  --dataset ark4-0 \
  --domain ark \
  --benchmark-service-url http://127.0.0.1:1944 \
  --server-url http://127.0.0.1:1933 \
  --api-key "$(jq -r '.bot.ov_server.api_key' /Users/bytedance/.openviking/ov-eval-vaka.conf)" \
  --epochs 1 \
  --train-index 0 \
  --batch-size 1 \
  --train-trials 1 \
  --concurrency 1 \
  --commit-concurrency 1 \
  --eval-split none \
  --skip-baseline-eval \
  --skip-final-eval
```

The `--api-key` must be a local OpenViking user/admin key. In `api_key` auth mode the server root
key can pass health checks but cannot create tenant-scoped training sessions.

This is the unmodified OpenViking CLI in
`openviking.session.train.run_batch_train_eval`; the Ark adapter only supplies its
`--benchmark-service-url`.

Training output is written under:

```text
result/ark4-0/train/run_ark_<timestamp>/
  report.json
  events.jsonl
  rollouts_index.json
  rollouts/
```

## 4. Send the completion signal explicitly

Only after `run_batch_train_eval` exits successfully and `report.json` has no training commit
errors:

```bash
curl -X POST http://127.0.0.1:1944/admin/external-training-completed \
  -H 'X-Ark4-Admin-Token: replace-with-a-local-admin-token'
```

The adapter translates this to:

```text
POST /inspect/training/tasks/{task_id}/signals/external-training-completed
```

The completion request uses a stable idempotency key. Do not complete a Task that still needs
additional training or evaluation; a completed Task cannot be resumed at `OV_WAIT`.

## Train/eval split

Without `casehub.split_field`, all cases belong to `casehub.default_split` (`train` by default), so
use `--eval-split none`.

If CaseHub envelopes contain, for example:

```json
{
  "metadata": {
    "split": "train"
  }
}
```

set this in the adapter configuration:

```json
{
  "casehub": {
    "split_field": "metadata.split"
  }
}
```

The native runner can then use `--eval-split test`. Supported aliases include
`training -> train`, `validation/valid -> dev`, and `evaluation/eval -> test`.

## Rollout message contract

Training requires the platform result to contain a useful trajectory in `result.messages`. The
adapter supports:

- OpenViking-style `{id, role, parts}` messages;
- OpenAI-style `{role, content, tool_calls}` messages;
- separate `role=tool` result messages.

When `rollout.require_messages_for_training` is `true`, an empty training trajectory fails instead
of silently learning only from `final_answer`. Set it to `false` only for temporary connectivity
testing; the adapter then synthesizes user/assistant messages.

## Tests

```bash
python3 -m pytest -q --no-cov benchmark/ark4-0/tests
ruff check benchmark/ark4-0
bash -n benchmark/ark4-0/start_adapter.sh
```

## OpenViking memory callback through KubeVPN

KubeVPN is used only for the Tool Server's OpenViking memory callback. It is not on the rollout
control path and it is not used to download traces. The actual data flow is:

```text
run_batch_train_eval -> local Ark adapter -> four platform APIs
  -> Ark Connector -> evolution Homepage/Agent/Tool Server
  -> openviking.target_urls[ov-ark-test]
  -> ov-proxy-ov-ark-test:8765
  -> KubeVPN -> local Ark adapter:1944
  -> local OpenViking:1933
```

Do not proxy `ai-search-rec-vaka-agent-server-evolution`. That workload is selected by the frozen
lane header, but it is not the local-memory callback target.

Three names must match exactly:

1. `rollout.runtime_params.memory.openviking_target` in `adapter_config.local.json`;
2. `memory_proxy.openviking_target` and `openviking.target_urls` in the evolution Tool Server's
   full Nacos runtime configuration;
3. the KubeVPN placeholder deployment name `ov-proxy-<openviking_target>`.

`openviking_target` is a Tool Server routing name and is independent of the platform Project.
It may be named `ov-ark-test` even when the training Task belongs to the default Project.

For this directory the intended target is `ov-ark-test`, so the required Tool Server entry is:

```json
{
  "openviking": {
    "target_urls": {
      "ov-ark-test": "http://ov-proxy-ov-ark-test.ai-search-rec.svc.cluster.local:8765"
    }
  }
}
```

The entry must be merged into the complete `tool_server_runtime_config_evolution`; do not replace
that Nacos document with this small fragment. The `ov-proxy-ov-ark-test` Deployment and Service
must also exist in namespace `ai-search-rec` before starting KubeVPN.

The matching Deployment and Service manifest is provided as
`kubevpn_target.example.yaml`. Review it before applying it to the shared STG cluster:

```bash
kubectl --kubeconfig /Users/bytedance/work/space/OV-train/evolution/.kube/config_stg \
  diff -f benchmark/ark4-0/kubevpn_target.example.yaml

kubectl --kubeconfig /Users/bytedance/work/space/OV-train/evolution/.kube/config_stg \
  apply -f benchmark/ark4-0/kubevpn_target.example.yaml
```

`kubectl diff` is read-only. `kubectl apply` changes the shared cluster and should only be run
after the target owner approves it. The manifest mirrors evolution's existing dedicated
`ov-proxy-*` placeholder pattern.

### Header injection

The lane header is required on the rollout path, but it is not a KubeVPN matcher for the dedicated
`ov-proxy-*` callback service:

- The Task's frozen binding injects `x-tt-backend: evolution`. It is protected and cannot appear
  in `rollout.extra_header`.
- `x-tt-sandbox.env.VAKA_REQUEST_SOURCE=ov-ark-test` selects the isolated source profile.
- `X-OpenViking-Target: ov-ark-test`/the hydrated target selects the Tool Server callback target.
- Neither header turns memory on. They only take effect after the Connector has set
  `extra.memory.enabled=true`.

The platform API documentation says `runtime_params` is Connector-defined. The adapter can send
the following block, but the currently deployed `ark@1` Connector does not consume it as a memory
enable switch:

```json
{
  "rollout": {
    "runtime_params": {
      "memory": {
        "enabled": true,
        "mode": "read_only",
        "openviking_target": "ov-ark-test"
      }
    },
    "extra_header": {
      "x-vaka-request-source": "vaka-agentmemory"
    }
  }
}
```

This limitation was verified against the default Project on 2026-08-09: Agent Server trace data
contained `extra.memory.enabled=false` while `openviking_target=ov-ark-test` was present. Variants
under `runtime_params`, `runtime_config`, `connector_config.options`, and `extra` produced the same
result. The Connector's published execution contract currently exposes only `model_ep`.

Consequently, the four external-training APIs and native OpenViking training work end to end, but
automatic memory recall will not call the local proxy until the `ark@1` Connector exposes and
honours a memory-enable field. KubeVPN and the target mapping can be validated independently; do
not treat `call_count=0` as a KubeVPN failure when the Agent trace says memory is disabled.

### Local OpenViking authentication

Tool Server does not know the local OpenViking user API key. The adapter's `memory_proxy` adds it
before forwarding to `http://127.0.0.1:1933`. Put the key directly in the ignored local JSON, or
read it from an existing OpenViking JSON config without an environment variable:

```json
{
  "memory_proxy": {
    "enabled": true,
    "openviking_target": "ov-ark-test",
    "openviking_url": "http://127.0.0.1:1933",
    "openviking_api_key": "",
    "openviking_config_file": "/Users/bytedance/.openviking/ov-eval-vaka.conf",
    "openviking_api_key_json_path": "bot.ov_server.api_key",
    "event_log_file": "memory_proxy_events.local.jsonl"
  }
}
```

The event log records only path, status, field names, timing, and a request hash. It does not
persist the API key or the query/body values.

### Start and verify

Create the local proxy config:

```bash
cp benchmark/ark4-0/kubevpn_config.example.json \
  benchmark/ark4-0/kubevpn_config.local.json
```

Start the local OpenViking server first. Then start the adapter, KubeVPN, and the native trainer as
three separate processes:

```bash
bash benchmark/ark4-0/start_adapter.sh
bash benchmark/ark4-0/start_kubevpn_proxy.sh
python3 -m openviking.session.train.run_batch_train_eval \
  --dataset ark4-0 \
  --domain ark \
  --benchmark-service-url http://127.0.0.1:1944 \
  --server-url http://127.0.0.1:1933 \
  --api-key "$(jq -r '.bot.ov_server.api_key' /Users/bytedance/.openviking/ov-eval-vaka.conf)" \
  --epochs 1 --train-index 0 --batch-size 1 --train-trials 1 \
  --concurrency 1 --commit-concurrency 1 \
  --eval-split none --skip-baseline-eval --skip-final-eval
```

Check local callback status and evidence:

```bash
curl http://127.0.0.1:1944/admin/memory-proxy \
  -H 'X-Ark4-Admin-Token: replace-with-a-local-admin-token'

tail -f benchmark/ark4-0/memory_proxy_events.local.jsonl
```

To prove the reverse path independently of Agent auto-recall, send the same source Header from an
STG Tool Server Pod through the dedicated callback Service:

```bash
POD=$(kubectl --kubeconfig /Users/bytedance/work/space/OV-train/evolution/.kube/config_stg \
  -n ai-search-rec get pod \
  -l app.kubernetes.io/instance=ai-search-rec-tool-server-evolution \
  -o jsonpath='{.items[0].metadata.name}')

kubectl --kubeconfig /Users/bytedance/work/space/OV-train/evolution/.kube/config_stg \
  -n ai-search-rec exec "$POD" -- curl -sS -m 30 -X POST \
  -H 'Content-Type: application/json' \
  -H 'x-vaka-request-source: vaka-agentmemory' \
  -d '{"query":"kubevpn network proof","limit":1}' \
  http://ov-proxy-ov-ark-test.ai-search-rec.svc.cluster.local:8765/api/v1/search/find
```

A response containing `result`, plus a new event with `status_code=200`, proves the complete
`STG Pod -> ov-proxy Service -> KubeVPN -> local adapter -> local OpenViking` path.

`call_count > 0` or a new event proves the native rollout's Tool Server reached this machine. A
successful event then proves the adapter also reached the authenticated local OpenViking server.

The optional `probe.enabled=true` mode in `kubevpn_config.local.json` is only for an isolated
network proof. It starts `traffic_probe.py` on `local_port` and deliberately returns HTTP 503 with
marker `ARK4_LOCAL_TRAFFIC_PROBE`; do not use probe mode for a real training run.

Inspect or stop it with:

```bash
python3 benchmark/ark4-0/kubevpn_proxy.py status
bash benchmark/ark4-0/stop_kubevpn_proxy.sh
```

`--config` is optional for both KubeVPN launchers. Their default is
`kubevpn_config.local.json` next to the scripts.
