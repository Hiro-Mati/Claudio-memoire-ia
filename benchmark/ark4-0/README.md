# Ark 4.0 Adapter

Adapter 和 `run_batch_train_eval` 是两个独立进程。

- Adapter 常驻，只负责 CaseHub、rollout 和平台 Task 接口转换。
- 每执行一次 `run_batch_train_eval`，就新建一个平台 Task。
- 本次 baseline、Train rollout、Train commit 和最终 eval 都使用这个 Task。
- runner 正常完成且 Train commit 没有错误后，自动结束这个 Task。
- runner 中途失败时不会发送“训练完成”，避免把失败任务标成成功。

## 配置

复制配置模板：

```bash
cp benchmark/ark4-0/adapter_config.example.json \
  benchmark/ark4-0/adapter_config.local.json
```

最小配置：

```json
{
  "platform": {
    "gateway_base_url": "https://replace-with-inspect-gateway",
    "api_key": "replace-with-platform-api-key",
    "project_id": "replace-with-32-character-project-id",
    "vaka_request_source": "ark-lx"
  },
  "rollout": {
    "runtime_params": {
      "memory": {
        "enabled": true,
        "mode": "read_only",
        "openviking_target": "evolving-dutao"
      }
    }
  },
  "memory_proxy": {
    "enabled": true,
    "openviking_config_file": "/absolute/path/to/openviking.conf",
    "openviking_api_key_json_path": "bot.ov_server.api_key"
  },
  "kubevpn": {
    "kubeconfig": "/absolute/path/to/kubeconfig",
    "namespace": "ai-search-rec"
  }
}
```

说明：

- `project_id` 是 32 位项目 ID，不是项目名称。
- `vaka_request_source` 会作为 `x-vaka-request-source` Header，自动添加到 Adapter 发出的所有平台请求。
- CaseHub 数据集和 Case 不再写在 Adapter 配置里，由每次 runner 的参数决定。
- `openviking_target` 同时决定远端回调目标和 KubeVPN 目标。
- Adapter 不再需要 `state_file`，也不需要提前填写 `task_id`。
- `--config` 可选；两个启动脚本默认读取同目录的 `adapter_config.local.json`。

## 执行

先启动本地 OpenViking，再开两个终端。

终端 1，启动 Adapter：

```bash
bash benchmark/ark4-0/start_adapter.sh
```

终端 2，开启 KubeVPN：

```bash
bash benchmark/ark4-0/start_kubevpn_proxy.sh
```

查看状态：

```bash
python3 benchmark/ark4-0/kubevpn_proxy.py status
curl http://127.0.0.1:1944/health
```

执行 runner：

```bash
.venv/bin/python -m openviking.session.train.run_batch_train_eval \
  --dataset ark4-0 \
  --domain ark \
  --casehub-dataset-id ds_85079cd1acb6481095e3a2d6c57cf9dc \
  --casehub-case-id case_0c8c2aaf15d940378e6f2742ac7b2f18 \
  --benchmark-service-url http://127.0.0.1:1944 \
  --server-url http://127.0.0.1:1933 \
  --api-key "<本地 OpenViking API Key>" \
  --epochs 1 \
  --batch-size 1 \
  --train-trials 1 \
  --concurrency 1 \
  --commit-concurrency 1 \
  --eval-split train \
  --skip-baseline-eval
```

runner 会打印本次 `task_id`，并写入报告的 `benchmark_task_id` 字段。无需手工调用完成接口。

Case 选择规则：

- `--casehub-dataset-id` 必填，可以重复传入多个数据集。
- `--casehub-case-id` 可选，可以重复传入多个 Case。
- 不传 `--casehub-case-id`，表示执行所选数据集里的全部 Case。
- 已经精确指定 Case 时，一般不再需要 `--train-index`。

停止 KubeVPN：

```bash
bash benchmark/ark4-0/stop_kubevpn_proxy.sh
```

停止 Adapter：在 Adapter 终端按 `Ctrl+C`。

## 调用链

```text
run_batch_train_eval
  -> POST /v1/runs/start
     携带本次 casehub dataset_ids 和 case_ids
  -> Adapter 校验并加载本次 Case
  -> Adapter 创建平台 Task，并等待 OV_WAIT
  -> 获取 Case
  -> baseline / Train rollout / Train commit / final eval
  -> POST /v1/runs/{run_id}/complete
  -> Adapter 通知平台 external-training-completed
```

每个 rollout 都带本次 `run_id`。Adapter 用它找到对应 `task_id`，所以一个 Adapter 可以连续或并行服务多个 runner，不会串 Task。

远端 Memory 回调地址：

```text
http://ov-proxy-evolving-dutao.ai-search-rec.svc.cluster.local:8765
  -> KubeVPN
  -> http://127.0.0.1:1944
  -> http://127.0.0.1:1933
```

## 排查

查看 Adapter 中的 run 和 task：

```bash
curl http://127.0.0.1:1944/admin/platform-runs
curl http://127.0.0.1:1944/admin/platform-runs/<run_id>
```

训练结果默认在：

```text
result/ark4-0/train/run_ark_<timestamp>/report.json
```

## 测试

```bash
.venv/bin/python -m pytest -q --no-cov benchmark/ark4-0/tests
ruff check benchmark/ark4-0 openviking/session/train/components/remote.py \
  openviking/session/train/batch_runner.py
```
