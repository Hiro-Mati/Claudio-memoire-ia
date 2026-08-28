# 外部 Compile Server 对接说明

## 整体流程

```text
ov compile
  -> OpenViking POST /api/v1/compile
  -> OV 创建 TaskRecord 并写入 ExternalTask QueueFS
  -> OV 调用外部 POST /bot/v1/compile
  -> 外部 Server 使用请求中的 OV 地址和用户 API Key 读写产物
  -> OV 轮询外部任务并更新统一 TaskRecord
  -> ov task status/cancel 通过 /api/v1/tasks 查询或取消
```

OV 负责持久化任务、重启恢复、重试、统一查询和取消。外部 Server 负责实际 Compile 执行和自己的执行状态。OV Task ID 与外部 Task ID 是两个 ID，CLI 只看到 OV Task ID。

## 外部 Server 必须实现

### 1. 创建任务

```http
POST /bot/v1/compile
Authorization: Bearer <compile_api.api_key>
Idempotency-Key: <OV task_id，例如 cmp_abc>
X-API-Key: <current user OV API key>
```

```json
{
  "from": ["viking://resources/source"],
  "to": "viking://resources/wiki",
  "skill": "viking://agent/skills/wiki",
  "reason": "optional",
  "runtime_timeout_seconds": 3600,
  "openviking_connection": {
    "server_url": "https://ov.example.com",
    "api_key": "current-user-key",
    "account_id": "account",
    "user_id": "user",
    "role": "user",
    "agent_id": "web-playground",
    "api_key_type": "user",
    "namespace_policy": {
      "isolate_user_scope_by_agent": false,
      "isolate_agent_scope_by_user": false
    }
  }
}
```

返回 HTTP 202，响应不要套 OV 的 `status/result` envelope：

```json
{
  "task_id": "external_cmp_123",
  "status": "accepted",
  "to": "viking://resources/wiki"
}
```

`Idempotency-Key` 是强制契约。同一个 Key 重复提交时必须返回同一个外部 Task ID，不能重复执行。OV 在进程崩溃或网络响应丢失后会重新提交。

### 2. 查询任务

```http
GET /bot/v1/compile/{external_task_id}
Authorization: Bearer <compile_api.api_key>
X-API-Key: <current user OV API key>
```

```json
{
  "task_id": "external_cmp_123",
  "status": "running",
  "stage": "agent"
}
```

支持的状态：`accepted`、`pending`、`running`、`committing`、`cancelling`、`completed`、`failed`、`cancelled`。

完成时必须返回 Compile 结果：

```json
{
  "task_id": "external_cmp_123",
  "status": "completed",
  "stage": "completed",
  "result": {
    "from": ["viking://resources/source"],
    "to": "viking://resources/wiki",
    "skill": "viking://agent/skills/wiki",
    "okf_version": "0.1",
    "created": [],
    "updated": ["viking://resources/wiki/index.md"],
    "unchanged": [],
    "page_count": 1,
    "link_count": 0,
    "warnings": []
  }
}
```

失败时返回：

```json
{
  "task_id": "external_cmp_123",
  "status": "failed",
  "stage": "writing",
  "error": {"code": "WRITE_FAILED", "message": "write failed"}
}
```

### 3. 取消任务

```http
POST /bot/v1/compile/{external_task_id}/cancel
Authorization: Bearer <compile_api.api_key>
X-API-Key: <current user OV API key>
```

返回与查询接口相同的任务结构。接口需要幂等；任务已经取消时继续返回 `cancelled`。

## 写回 OpenViking

外部 Server 使用 `openviking_connection.server_url` 选择 OV 实例，并使用其中的 `api_key` 调用 OV。不要使用 Compile Server 自己的固定 OV Key，否则多 OV 实例和多用户场景会写错位置或越权。

`openviking_connection` 包含用户凭证，禁止写日志。多实例 Compile Server 需要使用共享任务存储，以外部 Task ID 和 `Idempotency-Key` 建唯一索引；任务结束后按安全策略清理凭证。

## 错误与重试

- `408`、`425`、`429` 和 `5xx`：OV 认为是暂时错误并重试。
- 其他 `4xx`：OV 认为是永久错误并结束任务。
- 错误响应使用 `{"detail":{"code":"...","message":"..."}}` 或 `{"error":{"code":"...","message":"..."}}`。
- 外部响应必须是 JSON；完成状态缺少 `result` 会被 OV 判为 `INVALID_RESPONSE`。

OV 配置：

```json
{
  "compile_api": {
    "enable": true,
    "host": "https://compile.example.com",
    "api_key": "$OPENVIKING_COMPILE_API_KEY",
    "http_timeout_seconds": 10,
    "poll_interval_ms": 3000
  }
}
```
