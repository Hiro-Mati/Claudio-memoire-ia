# 服务端部署

OpenViking 可以作为独立的 HTTP 服务器运行，允许多个客户端通过网络连接。

## 快速开始

```bash
# 使用初始化向导创建或刷新 ~/.openviking/ov.conf
openviking-server init

# 如果你在向导中选择 OpenAI Codex，init 会帮你处理 Codex 登录/导入

# 启动前校验本地配置、模型访问和鉴权状态
openviking-server doctor

# 配置文件在默认路径 ~/.openviking/ov.conf 时，直接启动
openviking-server

# 配置文件在其他位置时，通过 --config 指定
openviking-server --config /path/to/ov.conf

# 验证服务器是否运行
curl http://localhost:1933/health
# {"status": "ok"}
```

## 命令行选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `~/.openviking/ov.conf` |
| `--host` | 绑定的主机地址 | `127.0.0.1` |
| `--port` | 绑定的端口 | `1933` |

**示例**

```bash
# 使用默认配置
openviking-server

# 使用自定义端口
openviking-server --port 8000

# 指定配置文件、主机地址和端口
openviking-server --config /path/to/ov.conf --host 127.0.0.1 --port 8000
```

## 配置

服务端从 `ov.conf` 读取所有配置。配置文件各段详情见 [配置指南](01-configuration.md)。

`ov.conf` 中的 `server` 段控制服务端行为：

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 1933,
    "root_api_key": "your-secret-root-key",
    "cors_origins": ["*"]
  },
  "storage": {
    "workspace": "./data",
    "agfs": { "backend": "local" },
    "vectordb": { "backend": "local" }
  }
}
```

## 部署模式

### 独立模式（嵌入存储）

服务器管理本地 RAGFS 和 VectorDB。在 `ov.conf` 中配置本地存储路径：

```json
{
  "storage": {
    "workspace": "./data",
    "agfs": { "backend": "local" },
    "vectordb": { "backend": "local" }
  }
}
```

```bash
openviking-server
```

## 使用 Systemd 部署服务（推荐）

对于 Linux 系统，可以使用 Systemd 服务来管理 OpenViking，实现自动重启、开机自启等功能。首先，你应该已经成功安装并配置了 OpenViking 服务器，确保它可以正常运行，再进行服务化部署。

### 创建 Systemd 服务文件

创建 `/etc/systemd/system/openviking.service` 文件：

```ini
[Unit]
Description=OpenViking HTTP Server
After=network.target

[Service]
Type=simple
# 替换为运行 OpenViking 的用户
User=your-username
# 替换为用户组
Group=your-group
# 替换为工作目录
WorkingDirectory=/var/lib/openviking
# 以下两种启动方式二选一
ExecStart=/path/to/your/python/bin/openviking-server
Restart=always
RestartSec=5
# 配置文件路径
Environment="OPENVIKING_CONFIG_FILE=/etc/openviking/ov.conf"

[Install]
WantedBy=multi-user.target
```

### 管理服务

创建好服务文件后，使用以下命令管理 OpenViking 服务：

```bash
# 重载 systemd 配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start openviking.service

# 设置开机自启
sudo systemctl enable openviking.service

# 查看服务状态
sudo systemctl status openviking.service

# 查看服务日志
sudo journalctl -u openviking.service -f
```

## 连接客户端

### Python SDK

```python
import openviking as ov

client = ov.SyncHTTPClient(url="http://localhost:1933", api_key="your-key")
client.initialize()

results = client.find(query="how to use openviking")
client.close()
```

### CLI

CLI 从 `ovcli.conf` 读取连接配置。在 `~/.openviking/ovcli.conf` 中配置：

```json
{
  "url": "http://localhost:1933",
  "api_key": "your-key"
}
```

也可通过 `OPENVIKING_CLI_CONFIG_FILE` 环境变量指定配置文件路径：

```bash
export OPENVIKING_CLI_CONFIG_FILE=/path/to/ovcli.conf
```

### curl

```bash
curl http://localhost:1933/api/v1/fs/ls?uri=viking:// \
  -H "X-API-Key: your-key"
```

## 云原生部署

### Docker

OpenViking 提供预构建的 Docker 镜像，发布在 GitHub Container Registry。容器内所有持久化状态（`ov.conf`、`ovcli.conf` 以及工作区数据）都放在 `/app/.openviking` 下，挂载一个目录即可：

```bash
docker run -d \
  --name openviking \
  -p 1933:1933 \
  -v ~/.openviking:/app/.openviking \
  --restart unless-stopped \
  ghcr.io/volcengine/openviking:latest
```

Docker 镜像默认会同时启动：
- OpenViking HTTP 服务，端口 `1933`（绑定 `0.0.0.0`），同时在 `/studio` 提供 Web Studio 前端
- `vikingbot` gateway

由于容器内服务绑定 `0.0.0.0`（Docker 端口映射所必需），你**必须**在 `ov.conf` 中设置 `root_api_key`：

```json
{
  "server": {
    "root_api_key": "your-secret-root-key"
  }
}
```

未设置时服务将拒绝启动。如需自定义绑定地址，可通过环境变量 `OPENVIKING_SERVER_HOST` 覆盖。

升级容器的方式
```bash
docker stop openviking
docker pull ghcr.io/volcengine/openviking:latest
docker rm -f openviking
# 然后重新 docker run ...
```

如果你希望本次容器启动时关闭 `vikingbot`，可以使用下面任一方式：

```bash
docker run -d \
  --name openviking \
  -p 1933:1933 \
  -v ~/.openviking:/app/.openviking \
  --restart unless-stopped \
  ghcr.io/volcengine/openviking:latest \
  --without-bot
```

```bash
docker run -d \
  --name openviking \
  -e OPENVIKING_WITH_BOT=0 \
  -p 1933:1933 \
  -v ~/.openviking:/app/.openviking \
  --restart unless-stopped \
  ghcr.io/volcengine/openviking:latest
```

#### 无法使用 `docker -v` 时

部分托管平台（如 Railway、Fly.io、Heroku 这类 PaaS）不支持把宿主机目录绑定挂载进容器。这种环境下，如果容器启动时找不到 `ov.conf`，entrypoint 不会崩溃 —— 它会打印一段修复指引并阻塞等待文件出现。你可以选用以下两种方式之一：

**方案 A：通过 `OPENVIKING_CONF_CONTENT` 注入完整配置内容。** entrypoint 会在启动 server 前把这个环境变量的值写入到 `OPENVIKING_CONFIG_FILE`（默认 `/app/.openviking/ov.conf`）：

```bash
docker run -d \
  --name openviking \
  -p 1933:1933 \
  -e OPENVIKING_CONF_CONTENT="$(cat ~/.openviking/ov.conf)" \
  --restart unless-stopped \
  ghcr.io/volcengine/openviking:latest
```

**方案 B：容器起来之后再 `docker exec` 进去用向导配置。** 容器在等待 `ov.conf` 期间是存活的，`exec` 进去运行 setup wizard，它会按 `OPENVIKING_CONFIG_FILE` 写到 server 正在监听的位置：

```bash
docker exec -it openviking openviking-server init
```

`ov.conf` 出现后，entrypoint 会自动恢复并启动 server。

也可以使用 Docker Compose，项目根目录提供了 `docker-compose.yml`：

```bash
docker compose up -d
```

启动后可以访问：
- API 服务：`http://localhost:1933`
- Web Studio：`http://localhost:1933/studio`（与 API 同源）
- 兼容入口：`http://localhost:1934`（Caddy 反代到 1933，仅为已有部署保留）

### 多实例部署注意事项

多实例部署时，通常建议注意这几项配置：

- 把 `server.temp_upload.default_mode` 设为 `"shared"`，这样临时上传文件可以被其他副本消费。
- 只有在多个实例明确共享同一个 `storage.workspace` 时，才考虑把 `storage.skip_process_lock` 设为 `true`。启用后，OpenViking 不会再检查或创建 `.openviking.pid`。
- 对 QueueFS，建议通过 `storage.agfs.queuefs.db_path` 显式指定实例本地的 SQLite 路径。如果启用了 usage audit，建议通过 `server.observability.usage_audit.sqlite_path` 显式指定实例本地的 SQLite 路径，不要默认和共享 workspace 卷混用。

示例：

```json
{
  "server": {
    "temp_upload": {
      "default_mode": "shared"
    }
  },
  "storage": {
    "skip_process_lock": true
  }
}
```

这个示例只适用于多个实例明确共享同一个 `workspace` 的场景。如果每个实例都有自己的本地 `workspace`，不要开启 `skip_process_lock`。

如果你还需要为 QueueFS 和 usage audit 显式指定本地 SQLite 路径，可以参考：

```json
{
  "server": {
    "temp_upload": {
      "default_mode": "shared"
    },
    "observability": {
      "usage_audit": {
        "sqlite_path": "/var/lib/openviking-local/usage_audit.sqlite3"
      }
    }
  },
  "storage": {
    "skip_process_lock": true,
    "agfs": {
      "queuefs": {
        "db_path": "/var/lib/openviking-local/queue.db"
      }
    }
  }
}
```

这个变体适用于多个实例共享同一个 `workspace`，但 QueueFS 和 usage audit 的 SQLite 文件仍然放在各实例本地路径的场景。

#### Semantic QueueFS 正确性边界

Semantic keyed 合并只在单个 QueueFS backend namespace 内保证原子性。Memory backend 是进程本地的。SQLite 的原子合并和单飞调度要求每个 SQLite 文件只能由一个活跃消费者进程或 namespace 使用。不要让多个活跃消费者打开同一个文件：一个消费者的启动恢复可能重置另一个消费者正在处理的行。每实例本地 SQLite 只做实例内合并，不同实例仍可能同时处理同一个 semantic 目录。跨副本全局单飞需要所有副本共享同一 Redis QueueFS `key_prefix`。不要通过不受支持的网络文件系统共享 SQLite 数据库；实例本地 SQLite 仍是本地部署的推荐方式。

发布时必须先上线实现 `/enqueue_keyed` 的 Rust QueueFS 服务，再上线 Python semantic producer。Python producer 请求旧 QueueFS 时，缺少 `/enqueue_keyed` 控制文件必须硬失败，不能回退到 `/enqueue`，否则会丢失原子合并和单飞保证。

回滚 Python 前，先停止产生新的 semantic 工作，再用兼容 keyed wrapper 的 Python 消费者排空全部 keyed 积压；确认 keyed 行清空后才能停止该消费者。如果无法排空，必须保留至少一个兼容消费者，并且不能让回滚后的消费者接入这个 QueueFS namespace。旧版 Python 消费者无法解析 keyed wrapper，绝不能让它领取 keyed 行。

`wait=true` 采用 wait 方案 A：每个请求只等待自己的文件 embedding 和已登记的 semantic root。共享目录 L0/L1 embedding 不归属于任何单一请求，可以随后完成；文件 embedding 和 semantic DAG 完成后，请求不会被尚未完成的目录 embedding 阻塞。

CI 必须编译 ignored Redis contract tests；提供 `QUEUEFS_REDIS_TEST_URL` 时，还必须让两个实例使用相同 `key_prefix` 运行 standalone Redis 双实例测试。真实 Redis 校验由环境变量控制；没有该 URL 时，本地 SQLite 验收仍是发布必需门禁，但不能据此宣称具备跨副本保证。

如需公网 HTTPS 访问，请参考 [公网访问指南](12-public-access.md)。

如需自行构建镜像，请显式传入 OpenViking 版本：
`docker build --build-arg OPENVIKING_VERSION=0.3.12 -t openviking:latest .`

### Kubernetes + Helm

项目提供了 Helm chart，位于 `examples/k8s-helm/`：

```bash
helm install openviking ./examples/k8s-helm \
  --set openviking.config.embedding.dense.api_key="YOUR_API_KEY" \
  --set openviking.config.vlm.api_key="YOUR_API_KEY"
```

详细的云上部署指南（包括火山引擎 TOS + VikingDB + 方舟配置）请参考 [云上部署指南](https://github.com/volcengine/OpenViking/blob/main/examples/cloud/GUIDE.md)。

## 健康检查

| 端点 | 认证 | 用途 |
|------|------|------|
| `GET /health` | 否 | 存活探针 — 立即返回 `{"status": "ok"}` |
| `GET /ready` | 否 | 就绪探针 — 检查 AGFS、VectorDB、APIKeyManager、Embedding、Ollama |

```bash
# 存活探针
curl http://localhost:1933/health

# 就绪探针
curl http://localhost:1933/ready
# {"status": "ready", "checks": {"agfs": "ok", "vectordb": "ok", "api_key_manager": "ok", "embedding": "ok", "ollama": "ok"}}
```

在 Kubernetes 中，使用 `/health` 作为存活探针，`/ready` 作为就绪探针。

## 相关文档

- [公网访问与反向代理](12-public-access.md) - HTTPS、Caddy、nginx
- [认证](04-authentication.md) - API Key 设置
- [OAuth 接入指南](11-oauth.md) - 面向 MCP 客户端的 OAuth 2.1
- [可观测性与排障](05-observability.md) - 健康检查、追踪与排障
- [API 概览](../api/01-overview.md) - 完整 API 参考
