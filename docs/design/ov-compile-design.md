# `ov compile` 技术设计

| 项目 | 信息 |
| --- | --- |
| 状态 | 已实现 |
| 目标版本 | v1 |
| 更新日期 | 2026-08-03 |

## 1. 概述

`ov compile` 使用指定 Skill 把 OpenViking 材料编译成 Skill 声明的产物。产物可以是 Wiki 页面、精确文件树，或两者的组合；Compile 本身不内置业务领域或 Wiki 分类规则。

命令由 VikingBot 执行。`ov` CLI 通过 OpenViking 的 Bot 代理调用 VikingBot，VikingBot 运行 AgentLoop，并使用当前用户身份读取和写入 OpenViking 数据。

```text
ov compile
  -> OpenViking /bot/v1/compile
  -> VikingBot Compile AgentLoop
  -> OpenViking content APIs
```

v1 的核心目标：

- 支持一个或多个来源目录；
- 加载用户指定的 OV Skill；
- 根据 Skill 生成 Wiki、文件或混合产物；
- 完整清点来源并在隔离上下文中分批处理；
- 通过结构化输出计划、可选 Skill Contract 和独立审计执行“验证—修复”闭环；
- 增量更新已有目标；
- 通过异步任务返回进度和结果。

## 2. 用户接口

### 2.1 命令格式

```bash
ov compile \
  --from viking://resources/周报 \
  --to viking://resources/团队知识库 \
  --reason "按月整理团队的成本优化进展" \
  --skill viking://agent/skills/monthly_wiki \
  --wait
```

| 参数 | 规则 |
| --- | --- |
| `--from` | 必填，可重复，也可使用逗号分隔多个目录 |
| `--to` | 必填，目标 resource、memory 或 Skill namespace 目录 |
| `--skill` | 必填，Skill 目录或 `SKILL.md` 的 Viking URI |
| `--reason` | 可选，本次整理任务的描述 |
| `--wait` | 可选，等待任务完成 |
| `--timeout` | 可选，仅与 `--wait` 一起使用；只限制 CLI 等待时间，不取消任务 |

参数在 OpenViking 用户身份下 canonicalize 后满足以下约束：

- `from` 必须是一个或多个可读目录；重复项去重，空项报错；
- `to` 必须是可写的 resource、memory 内容目录或受支持的 Skill namespace，不能是文件、无效 namespace 根或 OpenViking 派生目录；
- `skill` 必须解析为 Skill root，目录 URI 和其 `SKILL.md` URI 视为同一个 Skill；
- `from`、`to` 和 `skill` 的权限最终仍由 OpenViking Server 校验，CLI 不根据 URI 文本推断权限。

`--reason` 为空时，VikingBot 使用以下默认任务描述：

```text
Follow the loaded Skill's instructions to transform the provided source materials into the outputs required by the Skill.
```

### 2.2 返回结果

未指定 `--wait` 时，CLI 在任务创建后返回：

```text
task_id: cmp_01...
status: accepted
to: viking://resources/团队知识库
```

指定 `--wait` 时，CLI 轮询任务并返回最终结果：

```text
to: viking://resources/团队知识库
created: 1
updated: 2
unchanged: 3
page_count: 6
file_count: 1
link_count: 8
source_coverage: 32/32
validation_attempts: 1
contract_source: declared
```

完整 URI 列表通过全局 JSON 输出返回。

`created`、`updated` 和 `unchanged` 只统计本次提交的产物；未被草稿触达的目标文件不计入 `unchanged`。`page_count` 和 `file_count` 分别统计 Wiki 页面与精确文件，`link_count` 只统计最终正文中实际渲染出的 bundle 内 WikiLink。

`--wait` 使用单调时钟计算整体等待 deadline，并以有上限的 polling interval 查询任务；CLI timeout 或 Ctrl-C 只结束本地等待，不向 Bot 发送取消请求。

## 3. 架构

```text
┌──────────────┐
│ ov CLI       │
└──────┬───────┘
       │ POST /bot/v1/compile
       ▼
┌────────────────────────────┐
│ OpenViking Bot Proxy       │
│ auth + identity forwarding │
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│ VikingBot                  │
│                            │
│ Compile Task               │
│   ├─ Skill Loader          │
│   ├─ Evidence Work Queue   │
│   ├─ Source Work Queue     │
│   ├─ Context Tools         │
│   ├─ AgentLoop             │
│   ├─ Output Plan / Auditor │
│   ├─ Output Renderer       │
│   └─ OpenViking Writer     │
└──────────────┬─────────────┘
               │ read / search / batch-write
               ▼
┌────────────────────────────┐
│ OpenViking Data APIs       │
└────────────────────────────┘
```

职责划分：

| 模块 | 职责 |
| --- | --- |
| `crates/ov_cli` | 参数解析、HTTP 调用、任务轮询和结果展示 |
| `openviking/server/routers/bot.py` | 认证请求并代理到 VikingBot |
| `bot/vikingbot/compile` | Compile 任务、Skill、AgentLoop、渲染和写入编排 |
| OpenViking content service | 数据权限、内容读写和索引刷新 |

OpenViking Server 必须启用 Bot 服务。未启用时，命令返回与 `ov chat` 一致的 503 错误。

### 3.1 现有能力复用

Compile 只增加任务编排和领域规则，基础能力使用现有实现：

| 步骤 | 复用实现 | Compile 适配 |
| --- | --- | --- |
| CLI | `CliContext`、`HttpClient`、全局认证、`OutputFormat`、`output_success()` | compile request、状态轮询和 human formatter |
| Bot proxy | `get_bot_url()`、`_create_bot_proxy_client()`、`_attach_openviking_connection()` | create/status 路由 |
| Gateway 认证 | `OpenAPIChannel` 的 Gateway Token dependency、`OpenVikingConnection` 和 principal scope | compile request model 和 task owner 绑定 |
| URI 与权限 | `fs/attrs` 返回的 canonical URI、`validate_viking_uri()`、`canonicalize_uri()`、`context_type_for_uri()`、VikingFS access check | 用户上下文中的目录约束和 target containment |
| Skill | OpenViking Skills API、`SkillLoader.parse()`、VikingBot `SkillsLoader`、`SandboxManager` | OV bundle 快照和 task-local materialization |
| Agent | `AgentLoop._run_agent_loop()`、`ToolRegistry`、`register_default_tools()` | structured wrapper、分批 worker、scope guard 和通用提交工具 |
| 内容读取 | `openviking_list/search/grep/glob/multi_read` | 限定允许的 URI roots；不增加同义读取工具 |
| Link 与 metadata | `WikiLink`、`StoredLink`、`LinkRenderer`；Memory 目标额外复用 `MemoryFileUtils`、`next_memory_version()` 和 resource refs helper | OKF path、citation 和严格校验 |
| 写入与刷新 | `ContentWriteCoordinator` 的校验/refresh helper、`LockManager`、`VikingFS.write_file(..., lock_handle=...)`、`RequestWaitTracker` | batch precondition 和多文件编排 |

新增能力保持在以下边界内：

- Bot 侧 Compile request/task/result 和最小 task store；
- `submit_compile_work`、`submit_compile_plan`、`submit_compile_outputs`、`submit_compile_validation`、`submit_compile_bundle` 及其 schema；
- 可选的 `compile-contract.yaml`；
- Compile 特有的 OKF/path/citation 规则；
- `/bot/v1/compile` API family 和 `/api/v1/content/batch-write` 数据接口。

## 4. Compile API

### 4.1 创建任务

```http
POST /bot/v1/compile
```

```json
{
  "from": ["viking://resources/周报"],
  "to": "viking://resources/团队知识库",
  "reason": "按月整理团队的成本优化进展",
  "skill": "viking://agent/skills/monthly_wiki"
}
```

成功时返回 HTTP 202：

```json
{
  "task_id": "cmp_01...",
  "status": "accepted",
  "to": "viking://resources/团队知识库"
}
```

VikingBot 负责规范化参数并计算实际任务描述：

```python
effective_reason = (request.reason or "").strip() or DEFAULT_COMPILE_REASON
```

### 4.2 查询任务

```http
GET /bot/v1/compile/{task_id}
```

```json
{
  "task_id": "cmp_01...",
  "status": "running",
  "stage": "agent",
  "created_at": "2026-07-20T10:00:00Z",
  "updated_at": "2026-07-20T10:01:12Z"
}
```

任务状态：

| status | stage |
| --- | --- |
| `accepted` | `queued` |
| `running` | `loading_skill`、`collecting_context`、`planning`、`processing_sources`、`planning_outputs`、`validating_plan`、`materializing_outputs`、`validating`、`rendering`、`writing`、`refreshing` |
| `committing` | `writing`、`refreshing` |
| `completed` | `completed` |
| `failed` | 失败时所在阶段 |

完成结果：

```json
{
  "task_id": "cmp_01...",
  "status": "completed",
  "result": {
    "from": ["viking://resources/周报"],
    "to": "viking://resources/团队知识库",
    "skill": "viking://agent/skills/monthly_wiki",
    "okf_version": "0.1",
    "created": ["viking://resources/团队知识库/成本优化月度进展.md"],
    "updated": [],
    "unchanged": [],
    "page_count": 1,
    "file_count": 0,
    "link_count": 0,
    "source_count": 32,
    "processed_source_count": 32,
    "validation_attempts": 1,
    "contract_source": "declared",
    "warnings": []
  }
}
```

任务只能由创建它的用户查询。

失败结果使用同一查询接口返回稳定结构：

```json
{
  "task_id": "cmp_01...",
  "status": "failed",
  "stage": "writing",
  "error": {
    "code": "WRITE_CONFLICT",
    "message": "Target Wiki changed while the compile task was running."
  }
}
```

创建请求继续通过 body 中的 `openviking_connection` 传递当前用户身份。查询请求是 GET，没有 body；OpenViking proxy 转发原认证凭证，并从已认证的 `RequestContext` 设置 canonical `X-OpenViking-Account/User` header。VikingBot 先做现有 Gateway Token/loopback 校验，再通过 `_resolve_request_principal()` 向 OpenViking 验证凭证并计算 principal scope。无权查询与 task 不存在统一返回 `NOT_FOUND`，避免泄露其他用户的 task ID。

## 5. 执行流程

VikingBot 创建异步任务后依次执行：

1. 计算 `effective_reason`，并对 `from`、`to` 和 `skill` 做 URI 语法校验。
2. 通过 OpenViking 现有 `fs/attrs` 取得来源和目标的 canonical URI，再用 stat/list/read 路径验证形状与权限；Skill API 直接返回 canonical Skill root。VikingBot 后续只使用这些响应中的 canonical URI。
3. 通过 Skills API 取得 Skill root、定义和文件清单，通过现有 content read/download 路径读取辅助文件，在 task workspace 中物化快照，并交给 `SkillsLoader` 加载。
4. 递归建立完整来源 Manifest；超过硬上限时失败，不以截断目录继续生成。Manifest、工作报告和候选验证文件只写入 task workspace 的 `__compile_staging__/`。
5. Skill 未声明 `compile-contract.yaml` 时，不让模型推导 Contract。可信代码只按目标类型确定产物边界：Resource 为 `mixed`、Memory 为 `wiki`、Skill namespace 为 `files`；`SKILL.md` 递归引用的本地 Markdown 作为 Skill reference 参与规划和审计，但不会被冒充成显式 validation rubric。
6. 可信代码先把 `SKILL.md` 及其递归引用的本地 Markdown 切成稳定编号的原文证据块，再由模型选择证据 ID 并形成 task-local 的 Skill checklist；原文路径和引文由代码回填，模型不能伪造。带有 must/required/never/at least/create/emit 等强约束语气（以及常见中文对应词）的证据块必须全部进入 checklist，并同时参与计划与候选审计，避免模型漏选明确规则。checklist 不是 Contract，不推断固定产物，也不会写入目标目录。
7. 按 URI 稳定排序和单次读取上限生成 work items，以有界并发运行短上下文 worker。每个 worker 只提取证据，不决定最终页面、路径或文件树；它必须读取全部分配来源并提交独立报告。
8. 全局 Planner 读取完整 Manifest、Skill checklist、全部 worker 报告、Skill references 和显式 rubric，提交结构化 `CompileOutputPlan`。该计划是唯一的输出 TODO list，明确每个页面/文件的稳定 ID、最终路径、证据报告、coverage disposition，以及覆盖全部产物的 materialization groups/依赖 DAG，不包含正文。
9. 独立只读 Auditor 在生成正文前按有界规则批次逐条核对适用于计划阶段的 Skill checklist，并汇总全部批次结果；每个 Skill 批次只聚焦少量带原文证据的规则。task reason 则由宿主按分号、句号和换行稳定切成 `compile_task_reason_001...N`，每个子句各用一个完全独立的审计回合，只读取原始 reason 和最终计划决策的紧凑上下文，不读取 worker 报告，避免来源细节或其他子句稀释当前约束。每个子句必须分别且恰好检查一次；普通验证通过后再由一个新的 adversarial challenger 尝试寻找反例，最终只保留 challenger 的结论，任一阶段失败都拒绝计划。用户的具体指令不得被更一般的 Skill 规则放宽；全称或否定约束必须检查并列出全部受影响计划字段。审计覆盖计划完整性、产物粒度、命名、路径、来源映射、分组依赖和导航拓扑；任何适用规则失败都拒绝计划，通过证据必须引用实际计划字段，不合格时只修复计划。
10. 计划通过后按 DAG 拓扑层生成正文和精确文件；独立 group 可并发，强耦合产物在同一 group 中生成，依赖 group 必须读取前序文本产物。`submit_compile_outputs` 逐项核验所有计划路径真实存在；Wiki 页面还必须是无 frontmatter 的 UTF-8 Markdown，并引用该页面计划中的精确来源 URI。
11. 可信代码按计划组装 `CompileBundleDraft`，模型不能增删或改名。随后执行确定性的 Bundle/Contract 校验和声明的验证命令；命令执行后重新装载产物。候选验证分别运行 Skill/Contract 审计和 task-reason 审计：前者读取完整证据；后者为每个 `compile_task_reason_001...N` 单独发起 verifier + adversarial challenger，只读取紧凑计划、candidate 清单及每个生成的文本输出。验证失败时按结构化问题重新物化，超过上限则任务失败且不写目标目录。
12. 验证通过后，对每个 `update_uri` 读取最新 raw content 并生成 base hash，渲染 Wiki 与文件产物，区分 created、updated 和 unchanged。
13. 有 write operation 时通过 batch-write 一次提交并等待索引刷新；保存任务结果并清理 task workspace。

Compile 使用固定的 Manifest → Evidence Work → Output Plan → Plan Audit → Materialize → Validate/Repair → Commit 状态机，不引入第二套工作流框架。内容生成仍复用现有 AgentLoop，TODO 完整性、路径、覆盖和提交条件由可信代码校验。

## 6. Skill 与上下文

### 6.1 Skill 加载

`--skill` 支持 Skill 目录或目录内的 `SKILL.md`。

VikingBot 从 canonical Skill URI 拆出 `skill_name` 和 `target_uri`，调用现有 Skills API 取得 Skill root、`SKILL.md` 和文件列表，再通过同一用户连接调用现有 content read/download 路径读取辅助文件；确定性加载阶段不调用 Agent tool，也不解析 `openviking_multi_read` 的展示文本。`SKILL.md` 使用现有 `openviking.core.skill_loader.SkillLoader.parse()` 校验并取得 `allowed_tools`；该 parser 增加 `allowed_tools_declared` 布尔值，以保留“未声明”和“显式空数组”的区别，Compile 不为此再解析一遍 YAML。快照物化到 task-local workspace 后，使用现有 `vikingbot.agent.skills.SkillsLoader` 加载正文和 VikingBot metadata。requirements 使用 `SkillsLoader` 解析出的 `requires.bins/env`，但在实际 task sandbox 中做存在性检查，避免使用 Bot host 环境误判。

该层只负责远程 bundle 的快照和物化，不实现新的 frontmatter parser、Skill 目录规范或 requirements 协议。OpenViking 派生文件和 Skill source metadata 不进入快照；加载过程限制文件数量、单文件大小和总大小，并拒绝逃逸 Skill root 的相对路径。task workspace 只包含本次选择的 Skill，selected Skill 正文直接加入 structured system prompt。任务结束后先调用 `SandboxManager.cleanup_session()` 停止 backend，再删除 compile 专属 workspace；现有 `cleanup_session()` 本身不会删除 direct-backend 目录，不能把它当成文件清理。

Skill package 内的文件使用 `read_file` 读取 task workspace 路径 `skills/<skill-name>/...`；`openviking_*` 工具只读取任务范围内的 `viking://` URI。

Skill 用于描述整理方法，例如：

- 应关注哪些信息；
- 页面如何分层；
- 使用什么表达风格；
- 何时生成索引页或专题页。

### 6.2 Compile Contract

Skill 可以在根目录提供可选的 `compile-contract.yaml`。Contract 只描述通用完成条件，不包含任何内置的 Wiki、论文或客户领域规则：

```yaml
version: 1
output:
  kind: files # wiki | files | mixed
coverage:
  mode: all_sources # all_sources | selected | custom
required_outputs:
  - path: PAPER.md
  - glob: logic/*.md
capabilities:
  required: [exec]
validators:
  - type: rubric
    path: references/validation.md
  - type: command
    run: python3 skills/compiler/scripts/validate.py
```

`compile-contract.yaml` 不是 Skill 的必需文件。没有它时不运行 LLM Contract 推导，也不猜固定文件名；可信代码仅从目标类型确定 `output` 边界，并保留默认 `coverage=all_sources`。Skill 正文与其递归链接的本地 Markdown 规范直接参与全局规划，并产生带原文证据的临时 Skill checklist 供计划审计和候选审计逐条核对；结果标记 `contract_source=inferred`。

显式 Contract 用于作者确实需要机器可执行 validator、运行能力、固定产物路径或非默认 coverage 的场景；其未知字段、非法内部路径、缺失 rubric 或运行环境能力不足会在内容生成前失败。Contract 和内部 `CompileOutputPlan` 都只属于 Compile 任务，`__compile_staging__/`、`.compile/`、来源 Manifest、工作报告和 `output-plan.json` 不会进入最终 Bundle 或 `to` 目录。

### 6.3 来源上下文

VikingBot 按 canonical `from` 顺序为每个来源分配稳定的 request-local ID：

```text
source_id, directory_uri, overview
```

`source_id` 只标识用户传入的来源目录，例如 `src_1`。完整目录清单和明确的一级 coverage units 写入 `__compile_staging__/source-manifest.json`，避免把递归目录项数误当成一级材料数；Prompt 只携带有界摘要。叶子文件按稳定 URI 顺序切成 work items；worker 的 `submit_compile_work` 只有在所有分配 URI 都被 `openviking_multi_read` 成功读到且报告文件存在时才接受。Planner 必须读取 Manifest、Skill checklist 和全部报告；Skill 计划审计读取聚合的 `plan-audit-context.json`，task-reason 计划审计改读不含 worker 报告的 `task-reason-audit-context.json`。候选阶段同样把 Skill/Contract 与 task reason 分开验证，后者只增加 candidate 和实际文本产物。宿主为每个 `compile_task_reason_001...N` 创建独立审计回合，并要求它与适用的 checklist rule ID 各自恰好检查一次，避免“文件读过或提示词提过，但硬要求没有被实际检查”。

各阶段通过 VikingBot 已有 OpenViking 工具按需复查：

- `openviking_list`：浏览来源目录；
- `openviking_search`：语义检索；
- `openviking_grep` / `openviking_glob`：按内容或路径查找；
- `openviking_multi_read`：读取具体内容和 overview。

Compile 不注册另一组 source tools。它在现有工具执行前增加 request-local URI scope guard：所有 URI 参数必须位于 `from`、`to` 或 Skill root 内；`openviking_search/list/grep/glob` 不能省略 scope 后退化为全库查询；`multi_read` 的 URI 数量、递归 list 的节点数、单次结果和任务累计工具结果字节数受 Compile 上限约束。原工具没有上限的地方由这个 guard 补齐，但实际读取和权限判断仍由原工具完成。

### 6.4 目标上下文

运行 Agent 前，VikingBot 使用现有 list/tree/read API 建立目标 catalog：

```text
page_id, uri, title, type, summary
```

catalog 只保存目录项和 L0/L1 可得的轻量信息，不为了计算 hash 或 outgoing links 预读全部目标正文。Agent 可以按需读取已有页面，用于判断创建、更新或复用；只有最终草稿选中的 `update_uri` 会在渲染前读取 raw content 并计算 precondition hash。未被本次结果引用的已有页面保持不变。

### 6.5 工具集合

`request_tools` 从 `register_default_tools()` 创建的 task-local registry 中筛选：

```text
compile_tools = available_tools ∩ (_COMPILE_CORE_TOOLS ∪ _OV_READ_TOOLS)
request_tools = compile_tools + stage_submission_tool
```

`_COMPILE_CORE_TOOLS` 固定为 `read_file`、`write_file` 和 `edit_file`；隔离 backend 或显式启用的可信 direct backend 才增加 `exec`。`_OV_READ_TOOLS` 固定为 `openviking_list`、`openviking_search`、`openviking_grep`、`openviking_glob` 和 `openviking_multi_read`。每个 AgentLoop 只注册当前阶段的一个 submission tool；Planner 和 Auditor 使用只读 registry，批次 materializer 才获得写文件能力。

Compile 不使用 Skill 的 `allowed-tools` 推导、授权或限制工具，也不为 Skill 连接 MCP。该字段可作为其他 Skill 宿主的兼容 metadata 保留。Skill 需要飞书、方舟等外部能力时，通过 `exec` 调用 task sandbox 中预装的 CLI；可选的 `requires.bins/env` 只用于提前检查运行条件，不负责安装 CLI 或依赖。

固定 allowlist 已排除 `message`、`cron`、`spawn`、Web、image、MCP 和 OpenViking 写入/提交工具，无需维护额外 blocklist。`exec` 仍可能产生外部副作用；现有 `direct` sandbox 只提供 task cwd，不是 OS 级隔离。`bot.sandbox.backends.direct.allow_compile_exec` 默认为 `false`，使用 `direct` 时 Compile 仍可通过文件工具完成普通整理，但工具集中不会注册 `exec`；声明 `requires.bins` 或 `requires.env` 的 Skill 会在执行任何命令探测前返回 `SKILL_CAPABILITY_UNAVAILABLE`。将该选项设为 `true` 是明确的不安全 opt-in；生产或多用户部署应使用配置了文件系统和网络 policy 的隔离 backend。

## 7. AgentLoop 输出协议

Compile 不实现第二套 loop。VikingBot 在现有 `AgentLoop._run_agent_loop()` 上提供薄的 `run_structured_task()` 入口，并复用已有参数：

```python
await agent_loop.run_structured_task(
    system_prompt=compile_system_prompt,
    user_prompt=compile_user_prompt,
    session_key=SessionKey(type="compile", channel_id=task_id, chat_id=task_id),
    tool_registry=request_tools,
    openviking_tool_names=openviking_read_tool_names,
    stop_tool_names=[stage_submission_tool_name],
    openviking_connection=connection,
)
```

BotCompileService 使用当前 provider/config、`workspace=task_workspace` 和 task-local `SandboxManager` 创建 request-local `AgentLoop`。`run_structured_task()` 用显式的 system/user prompt 建立 messages 后委托给 `_run_agent_loop()`；后者增加可选 `tool_registry` 和 `openviking_tool_names` 参数，并以选定 registry 同时生成 definitions 和执行工具。只有名称属于 `openviking_tool_names` 的现有 OV adapter 才在 `ToolContext`/post-call hook 中收到用户 connection；file 和 shell tool 收到 `None`。普通 chat 未传这些参数时仍使用 `self.tools` 和现有 connection 行为。

该入口不使用普通 chat history、自动 memory/experience recall 或普通最终回答。它接受且只接受一个 stop tool 名称，因此同一薄封装可用于 work、plan、output receipt 和 validation 等结构化阶段。参数或领域校验返回 `Error:` 时继续当前 loop 修复；只有自然语言而没有 submit 时，wrapper 追加对应工具名的提交提醒。达到 iteration limit 时直接返回 `AGENT_OUTPUT_INVALID`，不执行现有聊天路径的“禁用工具后再回答一次”。

现有 `_run_agent_loop()` 的 stop 判定需要从“出现 stop tool name”改成“该 stop tool 的结果通过 `_is_tool_result_success()`”；这是 structured task 正确重试的必要条件，默认聊天未传 `stop_tool_names`，行为不变。

`request_tools` 仍使用现有 `ToolRegistry` 中的工具实例；OpenViking 权限不在 Bot 中模拟，实际调用继续由 Server 校验。当前阶段的 submission tool 最后注册。Skill 的文件操作和 CLI 命令继续在 task-local `SandboxManager` 中执行；Prompt 明确要求将 Bash、shell 或 CLI 指令交给 `exec`。

Planner 先通过 `submit_compile_plan` 提交不含正文的完整 TODO 和 materialization DAG。Materializer 按拓扑层写文件后，只能通过 `submit_compile_outputs` 提交当前 group 的 output IDs；工具会从 task workspace 读取每个绑定路径并拒绝缺失、空文件、非 UTF-8 Wiki、frontmatter 或无计划来源引用。最终 `CompileBundleDraft` 由可信代码从计划和真实文件组装，不再由模型提交路径清单。

`submit_compile_bundle` 保留为最终确定性 Bundle 校验边界，但在主流程中由可信代码调用。

核心结构：

```python
class WikiPageDraft(BaseModel):
    page_id: int
    title: str
    page_type: str
    summary: str
    body_markdown: str
    source_ids: list[str]
    tags: list[str] = Field(default_factory=list)
    path_hint: str | None = None
    update_uri: str | None = None

class CompileFileDraft(BaseModel):
    path: str | None = None
    update_uri: str | None = None
    content: str | None = None
    workspace_path: str | None = None

class CompileBundleDraft(BaseModel):
    pages: list[WikiPageDraft]
    files: list[CompileFileDraft] = Field(default_factory=list)
    links: list[WikiLink] = Field(default_factory=list)
```

`WikiLink` 直接复用 `openviking.session.memory.dataclass.WikiLink` 做运行时校验，使用其 `f/t/link_type/weight/match_text/description` 字段，不定义 compile 专属 link model。`submit_compile_bundle` 的 tool schema 将 `match_text` 描述覆盖为“必须出现在来源草稿正文中的锚点”，避免沿用 Memory 模型中“original conversation”的提示语义。

约束：

- `pages` 只包含真实 Wiki 页面；Skill 规定的 Markdown、YAML、JSON、图片或其他文件树全部进入 `files`；
- `pages=[]` 对纯文件 Skill 完全合法，但 `pages` 和 `files` 同时为空会被拒绝；
- Contract 的 `output=wiki/files/mixed` 限制可提交的产物类别，`required_paths/globs` 必须被最终路径满足；
- `__compile_staging__/` 和 `.compile/` 不能作为最终输出路径；
- `page_id` 在 bundle 内唯一；
- `update_uri` 必须来自目标 catalog；
- update 保持原 URI，不能通过 `path_hint` rename 或 move；create 的 `path_hint` 只能是 `to` 下的相对 Markdown 路径；
- create 的最终 canonical path 不能与 catalog 中的已有文件或本 bundle 的其他页面冲突；并发创建同一路径仍由 batch precondition 拦截；
- link 的 `f/t` 必须非空、非 self-link，并引用 bundle 中的页面；
- link 的 `match_text` 必须是未包含 Markdown 方括号的正文原文，避免生成嵌套链接；
- `pages` 非空时，每个页面至少引用一个 `source_id`，且必须来自本次请求的来源描述；页面正文还必须包含至少一个本次完整来源清单中的精确叶子文件 URI，只有来源根目录不算可追溯引用；
- Agent 不提供最终文件 URI，也不能直接写入 OpenViking。

Pydantic model 使用 `extra="forbid"`；计划阶段先校验输出类型、完整 coverage、路径冲突和来源映射，物化阶段校验真实文件，最终再执行 Bundle 和 CompileLimits 校验。失败问题结构化返回对应修复阶段；达到上限仍不合法时任务失败。

页面和文件数量由 reason、Skill、Contract 和材料决定。纯文件 Compile 的 `page_count=0`、`link_count=0` 是合法结果。

## 8. 产物渲染与写入

VikingBot renderer 将 `CompileBundleDraft` 转成最终写入计划。普通 `CompileFileDraft` 保持精确字节和路径；只有 `WikiPageDraft` 进入 OKF、链接和 citation 渲染：

1. 解析已有 OKF frontmatter；Memory 目标先用 `MemoryFileUtils.read()` 分离可见文档与 hidden metadata，Resource 目标不生成 `MEMORY_FIELDS`。
2. 将现有 `ExtractLoop._resolve_links()` 中 page ID 解析、self-link 和去重的纯逻辑提取为共享 helper；Compile 使用严格校验模式。
3. 使用 `LinkRenderer` 已有的 anchor 查找、竞争处理和 escaping 生成相对 WikiLink，并补充 canonical target-root 相对路径与 Markdown protected span 两个纯 helper。
4. 确定性生成 OKF v0.1 concept frontmatter、目标路径和 citation section，Agent 不直接生成 YAML。
5. Memory 目标把 resolved `StoredLink` 合并到 `links/backlinks`、复用 resource refs helper，并用 `MemoryFileUtils` round-trip metadata；Resource 目标只存储 OKF Markdown。
6. 对比最终 raw bytes，区分 created、updated 和 unchanged，并为更新绑定渲染前读取的 `content_hash`。

### 8.1 OKF 与 metadata

v1 以 [Open Knowledge Format v0.1 Draft](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) 为格式基线。每个 Compile 页面都是 UTF-8 Markdown concept document：YAML frontmatter 中 `type` 必填；OpenViking 额外要求 `title` 和单行 `description` 非空，`tags` 可选。

字段映射固定为：

| Draft | OKF frontmatter |
| --- | --- |
| `page_type` | `type` |
| `title` | `title` |
| `summary` | `description` |
| `tags` | `tags`，trim、去空并稳定去重；空列表不输出 |

renderer 使用 `yaml.safe_dump(allow_unicode=True, sort_keys=False)` 生成 YAML，拒绝 `body_markdown` 中的第二份 frontmatter。`title`、`page_type` 和 `summary` trim 后必须非空，`summary` 不允许换行。update 保留不与上述平台字段冲突的未知 frontmatter 字段；不自动生成 `timestamp`，避免重复执行仅因时间变化产生更新。

只有 Memory 目标使用 `MemoryFileUtils` round-trip `MEMORY_FIELDS`。create 写入 `category=page_type` 和 `version=1`；update 保留未知字段、同步 `category`，先以原 version 生成 candidate，除 version 外的最终 raw bytes 发生变化时才通过 `next_memory_version()` 推进 version。为避免 hidden links 再次命中 frontmatter，`MemoryFileUtils.write()` 增加默认保持现状的 `render_links=True` 参数，Compile 在已经渲染可见正文后以 `render_links=False` 调用。Resource 目标不写 `category`、`version` 或其他 Memory metadata。

concept 页面不写 `okf_version`；该字段按 OKF 只能出现在 bundle-root `index.md`。`index.md`、`log.md` 不是 concept page，Compile 将它们作为 Resource artifact 创建或更新。API/result 中的 `okf_version: "0.1"` 表示本次 renderer 的目标规范版本。

### 8.2 路径、链接与 Citations

create 的目标路径通过 `sanitize_relative_viking_path()` 和 `safe_join_viking_uri()` 约束在 canonical `to` 下。使用 workspace page body 时，`path_hint` 为空则保留 `__compile_staging__/wiki_pages/` 下的相对路径；直接提交正文时回退到 `VikingURI.sanitize_segment(title)`，并自动追加 `.md`。concept page 拒绝点号文件、`index.md`、`log.md`、OpenViking 派生文件名和清洗后的重复路径；Resource artifact 允许标准的 `index.md`、`log.md`，仍拒绝平台派生文件。update 始终使用已有 URI。

bundle link 的两端必须是本次提交的页面。`match_text` 必须实际命中来源页面的 `body_markdown`，且命中位置不能位于 YAML、代码块、inline code、已有 Markdown link 或 Citations section；renderer 只对正文做 link rendering，再拼接 frontmatter 和 Citations。它使用 target-root-aware 相对路径生成标准 Markdown link，未渲染出的 link 不计入 `link_count`。Resource 目标只保留可见链接；Memory 目标还将 resolved link/backlink 合并进 `MEMORY_FIELDS`，但 v1 不写独立 relation store。

renderer 把每页 `source_ids` 映射为用户传入的 canonical source directory URI，并在可见正文末尾合并成唯一的顶层 `# Citations`。已有 citation 先保留，再按 canonical target 去重追加本次来源；最终统一渲染为连续的 `[n] [label](target)` 列表，来源目录使用 canonical URI 的末级目录名作为 label，无法取得时回退为 `Source src_n`。代码块中的同名标题不视为 citation section。Agent 也可以在正文中引用来源范围内的具体文件 URI，这些 Markdown citation 的 label 和 target 会被保留并参与去重。`viking://` 是 OpenViking 对 citation target 的内部扩展，其他 OKF consumer 未必能够解析该 scheme。

renderer 按每页可见正文的主语言选择自动章节标题：中文正文使用 `相关页面`、`引用来源`，其他正文使用 `Related pages`、`Citations`。语言判断忽略代码、Markdown 链接目标和裸 `viking://` URI；增量更新同时识别中英文旧标题，避免重复章节。

渲染完成后，以最终 UTF-8 bytes 的 SHA-256 作为 hash。candidate 与当前 raw bytes 完全一致时归入 `unchanged` 且不提交 write operation。

写入使用通用内容接口。它是现有内容写入能力的批量入口，不实现新的存储或索引协议：

```http
POST /api/v1/content/batch-write
```

```json
{
  "root_uri": "viking://resources/团队知识库",
  "wait": true,
  "timeout": 300,
  "operations": [
    {
      "uri": "viking://resources/团队知识库/成本优化月度进展.md",
      "content": "...",
      "precondition": {"kind": "create_if_absent"}
    },
    {
      "uri": "viking://resources/团队知识库/既有页面.md",
      "content": "...",
      "precondition": {
        "kind": "replace_if_hash",
        "base_hash": "sha256:..."
      }
    }
  ]
}
```

`content` 是 renderer 生成的最终 UTF-8 存储内容，不是对已有正文执行 append/replace 的编辑指令。接口只接受 create/replace，不支持 delete；请求限制 operation 数量、单文件字节数和总字节数。

Batch write 负责：

- 要求 `root_uri` 是已存在的可写目录；canonicalize 所有 URI，拒绝空 operations、重复 URI、跨 context type 以及 root 之外的目标，并按 canonical URI 稳定排序；
- 校验用户对每个目标 URI 的写权限；
- 保证目标 URI 位于 `root_uri` 下；
- 在目标 tree lock 内读取当前 raw bytes 并检查所有 precondition；
- 若当前 hash 已等于本 operation 的最终 content hash，记为 unchanged 并加入 `refresh_uris`，再检查其余 operation；因此同一请求在响应丢失或前次写入后 refresh 失败时可以安全重试，不需要单独的 idempotency-key store；
- 完成全部底层写入后，以 `refresh_uris = desired-content matches + changed_uris` 刷新语义和向量索引；Bot 正常的全量 unchanged 重跑不会调用 batch-write，因此不会产生多余 refresh；
- resource/skill 按 refresh root 合并变更，每个 root 只提交一个包含全部变更的 `SemanticMsg`，由现有 semantic pipeline 自底向上更新 `.abstract.md` 和 `.overview.md`；
- memory 为变更文件分别更新 embedding，但每个受影响目录只调用一次 `refresh_schema_overview()`；
- 将本批次产生的 refresh 工作绑定到同一个 `RequestWaitTracker`，当 `wait=true` 时统一等待一次。

hash 定义为最终 raw UTF-8 bytes 的小写 SHA-256，API 表示为 `sha256:<hex>`。`create_if_absent` 要求文件不存在；`replace_if_hash` 要求当前 hash 等于 `base_hash`。任一非 unchanged operation 的 precondition 不满足时，在本次调用发生任何新写入前返回标准 `CONFLICT`；若同一重试请求中已有 desired-content match，释放 tree lock 后仍为这些 URI 补做 refresh。VikingBot 将该冲突映射为 task error `WRITE_CONFLICT`。

实现调用链：

```text
content.batch-write router
  -> validate_viking_uri / canonicalize_uri / existing target-shape check
  -> LockManager target tree lease
  -> read current raw bytes; classify unchanged; validate all remaining preconditions
  -> for each changed operation: VikingFS.write_file(..., lock_handle=lease.handle)
  -> collect changed_uris / refresh_uris and group them by refresh scope
  -> release target tree lease
  -> register one request in RequestWaitTracker
  -> existing ContentWriteCoordinator / MemoryUpdater refresh helpers
  -> one RequestWaitTracker waits for the batch's semantic and embedding work
```

Batch coordinator 放在现有 `openviking/storage/content_write.py` 附近，并从 `ContentWriteCoordinator` 下沉双方共同使用的 target validation、SemanticMsg 构造和 refresh helper。单文件 `write()` 与 batch 使用同一组底层实现，不复制 namespace、锁、Memory、semantic 或 embedding 逻辑。Batch coordinator 不得针对每个 operation 循环调用高层 `ContentWriteCoordinator.write()`，否则每个文件都会独立触发并等待 refresh；它必须先完成所有底层写入、释放 tree lock，再对汇总后的变更执行一次批量 refresh 编排，避免 semantic processor 与请求持有的 tree lock 相互阻塞。

Memory 现有 `refresh_schema_overview()` / `refresh_file_embedding()` 会记录 warning 后吞掉部分异常。Batch 路径需要为共享 helper 增加保持旧调用行为的 `strict=False` 默认值，并以 `strict=True` 调用；overview、semantic 或 embedding 任一登记工作失败，或 `wait=true` 得到 failed queue status 时，batch 返回失败，Compile task 不能标记 completed。

该接口不是跨文件原子存储事务：precondition conflict 不会产生本次调用的部分写入，但底层 I/O 在中途失败时可能已有少量文件可见。错误路径必须释放 tree lock，并为已成功写入的 `changed_uris` 触发一次 refresh。相同请求可依据 content hash 跳过完整落盘的文件并继续；若底层留下了不等于最终 content 的残缺文件，重试必须返回冲突，不能静默覆盖。

成功响应使用 OpenViking 标准 envelope：

```json
{
  "status": "ok",
  "result": {
    "created": ["viking://resources/团队知识库/新页面.md"],
    "updated": ["viking://resources/团队知识库/既有页面.md"],
    "unchanged": [],
    "queue_status": {}
  }
}
```

Bot 以该响应为最终提交事实，不根据请求计划假定所有文件都已写入；最终 Compile result 将 renderer 预先识别的 unchanged 与 batch 响应中的 unchanged 合并、稳定去重。

任一页面在读取后被其他请求修改时，本次写入以 `WRITE_CONFLICT` 失败，不覆盖新内容。

## 9. 身份与安全

OpenViking Bot proxy 认证 CLI 请求，并将当前用户的 OpenViking connection 转交给 VikingBot。VikingBot 使用同一身份完成所有读取和写入。

OpenViking proxy 复用 `bot.py` 现有 Bot URL、httpx client、Gateway Token、身份附加和错误映射。VikingBot compile router 复用 `OpenAPIChannel._verify_gateway_request()` 和 `OpenVikingConnection`，不定义第二套 Gateway 认证或 principal 格式。

安全要求：

- task 查询校验创建者身份；
- API key 只存在于运行中任务的内存，不写入 task store 和日志；
- Agent 的 OpenViking 读取范围只包含 `from`、`to` 和 Skill；
- OpenViking adapter 的写入和删除工具不进入 request registry；Compile 管理的 Wiki 写入只能由 batch-write 完成；
- 用户 connection 只注入 scope-guarded OpenViking read adapter，不传给 file 或 shell tool；
- Compile 忽略 Skill 的 `allowed-tools`，固定工具集合中的 `exec` 可能产生 Compile 之外的副作用，不纳入 batch-write 的一致性保证；
- Compile Prompt 明确把来源正文、catalog 和工具结果视为待整理数据，不能把其中的文本当作指令；只有用户的 reason、所选 Skill 和系统 Compile 规则构成指令层；
- file tool 只能访问 task workspace；shell 的隔离强度取决于 backend，多用户部署必须关闭 `direct` Compile exec 或使用隔离 backend；
- 最终 URI、写入条件和 metadata 由可信代码生成；
- 日志不记录 source 正文、Skill 正文、完整 Prompt 或凭证。

远程使用时，Bot 运行在 OpenViking Server 一侧。CLI 不在用户本机启动 Bot。

## 10. 任务存储与并发

Compile task 保存在 VikingBot 的 `bot_data_path/compile_tasks/`，包含：

```text
task_id, principal_scope, sanitized_request, status, stage, timestamps, result, error
```

Bot 当前没有通用的持久化后台任务管理器，因此这里实现一个最小 JSON task store，使用 per-task lock 和临时文件原子替换。进程内以有界的 `asyncio.Task` 集合和 semaphore 承载 accepted task；全局和单 principal admission 在任务创建前计数，超限同步返回 `RESOURCE_EXHAUSTED`。现有 `SessionManager` 继续只管理 chat JSONL，不承载 Compile 状态。

`sanitized_request` 只包含 canonical `from/to/skill` 和 effective reason；`openviking_connection` 仅由运行中 `asyncio.Task` 持有，不进入 JSON、异常详情或日志。

运行中任务目录可以保存有大小限制的 Skill 快照、catalog 和 draft，但不能保存用户凭证。任务进入终态后删除 workspace、Skill snapshot 和 draft；task/result/error JSON 最长保留 24 小时且最多保留 1,000 条，启动和任务结束时都会清理。

VikingBot 使用独立的 compile 并发限制，并对同一 canonical 目标目录串行执行。accepted task 最多排队 5 分钟，取得 target lock 和全局执行 slot 后才开始计算 4 小时 runtime。该锁只减少同一 Bot 进程内的浪费；跨进程或人工写入冲突仍由 batch-write 的 tree lock 和 content hash 检查解决。v1 task store 以单个 VikingBot gateway 进程为部署边界，不承诺多副本共享 task 查询。

VikingBot 启动时把 store 中所有非终态任务统一标记为 `BOT_RESTARTED`，包括处于 committing 的任务；因为 API key 不落盘，重启后不能安全恢复原任务。用户可以重新提交，batch-write 通过最终 content hash 跳过已落盘内容并继续收敛。

### 10.1 v1 资源上限

v1 先使用集中定义、可测试的 `CompileLimits`，不把常量散落在 router/tool/renderer 中：

| 项目 | 默认值 |
| --- | --- |
| source roots | 16 |
| source inventory / Prompt catalog entries | 5,000 / 200 |
| Skill files / 单文件 / 总大小 | 128 / 8 MiB / 32 MiB |
| target inventory entries / relevance catalog pages | 2,000 / 10 |
| initial prompt characters | 200,000 |
| tool URI count / 单次结果 / 任务累计结果 | 32 / 1 MiB / 8 MiB |
| work items / 每项来源 / 并发 / 单报告 | 64 / 24 / 3 / 64 KiB |
| repair attempts | 2 |
| output pages / files / 最终总大小 | 64 / 64 / 4 MiB |
| concurrent Compile tasks / task runtime | 2 / 4 h |
| accepted tasks（全局 / 单 principal）/ queue wait | 16 / 4 / 5 min |
| terminal task retention / records | 24 h / 1,000 |

OpenViking batch-write 自己还要设置独立的 request 上限，至少覆盖 Compile 的 64 pages / 4 MiB，但不能信任 Bot 已经做过限制。超限统一返回 `RESOURCE_EXHAUSTED`。

## 11. 错误处理

| code | 场景 |
| --- | --- |
| `INVALID_ARGUMENT` | 参数缺失或 URI 格式错误 |
| `UNAVAILABLE` | Bot 未启用或不可达；与现有 `ov chat` 一致 |
| `PERMISSION_DENIED` | 无权读取来源或写入目标 |
| `NOT_FOUND` | 来源、Skill、任务不存在，或 task 不属于当前用户 |
| `SKILL_INVALID` | Skill 结构或引用不合法 |
| `SKILL_CAPABILITY_UNAVAILABLE` | Skill 声明的 requirement 或 tool 不可用 |
| `AGENT_OUTPUT_INVALID` | Agent 未提交合法 bundle |
| `CONTRACT_VALIDATION_FAILED` | 候选产物在修复上限内仍未满足 Skill Contract |
| `MODEL_UNAVAILABLE` | 模型服务不可用 |
| `WRITE_CONFLICT` | 目标页面在任务期间发生变化 |
| `WRITE_FAILED` | 内容写入或索引刷新失败 |
| `RESOURCE_EXHAUSTED` | Skill、catalog、工具输入或输出超过 Compile 上限 |
| `DEADLINE_EXCEEDED` | Agent、batch refresh 或 CLI 等待超时 |
| `BOT_RESTARTED` | Bot 重启中断了非终态 Compile 任务 |

同步参数和服务错误沿用 OpenViking 标准 HTTP error code。任务执行错误通过 task 的 `status=failed` 和 `error` 返回；其中 batch API 的标准 `CONFLICT` 在 Compile task 中映射为更具体的 `WRITE_CONFLICT`。

## 12. 代码改动

### CLI

- `crates/ov_cli/src/main.rs`：注册 `compile` 子命令；
- `crates/ov_cli/src/commands/compile.rs`：使用 `CliContext`/`HttpClient` 请求和轮询，使用全局 `OutputFormat`/`output_success()` 输出；
- `crates/ov_cli/src/commands/mod.rs`：导出 command；
- `crates/ov_cli/src/client.rs`：增加 compile create/status 的 typed request 方法；
- `crates/ov_cli/src/help_ui.rs`：增加命令说明和示例。

### OpenViking

- `openviking/server/routers/bot.py`：基于现有 Bot proxy helper 增加 compile 创建和查询请求；
- `openviking/server/routers/content.py`：提供 batch write API；
- `openviking/service/fs_service.py`：暴露 batch coordinator，保持 router 不直接操作 VikingFS；
- `openviking/core/skill_loader.py`：继续兼容解析 `allowed-tools`，Compile 不消费该字段；
- `openviking/storage/content_write.py`：在现有 target validation、锁和 refresh helper 上增加 batch coordinator；
- `openviking/session/memory/`：仅下沉 Link、Memory 或 refresh 双方共用的小型纯 helper，为 `MemoryFileUtils.write()` 增加兼容默认值的 link-render 开关，并为 refresh 增加默认关闭的 strict 失败传播；
- `sdk/python/openviking_sdk/client.py`：为 Bot 使用的现有 async/sync HTTP client 增加 `batch_write()` 和 Skill 辅助文件 download 方法。

### VikingBot

```text
bot/vikingbot/compile/
  models.py
  router.py
  service.py
  store.py
  renderer.py
```

`service.py` 只编排现有 Skills API/loader、OpenViking tools、AgentLoop 和 batch-write client；不为这些能力增加一层同义 wrapper。只有某部分出现独立状态或被第二个调用者复用时再拆文件。

同时对现有模块做小型扩展：

- `bot/vikingbot/agent/loop.py`：为 `_run_agent_loop()` 增加可选 request registry，并提供薄的 `run_structured_task()`；
- `bot/vikingbot/channels/openapi.py`：接收 `BotCompileService` 并用现有 Gateway auth/principal resolver 注册 compile router；
- `bot/vikingbot/agent/tools/`：增加通用 work/bundle/validation 提交工具和 request-local URI scope guard；
- `bot/vikingbot/openviking_mount/ov_server.py`：在现有 request-scoped `VikingClient` 上薄封装 Skills/read/download/batch-write 调用；
- `bot/vikingbot/cli/commands.py`：gateway 先构造共享 provider/config 所属的 AgentLoop，再创建 `BotCompileService` 并注入 OpenAPIChannel；不增加全局 service holder。

## 13. 测试与验收

至少覆盖：

- CLI 参数展开、默认 reason、`--wait` 和 timeout；
- Bot proxy 的创建/GET 查询身份转交、未启用 Bot 的 503 和上游错误；
- Skill 复用现有 parser/loader、相对引用、requirements 和路径逃逸检查；`allowed-tools` 可正常解析但不影响 Compile 工具集合；
- request registry 固定包含当前阶段所需的本地工具、scope-guarded OpenViking 只读工具和唯一 submission tool，不包含 message/cron/spawn/Web/image/MCP/OV write，用户 connection 只进入 OV read adapter；
- Agent structured wrapper 复用原 loop；失败 submit 不停止、plain text 会修复、iteration limit 不额外生成普通回答，普通 chat 行为不回归；
- OpenViking 工具的 URI scope、缺省全库参数和数量/单次/累计输出上限，并确认没有注册第二组 source tools；
- 非法 output plan/bundle 的 loop 内修复、空计划拒绝和最终失败；
- Wiki-only、file-only、mixed bundle、单页面零 link、多页面互链和已有页面更新；
- 完整来源 Manifest、证据型 worker receipt、结构化 TODO、计划逐项落盘、独立审计、Contract validator 和修复上限；
- OKF frontmatter、保留未知字段、Resource/Memory 格式差异、protected anchor、路径 containment、citation merge、WikiLink、Memory version 和 resource refs；
- batch-write 复用现有锁/write/refresh helper，覆盖 canonical URI/重复 operation、权限、content hash conflict、响应丢失/refresh 失败/部分写入后的安全重试，并验证释放 tree lock 后才 refresh；
- 多文件 resource 每个 refresh root 只产生一个 SemanticMsg，memory 每个目录只刷新一次 overview，strict refresh 失败不会返回成功；
- task owner 隔离、同目标并发、终态 workspace 清理和 Bot 重启时所有非终态任务失败。

验收命令：

```bash
ov compile \
  --from viking://resources/周报 \
  --to viking://resources/团队知识库 \
  --reason "按月整理团队的成本优化进展" \
  --skill viking://agent/skills/monthly_wiki \
  --wait
```

验收结果：

1. VikingBot 加载指定 Skill 并运行 Compile AgentLoop。
2. 目标目录生成符合 OKF v0.1 的 Wiki 页面。
3. 重复执行只创建或更新发生变化的页面；最终 raw bytes 相同时不 write、不推进 Memory version。
4. 未触达的已有页面保持不变。
5. 多页面通过一次 batch-write 提交，并按 refresh scope 合并刷新。
6. 未启用 Bot 时命令返回与 `ov chat` 一致的明确错误。
