# OpenViking Data Reader

零额外依赖、只读的本地多目录数据浏览网站。它适合在本地排查
OpenViking 数据目录、历史快照或迁移产物；日常查看 session、memory 和
resource 的产品化结果时，优先使用 OpenViking Studio。

## 启动

从仓库根目录运行：

```bash
python3 data-reader/server.py --config data-reader/sources.json
```

然后访问：

```text
http://127.0.0.1:4180
```

支持数据源切换、目录浏览、全文搜索、Markdown/JSON/文本阅读、SQLite 按表分页查看、深色模式和移动端布局。

## 配置数据源

通过 `sources.json` 配置允许观察的数据源，例如：

```json
{
  "sources": [
    {
      "id": "current",
      "label": "Local data",
      "description": "OpenViking local data directory",
      "path": "../data"
    }
  ]
}
```

相对路径以配置文件所在目录为基准。数据源 ID 只能包含字母、数字、连字符和下划线。

页面可以切换配置的数据源，各数据源在首次打开时才建立索引。服务端没有写入、上传或删除接口，API 只能访问配置文件中明确列出的目录；浏览器不能提交任意磁盘路径。

不传 `--config` 时仍兼容单目录模式：

```bash
python3 data-reader/server.py --data ./data
```

默认仅监听 127.0.0.1。需要内网访问时显式传入 --host 0.0.0.0。

## 查看结果

Data Reader 不连接 OpenViking API，只读取配置中的本地目录。常用入口：

- `/api/sources`: 查看当前可选数据源。
- `/api/index?source=current`: 查看文件索引和统计。
- `/api/search?source=current&q=<keyword>`: 搜索路径和小文本文件内容。
- `/api/file?source=current&path=<relative-path>`: 读取 Markdown、JSON 或文本。
- `/api/sqlite?source=current&path=<db-path>&table=<table>`: 分页查看 SQLite 表。

如果目标是验证 memory extraction recall 的产品行为，启动 OpenViking Server 后访问 Studio：

```text
http://127.0.0.1:30303/studio
```

在 Studio 中查看 session commit 结果、Resources 面板和 Memories 面板。Data Reader 只作为本地数据排查补充。

## 测试

```bash
python3 -m unittest data-reader/tests/test_server.py
```
