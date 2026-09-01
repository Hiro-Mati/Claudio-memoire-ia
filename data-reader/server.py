#!/usr/bin/env python3
"""A dependency-free, read-only browser for configured local data directories."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlparse

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_SEARCH_RESULTS = 200
TEXT_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".xml", ".html", ".htm", ".csv", ".tsv", ".log",
    ".ini", ".cfg", ".conf", ".sql", ".py", ".js", ".ts", ".css",
}
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def source_id(value: str) -> str:
    value = value.strip()
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in value):
        raise ValueError(f"Invalid source id: {value!r}")
    return value


def load_sources(config_path: Path | None, fallback_root: Path) -> list[dict]:
    if config_path is None:
        raw_sources = [{"id": "default", "label": fallback_root.name, "path": str(fallback_root)}]
        base = Path.cwd()
    else:
        config_path = config_path.resolve()
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read source config {config_path}: {exc}") from exc
        raw_sources = config.get("sources") if isinstance(config, dict) else None
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError("Source config must contain a non-empty 'sources' list")
        base = config_path.parent

    sources = []
    seen = set()
    for item in raw_sources:
        if not isinstance(item, dict):
            raise ValueError("Each source must be an object")
        identifier = source_id(str(item.get("id", "")))
        if identifier in seen:
            raise ValueError(f"Duplicate source id: {identifier}")
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"Source {identifier} requires a path")
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"Source directory does not exist: {path}")
        sources.append({
            "id": identifier,
            "label": str(item.get("label") or path.name),
            "description": str(item.get("description") or ""),
            "path": path,
        })
        seen.add(identifier)
    return sources


def human_kind(path: Path, sample: bytes = b"") -> str:
    suffix = path.suffix.lower()
    if suffix in SQLITE_SUFFIXES and sample.startswith(b"SQLite format 3\x00"):
        return "sqlite"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if not sample:
        return "empty"
    if b"\x00" not in sample:
        try:
            sample.decode("utf-8")
            return "text"
        except UnicodeDecodeError:
            pass
    return "binary"


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a user path while guaranteeing it stays below root."""
    relative = unquote(relative).replace("\\", "/").lstrip("/")
    parts = PurePosixPath(relative).parts
    if any(part in {"..", ""} for part in parts):
        raise ValueError("Invalid path")
    candidate = root.joinpath(*parts).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escapes data root")
    return candidate


class DataIndex:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._lock = threading.Lock()
        self.entries: list[dict] = []
        self.by_path: dict[str, dict] = {}
        self.text_cache: dict[str, str] = {}
        self.refresh()

    def refresh(self) -> None:
        entries: list[dict] = []
        by_path: dict[str, dict] = {}
        text_cache: dict[str, str] = {}
        for current, dirs, files in os.walk(self.root):
            dirs.sort(key=str.casefold)
            files.sort(key=str.casefold)
            current_path = Path(current)
            for name in files:
                path = current_path / name
                try:
                    stat = path.stat()
                    with path.open("rb") as stream:
                        sample = stream.read(8192)
                except OSError:
                    continue
                relative = path.relative_to(self.root).as_posix()
                kind = human_kind(path, sample)
                parent = PurePosixPath(relative).parent.as_posix()
                entry = {
                    "path": relative,
                    "name": name,
                    "parent": "" if parent == "." else parent,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                    "kind": kind,
                    "hidden": any(part.startswith(".") for part in PurePosixPath(relative).parts),
                    "readable": kind in {"markdown", "json", "text", "empty", "sqlite"},
                }
                entries.append(entry)
                by_path[relative] = entry
                if kind in {"markdown", "json", "text"} and stat.st_size <= MAX_TEXT_BYTES:
                    try:
                        text_cache[relative] = path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        pass
        with self._lock:
            self.entries = entries
            self.by_path = by_path
            self.text_cache = text_cache

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        total = 0
        readable = 0
        for entry in self.entries:
            counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
            total += entry["size"]
            readable += int(entry["readable"])
        return {
            "root": self.root.name,
            "files": len(self.entries),
            "readable": readable,
            "bytes": total,
            "kinds": counts,
            "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def search(self, query: str) -> list[dict]:
        needle = query.strip().casefold()
        if not needle:
            return []
        results = []
        for entry in self.entries:
            path_hit = needle in entry["path"].casefold()
            content = self.text_cache.get(entry["path"], "")
            content_at = content.casefold().find(needle)
            if not path_hit and content_at < 0:
                continue
            snippet = ""
            if content_at >= 0:
                start = max(0, content_at - 90)
                end = min(len(content), content_at + len(query) + 150)
                snippet = " ".join(content[start:end].split())
            results.append({**entry, "snippet": snippet, "match": "path" if path_hit else "content"})
            if len(results) >= MAX_SEARCH_RESULTS:
                break
        return results


class SourceRegistry:
    def __init__(self, sources: list[dict]):
        self.sources = {source["id"]: source for source in sources}
        self._indexes: dict[str, DataIndex] = {}
        self._lock = threading.Lock()

    def public_sources(self) -> list[dict]:
        return [
            {"id": source["id"], "label": source["label"], "description": source["description"]}
            for source in self.sources.values()
        ]

    def get(self, identifier: str) -> DataIndex:
        source = self.sources.get(identifier)
        if source is None:
            raise ValueError("Unknown data source")
        with self._lock:
            index = self._indexes.get(identifier)
            if index is None:
                index = DataIndex(source["path"])
                self._indexes[identifier] = index
        return index

    def source(self, identifier: str) -> dict:
        source = self.sources.get(identifier)
        if source is None:
            raise ValueError("Unknown data source")
        return source


def printable(value):
    if isinstance(value, bytes):
        return f"<binary: {len(value)} bytes>"
    return value


def sqlite_description(path: Path, table: str | None = None, page: int = 1, limit: int = 50) -> dict:
    # mode=ro keeps the browser non-mutating while still allowing SQLite to
    # merge committed rows from an existing WAL file.
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=1)
    connection.row_factory = sqlite3.Row
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        result: dict = {"tables": tables}
        if table:
            if table not in tables:
                raise ValueError("Unknown table")
            quoted = '"' + table.replace('"', '""') + '"'
            count = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            offset = (page - 1) * limit
            rows = connection.execute(f"SELECT * FROM {quoted} LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            columns = list(rows[0].keys()) if rows else [row[1] for row in connection.execute(f"PRAGMA table_info({quoted})")]
            result.update({
                "table": table, "page": page, "limit": limit, "total": count,
                "columns": columns,
                "rows": [[printable(value) for value in row] for row in rows],
            })
        return result
    finally:
        connection.close()


class DataReaderHandler(SimpleHTTPRequestHandler):
    registry: SourceRegistry

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            params = parse_qs(parsed.query)
            if parsed.path == "/api/sources":
                self.send_json({"sources": self.registry.public_sources()})
                return
            if parsed.path == "/api/index":
                identifier, index = self.source_index(params)
                self.send_json({"source": identifier, "stats": index.stats(), "entries": index.entries})
                return
            if parsed.path == "/api/search":
                _, index = self.source_index(params)
                query = params.get("q", [""])[0][:200]
                self.send_json({"query": query, "results": index.search(query)})
                return
            if parsed.path == "/api/file":
                self.handle_file(params)
                return
            if parsed.path == "/api/sqlite":
                self.handle_sqlite(params)
                return
            if parsed.path == "/api/refresh":
                identifier, index = self.source_index(params)
                index.refresh()
                self.send_json({"source": identifier, "stats": index.stats()})
                return
            if parsed.path.startswith("/api/"):
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            if parsed.path not in {"/", "/index.html", "/app.css", "/app.js"}:
                self.path = "/index.html"
            super().do_GET()
        except (ValueError, OSError, sqlite3.Error) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def source_index(self, params: dict) -> tuple[str, DataIndex]:
        identifier = params.get("source", [""])[0]
        return identifier, self.registry.get(identifier)

    def handle_file(self, params: dict) -> None:
        _, index = self.source_index(params)
        relative = params.get("path", [""])[0]
        path = safe_path(index.root, relative)
        entry = index.by_path.get(relative)
        if not entry or not path.is_file():
            was_indexed = entry is not None
            if was_indexed:
                index.refresh()
            self.send_json({"error": "File not found", "stale": was_indexed}, HTTPStatus.NOT_FOUND)
            return
        if entry["kind"] not in {"markdown", "json", "text", "empty"}:
            self.send_json({"entry": entry, "content": None})
            return
        if entry["size"] > MAX_TEXT_BYTES:
            self.send_json({"error": f"Text file exceeds {MAX_TEXT_BYTES} byte preview limit"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        content = path.read_text(encoding="utf-8", errors="replace")
        if entry["kind"] == "json" and path.suffix.lower() == ".json":
            try:
                content = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass
        self.send_json({"entry": entry, "content": content})

    def handle_sqlite(self, params: dict) -> None:
        _, index = self.source_index(params)
        relative = params.get("path", [""])[0]
        table = params.get("table", [None])[0]
        page = max(1, int(params.get("page", ["1"])[0]))
        limit = min(100, max(1, int(params.get("limit", ["50"])[0])))
        path = safe_path(index.root, relative)
        entry = index.by_path.get(relative)
        if not entry or entry["kind"] != "sqlite":
            if entry is not None and not path.is_file():
                index.refresh()
            raise ValueError("Not a readable SQLite database")
        self.send_json({"entry": entry, **sqlite_description(path, table, page, limit)})


def create_server(sources: list[dict], host: str, port: int) -> ThreadingHTTPServer:
    DataReaderHandler.registry = SourceRegistry(sources)
    return ThreadingHTTPServer((host, port), DataReaderHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=APP_DIR.parent / "data")
    parser.add_argument("--config", type=Path, help="JSON file listing allowed data sources")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4180)
    args = parser.parse_args()
    try:
        sources = load_sources(args.config, args.data)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    server = create_server(sources, args.host, args.port)
    print(f"Data Reader: http://{args.host}:{args.port}")
    print("Read-only sources:")
    for source in sources:
        print(f"  {source['id']}: {source['path']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
