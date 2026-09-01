import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

MODULE_PATH = Path(__file__).parents[1] / "server.py"
SPEC = importlib.util.spec_from_file_location("data_reader_server", MODULE_PATH)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class DataReaderTests(unittest.TestCase):
    def test_safe_path_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(ValueError):
                server.safe_path(root, "../secret")

    def test_index_and_full_text_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes").mkdir()
            (root / "notes" / "hello.md").write_text("# 标题\nOpenViking 可阅读内容", encoding="utf-8")
            (root / "meta.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            index = server.DataIndex(root)
            self.assertEqual(index.stats()["files"], 2)
            self.assertEqual(index.by_path["notes/hello.md"]["kind"], "markdown")
            self.assertEqual(index.search("可阅读")[0]["path"], "notes/hello.md")

    def test_sqlite_is_browsable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.db"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE notes (id INTEGER, title TEXT)")
                connection.execute("INSERT INTO notes VALUES (1, 'hello')")
            data = server.sqlite_description(path, "notes")
            self.assertEqual(data["columns"], ["id", "title"])
            self.assertEqual(data["rows"], [[1, "hello"]])

    def test_source_config_resolves_paths_relative_to_config(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "first").mkdir()
            (base / "second").mkdir()
            config_path = base / "sources.json"
            config_path.write_text(json.dumps({"sources": [
                {"id": "one", "label": "One", "path": "first"},
                {"id": "two", "label": "Two", "path": "second"},
            ]}), encoding="utf-8")
            sources = server.load_sources(config_path, base / "unused")
            self.assertEqual([source["id"] for source in sources], ["one", "two"])
            self.assertEqual(sources[0]["path"], (base / "first").resolve())

    def test_registry_is_lazy_and_rejects_unknown_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hello.md").write_text("hello", encoding="utf-8")
            registry = server.SourceRegistry([{
                "id": "allowed", "label": "Allowed", "description": "", "path": root.resolve(),
            }])
            self.assertEqual(registry._indexes, {})
            self.assertEqual(registry.get("allowed").stats()["files"], 1)
            with self.assertRaises(ValueError):
                registry.get("not-configured")

    def test_invalid_source_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config_path = base / "sources.json"
            config_path.write_text(json.dumps({"sources": [
                {"id": "bad/source", "path": "."},
            ]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                server.load_sources(config_path, base)

    def test_missing_indexed_file_refreshes_index_and_marks_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "stale.md"
            path.write_text("old", encoding="utf-8")
            index = server.DataIndex(root)
            path.unlink()

            handler = object.__new__(server.DataReaderHandler)
            handler.source_index = Mock(return_value=("source", index))
            handler.send_json = Mock()
            handler.handle_file({"path": ["stale.md"]})

            payload, status = handler.send_json.call_args.args
            self.assertEqual(status, server.HTTPStatus.NOT_FOUND)
            self.assertTrue(payload["stale"])
            self.assertNotIn("stale.md", index.by_path)


if __name__ == "__main__":
    unittest.main()
