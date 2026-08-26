import sys
from pathlib import Path

import pytest
from kubevpn_proxy import SCRIPT_DIR, build_kubevpn_command, load_config, parse_args
from traffic_probe import _safe_headers, _summarize_body


def test_load_config_resolves_paths_and_builds_dedicated_target_command(tmp_path: Path) -> None:
    config_file = tmp_path / "adapter_config.local.json"
    kubeconfig = tmp_path / "config_stg"
    kubeconfig.write_text("clusters: []\n", encoding="utf-8")
    config_file.write_text(
        """
        {
          "service": {"port": 1944},
          "rollout": {
            "runtime_params": {
              "memory": {"openviking_target": "ov-ark-test"}
            }
          },
          "kubevpn": {
            "kubeconfig": "config_stg",
            "namespace": "ai-search-rec"
          }
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.kubeconfig == kubeconfig
    assert config.openviking_target == "ov-ark-test"
    assert config.deployment == "ov-proxy-ov-ark-test"
    assert config.remote_port == 8765
    assert config.local_port == 1944
    assert config.local_health_url == "http://127.0.0.1:1944/health"
    assert build_kubevpn_command(config) == [
        "kubevpn",
        "proxy",
        "deployment/ov-proxy-ov-ark-test",
        "--namespace",
        "ai-search-rec",
        "--portmap",
        "8765:1944",
    ]


def test_config_argument_defaults_to_combined_adapter_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["kubevpn_proxy.py", "status"])

    args = parse_args()

    assert Path(args.config) == SCRIPT_DIR / "adapter_config.local.json"


def test_load_config_rejects_target_deployment_mismatch(tmp_path: Path) -> None:
    config_file = tmp_path / "adapter_config.local.json"
    config_file.write_text(
        """
        {
          "kubevpn": {
            "kubeconfig": "config_stg",
            "namespace": "ai-search-rec",
            "openviking_target": "ov-ark-test",
            "deployment": "ov-proxy-someone-else",
            "remote_port": 8765,
            "local_port": 1944
          }
        }
        """,
        encoding="utf-8",
    )

    try:
        load_config(config_file)
    except ValueError as exc:
        assert "must match openviking_target" in str(exc)
    else:
        raise AssertionError("target/deployment mismatch should be rejected")


def test_load_config_rejects_combined_target_mismatch(tmp_path: Path) -> None:
    config_file = tmp_path / "adapter_config.local.json"
    config_file.write_text(
        """
        {
          "service": {"port": 1944},
          "memory_proxy": {"openviking_target": "target-a"},
          "kubevpn": {
            "kubeconfig": "config_stg",
            "namespace": "ai-search-rec",
            "openviking_target": "target-b",
            "deployment": "ov-proxy-target-b",
            "remote_port": 8765,
            "local_port": 1944
          }
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="values must match"):
        load_config(config_file)


def test_load_config_rejects_combined_adapter_port_mismatch(tmp_path: Path) -> None:
    config_file = tmp_path / "adapter_config.local.json"
    config_file.write_text(
        """
        {
          "service": {"port": 1944},
          "kubevpn": {
            "kubeconfig": "config_stg",
            "namespace": "ai-search-rec",
            "openviking_target": "target-a",
            "deployment": "ov-proxy-target-a",
            "remote_port": 8765,
            "local_port": 2999
          }
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must equal service.port"):
        load_config(config_file)


def test_traffic_probe_redacts_credentials_and_only_summarizes_json_body() -> None:
    headers = _safe_headers(
        {
            "X-API-Key": "secret-key",
            "Authorization": "Bearer secret",
            "X-TT-Backend": "ark4",
        }
    )
    body = _summarize_body(b'{"case_id":"case-secret","runtime_params":{}}')

    assert headers == {
        "x-api-key": "<redacted>",
        "authorization": "<redacted>",
        "x-tt-backend": "ark4",
    }
    assert body["format"] == "json"
    assert body["top_level_keys"] == ["case_id", "runtime_params"]
    assert "case-secret" not in str(body)
