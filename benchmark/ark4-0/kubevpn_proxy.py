#!/usr/bin/env python3
"""Route an OpenViking callback placeholder to the local Ark adapter."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class ProxyConfig:
    path: Path
    kubeconfig: Path
    namespace: str
    deployment: str
    openviking_target: str
    remote_port: int
    local_port: int
    headers: dict[str, str]
    image: str
    state_file: Path
    kubevpn_log_file: Path
    probe_log_file: Path
    probe_event_file: Path
    probe_host: str
    probe_enabled: bool
    local_health_url: str
    probe_response_status: int
    probe_marker: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "status"))
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "kubevpn_config.local.json"),
        help="JSON config (default: kubevpn_config.local.json next to this script)",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> ProxyConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"kubevpn config does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"kubevpn config is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("kubevpn config root must be a JSON object")
    probe = _object(raw.get("probe"), "probe")
    headers = _string_dict(raw.get("headers"), "headers")
    deployment = _text(raw.get("deployment"), "deployment")
    openviking_target = _text(raw.get("openviking_target"), "openviking_target")
    expected_deployment = f"ov-proxy-{openviking_target}"
    if deployment != expected_deployment:
        raise ValueError(
            "deployment must match openviking_target: expected "
            f"{expected_deployment!r}, got {deployment!r}"
        )
    probe_enabled = _boolean(probe.get("enabled"), "probe.enabled", default=False)
    probe_host = _text(probe.get("host") or "127.0.0.1", "probe.host")
    local_port = _port(raw.get("local_port"), "local_port")
    local_health_url = str(raw.get("local_health_url") or "").strip()
    if probe_enabled and not local_health_url:
        local_health_url = f"http://{probe_host}:{local_port}/healthz"
    return ProxyConfig(
        path=config_path,
        kubeconfig=_path(raw.get("kubeconfig"), "kubeconfig", base=config_path.parent),
        namespace=_text(raw.get("namespace"), "namespace"),
        deployment=deployment,
        openviking_target=openviking_target,
        remote_port=_port(raw.get("remote_port"), "remote_port"),
        local_port=local_port,
        headers=headers,
        image=str(raw.get("image") or "").strip(),
        state_file=_path(
            raw.get("state_file") or "kubevpn_state.local.json",
            "state_file",
            base=config_path.parent,
        ),
        kubevpn_log_file=_path(
            raw.get("kubevpn_log_file") or "kubevpn.local.log",
            "kubevpn_log_file",
            base=config_path.parent,
        ),
        probe_log_file=_path(
            probe.get("process_log_file") or "traffic_probe.local.log",
            "probe.process_log_file",
            base=config_path.parent,
        ),
        probe_event_file=_path(
            probe.get("event_log_file") or "traffic_probe_events.local.jsonl",
            "probe.event_log_file",
            base=config_path.parent,
        ),
        probe_host=probe_host,
        probe_enabled=probe_enabled,
        local_health_url=local_health_url,
        probe_response_status=_status_code(probe.get("response_status", 503)),
        probe_marker=_text(
            probe.get("marker") or "ARK4_LOCAL_TRAFFIC_PROBE",
            "probe.marker",
        ),
    )


def build_kubevpn_command(config: ProxyConfig) -> list[str]:
    command = [
        "kubevpn",
        "proxy",
        f"deployment/{config.deployment}",
        "--namespace",
        config.namespace,
        "--portmap",
        f"{config.remote_port}:{config.local_port}",
    ]
    for key, value in config.headers.items():
        command.extend(("--headers", f"{key}={value}"))
    if config.image:
        command.extend(("--image", config.image))
    return command


def start(config: ProxyConfig) -> None:
    _require_executable("kubevpn")
    _require_executable("kubectl")
    if not config.kubeconfig.is_file():
        raise RuntimeError(f"kubeconfig not found: {config.kubeconfig}")
    old_state = _read_state(config.state_file)
    if _pid_running(old_state.get("probe_pid")) or _pid_running(old_state.get("kubevpn_pid")):
        raise RuntimeError(
            f"ark4 kubevpn proxy appears to be running; use status or stop first: "
            f"{config.state_file}"
        )
    if _mapping_active(config):
        raise RuntimeError(
            f"kubevpn mapping already exists for deployment/{config.deployment}; "
            "use status or stop first"
        )
    pre_proxy_revision = _deployment_revision(config)

    probe: subprocess.Popen[Any] | None = None
    if config.probe_enabled:
        if _port_open(config.probe_host, config.local_port):
            raise RuntimeError(
                f"local probe port is already in use: {config.probe_host}:{config.local_port}"
            )
        config.probe_log_file.parent.mkdir(parents=True, exist_ok=True)
        probe_stream = config.probe_log_file.open("w", encoding="utf-8")
        probe = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT_DIR / "traffic_probe.py"),
                "--host",
                config.probe_host,
                "--port",
                str(config.local_port),
                "--log-file",
                str(config.probe_event_file),
                "--response-status",
                str(config.probe_response_status),
                "--marker",
                config.probe_marker,
            ],
            cwd=SCRIPT_DIR,
            stdin=subprocess.DEVNULL,
            stdout=probe_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        probe_stream.close()
    elif not _port_open(config.probe_host, config.local_port):
        raise RuntimeError(
            "local target is not listening; start the Ark adapter first: "
            f"{config.probe_host}:{config.local_port}"
        )
    vpn: subprocess.Popen[Any] | None = None
    try:
        _wait_for_local_target(config, probe)
        config.kubevpn_log_file.parent.mkdir(parents=True, exist_ok=True)
        vpn_stream = config.kubevpn_log_file.open("w", encoding="utf-8")
        env = dict(os.environ)
        env["KUBECONFIG"] = str(config.kubeconfig)
        command = build_kubevpn_command(config)
        vpn = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=vpn_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        vpn_stream.close()
        state = {
            "config_file": str(config.path),
            "deployment": config.deployment,
            "openviking_target": config.openviking_target,
            "namespace": config.namespace,
            "headers": config.headers,
            "remote_port": config.remote_port,
            "local_port": config.local_port,
            "probe_pid": probe.pid if probe is not None else None,
            "kubevpn_pid": vpn.pid,
            "pre_proxy_revision": pre_proxy_revision,
            "started_at": time.time(),
        }
        _write_state(config.state_file, state)
        _wait_for_kubevpn(config, vpn)
    except BaseException:
        if vpn is not None:
            _leave(config, fallback_revision=pre_proxy_revision)
            _terminate(vpn.pid)
        if probe is not None:
            _terminate(probe.pid)
        config.state_file.unlink(missing_ok=True)
        raise

    mode = "diagnostic probe" if config.probe_enabled else "existing local service"
    print(f"[ark4-kubevpn] {mode} ready: {config.probe_host}:{config.local_port}")
    print(
        f"[ark4-kubevpn] proxying deployment/{config.deployment} "
        f"{config.remote_port}->{config.local_port}"
    )
    print(f"[ark4-kubevpn] openviking target: {config.openviking_target}")
    if config.headers:
        print(f"[ark4-kubevpn] header match: {json.dumps(config.headers, ensure_ascii=False)}")
    else:
        print("[ark4-kubevpn] header match: none (dedicated ov-proxy target)")
    if config.probe_enabled:
        print(f"[ark4-kubevpn] probe events: {config.probe_event_file}")


def stop(config: ProxyConfig) -> None:
    state = _read_state(config.state_file)
    _leave(config, fallback_revision=str(state.get("pre_proxy_revision") or ""))
    _terminate(state.get("kubevpn_pid"))
    _terminate(state.get("probe_pid"))
    config.state_file.unlink(missing_ok=True)
    print(f"[ark4-kubevpn] stopped deployment/{config.deployment} proxy")


def status(config: ProxyConfig) -> None:
    state = _read_state(config.state_file)
    result = {
        "deployment": config.deployment,
        "openviking_target": config.openviking_target,
        "namespace": config.namespace,
        "headers": config.headers,
        "local_target": f"{config.probe_host}:{config.local_port}",
        "local_mode": "probe" if config.probe_enabled else "existing",
        "probe_pid": state.get("probe_pid"),
        "probe_running": _pid_running(state.get("probe_pid")),
        "launcher_pid": state.get("kubevpn_pid"),
        "launcher_running": _pid_running(state.get("kubevpn_pid")),
        "local_healthy": _local_healthy(config),
        "mapping_active": _mapping_active(config),
        "event_log": str(config.probe_event_file) if config.probe_enabled else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _wait_for_local_target(
    config: ProxyConfig,
    process: subprocess.Popen[Any] | None,
) -> None:
    for _ in range(50):
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"local traffic probe exited; see {config.probe_log_file}")
        if _local_healthy(config):
            return
        time.sleep(0.1)
    detail = (
        f"see {config.probe_log_file}"
        if config.probe_enabled
        else f"health URL={config.local_health_url or '<tcp-only>'}"
    )
    raise RuntimeError(f"local target did not become healthy; {detail}")


def _wait_for_kubevpn(config: ProxyConfig, process: subprocess.Popen[Any]) -> None:
    ready_markers = (
        "Now you can access resources in the kubernetes cluster",
        "tunnel established",
        "proxy traffic to local",
    )
    # A sidecar injection rolls the deployment. The STG cluster may need to
    # autoscale a node first, so keep this longer than the common 1-2 minute
    # scheduling delay while the launcher remains observable via its log.
    for _ in range(1200):
        if config.kubevpn_log_file.exists():
            log = config.kubevpn_log_file.read_text(encoding="utf-8", errors="replace")
            if any(marker in log for marker in ready_markers):
                return
        if process.poll() is not None:
            raise RuntimeError(f"kubevpn exited while starting; see {config.kubevpn_log_file}")
        time.sleep(0.5)
    raise RuntimeError(f"kubevpn did not report ready; see {config.kubevpn_log_file}")


def _local_healthy(config: ProxyConfig) -> bool:
    if not config.local_health_url:
        return _port_open(config.probe_host, config.local_port)
    try:
        with urllib.request.urlopen(
            config.local_health_url,
            timeout=0.5,
        ) as response:
            return 200 <= response.status < 300
    except OSError:
        return False


def _mapping_active(config: ProxyConfig) -> bool:
    if not shutil.which("kubevpn") or not config.kubeconfig.is_file():
        return False
    env = dict(os.environ)
    env["KUBECONFIG"] = str(config.kubeconfig)
    completed = subprocess.run(
        ["kubevpn", "status"],
        cwd=SCRIPT_DIR,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and f"deployments.apps/{config.deployment}" in completed.stdout


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _pid_running(value: Any) -> bool:
    try:
        pid = int(value)
        os.kill(pid, 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def _terminate(value: Any) -> None:
    if not _pid_running(value):
        return
    pid = int(value)
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(30):
        if not _pid_running(pid):
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _leave(config: ProxyConfig, *, fallback_revision: str = "") -> None:
    if not shutil.which("kubevpn") or not config.kubeconfig.is_file():
        return
    env = dict(os.environ)
    env["KUBECONFIG"] = str(config.kubeconfig)
    subprocess.run(
        [
            "kubevpn",
            "leave",
            f"deployment/{config.deployment}",
            "--namespace",
            config.namespace,
        ],
        cwd=SCRIPT_DIR,
        env=env,
        check=False,
    )
    if fallback_revision and _deployment_has_injected_sidecars(config):
        print(
            "[ark4-kubevpn] kubevpn leave did not restore the incomplete rollout; "
            f"rolling back to revision {fallback_revision}",
            file=sys.stderr,
        )
        subprocess.run(
            [
                "kubectl",
                "rollout",
                "undo",
                f"deployment/{config.deployment}",
                "--namespace",
                config.namespace,
                f"--to-revision={fallback_revision}",
            ],
            cwd=SCRIPT_DIR,
            env=env,
            check=False,
        )
        subprocess.run(
            [
                "kubectl",
                "rollout",
                "status",
                f"deployment/{config.deployment}",
                "--namespace",
                config.namespace,
                "--timeout=60s",
            ],
            cwd=SCRIPT_DIR,
            env=env,
            check=False,
        )


def _deployment_revision(config: ProxyConfig) -> str:
    payload = _deployment_json(config)
    annotations = payload.get("metadata", {}).get("annotations", {})
    revision = str(annotations.get("deployment.kubernetes.io/revision") or "").strip()
    if not revision:
        raise RuntimeError(f"deployment/{config.deployment} has no rollout revision")
    return revision


def _deployment_has_injected_sidecars(config: ProxyConfig) -> bool:
    payload = _deployment_json(config)
    containers = payload.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    names = {str(item.get("name") or "") for item in containers if isinstance(item, dict)}
    return bool(names & {"vpn", "envoy-proxy"})


def _deployment_json(config: ProxyConfig) -> dict[str, Any]:
    env = dict(os.environ)
    env["KUBECONFIG"] = str(config.kubeconfig)
    completed = subprocess.run(
        [
            "kubectl",
            "get",
            f"deployment/{config.deployment}",
            "--namespace",
            config.namespace,
            "--output=json",
        ],
        cwd=SCRIPT_DIR,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot read deployment/{config.deployment}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"kubectl returned invalid JSON for deployment/{config.deployment}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"deployment/{config.deployment} response is not an object")
    return payload


def _require_executable(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"required executable is not installed or not in PATH: {name}")


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _write_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _object(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _string_dict(value: Any, label: str) -> dict[str, str]:
    raw = _object(value, label)
    result = {str(key).strip(): str(item).strip() for key, item in raw.items()}
    if any(not key or not item for key, item in result.items()):
        raise ValueError(f"{label} keys and values must be non-empty strings")
    return result


def _text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} is required")
    return result


def _path(value: Any, label: str, *, base: Path) -> Path:
    path = Path(_text(value, label)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _port(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not 1 <= result <= 65535:
        raise ValueError(f"{label} must be between 1 and 65535")
    return result


def _status_code(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("probe.response_status must be an integer") from exc
    if not 400 <= result <= 599:
        raise ValueError("probe.response_status must be between 400 and 599")
    return result


def _boolean(value: Any, label: str, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{label} must be a boolean")


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        {"start": start, "stop": stop, "status": status}[args.command](config)
    except (ValueError, RuntimeError) as exc:
        print(f"[ark4-kubevpn] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
