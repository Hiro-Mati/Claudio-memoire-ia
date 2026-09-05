# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Development install without compiling the native components.

Building OpenViking from source needs Rust (ov CLI, ragfs-python) and a C++17
toolchain with CMake (vector engine). On a laptop, especially under Windows,
that is the single biggest obstacle to hacking on the Python code.

This script reuses the native artifacts shipped in the official wheel for the
current platform and points the virtualenv at the source tree instead:

1. download the ``openviking`` wheel for this Python/platform (or use --wheel);
2. install it with pip (dependencies + native artifacts);
3. copy the native artifacts into the source tree, where the runtime looks
   for them (all are gitignored):
     openviking/bin/ov*                       Rust CLI
     openviking/lib/ragfs_python*             RAGFS binding
     openviking/storage/vectordb/engine/_*    vector engine variants
     openviking/web_studio/dist/              built Studio bundle
4. remove the wheel's Python packages from site-packages, keep console
   scripts and dependencies, and add a ``.pth`` file so ``import openviking``
   resolves to the checkout.

Re-run it after ``git pull`` when ``crates/`` or ``src/`` changed (with the
matching wheel version) or simply to refresh the artifacts.

Usage (from the repository root, inside the venv you want to use):

    python scripts/dev_install_prebuilt.py [--wheel PATH] [--version X.Y.Z]
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ("openviking", "openviking_cli", "vikingbot")
NATIVE_GLOBS = (
    "bin/ov*",
    "lib/ragfs_python*",
    "storage/vectordb/engine/_*.pyd",
    "storage/vectordb/engine/_*.so",
)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def site_packages() -> Path:
    candidates = (
        [Path(p) for p in site.getsitepackages()] if hasattr(site, "getsitepackages") else []
    )
    for candidate in candidates:
        if candidate.name.lower() == "site-packages":
            return candidate
    return Path(site.getusersitepackages())


def download_wheel(version: str | None, dest: Path) -> Path:
    spec = f"openviking=={version}" if version else "openviking"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            spec,
            "--no-deps",
            "--only-binary=:all:",
            "-d",
            str(dest),
        ]
    )
    wheels = sorted(dest.glob("openviking-*.whl"))
    if not wheels:
        raise SystemExit("no binary wheel available for this Python/platform")
    return wheels[-1]


def copy_native_artifacts(installed_pkg: Path) -> int:
    copied = 0
    for pattern in NATIVE_GLOBS:
        for src in glob.glob(str(installed_pkg / pattern)):
            rel = Path(src).relative_to(installed_pkg)
            dst = REPO / "openviking" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print("copied", rel, os.path.getsize(dst))
            copied += 1
    studio_src = installed_pkg / "web_studio" / "dist"
    if studio_src.is_dir():
        studio_dst = REPO / "openviking" / "web_studio" / "dist"
        if studio_dst.exists():
            shutil.rmtree(studio_dst)
        shutil.copytree(studio_src, studio_dst)
        print("copied web_studio/dist")
        copied += 1
    if copied == 0:
        raise SystemExit(f"no native artifacts found under {installed_pkg}")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wheel", help="use this wheel instead of downloading one")
    parser.add_argument("--version", help="wheel version to download (default: latest)")
    args = parser.parse_args()

    if sys.prefix == sys.base_prefix:
        print("warning: not running inside a virtualenv; continuing anyway", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        wheel = Path(args.wheel) if args.wheel else download_wheel(args.version, Path(tmp))
        run([sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel)])

        sp = site_packages()
        installed_pkg = sp / "openviking"
        if not installed_pkg.is_dir():
            raise SystemExit(f"openviking not found in {sp}")
        copy_native_artifacts(installed_pkg)

        for name in PACKAGES:
            target = sp / name
            if target.is_dir():
                shutil.rmtree(target)
                print("removed from site-packages:", name)

        pth = sp / "openviking-dev.pth"
        lines = [str(REPO)]
        if (REPO / "bot" / "vikingbot").is_dir():
            lines.append(str(REPO / "bot"))
        pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("wrote", pth)

    check = subprocess.run(
        [sys.executable, "-c", "import openviking, openviking_cli; print(openviking.__file__)"],
        capture_output=True,
        text=True,
        cwd=str(Path.home()),
    )
    print(check.stdout.strip() or check.stderr.strip())
    if str(REPO) not in check.stdout.replace("/", os.sep):
        print("warning: import does not resolve to the checkout", file=sys.stderr)
        return 1
    print("done: run `openviking-server doctor` to validate the setup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
