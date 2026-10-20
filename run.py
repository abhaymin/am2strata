#!/usr/bin/env python3
"""Run the PolyStrata Rust server (strata-serve).

  python3 run.py                         build if needed, serve every model of ~/Development/PolyStrata
  python3 run.py --model minicpm --port 8001
  python3 run.py --take-over             stop the PolyStrata systemd service first (the GPU fits one engine), start it
                                         again when this server ends
  python3 run.py --test                  run the Rust test suite instead
  python3 run.py --build                 rebuild first even when the binary looks current

Everything after the known options is passed to strata-serve (see `run.py -- --help`).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BINARY = ROOT / "target" / "release" / "strata-serve"
SERVICE = "strata-orca"


def say(msg: str) -> None:
    print(f"[run] {msg}", flush=True)


def needs_build() -> bool:
    if not BINARY.exists():
        return True
    built = BINARY.stat().st_mtime
    sources = [p for p in (ROOT / "crates").rglob("*") if p.suffix in (".rs", ".toml") and "target" not in p.parts]
    sources.append(ROOT / "Cargo.toml")
    return any(p.stat().st_mtime > built for p in sources)


def build() -> None:
    say("building strata-serve (cargo build --release) ...")
    subprocess.run(["cargo", "build", "--release", "-p", "strata-serve"], cwd=ROOT, check=True)


def service_active() -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", "--quiet", SERVICE])
    return r.returncode == 0


def service(action: str) -> None:
    say(f"systemctl --user {action} {SERVICE}")
    subprocess.run(["systemctl", "--user", action, SERVICE], check=False)


def wait_ready(host: str, port: int, proc: subprocess.Popen, timeout: float = 900) -> bool:
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                health = json.load(r)
            say(f"ready: {health.get('model')} (images: {health.get('images')}, video: {health.get('video')})")
            return True
        except OSError:
            time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(Path.home() / "Development" / "PolyStrata"))
    ap.add_argument("--model", help="model loaded at start (id or a unique part of it)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--take-over", action="store_true", help=f"stop the {SERVICE} service while this server runs")
    ap.add_argument("--build", action="store_true", help="rebuild first")
    ap.add_argument("--test", action="store_true", help="run the Rust test suite and exit")
    args, rest = ap.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]

    if args.test:
        return subprocess.run(["cargo", "test", "--release"], cwd=ROOT).returncode
    if args.build or needs_build():
        build()

    restore = False
    if args.take_over and service_active():
        service("stop")
        restore = True
        time.sleep(3)                              # the engine gives its GPU memory back

    cmd = [str(BINARY), "--models-dir", args.models_dir, "--port", str(args.port), "--host", args.host, *rest]
    if args.model:
        cmd += ["--model", args.model]
    say(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=ROOT)
    try:
        if wait_ready(args.host, args.port, proc):
            say(f"web app: http://127.0.0.1:{args.port}/   API: http://127.0.0.1:{args.port}/v1   (Ctrl+C stops)")
        return proc.wait()
    except KeyboardInterrupt:
        say("stopping ...")
        proc.send_signal(signal.SIGINT)            # the server stops its engine and image encoder
        try:
            return proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 1
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                proc.kill()
        if restore:
            service("start")


if __name__ == "__main__":
    sys.exit(main())
