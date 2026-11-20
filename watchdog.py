"""Idle watchdog: destroys this Vast instance when it has been idle too long.

Vast's autoscaler only scales a worker down when it reports no in-flight load. A request
that never completes (e.g. an aborted client stream) leaves load pending forever and the
GPU keeps billing. This watchdog does not trust that accounting.

Idle means: llama-server is not processing any slot AND no request has completed
(per the PyWorker access log) for WATCHDOG_IDLE_MINUTES. The clock starts once the
model first answers /health, so download and load time never count.

Stdlib only. Uses the container's own CONTAINER_API_KEY, which Vast scopes to this
instance, via the vastai CLI from the PyWorker venv.
"""

import datetime as dt
import json
import os
import re
import subprocess
import time
import urllib.request

IDLE_MINUTES = float(os.environ.get("WATCHDOG_IDLE_MINUTES", "20"))
POLL_SECONDS = 60
LLAMA = "http://127.0.0.1:18000"
PYWORKER_LOG = os.environ.get("PYWORKER_LOG", "/workspace/pyworker.log")
VENV_VASTAI = os.path.join(os.environ.get("ENV_PATH", "/workspace/worker-env"), "bin", "vastai")
ACCESS_LINE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"POST /v1/')


def log(msg: str) -> None:
    print(f"{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S} [watchdog] {msg}", flush=True)


def env(name: str) -> str:
    """os.environ first, then /etc/environment (populated by start_server.sh)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        for line in open("/etc/environment"):
            k, _, v = line.strip().partition("=")
            if k == name:
                return v.strip().strip('"')
    except OSError:
        pass
    return ""


def get_json(path: str):
    with urllib.request.urlopen(LLAMA + path, timeout=10) as r:
        return json.load(r)


def healthy() -> bool:
    try:
        return get_json("/health").get("status") == "ok"
    except Exception:
        return False


def llama_busy() -> bool:
    """True if any slot is processing. Unknown counts as busy (never kill on doubt)."""
    try:
        slots = get_json("/slots")
        return any(s.get("is_processing") for s in slots)
    except Exception as e:
        log(f"/slots unavailable ({e}); treating as busy")
        return True


def last_completed_request() -> float:
    """Epoch seconds of the newest completed POST /v1/* in the PyWorker log, or 0."""
    try:
        with open(PYWORKER_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4_000_000))
            tail = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return 0.0
    for line in reversed(tail):
        m = ACCESS_LINE.match(line)
        if m:
            # PyWorker logs container-local time; containers run in UTC.
            return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=dt.timezone.utc).timestamp()
    return 0.0


def self_destruct() -> None:
    cid, key = env("CONTAINER_ID"), env("CONTAINER_API_KEY")
    if not cid or not key:
        log("CONTAINER_ID or CONTAINER_API_KEY missing; cannot destroy self")
        return
    vastai = VENV_VASTAI if os.path.exists(VENV_VASTAI) else "vastai"
    for action in (["destroy", "instance", cid, "-y"], ["stop", "instance", cid]):
        r = subprocess.run([vastai, *action, "--api-key", key], capture_output=True, text=True, timeout=60)
        log(f"vastai {' '.join(action[:2])} -> exit {r.returncode}: {(r.stdout + r.stderr).strip()[:300]}")
        if r.returncode == 0:
            return


def main() -> None:
    log(f"started; idle limit {IDLE_MINUTES:g} min")
    while not healthy():
        time.sleep(POLL_SECONDS)
    ready_at = time.time()
    log("model healthy; idle clock running")
    while True:
        time.sleep(POLL_SECONDS)
        now = time.time()
        if llama_busy():
            continue
        last = max(ready_at, last_completed_request())
        idle_min = (now - last) / 60
        if idle_min >= IDLE_MINUTES:
            log(f"idle for {idle_min:.1f} min with llama idle; destroying this instance")
            self_destruct()
            time.sleep(300)  # if destroy failed, try again later rather than spin


if __name__ == "__main__":
    main()
