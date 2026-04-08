#!/usr/bin/env python3
"""fcloud_workflow.py - Automated test workflow for fcloud.

Orchestrates: git pull → copy changed files → restart sglang → run tests.

Usage:
    # Full workflow: pull, sync, restart server, run accuracy test
    python3 scripts/fcloud/fcloud_workflow.py full

    # Just sync files (git pull + copy)
    python3 scripts/fcloud/fcloud_workflow.py sync

    # Just restart the server (kill old, start new)
    python3 scripts/fcloud/fcloud_workflow.py restart-server

    # Run accuracy test only (server must be running)
    python3 scripts/fcloud/fcloud_workflow.py accuracy

    # Run speed benchmark (s1, s8, smax)
    python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
    python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
    python3 scripts/fcloud/fcloud_workflow.py speed --variant smax

    # Run all speed benchmarks
    python3 scripts/fcloud/fcloud_workflow.py speed --variant all

    # Wait for server to be ready
    python3 scripts/fcloud/fcloud_workflow.py wait-server

    # Show server logs (last N lines)
    python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 100
"""

import argparse
import json
import os
import re
import sys
import time

# Add parent so we can import fcloud_exec
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcloud_exec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FCLOUD_REPO = "/root/sglang-minicpm"
FCLOUD_SIM = "/root/submission_sim"
FCLOUD_DATA = "/root/data"
MODEL_PATH = "/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8"
HOST = "0.0.0.0"
PORT = 30000
API_BASE = f"http://127.0.0.1:{PORT}"

# Server terminal name (stable, so we can reconnect)
SERVER_TERMINAL = None  # Will be set when starting server


def fcloud_run(base_url, token, cmd, timeout=300, cwd=None, background=False):
    """Execute a command on fcloud and return output."""
    name, output = fcloud_exec.exec_command(
        base_url, token, cmd, timeout=timeout, cwd=cwd, background=background
    )
    return name, output


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Workflow steps
# ---------------------------------------------------------------------------
def step_sync(base_url, token):
    """Git pull and copy changed files to submission_sim."""
    print_section("SYNC: git pull + copy files")

    # Git pull with auto-merge message
    _, out = fcloud_run(
        base_url, token,
        "cd /root/sglang-minicpm && git pull --no-edit 2>&1",
        timeout=60,
    )
    print(f"[git pull] {out}")

    # Find which files changed in the last pull
    _, changed = fcloud_run(
        base_url, token,
        "cd /root/sglang-minicpm && git diff --name-only HEAD~1 HEAD 2>/dev/null || git diff --name-only @{1} HEAD 2>/dev/null || echo 'DIFF_FAILED'",
        timeout=30,
    )
    print(f"[changed files]\n{changed}")

    if "DIFF_FAILED" in changed or not changed.strip():
        print("[sync] Could not determine changed files, copying all known paths")
        changed_files = []
    else:
        changed_files = [f.strip() for f in changed.strip().split("\n") if f.strip()]

    # Copy files based on path mapping
    copy_cmds = []
    sgl_kernel_changed = False

    for f in changed_files:
        if not f:
            continue
        if f.startswith("benchmark/soar/demo_sala/"):
            # Copy to /root/submission_sim (flat)
            src = f"{FCLOUD_REPO}/{f}"
            dst = f"{FCLOUD_SIM}/{f.replace('benchmark/soar/demo_sala/', '')}"
            copy_cmds.append(f"cp -v {src} {dst}")
        elif f.startswith("python/"):
            # Copy to /root/submission_sim/sglang/python/...
            src = f"{FCLOUD_REPO}/{f}"
            dst = f"{FCLOUD_SIM}/sglang/{f}"
            copy_cmds.append(f"mkdir -p $(dirname {dst}) && cp -v {src} {dst}")
        elif f.startswith("sgl-kernel/"):
            sgl_kernel_changed = True

    if copy_cmds:
        cmd = " && ".join(copy_cmds)
        _, out = fcloud_run(base_url, token, cmd, timeout=60)
        print(f"[copy] {out}")
    else:
        print("[sync] No files to copy (or using fallback)")

    if sgl_kernel_changed:
        print("[sync] sgl-kernel changed — building wheel...")
        _, out = fcloud_run(
            base_url, token,
            f"cd {FCLOUD_REPO}/sgl-kernel && pip wheel --no-build-isolation -w dist . 2>&1 | tail -5",
            timeout=600,
        )
        print(f"[sgl-kernel build] {out}")
        _, out = fcloud_run(
            base_url, token,
            f"cp -v {FCLOUD_REPO}/sgl-kernel/dist/sgl_kernel-*.whl {FCLOUD_SIM}/ 2>&1",
            timeout=30,
        )
        print(f"[sgl-kernel copy] {out}")

    print("[sync] Done")


def step_restart_server(base_url, token):
    """Kill existing sglang server and start a new one."""
    print_section("RESTART SERVER")

    # Kill existing
    _, out = fcloud_run(
        base_url, token,
        'pkill -f "sglang.launch_server" 2>/dev/null; sleep 2; echo "killed"',
        timeout=15,
    )
    print(f"[kill] {out}")

    # Source prepare_env.sh and start server in background
    server_cmd = f"""cd {FCLOUD_SIM} && source ./prepare_env.sh && \\
MODEL_PATH={MODEL_PATH} && \\
HOST={HOST} && \\
PORT={PORT} && \\
read -r -a EXTRA_ARGS <<< "${{SGLANG_SERVER_ARGS:-}}" && \\
python3 -m sglang.launch_server \\
  --model-path "$MODEL_PATH" \\
  --host "$HOST" \\
  --port "$PORT" \\
  "${{EXTRA_ARGS[@]}}" 2>&1"""

    term_name, out = fcloud_run(
        base_url, token, server_cmd, background=True
    )
    print(f"[server] {out}")
    return term_name


def step_wait_server(base_url, token, timeout=300):
    """Wait for sglang server to be ready."""
    print_section("WAITING FOR SERVER")
    start = time.time()
    check_cmd = f'curl -s -o /dev/null -w "%{{http_code}}" {API_BASE}/health 2>/dev/null || echo 000'

    while (time.time() - start) < timeout:
        _, out = fcloud_run(base_url, token, check_cmd, timeout=10)
        code = out.strip()
        # Extract just the numeric code in case of extra output
        m = re.search(r"\b(200|000)\b", code)
        status = m.group(1) if m else code
        if status == "200":
            elapsed = int(time.time() - start)
            print(f"[server] Ready after {elapsed}s")
            return True
        print(f"[server] Not ready yet (status={status}), waiting...")
        time.sleep(10)

    print(f"[server] TIMEOUT after {timeout}s")
    return False


def step_accuracy(base_url, token, timeout=1800):
    """Run accuracy evaluation."""
    print_section("ACCURACY TEST")
    cmd = (
        f"cd {FCLOUD_DATA} && python3 eval_model_001.py "
        f"--api_base {API_BASE} "
        f"--model_path {MODEL_PATH} "
        f"--data_path {FCLOUD_DATA}/perf_public_set.jsonl "
        f"--concurrency 32 2>&1"
    )
    _, out = fcloud_run(base_url, token, cmd, timeout=timeout)
    print(out)
    return out


def step_speed(base_url, token, variant="s1", timeout=600):
    """Run speed benchmark."""
    print_section(f"SPEED TEST: {variant}")

    if variant == "all":
        results = {}
        for v in ["s1", "s8", "smax"]:
            results[v] = step_speed(base_url, token, v, timeout)
        return results

    SPEED_DATA = {
        "s1": "/root/data/benchmark/soar/data/speed_s1.jsonl",
        "s8": "/root/data/benchmark/soar/data/speed_s8.jsonl",
        "smax": "/root/data/benchmark/soar/data/speed_smax.jsonl",
    }

    data_file = SPEED_DATA.get(variant)
    if not data_file:
        print(f"[speed] Unknown variant: {variant}")
        return ""

    # Use bench_serving.sh with the appropriate SPEED_DATA_* env var
    env_var = f"SPEED_DATA_{variant.upper()}"
    cmd = (
        f"cd /root/data && "
        f"{env_var}={data_file} "
        f"bash bench_serving.sh {API_BASE} 2>&1"
    )
    _, out = fcloud_run(base_url, token, cmd, timeout=timeout)
    print(out)
    return out


def step_server_logs(base_url, token, lines=100):
    """Show recent server logs."""
    terms = fcloud_exec.list_terminals(base_url, token)
    if not terms:
        print("[logs] No active terminals")
        return ""

    # Try to get output from terminals
    for t in terms:
        output = fcloud_exec.tail_terminal(base_url, token, t["name"], lines=lines, wait=3)
        if output.strip():
            print(f"--- Terminal: {t['name']} ---")
            print(output)
            return output

    print("[logs] No output captured from terminals")
    return ""


def workflow_full(base_url, token):
    """Full workflow: sync → restart → wait → accuracy."""
    step_sync(base_url, token)
    server_term = step_restart_server(base_url, token)
    ready = step_wait_server(base_url, token, timeout=300)
    if not ready:
        print("[ABORT] Server did not start in time")
        step_server_logs(base_url, token, lines=50)
        return

    accuracy_out = step_accuracy(base_url, token)

    # Parse accuracy from output
    m = re.search(r"Average Score:\s*([\d.]+)%", accuracy_out)
    if m:
        score = float(m.group(1))
        print(f"\n{'='*60}")
        print(f"  ACCURACY: {score}%")
        if score >= 80:
            print(f"  STATUS: PASS (>= 80%)")
        else:
            print(f"  STATUS: FAIL (< 80%)")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="fcloud automated test workflow")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("full", help="Full workflow: sync → restart → accuracy")
    sub.add_parser("sync", help="Git pull and copy changed files")
    sub.add_parser("restart-server", help="Restart sglang server")
    sub.add_parser("wait-server", help="Wait for server to be ready")
    sub.add_parser("accuracy", help="Run accuracy test")

    p_speed = sub.add_parser("speed", help="Run speed benchmark")
    p_speed.add_argument("--variant", choices=["s1", "s8", "smax", "all"], default="s1")

    p_logs = sub.add_parser("server-logs", help="Show server logs")
    p_logs.add_argument("--lines", type=int, default=100)

    sub.add_parser("shutdown", help="Shut down the fcloud instance")

    args = parser.parse_args()
    base_url, token = fcloud_exec.load_config()

    if args.action == "full":
        workflow_full(base_url, token)
    elif args.action == "sync":
        step_sync(base_url, token)
    elif args.action == "restart-server":
        step_restart_server(base_url, token)
    elif args.action == "wait-server":
        step_wait_server(base_url, token)
    elif args.action == "accuracy":
        step_accuracy(base_url, token)
    elif args.action == "speed":
        step_speed(base_url, token, args.variant)
    elif args.action == "server-logs":
        step_server_logs(base_url, token, args.lines)
    elif args.action == "shutdown":
        print_section("SHUTDOWN")
        fcloud_exec.shutdown_server(base_url, token)
        print("[shutdown] fcloud instance shutdown initiated")


if __name__ == "__main__":
    main()
