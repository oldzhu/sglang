#!/usr/bin/env python3
"""fcloud_exec.py - Execute commands on fcloud JupyterLab instance.

Usage:
    # Run a single command and get output:
    python3 scripts/fcloud/fcloud_exec.py exec "ls -la /root/submission_sim"

    # Run a command with custom timeout (seconds):
    python3 scripts/fcloud/fcloud_exec.py exec --timeout 600 "source ./prepare_env.sh"

    # Run command in background (don't wait for completion):
    python3 scripts/fcloud/fcloud_exec.py exec --background "python3 -m sglang.launch_server ..."

    # Check output from a background terminal:
    python3 scripts/fcloud/fcloud_exec.py tail <terminal_name> --lines 50

    # List active terminals:
    python3 scripts/fcloud/fcloud_exec.py list

    # Kill a terminal:
    python3 scripts/fcloud/fcloud_exec.py kill <terminal_name>

    # Kill all terminals:
    python3 scripts/fcloud/fcloud_exec.py killall

Environment variables:
    FCLOUD_URL   - JupyterLab base URL (e.g. http://host:port)
    FCLOUD_TOKEN - JupyterLab access token

Config file (~/.fcloud_config):
    FCLOUD_URL=http://host:port
    FCLOUD_TOKEN=your_token_here
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Ensure websocket-client is available
# ---------------------------------------------------------------------------
try:
    import websocket  # websocket-client package
except ImportError:
    print("[fcloud] Installing websocket-client ...", file=sys.stderr)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "websocket-client"],
        stdout=subprocess.DEVNULL,
    )
    import websocket


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config():
    """Load FCLOUD_URL and FCLOUD_TOKEN from env or ~/.fcloud_config."""
    url = os.environ.get("FCLOUD_URL", "")
    token = os.environ.get("FCLOUD_TOKEN", "")

    if not url or not token:
        config_path = os.path.expanduser("~/.fcloud_config")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k == "FCLOUD_URL" and not url:
                        url = v
                    elif k == "FCLOUD_TOKEN" and not token:
                        token = v

    if not url or not token:
        print(
            "[fcloud] Error: FCLOUD_URL and FCLOUD_TOKEN must be set (env or ~/.fcloud_config)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Normalise: strip trailing slash, extract base URL (remove /lab etc.)
    url = url.rstrip("/")
    # If URL ends with /lab or /lab/... strip it to get the base
    url = re.sub(r"/lab(/.*)?$", "", url)
    return url, token


# ---------------------------------------------------------------------------
# JupyterLab REST helpers
# ---------------------------------------------------------------------------
def _headers(token):
    return {
        "Content-Type": "application/json",
    }


def _url_with_token(base_url, path, token):
    """Append token as query parameter (works through proxies)."""
    sep = "&" if "?" in path else "?"
    return f"{base_url}{path}{sep}token={token}"


def api_get(base_url, token, path):
    req = urllib.request.Request(
        _url_with_token(base_url, path, token), headers=_headers(token), method="GET"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def api_post(base_url, token, path, data=None):
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(
        _url_with_token(base_url, path, token), headers=_headers(token), method="POST", data=body
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def api_delete(base_url, token, path):
    req = urllib.request.Request(
        _url_with_token(base_url, path, token), headers=_headers(token), method="DELETE"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


# ---------------------------------------------------------------------------
# Terminal operations
# ---------------------------------------------------------------------------
STRIP_ANSI = re.compile(
    r"\x1b\[[0-9;]*[a-zA-Z]"       # CSI sequences (ESC [ ... letter)
    r"|\x1b\].*?\x07"               # OSC sequences (ESC ] ... BEL)
    r"|\x1b\[.*?[@-~]"              # CSI with extended params (ESC [ ... @-~)
    r"|\x1b[=>()<]"                  # Simple mode switches (DECKPAM, DECKPNM, charset)
    r"|\x1b\([A-Z0-9]"              # Character set designation (ESC ( X)
)


def strip_ansi(text):
    return STRIP_ANSI.sub("", text)


def list_terminals(base_url, token):
    return api_get(base_url, token, "/api/terminals")


def create_terminal(base_url, token):
    return api_post(base_url, token, "/api/terminals")


def delete_terminal(base_url, token, name):
    return api_delete(base_url, token, f"/api/terminals/{name}")


def ws_url_for(base_url, token, terminal_name):
    ws_base = base_url.replace("http://", "ws://").replace("https://", "wss://")
    return f"{ws_base}/terminals/websocket/{terminal_name}?token={token}"


def _get_proxy():
    """Get HTTP proxy settings for WebSocket connections."""
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or ""
    if not proxy:
        return None, 0
    # Parse proxy URL: http://host:port
    proxy = proxy.replace("http://", "").replace("https://", "").rstrip("/")
    if ":" in proxy:
        host, port = proxy.rsplit(":", 1)
        return host, int(port)
    return proxy, 8080


def _ws_connect(base_url, token, terminal_name):
    """Create WebSocket connection, handling proxy if needed."""
    url = ws_url_for(base_url, token, terminal_name)
    proxy_host, proxy_port = _get_proxy()
    if proxy_host:
        return websocket.create_connection(
            url, timeout=10,
            http_proxy_host=proxy_host,
            http_proxy_port=proxy_port,
        )
    return websocket.create_connection(url, timeout=10)


def exec_command(base_url, token, command, timeout=300, background=False, cwd=None):
    """Execute a command on fcloud and return (terminal_name, output).

    If background=True, returns immediately after sending the command.
    """
    term = create_terminal(base_url, token)
    term_name = term["name"]

    ws = _ws_connect(base_url, token, term_name)

    # Drain any initial banner/prompt
    time.sleep(0.5)
    _drain(ws, drain_timeout=1.5)

    if background:
        # For background: send command, don't wait, return terminal name
        bg_cmd = f"{command}\n" if not cwd else f"cd {cwd} && {command}\n"
        ws.send(json.dumps(["stdin", bg_cmd]))
        time.sleep(0.3)
        ws.close()
        return term_name, f"[fcloud] Background command started in terminal '{term_name}'"

    # Use unique markers to delimit output
    ts = int(time.time() * 1000)
    begin_marker = f"__FCLOUD_BEGIN_{ts}__"
    end_marker = f"__FCLOUD_END_{ts}__"

    # Build: echo begin_marker, run command, ensure newline, echo end_marker $ec
    # Save $? before the echo (which would reset it to 0)
    if cwd:
        full_cmd = f"cd {cwd} && echo {begin_marker} && {{ {command} ; }}; _ec=$?; echo; echo {end_marker} $_ec\n"
    else:
        full_cmd = f"echo {begin_marker} && {{ {command} ; }}; _ec=$?; echo; echo {end_marker} $_ec\n"

    ws.send(json.dumps(["stdin", full_cmd]))

    # Collect output until end marker appears at start of a line (not in command echo)
    output_parts = []
    start_time = time.time()
    found_end = False
    exit_code = None
    # Pattern: end_marker at line start, followed by space and digit(s)
    end_pattern = re.compile(rf"(?:^|\n){re.escape(end_marker)}\s+(\d+)", re.MULTILINE)

    while (time.time() - start_time) < timeout:
        ws.settimeout(2.0)
        try:
            msg = ws.recv()
            data = json.loads(msg)
            if data[0] == "stdout":
                output_parts.append(data[1])
                # Strip ANSI and \r before checking (terminal sends \r\n line endings)
                joined = strip_ansi("".join(output_parts)).replace("\r", "")
                m = end_pattern.search(joined)
                if m:
                    found_end = True
                    exit_code = int(m.group(1))
                    break
        except websocket.WebSocketTimeoutException:
            continue
        except Exception:
            break

    ws.close()

    if not found_end:
        raw = strip_ansi("".join(output_parts)).replace("\r\n", "\n").replace("\r", "")
        return term_name, f"[fcloud] TIMEOUT after {timeout}s. Partial output:\n{raw}"

    # Clean up terminal
    delete_terminal(base_url, token, term_name)

    # Parse: extract text between begin_marker and end_marker
    # Use \n prefix to skip markers in the command echo line
    raw = strip_ansi("".join(output_parts)).replace("\r\n", "\n").replace("\r", "")
    begin_idx = raw.find("\n" + begin_marker)
    end_idx = raw.find("\n" + end_marker)

    if begin_idx >= 0 and end_idx >= 0:
        # Content starts after begin_marker line
        content_start = begin_idx + 1 + len(begin_marker)
        # Skip the newline after begin_marker
        if content_start < len(raw) and raw[content_start] == "\n":
            content_start += 1
        content_end = end_idx + 1  # +1 to include the \n we matched
        content = raw[content_start:content_end]
    else:
        # Fallback: return everything
        content = raw

    output = content.strip()
    if exit_code is not None and exit_code != 0:
        output += f"\n[fcloud] Exit code: {exit_code}"

    return term_name, output


def tail_terminal(base_url, token, terminal_name, lines=50, wait=3):
    """Connect to an existing terminal and capture recent output."""
    ws = _ws_connect(base_url, token, terminal_name)

    output_parts = []
    start = time.time()
    while (time.time() - start) < wait:
        ws.settimeout(1.0)
        try:
            msg = ws.recv()
            data = json.loads(msg)
            if data[0] == "stdout":
                output_parts.append(data[1])
        except websocket.WebSocketTimeoutException:
            continue
        except Exception:
            break

    ws.close()
    raw = strip_ansi("".join(output_parts))
    result_lines = raw.split("\n")
    return "\n".join(result_lines[-lines:])


def _drain(ws, drain_timeout=1.0):
    """Drain initial output from websocket."""
    parts = []
    start = time.time()
    while (time.time() - start) < drain_timeout:
        ws.settimeout(0.3)
        try:
            msg = ws.recv()
            data = json.loads(msg)
            if data[0] == "stdout":
                parts.append(data[1])
        except (websocket.WebSocketTimeoutException, Exception):
            break
    return "".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Execute commands on fcloud JupyterLab")
    sub = parser.add_subparsers(dest="action", required=True)

    # exec
    p_exec = sub.add_parser("exec", help="Execute a command")
    p_exec.add_argument("command", help="Shell command to execute")
    p_exec.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    p_exec.add_argument("--background", action="store_true", help="Run in background")
    p_exec.add_argument("--cwd", help="Working directory")

    # tail
    p_tail = sub.add_parser("tail", help="Tail a background terminal")
    p_tail.add_argument("terminal", help="Terminal name")
    p_tail.add_argument("--lines", type=int, default=50, help="Number of lines")
    p_tail.add_argument("--wait", type=int, default=3, help="Seconds to wait for output")

    # list
    sub.add_parser("list", help="List active terminals")

    # kill
    p_kill = sub.add_parser("kill", help="Kill a terminal")
    p_kill.add_argument("terminal", help="Terminal name")

    # killall
    sub.add_parser("killall", help="Kill all terminals")

    args = parser.parse_args()
    base_url, token = load_config()

    if args.action == "exec":
        name, output = exec_command(
            base_url, token, args.command,
            timeout=args.timeout, background=args.background, cwd=args.cwd,
        )
        if args.background:
            print(output)
            print(f"[fcloud] Terminal: {name}")
        else:
            print(output)

    elif args.action == "tail":
        output = tail_terminal(base_url, token, args.terminal, args.lines, args.wait)
        print(output)

    elif args.action == "list":
        terms = list_terminals(base_url, token)
        if not terms:
            print("[fcloud] No active terminals")
        else:
            for t in terms:
                print(f"  {t['name']}")

    elif args.action == "kill":
        delete_terminal(base_url, token, args.terminal)
        print(f"[fcloud] Killed terminal: {args.terminal}")

    elif args.action == "killall":
        terms = list_terminals(base_url, token)
        for t in terms:
            delete_terminal(base_url, token, t["name"])
            print(f"[fcloud] Killed terminal: {t['name']}")
        print(f"[fcloud] Killed {len(terms)} terminal(s)")


if __name__ == "__main__":
    main()
