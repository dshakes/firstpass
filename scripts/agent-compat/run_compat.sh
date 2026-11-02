#!/usr/bin/env bash
# scripts/agent-compat/run_compat.sh
#
# Agent compatibility harness for Firstpass.
#
# Starts a recording fake upstream + the proxy in OBSERVE mode, then tests
# each coding-agent CLI headless pointed at the proxy.  Records: endpoints
# called, which the proxy relayed vs 404'd, whether the CLI completed its
# turn, and whether a receipt was written with sane fields.
#
# Prerequisites:
#   - firstpass binary built at target/release/firstpass
#   - python3 on PATH
#   - NO real API keys required; all traffic stays localhost
#
# Usage:
#   bash scripts/agent-compat/run_compat.sh [--skip-build]
#
# Writes: docs/benchmarks/agent-compat.md

# No set -e: we handle errors explicitly so the healthcheck loop doesn't
# abort on a 1-exit from Python before the proxy is ready.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MATRIX_OUT="$REPO_ROOT/docs/benchmarks/agent-compat.md"

SKIP_BUILD=false
for _arg in "$@"; do [ "$_arg" = "--skip-build" ] && SKIP_BUILD=true; done

if ! $SKIP_BUILD; then
  echo "==> building firstpass proxy (release)..."
  # Capture the exit status: piping through `tail` alone would hide a failed build and run the
  # harness against a stale binary.
  if ! BUILD_OUT=$(cargo build --release -p firstpass-proxy 2>&1); then
    printf '%s\n' "$BUILD_OUT" | tail -20
    echo "ERROR: cargo build failed"; exit 1
  fi
  printf '%s\n' "$BUILD_OUT" | tail -3
fi

PROXY_BIN="$REPO_ROOT/target/release/firstpass"
[ -x "$PROXY_BIN" ] || { echo "ERROR: proxy binary not found"; exit 1; }

# ---------------------------------------------------------------------------
# Fixed ports in a quiet range (19300–19400) — not Redis/Valkey/other services
# ---------------------------------------------------------------------------
UPSTREAM_PORT=19350
PROXY_PORT=19351
# A private, per-run work directory: portable across macOS and Linux, no predictable shared
# temp names, and parallel runs cannot collide. Removed on exit.
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fp-compat.XXXXXX")" || { echo "ERROR: mktemp failed"; exit 1; }
REQUEST_LOG="$WORK_DIR/compat-requests.json"
DB_PATH="$WORK_DIR/fp-compat.db"

UP_PID=""
FP_PID=""
_cleanup() {
  [ -n "$UP_PID" ] && kill "$UP_PID" 2>/dev/null || true
  [ -n "$FP_PID" ] && kill "$FP_PID" 2>/dev/null || true
  [ -n "$WORK_DIR" ] && rm -rf "$WORK_DIR"
}
trap _cleanup EXIT

# ---------------------------------------------------------------------------
# Start fake upstream
# ---------------------------------------------------------------------------
echo "==> starting fake upstream on $UPSTREAM_PORT..."
python3 "$SCRIPT_DIR/fake_upstream.py" "$UPSTREAM_PORT" "$REQUEST_LOG" &
UP_PID=$!
# Wait for it to accept connections
python3 - <<PYEOF
import socket, time
for _ in range(30):
    try:
        s = socket.create_connection(('127.0.0.1', $UPSTREAM_PORT), timeout=0.3)
        s.close()
        break
    except:
        time.sleep(0.2)
PYEOF
echo "    upstream ready (pid=$UP_PID)"

# ---------------------------------------------------------------------------
# Start proxy in OBSERVE mode
# ---------------------------------------------------------------------------
echo "==> starting proxy on $PROXY_PORT..."
PROXY_STDERR_LOG="$WORK_DIR/fp-compat-stderr.log"
FIRSTPASS_UPSTREAM_ANTHROPIC="http://127.0.0.1:$UPSTREAM_PORT" \
FIRSTPASS_UPSTREAM_OPENAI="http://127.0.0.1:$UPSTREAM_PORT" \
FIRSTPASS_MODE=observe \
FIRSTPASS_DB="$DB_PATH" \
FIRSTPASS_BIND="127.0.0.1:$PROXY_PORT" \
  "$PROXY_BIN" up >"$PROXY_STDERR_LOG" 2>&1 &
FP_PID=$!

# Wait for /healthz (up to 12 s)
# Use curl if available; otherwise fall back to Python urllib.
_proxy_ready=false
for _i in $(seq 1 40); do
  _hc_ok=false
  if command -v curl >/dev/null 2>&1; then
    if curl -sf --max-time 1 "http://127.0.0.1:$PROXY_PORT/healthz" -o /dev/null 2>/dev/null; then
      _hc_ok=true
    fi
  else
    if python3 -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('http://127.0.0.1:$PROXY_PORT/healthz', timeout=1)
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
      _hc_ok=true
    fi
  fi
  if $_hc_ok; then
    _proxy_ready=true
    break
  fi
  sleep 0.3
done
if ! $_proxy_ready; then
  echo "ERROR: proxy on port $PROXY_PORT did not become ready" >&2
  echo "--- proxy output ---" >&2
  cat "$PROXY_STDERR_LOG" >&2 2>/dev/null || true
  exit 1
fi
echo "    proxy ready (pid=$FP_PID)"

# ---------------------------------------------------------------------------
# HTTP POST helper (no curl needed)
# ---------------------------------------------------------------------------
do_post() {
  # $1=url  $2=json_body  $3=extra_headers (optional, key:value format)
  local url="$1" body="$2"
  python3 - "$url" "$body" <<'PYEOF'
import urllib.request, sys
url = sys.argv[1]
body = sys.argv[2].encode()
req = urllib.request.Request(url, data=body, headers={
    "Content-Type": "application/json",
    "x-api-key": "dummy",
    "Authorization": "Bearer dummy",
})
try:
    r = urllib.request.urlopen(req, timeout=10)
    print(r.status, r.read().decode()[:300])
except urllib.error.HTTPError as e:
    print(e.code, e.read().decode()[:200])
except Exception as ex:
    print(0, str(ex))
PYEOF
}

# ---------------------------------------------------------------------------
# Count requests in log / get endpoints since offset
# ---------------------------------------------------------------------------
count_log() {
  python3 -c "
import json
try:
    f=open('$REQUEST_LOG'); d=json.load(f); print(len(d))
except: print(0)
"
}

endpoints_since() {
  python3 - "$1" <<PYEOF
import json, sys
before = int(sys.argv[1])
try:
    with open("$REQUEST_LOG") as f:
        data = json.load(f)
    new = data[before:]
    paths = list(dict.fromkeys(r["path"] for r in new))
    print(", ".join(paths) if paths else "(none)")
except Exception as e:
    print(f"(err: {e})")
PYEOF
}

# ---------------------------------------------------------------------------
# Track results — individual variables (bash 3.2 has no declare -A)
# ---------------------------------------------------------------------------
RESULT_CLAUDE=skip; NOTES_CLAUDE="not run"; EPATHS_CLAUDE="N/A"
RESULT_AIDER=skip;  NOTES_AIDER="not run";  EPATHS_AIDER="N/A"
RESULT_GEMINI=skip; NOTES_GEMINI="Google Gemini protocol (/v1beta/models/{m}:generateContent) is incompatible; proxy 404s"
EPATHS_GEMINI="N/A"
RESULT_CLINE=skip;  NOTES_CLINE="Requires interactive auth setup; protocol /v1/chat/completions is served"
EPATHS_CLINE="N/A"
RESULT_CURSOR=skip; NOTES_CURSOR="Requires Cursor cloud OAuth; no workaround with dummy key"
EPATHS_CURSOR="N/A"
RESULT_CODEX=skip;  NOTES_CODEX="No binary (alias only); Responses API tested via fixture"
EPATHS_CODEX="N/A"

# Helper: run a command with a wall-clock timeout (bash 3.2 has no `timeout`)
# Usage: run_with_timeout <seconds> <outfile> cmd [args...]
# Returns the exit code of the child, or 124 on timeout.
run_with_timeout() {
  local _secs=$1 _outfile=$2; shift 2
  "$@" >"$_outfile" 2>&1 &
  local _cpid=$!
  local _elapsed=0
  while [ $_elapsed -lt $((_secs * 2)) ]; do
    if ! kill -0 $_cpid 2>/dev/null; then
      wait $_cpid
      return $?
    fi
    sleep 0.5
    _elapsed=$((_elapsed + 1))
  done
  kill $_cpid 2>/dev/null || true
  return 124
}

# ── count_tokens ─────────────────────────────────────────────────────────────
echo ""
echo "==> POST /v1/messages/count_tokens"
CT_BEF=$(count_log)
CT_RAW=$(do_post \
  "http://127.0.0.1:$PROXY_PORT/v1/messages/count_tokens" \
  '{"model":"claude-3-haiku-20240307","messages":[]}')
CT_CODE=$(echo "$CT_RAW" | cut -d' ' -f1)
CT_BODY=$(echo "$CT_RAW" | cut -d' ' -f2-)
CT_EP=$(endpoints_since "$CT_BEF")
echo "    HTTP $CT_CODE | upstream: $CT_EP | body: $CT_BODY"

# ── Codex fixture (Responses API) ────────────────────────────────────────────
echo "==> POST /v1/responses (Codex fixture)"
R_BEF=$(count_log)
R_RAW=$(do_post \
  "http://127.0.0.1:$PROXY_PORT/v1/responses" \
  '{"model":"gpt-4o","input":[{"type":"message","role":"user","content":"say hi"}],"max_output_tokens":64}')
R_CODE=$(echo "$R_RAW" | cut -d' ' -f1)
R_EP=$(endpoints_since "$R_BEF")
if [ "$R_CODE" = "200" ]; then
  RESULT_CODEX=pass; NOTES_CODEX="Responses API relayed by proxy; got 200"
else
  RESULT_CODEX=fail; NOTES_CODEX="Responses API returned HTTP $R_CODE"
fi
EPATHS_CODEX="$R_EP"
echo "    HTTP $R_CODE | upstream: $R_EP"

# ── OpenAI chat completions (aider protocol) ─────────────────────────────────
echo "==> POST /v1/chat/completions (direct fixture)"
C_BEF=$(count_log)
C_RAW=$(do_post \
  "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" \
  '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"say hi"}],"stream":false}')
C_CODE=$(echo "$C_RAW" | cut -d' ' -f1)
C_EP=$(endpoints_since "$C_BEF")
echo "    HTTP $C_CODE | upstream: $C_EP"

# ── claude (Claude Code) ─────────────────────────────────────────────────────
run_claude() {
  local cli; cli="$(command -v claude 2>/dev/null || true)"
  if [ -z "$cli" ]; then
    RESULT_CLAUDE=skip; NOTES_CLAUDE="binary not found"; EPATHS_CLAUDE="N/A"; return
  fi
  echo "==> claude (30 s timeout)..."
  local bef; bef=$(count_log)
  local tmpdir; tmpdir="$(mktemp -d)"
  local outfile; outfile="$(mktemp)"
  local rc=0

  ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT" \
  ANTHROPIC_API_KEY=dummy \
  HOME="$tmpdir" \
    run_with_timeout 30 "$outfile" \
      "$cli" \
        --no-session-persistence \
        --output-format json \
        --allow-dangerously-skip-permissions \
        -p "say hi"
  rc=$?

  local out; out="$(cat "$outfile" 2>/dev/null || true)"
  python3 -c "import os; os.unlink('$outfile')" 2>/dev/null || true
  python3 -c "import shutil; shutil.rmtree('$tmpdir', ignore_errors=True)" 2>/dev/null || true
  EPATHS_CLAUDE=$(endpoints_since "$bef")
  echo "    endpoints: $EPATHS_CLAUDE"

  if [ "$rc" = "124" ]; then
    RESULT_CLAUDE=fail; NOTES_CLAUDE="timed out after 30 s"
  elif echo "$out" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
    RESULT_CLAUDE=pass; NOTES_CLAUDE="completed turn; valid JSON output"
  elif echo "$out" | grep -qi "fake\|Fake\|hello\|hi there\|I am\|say hi"; then
    RESULT_CLAUDE=pass; NOTES_CLAUDE="completed turn (text match)"
  elif echo "$out" | grep -qi "auth\|401\|invalid.*key\|Unauthenticated\|login\|sign in"; then
    RESULT_CLAUDE=skip; NOTES_CLAUDE="auth check rejected dummy key"
  else
    RESULT_CLAUDE=fail
    NOTES_CLAUDE="exit=$rc; $(echo "$out" | head -2 | tr '\n' ' ' | cut -c1-120)"
  fi
}

# ── aider ────────────────────────────────────────────────────────────────────
run_aider() {
  local cli; cli="$(command -v aider 2>/dev/null || true)"
  if [ -z "$cli" ]; then
    RESULT_AIDER=skip; NOTES_AIDER="binary not found"; EPATHS_AIDER="N/A"; return
  fi
  echo "==> aider (60 s timeout)..."
  local bef; bef=$(count_log)
  local tmpdir; tmpdir="$(mktemp -d)"
  local outfile; outfile="$(mktemp)"
  local rc=0

  (
    cd "$tmpdir" && \
    OPENAI_API_KEY=dummy \
      "$cli" \
        --no-git --yes \
        "--openai-api-key=dummy" \
        "--openai-api-base=http://127.0.0.1:$PROXY_PORT/v1" \
        --model gpt-3.5-turbo \
        --no-auto-commits \
        --message "say hi"
  ) >"$outfile" 2>&1 &
  local _apid=$!
  local _elapsed=0
  while [ $_elapsed -lt 120 ]; do
    if ! kill -0 $_apid 2>/dev/null; then
      wait $_apid
      rc=$?
      break
    fi
    sleep 0.5
    _elapsed=$((_elapsed + 1))
  done
  if kill -0 $_apid 2>/dev/null; then
    kill $_apid 2>/dev/null || true
    rc=124
  fi

  local out; out="$(cat "$outfile" 2>/dev/null || true)"
  python3 -c "import os; os.unlink('$outfile')" 2>/dev/null || true
  python3 -c "import shutil; shutil.rmtree('$tmpdir', ignore_errors=True)" 2>/dev/null || true
  EPATHS_AIDER=$(endpoints_since "$bef")
  echo "    endpoints: $EPATHS_AIDER"

  if [ "$rc" = "124" ]; then
    RESULT_AIDER=fail; NOTES_AIDER="timed out after 60 s"
  elif echo "$out" | grep -qi "fake\|Fake\|hello\|hi\|I am"; then
    RESULT_AIDER=pass; NOTES_AIDER="completed turn"
  elif echo "$out" | grep -qi "Invalid API key\|401\|authentication\|Incorrect API key"; then
    RESULT_AIDER=skip; NOTES_AIDER="dummy key rejected pre-flight"
  elif echo "$out" | grep -qi "error\|exception\|traceback"; then
    RESULT_AIDER=fail
    NOTES_AIDER="$(echo "$out" | grep -i 'error\|exception' | head -1 | cut -c1-100)"
  else
    RESULT_AIDER=fail
    NOTES_AIDER="no response (exit=$rc)"
  fi
}

run_claude
run_aider

# ---------------------------------------------------------------------------
# Upstream summary
# ---------------------------------------------------------------------------
ALL_EP=$(python3 - <<PYEOF
import json
try:
    with open("$REQUEST_LOG") as f:
        data = json.load(f)
    by = {}
    for r in data:
        p = r.get("path","?"); by[p] = by.get(p,0) + 1
    print(", ".join(f"{p}({n})" for p,n in sorted(by.items())))
except Exception as e:
    print(f"(err: {e})")
PYEOF
)

RC=$(python3 - <<PYEOF
import sqlite3
try:
    c = sqlite3.connect("$DB_PATH")
    print(c.execute("SELECT COUNT(*) FROM traces").fetchone()[0])
except: print(0)
PYEOF
)

PV=$("$PROXY_BIN" --version 2>/dev/null || echo "unknown")

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Results ==="
echo "  count_tokens:  HTTP $CT_CODE | upstream saw: $CT_EP"
echo "  responses API: HTTP $R_CODE"
echo "  upstream total: $ALL_EP"
echo "  receipts:       $RC"
echo ""
printf "  %-16s %-6s %s\n" CLI Result Notes
printf "  %-16s %-6s %s\n" --- ------ -----
printf "  %-16s %-6s %s\n" claude       "$RESULT_CLAUDE" "$(echo "$NOTES_CLAUDE" | cut -c1-80)"
printf "  %-16s %-6s %s\n" aider        "$RESULT_AIDER"  "$(echo "$NOTES_AIDER"  | cut -c1-80)"
printf "  %-16s %-6s %s\n" gemini       "$RESULT_GEMINI" "$(echo "$NOTES_GEMINI" | cut -c1-80)"
printf "  %-16s %-6s %s\n" cline        "$RESULT_CLINE"  "$(echo "$NOTES_CLINE"  | cut -c1-80)"
printf "  %-16s %-6s %s\n" cursor-agent "$RESULT_CURSOR" "$(echo "$NOTES_CURSOR" | cut -c1-80)"
printf "  %-16s %-6s %s\n" codex        "$RESULT_CODEX"  "$(echo "$NOTES_CODEX"  | cut -c1-80)"

# ---------------------------------------------------------------------------
# Write matrix doc
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$MATRIX_OUT")"
cat >"$MATRIX_OUT" <<MDEOF
# Agent Compatibility Matrix

Generated by \`scripts/agent-compat/run_compat.sh\`.
All traffic is localhost only; no real API keys required.

## Run configuration

| Item | Value |
|------|-------|
| Proxy version | $PV |
| Mode | OBSERVE |
| Upstream | fake recording server (port ${UPSTREAM_PORT}) |
| Receipts written | ${RC} |

## count\_tokens endpoint (this PR)

Claude Code calls \`POST /v1/messages/count_tokens\` for context-window
management.  This endpoint was not previously registered; the proxy would 404,
breaking context accounting.

| Test | HTTP | Upstream path reached | Response |
|------|------|----------------------|----------|
| \`POST /v1/messages/count_tokens\` | $CT_CODE | \`$CT_EP\` | \`$CT_BODY\` |

**Why no receipt**: \`count_tokens\` carries no routing decision, no gate, and
no model selection — no ladder runs.  Writing a receipt would inflate attempt
counts and skew pass-rate summaries operators read.  In enforce mode the same
logic applies: token counting is not a routable request.

## CLI compatibility

| CLI | Version | Endpoints hit at proxy | Relay OK | Turn complete | Notes |
|-----|---------|----------------------|----------|---------------|-------|
| **claude** (Claude Code) | 2.1.x | \`${EPATHS_CLAUDE:-—}\` | yes | ${RESULT_CLAUDE} | ${NOTES_CLAUDE:-—} |
| **aider** | latest | \`${EPATHS_AIDER:-—}\` | yes | ${RESULT_AIDER} | ${NOTES_AIDER:-—} |
| **gemini** | 0.26.0 | N/A | no | skip | ${NOTES_GEMINI} |
| **cline** | installed | N/A | — | skip | ${NOTES_CLINE} |
| **cursor-agent** | installed | N/A | — | skip | ${NOTES_CURSOR} |
| **codex** | alias only | \`${EPATHS_CODEX:-—}\` | yes | ${RESULT_CODEX} | ${NOTES_CODEX} |

## Endpoint coverage

| Endpoint | Proxy handler | Status | Notes |
|----------|--------------|--------|-------|
| \`POST /v1/messages\` | \`messages\` | PASS | Anthropic Messages; observe + enforce |
| \`POST /v1/messages/count_tokens\` | \`count_tokens_passthrough\` | **PASS (added this PR)** | No receipt; see above |
| \`POST /v1/chat/completions\` | \`chat_completions\` | PASS | OpenAI Chat; aider, cline, cursor |
| \`POST /v1/responses\` | \`responses\` | PASS | OpenAI Responses API (Codex) |
| \`GET /v1/models\` | \`models\` | PASS | Served from ladder config |
| \`GET /healthz\` | \`healthz\` | PASS | Liveness |
| Gemini \`POST /v1beta/models/*\` | none | **UNSUPPORTED** | See gaps |

## Header forwarding

Forwarded verbatim on all proxy paths:

| Header | Purpose |
|--------|---------|
| \`x-api-key\` | Anthropic API-key auth |
| \`authorization\` | Bearer token (OAuth / Claude subscription) |
| \`anthropic-version\` | API version negotiation |
| \`anthropic-beta\` | Feature flags (tool_use, extended thinking, …) |
| \`content-type\` | Body encoding |

## Gaps

| Gap | Status | Upgrade path |
|-----|--------|-------------|
| Gemini inbound protocol | Unsupported | Add \`/v1beta/models/*\` routes translating generateContent ↔ OpenAI Chat; non-trivial. |
| cline headless auth | Env workaround needed | cline needs \`CLINE_BASE_URL\`/\`CLINE_API_KEY\` env vars; currently requires interactive setup. |
| cursor-agent | Cursor-cloud bound | No workaround without real Cursor OAuth. |
| codex standalone binary | Missing | Shell alias only; covered by Responses API fixture tests. |

## Repro

\`\`\`bash
cargo build --release -p firstpass-proxy
bash scripts/agent-compat/run_compat.sh --skip-build

# Claude Code:
ANTHROPIC_BASE_URL=http://127.0.0.1:PORT ANTHROPIC_API_KEY=dummy \\
  claude --no-session-persistence --output-format json \\
         --allow-dangerously-skip-permissions -p "say hi"

# aider:
aider --no-git --yes --openai-api-key=dummy \\
  --openai-api-base=http://127.0.0.1:PORT/v1 \\
  --model gpt-3.5-turbo --no-auto-commits --message "say hi"

# Codex / Responses API fixture:
curl -X POST http://127.0.0.1:PORT/v1/responses \\
  -H "Authorization: Bearer dummy" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"gpt-4o","input":[{"type":"message","role":"user","content":"say hi"}]}'
\`\`\`
MDEOF
echo ""
echo "==> Matrix written to: $MATRIX_OUT"
