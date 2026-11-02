#!/usr/bin/env bash
# scripts/agent-compat/run_enforce.sh
#
# ENFORCE-mode end-to-end harness: real coding-agent CLIs (Claude Code, aider) driven through the
# proxy with a 2-rung ladder, against a scriptable fake upstream. $0, localhost only, dummy keys.
#
# The cheap model's first reply fails a gate, so a real escalation happens; the top model passes.
# Exits NON-ZERO on any failed assertion (see verify_enforce.py). CLIs not on PATH are skipped and
# the skip is recorded in the generated "Enforce mode" section of docs/benchmarks/agent-compat.md.
#
# Usage: bash scripts/agent-compat/run_enforce.sh [--skip-build]
# Env:   FP_ENFORCE_UPSTREAM_PORT (19360) FP_ENFORCE_PROXY_PORT (19361)
#        FAKE_TOP_FAILS=1  negative control: the top model also fails, the harness must exit non-zero
#        FP_ENFORCE_NO_DOC=1  do not touch the docs file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SKIP_BUILD=false
for _arg in "$@"; do [ "$_arg" = "--skip-build" ] && SKIP_BUILD=true; done

if ! $SKIP_BUILD; then
  echo "==> building firstpass proxy (release)..."
  if ! BUILD_OUT=$(cargo build --release -p firstpass-proxy 2>&1); then
    printf '%s\n' "$BUILD_OUT" | tail -20
    echo "ERROR: cargo build failed"; exit 1
  fi
fi
PROXY_BIN="$REPO_ROOT/target/release/firstpass"
[ -x "$PROXY_BIN" ] || { echo "ERROR: proxy binary not found"; exit 1; }

UPSTREAM_PORT="${FP_ENFORCE_UPSTREAM_PORT:-19360}"
PROXY_PORT="${FP_ENFORCE_PROXY_PORT:-19361}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fp-enforce.XXXXXX")" || { echo "ERROR: mktemp failed"; exit 1; }
REQUEST_LOG="$WORK_DIR/requests.json"
DB_PATH="$WORK_DIR/fp-enforce.db"
CONFIG="$WORK_DIR/firstpass.toml"
UP_PID=""; FP_PID=""
_cleanup() {
  [ -n "$UP_PID" ] && kill "$UP_PID" 2>/dev/null
  [ -n "$FP_PID" ] && kill "$FP_PID" 2>/dev/null
  if [ -n "$FP_ENFORCE_KEEP" ]; then echo "kept work dir: $WORK_DIR"; elif [ -n "$WORK_DIR" ]; then rm -rf "$WORK_DIR"; fi
}
trap _cleanup EXIT
trap 'exit 130' INT TERM

# ---- fake upstream (scripted per model) -------------------------------------------------------
FAKE_SCRIPT=enforce python3 "$SCRIPT_DIR/fake_upstream.py" "$UPSTREAM_PORT" "$REQUEST_LOG" >/dev/null &
UP_PID=$!
for _i in $(seq 1 50); do
  [ -f "$REQUEST_LOG" ] && break
  kill -0 "$UP_PID" 2>/dev/null || break
  sleep 0.1
done
if [ ! -f "$REQUEST_LOG" ] || ! kill -0 "$UP_PID" 2>/dev/null; then
  echo "ERROR: fake upstream did not start on $UPSTREAM_PORT (set FP_ENFORCE_UPSTREAM_PORT)" >&2; exit 1
fi

# ---- enforce config ---------------------------------------------------------------------------
# Both ladders use built-in prices. The claude-code route matches by agent; everything else (aider,
# plain OpenAI clients) falls to the OpenAI route. Provider base URLs come from the
# FIRSTPASS_UPSTREAM_* env below, so no key or hostname lives in this file.
cat >"$CONFIG" <<EOF
[[route]]
match = { agent = "claude-code" }
mode = "enforce"
ladder = ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]
gates = ["non-empty", "tool-args-valid"]

[[route]]
match = {}
mode = "enforce"
ladder = ["openai/gpt-4.1-mini", "openai/gpt-5.5"]
gates = ["non-empty", "tool-args-valid"]

[escalation]
max_rungs_per_request = 2
session_promotion = { after_failures = 3, window = "30m" }
EOF

export FIRSTPASS_CONFIG="$CONFIG"
export FIRSTPASS_UPSTREAM_ANTHROPIC="http://127.0.0.1:$UPSTREAM_PORT"
export FIRSTPASS_UPSTREAM_OPENAI="http://127.0.0.1:$UPSTREAM_PORT"
export FIRSTPASS_DB="$DB_PATH"
export FIRSTPASS_BIND="127.0.0.1:$PROXY_PORT"
# Keys the doctor/provider build expects; dummy values, never sent anywhere but localhost.
export ANTHROPIC_API_KEY=dummy OPENAI_API_KEY=dummy

echo "==> firstpass doctor"
DOCTOR_OUT="$("$PROXY_BIN" doctor 2>&1)"; DOCTOR_RC=$?
printf '%s\n' "$DOCTOR_OUT" | tee "$WORK_DIR/doctor.txt"
if [ "$DOCTOR_RC" -ne 0 ]; then echo "FAIL: firstpass doctor unhealthy (exit $DOCTOR_RC)" >&2; exit 1; fi

# ---- proxy ------------------------------------------------------------------------------------
"$PROXY_BIN" up >"$WORK_DIR/proxy.log" 2>&1 &
FP_PID=$!
_ready=false
for _i in $(seq 1 40); do
  if curl -sf --max-time 1 "http://127.0.0.1:$PROXY_PORT/healthz" -o /dev/null 2>/dev/null; then _ready=true; break; fi
  kill -0 "$FP_PID" 2>/dev/null || break
  sleep 0.3
done
if ! $_ready; then echo "ERROR: proxy not ready"; cat "$WORK_DIR/proxy.log" >&2; exit 1; fi
echo "==> proxy ready on $PROXY_PORT (enforce route config: $CONFIG)"

# Run a command with a wall-clock timeout; rc 124 on timeout. Output goes to $2.
run_with_timeout() {
  local _secs=$1 _out=$2; shift 2
  "$@" >"$_out" 2>&1 &
  local _p=$! _n=0
  while [ $_n -lt $((_secs * 2)) ]; do
    kill -0 $_p 2>/dev/null || { wait $_p; return $?; }
    sleep 0.5; _n=$((_n + 1))
  done
  kill $_p 2>/dev/null; return 124
}

HAVE=""
# ---- claude: two headless runs so one shows an escalation and one a cheap valid tool turn -----
CLAUDE_BIN="$(command -v claude 2>/dev/null || true)"
if [ -n "$CLAUDE_BIN" ]; then
  HAVE="$HAVE claude"
  for n in 1 2; do
    echo "==> claude -p run $n"
    _home="$WORK_DIR/claude-home-$n"; mkdir -p "$_home" "$WORK_DIR/claude-cwd-$n"
    (
      cd "$WORK_DIR/claude-cwd-$n" || exit 1
      ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT" ANTHROPIC_API_KEY=dummy HOME="$_home" \
      ANTHROPIC_CUSTOM_HEADERS="x-firstpass-agent: claude-code" \
      DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
      run_with_timeout 90 "$WORK_DIR/claude-$n.out" \
        "$CLAUDE_BIN" --no-session-persistence --output-format json --dangerously-skip-permissions \
          -p "Run the shell command: echo fp-enforce-ok"
      echo $? >"$WORK_DIR/claude-$n.rc"
    )
    echo "    exit $(cat "$WORK_DIR/claude-$n.rc")"
  done
fi

# ---- aider ------------------------------------------------------------------------------------
AIDER_BIN="$(command -v aider 2>/dev/null || true)"
if [ -n "$AIDER_BIN" ]; then
  HAVE="$HAVE aider"
  echo "==> aider --message"
  mkdir -p "$WORK_DIR/aider-home" "$WORK_DIR/aider-cwd"
  (
    cd "$WORK_DIR/aider-cwd" || exit 1
    HOME="$WORK_DIR/aider-home" OPENAI_API_KEY=dummy \
    run_with_timeout 120 "$WORK_DIR/aider.out" \
      "$AIDER_BIN" --no-git --yes --no-auto-commits --analytics-disable --no-check-update \
        --no-show-model-warnings --no-show-release-notes --openai-api-key=dummy \
        --openai-api-base="http://127.0.0.1:$PROXY_PORT/v1" --model openai/gpt-4.1-mini \
        --message "say hi"
    echo $? >"$WORK_DIR/aider.rc"
  )
  echo "    exit $(cat "$WORK_DIR/aider.rc")"
fi

# ---- assertions + doc section -----------------------------------------------------------------
DOC_ARG=()
[ -z "$FP_ENFORCE_NO_DOC" ] && DOC_ARG=(--doc "$REPO_ROOT/docs/benchmarks/agent-compat.md")
python3 -I "$SCRIPT_DIR/verify_enforce.py" \
  --work "$WORK_DIR" --db "$DB_PATH" --requests "$REQUEST_LOG" --bin "$PROXY_BIN" \
  --have "${HAVE# }" --doctor-file "$WORK_DIR/doctor.txt" \
  --top-fails "${FAKE_TOP_FAILS:-0}" --upstream-port "$UPSTREAM_PORT" "${DOC_ARG[@]}"
RC=$?
echo "==> harness exit code: $RC"
exit $RC
