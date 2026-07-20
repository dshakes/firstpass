#!/usr/bin/env bash
# One provider's live smoke: boot the enforcing proxy with a one-rung ladder on this provider,
# send a tiny real request through it, and assert (1) HTTP 200, (2) non-empty served content,
# (3) a receipt landed in the trace store. This is what "wire-verified" means — the whole
# firstpass path (parse → route → provider adapter → gate → serve → receipt) against the real
# endpoint, not a unit test of a recorded shape.
#
# Inputs (env): PROVIDER_KEY (the secret), KEY_ENV_NAME (env var name the provider def reads,
# e.g. GROQ_API_KEY), MODEL (ladder rung, e.g. groq/llama-3.3-70b-versatile), EXTRA_TOML
# (optional [[provider]] block for non-builtin providers).
#
# Cost: one request, max_tokens=32 — cents at most.
set -euo pipefail

: "${PROVIDER_KEY:?PROVIDER_KEY required}"
: "${MODEL:?MODEL required}"
KEY_ENV_NAME="${KEY_ENV_NAME:-ANTHROPIC_API_KEY}"
EXTRA_TOML="${EXTRA_TOML:-}"

workdir="$(mktemp -d)"
trap 'kill "${proxy_pid:-0}" 2>/dev/null || true; rm -rf "$workdir"' EXIT

cat > "$workdir/firstpass.toml" <<TOML
${EXTRA_TOML}
[[route]]
match = {}
mode = "enforce"
ladder = ["${MODEL}"]
gates = ["non-empty"]
TOML

# The provider def reads its key from KEY_ENV_NAME at call time; builtins take BYOK headers too.
export "${KEY_ENV_NAME}=${PROVIDER_KEY}"
export FIRSTPASS_MODE=enforce
export FIRSTPASS_CONFIG="$workdir/firstpass.toml"
export FIRSTPASS_DB="$workdir/firstpass.db"
export FIRSTPASS_BIND=127.0.0.1:18099

./target/debug/firstpass-proxy > "$workdir/proxy.log" 2>&1 &
proxy_pid=$!

for _ in $(seq 1 50); do
  curl -sf http://127.0.0.1:18099/healthz > /dev/null 2>&1 && break
  sleep 0.2
done
curl -sf http://127.0.0.1:18099/healthz > /dev/null || { echo "proxy never became healthy"; cat "$workdir/proxy.log"; exit 1; }

body='{"model":"'"${MODEL}"'","max_tokens":32,"messages":[{"role":"user","content":"Reply with the single word OK"}]}'
status=$(curl -s -o "$workdir/resp.json" -w "%{http_code}" \
  -H "content-type: application/json" \
  -H "x-api-key: ${PROVIDER_KEY}" \
  -H "authorization: Bearer ${PROVIDER_KEY}" \
  -X POST http://127.0.0.1:18099/v1/messages \
  --data "$body")

echo "--- response (${status}):"
cat "$workdir/resp.json"; echo

[ "$status" = "200" ] || { echo "FAIL: expected 200, got ${status}"; cat "$workdir/proxy.log"; exit 1; }
python3 - "$workdir/resp.json" <<'PY'
import json, sys
resp = json.load(open(sys.argv[1]))
content = resp.get("content") or []
text = "".join(b.get("text", "") for b in content if isinstance(b, dict))
assert text.strip(), f"served content is empty: {resp}"
print(f"served text: {text.strip()[:80]!r}")
PY

# The receipt: at least one trace recorded for this request (written off-path — give the
# background writer a beat).
sleep 1
export FIRSTPASS_CONFIG="$workdir/firstpass.toml"
traces=$(./target/debug/firstpass trace --limit 1)
echo "--- receipt:"; echo "$traces" | head -c 400; echo
[ "$traces" != "no traces recorded yet" ] || { echo "FAIL: no receipt recorded"; exit 1; }

echo "PASS: ${MODEL} wire-verified end-to-end (served + receipt)"
