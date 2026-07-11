# Runbook: running a soak

A soak is how you earn confidence in a Firstpass deployment before flipping it
from **observe** to **enforce** (or before trusting an enforce deployment at
higher volume). This runbook covers a self-hosted, single-operator soak — no
hosted plane exists yet (see [ADR 0001](../adr/0001-hosted-ga-architecture.md)),
and no soak period has been run and closed out yet
([ADR 0003](../adr/0003-production-ga-readiness.md) lists a real soak as a
GA exit criterion, not a completed one). Treat this as the procedure, not a
report of results.

## 1. Deploy in observe mode

Observe mode routes and gates every request but **never changes what's
served** — the top-rung/default response goes to the caller regardless of
gate verdict, so a soak adds zero latency and zero risk to production traffic
while you collect signal on what Firstpass *would* have decided.

```bash
FIRSTPASS_MODE=observe firstpass up
```

**How you know it worked:** `firstpass doctor` reports a healthy config, and
`GET /healthz` returns `200`. Pull a few recent traces with
`firstpass trace --limit 20` and confirm each has a `mode: "observe"` field
and that the served output matches the top-rung/default candidate, not
whatever the gate picked.

## 2. What to watch during the soak

There is no metrics/alerting exporter in the codebase today (see
[`docs/compliance/soc2-controls.md`](../compliance/soc2-controls.md) —
monitoring is a documented gap, not a shipped Prometheus endpoint). Watch
these signals by tailing structured `tracing` output and querying the trace
store directly:

| Signal | How to check | What "healthy" looks like |
| --- | --- | --- |
| Trace-writer liveness | grep logs for `"trace writer: failed to load chain head, stopping"` (`crates/firstpass-proxy/src/store.rs:107`) | Never appears. If it does, the writer thread is dead and traces have stopped — this is a Sev-1, not a warning to note and move on |
| Dropped traces under load | grep logs for `"trace channel full; dropping trace"` / `"trace writer is gone; dropping trace"` (`proxy.rs:66,69`) | Zero, or rare and correlated with a known traffic spike |
| Gate error-budget trips | grep logs for `"gate exceeded its error budget — auto-disabled (ALARM)"` (`crates/firstpass-proxy/src/gate.rs:192`) | Zero. Any trip means a gate is silently being skipped for every subsequent request — investigate before continuing the soak |
| Gate crashes / timeouts | grep logs for gate `abstain` verdicts with reason `gate_crash` or `timeout` | Rare, and each one should map to a known-flaky gate you're already tracking |
| Provider failover rate | Query traces for `verdict: "fail"` entries followed by a cross-provider retry | Consistent with the failure rate of the upstream provider you're depending on — a spike means investigate the provider, not Firstpass |
| Hash-chain integrity | Periodically re-derive the chain with `firstpass_core::verify_chain` over `load_all_traces` (there is no dedicated CLI verify flag today — `cargo run -p firstpass-proxy --example demo` shows the pattern: it drives a decision, then re-verifies the sealed chain) | Verifies clean every time. A verification failure is a Sev-1 — it means the trace file was tampered with or corrupted |

Pull this from the trace DB directly if you want it in a spreadsheet instead
of grepping logs — `firstpass trace --limit N` prints recent traces as JSON
lines, each carrying `mode`, `verdict`, `gates[]`, and cost fields (see
`crates/firstpass-proxy/src/store.rs` schema).

## 3. Success criteria

Before calling a soak "passed" and switching to enforce (or before trusting
an existing enforce deployment):

- **No Sev-1** (trace-writer death, hash-chain verification failure, or a
  sustained gate error-budget trip) for the soak window. 30 days is the
  target window referenced in the [README roadmap](../../README.md); shorter
  is acceptable for a low-volume deployment if you say so explicitly.
- **Error budget:** dropped-trace rate and gate-crash rate each stay under
  whatever threshold you've pre-committed to (there is no built-in default —
  set one before you start, not after you see the numbers).
- **Observe-mode decisions match your expectation** — spot-check a sample of
  traces where the gate would have failed the cheap tier and confirm the
  escalation decision looks right before trusting it to actually change what
  gets served.

## 4. Rollback

If the soak surfaces a Sev-1 or you need to back out of enforce mode:

1. Flip back to observe immediately: `FIRSTPASS_MODE=observe` (restart, or
   redeploy with the env var changed — there is no hot-reload).
2. If a specific gate is the problem, disable it in your gate config rather
   than the whole deployment — the error-budget mechanism will already have
   auto-disabled a gate that's failing beyond its threshold; confirm it did
   and leave it disabled while you fix the gate.
3. If you're rolling back a *release* (not just a mode), see
   [`docs/runbooks/release.md`](release.md) for pinning to a prior tagged
   version.
4. Re-run the soak from step 1 after the fix — don't resume mid-window;
   Sev-1s reset the clock.
