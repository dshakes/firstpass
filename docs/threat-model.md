# Threat model

STRIDE analysis of Firstpass's trust boundaries, as the code actually
implements them today. Scope matches [`SECURITY.md`](../SECURITY.md): the
`firstpass-proxy` crate (proxy, gates, trace store, feedback/MCP/CLI), the
`firstpass-bench` code-execution sandbox, and `firstpass-core`. Firstpass is
**single-operator** today — no hosted multi-tenant plane exists (see
[ADR 0001](adr/0001-hosted-ga-architecture.md), which is `Proposed`, not
built).

## Assets

- **Provider API keys** (BYOK) — Anthropic/OpenAI credentials the operator's
  agent supplies per request.
- **The tamper-evident trace** — the SQLite hash chain that is Firstpass's
  audit product; its integrity is the thing customers are meant to be able to
  trust.
- **The served model output** — what actually reaches the caller; an attacker
  who can force a bad answer to pass a gate defeats the entire product thesis.
- **Host resources** — CPU/memory/disk on the machine running the proxy or
  bench harness, and (for the sandbox) the host OS itself.

## Entry points

| Entry point | Code | Who talks to it |
| --- | --- | --- |
| `POST /v1/messages` | `crates/firstpass-proxy/src/proxy.rs:82` | The operator's own agent (BYOK key on the request) |
| `GET /v1/capabilities`, `GET /healthz` | `proxy.rs:83-91` | Operator tooling, health checks |
| Feedback API (deferred verdicts) | `proxy.rs` (`append_deferred`) | Operator, post-hoc |
| `firstpass` CLI (`doctor`, `trace`, `calibrate`, `mcp`) | `crates/firstpass-proxy/src/bin/firstpass.rs` | Local operator / MCP client |
| Gate subprocess stdin | `crates/firstpass-proxy/src/subprocess.rs` | Proxy → operator-supplied gate binary |
| Sandbox exec | `crates/firstpass-bench/src/sandbox.rs` | Bench harness → `runsc`/`runc` running **model-generated code** |
| Outbound provider calls | `crates/firstpass-proxy/src/provider.rs` | Proxy → Anthropic/OpenAI, carrying the forwarded BYOK key |
| On-disk trace DB | `crates/firstpass-proxy/src/store.rs` | Background writer thread; anyone with filesystem access to the `.db` file |

## Trust boundaries × STRIDE

### 1. BYOK provider keys (per-request, in-flight)

The proxy reads `x-api-key` (Anthropic) / `Authorization: Bearer` (OpenAI)
straight off the inbound request (`provider.rs:117,218`) or an env var
fallback for local single-operator use (`config.rs:61-70`) and forwards it
to the upstream provider. It is never written to the trace store, and a gate
subprocess never receives it (`subprocess.rs:11`: "A gate never gets the API
keys or anything beyond the candidate + request metadata it needs.").

| STRIDE | Risk | Mitigation in place | Residual risk |
| --- | --- | --- | --- |
| Spoofing | A caller impersonates a legitimate agent to spend the operator's provider budget | None at the proxy — it trusts whatever key arrives on the request, same as calling the provider directly | Deploy Firstpass behind the same network boundary you'd put a raw provider call behind; no additional authn layer exists yet |
| Information disclosure | Key leaks via logs, error responses, or the trace | Structured `tracing` logs never log the key; error responses are opaque (`error.rs:101`, "operators keep the diagnostic while clients get an opaque message"); trace records carry request/response content, not credentials | A `Debug` derive added carelessly to a struct that transiently holds the key could leak it into a log line — no automated secret-scanning in CI today (tracked as a GA gap in ADR 0003) |
| Tampering | Key modified in flight | TLS is the operator's responsibility (ingress), not provided by Firstpass itself | Operator must terminate TLS in front of Firstpass; plaintext HTTP exposes the key on the wire |
| Repudiation | N/A — keys aren't tied to trace entries | — | — |

### 2. The on-disk SQLite trace store

Every decision is written as a link in a hash chain (`store.rs`): each row's
hash is `H(prev_hash || trace fields)`, starting from `GENESIS_HASH`, so
altering or deleting a row breaks every hash after it. The writer runs off a
**bounded** channel (`proxy.rs:66-69`) — the hot request path `try_send`s and
never blocks; a full or gone writer causes traces to be dropped with a
warning, not an OOM.

| STRIDE | Risk | Mitigation in place | Residual risk |
| --- | --- | --- | --- |
| Tampering | Operator (or an attacker with disk access) edits a past trace row | Hash chain makes any edit detectable by re-deriving the chain (`firstpass trace` / `verify`) | Detection, not prevention — nothing stops the file from being edited or replaced wholesale (e.g. truncated and re-genesis'd) if an attacker controls the filesystem; there's no off-host, append-only mirror |
| Repudiation | Operator denies a decision was made | Hash-chained trace is designed to make this costly to fake | No third-party notarization/anchoring (e.g. periodic external timestamp) exists yet — a sufficiently privileged attacker with time could regenerate a self-consistent forged chain from genesis |
| Denial of service | Trace writer starved (disk full, DB locked) blocks or crashes request serving | Bounded channel + load-shedding (`proxy.rs:66-69`) decouples the hot path; writer failures are logged, not fatal to serving | If the writer thread dies (`store.rs:107`, "trace writer: failed to load chain head, stopping"), traces silently stop being written until restart — no alert wired to that condition yet (see SOC2 monitoring gap) |
| Information disclosure | Trace DB contains full request/response content (potentially sensitive prompts/candidates) | File lives on the operator's own disk under the operator's own permissions — no network exposure by Firstpass | No at-rest encryption of the trace DB; operator must supply disk-level encryption if required |

### 3. Untrusted model-generated code (bench sandbox)

The coding-with-tests benchmark executes **model-generated code**, the
highest-risk input class in the system. [ADR 0002](adr/0002-bench-code-execution-sandbox.md)
governs this boundary explicitly.

| STRIDE | Risk | Mitigation in place | Residual risk |
| --- | --- | --- | --- |
| Elevation of privilege | Candidate code escapes its container to the host | Prefers `runsc` (gVisor, syscall-filtering user-space kernel); falls back to `runc` **only with a printed warning naming the weaker isolation** (ADR 0002 D2) | `runc`-only hosts (no gVisor available) run with process-namespace isolation only, not a security boundary against a determined kernel-exploit attacker — the ADR treats this as an accepted, disclosed gap, not a silent one |
| Denial of service | Candidate code spins forever or exhausts host resources | `--network none`, capability drop, resource caps (CPU/mem/pids), no host mounts (ADR 0002 D1); `TimedOut`/`Killed` outcomes are counted as gate fails, not silently dropped (ADR 0002 D3) | Resource caps are per-container; no cluster-wide concurrency cap on the bench harness itself |
| Tampering / spoofing of results | Candidate code tries to fake a passing test result | Sandbox is fail-**closed**: if the runtime is missing, the image is unavailable, or the fail-closed preflight doesn't demonstrate isolation, the harness **aborts** rather than falling back to running on the host (ADR 0002 D3) | Preflight only checks isolation, not that the specific test harness inside the container is itself uncompromised — trusts the container image build |

### 4. Operator-supplied gate code (subprocess / judge)

Gates are the second execution surface, but the trust direction is inverted
from the sandbox: the **gate program is operator-trusted**, and only the
**candidate** (model output) it evaluates is untrusted.

| STRIDE | Risk | Mitigation in place | Residual risk |
| --- | --- | --- | --- |
| Tampering (command/argument injection) | Model output crafted to be interpreted as shell/CLI flags by the gate process | Candidate is passed as JSON **on stdin, never as an argv element** (`subprocess.rs:6-11`) — no shell string interpolation of model output | If an operator's *own* gate script naively `eval`s or shells out with stdin content, that's outside Firstpass's control surface |
| Denial of service | A gate hangs or errors repeatedly, stalling every request behind it | Per-gate timeout kills the child (`abstain`, reason `timeout`); an **error budget** auto-disables a gate that errors past its threshold and logs an ALARM (`gate.rs:143-192`) | `gate_health` is shared mutable state across all requests process-wide — flagged in ADR 0003 as a hosted/multi-tenant risk (one tenant's flaky gate could trip another's budget); not a concern in today's single-operator model |
| Elevation of privilege | Gate subprocess itself is malicious or compromised | None beyond OS process isolation — the gate runs with the same privileges as the proxy process | The gate command is 100% operator-configured and operator-trusted by design (SPEC §8); Firstpass does not sandbox gate *processes* the way it sandboxes bench *candidates* |

### 5. Multi-tenant / hosted plane (future)

There is no hosted, multi-tenant deployment mode today. [ADR 0001](adr/0001-hosted-ga-architecture.md)
proposes one (cloud + KMS-backed envelope encryption for keys, per-tenant
isolation) but it is **not implemented** — encryption-at-rest via KMS is a
documented gap, not a shipped control (see
[`docs/compliance/soc2-controls.md`](compliance/soc2-controls.md)). Anyone
running Firstpass today is running it single-tenant, on their own
infrastructure, under their own trust.

## Summary table

| Trust boundary | Primary risk | Status |
| --- | --- | --- |
| BYOK keys in flight | Leak via logs/errors | Mitigated (opaque errors, no key in trace); TLS is operator's job |
| Trace store on disk | Silent tamper / silent writer death | Tamper *detected* via hash chain; no alerting on writer death yet |
| Bench sandbox | Container escape | Fail-closed + gVisor-preferred; `runc` fallback is a disclosed, not eliminated, gap |
| Gate subprocess | Injection via candidate | Stdin-only contract closes the injection vector; gate binary itself is operator-trusted |
| Gate error budget | Cross-tenant DoS | Not a live risk single-tenant; would need per-tenant budgets before multi-tenant GA |
| Multi-tenant / hosted plane | Everything above, at higher stakes | Does not exist — ADR 0001 proposed, not built |
