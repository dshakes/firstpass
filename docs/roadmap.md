# Firstpass roadmap — verified cascade routing, phased to GA

Firstpass is **verified cascade routing**: predict where to *start* (a learned start-rung),
verify before you *serve* (gates on the actual output), bound what slips through (a
distribution-free served-failure guarantee), and give every decision a tamper-evident
receipt. The routing literature's two documented failure modes — predictive routers that
collapse toward always-picking-the-strongest model, and cascades that pay for every rung —
are exactly what the predict-to-start + verify-to-serve split avoids; the guarantee and the
receipt are the parts no category incumbent ships.

This roadmap is ordered by a single rule: **never claim what a committed command can't
reproduce**, then widen the traffic the engine can route, then deepen the science, then
harden for GA. Each phase has a hard exit gate.

## Phase 0 — Truth & trust

Make every public claim reproducible or remove it.

- [x] `schema` gate kind wired end-to-end (`[[gate]] schema = {…}`).
- [x] Per-gate `on_abstain = "fail_open" | "fail_closed"` (§7.2) — a gate whose silence must
      never mean approval can now block serving, while the receipt still records the honest
      abstain.
- [x] Judge and self-consistency gates price their own model calls onto the receipt
      (`GateResult.cost_usd`), so budget and savings math see the true cost of proof.
- [x] Value metrics exported: `firstpass_cost_usd_total`, `firstpass_gate_cost_usd_total`,
      `firstpass_baseline_usd_total`, `firstpass_savings_usd_total`,
      `firstpass_served_rung_total{rung,model}`.
- [x] `firstpass savings [--json]` — spend vs the always-top counterfactual, measured from
      the operator's own receipts.
- [x] Benchmark result artifacts committed alongside the numbers the docs cite
      (`docs/benchmarks/`); the repro command for each number stated next to it.
- [x] Provider table labeled honestly: **wire-verified** vs **implemented, awaiting live
      verification** — labels flip only via CI evidence (provider-smoke workflow).
- [ ] crates.io publish (token-gated release step).

**Exit gate:** every number and provider named in public surfaces regenerates from a
committed command; no "unverified" code path sits behind a "live" claim.

## Phase 1 — Route real traffic

The target workload is agents, and agents send tools on nearly every call. Enforce mode
must route that traffic, not fall back around it.

- [x] Full tool/multimodal round-trip through the ladder — verbatim raw-body carry per rung;
      `enforce_structured` default-on behind the fidelity guard (ADR 0005 P4/P5).
- [x] Enforce SSE: connection opens immediately with spec-compliant keepalives while the
      ladder routes; the gated result then streams as the full event sequence. (Token-level
      pass-through streaming is impossible under verify-before-serve — by design.)
- [x] OpenAI-compatible inbound endpoint (`/v1/chat/completions`) alongside the Anthropic
      one (SPEC §M1).
- [x] Per-provider live smoke tests in CI (`provider-smoke.yml`, key-gated); badges flip to
      verified only on CI green. Anthropic proven; others await repo secrets (human gate).

**Exit gate:** a real coding agent completes a full session through enforce with tools and
streaming — receipts intact, zero fidelity loss, measured not asserted.

## Phase 2 — Science upgrade

Adopt the strongest published techniques, then benchmark against the strongest published
baseline.

- [x] Thompson sampling (with discounting) for the start-rung (`algorithm = "thompson"`,
      ADR 0007): native Monte-Carlo propensities (clean off-policy estimates without the
      epsilon overlay), geometric forgetting for model churn. Receipts remain the durable
      state (warm-start on boot). Default stays `ucb1` until a live A/B promotes it.
- [ ] Per-rung P(gate-pass | features) prediction rather than a single difficulty scalar.
- [ ] **Elastic verification** (ADR 0008, research — go/no-go gated): probe-before-commit, verify
      *proportional to doubt*, with a conformal guarantee over the *verify/skip* decision so
      un-verified serves keep the same served-failure bound. Validated by `firstpass-bench
      --probe-study` before any skip logic ships.
- [x] Learn-then-Test threshold calibration (`--method ltt` in `firstpass calibrate`):
      distribution-free finite-sample risk control via fixed-sequence exact-binomial testing
      (Angelopoulos et al. 2021 / RCPS). Includes per-λ diagnostics and the gate's empirical
      false-accept rate (verifier ROC point) at the chosen threshold.
- [ ] Live adaptive-conformal loop closing the guarantee under drift: ACI wired to the
      realized-served-failure gauge so the bound holds as the workload shifts.
- [x] Verifier-imperfection rails: sample counts are hard-capped (consistency k &le; 8) and
      LTT calibration reports the gate's observed false-accept rate at the chosen threshold
      (an imperfect verifier inverse-scales — bounded use is a feature, not a limit).
- [ ] Joint (rung, sample-count) routing: cheap model + k-sample agreement often dominates
      escalation on the easy/medium slice, and the gate makes exploiting that safe.
- [x] Speculative deferral: `speculation_band = [low, high]` — prefetch fires only when the
      bandit's gate-pass estimate is in the marginal zone; confident requests run serial and
      keep the tokens (`firstpass_speculation_skipped_total`).
- [x] Doubly-robust estimator alongside IPS/SNIPS in `firstpass ope`.
- [ ] Reproducible benchmark vs the unified routing+cascading policy from the literature,
      including a drift scenario in which the bound must hold.

**Exit gate:** a committed artifact showing ≥ cascade-routing cost/quality *plus* the bound
holding under induced drift.

## Phase 3 — GA hardening

GA is an audit + soak + process stamp, not a code stamp (ADR 0003).

- [ ] External security review of tenant auth + key custody (ADR 0004 §D7) — human gate.
- [x] Durable receipts: `FIRSTPASS_RECEIPTS=durable` spills to `<db>.spill.jsonl` under
      backpressure (synced, ordered) and drains on boot with the hash chain verified valid —
      no receipt is ever silently dropped. (Postgres store option remains open.)
- [ ] Observability suite: per-provider/per-rung/per-gate latency + failure + cost series,
      committed dashboards, false-pass SLO alarm (SPEC §M3).
- [ ] 30-day soak on real agent traffic — calendar gate.
- [x] Price-table refresh mechanism: `[[price]]` per-deployment overrides.

**Exit gate:** ADR 0003 checklist green.

## Phase 4 — Hosted plane & compounding moats

- [ ] Hosted control plane (ADR 0004, post-review) with team dashboards.
- [x] Policy rehearsal as a product surface: `firstpass ope` (CLI) plus the MCP
      `rehearse_policy` tool — an agent replays a candidate policy over its own logged traffic
      and reads back estimated cost + served-failure before enforcing anything.
- [x] Receipts as a compliance artifact: `firstpass export` (sealed JSONL) + `firstpass verify`
      re-derive the hash chain from genesis with no proxy/DB in the loop — tamper and reorder
      break the chain at their index and exit non-zero. Live-proven end-to-end.
- [x] SLO-backed guarantee language: [docs/compliance/slo.md](compliance/slo.md) states the
      served-failure SLO contractually and ties it to the same receipts an auditor verifies.
- [x] MCP tools for savings, rehearsal, and route explanation: `get_savings`, `get_evals`,
      `rehearse_policy`, `explain_route`, `verify_receipts` (plus trace read + feedback).

**Exit gate:** first external design partner running the hosted plane with the guarantee in
their contract.
