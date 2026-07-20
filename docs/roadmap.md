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
- [ ] Benchmark result artifacts committed alongside the numbers the docs cite; the repro
      command for each number stated next to it.
- [ ] Provider table labeled honestly: **wire-verified** vs **implemented, awaiting live
      verification** — labels flip only via CI evidence.
- [ ] crates.io publish (token-gated release step).

**Exit gate:** every number and provider named in public surfaces regenerates from a
committed command; no "unverified" code path sits behind a "live" claim.

## Phase 1 — Route real traffic

The target workload is agents, and agents send tools on nearly every call. Enforce mode
must route that traffic, not fall back around it.

- [ ] Full tool/multimodal round-trip through the ladder; `enforce_structured` default-on
      once fidelity-tested.
- [ ] True SSE streaming on the enforce path (stream the winning rung; no buffer-then-replay).
- [ ] OpenAI-compatible inbound endpoint (`/v1/chat/completions`) alongside the Anthropic
      one (SPEC §M1).
- [ ] Per-provider live smoke tests in CI (key-gated, cents/day); provider badges flip to
      verified only on CI green.

**Exit gate:** a real coding agent completes a full session through enforce with tools and
streaming — receipts intact, zero fidelity loss, measured not asserted.

## Phase 2 — Science upgrade

Adopt the strongest published techniques, then benchmark against the strongest published
baseline.

- [ ] Contextual Thompson sampling (with discounting) for the start-rung: native
      non-degenerate propensities (clean off-policy estimates), non-stationarity handling
      under model churn, persistent state keyed by ladder identity.
- [ ] Per-rung P(gate-pass | features) prediction rather than a single difficulty scalar.
- [ ] Learn-then-Test threshold calibration + the live adaptive-conformal loop, closing the
      guarantee with the realized-served-failure gauge: a served-failure bound that holds
      under drift.
- [ ] Verifier-imperfection rails: cap samples per rung; feed each gate's observed
      error profile into calibration (an imperfect verifier inverse-scales — bounded use is
      a feature, not a limit).
- [ ] Joint (rung, sample-count) routing: cheap model + k-sample agreement often dominates
      escalation on the easy/medium slice, and the gate makes exploiting that safe.
- [ ] Speculative deferral: prefetch the next rung only in the marginal serve-probability
      band, under a latency SLA.
- [ ] Doubly-robust estimator alongside IPS/SNIPS in `firstpass ope`.
- [ ] Reproducible benchmark vs the unified routing+cascading policy from the literature,
      including a drift scenario in which the bound must hold.

**Exit gate:** a committed artifact showing ≥ cascade-routing cost/quality *plus* the bound
holding under induced drift.

## Phase 3 — GA hardening

GA is an audit + soak + process stamp, not a code stamp (ADR 0003).

- [ ] External security review of tenant auth + key custody (ADR 0004 §D7) — human gate.
- [ ] Durable receipts: never-drop mode (block or spill) and a Postgres store option.
- [ ] Observability suite: per-provider/per-rung/per-gate latency + failure + cost series,
      committed dashboards, false-pass SLO alarm (SPEC §M3).
- [ ] 30-day soak on real agent traffic — calendar gate.
- [ ] Price-table refresh mechanism (prices drift; savings math must not).

**Exit gate:** ADR 0003 checklist green.

## Phase 4 — Hosted plane & compounding moats

- [ ] Hosted control plane (ADR 0004, post-review) with team dashboards.
- [ ] Policy rehearsal as a product surface: replay a candidate policy over your own logged
      traffic before enforcing it.
- [ ] Receipts as a compliance artifact: export/verify tooling an external auditor can run.
- [ ] SLO-backed guarantee language.
- [ ] MCP tools for savings, rehearsal, and route explanation.

**Exit gate:** first external design partner running the hosted plane with the guarantee in
their contract.
