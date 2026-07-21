# ADR 0008 — Elastic verification: value-of-information routing with a conformal guarantee over the verify/skip decision

Status: **proposed — gated on empirical validation** · 2026-07-21

## Context

Firstpass today is a *verification-gated cascade*: run the cheapest rung to completion, run the
full gate on its output, serve on pass, escalate on fail (with a learned start-rung to warm-start
where the ladder begins). This is safe and guaranteed, but it carries two structural taxes,
quantified below with `c` = cheap-model cost, `e` = expensive-model cost, `g` = gate cost per
attempt, `p` = P(cheap output passes the gate):

- **Doomed-attempt tax.** On a query the cheap model was never going to get right, the cascade
  still pays `c + g` before escalating — a full wasted cheap attempt *and* a full wasted gate.
- **Uniform-verification tax.** The gate runs on *every* served output at cost `g`, even when the
  output is obviously fine. When the gate is an LLM judge or k-sample self-consistency, `g` can
  approach `e`, and the cascade can cost **more** than always-calling-the-expensive-model. The
  break-even is `c + (1−p)·g < p·e`: with a cheap deterministic gate (`g≈0`) it holds easily; with
  an expensive gate it frequently fails.

Both taxes come from treating "which model" and "verify or not" as *fixed steps* rather than
*decisions under uncertainty*.

## Decision

Reframe routing as a **value-of-information (VoI) problem** and make verification **elastic** —
spend verification budget proportional to *doubt*, not uniformly — while preserving the
distribution-free served-failure guarantee by extending conformal risk control to cover the
decision of *whether to verify*.

The mechanism, per request:

1. **Probe before commit.** Instead of a full cheap attempt, draw a cheap *probe*: the small model
   at low token budget, k=2–4 samples. The probe yields (a) a candidate output and (b) a calibrated
   self-consistency **uncertainty** signal (semantic-entropy family; a few samples, or a single-pass
   entropy probe). Cost is a fraction of a full attempt.

2. **A learned value function chooses the action.** From the probe uncertainty + request features +
   receipt history, estimate `P(pass | rung, probe signal)` per rung, then take the action with the
   best expected quality-per-dollar among: **serve the probe answer**, **verify it with the gate**,
   **draw more samples**, or **escalate**. The action space is `(model, #samples, verify?)`, not just
   "which model."

3. **Elastic verification (the money-saver).**
   - Probe **confident** + predicted `P(pass)` high → serve **without** the full gate.
   - Probe **ambiguous** → run the gate; this is where verification's information is worth its cost.
   - Probe **confidently hard** → skip the cheap attempt *and* its gate, escalate directly.
   Verification cost concentrates in the ambiguous middle instead of being paid everywhere.

4. **Conformal guarantee over the verify/skip decision (the novel, load-bearing part).** The
   threshold above which we *serve-without-verifying* is calibrated conformally (Learn-then-Test /
   RCPS, already in `firstpass-core::ltt`): choose λ such that served-failure **among the
   un-verified serves** is provably ≤ α at confidence 1−δ. The un-verified serves therefore carry
   the *same* distribution-free bound as the verified ones. We are not trusting the probe — we skip
   verification only where the finite-sample statistics prove it is safe. Adaptive conformal
   (`AdaptiveConformal`) maintains λ under drift.

5. **Modes as constraints, not presets.** A "mode" is an `(α, latency budget, $ budget)` triple fed
   to the same optimizer. `cost` = looser α; `quality` = tight α + verify aggressively; `latency` =
   add a wall-clock constraint that triggers speculative parallel probes; `max` = α→0. The caller or
   agent may set the triple **per turn** via `x-firstpass-mode`, so routing intent is declared per
   request without changing the engine. (Modes ship first as presets over the existing knobs, ADR
   forthcoming; the constraint-optimizer form is the target once the value function lands.)

## The validation gate (why this ADR is "proposed", not "accepted")

The entire architecture rests on one empirical claim: **a cheap probe's self-consistency
uncertainty predicts whether the served output will clear the (hidden) oracle.** If it does not,
elastic verification is unfounded and we keep uniform verification.

This is pre-registered and measured by `firstpass-bench --probe-study` (committed): draw k cheap
samples per real MBPP task, compute uncertainty, serve the candidate a cascade would serve, and
record the oracle outcome. The decision metric is **AUC(uncertainty → oracle-failure)** plus
**skip-safety when confident** (`P(oracle correct | probe confident)`):

- AUC ≤ ~0.55 or low skip-safety → **do not build the skip logic**; the probe is noise. The engine
  stays a uniform-verification cascade and this ADR is withdrawn.
- AUC ≳ 0.65 with high skip-safety → the signal is real; implement in phases below.

The result artifact lands in `docs/benchmarks/` next to the MBPP bound. **No skip logic ships until
this gate is green.**

## Invariants (must never regress)

- **Prediction never overrides proof for the *verified* path.** Elastic verification only changes
  *whether* we verify; when we do verify, the gate still decides. The skip path is governed by the
  conformal bound, not by optimism.
- **The served-failure guarantee holds over the union of verified and un-verified serves**, or the
  skip is not taken. This is the whole point; it is the acceptance criterion.
- **Default-off / byte-identical.** Ships behind config; absent config reproduces today's uniform
  cascade exactly.
- **Auditable.** Every decision records which action the value function took and, for un-verified
  serves, the λ and calibration that authorized the skip — so an auditor can see *why* verification
  was skipped and check the bound held.

## Phased implementation (each phase default-off, gate-clean, ADR-consistent)

1. **Probe + uncertainty signal** in the router (k-sample cheap probe, semantic-consistency score),
   recorded on the receipt. No behavior change yet.
2. **Learned `P(pass | rung, features, probe)`** predictor, trained on receipts; reported only
   (shadow), validated against realized outcomes via the existing OPE machinery.
3. **Elastic-verification serving** behind the LTT-calibrated skip threshold; the conformal-over-skip
   bound validated on held-out + drift splits before default-on is even proposed.
4. **`(rung, n, verify?)` joint optimizer** under per-turn `(α, latency, $)` mode constraints.

## Consequences

- If validated, this is a genuine research contribution: conformal risk control applied to the
  *decision to verify*, which is what lets a router save money without weakening its accuracy
  guarantee. It is assembled from proven parts (semantic entropy; compute-optimal allocation;
  RCPS/LTT; adaptive conformal) — the novelty is the synthesis and the guarantee-over-verification.
- If not validated, we have spent one benchmark run to avoid building a speculative engine on a
  false premise — which is the point of the gate.
- Related: ADR 0007 (Thompson start-rung — the predict-to-start half this builds on), SPEC §10.1
  (conformal serving), `firstpass-core::ltt` (the risk-control machinery reused for the skip bound).
