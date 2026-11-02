# Routing granularity for multi-turn coding agents

date: 2026-11-01 · spend: $0 (public runs) · data: github.com/SWE-bench/experiments@40f164d ·
command: `python3 -I scripts/analyze-agent-granularity.py [CACHE_DIR]`

Pre-registration: [agent-granularity-preregistration.md](agent-granularity-preregistration.md)
(committed at e848709 before any result was computed). See **Deviations from pre-registration**
for all post-preregistration changes to script and interpretation.

---

## Validity caveat

Whole-run policies (always-X, per-session, workload, run-level cascade, escalate-and-pin)
are scored exactly from recorded runs. Per-call cascade (P5) runs cheap to completion then
top from scratch — whole recorded runs, no stitching; quality IS identifiable.

Escalate-and-pin (P4) post-switch quality ≈ top model's own full run (restart semantics;
optimistic for quality since the top model starts fresh, pessimistic for cost; see
pre-registration §Validity caveat).

**Two P4 rows** are reported:
- **Primary (pre-registration)**: cost = partial_cheap + switch_overhead + top_leaderboard_cost,
  per the pre-registration cost model. The top model's first call re-reads the accumulated
  prefix at the full uncached input rate. The overhead applies only when a switch actually
  happens after at least one cheap step: escalating at step 0 is not a switch, and the top
  model's recorded run already includes its own first read.
- **Secondary (restart-semantics)**: cost = partial_cheap + top_leaderboard_cost. No separate
  overhead because the top model restarts from scratch and its full recorded cost already
  includes its own first read. Listed as a deviation below.

Context model calibrated from CALIB_N=20 seeded-random (seed=42) cheap-model trajectories;
I_0 = system + first user message only.

---

## Models

Primary ladder: `claude-4-5-haiku-high` → `claude-4-5-opus-high` (v2.0.0, n=500)
Secondary ladder: `gpt-5-mini` → `gpt-5-2-high` (v2.0.0, n=500)
Pool for P2/P3-pool: 11 models from `20260217_mini-v2.0.0_*`, reference = index 2
(`claude-4-5-opus-high`); n=500

---

## Context calibration (CALIB_N=20 seeded-random sample, cheap model only)

| model | I_0 (tokens) | delta (tokens/step) | out/step (tokens) | avg nc_factor |
|---|---|---|---|---|
| claude-4-5-haiku-high | 2,328 | 722 | 255 | 5.56× |
| gpt-5-mini | 1,349 | 905 | 266 | 3.46× |

nc_factor = estimated no-cache / estimated with-cache for single-model sessions (informational;
not in table). Context model calibrated on cheap model only; top-model parameters not estimated.
Haiku: role-based message format, chars / 2.5. GPT-5-mini: OpenAI Responses API with
authoritative per-step token counts.

---

## Results

### Claude haiku→opus (v2.0.0)  (n=500)

Cross-fit per fold: cascade tau=[19, 34], repo-eps=[0.0, 0.0], P4 tau=[-1, -1] (both variants)

| arm | resolved | $/inst | $/resolved | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|
| always-cheap [P1] | 333/500 (0.666) | $0.331 | $0.497 | [-0.136, -0.072] | [-0.460, -0.387] |
| always-top [P1] | 384/500 (0.768) | $0.754 | $0.982 | — | — |
| run-level cascade (cross-fit) [P5] | 382/500 (0.764) | $1.071 | $1.401 | [-0.010, +0.000] | [+0.295, +0.340] |
| per-session / repo (cross-fit) [P3] | 378/500 (0.756) | $0.720 | $0.953 | [-0.024, -0.002] | [-0.048, -0.021] |
| escalate-and-pin prereg [P4] | 384/500 (0.768) | $0.754 | $0.982 | [+0.000, +0.000] | [+0.000, +0.000] |
| escalate-and-pin restart (secondary) | 384/500 (0.768) | $0.754 | $0.982 | [+0.000, +0.000] | [+0.000, +0.000] |
| random (seed=42) [P6] | 359/500 (0.718) | $0.555 | $0.772 | [-0.074, -0.026] | [-0.232, -0.170] |
| oracle-per-instance (bound) [P6] | 395/500 (0.790) | $0.384 | $0.486 | [+0.010, +0.036] | [-0.408, -0.334] |
| oracle-cascade (bound) | 395/500 (0.790) | $0.679 | $0.859 | [+0.010, +0.036] | [-0.116, -0.032] |

P4 escalation: 500/500 trigger (tau=-1 in both folds).
Fitted tau=−1 means P4 escalates before the cheap model takes a step, so no switch happens and no
switch overhead is charged in either P4 row; both equal always-top. (A switch after the first cheap
step would have cost ≈ 2,328 tokens × $15/1M ≈ $0.035/inst.)

### GPT-5-mini→GPT-5.2-high (v2.0.0)  (n=500)

Cross-fit per fold: cascade tau=[6, 8], repo-eps=[0.05, 0.1], P4 tau=[-1, -1] (both variants)

| arm | resolved | $/inst | $/resolved | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|
| always-cheap [P1] | 281/500 (0.562) | $0.047 | $0.084 | [-0.206, -0.126] | [-0.459, -0.396] |
| always-top [P1] | 364/500 (0.728) | $0.474 | $0.651 | — | — |
| run-level cascade (cross-fit) [P5] | 364/500 (0.728) | $0.520 | $0.714 | [+0.000, +0.000] | [+0.043, +0.050] |
| per-session / repo (cross-fit) [P3] | 353/500 (0.706) | $0.416 | $0.589 | [-0.042, -0.002] | [-0.074, -0.042] |
| escalate-and-pin prereg [P4] | 364/500 (0.728) | $0.474 | $0.651 | [+0.000, +0.000] | [+0.000, +0.000] |
| escalate-and-pin restart (secondary) | 364/500 (0.728) | $0.474 | $0.651 | [+0.000, +0.000] | [+0.000, +0.000] |
| random (seed=42) [P6] | 321/500 (0.642) | $0.284 | $0.442 | [-0.116, -0.056] | [-0.215, -0.166] |
| oracle-per-instance (bound) [P6] | 383/500 (0.766) | $0.154 | $0.201 | [+0.022, +0.056] | [-0.353, -0.289] |
| oracle-cascade (bound) | 383/500 (0.766) | $0.317 | $0.414 | [+0.022, +0.056] | [-0.182, -0.133] |

Fitted tau=−1: no switch, no overhead in either P4 row; both equal always-top. (A switch after the first cheap step would have cost ≈ 1,349 tokens × $5/1M ≈ $0.007/inst.)

### Pool policies  (n=500)

P2: even half → minimax-2-5-high, odd half → gemini-3-flash-high (replicates #239).
P3-pool: fitted eps per fold = [0.2, 0.05]; repo_router picks cheapest within eps of best rate.

| arm | resolved | $/inst | Δ resolved vs always-opus | Δ $/inst vs always-opus |
|---|---|---|---|---|
| pool-selection (P2, cross-fit) | 374/500 (0.748) | $0.219 | [-0.050, +0.008] | [-0.582, -0.490] |
| pool-session (P3-pool, cross-fit) | 382/500 (0.764) | $0.122 | [-0.032, +0.024] | [-0.679, -0.585] |
| always-claude-4-5-opus-high (pool ref) | 384/500 (0.768) | $0.754 | — | — |

**P3-pool vs P2 paired CI** (same 500 instances, pool-controlled comparison):
Δresolved = [-0.004, +0.036], Δ$/inst = [-0.116, -0.077]

### P3-ladder vs P2 cross-comparison (confounded: 2-model ladder vs 11-model pool)

| ladder | P3-ladder | P2 | Δresolved CI | Δ$/inst CI |
|---|---|---|---|---|
| haiku→opus | 378/500, $0.720/inst | 374/500, $0.219/inst | [-0.020, +0.036] | [+0.456, +0.550] |
| gpt-5-mini→5.2 | 353/500, $0.416/inst | 374/500, $0.219/inst | [-0.074, -0.008] | [+0.165, +0.230] |

---

## Verdicts (per pre-registered decision rules)

**Decision rule**: A policy "beats" always-top if BOTH: Δresolved CI lb ≥ −0.01 AND
Δ$/inst CI ub < −0.01. A policy "loses on quality" if lb < −0.01.

| policy | Δresolved CI lb | Δ$/inst CI ub | verdict |
|---|---|---|---|
| P2: workload selection (pool) | −0.050 | −$0.490 | **loses on quality** — lb=−0.050 < −0.01; non-inferiority at 1pp NOT shown; cost decisively better |
| P3-pool vs always-opus | −0.032 | −$0.585 | **loses on quality** vs always-opus (lb=−0.032 < −0.01) |
| P3: per-session/repo, haiku→opus | −0.024 | −$0.021 | **loses on quality** (lb < −0.01) |
| P3: per-session/repo, gpt-5-mini→5.2 | −0.042 | −$0.042 | **loses on quality** (lb < −0.01) |
| P4 prereg: haiku→opus | +0.000 | +$0.000 | **no gain — equals always-top** (fitted tau=−1: escalate at step 0, never a mid-run switch) |
| P4 prereg: gpt-5-mini→5.2 | +0.000 | +$0.000 | **no gain — equals always-top** (fitted tau=−1) |
| P4 restart (secondary): both | +0.000 | +$0.000 | **equals always-top exactly** — same as primary at tau=−1 |
| P5: cascade, haiku→opus | −0.010 | +$0.340 | **negative** — more expensive; quality borderline (lb=−0.010, at threshold) |
| P5: cascade, gpt-5-mini→5.2 | +0.000 | +$0.050 | **negative** — more expensive, same quality |

**P2 quality note**: The #239 "n.s. quality loss" framing uses a two-sided test; the CI spans 0
(upper bound +0.008), so no significant difference in a two-sided test at α=0.05. The
pre-registered rule tests **non-inferiority at 1pp**: lb=−0.050 < −0.01, so non-inferiority is
not shown. Both are reported; they answer different questions.

**P3-pool vs P2** (pool-controlled comparison — **exploratory**: the P2-vs-P3 rule is
pre-registered but the P3-pool arm is not; it was added after the first results, deviation #5):
Δresolved CI lb = −0.004 ≥ −0.01 AND Δ$/inst CI ub = −$0.077 < −$0.01.
Per the pre-registered "adds value" rule: **P3-pool adds value over P2** — it resolves more at
lower cost. This is the result after fixing `repo_router` to pick the cheapest eligible model
(not the first in list order) and fixing `fit_eps` to use the pool reference model (opus) as the
quality bar instead of the last pool model (minimax). See deviation #8 and #9.

**P3-ladder vs P2** (confounded comparison, required per auditors):
- haiku→opus P3-ladder: Δresolved CI spans 0, Δ$/inst CI entirely positive (more expensive). P3-ladder does not add value over P2.
- gpt-5-mini→5.2 P3-ladder: loses on quality (lb=−0.074) and more expensive. Does not add value.
- Note: this comparison confounds pool diversity with routing granularity. Use P3-pool vs P2 for the clean comparison.

---

## Prompt caching and mid-session switching (P4)

With provider prompt caching, a mid-session model switch forces the top model's first call to
re-read the accumulated context at full (uncached) input price. Under the calibrated context
model (cheap model only):

What a switch would cost if it happened after the cheap model's first step (the smallest real
switch, context ≈ I_0): about 2,328 tokens × $15/M ≈ $0.035/inst on haiku→opus and 1,349 tokens ×
$5/M ≈ $0.007/inst on gpt-5-mini→5.2. The cross-fit, which includes this overhead, still picks
tau=−1 (escalate before any cheap step). That is not a switch, so no overhead is charged and P4
equals always-top under both cost readings.

---

## Deviations from pre-registration

All changes made after e848709 (pre-registration commit), before publication.

| # | deviation | reason |
|---|---|---|
| 1 | "est. no caching" column removed | Mixed-model nc-factor scaling invalid (cascade: cost<$0.50 heuristic; per-session/oracle: one model's factor; escalate: FALLBACK_OUT). Column removed; handoff context estimate kept labelled as from the cheap-model calibration only. |
| 2 | P5 (run-level cascade) quality restored as identifiable | PR commits after e848709 labelled cascade quality "not identifiable." Incorrect: cascade() runs cheap to completion then top from scratch — whole recorded runs, no stitching. |
| 3 | P4 trigger fixed: `>= tau_pin` (was `> tau_pin`) | Pre-registration: "escalate when api_calls >= tau_pin." Initial code used `cheap_steps <= tau` → keep cheap, which triggered escalation at `>`, not `>=`. Fixed to `cheap_steps < tau`. For this dataset tau=−1 in both folds, so numerical results are unchanged. |
| 4 | Random arm (P6) implemented | Listed in the pre-registration but missing from the initial draft. Added seed=42. |
| 5 | P3-pool added (not in pre-registration) | Auditors noted the P3-ladder vs P2 comparison confounds pool diversity with routing granularity. Added P3-pool (per-session from the same 11-model pool) for a pool-controlled comparison. |
| 6 | `_traj_context` I_0 = system + first user message only | Initial code summed ALL user messages into I_0. Fixed to use system + first user message; subsequent messages count toward delta. |
| 7 | n_switched uses fold-specific fitted tau | Initial code used fitted_pin[0] for all instances. Fixed to use the tau fitted on each instance's own training fold. |
| 8 | `repo_router` picks cheapest eligible (not first in list) | Initial code picked the first model in POOL list order within eps, which is haiku/sonnet/opus (expensive). Fixed to pick `min(eligible, key=lambda k: avg_cost[k])`, matching the same selection rule as `select_cheapest` (P2). This materially changes P3-pool results: from 363/500 at $0.433/inst to 382/500 at $0.122/inst. P3-pool now beats P2 by the pre-registered rule. |
| 9 | `fit_eps` uses explicit `ref_idx` (was always `r["runs"][-1]`) | For the 11-model pool, `runs[-1]` is minimax (last in POOL list), not the reference model. Fixed by passing `POOL_REF_IDX=2` (claude-4-5-opus-high) to `fit_eps` for pool rows. The 2-model ladders are unchanged (ref_idx=-1 = top rung = correct). |
| 10 | `withcache_cost` prefix billing fixed (k-1, not k) | At step k the cached prefix is I_0+(k-1)×delta (tokens written before this step); the bug used I_0+k×delta (one step ahead). Affects the informational nc_factor only; observed leaderboard costs in the table are unchanged. |
| 11 | Calibration sample uses seeded random (not alphabetical first 20) | `ids[:CALIB_N]` sampled all astropy instances (alphabetically first). Fixed to `random.sample(ids, CALIB_N, seed=42)`. I_0 values changed (haiku: 2816→2328; gpt-5: 1720→1349); qualitative results unchanged. |
| 12 | P4 split into primary (pre-registration) and secondary (restart) | Pre-registration specifies switch_overhead in the cost model. The restart interpretation (no overhead since top model restarts from scratch) is also valid and reported as a secondary row. Audit comment: these are the same experiment viewed through two cost models; the pre-registration is the primary. |
| 13 | Trajectory cache writes use atomic tmp+replace; retry once on JSON decode failure | Hardening for concurrent / interrupted writes; no effect on results. |

---

## What this means

1. **P3-pool beats P2** at the pre-registered threshold — an **exploratory** result (the arm was
   added after the first results; its fitted eps differs by fold, [0.2, 0.05]; and every repo is
   seen in both folds, so a new repo gets no evidence). It is a hypothesis for a fresh,
   pre-registered run, not a shipped claim: per-session selection using the repo
   feature, when applied to the same 11-model pool and corrected to pick the cheapest eligible
   model, resolves more at lower cost than workload selection. The key driver is the repo
   feature identifying repos where cheap models are sufficient: the corrected router sends those
   to minimax/gemini-flash/deepseek at <$0.05/inst rather than paying the workload-selected
   single model for all 500.

2. **P3-ladder does not add value over P2** on either ladder: it is more expensive (2-model
   pool limits to cheap/top with no very cheap near-frontier options) and the repo feature is
   not strong enough to compensate.

3. **P4 (escalate-and-pin) gains nothing**: the cross-fitted trigger is tau=−1 on both ladders,
   i.e. escalate before the cheap model takes a step. Any later switch point wastes cheap spend
   without closing the resolve gap, so the best mid-session policy is "use the top model from
   the start" — always-top. Both cost readings agree at tau=−1.

4. **Run-level cascade (P5) costs more** with no quality gain. Negative result confirmed
   from #239 on both ladders.

5. **The oracle gap on the two-model ladder is large**: 395/500 at $0.384/inst vs always-top
   384/500 at $0.754/inst. (This oracle chooses between two models only, so it is not comparable
   with the 11-model pool arms.)

---

## Limitations

1. **Replay validity**: P4 and P5 are scored from whole recorded runs; true mid-run handoff
   quality is not measurable from this data.
2. **Single run per model per instance**: run-to-run variance not captured.
3. **Context model approximated** from 20 cheap-model trajectories. Top-model context not
   estimated. Overhead figures are estimates.
4. **No-cache costs not reported**: analytical model is invalid for mixed-model scenarios
   (deviation 1). Single-model nc_factor is shown as informational only.
5. **One scaffold, one benchmark**: mini-SWE-agent bash-only, SWE-bench Verified 500 instances.
6. **Approximate price table**: list prices for overhead illustration; enterprise discounts
   would reduce estimates.

---

## How to rerun

```bash
python3 -I scripts/analyze-agent-granularity.py ~/.cache/firstpass-swebench-experiments
python3 -I scripts/analyze-agent-granularity.py --self-check
```

Data fetched at runtime from `github.com/SWE-bench/experiments@40f164d` into cache.
Not committed (no data licence).
