# Routing granularity for multi-turn coding agents

date: 2026-11-01 · spend: $0 (public runs) · data: github.com/SWE-bench/experiments@40f164d ·
command: `python3 -I scripts/analyze-agent-granularity.py [CACHE_DIR]`

Pre-registration: [agent-granularity-preregistration.md](agent-granularity-preregistration.md)
(committed at e848709 before any result was computed). See **Deviations from pre-registration**
for all post-preregistration changes to script and interpretation.

---

## Validity caveat

Whole-run policies (always-X, per-session, workload) are scored exactly from recorded runs.
Per-call cascade (P5) as pre-registered is **not identifiable** from this data: the leaderboard
records only whole runs, not per-step gate verdicts or counterfactual continuations. The replayable
proxy is a run-level cascade (cheap to completion, then decide); this is what the script measures.
See Deviations #2 and #15.

Escalate-and-pin (P4) post-switch quality ≈ top model's own full run (restart semantics;
optimistic for quality since the top model starts fresh; see pre-registration validity caveat).

**Two P4 rows** are reported:
- **Primary (pre-registration)**: cost = partial_cheap + switch_overhead + top_leaderboard_cost,
  per the pre-registration cost model. The top model's first call re-reads the accumulated
  prefix at the full uncached input rate.
- **Secondary (restart-semantics)**: cost = partial_cheap + top_leaderboard_cost. No separate
  overhead because the top model restarts from scratch and its full recorded cost already
  includes its own first read. Listed as a deviation below.

Context model calibrated from CALIB_N=20 seeded-random (seed=42) cheap-model trajectories;
I_0 = system + first user message only.

**tau_pin >= 1**: the pre-registration requires at least one cheap step before any switch;
tau_pin = -1 is not a valid grid value. See Deviation #14.

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

nc_factor = estimated no-cache / estimated with-cache for single-model sessions (informational).
Context model calibrated on cheap model only; top-model parameters not estimated.

---

## Results

### Claude haiku→opus (v2.0.0)  (n=500)

Cross-fit per fold: cascade tau=[19, 34], repo-eps=[0.0, 0.0], P4 tau_pin=[19, 17] (both variants)

| arm | resolved | $/inst | $/resolved | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|
| always-cheap [P1] | 333/500 (0.666) | $0.331 | $0.497 | [-0.136, -0.072] | [-0.460, -0.387] |
| always-top [P1] | 384/500 (0.768) | $0.754 | $0.982 | — | — |
| run-level cascade, proxy for P5 (cross-fit) | 382/500 (0.764) | $1.071 | $1.401 | [-0.010, +0.000] | [+0.295, +0.340] |
| per-session / repo (cross-fit) [P3] | 378/500 (0.756) | $0.720 | $0.953 | [-0.024, -0.002] | [-0.048, -0.021] |
| escalate-and-pin prereg [P4] | 384/500 (0.768) | $1.066 | $1.389 | [+0.000, +0.000] | [+0.309, +0.316] |
| escalate-and-pin restart (secondary) | 384/500 (0.768) | $0.838 | $1.091 | [+0.000, +0.000] | [+0.081, +0.086] |
| random (seed=42) [P6] | 359/500 (0.718) | $0.555 | $0.772 | [-0.074, -0.026] | [-0.232, -0.170] |
| oracle-per-instance (bound) [P6] | 395/500 (0.790) | $0.384 | $0.486 | [+0.010, +0.036] | [-0.408, -0.334] |
| oracle-cascade (bound) | 395/500 (0.790) | $0.679 | $0.859 | [+0.010, +0.036] | [-0.116, -0.032] |

P4 escalation: 248/250 (fold 0, tau_pin=19) + 250/250 (fold 1, tau_pin=17) = 498/500 trigger.
Context at switch: 16,046 tokens (fold 0, I_0 + 19×delta) / 14,602 tokens (fold 1, I_0 + 17×delta).
Switch overhead (prereg arm): ~$0.241/inst (fold 0) / ~$0.219/inst (fold 1) at $15/1M.
Restart arm: no overhead; pays partial cheap only (19 or 17 steps / full cheap run).

### GPT-5-mini→GPT-5.2-high (v2.0.0)  (n=500)

Cross-fit per fold: cascade tau=[6, 8], repo-eps=[0.05, 0.1], P4 tau_pin=[6, 8] (both variants)

| arm | resolved | $/inst | $/resolved | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|
| always-cheap [P1] | 281/500 (0.562) | $0.047 | $0.084 | [-0.206, -0.126] | [-0.459, -0.396] |
| always-top [P1] | 364/500 (0.728) | $0.474 | $0.651 | — | — |
| run-level cascade, proxy for P5 (cross-fit) | 364/500 (0.728) | $0.520 | $0.714 | [+0.000, +0.000] | [+0.043, +0.050] |
| per-session / repo (cross-fit) [P3] | 353/500 (0.706) | $0.416 | $0.589 | [-0.042, -0.002] | [-0.074, -0.042] |
| escalate-and-pin prereg [P4] | 364/500 (0.728) | $0.527 | $0.724 | [+0.000, +0.000] | [+0.052, +0.054] |
| escalate-and-pin restart (secondary) | 364/500 (0.728) | $0.488 | $0.671 | [+0.000, +0.000] | [+0.014, +0.015] |
| random (seed=42) [P6] | 321/500 (0.642) | $0.284 | $0.442 | [-0.116, -0.056] | [-0.215, -0.166] |
| oracle-per-instance (bound) [P6] | 383/500 (0.766) | $0.154 | $0.201 | [+0.022, +0.056] | [-0.353, -0.289] |
| oracle-cascade (bound) | 383/500 (0.766) | $0.317 | $0.414 | [+0.022, +0.056] | [-0.182, -0.133] |

P4 escalation: 250/250 (fold 0, tau_pin=6) + 249/250 (fold 1, tau_pin=8) = 499/500 trigger.
Context at switch: 6,779 tokens (fold 0) / 8,589 tokens (fold 1).
Switch overhead (prereg arm): ~$0.034/inst (fold 0) / ~$0.043/inst (fold 1) at $5/1M.

### Pool policies  (n=500)

P2: even half → minimax-2-5-high, odd half → gemini-3-flash-high (replicates #239).
P3-pool: fitted eps per fold = [0.2, 0.05]; repo_router picks cheapest within eps of best rate.

| arm | resolved | $/inst | Δ resolved vs always-opus | Δ $/inst vs always-opus |
|---|---|---|---|---|
| pool-selection (P2, cross-fit) | 374/500 (0.748) | $0.219 | [-0.050, +0.008] | [-0.582, -0.490] |
| pool-session (P3-pool, cross-fit) | 382/500 (0.764) | $0.122 | [-0.032, +0.024] | [-0.679, -0.585] |
| always-claude-4-5-opus-high (pool ref) | 384/500 (0.768) | $0.754 | — | — |

**P3-pool vs P2 paired CI** (same 500 instances):
Δresolved = [-0.004, +0.036], Δ$/inst = [-0.116, -0.077]

### P3-ladder vs P2 cross-comparison (confounded: 2-model ladder vs 11-model pool)

| ladder | P3-ladder | P2 | Δresolved CI | Δ$/inst CI |
|---|---|---|---|---|
| haiku→opus | 378/500, $0.720/inst | 374/500, $0.219/inst | [-0.020, +0.036] | [+0.456, +0.550] |
| gpt-5-mini→5.2 | 353/500, $0.416/inst | 374/500, $0.219/inst | [-0.074, -0.008] | [+0.165, +0.230] |

---

## Verdicts (per pre-registered decision rules)

**"Beats" always-top** (prereg §Decision rules lines 191-193): BOTH Δresolved CI lb ≥ −0.01
AND Δ$/inst CI ub < −0.01. "Loses on quality": lb < −0.01. Otherwise "inconclusive."

| policy | Δresolved CI lb | Δ$/inst CI ub | verdict |
|---|---|---|---|
| P2: workload selection (pool) | −0.050 | −$0.490 | **loses on quality** — lb < −0.01; quality non-inferiority at 1pp not shown; cost decisively better |
| P3-pool vs always-opus | −0.032 | −$0.585 | **loses on quality** vs always-opus (lb=−0.032 < −0.01) |
| P3: per-session/repo, haiku→opus | −0.024 | −$0.021 | **loses on quality** (lb=−0.024 < −0.01) |
| P3: per-session/repo, gpt-5-mini→5.2 | −0.042 | −$0.042 | **loses on quality** (lb=−0.042 < −0.01) |
| P4 prereg: haiku→opus | +0.000 | +$0.316 | **costs significantly more than always-top** — same quality, switch overhead ~$0.230/inst at tau_pin=19 steps |
| P4 prereg: gpt-5-mini→5.2 | +0.000 | +$0.054 | **costs more than always-top** — same quality, switch overhead ~$0.038/inst at tau_pin=6–8 steps |
| P4 restart (secondary): haiku→opus | +0.000 | +$0.086 | **costs more than always-top** — partial cheap cost for 19 steps adds up |
| P4 restart (secondary): gpt-5-mini→5.2 | +0.000 | +$0.015 | **costs more than always-top** — partial cheap cost for 6–8 steps |
| run-level cascade (proxy for P5): haiku→opus | −0.010 | +$0.340 | **confirmed negative proxy** (Δresolved lb=−0.010 < −0.01; also more expensive) |
| run-level cascade (proxy for P5): gpt-5-mini→5.2 | +0.000 | +$0.050 | **confirmed negative proxy** (Δ$/inst ub=+0.050 ≥ 0) |

**P2 quality note**: The #239 "n.s. quality loss" framing uses a two-sided test; the CI spans 0
(upper bound +0.008). The pre-registered rule tests non-inferiority at 1pp: lb=−0.050 < −0.01,
so non-inferiority is not shown. Both framings are noted; they answer different questions.

**Run-level cascade (proxy for P5)**: the pre-registered P5 is per-call cascade, which is not
identifiable from this data (no per-step gate results). The confirmed-negative rule (prereg
lines 206-208: lb < −0.01 OR Δ$/inst ub ≥ 0) is applied to the run-level proxy. Both ladders
satisfy at least one condition. The per-call negative result is prior evidence from #239, cited
below.

**P4 findings**: under the pre-registration cost model (tau_pin >= 1), switching overhead at
tau_pin=19 (haiku) accumulates to ~$0.230/inst against a top model that costs $0.754/inst base.
The fitted tau_pin equals the cascade threshold because both optimisers share the same quality
bar. Under restart semantics, partial cheap cost at 19 steps is substantial (~$0.08/inst extra).
Neither variant reduces cost relative to always-top.

**P3-pool vs P2** — pre-registered "adds value" rule (prereg lines 199-201):
per-session adds value over per-workload if Δresolved CI lb ≥ 0 (strictly better quality) OR
(CI contains 0 AND Δ$/inst CI ub < 0).

With Δresolved CI [−0.004, +0.036]: lb=−0.004 < 0, ub=+0.036 > 0 → CI contains 0.
With Δ$/inst CI [−0.116, −0.077]: ub=−0.077 < 0 → strictly cheaper.
Second clause satisfied → **P3-pool adds value over P2.**

**P3-ladder vs P2** (confounded comparison, required per auditors):
- haiku→opus P3-ladder: CI spans 0 on resolved, Δ$/inst CI entirely positive (more expensive). Does not add value over P2.
- gpt-5-mini→5.2 P3-ladder: loses on quality (lb=−0.074) and more expensive. Does not add value.
- Note: this comparison confounds pool diversity with routing granularity. Use P3-pool vs P2 for the clean comparison.

---

## Prompt caching and mid-session switching (P4)

With provider prompt caching, a mid-session model switch forces the top model's first call to
re-read the accumulated context at full (uncached) input price. Under the calibrated context
model (cheap model only):

| ladder | tau_pin (fold 0 / fold 1) | context at switch | switch overhead (approx) | overhead as % of always-top |
|---|---|---|---|---|
| haiku→opus | 19 / 17 | 16,046 / 14,602 tokens | ~$0.241 / ~$0.219 /inst | ~31% / ~29% |
| gpt-5-mini→5.2 | 6 / 8 | 6,779 / 8,589 tokens | ~$0.034 / ~$0.043 /inst | ~7% / ~9% |

The fitted tau_pin equals the cascade threshold in both folds. Both optimisers share the same
quality bar (resolved count >= always-top on training half), and both maximise quality first,
then minimise cost. Any smaller tau_pin is infeasible because fast-finishing cheap instances
that serve cheap don't resolve enough to meet the bar.

**Conclusion**: under the pre-registration cost model, escalate-and-pin is materially more
expensive than always-top on this data. The overhead at tau_pin=19 steps (haiku) is ~30% of
always-top's cost per instance, and the restart variant also costs more due to partial cheap
costs at 17-19 steps. Neither variant saves cost, and both match always-top's quality exactly.

---

## Deviations from pre-registration

All changes made after e848709 (pre-registration commit), before publication.

| # | deviation | reason |
|---|---|---|
| 1 | "est. no caching" column removed | Mixed-model nc-factor scaling invalid. Column removed; handoff context estimate kept labelled as from the cheap-model calibration only. |
| 2 | Run-level cascade implemented as proxy for P5 | The pre-registration defines P5 as per-call cascade (gate each step and re-run on failure). Per-call gating is not identifiable from recorded whole-runs: there are no per-step gate verdicts and no counterfactual continuation data. The run-level variant (run cheap to completion, then decide from final api_calls) is the closest replayable proxy. The arm is labelled "run-level cascade, proxy for P5." The pre-registered per-call cascade negative result is prior evidence from #239 (not claimed as a result of this run). |
| 3 | P4 trigger fixed: `>= tau_pin` (was `> tau_pin`) | Fixed to `cheap_steps < tau` (keep cheap) → escalate when `cheap_steps >= tau`. For this dataset numerical results unchanged (tau_pin > 0 always). |
| 4 | Random arm (P6) implemented | Listed in the pre-registration but missing from the initial draft. Added seed=42. |
| 5 | P3-pool added (not in pre-registration) | Auditors noted the P3-ladder vs P2 comparison confounds pool diversity with routing granularity. Added P3-pool (per-session from the same 11-model pool) for a pool-controlled comparison. |
| 6 | `_traj_context` I_0 = system + first user message only | Initial code summed ALL user messages into I_0. Fixed to use system + first user message; subsequent messages count toward delta. |
| 7 | n_switched uses fold-specific fitted tau | Initial code used fitted_pin[0] for all instances. Fixed to use the tau fitted on each instance's own training fold. |
| 8 | `repo_router` picks cheapest eligible (not first in list) | Initial code picked the first model in POOL list order within eps (haiku/sonnet/opus). Fixed to pick `min(eligible, key=avg_cost)`. Materially changes P3-pool: 363/500 $0.433/inst → 382/500 $0.122/inst; verdict reversed. |
| 9 | `fit_eps` uses explicit `ref_idx` (was always `r["runs"][-1]`) | For the 11-model pool, `runs[-1]` is minimax (last in POOL), not the reference model. Fixed by passing `POOL_REF_IDX=2` (claude-4-5-opus-high) to `fit_eps` for pool rows. |
| 10 | `withcache_cost` prefix billing fixed (k-1, not k) | At step k the cached prefix is I_0+(k-1)×delta; the bug used I_0+k×delta. Affects the informational nc_factor only; observed leaderboard costs unchanged. |
| 11 | Calibration sample uses seeded random (not alphabetical first 20) | `ids[:CALIB_N]` sampled all astropy instances. Fixed to `random.sample(ids, CALIB_N, seed=42)`. I_0 values corrected (haiku: 2816→2328; gpt-5: 1720→1349); qualitative results unchanged. |
| 12 | P4 split into primary (pre-registration) and secondary (restart) | Pre-registration specifies switch_overhead in the cost model. The restart interpretation (no overhead since top model restarts from scratch) is also valid and reported as a secondary row. |
| 13 | Trajectory cache writes use atomic tmp+replace; retry once on decode failure | Hardening for concurrent/interrupted writes; no effect on results. |
| 14 | tau_pin grid restricted to >= 1 (removed -1) | Raised in review by the maintainer, not by an automated auditor. An earlier draft of this analysis allowed tau=−1, which is always-top rather than escalate-and-pin; always-top remains in the tables as the reference row. The pre-registration states "Start every session with the cheap model" and switch when `api_calls >= tau_pin`, so tau_pin must be >= 1. The previous grid included -1 as a special value (always-top, no cheap step). Removing -1 changes P4 results materially: fitted tau_pin is now 19/17 (haiku) and 6/8 (gpt-5), matching the cascade threshold. P4 prereg now costs +$0.31/inst (haiku) and +$0.05/inst (gpt-5) more than always-top, not marginally more. The `actual_tau == 0` branch in `escalate_pin_prereg` is now a guarded assertion. |
| 15 | P5 arm renamed to "run-level cascade, proxy for P5" | Follows from deviation #2. The table row label now states plainly that this is a run-level proxy, not the pre-registered per-call granularity. |
| 16 | `_atomic_write` uses buffered write | `os.write` may write partially on large data; replaced with `with open(fd, "wb") as f: f.write(); f.flush(); os.fsync()`. No effect on results. |
| 17 | `json.loads(path.read_bytes())` everywhere; catch ValueError | Locale-independent; ValueError covers UnicodeDecodeError in addition to JSONDecodeError. No effect on results. |

---

## What this means

1. **P3-pool adds value over P2** at the pre-registered threshold: per-session selection using the
   repo feature, when applied to the same 11-model pool and corrected to pick the cheapest eligible
   model, resolves more at lower cost. The key driver is the repo feature routing instances from
   easy repos to cheap near-frontier models at <$0.05/inst.

2. **P3-ladder does not add value over P2** on either ladder: the 2-model pool limits choices to
   cheap/top, and the repo feature alone is insufficient to recover P2's gain from pool diversity.

3. **P4 (escalate-and-pin) costs more than always-top** under both cost models. The fitted
   tau_pin equals the cascade threshold (19 steps for haiku, 6-8 for gpt-5) because both
   optimisers share the same quality bar. At 19 steps, the accumulated context is ~16,000
   tokens; the switch overhead at $15/1M input = ~$0.23/inst (30% of always-top's cost).
   Under restart semantics, partial cheap cost at 19 steps is also substantial. Neither variant
   saves cost; both match always-top quality.

4. **Run-level cascade (proxy for P5) costs more** with no quality gain on both ladders.
   Negative result confirmed as replayable proxy. Per-call cascade negative result is prior
   evidence from #239.

5. **The oracle gap on the two-model ladder is large**: 395/500 at $0.384/inst vs always-top
   384/500 at $0.754/inst. (That oracle chooses between two models only, so it is not comparable
   with the 11-model pool arms such as P3-pool.)

---

## Limitations

1. **Replay validity**: P4 and proxy-P5 are scored from whole recorded runs; true mid-run handoff
   quality and per-step gate verdicts are not measurable from this data.
2. **Single run per model per instance**: run-to-run variance not captured.
3. **Context model approximated** from 20 cheap-model trajectories. Top-model context not
   estimated. Overhead figures are estimates.
4. **No-cache costs not reported**: analytical model is invalid for mixed-model scenarios.
   Single-model nc_factor is shown as informational only.
5. **One scaffold, one benchmark**: mini-SWE-agent bash-only, SWE-bench Verified 500 instances.
6. **Approximate price table**: list prices for overhead illustration; enterprise discounts
   would reduce estimates.
7. **Per-repo routing (P3-pool) needs per-repo history**: an unseen repo falls back to the global
   rate and gets no per-repo benefit. Every repo in this dataset appears in both training and test
   folds; out-of-distribution repos are not tested.

---

## How to rerun

```bash
python3 -I scripts/analyze-agent-granularity.py ~/.cache/firstpass-swebench-experiments
python3 -I scripts/analyze-agent-granularity.py --self-check
```

Data fetched at runtime from `github.com/SWE-bench/experiments@40f164d` into cache.
Not committed (no data licence).
