# Routing granularity for multi-turn coding agents

date: 2026-11-01 · spend: $0 (public runs) · data: github.com/SWE-bench/experiments@40f164d ·
command: `python3 -I scripts/analyze-agent-granularity.py [CACHE_DIR]`

Pre-registration: [agent-granularity-preregistration.md](agent-granularity-preregistration.md)
(committed at e848709 before any result was computed; policies, metrics, and decision rules fixed
in advance). See **Deviations from pre-registration** below for changes made after review.

---

## Validity caveat

Whole-run policies (always-X, per-session, workload, run-level cascade, escalate-by-restart) are
scored exactly from recorded runs. Per-call cascade (P5) is scored from whole recorded runs (cheap
to completion, then top from scratch) — quality IS identifiable.

Escalate-by-restart (P4) post-switch quality ≈ top model's own full run on that instance
(restart semantics; optimistic for quality since the top model starts fresh, pessimistic for cost
since it runs to full length; see pre-registration §Validity caveat).

Context growth model calibrated from the cheap model's trajectories only. I_0 = system prompt +
first user message (initial problem statement). The handoff re-read illustration is an estimate
from this model.

---

## Models

Primary ladder: `claude-4-5-haiku-high` → `claude-4-5-opus-high` (v2.0.0, n=500)
Secondary ladder: `gpt-5-mini` → `gpt-5-2-high` (v2.0.0, n=500)
Pool for P2/P3-pool: 11 models from `20260217_mini-v2.0.0_*` (n=500)

---

## Context calibration (20 trajectory sample, cheap model only)

| model | I_0 (tokens) | delta (tokens/step) | out/step (tokens) | avg nc_factor |
|---|---|---|---|---|
| claude-4-5-haiku-high | 2,816 | 734 | 289 | 5.38× |
| gpt-5-mini | 1,720 | 842 | 245 | 3.45× |

nc_factor = estimated no-cache / estimated with-cache for single-model sessions.
Haiku: role-based message format, chars / 2.5 tokens. GPT-5-mini: OpenAI Responses API format
with authoritative per-step token counts. Context model calibrated on cheap model only; top model
parameters not estimated.

---

## Results

### Claude haiku→opus (v2.0.0)  (n=500)

Cross-fit per fold: cascade tau=[19, 34], repo-eps=[0.0, 0.0], escalate-by-restart tau=[-1, -1]

| arm | resolved | $/inst | $/resolved | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|
| always-cheap [P1] | 333/500 (0.666) | $0.331 | $0.497 | [-0.136, -0.072] | [-0.460, -0.387] |
| always-top [P1] | 384/500 (0.768) | $0.754 | $0.982 | — | — |
| run-level cascade (cross-fit) [P5] | 382/500 (0.764) | $1.071 | $1.401 | [-0.010, +0.000] | [+0.295, +0.340] |
| per-session / repo (cross-fit) [P3] | 378/500 (0.756) | $0.720 | $0.953 | [-0.024, -0.002] | [-0.048, -0.021] |
| escalate-by-restart (cross-fit) [P4] | 384/500 (0.768) | $0.754 | $0.982 | [+0.000, +0.000] | [+0.000, +0.000] |
| random (seed=42) [P6] | 359/500 (0.718) | $0.555 | $0.772 | [-0.074, -0.026] | [-0.232, -0.170] |
| oracle-per-instance (bound) [P6] | 395/500 (0.790) | $0.384 | $0.486 | [+0.010, +0.036] | [-0.408, -0.334] |
| oracle-cascade (bound) | 395/500 (0.790) | $0.679 | $0.859 | [+0.010, +0.036] | [-0.116, -0.032] |

Escalate-by-restart: 500/500 trigger (250/250 fold 0 tau=-1, 250/250 fold 1 tau=-1).
Handoff illustration (cost-only estimate): a mid-run HANDOFF at step 0 (tau=-1) would re-read
2,816 tokens at uncached rate = est. $0.042/inst (context model, cheap-model calibration only);
not charged to escalate-by-restart (the top model's recorded run already includes its own first read).

### GPT-5-mini→GPT-5.2-high (v2.0.0)  (n=500)

Cross-fit per fold: cascade tau=[6, 8], repo-eps=[0.05, 0.1], escalate-by-restart tau=[-1, -1]

| arm | resolved | $/inst | $/resolved | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|
| always-cheap [P1] | 281/500 (0.562) | $0.047 | $0.084 | [-0.206, -0.126] | [-0.459, -0.396] |
| always-top [P1] | 364/500 (0.728) | $0.474 | $0.651 | — | — |
| run-level cascade (cross-fit) [P5] | 364/500 (0.728) | $0.520 | $0.714 | [+0.000, +0.000] | [+0.043, +0.050] |
| per-session / repo (cross-fit) [P3] | 353/500 (0.706) | $0.416 | $0.589 | [-0.042, -0.002] | [-0.074, -0.042] |
| escalate-by-restart (cross-fit) [P4] | 364/500 (0.728) | $0.474 | $0.651 | [+0.000, +0.000] | [+0.000, +0.000] |
| random (seed=42) [P6] | 321/500 (0.642) | $0.284 | $0.442 | [-0.116, -0.056] | [-0.215, -0.166] |
| oracle-per-instance (bound) [P6] | 383/500 (0.766) | $0.154 | $0.201 | [+0.022, +0.056] | [-0.353, -0.289] |
| oracle-cascade (bound) | 383/500 (0.766) | $0.317 | $0.414 | [+0.022, +0.056] | [-0.182, -0.133] |

Escalate-by-restart: 500/500 trigger (250/250 each fold, tau=-1).
Handoff illustration: est. $0.009/inst (1,720 tokens uncached at $5/1M; cheap-model calibration).

### Pool policies (P2 and P3-pool)  (n=500)

P2 cross-fit: even half → minimax-2-5-high, odd half → gemini-3-flash-high.
P3-pool cross-fit: repo-eps=[0.0, 0.1].

| arm | resolved | $/inst | Δ resolved vs always-opus | Δ $/inst vs always-opus |
|---|---|---|---|---|
| pool-selection (P2, cross-fit) | 374/500 (0.748) | $0.219 | [-0.050, +0.008] | [-0.582, -0.490] |
| pool-session (P3-pool, cross-fit) | 363/500 (0.726) | $0.433 | [-0.070, -0.016] | [-0.368, -0.275] |
| always-claude-4-5-opus-high (pool ref) | 384/500 (0.768) | $0.754 | — | — |

**P3-pool vs P2 paired CI** (pool-controlled comparison, same 500 instances):
Δresolved = [-0.050, +0.004], Δ$/inst = [+0.182, +0.249]

### P3-ladder vs P2 cross-comparison (confounded: different pools)

| ladder | P3-ladder | P2 | Δresolved CI | Δ$/inst CI |
|---|---|---|---|---|
| haiku→opus | 378/500, $0.720/inst | 374/500, $0.219/inst | [-0.020, +0.036] | [+0.456, +0.550] |
| gpt-5-mini→5.2 | 353/500, $0.416/inst | 374/500, $0.219/inst | [-0.074, -0.008] | [+0.165, +0.230] |

Note: P3-ladder uses a 2-model ladder; P2 uses an 11-model pool. The comparison confounds
routing granularity with pool diversity. Use the P3-pool vs P2 comparison above for a
pool-controlled test of granularity value.

---

## Verdicts (per pre-registered decision rules)

**Decision rule**: A policy "beats" always-top if BOTH: (1) Δresolved CI lower bound ≥ −0.01
AND (2) Δ$/inst CI upper bound < −0.01. A policy "loses on quality" if lb < −0.01.

| policy | Δresolved CI lb | Δ$/inst CI ub | verdict |
|---|---|---|---|
| P2: workload selection (pool) | −0.050 | −$0.490 | **loses on quality** — lb < −0.01, so non-inferiority at 1pp tolerance is NOT shown; cost decisively better |
| P3-pool: pool-session vs always-opus | −0.070 | −$0.275 | **loses on quality** — lb < −0.01 |
| P3: per-session/repo, haiku→opus | −0.024 | −$0.021 | **loses on quality** — lb < −0.01 |
| P3: per-session/repo, gpt-5-mini→5.2 | −0.042 | −$0.042 | **loses on quality** — lb < −0.01 |
| P4: escalate-by-restart, haiku→opus | +0.000 | +$0.000 | **no gain: equals always-top exactly** — cross-fit chose tau=−1 (always escalate) in both folds |
| P4: escalate-by-restart, gpt-5-mini→5.2 | +0.000 | +$0.000 | **no gain: equals always-top exactly** |
| P5: cascade, haiku→opus | −0.010 | +$0.340 | **negative** — significantly more expensive; quality borderline (lb=−0.010, at the −0.01 threshold) |
| P5: cascade, gpt-5-mini→5.2 | +0.000 | +$0.050 | **negative** — more expensive, same quality |

**P2 quality note**: The #239 "not significant" framing used a two-sided test of whether there
is a measurable quality difference; the CI spans 0 (upper bound +0.008), so a two-sided test at
α=0.05 does not find a significant difference. The pre-registered decision rule here tests
**non-inferiority at 1pp**: is the lower CI bound ≥ −0.01? With lb=−0.050, it is not, so the
verdict is "loses on quality" even though the point estimate and upper CI are close to zero.
These are different statistical questions; both are reported.

**P3-pool vs P2** (pre-registered P2-vs-P3 comparison, pool-controlled):
Δresolved CI lb = −0.050 < 0 (not strictly better quality) and Δ$/inst CI ub = +$0.249 > 0
(P3-pool is more expensive). Per pre-registered rule: P3-pool does NOT add value over P2.
Per-session selection from the same 11-model pool, using the repo feature, costs 2× more than
workload selection with no quality benefit.

**P3-ladder vs P2** (confounded comparison, noted as required by auditors):
- haiku→opus P3-ladder: Δ$/inst CI = [+0.456, +0.550] (more expensive), Δresolved CI spans 0.
  P3-ladder does not add value over P2.
- gpt-5-mini→5.2 P3-ladder: loses on quality (lb=−0.074) and more expensive.
  P3-ladder does not add value over P2.

---

## Prompt caching and mid-run switching (illustration only)

A mid-run handoff makes the top model re-read the whole accumulated context without a cache hit
(each model has its own KV cache). Under the calibrated context model (cheap model only), the
cost of that first read at step k is estimated as (I_0 + k·delta) × price_top_in / 1M:
- haiku→opus: $0.042 at step 0 (2,816 tokens × $15/1M), growing to $0.255 at step 19
  (2,816 + 19×734 = 16,762 tokens × $15/1M)
- gpt-5-mini→5.2: $0.009 at step 0, growing to $0.045 at step 6 (6,772 tokens × $5/1M)

This is a cost illustration, not a quality result. The cross-fit chose tau=−1 for escalate-by-restart
on both ladders, meaning the optimal policy under these conditions is immediate escalation
(= always-top). Escalate-by-restart does not incur this overhead since the top model's recorded
run already includes its own first read.

---

## Deviations from pre-registration

Committed after the initial script run (e848709) and before publication.

| # | deviation | reason |
|---|---|---|
| 1 | "est. no caching" column removed | Mixed-model nc-factor scaling invalid: cascade selects factor by cost<$0.50 heuristic; per-session and oracle use one model's factor; escalate uses FALLBACK_OUT=219 vs calibrated value. Column dropped entirely. Handoff re-read kept as a point estimate labelled "cheap-model calibration only." |
| 2 | P5 (run-level cascade) quality restored as identifiable | PR commits after e848709 labelled cascade quality "not identifiable." This was wrong: cascade() runs cheap to completion then top from scratch — both are whole recorded runs with no stitching. Label removed; resolved counts restored as measured. |
| 3 | P4 trigger fixed: `>= tau_pin` (was `> tau_pin`) | Pre-registration says "escalate when api_calls >= tau_pin." Initial code used `cheap_steps <= tau` → keep cheap, which triggers escalation at `>` not `>=`. Fixed to `cheap_steps < tau`. For this dataset tau=−1 in both folds, so numerical results are unchanged. |
| 4 | Random arm (P6) implemented | Was listed in the pre-registration but missing from the initial implementation. Added with seed=42. |
| 5 | P3-pool added (not in pre-registration) | Auditors noted the P3-ladder vs P2 comparison confounds pool diversity with routing granularity. Added P3-pool (per-session from same 11-model pool) to provide a pool-controlled comparison. |
| 6 | _traj_context I_0 uses system + first user message only | Initial code summed ALL user messages into I_0. Fixed to use system + first user message; subsequent messages count toward delta. I_0 changed from 2,873 to 2,816 for haiku; results qualitatively unchanged. |
| 7 | n_switched uses fold-specific fitted tau | Initial code used fitted_pin[0] for all instances. Fixed to use the tau fitted on each instance's own training fold. |

---

## What this means

1. **No routing granularity finer than workload level adds value on these runs.** Per-session,
   mid-session, and per-call policies all either cost more, lose quality, or both vs their
   respective baselines.

2. **Workload selection (P2) shows a decisive cost signal** (-$0.490/inst vs always-top) with
   borderline quality under the pre-registered non-inferiority criterion. The #239 finding
   (−71% cost, n.s. quality) is reproduced; the verdict difference is the choice between a
   non-inferiority framing (this study) and a two-sided difference framing (#239).

3. **Escalate-by-restart degenerates to always-top** on both ladders: the cross-fit finds
   immediate escalation optimal, because no step-count threshold identifies cheap-sufficient
   instances while holding quality at always-top. The policy equals always-top exactly.

4. **Run-level cascade (P5) costs more** than always-top on both ladders (confirmed negative
   result from #239). It adds the top model's full run cost on top of the cheap model's run
   with no net quality gain.

5. **Per-session routing (P3) loses quality** on both ladders and both pools. The repo feature
   is insufficient to route instances reliably; richer features (issue text, complexity signals)
   would be needed.

6. **The oracle gap is large**: oracle-per-instance achieves 395/500 at $0.384/inst vs
   always-top 384/500 at $0.754/inst. The gap (2× cheaper, 2.9pp better quality) measures the
   theoretical value of a perfect instance-level router — beyond what any cross-fitted policy on
   available features achieves here.

---

## Limitations

1. **Replay validity**: escalate-by-restart and cascade are scored from whole recorded runs;
   true mid-run handoff quality is not identifiable from this data.
2. **Single run per model per instance**: run-to-run variance not captured.
3. **Context model approximation**: I_0, delta estimated from 20 cheap-model trajectories.
   Top-model parameters not estimated. Handoff cost illustration is approximate.
4. **No-cache costs not reported**: analytical model is invalid for mixed-model scenarios
   (see deviation 1). Single-model nc_factor is printed as informational.
5. **One scaffold, one benchmark**: mini-SWE-agent bash-only, SWE-bench Verified 500 instances.
6. **Approximate price table**: list prices used for handoff illustration; enterprise discounts
   would reduce overhead estimates.

---

## How to rerun

```bash
python3 -I scripts/analyze-agent-granularity.py ~/.cache/firstpass-swebench-experiments
# delete *.ctx.*.json in cache dir to force trajectory re-calibration
python3 -I scripts/analyze-agent-granularity.py --self-check
```

Data fetched at runtime from `github.com/SWE-bench/experiments@40f164d` into cache.
Not committed (no licence on the data).
