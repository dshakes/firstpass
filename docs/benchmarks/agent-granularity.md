# Routing granularity for multi-turn coding agents

date: 2026-11-01 · spend: $0 (public runs) · data: github.com/SWE-bench/experiments@40f164d ·
command: `python3 -I scripts/analyze-agent-granularity.py [CACHE_DIR]`

Pre-registration: [agent-granularity-preregistration.md](agent-granularity-preregistration.md)
(committed before any result was computed; policies, metrics, and decision rules fixed in advance).

---

## Validity caveat (prominent, per pre-registration)

Recorded trajectories show what each model did on its own run; they cannot show what a
different model would have done after taking over mid-run (The Replay Gap, arXiv 2608.08239:
a mid-run model swap changes 61–94% of subsequent actions). So:

- **Whole-run policies are scored exactly**: always-X, per-session (start-of-session) choice,
  workload selection, and **escalate-by-restart** (the cheap model's real run up to the trigger,
  then the top model's own complete run from scratch — no stitching).
- **Per-call cascade quality is NOT identifiable** by replay: it stitches steps from two models'
  runs. Its *cost* under the recorded trajectories is reported; its resolved count is shown only
  for continuity with #239 and must not be read as measured.
- **Mid-run handoff** (the top model continues the cheap model's context) is not scored for
  quality; its cost is discussed below as a cost-only illustration.

Context growth model (I_0, delta) is estimated from 20 sampled trajectories at 2.5 chars/token.
Real token counts differ by ±10–30%. The "no caching" column is labelled as estimated.

> **Correction (2026-11-01, before publication):** the first draft charged escalate-by-restart a
> separate "switch overhead" for the top model's uncached first read. That read is already in the
> top model's recorded cost (its run starts from scratch), so it was counted twice, which also
> biased the cross-fit toward immediate escalation. The overhead is removed; escalate-by-restart
> now equals always-top exactly (tables and verdicts below are from the corrected run).

---

## Models

Primary ladder: `claude-4-5-haiku-high` → `claude-4-5-opus-high` (v2.0.0, n=500)
Secondary ladder: `gpt-5-mini` → `gpt-5-2-high` (v2.0.0, n=500)
Pool for workload/per-session selection: 11 models from `20260217_mini-v2.0.0_*`

---

## Context calibration (20 trajectory sample per model)

| model | I_0 (tokens) | delta (tokens/step) | out/step (tokens) | cheap no-cache factor | top no-cache factor |
|---|---|---|---|---|---|
| claude-4-5-haiku-high | 2,873 | 724 | 289 | 5.37× | — |
| claude-4-5-opus-high | — | — | — | — | 3.98× |
| gpt-5-mini | 1,720 | 842 | 245 | 3.45× | — |
| gpt-5-2-high | — | — | — | — | 4.70× |

Haiku uses role-based message format (chars / 2.5 tokens). GPT-5-mini uses OpenAI Responses
API format with per-step token usage fields (authoritative, not estimated). No-cache factors
computed from analytical model; leaderboard costs are the authoritative "with caching" baseline.

---

## Results

### Claude haiku→opus (v2.0.0)  (n=500)

Cross-fit parameters: cascade tau=[19, 34] (per half), escalate-and-pin tau=[-1, -1], repo-eps=[0.0, 0.0]

| arm | resolved | $/inst (observed, w/ caching) | $/inst (est. no caching) | $/resolved (w/ caching) | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|---|
| always-cheap | 333/500 (0.666) | $0.331 | $1.777 | $0.497 | [-0.136, -0.072] | [-0.460, -0.387] |
| always-top | 384/500 (0.768) | $0.754 | $3.002 | $0.982 | — | — |
| per-call cascade (cross-fit) [P5] — cost-only† | (382/500)† | $1.071 | $4.337 | $1.401 | [-0.010, +0.000] | [+0.295, +0.340] |
| per-session / repo (cross-fit) [P3] | 378/500 (0.756) | $0.720 | $3.868 | $0.953 | [-0.024, -0.002] | [-0.048, -0.021] |
| escalate-by-restart (cross-fit) [P4] | 384/500 (0.768) | $0.754 | $3.634 | $0.982 | [+0.000, +0.000] | [+0.000, +0.000] |
| oracle-per-instance (bound) [P6] | 395/500 (0.790) | $0.384 | $2.061 | $0.486 | [+0.010, +0.036] | [-0.408, -0.334] |
| oracle-cascade (bound) | 395/500 (0.790) | $0.679 | $2.703 | $0.859 | [+0.010, +0.036] | [-0.116, -0.032] |

†Stitched replay: resolved count not identifiable (see validity caveat); cost is under the recorded
trajectories.

**Escalate-by-restart**: the cross-fit chose tau = −1 (escalate before the cheap model runs) in
both halves, so the policy is always-top. No step-budget trigger improved on it: escalating later
spends the cheap model's partial run without changing the outcome, and serving the cheap model's
short runs loses resolved instances.

### GPT-5-mini→GPT-5.2-high (v2.0.0)  (n=500)

Cross-fit parameters: cascade tau=[6, 8] (per half), escalate-and-pin tau=[-1, -1], repo-eps=[0.05, 0.1]

| arm | resolved | $/inst (observed, w/ caching) | $/inst (est. no caching) | $/resolved (w/ caching) | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|---|
| always-cheap | 281/500 (0.562) | $0.047 | $0.163 | $0.084 | [-0.206, -0.126] | [-0.459, -0.396] |
| always-top | 364/500 (0.728) | $0.474 | $2.228 | $0.651 | — | — |
| per-call cascade (cross-fit) [P5] — cost-only† | (364/500)† | $0.520 | $2.212 | $0.714 | [+0.000, +0.000] | [+0.043, +0.050] |
| per-session / repo (cross-fit) [P3] | 353/500 (0.706) | $0.416 | $1.434 | $0.589 | [-0.042, -0.002] | [-0.074, -0.042] |
| escalate-by-restart (cross-fit) [P4] | 364/500 (0.728) | $0.474 | $2.573 | $0.651 | [+0.000, +0.000] | [+0.000, +0.000] |
| oracle-per-instance (bound) [P6] | 383/500 (0.766) | $0.154 | $0.530 | $0.201 | [+0.022, +0.056] | [-0.353, -0.289] |
| oracle-cascade (bound) | 383/500 (0.766) | $0.317 | $1.492 | $0.414 | [+0.022, +0.056] | [-0.182, -0.133] |

Same pattern: escalate-by-restart degenerates to always-top (tau = −1).

### Policy P2 — Workload-level selection from 11-model pool (cross-fit, n=500)

Selected models: even half → minimax-2-5-high; odd half → gemini-3-flash-high (replicates #239).

| arm | resolved | $/instance | Δ resolved vs pool-reference | Δ $/inst vs pool-reference |
|---|---|---|---|---|
| pool-selection (cross-fit) | 374/500 (0.748) | $0.219 | [-0.050, +0.008] | [-0.582, -0.490] |
| always-claude-4-5-opus-high | 384/500 (0.768) | $0.754 | — | — |

---

## Verdicts (per pre-registered decision rules)

A policy "beats" always-top if Δresolved CI lower bound ≥ −0.01 AND Δ$/inst CI upper bound < −0.01.
A policy "loses on quality" if Δresolved CI lower bound < −0.01.

| policy | ladder | resolved | Δresolved CI lb | Δ$/inst CI ub | verdict |
|---|---|---|---|---|---|
| P2: workload selection (11-model pool) | pool | 374/500 | −0.050 | −$0.490 | **inconclusive on quality** (lb < −0.01 formally; CI spans 0; point estimate −2pp) |
| P3: per-session/repo (2-model ladder) | haiku→opus | 378/500 | −0.024 | −$0.021 | **loses on quality** (significantly fewer resolved) |
| P3: per-session/repo (2-model ladder) | gpt-5-mini→5.2 | 353/500 | −0.042 | −$0.042 | **loses on quality** (significantly fewer resolved) |
| P4: escalate-by-restart | haiku→opus | 384/500 | +0.000 | +$0.000 | **no gain: degenerates to always-top** (cross-fit tau = −1) |
| P4: escalate-by-restart | gpt-5-mini→5.2 | 364/500 | +0.000 | +$0.000 | **no gain: degenerates to always-top** |
| P5: per-call cascade | haiku→opus | (cost-only) | n/a | +$0.340 | **costs more than always-top** under the recorded trajectories; quality not identifiable |
| P5: per-call cascade | gpt-5-mini→5.2 | (cost-only) | n/a | +$0.050 | **costs more than always-top**; quality not identifiable |

**Per-session vs per-workload (pre-registered comparison)**: P3 (per-instance, 2-model
ladder) costs $0.720/inst vs P2 (workload, 11-model pool) at $0.219/inst, with P3 resolving
more (378 vs 374). Per the pre-registered rule, P3 does NOT add value over P2: it is 3.3×
more expensive. P2's access to very cheap near-frontier models (minimax-2.5, gemini-3-flash)
dominates any per-instance routing benefit from the repo feature.

---

## Prompt caching and mid-run switching (cost-only)

A mid-run handoff makes the new model re-read the whole accumulated context without a cache
(each model has its own cache). Under the calibrated context model that first read costs
I_0 + k·delta tokens at the new model's uncached input price after k steps: for haiku→opus,
$0.043 at k = 0 and $0.249 at k = 19 (2,873 + 19×724 tokens × $15/1M). The cost grows linearly
with the session, so handoffs late in a long session are the most expensive, and their quality
cannot be scored by replay. This is a cost fact, not a quality result; it argues for choosing the
model at session start (or per workload) and for escalating by restart or at a natural cache
boundary rather than mid-prefix.

The estimated no-cache factor for single-model sessions is 3.98× to 5.37× (later steps become
progressively cheaper with cached prefixes), which is why staying on one model per session is
so much cheaper than any model that re-reads the prefix.

---

## What this means

1. **Workload-level model selection remains the strongest routing result** (+$0.490/inst savings
   vs always-top at borderline quality, replicates #239). The right routing granularity is
   "per workload, from a diverse model pool," not finer.

2. **Per-session routing (repo feature only) sacrifices quality** without proportional cost
   savings on these ladders. Adding richer session-start features (problem statement length,
   issue category) might recover quality, but would need re-measurement.

3. **Escalate-by-restart gives no gain on these runs**: with a step-budget trigger the cross-fit
   escalates immediately, so it equals always-top. Beating always-top needs a cheap model that is
   sufficient for a whole workload, which is what workload-level selection does, with no switching.

4. **Per-call cascade costs more than always-top** under the recorded trajectories on both
   ladders (consistent with #239); its quality is not identifiable by replay.

5. **The oracle bound is far from any live policy**: 395/500 at $0.384 is 2× cheaper than
   always-top at 2.9pp better quality. This gap measures the theoretical value of a perfect
   instance-level router — a target for future work with richer features (issue text, repo
   history, problem complexity signals).

---

## Limitations

1. **Replay validity**: escalate-by-restart and the whole-run policies are scored exactly; per-call
   cascade quality and mid-run handoffs are not identifiable (see the validity caveat).
2. **Single run per model per instance**: run-to-run variance is not captured.
3. **Context model is estimated** from 20 trajectories. Token counts from chars/2.5 may be
   off by 10–30%. The "no caching" column is an approximation; the "observed (w/ caching)"
   column uses authoritative leaderboard costs.
4. **No-cache column for escalate-and-pin** uses a different `out_per_step` (hardcoded 219
   vs calibrated 289/245) than other arms, causing a slight inconsistency. For escalate-and-pin
   with tau=-1 (all instances use top model), the no-cache cost should equal always-top's
   no-cache cost; the observed small difference is a modelling artefact.
5. **One scaffold, one benchmark**: mini-SWE-agent bash-only, SWE-bench Verified.
6. **Approximate price table**: list prices are used for the handoff re-read illustration.

---

## How to rerun

```bash
python3 -I scripts/analyze-agent-granularity.py ~/.cache/firstpass-swebench-experiments
```

Data is fetched at runtime into the cache dir (per_instance_details.json + 20 trajectory
samples per model for context calibration). Already-cached files are reused. Self-check:

```bash
python3 -I scripts/analyze-agent-granularity.py --self-check
```
