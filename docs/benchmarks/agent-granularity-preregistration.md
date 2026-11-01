# Routing granularity for multi-turn coding agents — pre-registration

Written **before** any routing result was computed (before the script ran on the live data).
The policies, metrics and decision rules below fix the interpretation in advance.

date: 2026-11-01 · data: github.com/SWE-bench/experiments@40f164d (per-instance
`{resolved, cost, api_calls}` + trajectory messages for context calibration), fetched at
run time into a gitignored cache, never committed.
command: `python3 -I scripts/analyze-agent-granularity.py [CACHE_DIR]`

---

## The question under test

Is there value in routing at finer granularity than a whole workload?
Granularity ladder (coarsest to finest):

| level | description |
|---|---|
| per-workload | one model for all 500 instances, chosen by cross-fitted evidence |
| per-session | one model per instance, chosen before the session starts from observable session-start features |
| mid-session escalate | start cheap; switch to top on a session-level failure signal mid-run |
| per-call cascade | gate each step and re-run on failure (existing policy, replicate negative result) |
| oracle per-instance | cheapest model that resolves each instance (upper bound) |

The null hypothesis for each comparison: finer granularity does **not** reduce cost-per-resolved
instance (or increase resolved rate) beyond the coarser granularity.

---

## Available data

- **Source**: SWE-bench Verified mini-SWE-agent bash-only leaderboard, commit `40f164d`.
  Per-instance `{resolved, cost_usd, api_calls}` for each model-system.
  Trajectory JSON for context calibration (message lengths, not token counts; fetched for a
  small calibration sample only).
- **Validity caveat (prominent)**: trajectories are recorded runs, not live experiments.
  A mid-session model switch cannot be replayed: the continued trajectory on a different model
  does not exist in the data. The post-switch outcome for escalate-and-pin is approximated by
  the top model's own full run on that instance. This is an **optimistic proxy for success**
  (the top model runs from a clean start, not confused by the cheap model's partial exploration;
  real performance may be lower) and a **pessimistic proxy for cost** (the top model might finish
  in fewer steps if it inherits useful context; real cost may be lower). Sensitivity bounds are
  reported rather than single-point estimates where this uncertainty is material.

---

## Models and ladders

Primary ladder: **claude haiku→opus (v2.0.0)**
- Cheap: `20260217_mini-v2.0.0_claude-4-5-haiku-high`
- Top: `20260217_mini-v2.0.0_claude-4-5-opus-high`

Secondary ladder: **gpt-5-mini→gpt-5.2-high (v2.0.0)**
- Cheap: `20260217_mini-v2.0.0_gpt-5-mini`
- Top: `20260217_mini-v2.0.0_gpt-5-2-high`

Pool for workload/per-session selection (11 models from `20260217_mini-v2.0.0_*`):
haiku, sonnet, opus, claude-4-6-opus, deepseek-3-2, gemini-3-flash, glm-5, gpt-5-2, gpt-5-mini,
kimi-k2-5, minimax-2-5. Excluded: gemini-3-pro-high (resolved=false on all 500, data defect).

---

## Policies (fixed before results were computed)

### P1 — always-cheap / always-top
Use one model for every instance. No routing. Two arms per ladder.

### P2 — per-workload selection (cross-fit)
From the 11-model pool, pick the cheapest model whose resolve rate on the training half is within
`SELECT_EPS = 0.01` of the best. Apply that single model to the test half (all instances).
Cross-fit: train on even-indexed instances, score on odd, and vice versa (by sorted instance id).
Replicates #239's workload-level selection result.

### P3 — per-session selection (cross-fit, repo-level feature)
From the 2-model ladder (cheap vs top), for each instance use the instance's repo as the sole
session-start feature. A repo-level router learns each repo's per-model resolve rate on the
training half and routes each test instance to the cheapest model whose repo rate is within
`eps` (cross-fitted on training half from `EPS_GRID = [0.0, 0.05, 0.1, 0.2, 0.3]`) of the best.
Falls back to the global rate for unseen repos.

**Scope**: only features observable before the session starts. `api_calls` is NOT available at
session start and is excluded. Repo name is encoded in the instance ID and costs nothing to
observe.

**Comparison to P2**: P3 makes a per-instance decision from limited features (repo only);
P2 makes a per-workload decision from aggregate evidence (full training half). Neither is
strictly more powerful: P2 has access to 11 models; P3 has per-repo signal. Both are
cross-fitted on the same parity split.

### P4 — escalate-and-pin (mid-session, with cache-aware cost)
Start every session with the cheap model. Monitor `api_calls` as the session progresses.
When `api_calls >= tau_pin` (a pre-session-failure signal: the agent has used many steps and
is likely struggling), switch immediately to the top model **for the rest of the session**.

**Signal available online**: step count is visible to any router watching the session;
`tau_pin` is cross-fitted on the training half.

**Post-switch approximation**: the top model's outcome is taken from its own full recorded run
on that instance (the only data available; see validity caveat above).

**Cost model (cache-aware)**:
- Partial cheap cost = `cheap_leaderboard_cost × tau_pin / cheap_api_calls` (linear
  interpolation; assumes cost is proportional to steps run).
- Switch overhead = `context_tokens_at_tau × price_top_input / 1e6` where `context_tokens_at_tau`
  is estimated from a sample of trajectories (message lengths / 2.5 chars per token; see
  context model below).
- Top model continuation = `top_leaderboard_cost` (full run, optimistic for success,
  pessimistic for cost; see validity caveat).
- Total cost with caching = `partial_cheap + switch_overhead + top_leaderboard_cost`.
- Total cost without caching = `partial_cheap_nocache + top_nocache` (no switch overhead: without
  caching, switching models carries no cache-miss penalty).

**tau_pin fitted objective** (same as cascade): cheapest `tau_pin` (from step-count quantile grid)
whose resolved count on the training half equals or exceeds always-top; if none does, the most
resolved. Cost includes switch overhead in the fit.

### P5 — per-call cascade (cross-fit, replication)
Same as `first-pass (steps gate, cross-fit)` in `scripts/analyze-agent-cascade.py` (PR #239).
Run cheap model to completion; if `api_calls <= tau`, serve cheap outcome; else re-run top model
from scratch. `tau` cross-fitted from step-count quantile grid. **Expected result: negative
(cascade with step gate does not beat always-top on quality or cost)**. This is a pre-registered
replication of the negative result.

### P6 — oracle bounds
- **Oracle per-instance**: cheapest model that resolves each instance; served as-is (upper bound
  for any routing policy on this data). Note: not a reachable policy without cheating.
- **Random**: with probability 0.5 serve cheap, probability 0.5 serve top (independently per
  instance, seeded). Lower-bound sanity check.

---

## Cache-aware cost model

**Why caching matters for routing**: with provider prompt caching, the conversation prefix is
re-read at 0.1× the input rate after the first write at 1.25×. Within a single-model session,
later steps become progressively cheaper as the prefix grows. A mid-session model switch resets
the cache: the top model's first call must read the entire accumulated prefix at the full (1×)
input rate, paying 1.0 − 0.1 = 0.9× more than it would have if it had been running from the start.

**Context model**: from a calibration sample of `N_CALIB = 20` trajectories of the cheap model,
estimate:
- `I_0`: initial context size (system + problem statement, chars / 2.5 tokens per char)
- `delta`: average tokens added per API call step (assistant + tool chars / api_calls / 2.5)
- `out_per_step`: average output tokens per step (assistant chars only / api_calls / 2.5)

Context at step `k ≈ I_0 + k × delta`. If trajectory fetch fails, fall back to calibrated
defaults from one manually analysed haiku instance (`I_0 = 2333`, `delta = 574`, `out_per_step = 219`).
Label clearly whether calibrated or hardcoded.

**No-cache cost** for a session of `N` steps (analytical from context model):
```
uncached_input = sum(I_0 + k*delta for k in range(N))
uncached_output = N × out_per_step
cost_nocache = (uncached_input × price_in + uncached_output × price_out) / 1e6
```

**With-cache cost** (analytical, single model):
```
cost_withcache = (I_0 × 1.25 × price_in                            # first write
  + sum((I_0+k*delta)*0.1 + delta*1.25)*price_in for k=1..N-1      # reads + new writes
  + N × out_per_step × price_out) / 1e6
```

**Observed cost** (leaderboard): taken as the ground truth for "with caching active." It will
not exactly match the analytical model (different caching behaviour, model verbosity, etc.)
but is used as the authoritative per-instance cost for all policies except escalate-and-pin
(whose cost is modelled, not observed, since it never ran).

---

## Metrics

For each policy arm (baseline for CIs: always-top):

| metric | definition |
|---|---|
| resolved | number of instances resolved / n |
| $/instance (observed) | mean leaderboard cost per instance |
| $/instance (with cache model) | modelled cost including switch overhead where applicable |
| $/instance (no cache, estimated) | estimated cost under flat token pricing |
| $/resolved (observed) | total leaderboard cost / resolved count |
| Δ resolved | paired bootstrap CI vs always-top (n_boot=4000, seed=42) |
| Δ $/instance | paired bootstrap CI vs always-top |
| Δ $/resolved | paired bootstrap CI vs always-top |

---

## Decision rules (pre-registered)

A policy "beats" always-top if **both**:
1. Δ resolved 95% CI lower bound ≥ −0.01 (no more than 1pp quality loss), AND
2. Δ $/instance 95% CI upper bound < −0.01 (significantly cheaper).

A policy "loses on quality" if Δ resolved 95% CI lower bound < −0.01.

A policy is "inconclusive" if it fails both rules above.

**Per-workload vs per-session comparison** (policies P2 vs P3): per-session selection "adds value"
over per-workload selection if the Δ resolved CI (P3 − P2) lower bound ≥ 0 (strictly better quality)
OR (CI contains 0 AND Δ $/instance CI upper bound < 0 — strictly cheaper). Otherwise inconclusive.

**Cache analysis for escalate-and-pin**: the switch overhead "materially changes" the cost verdict
if the $/instance verdict (beat / lose / inconclusive) changes between the "no switch overhead"
and "with switch overhead" cost columns.

**Cascade replication**: confirmed negative if cascade arm Δ resolved CI lower bound < −0.01
OR Δ $/instance CI upper bound ≥ 0 (not cheaper). Expected from #239.

---

## Cross-fitting and no data leakage

All threshold parameters (`tau`, `tau_pin`, `eps`) are fitted on one parity half of sorted
instance IDs and scored on the other. No reported metric saw its own training data. The pool
model choice (P2) and repo-router eps (P3) follow the same parity split.

The cascade tau and escalate-and-pin tau are fitted independently using the same objective
(cheapest config matching or exceeding always-top resolved count on training half).

---

## Limitations (fixed before results)

1. **Replay validity**: recorded trajectories cannot show counterfactual outcomes. Post-switch
   quality for escalate-and-pin is an approximation (see validity caveat above).
2. **Single recorded run per model per instance**: run-to-run variance is not modelled.
3. **Context model approximation**: I_0 and delta estimated from 20 sample trajectories using
   2.5 chars/token. Real token counts will differ by ±10–30%.
4. **Leaderboard costs may not match analytic model**: provider may cache subsets, use batching,
   or bill at non-list prices. The analytic no-cache cost is labelled "estimated."
5. **One scaffold, one benchmark**: results may not generalise to other agent scaffolds or tasks.
6. **Price table**: approximate list prices are used for the top model's switch overhead
   calculation; enterprise discounts would reduce the switch overhead.
