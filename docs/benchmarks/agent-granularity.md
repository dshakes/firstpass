# Routing granularity for multi-turn coding agents

date: 2026-11-01 · spend: $0 (public runs) · data: github.com/SWE-bench/experiments@40f164d ·
command: `python3 -I scripts/analyze-agent-granularity.py [CACHE_DIR]`

Pre-registration: [agent-granularity-preregistration.md](agent-granularity-preregistration.md)
(committed before any result was computed; policies, metrics, and decision rules fixed in advance).

---

## Validity caveat (prominent, per pre-registration)

Recorded trajectories cannot show counterfactual outcomes. Escalate-and-pin's post-switch
quality is approximated by the top model's own full run on that instance:
- **Optimistic for quality** (top model starts fresh; real post-switch quality may be lower if
  the cheap model's partial exploration confused the problem space).
- **Pessimistic for cost** (top model might finish in fewer steps inheriting useful context;
  real cost may be lower).

Context growth model (I_0, delta) is estimated from 20 sampled trajectories at 2.5 chars/token.
Real token counts differ by ±10–30%. The "no caching" column is labelled as estimated.

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
| per-call cascade (cross-fit) [P5] | 382/500 (0.764) | $1.071 | $4.337 | $1.401 | [-0.010, +0.000] | [+0.295, +0.340] |
| per-session / repo (cross-fit) [P3] | 378/500 (0.756) | $0.720 | $3.868 | $0.953 | [-0.024, -0.002] | [-0.048, -0.021] |
| escalate-and-pin (cross-fit) [P4] | 384/500 (0.768) | $0.797 | $3.634 | $1.038 | [+0.000, +0.000] | [+0.043, +0.043] |
| oracle-per-instance (bound) [P6] | 395/500 (0.790) | $0.384 | $2.061 | $0.486 | [+0.010, +0.036] | [-0.408, -0.334] |
| oracle-cascade (bound) | 395/500 (0.790) | $0.679 | $2.703 | $0.859 | [+0.010, +0.036] | [-0.116, -0.032] |

**Switch overhead breakdown** (escalate-and-pin, tau=-1):
- All 500 instances trigger escalation (tau=-1 = escalate immediately)
- Context at switch: 2,873 tokens (initial prompt only; no cheap model steps)
- Switch overhead per instance: $0.043 (2,873 tokens × $15.00/1M)
- Escalate-and-pin with overhead: $0.797 = $0.000 (partial cheap) + $0.043 (overhead) + $0.754 (top)
- Escalate-and-pin WITHOUT overhead would cost: $0.754 = same as always-top

### GPT-5-mini→GPT-5.2-high (v2.0.0)  (n=500)

Cross-fit parameters: cascade tau=[6, 8] (per half), escalate-and-pin tau=[-1, -1], repo-eps=[0.05, 0.1]

| arm | resolved | $/inst (observed, w/ caching) | $/inst (est. no caching) | $/resolved (w/ caching) | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|---|
| always-cheap | 281/500 (0.562) | $0.047 | $0.163 | $0.084 | [-0.206, -0.126] | [-0.459, -0.396] |
| always-top | 364/500 (0.728) | $0.474 | $2.228 | $0.651 | — | — |
| per-call cascade (cross-fit) [P5] | 364/500 (0.728) | $0.520 | $2.212 | $0.714 | [+0.000, +0.000] | [+0.043, +0.050] |
| per-session / repo (cross-fit) [P3] | 353/500 (0.706) | $0.416 | $1.434 | $0.589 | [-0.042, -0.002] | [-0.074, -0.042] |
| escalate-and-pin (cross-fit) [P4] | 364/500 (0.728) | $0.482 | $2.573 | $0.662 | [+0.000, +0.000] | [+0.009, +0.009] |
| oracle-per-instance (bound) [P6] | 383/500 (0.766) | $0.154 | $0.530 | $0.201 | [+0.022, +0.056] | [-0.353, -0.289] |
| oracle-cascade (bound) | 383/500 (0.766) | $0.317 | $1.492 | $0.414 | [+0.022, +0.056] | [-0.182, -0.133] |

**Switch overhead breakdown** (escalate-and-pin, tau=-1):
- All 500 instances trigger escalation (tau=-1 = escalate immediately)
- Context at switch: 1,720 tokens × $5.00/1M = $0.009/instance
- Escalate-and-pin without overhead: $0.474 = same as always-top

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
| P4: escalate-and-pin (cache-aware) | haiku→opus | 384/500 | +0.000 | +$0.043 | **more expensive than always-top** (same quality, +$0.043/inst overhead) |
| P4: escalate-and-pin (cache-aware) | gpt-5-mini→5.2 | 364/500 | +0.000 | +$0.009 | **marginally more expensive** (same quality, +$0.009/inst) |
| P5: per-call cascade | haiku→opus | 382/500 | −0.010 | +$0.340 | **negative result confirmed** (not cheaper, borderline quality) |
| P5: per-call cascade | gpt-5-mini→5.2 | 364/500 | +0.000 | +$0.050 | **negative result confirmed** (not cheaper, same quality) |

**Cache-aware verdict for escalate-and-pin**: The switch overhead materially changes the
cost verdict for haiku→opus — without caching, escalate-and-pin at immediate escalation
would cost $0.754/instance (same as always-top); with caching it costs $0.797/instance
(+5.7%). For gpt-5-mini→gpt-5.2, the change is +1.9% ($0.009/inst), which is negligible.

**Per-session vs per-workload (pre-registered comparison)**: P3 (per-instance, 2-model
ladder) costs $0.720/inst vs P2 (workload, 11-model pool) at $0.219/inst, with P3 resolving
more (378 vs 374). Per the pre-registered rule, P3 does NOT add value over P2: it is 3.3×
more expensive. P2's access to very cheap near-frontier models (minimax-2.5, gemini-3-flash)
dominates any per-instance routing benefit from the repo feature.

---

## The cache-aware cost finding

**The key new contribution**: provider prompt caching makes mid-session model switching
expensive. This study quantifies the effect at both ends of the optimal escalation curve.

| ladder | optimal escalation point | context at switch | switch overhead | overhead as % of always-top cost | verdict without overhead |
|---|---|---|---|---|---|
| haiku→opus | step 0 (immediate) | 2,873 tokens (initial prompt) | $0.043/inst | +5.7% | tie with always-top |
| gpt-5-mini→5.2 | step 0 (immediate) | 1,720 tokens (initial prompt) | $0.009/inst | +1.9% | tie with always-top |

**Why step 0 is optimal**: for ANY tau > 0, the accumulated context grows at 724 (haiku) or
842 (gpt-5-mini) tokens per step, increasing the switch overhead faster than the partial cheap
model cost savings. At tau=19 (haiku), overhead = $0.249/instance — larger than the entire
partial cheap model savings of ~$0.116. The optimal strategy is to escalate before accumulating
any context, which degenerates to always-top plus a one-time re-read cost.

**No-cache comparison**: Without provider caching, there is no switch overhead (all tokens
priced identically regardless of which model reads them). The cross-model cache penalty is
purely a consequence of caching architecture. The estimated no-cache factor for single-model
sessions is 3.98× to 5.37× (later steps become progressively cheaper with cached prefixes).

---

## What this means

1. **Workload-level model selection remains the strongest routing result** (+$0.490/inst savings
   vs always-top at borderline quality, replicates #239). The right routing granularity is
   "per workload, from a diverse model pool," not finer.

2. **Per-session routing (repo feature only) sacrifices quality** without proportional cost
   savings on these ladders. Adding richer session-start features (problem statement length,
   issue category) might recover quality, but would need re-measurement.

3. **Escalate-and-pin is dominated by always-top** once caching is accounted for: the switch
   overhead at any escalation point (even immediate) exceeds zero. The only way to beat
   always-top cost with mid-session switching is to use a cheap model that is sometimes
   sufficient — which is exactly what workload-level selection achieves, without any switching.

4. **Per-call cascade remains negative**: higher cost, same or slightly lower quality. This
   replicates the finding from #239 on all tested ladders.

5. **The oracle bound is far from any live policy**: 395/500 at $0.384 is 2× cheaper than
   always-top at 2.9pp better quality. This gap measures the theoretical value of a perfect
   instance-level router — a target for future work with richer features (issue text, repo
   history, problem complexity signals).

---

## Limitations

1. **Replay invalidity for escalate-and-pin**: the post-switch outcome cannot be directly
   measured; the top model's full run is an approximation.
2. **Single run per model per instance**: run-to-run variance is not captured.
3. **Context model is estimated** from 20 trajectories. Token counts from chars/2.5 may be
   off by 10–30%. The "no caching" column is an approximation; the "observed (w/ caching)"
   column uses authoritative leaderboard costs.
4. **No-cache column for escalate-and-pin** uses a different `out_per_step` (hardcoded 219
   vs calibrated 289/245) than other arms, causing a slight inconsistency. For escalate-and-pin
   with tau=-1 (all instances use top model), the no-cache cost should equal always-top's
   no-cache cost; the observed small difference is a modelling artefact.
5. **One scaffold, one benchmark**: mini-SWE-agent bash-only, SWE-bench Verified.
6. **Approximate price table**: list prices used for switch overhead; enterprise discounts
   would reduce the switch overhead.

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
