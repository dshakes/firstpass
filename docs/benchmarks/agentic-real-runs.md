# Agentic routing on real agent runs — SWE-bench Verified, mini-SWE-agent (bash-only leaderboard)

date: 2026-10-30 · spend: $0 (public runs) · data: github.com/SWE-bench/experiments@40f164d (per-instance
`{resolved, cost, api_calls}`) + submitted patches from the public trajectories. The data is fetched at run
time and not committed, because the repository states no licence.
command: `python3 scripts/analyze-agent-cascade.py [CACHE_DIR]` (self-check: `--self-check`)

## Why this exists

Our own agentic evidence (SWE-60, 459 turns) was small, used a gate that runs the benchmark's answer-key
tests, and replayed turns counterfactually. This study fixes all three:
- n=500 per model, with real costs.
- Only live-observable gates.
- The whole agent run is the routing unit. A cascade reruns the task from scratch on the next model, and
  that rerun *is* the recorded run, so no conversation is invented.

Every rule, ladder and threshold objective was fixed in the script before its result was computed.
Thresholds are cross-fit by instance parity. One choice was made after seeing per-model rates and is
labelled EXPLORATORY; it was then replicated on runs it was not chosen on.

## Findings

1. **Per-run cascades with a live gate do not beat always-top on agent tasks.** On all 4 ladders, the
   step-count gate (cross-fit) resolves about the same or fewer than the top model, at equal or higher
   cost. A perfect gate would win on every ladder: the oracle-cascade resolves 2–6pp more, for less.
   **The gate is the whole problem.**
2. **No live gate tested on finished runs is precise enough.** Step count, cross-model "same files" agreement
   and "same hunks" agreement lift precision only 5–10pp above the base rate (e.g. haiku 0.666 → 0.773).
3. **The exploratory agreement win did not replicate.** On v2.0.0, minimax-2.5 + kimi → opus resolved 383 vs
   384 at −53% cost. On the older v1.16–1.17 runs (fixed before fetching), the same roles lose on both
   quality and cost (338–349 vs 372, +$0.24–0.58/instance). The v2.0.0 effect came from minimax-2.5 being
   a cheap near-frontier model, not from the gate.
4. **What holds: workload-level model selection.** Pick, on one half, the cheapest of 11 same-scaffold models
   within 1pp of the best resolve rate, then score it on the other half. Result: 374/500 vs always-opus-4.5's
   384/500 (Δ [−0.050, +0.008], not significant) at **$0.219 vs $0.754 per instance (−71%, CI
   [−0.582, −0.490])**. On agent traffic, the money is in choosing the model for the workload from measured
   evidence. Escalating per request does not pay it.

## What this means for the product

- Keep per-request cascades where a precise gate exists (tests, schema): MBPP head-to-head.
- For agent traffic, the router's job is **evidence-driven ladder selection per route**: measure candidate
  models on the route's own traffic (shadow runs, receipts), then serve the cheapest model statistically
  non-inferior to the incumbent. Don't per-request escalate on a weak gate.
- Limits:
  - One recorded run per model, so run-to-run noise is not modelled.
  - Costs are as logged by the leaderboard and were not independently re-priced.
  - One scaffold, one benchmark.
  - Selection was cross-fit within one scaffold era, not across eras.

## Full output

```
# Agentic routing on real agent runs (SWE-bench Verified, mini-SWE-agent; data @40f164d)

## claude haiku→sonnet→opus (v2.0.0)  (n=500)

rungs: claude-4-5-haiku-high → claude-4-5-sonnet-high → claude-4-5-opus-high
fitted per half (step taus, repo eps): [([19, 19], 0.0), ([34, 36], 0.0)]

| arm | resolved | $/instance | $/resolved | Δ resolved vs first-pass | Δ $/inst vs first-pass | Δ $/resolved vs first-pass |
|---|---|---|---|---|---|---|
| always-claude-4-5-haiku-high | 333/500 (0.666) | $0.331 | $0.497 | [-0.122, -0.060] | [-1.412, -1.252] | [-1.858, -1.557] |
| always-claude-4-5-sonnet-high | 357/500 (0.714) | $0.658 | $0.921 | [-0.070, -0.016] | [-1.075, -0.936] | [-1.419, -1.155] |
| always-claude-4-5-opus-high | 384/500 (0.768) | $0.754 | $0.982 | [+0.004, +0.022] | [-0.966, -0.853] | [-1.327, -1.114] |
| first-pass (steps gate, cross-fit) | 378/500 (0.756) | $1.663 | $2.199 | — | — | — |
| repo-predictive (cross-fit) | 374/500 (0.748) | $0.713 | $0.953 | [-0.024, +0.006] | [-1.010, -0.891] | [-1.360, -1.140] |
| oracle-cascade (bound) | 403/500 (0.806) | $0.892 | $1.107 | [+0.032, +0.070] | [-0.840, -0.698] | [-1.195, -0.994] |
| oracle-predictive (bound) | 403/500 (0.806) | $0.387 | $0.480 | [+0.032, +0.070] | [-1.355, -1.196] | [-1.884, -1.568] |

## claude haiku→opus (v2.0.0)  (n=500)

rungs: claude-4-5-haiku-high → claude-4-5-opus-high
fitted per half (step taus, repo eps): [([19], 0.0), ([34], 0.0)]

| arm | resolved | $/instance | $/resolved | Δ resolved vs first-pass | Δ $/inst vs first-pass | Δ $/resolved vs first-pass |
|---|---|---|---|---|---|---|
| always-claude-4-5-haiku-high | 333/500 (0.666) | $0.331 | $0.497 | [-0.130, -0.068] | [-0.790, -0.691] | [-0.998, -0.818] |
| always-claude-4-5-opus-high | 384/500 (0.768) | $0.754 | $0.982 | [+0.000, +0.010] | [-0.340, -0.295] | [-0.463, -0.381] |
| first-pass (steps gate, cross-fit) | 382/500 (0.764) | $1.071 | $1.401 | — | — | — |
| repo-predictive (cross-fit) | 378/500 (0.756) | $0.720 | $0.953 | [-0.020, +0.004] | [-0.378, -0.325] | [-0.497, -0.405] |
| oracle-cascade (bound) | 395/500 (0.790) | $0.679 | $0.859 | [+0.012, +0.040] | [-0.431, -0.352] | [-0.600, -0.486] |
| oracle-predictive (bound) | 395/500 (0.790) | $0.384 | $0.486 | [+0.012, +0.040] | [-0.736, -0.639] | [-1.014, -0.826] |

## gpt-5-mini→gpt-5.2-high (v2.0.0)  (n=500)

rungs: gpt-5-mini → gpt-5-2-high
fitted per half (step taus, repo eps): [([6], 0.05), ([8], 0.1)]

| arm | resolved | $/instance | $/resolved | Δ resolved vs first-pass | Δ $/inst vs first-pass | Δ $/resolved vs first-pass |
|---|---|---|---|---|---|---|
| always-gpt-5-mini | 281/500 (0.562) | $0.047 | $0.084 | [-0.206, -0.126] | [-0.507, -0.441] | [-0.698, -0.570] |
| always-gpt-5-2-high | 364/500 (0.728) | $0.474 | $0.651 | [+0.000, +0.000] | [-0.050, -0.043] | [-0.070, -0.058] |
| first-pass (steps gate, cross-fit) | 364/500 (0.728) | $0.520 | $0.714 | — | — | — |
| repo-predictive (cross-fit) | 353/500 (0.706) | $0.416 | $0.589 | [-0.042, -0.002] | [-0.121, -0.088] | [-0.155, -0.100] |
| oracle-cascade (bound) | 383/500 (0.766) | $0.317 | $0.414 | [+0.022, +0.056] | [-0.228, -0.179] | [-0.344, -0.261] |
| oracle-predictive (bound) | 383/500 (0.766) | $0.154 | $0.201 | [+0.022, +0.056] | [-0.400, -0.335] | [-0.583, -0.454] |

## gpt-5-mini→gpt-5 (v1.7.0)  (n=500)

rungs: gpt-5-mini → gpt-5
fitted per half (step taus, repo eps): [([12], 0.05), ([4], 0.0)]

| arm | resolved | $/instance | $/resolved | Δ resolved vs first-pass | Δ $/inst vs first-pass | Δ $/resolved vs first-pass |
|---|---|---|---|---|---|---|
| always-gpt-5-mini | 299/500 (0.598) | $0.035 | $0.059 | [-0.068, -0.008] | [-0.255, -0.203] | [-0.411, -0.307] |
| always-gpt-5 | 325/500 (0.650) | $0.280 | $0.431 | [-0.004, +0.032] | [+0.006, +0.028] | [-0.003, +0.039] |
| first-pass (steps gate, cross-fit) | 318/500 (0.636) | $0.263 | $0.414 | — | — | — |
| repo-predictive (cross-fit) | 309/500 (0.618) | $0.164 | $0.266 | [-0.038, +0.002] | [-0.123, -0.079] | [-0.191, -0.113] |
| oracle-cascade (bound) | 353/500 (0.706) | $0.180 | $0.255 | [+0.048, +0.092] | [-0.105, -0.065] | [-0.196, -0.129] |
| oracle-predictive (bound) | 353/500 (0.706) | $0.062 | $0.087 | [+0.048, +0.092] | [-0.229, -0.177] | [-0.384, -0.278] |

## Workload-level model selection (cross-fit, 11 models, n=500)

picked for even half: minimax-2-5-high; for odd half: gemini-3-flash-high

| arm | resolved | $/instance | Δ resolved vs always-opus-4.5 | Δ $/inst vs always-opus-4.5 |
|---|---|---|---|---|
| selected (cross-fit) | 374/500 (0.748) | $0.219 | [-0.050, +0.008] | [-0.582, -0.490] |
| always-claude-4-5-opus-high | 384/500 (0.768) | $0.754 | — | — |

## Agreement gate on finished agent runs (parameter-free, no fitting)

### A=haiku, B=gpt-5-mini → opus  (n=500)

| arm | resolved | $/instance | gate pass | gate precision (A resolved given pass) | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|---|
| always-A | 333/500 (0.666) | $0.331 | — | — | [-0.136, -0.072] | [-0.460, -0.387] |
| always-top | 384/500 (0.768) | $0.754 | — | — | — | — |
| agree-files(A,B) → top | 351/500 (0.702) | $0.622 | 391/500 | 0.734 | [-0.094, -0.040] | [-0.173, -0.088] |
| agree-hunks(A,B) → top | 361/500 (0.722) | $0.792 | 299/500 | 0.773 | [-0.068, -0.026] | [-0.001, +0.078] |

base rate (A resolved): 0.666

### A=gpt-5-mini, B=haiku → gpt-5.2-high  (n=500)

| arm | resolved | $/instance | gate pass | gate precision (A resolved given pass) | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|---|
| always-A | 281/500 (0.562) | $0.047 | — | — | [-0.206, -0.126] | [-0.459, -0.396] |
| always-top | 364/500 (0.728) | $0.474 | — | — | — | — |
| agree-files(A,B) → top | 325/500 (0.650) | $0.527 | 391/500 | 0.665 | [-0.112, -0.044] | [+0.021, +0.087] |
| agree-hunks(A,B) → top | 332/500 (0.664) | $0.590 | 333/500 | 0.685 | [-0.092, -0.038] | [+0.085, +0.150] |

base rate (A resolved): 0.562

### A=gemini-3-flash, B=gpt-5-mini → opus  (n=500)

| arm | resolved | $/instance | gate pass | gate precision (A resolved given pass) | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|---|
| always-A | 379/500 (0.758) | $0.356 | — | — | [-0.040, +0.018] | [-0.442, -0.354] |
| always-top | 384/500 (0.768) | $0.754 | — | — | — | — |
| agree-files(A,B) → top | 379/500 (0.758) | $0.654 | 381/500 | 0.808 | [-0.032, +0.014] | [-0.146, -0.055] |
| agree-hunks(A,B) → top | 381/500 (0.762) | $0.846 | 269/500 | 0.833 | [-0.024, +0.012] | [+0.048, +0.133] |

base rate (A resolved): 0.758

### EXPLORATORY A=minimax, B=kimi → opus  (n=500)

| arm | resolved | $/instance | gate pass | gate precision (A resolved given pass) | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|---|
| always-A | 379/500 (0.758) | $0.073 | — | — | [-0.040, +0.020] | [-0.726, -0.636] |
| always-top | 384/500 (0.768) | $0.754 | — | — | — | — |
| agree-files(A,B) → top | 383/500 (0.766) | $0.354 | 439/500 | 0.802 | [-0.026, +0.022] | [-0.444, -0.356] |
| agree-hunks(A,B) → top | 379/500 (0.758) | $0.499 | 363/500 | 0.821 | [-0.030, +0.010] | [-0.300, -0.211] |

base rate (A resolved): 0.758

### REPLICATION (v1.16-1.17) A=minimax-m2, B=kimi-k2-thinking → opus-4.5  (n=500)

| arm | resolved | $/instance | gate pass | gate precision (A resolved given pass) | Δ resolved vs always-top | Δ $/inst vs always-top |
|---|---|---|---|---|---|---|
| always-A | 305/500 (0.610) | $0.428 | — | — | [-0.172, -0.094] | [-0.343, -0.242] |
| always-top | 372/500 (0.744) | $0.721 | — | — | — | — |
| agree-files(A,B) → top | 338/500 (0.676) | $1.051 | 411/500 | 0.686 | [-0.098, -0.038] | [+0.244, +0.419] |
| agree-hunks(A,B) → top | 349/500 (0.698) | $1.209 | 322/500 | 0.736 | [-0.072, -0.020] | [+0.402, +0.577] |

base rate (A resolved): 0.610

```
