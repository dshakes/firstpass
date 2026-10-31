# Certified model selection: evidence ($0, real recorded runs)

The question for [ADR 0014](../adr/0014-certified-model-selection.md) is whether the experiment
reaches the right decision, how often it is wrong, and how much traffic it needs to decide. Both
replays mirror `crates/firstpass-core/src/select.rs` exactly: same λ grid, same pairing, same
stopping rule. Neither spends anything.

## 1. Agent sessions, live mode: SWE-bench Verified (`scripts/sim-certified-selection.py`)

**Data.** 500 real mini-SWE-agent runs per model from the public leaderboard (data @40f164d). The
incumbent is opus-4.5. Sessions are random instances split 50/50 between arms. Margin 0.05,
alpha 0.05, 400 replays, capped at 20k sessions.

| challenger | true gap | cost ratio | certified | halted | median sessions |
|---|---|---|---|---|---|
| minimax-2.5 | +0.010 | 0.10 | 0.99 | 0.01 | 3,667 |
| gemini-3-flash | +0.010 | 0.47 | 0.99 | 0.01 | 3,600 |
| opus-4.6 | +0.012 | 0.73 | 0.99 | 0.01 | 4,154 |
| kimi-k2.5 | +0.060 | 0.19 | 0.03 | 0.15 | 1,481 |
| haiku-4.5 | +0.102 | 0.44 | 0.01 | 0.99 | 2,206 |
| gpt-5-mini | +0.206 | 0.06 | 0.00 | 1.00 | 223 |
| synthetic, exactly 0.05 worse | 0.050 | — | 0.045 | 0.058 | — |

**Reading.**
- Every challenger within about a point of opus certified, including one 10× cheaper.
- Both clearly worse models halted.
- At the null boundary the test certified 4.5% of the time, inside its 5% error rate.
- kimi sits just past the margin (6 points), which is the hardest case to decide. Mostly it stays
  undecided and never wrongly certifies.

## 2. Single-turn coding, paired vs live: MBPP (`scripts/sim-certified-selection-mbpp.py`)

**Data.** Real paid calls of haiku-4.5, sonnet-5 and opus-4.8 on the **same** 974 MBPP tasks
(`docs/benchmarks/mbpp-974-{sonnet-k5,opus}.checkpoint.jsonl`). Each attempt carries the visible-test
gate verdict, which the router can see, and the hidden-test oracle verdict, which it cannot. Margin
0.05, alpha 0.05, 300 replays.

| incumbent → challenger | gate gap | oracle gap | cost ratio |
|---|---|---|---|
| opus-4.8 → sonnet-5 | −0.001 | +0.010 | 1.87 |
| opus-4.8 → haiku-4.5 | +0.151 | +0.160 | 0.18 |
| sonnet-5 → haiku-4.5 | +0.152 | +0.150 | 0.10 |

| incumbent → challenger | mode | metric | certified | halted | median requests |
|---|---|---|---|---|---|
| opus-4.8 → sonnet-5 | **paired** | gate | 1.00 | 0.00 | **127** |
| opus-4.8 → sonnet-5 | paired | oracle | 1.00 | 0.00 | 242 |
| opus-4.8 → sonnet-5 | live | gate | 1.00 | 0.00 | 425 |
| opus-4.8 → sonnet-5 | live | oracle | 1.00 | 0.00 | 957 |
| opus-4.8 → haiku-4.5 | **paired** | gate | 0.00 | 1.00 | **76** |
| opus-4.8 → haiku-4.5 | live | gate | 0.00 | 1.00 | 229 |
| sonnet-5 → haiku-4.5 | **paired** | gate | 0.00 | 1.00 | **76** |
| sonnet-5 → haiku-4.5 | live | gate | 0.00 | 1.00 | 222 |

**Reading.**
- **Paired is 3–4× faster.** Running both models on the same request cancels task difficulty,
  which is most of the variance.
- **On this workload the gate reached the oracle's decision every time.** The gate is visible
  tests and the oracle is hidden tests; their gaps agree to within a point. This holds only where
  the gate is a real check. Paired mode refuses a route whose only gate is `non-empty`, because
  certifying "passes a check that everything passes" is meaningless.
- **Certified is not adopt.** sonnet-5 is proven non-inferior to opus-4.8 but costs 1.87× more, so
  the report says to keep the incumbent. The cost check is what stops a "proven as good" model that
  costs more from being recommended.

## What this does not show

- **No live serving yet.** All of it comes from recorded runs. One recorded run per model stands in
  for that model's distribution, and run-to-run noise is not modelled.
- **Prices are the recorded ones.** Your contract prices change the cost ratio, not the quality
  decision.
- **Paired mode certifies gate-pass non-inferiority.** That matches outcomes only as well as your
  gate does. On MBPP the gate is good; on a route gated only by a judge, measure the judge first.
