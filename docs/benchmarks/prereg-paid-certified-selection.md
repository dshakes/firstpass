# Pre-registration: paid ADR 0014 certified selection, one fixed challenger

Status: **registered, not run** · date: 2026-11-04 · everything below is fixed before any spend. Amendments after the first paid call are not allowed; a change is a new pre-registration.

## Why one challenger

The $0 replay ([agent-paired-replay.txt](agent-paired-replay.txt)) returned STOP on three datasets. On SWE-bench, 10 challengers at alpha/10 = 0.005 halted or starved the single winner (certified in 10% of orders). Alone at alpha = 0.05 the best candidates adopt in 56% of orders (112 and 115 of 200) inside the 500 recorded sessions. So the fix is a list of length 1, not more tasks per challenger across ten. alpha stays 0.05.

## Challenger and incumbent

| | model | why |
|---|---|---|
| incumbent (top) | `anthropic/claude-sonnet-5` ($3 / $15 per Mtok) | the incumbent in the 2026-10-31 live run ([live-certified-selection.md](live-certified-selection.md)), so the harness, gates and sonnet cost per APPS request ($0.0668) are already measured |
| challenger | `google/gemini-3.6-flash` ($1.5 / $7.5 per Mtok) | see below |

Replay evidence against opus-4.5 on SWE-bench (n=500, paired, recomputed from `~/.cache/firstpass-swebench-experiments`):

| cheap model | resolved vs top 0.768 | gap | $/session vs $0.754 | adopts alone at alpha 0.05 | in the price table |
|---|---|---|---|---|---|
| minimax-2.5 | 0.758 | 0.010 | 0.073 | 112/200 | no (needs `[[price]]`, price unsourced here) |
| **gemini-3-flash** | 0.758 | 0.010 | 0.356 | 115/200 | **yes** (`google/gemini-3.6-flash`) |
| claude-4-6-opus | 0.756 | 0.012 | 0.552 | 77/200 | n/a |
| claude-haiku-4.5 | 0.666 | 0.102 | 0.331 | 0/200 (halts) | yes; also halted live on MBPP (84.6 vs 95.4) |
| kimi-k2.5, gpt-5-mini | 0.708, 0.562 | 0.06, 0.21 | | halt or starve | |

minimax-2.5 and gemini-3-flash tie on quality evidence. Gemini is chosen because its price is in the repo's table, so the spend ceiling below is computable and `$/request` is checkable from receipts without inventing a price. Minimax is cheaper per session and is the better follow-up if this one certifies. Caveat stated up front: the evidence is SWE-bench agent runs with a different top model and `gemini-3-flash-high`, not `gemini-3.6-flash` on APPS. This run tests that transfer; it does not assume it.

## Dataset

APPS, interview + competition, hidden stdin/stdout tests, from `scripts/live-paired/fetch_apps.py` with a seeded draw of **800** problems (seed 20261031; the script took a fixed 300 until this branch added an optional `N` argument). APPS rather than MBPP: sonnet's APPS pass rate (65.2%) is close to the SWE-bench rate (0.768) the variance below was measured on, and it is not ceiling-limited (sonnet passes 95.4% of MBPP). Gate = the task's own tests in a network-less container, so gate pass is task success. Caps: `max_tokens = 12000` both arms (the live run showed 4096 truncates sonnet).

## Power (the math)

Test: ADR 0014 betting e-processes on `D = X_inc - X_ch`, margin m = 0.05, alpha = 0.05 (certify at wealth >= 20, halt at wealth >= 20), paired mode so every task gives one pair.

Variance of `D` from the SWE-bench pairs (gemini-3-flash vs opus-4.5): P(top solves, challenger fails) = 0.058, P(challenger solves, top fails) = 0.048.
`Var(D) = 0.058 + 0.048 - (0.058 - 0.048)^2 = 0.106`, `sd(D) = 0.325` (minimax: 0.332). Pairing is what makes this small; two unpaired arms give about 0.6 (ADR 0014).

Fixed-sample reference: to reject "gap >= 0.05" when the true gap is 0.01, one-sided alpha 0.05, 80% power: `n = ((1.645 + 0.842) * 0.325 / (0.05 - 0.01))^2 = 408`. A sequential test that may be read at any time pays for that, so the number that matters is simulated. Simulation (scratch script, not committed; same lambda grid and stopping rule as `crates/firstpass-core/src/select.rs`, 2000 reps per row; the n=500 row reproduces the replay's 56%):

| true gap (top minus challenger) | P(certify by n=300) | n=400 | n=500 | n=600 | **n=800** | P(halt) |
|---|---|---|---|---|---|---|
| 0.010 (observed, discordance 0.106) | 0.37 | 0.47 | 0.56 | 0.66 | **0.79** | 0.0005 |
| 0.020 (same discordance) | 0.21 | 0.28 | 0.34 | 0.39 | 0.50 | 0.002 |
| 0.030 | 0.11 | 0.14 | 0.17 | 0.19 | 0.24 | 0.004 |
| 0.050 (null boundary, certifying is the error) | 0.03 | 0.03 | 0.03 | 0.03 | 0.035 | 0.035 |
| 0.100 | 0 | 0 | 0 | 0 | 0 | 0.97 (median 51 pairs) |

So **N = 800 pairs gives about 79% power if the challenger is within 1 point of sonnet**, 5% type-I (3.5% simulated at the boundary), and a clear halt if it is 10 points worse. The earlier ~300-400 estimate gives 37-47% power: too low to be worth paying for. This run is deliberately **not** powered for gaps of 2-3 points; those end undecided, and that is a stated outcome, not a failure.

## Decision rule (fixed)

Read after every pair; no peeking penalty (Ville). Stop at the first of: certified, halted, 800 valid pairs, or the spend cap.

- **PROCEED** (adopt gemini-3.6-flash for APPS-like single-turn code routes, by a human): status `certified` AND challenger `$/request` <= 0.9 x sonnet `$/request` from receipts AND per-arm `health` shows < 2% errors and no arm with > 5% answers cut off at `max_tokens` more than the other.
- **STOP** otherwise: `halted`; or `undecided` at 800 pairs; or `certified` but cost ratio > 0.9; or the spend cap reached first; or health bars missed. No extension, no second look at a subset, no switching to MBPP. `undecided` means "not shown within 5 points at this n", not "equivalent".
- Certified means gate-pass non-inferiority on this workload, not outcome non-inferiority elsewhere (ADR 0014 "What it does not claim").

## Spend ceiling

Per-pair cost from measured data: sonnet APPS `$0.0668`/request (receipts, 2026-11-01, `max_tokens` 12000). Implied output about 4.2k tokens (assuming ~1k input: `(0.0668 - 0.0015) / 15e-6`). Gemini at the same token counts and the table price: `1k x 1.5e-6 + 4.2k x 7.5e-6 = $0.033`. Gemini thinking tokens may raise that; the cap covers it.

| | USD |
|---|---|
| expected, all 800 pairs: `800 x (0.0668 + 0.033)` | 79.8 |
| smoke test first (20 pairs, abort if per-pair > $0.20) | <= 3 |
| **hard ceiling** (`FP_CAP_USD`, incumbent + challenger) | **120** |
| challenger-only daily cap (`max_usd_per_day`) | 50 |

Expected spend is lower: the median certify time is about 370 pairs (about $37). The ceiling is a stop, not a bound: at `max_tokens` 12000 a single pair could in principle cost $0.28, so a pathological run stops at $120 and is scored STOP.

## Exact commands

Constraint, stated plainly: **`firstpass-bench` has no live paired mode.** Its flags are `--paired-replay` ($0, recorded outcomes), `--live`, `--coding-live`, `--agentic-multiturn`, `--swe-agentic` and others; none runs ADR 0014 paired selection against a live proxy. The live path is `firstpass-proxy` plus `scripts/live-paired/drive.py` (as in the 2026-10-31 run). The one gap found was the fixed 300-task sample in `fetch_apps.py`, fixed here with an optional `N`.

```bash
# 1. data (seeded; file keeps the legacy name apps300.jsonl but holds 800 tasks)
python3 scripts/live-paired/fetch_apps.py data/apps 800

# 2. firstpass.toml  (a google provider block is required; confirm with `firstpass doctor` before spending: UNVERIFIED field values)
#    [[provider]]  id = "google"  base_url = "https://generativelanguage.googleapis.com"  dialect = "gemini"  api_key_env = "GEMINI_API_KEY"
#    [[gate]]      id = "tests"   cmd = ["python3", "scripts/live-paired/gate_apps.py", "data/apps/tests.json"]  timeout_ms = 180000
#    [[route]]     mode = "enforce"  ladder = ["anthropic/claude-sonnet-5"]  gates = ["tests"]
#    [route.challenger]  ladder = ["google/gemini-3.6-flash"]  mode = "paired"  margin = 0.05  alpha = 0.05  sample_rate = 1.0  max_usd_per_day = 50
FIRSTPASS_CONFIG=./firstpass.toml firstpass doctor
FIRSTPASS_CONFIG=./firstpass.toml firstpass-proxy &

# 3. smoke: 20 pairs, check $/pair <= 0.20 in receipts, then continue
head -20 data/apps/apps300.jsonl > data/apps/smoke.jsonl
FP_BASE=http://127.0.0.1:8080 FP_CAP_USD=3 FP_MAX_TOKENS=12000 python3 scripts/live-paired/drive.py data/apps/smoke.jsonl smoke.out

# 4. the run
FP_BASE=http://127.0.0.1:8080 FP_CAP_USD=120 FP_MAX_TOKENS=12000 python3 scripts/live-paired/drive.py data/apps/apps300.jsonl apps800.out

# 5. read the decision
firstpass experiment
```

Needs `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` in the proxy environment, Docker for the gate, and the human's spend approval for the $120 ceiling before step 3. The smoke pairs count toward the 800 (same seeded order); they are not a separate sample.

## Known limits (before the data)

- Variance and the 0.01 gap come from SWE-bench; APPS discordance for this pair is unmeasured. If it is higher, power at n=800 falls (not simulated). The first 300 pairs will show it; the rule does not change.
- One vendor pair, one workload, one run per model per task. Gemini `max_tokens` behaviour with thinking is unmeasured; the cut-off health bar guards it.
- Spend figures use list prices from `crates/firstpass-core/src/cost.rs`; the provider bill is the real number and goes in the result.
