# Certified model selection, live: sonnet-5 vs haiku-4-5 in paired mode

date: 2026-10-31, APPS extended 2026-11-01 · spend: ~$29.70 real Anthropic calls (all runs below,
including the invalid one) · binary: firstpass-proxy at main (fixes #244–#246, #248 as noted per run) · harness:
[`scripts/live-paired/`](../../scripts/live-paired/) · data fetched at run time, not committed.

This is the first live run of ADR 0014 paired mode: the real proxy, real Anthropic models, and gates
that execute each task's tests in a network-less container. The incumbent ladder is
`anthropic/claude-sonnet-5`; the challenger is `anthropic/claude-haiku-4-5`, run off-path on every
request (`sample_rate = 1.0`). Defaults: `margin = 0.05`, `alpha = 0.05`.

## Results

| Workload | Pairs | sonnet-5 gate pass | haiku-4-5 gate pass | haiku $/req ÷ sonnet $/req | Decision |
|---|---:|---:|---:|---:|---|
| MBPP, all 974 (tests shown in prompt) | 974 | 95.4% [93.9, 96.5] | 84.6% [82.2, 86.7] | 0.30 | **Halted** at pair 137: haiku is worse by more than the margin |
| APPS, 290 interview + competition (hidden stdin/stdout tests) | 290 | 65.2% | 60.7% | 0.09 | **Undecided**: a 4.5-point gap, inside the margin but not yet provably so |

Brackets are 95% Wilson intervals. Gate pass is the task's own tests passing, so on this workload it is
task success, not a proxy for it.

Cost ratios are per request from the receipts. The APPS ratio was first published as 0.21. That
divided sonnet's spend by 150 requests, of which 33 were zero-cost timeouts, and left out their
reruns. With every problem answered it is 0.09 ($0.00615 vs $0.0668), because sonnet-5's APPS
answers are long.

**MBPP.** The router refused the cheaper model, and the refusal is robust: replaying the 974 stored
pairs through `firstpass experiment` in 100 random orders halts every time. Haiku's failures were
inspected by hand (80 tasks re-sent with haiku as the incumbent, answers captured): they are wrong
code, not harness artefacts. For example, `odd_bit_set_number` ORs in a 32-bit mask, and
`find_missing` uses the wrong arithmetic-series formula. Several answers paste the failing asserts in
verbatim.

**APPS.** Undecided is the correct answer. The first 150 problems gave 64.7% vs 58.0%. A second
batch of 140 (problems 151–290 of the same seeded draw, run on the #248 binary) gave 65.7% vs 63.6%,
with 0 errors. Pooled over 290, 200 random orders end Undecided 194 times, Certified 5 times and
Halted once. Both decided outcomes fall inside alpha. The gap is now below the margin, but 290 pairs
cannot yet certify it. The live stream of the first batch did halt at pair 37, but that halt came
from a bug, described next. Replaying the pooled store in arrival order still shows that halt, and
`firstpass experiment` now flags why: `incumbent: 33 errors`.

## What the run caught in the product

The point of a live run is to find what replays cannot. This one found three defects. Each is fixed,
tested, and merged before any number above was taken.

1. **#244: a paired challenger was graded on "served", not on "passed".** The router serves a best
   attempt when every rung fails its gates. Paired mode counted any served answer as a pass, so a
   challenger that never passed a gate could be certified. The first MBPP run reported haiku at
   974/974 and recommended *adopt* at pair 51. That result is invalid and appears here only as the
   reason for the fix. Shadow scoring had the same bug.
2. **#245: a halted challenger kept running.** After halting at pair 137, paired mode kept paying for
   the challenger on every request.
3. **#246: the enforce path timed out at 120 s.** sonnet-5 at `max_tokens = 12000` needs up to 148 s on
   long APPS problems. The 33 longest problems returned 502s and their pairs were dropped. On those 33
   problems sonnet-5 passed 3 and haiku passed 15, so dropping them is not random. That is what produced
   the live halt at pair 37. Rerun on the fixed binary, all 33 complete. The pooled 150 pairs are what
   the table reports.

## Caveats

- **Output caps decide results.** At `max_tokens = 1024`, 35 of sonnet-5's MBPP answers were
  truncated. At 4096, 14 of 974 still were. At 12000 on APPS, 26 of the first 150 still were (25 of
  them among the 33 longest problems), and 30 of the second 140, against 0 for haiku in that batch. The runs above use 4096 on MBPP and 12000 on APPS for both arms. A cap is a property of the workload,
  and it can favour the terser model.
- **Dropped pairs and cut-off answers.** When the incumbent errors, the pair is skipped. Since this
  run, each experiment reports per-arm `health` (errors and answers cut off at `max_tokens`), the
  panel warns when either is non-zero, and a challenger error counts as a failed pair rather than a
  skipped one.
- **APPS sample size.** 290 problems (221 interview, 69 competition) of a seeded 300-problem draw;
  the second batch stopped at its spend cap after 140. The $30 budget is spent. At a 4.5-point gap a
  decision needs a much larger sample.
- **Same vendor, one pair of models.** This measures the selection mechanism on one model pair, not
  the market.

## Reproduce

```bash
python3 scripts/live-paired/fetch_apps.py data/apps                     # APPS sample (seeded)
curl -sfLo data/mbpp.jsonl https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
# route: ladder = ["anthropic/claude-sonnet-5"], gates = ["tests"],
# [route.challenger] ladder = ["anthropic/claude-haiku-4-5"], mode = "paired", max_usd_per_day = 8
# [[gate]] id = "tests", cmd = ["python3", "scripts/live-paired/gate_mbpp.py"]
#   (APPS: cmd = ["python3", "scripts/live-paired/gate_apps.py", "data/apps/tests.json"], timeout_ms = 180000)
FP_BASE=http://127.0.0.1:8080 FP_CAP_USD=5 FP_MAX_TOKENS=4096 \
  python3 scripts/live-paired/drive.py data/mbpp.jsonl mbpp.out
```

`drive.py` stops when incumbent plus challenger spend reaches `FP_CAP_USD`. Both gates run the
candidate under `docker run --network none --read-only --cap-drop ALL` with memory and pid limits.
