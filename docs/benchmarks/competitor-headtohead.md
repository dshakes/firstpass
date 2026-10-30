# Head-to-head vs competitor routing mechanisms — MBPP-974, offline replay

date: 2026-10-28 · spend: $0 (replay of committed checkpoints) · commit: 6acfde5+competitors.rs
command: `firstpass-bench --replay <checkpoint.jsonl>` (section "Head-to-head")

Each competitor is represented by its **decision rule**, replayed over the same measured candidates,
prices and hidden-oracle ground truth — not by calling the vendor. That isolates the routing decision
from each vendor's prompt classifier. It also means the predictive row is a *perfect-foresight upper bound*:
no shipped predictive router knows correctness before generating.

Caveats: (1) MBPP is code, where an executable gate exists — the case most favourable to Firstpass.
The judge arms are the relevant comparison where tests do not exist. (2) The k5 checkpoint bills five
haiku judge calls into every rung visit, for every arm including test-only first-pass, which handicaps
first-pass rather than the competitors. That is why first-pass costs more than always-top on that ladder.
(3) Switchyard's escalation router judges multi-turn session progress; single-shot MBPP exercises only
its per-turn judge decision. (4) The gate and the grader overlap: the gate runs MBPP's first two asserts,
the hidden oracle runs all three, and the first two are also shown in the prompt. So the gate can never
reject a correct answer (0 of 1948 rung evaluations), and its precision (95.9% / 97.1%) is partly
self-grading. A grader independent of the gate, such as EvalPlus-style extra tests, would likely lower it.
That has not been measured.
(5) **Always-top beats first-pass on success here**, by 1.0pp on both ladders (rows below). On the opus ladder
first-pass is much cheaper; on k5 it is not (see 2). The competitor arms lose to always-top too.

Audit (2026-10-29): an independent, fresh-context re-derivation reproduced the +3.3pp and +5.1pp rows.
They survive threshold tuning: the best judge threshold reaches 0.894, still Δ +0.029 [+0.014, +0.043].
It added the always-top rows and caveat (4).

### `mbpp-974-opus.checkpoint.jsonl`

n = 974 tasks · every arm sees identical candidates, prices and ground truth

| arm | stands in for | success | $/success | Δ success (fp − arm) | Δ $/task (fp − arm) |
|---|---|---|---|---|---|
| **first-pass** | Firstpass (executed-test gate) | 0.9333 | $0.00330 | — | — |
| oracle-predictive (upper bound) | Not Diamond · Martian · OpenRouter Auto · Foundry · Bedrock IPR | 0.9497 | $0.00250 | -0.016 [-0.025, -0.009] | +0.00070 [+0.00051, +0.00093] |
| always-top (ungated) | the no-router baseline | 0.9435 | $0.00793 | -0.0103 [-0.0195, -0.0010] | -0.00440 [-0.00468, -0.00413] |

First-pass closes 86.3% of the always-top → perfect-predictive-router cost gap. The oracle row is an upper bound on every predictive router: it knows each answer's correctness before generating it.


### `mbpp-974-sonnet-k5.checkpoint.jsonl`

n = 974 tasks · every arm sees identical candidates, prices and ground truth

| arm | stands in for | success | $/success | Δ success (fp − arm) | Δ $/task (fp − arm) |
|---|---|---|---|---|---|
| **first-pass** | Firstpass (executed-test gate) | 0.9230 | $0.01697 | — | — |
| oracle-predictive (upper bound) | Not Diamond · Martian · OpenRouter Auto · Foundry · Bedrock IPR | 0.9456 | $0.01289 | -0.023 [-0.032, -0.013] | +0.00347 [+0.00280, +0.00416] |
| always-top (ungated) | the no-router baseline | 0.9333 | $0.01500 | -0.0103 [-0.0226, +0.0010] | +0.00166 [+0.00091, +0.00241] |
| judge-escalate | NVIDIA NeMo Switchyard escalation router | 0.8901 | $0.01970 | +0.033 [+0.018, +0.048] | -0.00187 [-0.00256, -0.00117] |
| conformal-cascade (α=0.10) | Conformal Cascade, arXiv 2607.25018 | 0.8717 | $0.01763 | +0.051 [+0.036, +0.068] | +0.00029 [-0.00033, +0.00091] |

First-pass costs MORE than always-top on this ladder, so it closes none of the always-top → perfect-predictive-router cost gap. The oracle row is an upper bound on every predictive router: it knows each answer's correctness before generating it. Conformal thresholds per fold: [0.2, 0.4].


## Agent sessions — SWE-bench Verified, 60 repo-stratified instances, 459 turns

Command: `python3 scripts/analyze-swe-checkpoint.py docs/benchmarks/swebench-60-stratified.checkpoint.jsonl`
(section "Head-to-head"). $0: replay of the recorded $51.59 run. Every policy is scored on the same recorded
turns (each turn measured at both rungs, conversation history held as recorded). Intervals are a paired
bootstrap that resamples whole instances, because turns of one session are not independent.

| arm | correct turns | spend | $/correct | Δ rate (fp − arm) 95% CI | Δ $ (fp − arm) 95% CI |
|---|---|---|---|---|---|
| **first-pass** (gate per turn) | **40/459** | $48.70 | $1.217 | — | — |
| turn-index rule, ≥1 (labelled "stage router") | 35/459 | $43.26 | $1.236 | **[+0.0022, +0.0218]** | [+2.70, +8.06] |
| turn-index rule, ≥3 (labelled "stage router") | 28/459 | $30.72 | $1.097 | **[+0.0109, +0.0436]** | [+14.35, +21.85] |
| always-top (ungated) | 38/459 | $48.18 | $1.268 | [+0.0000, +0.0109] | [−1.64, +2.34] |

Reading it:
- **Correction (2026-10-29 audit): these rows are not Switchyard.** The recorded `tool_errors` counts
  failed earlier attempts in a 6-turn window (`agentic.rs:161`). On SWE-bench every earlier attempt
  failed, so it equals min(turn index, 6) on all 459 turns. The rule is effectively "cheap on turn 1,
  top after", not a session-progress signal.
- Against that turn-index rule: ≥3 serves fewer correct turns robustly (9 instances favour first-pass,
  0 favour the rule, sign-flip p=0.002). ≥1 is fragile: 5 turns, one per instance, sign-flip
  p=0.031 one-sided / 0.062 two-sided. Don't cite it as significant.
- First-pass ≈ always-top here: they differ on 2 turns, and first-pass costs slightly more. SWE shows no cost win.
- **The gate runs the benchmark's answer-key tests** (FAIL_TO_PASS plus same-file PASS_TO_PASS;
  `swebench-gate-precision.txt`). A real customer does not have those at serve time. Even so,
  top-rung gate precision is 40%.
- The solver is weak (haiku solves 3/459 turns), so the cheap rung saves little here. The perfect-foresight
  bound would spend $6.74 for the same 40 correct turns. Almost all of that saving comes from **not attempting
  turns neither rung solves (419/459, ~91%)**, not from choosing a model. For agent sessions the open lever is
  recognising hopeless turns, not model selection.
- Limits: n=60 instances; Switchyard's real escalation router also uses an LLM judge over session progress,
  which these recorded turns do not contain; the stage router is replayed as a rule, not by calling Switchyard.

### Stop escalating a hopeless session (2026-10-29)

Same command, section "Stop escalating a hopeless session". The rule: once the **top** rung's gate has failed
on `k` consecutive escalations in a session, serve a failed cheap rung as-is for the rest of the session.
It only uses what a live router sees: the top gate is observed only on turns that actually escalated. `k` is
fitted on one half of the instances (the cheapest `k` that loses no correct turn there) and scored on the
other half by index parity, so the reported number never sees the data its threshold came from. Fitted:
`k=6` and `k=5`.

| arm | correct turns | spend | $/correct |
|---|---|---|---|
| first-pass | 40/459 | $48.70 | $1.217 |
| first-pass + stop-after-k (cross-fit) | 38/459 | $41.58 | $1.094 |

Paired instance bootstrap, stop − first-pass: Δcorrect [−5, 0], **Δ$ [−$9.98, −$4.53]**, Δ$/correct
[−$0.310, +$0.003].

Verdict: **not shipped as a default.** It cuts spend significantly (about 15%), but $/correct does not
clear 0, and it gives up 2 correct turns (not significant at n=60). It also replays a counterfactual
optimistically: a turn served cheap instead of escalated would change the rest of the real conversation,
which recorded turns can't show. Re-test on a larger agent run before turning it on. The in-sample sweep
(k=4: 37 correct at $0.995/correct) is selection on the test set and is not the result.

### Anytime-valid session controller (2026-10-30)

Same command; row "e-process stop". Escalation pauses once a betting e-process rejects "the top gate's pass
rate in this session ≥ 0.25" at δ=0.2 (bet 0.9). It probes every 3rd paused turn. Constants were fixed
before the run. Result: 40/459 correct (no loss), $46.11 vs $48.70 (−5%). It only ever fires on a session's
8th, final turn: at δ=0.2 it needs 7 consecutive top-gate failures, and these sessions are capped at 8 turns.
The guarantee holds, but on this data the effect is small. Real agentic evidence at n=500 is in
[agentic-real-runs.md](agentic-real-runs.md).
