# Agentic multi-turn MBPP — what this run can and cannot establish

Written **during** the run, before the verdict was known, so the interpretation is fixed in advance
rather than fitted to whatever number came out.

## The claim under test

ADR 0012 added a trajectory hint: signals read from the agent's own conversation (failing tool
results, repeated attempts, conversation depth) used to pick a **start rung**. The pre-registered
bar in `multiturn::PreRegistered` is:

1. **≥5% improvement in $/success**, and
2. **paired quality CI lower bound ≥ −0.01** — judged on the bound, not the point estimate.

Miss either and the feature does not ship.

## Structural limits discovered during the run

These are properties of the harness and the workload, not of the router, and they were measured on
the first ~120 tasks:

| observation | value | consequence |
|---|---|---|
| tasks going multi-turn | ~15% | 85% of tasks produce one turn with no trajectory at all |
| turns carrying any signal | ~20% | the other 80% score `None`, where both policies decide identically |
| **maximum conversation depth observed** | **2** | `deep` needs ≥8, so `DifficultyHint::High` is **unreachable in this harness** |
| hint levels actually produced | only `None` and `Medium` | `Low` never occurs — every recorded tool result is a failure, because the loop only continues when the cheap rung fails |

The depth ceiling is the load-bearing one and it is a direct consequence of the loop design: the
conversation continues only while the cheap rung is still failing, and `max_turns = 3`. An MBPP task
is one function; a model that cannot write it in three attempts is rare, and one that needs eight
does not exist in this dataset.

## What follows, stated before seeing the result

- **A "ships" verdict here does not validate the feature.** It would rest on the ~20% of turns
  scoring `Medium`, with the remaining 80% contributing exactly zero paired difference. That
  inflates apparent precision: a CI computed mostly over identical decisions is narrow because the
  policies agree, not because the effect is well-measured.
- **A "killed" verdict here does not refute the feature either.** Two of four difficulty levels
  never occur, so the mechanism is only partly exercised.
- **The honest reading of either outcome is the same**: *MBPP is too easy to test trajectory
  routing.* The workload where this feature should pay — long agentic sessions that grind, retry,
  and drift — is not what a single-function benchmark produces.

## What would actually test it

A workload with genuine depth: SWE-bench-style repository tasks (explore → edit → test → retry),
where eight-plus turns are ordinary rather than pathological. ADR 0010 notes the SWE-bench harness
exists but is not wired to the CLI; wiring it is the prerequisite for a real answer here.

Reporting this run's number regardless, labelled for what it is, because a benchmark that only
publishes its favourable configurations is not evidence.
