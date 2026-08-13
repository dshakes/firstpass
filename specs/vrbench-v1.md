# VRBench v1 — a benchmark for **verification** routers

**Status:** draft v1. Breaking changes bump the major version; the format carries `vrbench_version`
so a harness can refuse data it does not understand.

## Why this exists

The field's benchmarks evaluate *pre-judgment* routers — systems that read a prompt and choose a
model before generation. RouterBench, RouterArena, RouterEval, RouterXBench and VL-RouterBench all
share that shape. A **verification router** — one that generates, checks the real output, and
escalates only when the check fails — cannot be scored by any of them, and the reasons are
structural rather than incidental:

- **No executable oracle.** RouterBench ships one stored response per model with a quality score.
  A verification router needs to *run a real check*, and the only signal available is the ground
  truth itself. Gate and oracle collapse into the same measurement, which measures nothing.
- **No cost field the router controls.** RouterArena's submission is a per-query model choice, and
  the harness derives cost from the named model. A cascade that escalates has paid for the rejected
  attempt too, and there is nowhere to say so. Reporting only the serving model under-states cost
  on exactly the queries the router found hard.
- **Robustness defined against verification.** RouterArena scores model-selection stability under
  prompt perturbation. A verification router's selection is a function of the *output*, which
  legitimately changes when the prompt does — so correct behaviour scores as fragility.

VRBench fixes exactly these three things and changes nothing else. It is **complementary to
RouterBench, not a replacement**: pre-judgment routers should still be scored there, and VRBench
reports RouterBench's AIQ metric verbatim (§5) so results are comparable across both.

## 1. The central property: the gate is not the oracle

Every task carries **two disjoint check sets**:

| set | who may see it | what it is for |
|---|---|---|
| **visible** | the router, freely, at any time | the *gate* — what a real operator would write |
| **hidden** | the harness only, never the router | the *oracle* — ground truth for scoring |

This split is the entire scientific content of the benchmark. It is what makes gate **error**
measurable: a gate can pass an answer the oracle fails (false accept, a wrong answer served) or
fail an answer the oracle passes (false reject, a needless escalation). A benchmark with only one
check set can measure neither, and a router evaluated on it can trivially score perfectly by
treating the check as ground truth.

**Rule 1.** A submission that reads, infers, or is trained against hidden cases is invalid. Harness
implementations MUST NOT expose them to router code; reference harnesses execute hidden cases in a
separate sandbox invocation after the router has committed to an answer.

**Rule 2.** The visible set MUST be a plausible operator-authored suite, not a weakened copy of the
hidden set. Deriving visible cases by sampling from the hidden ones produces an unrealistically
well-calibrated gate and inflates every verification router's score.

## 2. Task format

JSONL, one task per line.

```json
{
  "vrbench_version": 1,
  "id": "mbpp/602",
  "domain": "code/python",
  "prompt": "Write a function to find the first repeated character in a given string.",
  "visible": {
    "kind": "pytest",
    "cases": ["assert first_repeated_char('abcabc') == 'a'"],
    "entrypoint": "first_repeated_char"
  },
  "hidden": {
    "kind": "pytest",
    "cases": ["assert first_repeated_char('abc') is None", "..."]
  },
  "difficulty": "medium"
}
```

| field | required | meaning |
|---|---|---|
| `vrbench_version` | yes | format version; harness refuses unknown majors |
| `id` | yes | stable, globally unique, namespaced by source dataset |
| `domain` | yes | `code/python`, `code/js`, `math`, `structured/json`, … |
| `prompt` | yes | verbatim text given to the model |
| `visible` | yes | the gate's check set — router may read this |
| `hidden` | yes | the oracle's check set — router MUST NOT read this |
| `difficulty` | no | free-form; used only for slicing results |

`kind` selects the executor (`pytest`, `jest`, `cargo-test`, `json-schema`, `exact-match`). A
harness MUST fail loudly on a `kind` it does not implement rather than skipping the task — a
silently skipped task is a silently inflated score.

## 3. Submission format

JSONL, one line per task. **This is where VRBench differs most from existing benchmarks.**

```json
{
  "vrbench_version": 1,
  "id": "mbpp/602",
  "answer": "def first_repeated_char(s):\n    ...",
  "cost_usd": 0.00412,
  "attempts": [
    {"model": "anthropic/claude-haiku-4-5", "cost_usd": 0.00104, "gate_verdict": "fail"},
    {"model": "anthropic/claude-opus-4-8", "cost_usd": 0.00308, "gate_verdict": "pass"}
  ],
  "latency_ms": 3120
}
```

**`cost_usd` is reported by the router, and MUST include every attempt — including those whose
output was discarded.** This is the field RouterArena lacks and the reason a cascade cannot submit
there honestly.

**Rule 3.** `cost_usd` MUST equal the sum of `attempts[].cost_usd`. A harness MUST reject a
submission where it does not, rather than silently preferring one. A router that pays for three
generations and reports one is not making an approximation; it is misreporting, and the arithmetic
check is what makes that detectable by a third party.

**Rule 4.** `attempts` MUST be present and non-empty even for single-shot routers, which simply
report one attempt. This keeps cascades and non-cascades on identical footing and makes escalation
rate a derived quantity rather than a self-declared one.

Prices come from a published table (`prices.json`, versioned alongside the tasks) so that two
submissions are priced identically. A router using a model absent from the table MUST supply a
`[[price]]` entry; **there is no silent default**, because a missing price would silently record a
free model.

## 4. What the harness does

1. Refuses tasks and submissions whose `vrbench_version` major it does not implement.
2. Validates every submission line against Rules 3–4 before scoring anything.
3. Runs the **hidden** set against each `answer` in an isolated, network-free sandbox.
4. Emits per-task outcomes and the aggregate metrics of §5.

The sandbox is not optional. Answers are model-generated code from an untrusted source; a harness
that executes them on the host is a remote-code-execution vector wearing an evaluation costume.

## 5. Metrics

Reported together. Every one is computable from the submission plus the hidden-set outcomes.

| metric | definition |
|---|---|
| **success** | fraction of tasks whose served answer passes the full hidden set |
| **$/success** | total `cost_usd` ÷ successful tasks |
| **served-failure** | fraction of *served* answers that fail the oracle |
| **gate false-accept** | of answers the router's gate passed, fraction the oracle fails |
| **gate false-reject** | of answers the router's gate failed, fraction the oracle passes |
| **escalation rate** | fraction of tasks with more than one attempt |
| **AIQ** | RouterBench's Average Improvement in Quality over the non-decreasing convex hull, computed exactly as arXiv:2403.12031 defines it |

AIQ is included deliberately and unmodified. It makes VRBench results directly comparable to
RouterBench's, and it carries their **Zero Router** bar — the parameter-free probabilistic mix of
the raw models — which is the floor a router must clear to have accomplished anything. RouterBench
reports no learned router significantly clearing it; that is the reference point, not ours.

The two gate-error metrics have no counterpart in any existing router benchmark, and are the point
of the visible/hidden split.

## 6. Rules for a valid submission

1. **Evaluation-only.** No training, fitting, or threshold tuning on VRBench tasks. Calibrate on
   your own data; report on these.
2. **Hidden cases are never read.** See Rule 1.
3. **Costs include discarded attempts.** See Rule 3.
4. **Deterministic replay.** Submissions SHOULD include enough per-attempt detail to recompute
   every metric offline without re-running any model. Third parties should not need API keys or a
   budget to check a published number.

Rule 4 is not decoration. A result nobody can recompute is a claim, not a measurement.

## 7. Open questions for v2

- **Non-executable domains.** Prose has no oracle short of human judgment or an LLM judge; both are
  expensive and themselves error-prone. v1 restricts to domains with a machine-checkable answer and
  says so, rather than pretending coverage it does not have.
- **Gate realism.** Rule 2 states the requirement in prose. A quantitative test of "is this visible
  set a plausible operator suite" would be better, and we do not have one.
- **Judge-gated routers.** A router whose gate is an LLM judge pays real tokens for verification.
  §3 counts that under `attempts[].cost_usd`, but a judge call is not a generation and may deserve
  its own field.
