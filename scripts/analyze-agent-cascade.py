#!/usr/bin/env python3
"""Agentic routing on real agent runs, at $0: SWE-bench Verified, mini-SWE-agent bash-only leaderboard.

Each leaderboard system runs the same agent scaffold with a different model over all 500 instances and
publishes per-instance `{resolved, cost, api_calls}`. A whole agent run is the routing unit: a cascade
runs the cheap model's agent, gates the finished run on what a live router can see, and on failure
reruns the task from scratch with the next model. That rerun is exactly the recorded run, so unlike a
turn-level replay there is no invented conversation. The one assumption: one recorded run per model
stands in for that model's run distribution.

Gate (live-observable, no hidden tests): the cheap run's agent step count `api_calls <= tau`. An agent
that needed many steps was struggling. `tau` per rung is fitted on one half of the instances and scored
on the other (parity of the sorted instance id), so no reported number saw its own threshold.

Fitting objective (Firstpass's promise, fixed in advance): the cheapest setting whose resolved count on
the training half is at least the top model's; if none reaches it, the most resolved. Minimising
$/resolved instead would reward serving the cheap model's failures.

Arms:
  always-<model>       serve one model's run
  first-pass (steps)   cascade with the step-count gate, cross-fit
  repo-predictive      fair stand-in for a predictive router (Not Diamond/Martian-style): learns each
                       repo's per-model resolve rate on the other half and picks, before running, the
                       cheapest model whose rate is within `eps` of the best; `eps` cross-fit too
  oracle-cascade       escalate iff the run is unresolved (upper bound for ANY gate)
  oracle-predictive    cheapest model that resolves, run once (upper bound for ANY predictive router)
Not replayable from these data: a mid-run switch (Switchyard's escalation router), which needs the
partial trajectory to continue on another model.

Data: github.com/SWE-bench/experiments@PINNED evaluation/verified/<system>/per_instance_details.json,
fetched at run time into a cache dir and never committed (the repository states no licence).

Usage: analyze-agent-cascade.py [CACHE_DIR]     |     analyze-agent-cascade.py --self-check
"""

from __future__ import annotations

import json
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PINNED = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
RAW = f"https://raw.githubusercontent.com/SWE-bench/experiments/{PINNED}/evaluation/verified"

# Fixed before any routing result was computed. Same scaffold version within a ladder, cheapest first.
# Excluded: gemini-3-pro-high (resolved=false on all 500: a data defect), gpt-5-nano (costs more than
# gpt-5-mini on the same scaffold), gemini-2.5-flash (per_instance_details.json missing).
LADDERS = {
    "claude haiku→sonnet→opus (v2.0.0)": [
        "20260217_mini-v2.0.0_claude-4-5-haiku-high",
        "20260217_mini-v2.0.0_claude-4-5-sonnet-high",
        "20260217_mini-v2.0.0_claude-4-5-opus-high",
    ],
    "claude haiku→opus (v2.0.0)": [
        "20260217_mini-v2.0.0_claude-4-5-haiku-high",
        "20260217_mini-v2.0.0_claude-4-5-opus-high",
    ],
    "gpt-5-mini→gpt-5.2-high (v2.0.0)": [
        "20260217_mini-v2.0.0_gpt-5-mini",
        "20260217_mini-v2.0.0_gpt-5-2-high",
    ],
    "gpt-5-mini→gpt-5 (v1.7.0)": [
        "20250807_mini-v1.7.0_gpt-5-mini",
        "20250807_mini-v1.7.0_gpt-5",
    ],
}
TRAJ = "https://swe-bench-submissions.s3.amazonaws.com/bash-only/{s}/trajs/{i}/{i}.traj.json"

# Agreement gate (Firstpass's consistency gate, applied to finished agent runs). Fixed before any patch
# was inspected: run two independent cheap agents A and B; serve A's patch if the two agree, else run
# the top model. Parameter-free, so nothing is fitted and nothing can leak.
AGREEMENT = {
    "A=haiku, B=gpt-5-mini → opus": (
        "20260217_mini-v2.0.0_claude-4-5-haiku-high",
        "20260217_mini-v2.0.0_gpt-5-mini",
        "20260217_mini-v2.0.0_claude-4-5-opus-high",
    ),
    "A=gpt-5-mini, B=haiku → gpt-5.2-high": (
        "20260217_mini-v2.0.0_gpt-5-mini",
        "20260217_mini-v2.0.0_claude-4-5-haiku-high",
        "20260217_mini-v2.0.0_gpt-5-2-high",
    ),
    "A=gemini-3-flash, B=gpt-5-mini → opus": (
        "20260217_mini-v2.0.0_gemini-3-flash-high",
        "20260217_mini-v2.0.0_gpt-5-mini",
        "20260217_mini-v2.0.0_claude-4-5-opus-high",
    ),
}
# Workload-level model selection over every v2.0.0 system with data (fixed rule, cross-fit): on the
# training half, the cheapest model whose resolve rate is within SELECT_EPS of the best; scored on the
# other half. Excluded: gemini-3-pro-high (resolved=false on all 500), gpt-5-2-codex (no data file).
POOL = [
    f"20260217_mini-v2.0.0_{m}"
    for m in (
        "claude-4-5-haiku-high",
        "claude-4-5-sonnet-high",
        "claude-4-5-opus-high",
        "claude-4-6-opus",
        "deepseek-3-2-high",
        "gemini-3-flash-high",
        "glm-5-high",
        "gpt-5-2-high",
        "gpt-5-mini",
        "kimi-k2-5-high",
        "minimax-2-5-high",
    )
]
REFERENCE = "20260217_mini-v2.0.0_claude-4-5-opus-high"
SELECT_EPS = 0.01
# Chosen AFTER seeing per-model resolve rates (not routing outcomes): exploratory, not confirmatory.
EXPLORATORY_AGREEMENT = (
    "20260217_mini-v2.0.0_minimax-2-5-high",
    "20260217_mini-v2.0.0_kimi-k2-5-high",
    "20260217_mini-v2.0.0_claude-4-5-opus-high",
)
# Confirmatory replication of the exploratory pair on runs it was not chosen on (older v1.16-1.17
# scaffold; fixed before these files were fetched). Same roles: minimax A, kimi B, opus-4.5 top.
REPLICATION = (
    "20251124_mini-v1.17.0_minimax-m2",
    "20251210_mini-v1.17.2_kimi-k2-thinking",
    "20251124_mini-v1.16.0_claude-opus-4-5-20251101",
)
QUANTILES = [i / 10 for i in range(11)]
EPS_GRID = [0.0, 0.05, 0.1, 0.2, 0.3]

Run = tuple[bool, float, int]  # resolved, cost_usd, api_calls
Row = dict  # {"id", "repo", "runs": [Run per rung]}


def load(cache: Path, system: str) -> dict:
    path = cache / f"{system}.json"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            f"{RAW}/{system}/per_instance_details.json", timeout=60
        ) as r:
            path.write_bytes(r.read())
    return json.loads(path.read_text())


def rows_for(cache: Path, ladder: list[str]) -> list[Row]:
    data = [load(cache, s) for s in ladder]
    ids = sorted(set.intersection(*(set(d) for d in data)))
    return [
        {
            "id": i,
            "repo": i.split("__")[0],
            "runs": [
                (
                    bool(d[i]["resolved"]),
                    float(d[i]["cost"] or 0.0),
                    int(d[i]["api_calls"] or 0),
                )
                for d in data
            ],
        }
        for i in ids
    ]


def load_patches(cache: Path, system: str, ids: list[str]) -> dict[str, str]:
    """Each run's submitted patch, from its trajectory (only `info.submission` is kept; JSON only)."""
    path = cache / f"{system}.patches.json"
    if not path.exists():

        def one(i: str) -> tuple[str, str]:
            with urllib.request.urlopen(TRAJ.format(s=system, i=i), timeout=60) as r:
                return i, json.loads(r.read()).get("info", {}).get("submission") or ""

        with ThreadPoolExecutor(16) as ex:
            path.write_text(json.dumps(dict(ex.map(one, ids))))
    got = json.loads(path.read_text())
    return {
        i: (v.get("submission") or "" if isinstance(v, dict) else v)
        for i, v in got.items()
    }


_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? ", re.M)
_SCRATCH = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_test\.py$|(^|/)(reproduce|repro|debug)[^/]*$"
)


def edits(patch: str) -> dict[str, list[tuple[int, int]]]:
    """Source files the patch modifies -> original-line ranges of its hunks. New files, tests and
    reproduction scripts are dropped: agents leave those behind and they say nothing about the fix."""
    out: dict[str, list[tuple[int, int]]] = {}
    for block in re.split(r"(?=^diff --git )", patch, flags=re.M):
        m = _FILE.match(block)
        if (
            not m
            or "\nnew file mode" in block.split("@@", 1)[0]
            or _SCRATCH.search(m.group(1))
        ):
            continue
        out[m.group(1)] = [
            (int(a), int(a) + int(b or 1)) for a, b in _HUNK.findall(block)
        ]
    return out


def agree(a: str, b: str, hunks: bool, slack: int = 3) -> bool:
    """`files`: both patches modify the same non-empty set of source files. `hunks`: also every hunk
    of A overlaps (within `slack` lines) a hunk of B in the same file."""
    ea, eb = edits(a), edits(b)
    if not ea or set(ea) != set(eb):
        return False
    if not hunks:
        return True
    return all(
        any(s0 - slack <= t1 and t0 - slack <= s1 for t0, t1 in eb[f])
        for f, rs in ea.items()
        for s0, s1 in rs
    )


def select_cheapest(train: list[list[tuple[bool, float]]], eps: float) -> int:
    """Index of the cheapest model whose training resolve rate is within `eps` of the best."""
    n = len(train)
    rates = [sum(r[k][0] for r in train) / n for k in range(len(train[0]))]
    costs = [sum(r[k][1] for r in train) for k in range(len(train[0]))]
    ok = [k for k in range(len(rates)) if rates[k] >= max(rates) - eps]
    return min(ok, key=lambda k: costs[k])


def selection_report(cache: Path) -> None:
    data = [load(cache, s) for s in POOL]
    ids = sorted(set.intersection(*(set(d) for d in data)))
    res = [
        [(bool(d[i]["resolved"]), float(d[i]["cost"] or 0.0)) for d in data]
        for i in ids
    ]
    picks = [
        select_cheapest(res[1::2], SELECT_EPS),
        select_cheapest(res[0::2], SELECT_EPS),
    ]
    sel = [r[picks[j % 2]] for j, r in enumerate(res)]
    ref = [r[POOL.index(REFERENCE)] for r in res]
    n = len(ids)
    name = lambda k: POOL[k].split("v2.0.0_")[1]  # noqa: E731
    print(f"## Workload-level model selection (cross-fit, {len(POOL)} models, n={n})\n")
    print(f"picked for even half: {name(picks[0])}; for odd half: {name(picks[1])}\n")
    print(
        "| arm | resolved | $/instance | Δ resolved vs always-opus-4.5 | Δ $/inst vs always-opus-4.5 |"
    )
    print("|---|---|---|---|---|")
    for arm, outs in [
        ("selected (cross-fit)", sel),
        ("always-claude-4-5-opus-high", ref),
    ]:
        ok, cost = total(outs)
        d = "— | —"
        if outs is sel:
            (r0, r1), (c0, c1), _ = paired_ci(sel, ref)
            d = f"[{r0:+.3f}, {r1:+.3f}] | [{c0:+.3f}, {c1:+.3f}]"
        print(f"| {arm} | {ok}/{n} ({ok / n:.3f}) | ${cost / n:.3f} | {d} |")
    print()


def agreement_report(cache: Path) -> None:
    print("## Agreement gate on finished agent runs (parameter-free, no fitting)\n")
    configs = {
        **AGREEMENT,
        "EXPLORATORY A=minimax, B=kimi → opus": EXPLORATORY_AGREEMENT,
        "REPLICATION (v1.16-1.17) A=minimax-m2, B=kimi-k2-thinking → opus-4.5": REPLICATION,
    }
    for name, (sa, sb, st) in configs.items():
        data = [load(cache, x) for x in (sa, sb, st)]
        ids = sorted(set.intersection(*(set(d) for d in data)))
        pa, pb = load_patches(cache, sa, ids), load_patches(cache, sb, ids)
        res = [
            [(bool(d[i]["resolved"]), float(d[i]["cost"] or 0.0)) for d in data]
            for i in ids
        ]
        n = len(ids)
        top = [r[2] for r in res]
        print(f"### {name}  (n={n})\n")
        print(
            "| arm | resolved | $/instance | gate pass | gate precision (A resolved given pass) | Δ resolved vs always-top | Δ $/inst vs always-top |"
        )
        print("|---|---|---|---|---|---|---|")
        arms: dict[str, tuple[list[tuple[bool, float]], str, str]] = {
            "always-A": ([r[0] for r in res], "—", "—"),
            "always-top": (top, "—", "—"),
        }
        for kind in ("files", "hunks"):
            gate = [agree(pa.get(i, ""), pb.get(i, ""), kind == "hunks") for i in ids]
            outs = [
                (r[0][0], r[0][1] + r[1][1])
                if g
                else (r[2][0], r[0][1] + r[1][1] + r[2][1])
                for r, g in zip(res, gate)
            ]
            passed = sum(gate)
            prec = (
                sum(r[0][0] for r, g in zip(res, gate) if g) / passed
                if passed
                else float("nan")
            )
            arms[f"agree-{kind}(A,B) → top"] = (outs, f"{passed}/{n}", f"{prec:.3f}")
        base = sum(r[0][0] for r in res) / n
        for arm, (outs, gp, pr) in arms.items():
            ok, cost = total(outs)
            if arm == "always-top":
                d = "— | —"
            else:
                (r0, r1), (c0, c1), _ = paired_ci(outs, top)
                d = f"[{r0:+.3f}, {r1:+.3f}] | [{c0:+.3f}, {c1:+.3f}]"
            print(
                f"| {arm} | {ok}/{n} ({ok / n:.3f}) | ${cost / n:.3f} | {gp} | {pr} | {d} |"
            )
        print(f"\nbase rate (A resolved): {base:.3f}\n")


def cascade(row: Row, taus: list[float]) -> tuple[bool, float]:
    """Run rungs in order; serve the first whose step count passes its gate, or the last rung."""
    spent = 0.0
    for k, (ok, cost, steps) in enumerate(row["runs"]):
        spent += cost
        if k == len(row["runs"]) - 1 or steps <= taus[k]:
            return ok, spent
    raise AssertionError("unreachable")


def oracle_cascade(row: Row) -> tuple[bool, float]:
    spent = 0.0
    for ok, cost, _ in row["runs"]:
        spent += cost
        if ok:
            return True, spent
    return False, spent


def oracle_predictive(row: Row) -> tuple[bool, float]:
    hit = next(((ok, c) for ok, c, _ in row["runs"] if ok), None)
    return hit if hit else (False, row["runs"][0][1])


def total(outs: list[tuple[bool, float]]) -> tuple[int, float]:
    return sum(o for o, _ in outs), sum(c for _, c in outs)


def pick(train: list[Row], candidates: list, serve) -> object:
    """The cheapest candidate matching the top model's resolved count on `train`, else the most resolved."""
    bar = sum(r["runs"][-1][0] for r in train)
    scored = [
        (*total([serve(c, r) for r in train]), i) for i, c in enumerate(candidates)
    ]
    feasible = [(cost, i) for ok, cost, i in scored if ok >= bar]
    i = min(feasible)[1] if feasible else max(scored, key=lambda x: (x[0], -x[1]))[2]
    return candidates[i]


def fit_taus(train: list[Row]) -> list[float]:
    """Per-rung step thresholds over a grid of step-count quantiles (-1 = always escalate)."""
    n_gated = len(train[0]["runs"]) - 1
    grids = []
    for k in range(n_gated):
        steps = sorted(r["runs"][k][2] for r in train)
        grids.append(
            sorted(
                {
                    steps[min(int(q * (len(steps) - 1)), len(steps) - 1)]
                    for q in QUANTILES
                }
                | {-1}
            )
        )

    def combos(k: int) -> list[list[float]]:
        if k == n_gated:
            return [[]]
        return [[t, *rest] for t in grids[k] for rest in combos(k + 1)]

    return pick(train, combos(0), lambda taus, r: cascade(r, taus))


def repo_router(train: list[Row], eps: float):
    n_rungs = len(train[0]["runs"])
    stats: dict[str, list[list[int]]] = {}
    for r in train:
        s = stats.setdefault(r["repo"], [[0, 0] for _ in range(n_rungs)])
        for k, (ok, _, _) in enumerate(r["runs"]):
            s[k][0] += ok
            s[k][1] += 1
    overall = [sum(r["runs"][k][0] for r in train) / len(train) for k in range(n_rungs)]

    def route(row: Row) -> tuple[bool, float]:
        s = stats.get(row["repo"])
        rates = [s[k][0] / s[k][1] for k in range(n_rungs)] if s else overall
        k = next(k for k in range(n_rungs) if rates[k] >= max(rates) - eps)
        ok, cost, _ = row["runs"][k]
        return ok, cost

    return route


def fit_eps(train: list[Row]) -> float:
    routers = {eps: repo_router(train, eps) for eps in EPS_GRID}
    return pick(train, EPS_GRID, lambda eps, r: routers[eps](r))


def cross_fit(rows: list[Row]) -> dict[str, list[tuple[bool, float]]]:
    halves = [rows[0::2], rows[1::2]]
    fp: list = [None] * len(rows)
    rp: list = [None] * len(rows)
    fitted = []
    for h in (0, 1):
        train = halves[1 - h]
        taus, eps = fit_taus(train), fit_eps(train)
        route = repo_router(train, eps)
        fitted.append((taus, eps))
        for j in range(h, len(rows), 2):
            fp[j] = cascade(rows[j], taus)
            rp[j] = route(rows[j])
    arms = {
        f"always-rung{k}": [(r["runs"][k][0], r["runs"][k][1]) for r in rows]
        for k in range(len(rows[0]["runs"]))
    }
    arms["first-pass (steps gate, cross-fit)"] = fp
    arms["repo-predictive (cross-fit)"] = rp
    arms["oracle-cascade (bound)"] = [oracle_cascade(r) for r in rows]
    arms["oracle-predictive (bound)"] = [oracle_predictive(r) for r in rows]
    arms["_fitted"] = fitted  # type: ignore[assignment]
    return arms


def paired_ci(
    a: list[tuple[bool, float]],
    b: list[tuple[bool, float]],
    n_boot: int = 4000,
    seed: int = 42,
):
    """Paired instance bootstrap of a − b: Δresolved-rate, Δ$/instance, Δ$/resolved (95%)."""
    n = len(a)
    rng = random.Random(seed)
    dr, dc, dp = [], [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ao, ac = sum(a[i][0] for i in idx), sum(a[i][1] for i in idx)
        bo, bc = sum(b[i][0] for i in idx), sum(b[i][1] for i in idx)
        dr.append((ao - bo) / n)
        dc.append((ac - bc) / n)
        if ao and bo:
            dp.append(ac / ao - bc / bo)
    out = []
    for v in (dr, dc, dp):
        v.sort()
        out.append((v[int(0.025 * len(v))], v[int(0.975 * len(v))]))
    return out


def report(cache: Path) -> None:
    print(
        f"# Agentic routing on real agent runs (SWE-bench Verified, mini-SWE-agent; data @{PINNED[:7]})\n"
    )
    for name, ladder in LADDERS.items():
        rows = rows_for(cache, ladder)
        arms = cross_fit(rows)
        fitted = arms.pop("_fitted")
        n = len(rows)
        print(f"## {name}  (n={n})\n")
        print("rungs: " + " → ".join(s.split("_", 2)[2] for s in ladder))
        print(f"fitted per half (step taus, repo eps): {fitted}\n")
        print(
            "| arm | resolved | $/instance | $/resolved | Δ resolved vs first-pass | Δ $/inst vs first-pass | Δ $/resolved vs first-pass |"
        )
        print("|---|---|---|---|---|---|---|")
        fp = arms["first-pass (steps gate, cross-fit)"]
        for arm, outs in arms.items():
            ok, cost = total(outs)
            if arm.startswith("first-pass"):
                cis = "— | — | —"
            else:
                (r0, r1), (c0, c1), (p0, p1) = paired_ci(outs, fp)
                cis = f"[{r0:+.3f}, {r1:+.3f}] | [{c0:+.3f}, {c1:+.3f}] | [{p0:+.3f}, {p1:+.3f}]"
            label = arm
            if arm.startswith("always-rung"):
                label = "always-" + ladder[int(arm[-1])].split("_", 2)[2]
            print(
                f"| {label} | {ok}/{n} ({ok / n:.3f}) | ${cost / n:.3f} | ${cost / ok:.3f} | {cis} |"
            )
        print()


def _self_check() -> None:
    cheap_ok = {"id": "a__1", "repo": "a", "runs": [(True, 1.0, 10), (True, 5.0, 10)]}
    cheap_struggles = {
        "id": "a__2",
        "repo": "a",
        "runs": [(False, 1.0, 90), (True, 5.0, 20)],
    }
    # Gate passes (few steps): serve cheap, pay cheap only.
    assert cascade(cheap_ok, [50]) == (True, 1.0)
    # Gate fails (many steps): rerun on the top model, pay both.
    assert cascade(cheap_struggles, [50]) == (True, 6.0)
    # tau = -1 never passes: always escalate.
    assert cascade(cheap_ok, [-1]) == (True, 6.0)
    assert oracle_cascade(cheap_struggles) == (True, 6.0)
    assert oracle_predictive(cheap_struggles) == (True, 5.0)
    assert oracle_predictive({"runs": [(False, 1.0, 1), (False, 5.0, 1)]}) == (
        False,
        1.0,
    )
    # The fit matches the top model (2 resolved) at the least cost: tau=10 ($7), not -1 ($10), and
    # not 90, which is cheaper ($2) but serves the cheap model's failure.
    assert fit_taus([cheap_ok, cheap_struggles]) == [10]
    # Repo router: cheap resolves 1/2 vs top 2/2; eps 0.5 tolerates that gap, eps 0 does not.
    assert repo_router([cheap_ok, cheap_struggles], 0.5)(cheap_struggles) == (
        False,
        1.0,
    )
    assert repo_router([cheap_ok, cheap_struggles], 0.0)(cheap_struggles) == (True, 5.0)
    # Bootstrap of an arm against itself is exactly zero.
    same = [(True, 1.0), (False, 2.0)]
    assert paired_ci(same, same, n_boot=50) == [(0.0, 0.0)] * 3
    pa = (
        "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -40,6 +40,7 @@ def f\n"
        "diff --git a/reproduce.py b/reproduce.py\nnew file mode 100644\n--- /dev/null\n+++ b/reproduce.py\n@@ -0,0 +1,3 @@\n"
    )
    pb_near = "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -44,2 +44,3 @@ def f\n"
    pb_far = "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -400,2 +400,3 @@ def g\n"
    pb_test = (
        "diff --git a/pkg/tests/test_mod.py b/pkg/tests/test_mod.py\n@@ -1,2 +1,3 @@\n"
    )
    # The scratch reproduce.py (a new file) is ignored; only pkg/mod.py counts.
    assert edits(pa) == {"pkg/mod.py": [(40, 46)]}
    assert agree(pa, pb_near, hunks=True) and agree(pa, pb_far, hunks=False)
    assert not agree(pa, pb_far, hunks=True)
    # A patch that only touches tests is no fix; an empty patch never agrees.
    assert not agree(pa, pb_test, hunks=False) and not agree("", "", hunks=False)
    # Selection: model 1 is within eps of the best and cheaper, so it wins; with eps=0 the best wins.
    toy = [[(True, 5.0), (True, 1.0)], [(True, 5.0), (False, 1.0)]] * 50
    assert select_cheapest(toy, 0.6) == 1 and select_cheapest(toy, 0.0) == 0
    print("self-check ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        _self_check()
    else:
        cache = (
            Path(sys.argv[1])
            if len(sys.argv) > 1
            else Path.home() / ".cache" / "firstpass-swebench-experiments"
        )
        report(cache)
        selection_report(cache)
        agreement_report(cache)
