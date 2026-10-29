#!/usr/bin/env python3
"""Derive the reported SWE-bench numbers from a run's checkpoint.

Every figure in `docs/benchmarks/swebench-*.txt` must be recomputable from the raw per-instance
records by someone who did not run the benchmark. Pasting numbers out of an ad-hoc script is how a
report drifts from its data; this file is the single place the arithmetic lives.

The three framings are reported side by side on purpose. `pass@any` (was any of the N attempts
correct) is the flattering one and is NOT comparable to a SWE-bench leaderboard entry, which allows
one submission per instance. `turn-1` is the comparable one. `served` is the one that describes
Firstpass, because it scores what the router would actually have returned: the first rung whose
gate passed, or the top rung when none did.

    python3 scripts/analyze-swe-checkpoint.py docs/benchmarks/<run>.checkpoint.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys


def _betainc(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a,b), Lentz continued fraction.

    Hand-rolled because the repo takes no scipy dependency for one CDF, and a normal
    approximation is wrong at exactly the counts this benchmark produces (k=0 or k=n at n=3).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log1p(-x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1 - c * d) < 1e-12:
            break
    return front * (f - 1)


def _betaq(p: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _betainc(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI. Exact, not Wald: at n=3 a Wald interval leaves [0,1]."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else 1 - _betaq(1 - alpha / 2, n - k + 1, k)
    hi = 1.0 if k == n else 1 - _betaq(alpha / 2, n - k, k + 1)
    return lo, hi


def served_rung(rungs: list[dict]) -> dict:
    """What the router returns for one turn.

    The ladder serves the first rung whose gate passes. When no rung passes the ladder is
    exhausted and the top rung's answer is what the caller gets, so that is what gets scored —
    counting an exhausted ladder as "served nothing" would quietly discount its failures.
    """
    return next((g for g in rungs if g["gate_pass"]), rungs[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
        return 0
    if not args.checkpoint:
        ap.error("need a checkpoint path (or --self-check)")

    per: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "res": 0, "gate": 0, "ok": 0, "cost": 0.0}
    )
    total = t1_cost = 0.0
    n_inst = n_turns = t1_ok = t1_gate = t1_gate_ok = 0
    served_ok = served_gated = served_gated_ok = 0

    with open(args.checkpoint) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            repo = rec["id"].split("__")[0]
            all_rungs = [g for t in rec["turns"] for g in t["rungs"]]
            cost = sum(g["cost_usd"] for g in all_rungs)

            p = per[repo]
            p["n"] += 1
            p["res"] += any(g["oracle_correct"] for g in all_rungs)
            p["gate"] += sum(g["gate_pass"] for g in all_rungs)
            p["ok"] += sum(g["oracle_correct"] for g in all_rungs)
            p["cost"] += cost
            total += cost
            n_inst += 1

            for ti, turn in enumerate(rec["turns"]):
                n_turns += 1
                final = served_rung(turn["rungs"])
                served_ok += final["oracle_correct"]
                gated = next((g for g in turn["rungs"] if g["gate_pass"]), None)
                if gated is not None:
                    served_gated += 1
                    served_gated_ok += gated["oracle_correct"]
                if ti == 0:
                    t1_cost += sum(g["cost_usd"] for g in turn["rungs"])
                    t1_ok += final["oracle_correct"]
                    if gated is not None:
                        t1_gate += 1
                        t1_gate_ok += gated["oracle_correct"]

    if n_inst == 0:
        print("no records", file=sys.stderr)
        return 1

    res = sum(v["res"] for v in per.values())
    gate = sum(v["gate"] for v in per.values())
    ok = sum(v["ok"] for v in per.values())

    def pct(k: int, n: int) -> str:
        return f"{100 * k / n:.1f}%" if n else "—"

    def dollars(c: float, k: int) -> str:
        return f"${c / k:.2f}" if k else "—"

    print(f"# {args.checkpoint}\n")
    print("## Per repo (pass@any-of-all-attempts)\n")
    hdr = f"{'repo':14s} {'res/n':>7s} {'rate':>6s} {'95% CI':>16s} {'gate':>5s} {'ok':>4s} {'prec':>6s} {'cost':>8s} {'$/res':>8s}"
    print(hdr)
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]["res"] / max(kv[1]["n"], 1)):
        lo, hi = clopper_pearson(v["res"], v["n"])
        prec = pct(v["ok"], v["gate"]) if v["gate"] else "—"
        print(
            f"{k:14s} {v['res']:3d}/{v['n']:<3d} {pct(v['res'], v['n']):>6s} "
            f"[{100 * lo:5.1f}%,{100 * hi:5.1f}%] {v['gate']:5d} {v['ok']:4d} "
            f"{prec:>6s} ${v['cost']:7.2f} {dollars(v['cost'], v['res']):>8s}"
        )

    lo, hi = clopper_pearson(res, n_inst)
    print(
        f"\n{'POOLED':14s} {res:3d}/{n_inst:<3d} {pct(res, n_inst):>6s} "
        f"[{100 * lo:5.1f}%,{100 * hi:5.1f}%] {gate:5d} {ok:4d} "
        f"{pct(ok, gate):>6s} ${total:7.2f} {dollars(total, res):>8s}"
    )

    lo1, hi1 = clopper_pearson(t1_ok, n_inst)
    print("\n## The three framings\n")
    print(f"instances {n_inst}, turns {n_turns}, spend ${total:.2f}\n")
    print(
        f"turn-1 (SWE-bench convention, one submission)  {t1_ok}/{n_inst} = {pct(t1_ok, n_inst)} "
        f"[{100 * lo1:.1f}%, {100 * hi1:.1f}%]   ${t1_cost:.2f}  ->  {dollars(t1_cost, t1_ok)}/resolved"
        f"   gate {t1_gate_ok}/{t1_gate}"
    )
    print(
        f"served   (what the router returns, per turn)   {served_ok}/{n_turns} = {pct(served_ok, n_turns)}"
        f"                     ${total:.2f}  ->  {dollars(total, served_ok)}/correct"
        f"   gate {served_gated_ok}/{served_gated}"
    )
    print(
        f"pass@any (NOT leaderboard-comparable)          {res}/{n_inst} = {pct(res, n_inst)} "
        f"[{100 * lo:.1f}%, {100 * hi:.1f}%]   ${total:.2f}  ->  {dollars(total, res)}/resolved"
    )
    with open(args.checkpoint) as fh:
        head_to_head([json.loads(line) for line in fh if line.strip()])
    return 0


def first_pass(rungs: list[dict]) -> tuple[bool, float]:
    """Gate each rung in order and serve the first pass; pay only for the rungs actually tried."""
    spent = 0.0
    for g in rungs:
        spent += g["cost_usd"]
        if g["gate_pass"]:
            return bool(g["oracle_correct"]), spent
    return bool(rungs[-1]["oracle_correct"]), spent


def stage_router(turn: dict, min_errors: int) -> tuple[bool, float]:
    """NVIDIA NeMo Switchyard's stage router, as a decision rule: read the session's recent tool
    errors and loops, send a troubled turn to the top model, everything else to the cheap one —
    and serve it. No gate runs on the output."""
    troubled = turn["tool_errors"] >= min_errors or turn["repeated_tool_calls"] >= 1
    g = turn["rungs"][-1 if troubled else 0]
    return bool(g["oracle_correct"]), g["cost_usd"]


def head_to_head(records: list[dict], b: int = 4000, seed: int = 42) -> None:
    """First-pass vs session-signal routing on the same recorded turns.

    Turns of one instance share a conversation, so they are not independent: the bootstrap
    resamples whole instances, and the interval is on first-pass minus the other arm.
    """
    arms = {
        f"stage-router err>={k} (Switchyard-style)": (lambda t, k=k: stage_router(t, k))
        for k in (1, 3)
    }
    arms["always-top (ungated)"] = lambda t: (
        bool(t["rungs"][-1]["oracle_correct"]),
        t["rungs"][-1]["cost_usd"],
    )
    fp = [[first_pass(t["rungs"]) for t in r["turns"]] for r in records]
    n_turns = sum(len(x) for x in fp)
    fp_ok = sum(o for x in fp for o, _ in x)
    fp_cost = sum(c for x in fp for _, c in x)
    print("\n## Head-to-head: session-signal routing (offline replay, $0)\n")
    print(
        f"first-pass (gate per turn)  {fp_ok}/{n_turns} correct  ${fp_cost:.2f}  ->  ${fp_cost / fp_ok:.3f}/correct"
    )
    rng = random.Random(seed)
    idx_sets = [[rng.randrange(len(records)) for _ in records] for _ in range(b)]
    for name, rule in arms.items():
        other = [[rule(t) for t in r["turns"]] for r in records]
        d_ok = [sum(a[0] - o[0] for a, o in zip(f, x)) for f, x in zip(fp, other)]
        d_c = [sum(a[1] - o[1] for a, o in zip(f, x)) for f, x in zip(fp, other)]
        boot_ok = sorted(sum(d_ok[i] for i in s) / n_turns for s in idx_sets)
        boot_c = sorted(sum(d_c[i] for i in s) for s in idx_sets)
        ok = sum(o for x in other for o, _ in x)
        cost = sum(c for x in other for _, c in x)
        per = f"${cost / ok:.3f}/correct" if ok else "—"
        print(
            f"{name:38s} {ok}/{n_turns} correct  ${cost:.2f}  ->  {per}   "
            f"fp−arm: Δrate [{boot_ok[int(0.025 * b)]:+.4f}, {boot_ok[int(0.975 * b)]:+.4f}]  "
            f"Δ$ [{boot_c[int(0.025 * b)]:+.2f}, {boot_c[int(0.975 * b)]:+.2f}]"
        )


def _self_check() -> None:
    """`python3 scripts/analyze-swe-checkpoint.py --self-check`

    Clopper-Pearson bounds are checked against published values rather than against this
    implementation's own output, which would only prove it is self-consistent. k=0 and k=n are
    included because they are the closed forms (1-(a/2)^(1/n)) and the cases this benchmark
    actually hits at n=3.
    """
    for k, n, elo, ehi in [
        (2, 10, 0.02521, 0.55610),
        (0, 4, 0.0, 0.60236),
        (4, 4, 0.39764, 1.0),
        (1, 3, 0.00840, 0.90570),
    ]:
        lo, hi = clopper_pearson(k, n)
        assert abs(lo - elo) < 2e-4, (k, n, lo, elo)
        assert abs(hi - ehi) < 2e-4, (k, n, hi, ehi)

    # The ladder serves the first passing rung; with none passing it serves the top rung, whose
    # answer the caller gets and whose failure must therefore be scored.
    cheap_ok = {"gate_pass": True, "oracle_correct": True}
    cheap_bad = {"gate_pass": False, "oracle_correct": False}
    top_bad = {"gate_pass": False, "oracle_correct": False}
    # A passing cheap rung short-circuits the ladder.
    assert served_rung([cheap_ok, top_bad]) is cheap_ok
    # Exhausted ladder serves the TOP rung, not the bottom one. The two must be distinguishable
    # objects or this assertion passes for an implementation that returns rungs[0].
    assert served_rung([cheap_bad, top_bad]) is top_bad

    # The stage router ignores the gate: a quiet session serves the cheap rung even when its gate
    # failed, and a troubled one goes straight to the top without paying for the cheap rung.
    cheap = {"gate_pass": False, "oracle_correct": False, "cost_usd": 1.0}
    top = {"gate_pass": True, "oracle_correct": True, "cost_usd": 5.0}
    quiet = {"tool_errors": 0, "repeated_tool_calls": 0, "rungs": [cheap, top]}
    loop = {"tool_errors": 0, "repeated_tool_calls": 1, "rungs": [cheap, top]}
    assert stage_router(quiet, 1) == (False, 1.0)
    assert stage_router(loop, 3) == (True, 5.0)
    # First-pass pays for every rung it tries.
    assert first_pass([cheap, top]) == (True, 6.0)
    print("self-check ok")


if __name__ == "__main__":
    raise SystemExit(main())
