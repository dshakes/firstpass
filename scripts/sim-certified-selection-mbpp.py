#!/usr/bin/env python3
"""Certified selection on real paid MBPP outcomes, at $0: paired vs live mode (ADR 0014 stage 2).

`docs/benchmarks/mbpp-974-{sonnet-k5,opus}.checkpoint.jsonl` hold real calls of haiku-4.5,
sonnet-5 and opus-4.8 on the SAME 974 MBPP tasks. Each attempt records `gate_full_pass` (the
visible tests the router can run) and `oracle_correct` (hidden tests). That is exactly the paired
design: one request, both models, both gated.

Two questions:
  1. How many requests does each mode need to decide? Paired mode pairs the same task, so the
     between-task difficulty cancels; live mode pairs different sessions and pays for it.
  2. Does paired mode, which can only see the GATE, reach the decision the hidden-test ORACLE
     would? A gate that is a subset of the grader can certify a model the oracle would not.

The test mirrors `crates/firstpass-core/src/select.rs`: same lambda grid, same stopping rule.
Each replay draws tasks with replacement (the traffic distribution is the task set).

Usage: sim-certified-selection-mbpp.py [--margin M] [--reps N]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
GRID = [0.05 * j for j in range(1, 19)]  # == select.rs LAMBDA_GRID
ALPHA = 0.05
CAP = 20_000


def load() -> dict[str, dict[str, tuple[float, float, float]]]:
    """task -> model -> (gate_pass, oracle_correct, cost_usd)."""
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for name in ("mbpp-974-sonnet-k5", "mbpp-974-opus"):
        for line in (ROOT / f"{name}.checkpoint.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for model, rung in zip(r["ladder"], r["rungs"]):
                if rung is None:
                    continue
                short = model.split("/")[-1]
                out.setdefault(r["task_id"], {})[short] = (
                    1.0 if rung["gate_full_pass"] else 0.0,
                    1.0 if rung["oracle_correct"] else 0.0,
                    float(rung["cost_usd"]),
                )
    return out


class Mixture:
    def __init__(self, m: float, sign: float) -> None:
        self.m, self.sign = m, sign
        worst = 1 + m if sign > 0 else 1 - m
        self.lams = [g / worst for g in GRID]
        self.w = [1.0] * len(GRID)

    def step(self, d: float) -> None:
        x = self.sign * (d - self.m)
        self.w = [w * (1 + lam * x) for w, lam in zip(self.w, self.lams)]

    def e(self) -> float:
        return sum(self.w) / len(self.w)


def decide(cert: Mixture, halt: Mixture, d: float) -> str | None:
    cert.step(d)
    halt.step(d)
    if cert.e() >= 1 / ALPHA:
        return "certified"
    if halt.e() >= 1 / ALPHA:
        return "halted"
    return None


def paired(rows, inc, ch, col, m, rng):
    cert, halt = Mixture(m, -1.0), Mixture(m, 1.0)
    for n in range(1, CAP + 1):
        t = rng.choice(rows)
        verdict = decide(cert, halt, t[inc][col] - t[ch][col])
        if verdict:
            return verdict, n
    return "undecided", CAP


def live(rows, inc, ch, col, m, rng):
    cert, halt = Mixture(m, -1.0), Mixture(m, 1.0)
    qi: deque[float] = deque()
    qc: deque[float] = deque()
    for n in range(1, CAP + 1):
        t = rng.choice(rows)
        if rng.random() < 0.5:
            qi.append(t[inc][col])
        else:
            qc.append(t[ch][col])
        while qi and qc:
            verdict = decide(cert, halt, qi.popleft() - qc.popleft())
            if verdict:
                return verdict, n
    return "undecided", CAP


def row(label, runs, reps):
    t = {
        s: sum(1 for r in runs if r[0] == s) / reps
        for s in ("certified", "halted", "undecided")
    }
    dec = sorted(n for s, n in runs if s != "undecided")
    med = dec[len(dec) // 2] if dec else "—"
    return f"{label} | {t['certified']:.2f} | {t['halted']:.2f} | {t['undecided']:.2f} | {med} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    data = load()
    models = ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8")
    rows = [t for t in data.values() if all(mod in t for mod in models)]
    print(
        f"n={len(rows)} tasks with all three models; margin={a.margin}, alpha={ALPHA}, reps={a.reps}\n"
    )
    print("| incumbent → challenger | gate gap | oracle gap | cost ratio |")
    print("|---|---|---|---|")
    pairs = [
        ("claude-opus-4-8", "claude-sonnet-5"),
        ("claude-opus-4-8", "claude-haiku-4-5"),
        ("claude-sonnet-5", "claude-haiku-4-5"),
    ]
    for inc, ch in pairs:
        gg = sum(t[inc][0] - t[ch][0] for t in rows) / len(rows)
        og = sum(t[inc][1] - t[ch][1] for t in rows) / len(rows)
        cr = sum(t[ch][2] for t in rows) / sum(t[inc][2] for t in rows)
        print(f"| {inc} → {ch} | {gg:+.3f} | {og:+.3f} | {cr:.2f} |")
    print()
    print(
        "| incumbent → challenger | mode | metric | certified | halted | undecided | median requests |"
    )
    print("|---|---|---|---|---|---|---|")
    for inc, ch in pairs:
        for mode, fn in (("paired", paired), ("live", live)):
            for col, metric in ((0, "gate"), (1, "oracle")):
                rng = random.Random(13)
                runs = [fn(rows, inc, ch, col, a.margin, rng) for _ in range(a.reps)]
                print(row(f"| {inc} → {ch} | {mode} | {metric}", runs, a.reps))


if __name__ == "__main__":
    main()
