#!/usr/bin/env python3
"""Certified model selection (ADR 0014) replayed as a live A/B on real agent runs, at $0.

The serving-side experiment randomizes sessions to the route's incumbent ladder or a challenger
ladder and runs two betting e-processes on the paired outcome difference `D = X_inc - X_ch`:
certify (rejects "challenger worse by >= m") and halt (rejects "challenger worse by <= m"). This
script replays that design on SWE-bench Verified with the incumbent fixed to opus-4.5: each arriving
session is a uniformly random instance, assigned to an arm by a fair coin, and its outcome is that
model's recorded resolved bit. It mirrors `crates/firstpass-core/src/select.rs` exactly (same lambda
grid, same pairing, same stopping rule) so the numbers below are the numbers the product would see.

Two synthetic Bernoulli rows check validity: a challenger exactly `m` worse (the null boundary,
where certifying is the error the test caps at alpha) and one `2m` worse.

Data: github.com/SWE-bench/experiments@PINNED evaluation/verified/<system>/per_instance_details.json,
fetched at run time into a cache dir and never committed (the repository states no licence).

Usage: sim-certified-selection.py [--margin M] [--reps N] [CACHE_DIR]
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import deque
from pathlib import Path

PINNED = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
RAW = f"https://raw.githubusercontent.com/SWE-bench/experiments/{PINNED}/evaluation/verified"
INCUMBENT = "20260217_mini-v2.0.0_claude-4-5-opus-high"
CHALLENGERS = [
    "20260217_mini-v2.0.0_minimax-2-5-high",
    "20260217_mini-v2.0.0_gemini-3-flash-high",
    "20260217_mini-v2.0.0_claude-4-6-opus",
    "20260217_mini-v2.0.0_kimi-k2-5-high",
    "20260217_mini-v2.0.0_claude-4-5-haiku-high",
    "20260217_mini-v2.0.0_gpt-5-mini",
]
GRID = [0.05 * j for j in range(1, 19)]  # == select.rs LAMBDA_GRID
ALPHA = 0.05
CAP = 20_000  # sessions per replay before calling it undecided

Outcomes = dict[str, tuple[float, float]]  # instance -> (resolved, cost_usd)


def load(cache: Path, system: str) -> Outcomes:
    path = cache / f"{system}.json"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            f"{RAW}/{system}/per_instance_details.json", timeout=60
        ) as r:
            path.write_bytes(r.read())
    d = json.loads(path.read_text())
    return {k: (1.0 if v["resolved"] else 0.0, float(v["cost"])) for k, v in d.items()}


class Mixture:
    """Equal-weight mixture of fixed-lambda wealths for H0: E[sign * (D - m)] <= 0."""

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


def replay(inc: Outcomes, ch: Outcomes, keys: list[str], m: float, rng: random.Random):
    cert, halt = Mixture(m, -1.0), Mixture(m, 1.0)
    qi: deque[str] = deque()
    qc: deque[str] = deque()
    for n in range(1, CAP + 1):
        k = rng.choice(keys)
        (qi if rng.random() < 0.5 else qc).append(k)
        while qi and qc:
            d = inc[qi.popleft()][0] - ch[qc.popleft()][0]
            cert.step(d)
            halt.step(d)
            if cert.e() >= 1 / ALPHA:
                return "certified", n
            if halt.e() >= 1 / ALPHA:
                return "halted", n
    return "undecided", CAP


def bern(p: float) -> Outcomes:
    return {str(i): (1.0 if i < p * 10_000 else 0.0, 1.0) for i in range(10_000)}


def summarize(name, gap, cost_ratio, runs, reps):
    tally = {
        s: sum(1 for r in runs if r[0] == s) / reps
        for s in ("certified", "halted", "undecided")
    }
    decided = sorted(n for s, n in runs if s != "undecided")
    med = decided[len(decided) // 2] if decided else "—"
    return (
        f"| {name} | {gap:+.3f} | {cost_ratio:.2f} | {tally['certified']:.3f} | {tally['halted']:.3f} "
        f"| {tally['undecided']:.3f} | {med} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument(
        "cache",
        nargs="?",
        default=Path.home() / ".cache" / "firstpass-swebench-experiments",
    )
    a = ap.parse_args()
    cache, m = Path(a.cache), a.margin
    inc = load(cache, INCUMBENT)
    print(
        f"margin m={m}, alpha={ALPHA}, reps={a.reps}, cap={CAP} sessions, incumbent opus-4.5\n"
    )
    print(
        "| challenger | true gap | cost ratio | certified | halted | undecided | median sessions to decide |"
    )
    print("|---|---|---|---|---|---|---|")
    for system in CHALLENGERS:
        ch = load(cache, system)
        keys = sorted(k for k in inc if k in ch)
        gap = sum(inc[k][0] - ch[k][0] for k in keys) / len(keys)
        cost_ratio = sum(ch[k][1] for k in keys) / sum(inc[k][1] for k in keys)
        rng = random.Random(7)
        runs = [replay(inc, ch, keys, m, rng) for _ in range(a.reps)]
        print(summarize(system.split("_", 2)[2], gap, cost_ratio, runs, a.reps))
    for p_ch in (0.76 - m, 0.76 - 2 * m):
        i, c = bern(0.76), bern(p_ch)
        rng = random.Random(11)
        runs = [replay(i, c, list(i), m, rng) for _ in range(a.reps)]
        print(
            summarize(
                f"synthetic, {0.76 - p_ch:.2f} worse", 0.76 - p_ch, 1.0, runs, a.reps
            )
        )


if __name__ == "__main__":
    main()
