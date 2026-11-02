#!/usr/bin/env python3
"""Routing granularity for multi-turn coding agents: per-workload vs per-session vs mid-session.

Compares routing granularities on n=500 SWE-bench Verified mini-SWE-agent instances.
Policies and decision rules pre-registered in docs/benchmarks/agent-granularity-preregistration.md.

Corrections vs initial draft (see Deviations section in results doc):
  - P4 trigger: >= tau_pin (was > tau_pin, off-by-one vs pre-registration)
  - Per-call cascade (P5): whole recorded runs, quality IS identifiable; previous "not identifiable"
    label was wrong
  - Removed "est. no caching" column: mixed-model nc-factor scaling is invalid
  - Random arm (P6) implemented
  - _traj_context: I_0 = system + first user message only (was summing all user messages)
  - P3-pool added: per-session from 11-model pool to isolate pool-vs-granularity confound
  - n_switched uses fold-specific fitted tau

Usage: analyze-agent-granularity.py [CACHE_DIR]  |  analyze-agent-granularity.py --self-check
"""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ── pinned data source ────────────────────────────────────────────────────────
PINNED = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
RAW = f"https://raw.githubusercontent.com/SWE-bench/experiments/{PINNED}/evaluation/verified"
TRAJ = "https://swe-bench-submissions.s3.amazonaws.com/bash-only/{s}/trajs/{i}/{i}.traj.json"

# ── ladders ───────────────────────────────────────────────────────────────────
LADDERS: dict[str, list[str]] = {
    "claude haiku→opus (v2.0.0)": [
        "20260217_mini-v2.0.0_claude-4-5-haiku-high",
        "20260217_mini-v2.0.0_claude-4-5-opus-high",
    ],
    "gpt-5-mini→gpt-5.2-high (v2.0.0)": [
        "20260217_mini-v2.0.0_gpt-5-mini",
        "20260217_mini-v2.0.0_gpt-5-2-high",
    ],
}

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
SELECT_EPS = 0.01

# Price table ($/1M tokens); used only for the handoff re-read illustration.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-4-5-haiku-high": (1.0, 5.0),
    "claude-4-5-sonnet-high": (3.0, 15.0),
    "claude-4-5-opus-high": (15.0, 75.0),
    "claude-4-6-opus": (15.0, 75.0),
    "gpt-5-mini": (0.4, 1.6),
    "gpt-5-2-high": (5.0, 15.0),
    "gpt-5": (5.0, 15.0),
    "gemini-3-flash-high": (0.4, 1.6),
    "deepseek-3-2-high": (0.27, 1.1),
    "glm-5-high": (0.5, 1.5),
    "kimi-k2-5-high": (0.6, 2.5),
    "minimax-2-5-high": (0.3, 1.1),
}

CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.1
CHARS_PER_TOKEN = 2.5  # for role-based trajectory format only

FALLBACK_I0 = 2333
FALLBACK_DELTA = 574
FALLBACK_OUT = 219
CALIB_N = 20

QUANTILES = [i / 10 for i in range(11)]
EPS_GRID = [0.0, 0.05, 0.1, 0.2, 0.3]

Run = tuple[bool, float, int]  # resolved, cost_usd, api_calls
Row = dict  # {"id", "repo", "runs": [Run]}


# ── data loading ──────────────────────────────────────────────────────────────


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


# ── context model: calibrated from trajectory sample ─────────────────────────


def _traj_context(
    cache: Path, system: str, instance_id: str
) -> tuple[int, int, int] | None:
    """(I_0, step_delta, out_per_step) in tokens.

    I_0 = system prompt + first user message (the initial problem statement only).
    Later user/tool/assistant messages count toward per-step growth (delta).

    Two formats:
    - OpenAI Responses API: response objects with 'usage' field give authoritative per-step counts.
    - Claude role-based: char-length estimate; I_0 from system + FIRST user message only.
    """
    path = cache / f"{system}.ctx.{instance_id}.json"
    if not path.exists():
        try:
            url = TRAJ.format(s=system, i=instance_id)
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
        except OSError:
            return None
        path.write_bytes(raw)
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    msgs = d.get("messages") or d.get("history") or []
    if not msgs:
        return None
    n_steps = d.get("info", {}).get("model_stats", {}).get("api_calls") or 0
    if n_steps <= 0:
        return None

    # ── OpenAI Responses API: authoritative per-step token counts ──
    response_usages = [
        m["usage"]
        for m in msgs
        if isinstance(m, dict) and "usage" in m and isinstance(m["usage"], dict)
    ]
    if response_usages:
        input_toks = [u.get("input_tokens", 0) or 0 for u in response_usages]
        output_toks = [u.get("output_tokens", 0) or 0 for u in response_usages]
        if not input_toks:
            return None
        I_0 = input_toks[0]
        delta = (
            (input_toks[-1] - input_toks[0]) // max(1, len(input_toks) - 1)
            if len(input_toks) > 1
            else 0
        )
        out = sum(output_toks) // len(output_toks) if output_toks else 0
        return I_0, max(0, delta), out

    # ── Claude role-based: estimate from char lengths ──
    def mlen(m: dict) -> int:
        c = m.get("content") or ""
        tc = m.get("tool_calls") or []
        chars = len(c) if isinstance(c, str) else len(json.dumps(c))
        chars += sum(len(json.dumps(t)) for t in tc)
        return int(chars / CHARS_PER_TOKEN)

    # I_0 = system + FIRST user message only; subsequent messages count toward delta.
    system_len = 0
    first_user_len = 0
    seen_first_user = False
    later_len = 0
    assistant_len = 0
    for m in msgs:
        role = m.get("role", "?")
        length = mlen(m)
        if role == "system":
            system_len += length
        elif role == "user" and not seen_first_user:
            first_user_len = length
            seen_first_user = True
        else:
            later_len += length
            if role == "assistant":
                assistant_len += length

    I_0 = system_len + first_user_len
    return I_0, later_len // n_steps, assistant_len // n_steps


def calibrate_context(cache: Path, system: str, ids: list[str]) -> tuple[int, int, int]:
    """Median (I_0, delta, out_per_step) from up to CALIB_N trajectories."""
    sample = ids[:CALIB_N]

    def fetch(i: str) -> tuple[int, int, int] | None:
        return _traj_context(cache, system, i)

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(fetch, sample))

    valid = [r for r in results if r is not None]
    if not valid:
        print(
            f"  [context model] fallback to hardcoded defaults (no trajectories for {system})"
        )
        return FALLBACK_I0, FALLBACK_DELTA, FALLBACK_OUT

    def median(xs: list[int]) -> int:
        s = sorted(xs)
        return s[len(s) // 2]

    I_0 = median([r[0] for r in valid])
    delta = median([r[1] for r in valid])
    out = median([r[2] for r in valid])
    print(
        f"  [context model] calibrated from {len(valid)}/{len(sample)}: "
        f"I_0={I_0}, delta={delta}, out_per_step={out}"
    )
    return I_0, delta, out


def model_suffix(system: str) -> str:
    parts = system.split("_", 3)
    return parts[-1] if len(parts) >= 3 else system


def get_price(system: str) -> tuple[float, float]:
    return MODEL_PRICES.get(model_suffix(system), (0.0, 0.0))


# ── cost model (used only for handoff re-read illustration) ───────────────────


def nocache_cost(
    N: int, I0: int, delta: int, out: int, price_in: float, price_out: float
) -> float:
    total_in = sum(I0 + k * delta for k in range(N))
    return (total_in * price_in + N * out * price_out) / 1e6


def withcache_cost(
    N: int, I0: int, delta: int, out: int, price_in: float, price_out: float
) -> float:
    if N == 0:
        return 0.0
    cost = I0 * CACHE_WRITE_MULT * price_in / 1e6 + out * price_out / 1e6
    for k in range(1, N):
        prefix = I0 + k * delta
        cost += (prefix * CACHE_READ_MULT + delta * CACHE_WRITE_MULT) * price_in / 1e6
        cost += out * price_out / 1e6
    return cost


def switch_overhead_usd(tau: int, price_top_in: float, I0: int, delta: int) -> float:
    """Cost for top model's uncached re-read of prefix at a mid-run HANDOFF point.
    Not charged to escalate-by-restart (top model's recorded run already paid its first read).
    tau < 0 treated as 0 (switch before any cheap step: context = I_0 only)."""
    actual_steps = max(0, tau)
    return (I0 + actual_steps * delta) * price_top_in / 1e6


# ── policy helpers ────────────────────────────────────────────────────────────


def total(outs: list[tuple[bool, float]]) -> tuple[int, float]:
    return sum(o for o, _ in outs), sum(c for _, c in outs)


def paired_ci(
    a: list[tuple[bool, float]],
    b: list[tuple[bool, float]],
    n_boot: int = 4000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Paired bootstrap CI a − b: [Δresolved-rate, Δ$/instance, Δ$/resolved] (95%)."""
    n = len(a)
    rng = random.Random(seed)
    dr, dc, dp = [], [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ao = sum(a[i][0] for i in idx)
        ac = sum(a[i][1] for i in idx)
        bo = sum(b[i][0] for i in idx)
        bc = sum(b[i][1] for i in idx)
        dr.append((ao - bo) / n)
        dc.append((ac - bc) / n)
        if ao and bo:
            dp.append(ac / ao - bc / bo)
    out = []
    for v in (dr, dc, dp):
        v.sort()
        out.append((v[int(0.025 * len(v))], v[int(0.975 * len(v))]))
    return out


def cascade(row: Row, taus: list[float]) -> tuple[bool, float]:
    """Run-level cascade (P5): run cheap model to completion; if cheap api_calls <= tau
    serve cheap outcome; else run top model from scratch. Both are whole recorded runs —
    no stitching, quality IS identifiable."""
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


def random_arm(rows: list[Row], seed: int = 42) -> list[tuple[bool, float]]:
    """P6 random: serve cheap or top with equal probability per instance (seeded)."""
    rng = random.Random(seed)
    result = []
    for r in rows:
        if rng.random() < 0.5:
            result.append((r["runs"][0][0], r["runs"][0][1]))
        else:
            result.append((r["runs"][-1][0], r["runs"][-1][1]))
    return result


def fit_taus(train: list[Row]) -> list[float]:
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

    bar = sum(r["runs"][-1][0] for r in train)
    scored = [(*total([cascade(r, c) for r in train]), c) for c in combos(0)]
    feasible = [(cost, c) for ok, cost, c in scored if ok >= bar]
    return min(feasible)[1] if feasible else max(scored, key=lambda x: (x[0], -x[1]))[2]


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
    bar = sum(r["runs"][-1][0] for r in train)
    scored = [
        (
            sum(routers[eps](r)[0] for r in train),
            -sum(routers[eps](r)[1] for r in train),
            eps,
        )
        for eps in EPS_GRID
    ]
    feasible = [eps for ok, _, eps in scored if ok >= bar]
    if feasible:
        return min(feasible, key=lambda e: sum(routers[e](r)[1] for r in train))
    return max(scored)[2]


def select_cheapest(train: list[list[tuple[bool, float]]], eps: float) -> int:
    n_models = len(train[0])
    n = len(train)
    rates = [sum(r[k][0] for r in train) / n for k in range(n_models)]
    costs = [sum(r[k][1] for r in train) for k in range(n_models)]
    ok = [k for k in range(n_models) if rates[k] >= max(rates) - eps]
    return min(ok, key=lambda k: costs[k])


# ── P4: escalate-and-pin (escalate-by-restart) ───────────────────────────────


def escalate_pin(row: Row, tau: float) -> tuple[bool, float]:
    """Escalate-by-restart (P4): escalate when cheap api_calls >= tau_pin (pre-reg rule).

    Post-switch outcome = top model's own full recorded run (restart; no stitching).
    Cost = partial cheap spend (proportional to tau/cheap_steps) + top model full run.
    tau = -1 means always escalate with zero partial cheap spend.
    """
    cheap_res, cheap_cost, cheap_steps = row["runs"][0]
    top_res, top_cost, _ = row["runs"][-1]

    # Pre-registration: keep cheap if api_calls < tau_pin; escalate if api_calls >= tau_pin
    if cheap_steps < tau:
        return cheap_res, cheap_cost

    actual_tau = max(0, int(tau))
    partial_frac = actual_tau / cheap_steps if cheap_steps > 0 else 0.0
    return top_res, cheap_cost * partial_frac + top_cost


def fit_tau_pin(train: list[Row]) -> float:
    """Cross-fit tau_pin: cheapest tau (step-count quantile grid) where resolved >= always-top."""
    steps = sorted(r["runs"][0][2] for r in train)
    grid = sorted(
        {steps[min(int(q * (len(steps) - 1)), len(steps) - 1)] for q in QUANTILES}
        | {-1}
    )
    bar = sum(r["runs"][-1][0] for r in train)

    def run_policy(tau: float) -> tuple[int, float]:
        outs = [escalate_pin(r, tau) for r in train]
        return sum(o for o, _ in outs), sum(c for _, c in outs)

    scored = [(*run_policy(tau), tau) for tau in grid]
    feasible = [(cost, tau) for ok, cost, tau in scored if ok >= bar]
    return min(feasible)[1] if feasible else max(scored, key=lambda x: (x[0], -x[1]))[2]


# ── pool helpers ──────────────────────────────────────────────────────────────


def load_pool_data(cache: Path) -> tuple[list[str], list[dict]]:
    """Returns (sorted_ids, list_of_per_instance_dicts_per_model)."""
    data = [load(cache, s) for s in POOL]
    ids = sorted(set.intersection(*(set(d) for d in data)))
    return ids, data


def workload_selection(
    ids: list[str], data: list[dict]
) -> dict[str, list[tuple[bool, float]]]:
    """P2: pick cheapest pool model within SELECT_EPS of best on training half."""
    res = [
        [(bool(d[i]["resolved"]), float(d[i]["cost"] or 0.0)) for d in data]
        for i in ids
    ]
    picks = [
        select_cheapest(res[1::2], SELECT_EPS),
        select_cheapest(res[0::2], SELECT_EPS),
    ]
    sel = [r[picks[j % 2]] for j, r in enumerate(res)]
    ref_idx = next(k for k, s in enumerate(POOL) if "opus-high" in s and "4-5" in s)
    ref = [r[ref_idx] for r in res]
    return {"pool-selection (P2, cross-fit)": sel, "always-opus (pool ref)": ref}


def pool_session_selection(
    ids: list[str], data: list[dict]
) -> tuple[list[tuple[bool, float]], list[float]]:
    """P3-pool: per-session repo-level selection from 11-model pool, cross-fit.
    Returns (outcomes, fitted_eps_vals)."""
    rows = [
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
    n = len(rows)
    halves = [rows[0::2], rows[1::2]]
    rp: list[tuple[bool, float] | None] = [None] * n
    fitted_eps_vals = []
    for h in (0, 1):
        train = halves[1 - h]
        eps_val = fit_eps(train)
        fitted_eps_vals.append(eps_val)
        route = repo_router(train, eps_val)
        for j in range(h, n, 2):
            rp[j] = route(rows[j])
    return list(rp), fitted_eps_vals  # type: ignore[return-value]


# ── per-ladder analysis ───────────────────────────────────────────────────────


def analyse_ladder(
    cache: Path,
    name: str,
    ladder: list[str],
    ctx: tuple[int, int, int],
) -> tuple[list[str], list[tuple[bool, float]]]:
    """Analyse one ladder. Returns (instance_ids, p3_outcomes) for cross-ladder comparison."""
    rows = rows_for(cache, ladder)
    n = len(rows)
    cheap_sys, top_sys = ladder[0], ladder[-1]
    price_top_in = get_price(top_sys)[0]
    I0, delta, out_per_step = ctx

    # Compute avg nc_factor for cheap model (informational only; not in table)
    price_cheap_in, price_cheap_out = get_price(cheap_sys)
    cheap_nc_factors = []
    for row in rows:
        steps = row["runs"][0][2]
        if steps > 0 and price_cheap_in > 0:
            wc = withcache_cost(
                steps, I0, delta, out_per_step, price_cheap_in, price_cheap_out
            )
            nc = nocache_cost(
                steps, I0, delta, out_per_step, price_cheap_in, price_cheap_out
            )
            if wc > 0:
                cheap_nc_factors.append(nc / wc)
    avg_cheap_nc = (
        sum(cheap_nc_factors) / len(cheap_nc_factors) if cheap_nc_factors else 3.0
    )

    # Cross-fit
    halves = [rows[0::2], rows[1::2]]
    fp: list[tuple[bool, float] | None] = [None] * n
    rp: list[tuple[bool, float] | None] = [None] * n
    ep: list[tuple[bool, float] | None] = [None] * n
    fitted_cascade: list[list[float]] = []
    fitted_pin: list[float] = []
    fitted_eps_val: list[float] = []

    for h in (0, 1):
        train = halves[1 - h]
        taus = fit_taus(train)
        tau_pin = fit_tau_pin(train)
        eps_val = fit_eps(train)
        route = repo_router(train, eps_val)
        fitted_cascade.append(taus)
        fitted_pin.append(tau_pin)
        fitted_eps_val.append(eps_val)
        for j in range(h, n, 2):
            fp[j] = cascade(rows[j], taus)
            rp[j] = route(rows[j])
            ep[j] = escalate_pin(rows[j], tau_pin)

    print(f"## {name}  (n={n})\n")
    print(
        f"cascade tau per fold: {fitted_cascade}  repo-eps per fold: {fitted_eps_val}"
    )
    print(f"escalate-by-restart tau per fold: {fitted_pin}")
    print(
        f"context model (cheap model only): I_0={I0}, delta={delta}, "
        f"out/step={out_per_step} tokens  (avg nc_factor={avg_cheap_nc:.2f}×)"
    )
    print()

    always_cheap = [(r["runs"][0][0], r["runs"][0][1]) for r in rows]
    always_top = [(r["runs"][-1][0], r["runs"][-1][1]) for r in rows]
    rand = random_arm(rows)
    oracle_pi = [oracle_predictive(r) for r in rows]
    oracle_cas = [oracle_cascade(r) for r in rows]

    arms: dict[str, list[tuple[bool, float]]] = {
        "always-cheap [P1]": always_cheap,
        "always-top [P1]": always_top,
        "run-level cascade (cross-fit) [P5]": list(fp),  # type: ignore[arg-type]
        "per-session / repo (cross-fit) [P3]": list(rp),  # type: ignore[arg-type]
        "escalate-by-restart (cross-fit) [P4]": list(ep),  # type: ignore[arg-type]
        "random (seed=42) [P6]": rand,
        "oracle-per-instance (bound) [P6]": oracle_pi,
        "oracle-cascade (bound)": oracle_cas,
    }
    top_wc = arms["always-top [P1]"]

    print(
        "| arm | resolved | $/inst | $/resolved | Δ resolved vs always-top | Δ $/inst vs always-top |"
    )
    print("|---|---|---|---|---|---|")
    for arm, outs in arms.items():
        ok_w, cost_w = total(outs)
        if arm == "always-top [P1]":
            dr_ci = "— | —"
        else:
            (r0, r1), (c0, c1), _ = paired_ci(outs, top_wc)
            dr_ci = f"[{r0:+.3f}, {r1:+.3f}] | [{c0:+.3f}, {c1:+.3f}]"
        print(
            f"| {arm} | {ok_w}/{n} ({ok_w / n:.3f}) | ${cost_w / n:.3f} | "
            f"${cost_w / ok_w:.3f} | {dr_ci} |"
        )
    print()

    # Escalate-by-restart summary and handoff re-read illustration
    n_switched_folds = [0, 0]
    for j, row in enumerate(rows):
        fold = j % 2
        tau = fitted_pin[fold]
        if row["runs"][0][2] >= int(tau):  # >= per pre-registration
            n_switched_folds[fold] += 1
    n_switched = sum(n_switched_folds)

    print(
        f"  escalate-by-restart: {n_switched_folds[0]}/{n // 2} (fold 0, tau={fitted_pin[0]}) "
        f"+ {n_switched_folds[1]}/{n - n // 2} (fold 1, tau={fitted_pin[1]}) "
        f"= {n_switched}/{n} trigger"
    )
    for fold_h, tau_p in enumerate(fitted_pin):
        context_toks = I0 + max(0, int(tau_p)) * delta
        overhead = switch_overhead_usd(int(tau_p), price_top_in, I0, delta)
        print(
            f"  [handoff illustration, fold {fold_h}] a mid-run HANDOFF at tau={tau_p} "
            f"would re-read {context_toks:,} tokens uncached "
            f"(est. ${overhead:.3f}/inst from context model, cheap-model calibration only); "
            "not charged to escalate-by-restart"
        )
    print()

    instance_ids = [r["id"] for r in rows]
    return instance_ids, list(rp)  # type: ignore[return-value]


# ── main report ───────────────────────────────────────────────────────────────


def report(cache: Path) -> None:
    print(
        f"# Routing granularity for multi-turn coding agents "
        f"(SWE-bench Verified, data @{PINNED[:7]})\n"
    )
    print("## Validity caveat\n")
    print(
        "Whole-run policies (always-X, per-session, workload, run-level cascade, "
        "escalate-by-restart) are scored exactly from recorded runs. "
        "Escalate-by-restart post-switch quality ≈ top model's own full run (optimistic; "
        "see pre-registration). Context growth model calibrated on the cheap model only; "
        "I_0 = system + first user message. See pre-registration for full caveats.\n"
    )

    ladder_ids_map: dict[str, list[str]] = {}
    ladder_p3_map: dict[str, list[tuple[bool, float]]] = {}

    for lname, ladder in LADDERS.items():
        cheap_sys = ladder[0]
        print(f"### Calibrating context model for {model_suffix(cheap_sys)} ...")
        data_cheap = load(cache, cheap_sys)
        ids_cheap = sorted(data_cheap.keys())
        ctx = calibrate_context(cache, cheap_sys, ids_cheap)
        print()
        lids, lp3 = analyse_ladder(cache, lname, ladder, ctx)
        ladder_ids_map[lname] = lids
        ladder_p3_map[lname] = lp3

    # Pool policies: P2 and P3-pool
    print("## Pool policies (P2 workload-selection and P3-pool session-selection)\n")
    print("Loading pool data ...")
    pool_ids, pool_data = load_pool_data(cache)
    n_pool = len(pool_ids)
    print(f"  pool n={n_pool}")

    wl = workload_selection(pool_ids, pool_data)
    p3pool_outs, p3pool_eps = pool_session_selection(pool_ids, pool_data)
    print(f"  P3-pool fitted eps per fold: {p3pool_eps}\n")

    p2_sel = wl["pool-selection (P2, cross-fit)"]
    p2_ref = wl["always-opus (pool ref)"]

    print(
        "| arm | resolved | $/instance | Δ resolved vs always-opus | Δ $/inst vs always-opus |"
    )
    print("|---|---|---|---|---|")
    pool_arms = {
        "pool-selection (P2, cross-fit)": p2_sel,
        "pool-session (P3-pool, cross-fit)": p3pool_outs,
        "always-claude-4-5-opus-high (pool ref)": p2_ref,
    }
    for arm, outs in pool_arms.items():
        ok, cost = total(outs)
        if "pool ref" in arm:
            cis = "— | —"
        else:
            (r0, r1), (c0, c1), _ = paired_ci(outs, p2_ref)
            cis = f"[{r0:+.3f}, {r1:+.3f}] | [{c0:+.3f}, {c1:+.3f}]"
        print(
            f"| {arm} | {ok}/{n_pool} ({ok / n_pool:.3f}) | ${cost / n_pool:.3f} | {cis} |"
        )
    print()

    # P3-pool vs P2 paired CI (same instances, clean comparison)
    (r0, r1), (c0, c1), _ = paired_ci(p3pool_outs, p2_sel)
    print(
        f"P3-pool vs P2 paired CI (P3-pool − P2): "
        f"Δresolved [{r0:+.3f}, {r1:+.3f}], Δ$/inst [{c0:+.3f}, {c1:+.3f}]"
    )
    print()

    # P3 (2-model ladder) vs P2 paired CI: match by instance ID
    print("## P3 (2-model ladder) vs P2 (pool): cross-comparison\n")
    pool_id_set = set(pool_ids)
    for lname in LADDERS:
        lids = ladder_ids_map[lname]
        lp3 = ladder_p3_map[lname]
        lid_to_p3 = dict(zip(lids, lp3))
        # Match ladder P3 to pool P2 by instance ID
        matched_p3 = []
        matched_p2 = []
        for pid, p2_out in zip(pool_ids, p2_sel):
            if pid in lid_to_p3:
                matched_p3.append(lid_to_p3[pid])
                matched_p2.append(p2_out)
        n_matched = len(matched_p3)
        n_total = len(lids)
        if n_matched < n_total:
            print(
                f"  NOTE: {n_total - n_matched} ladder instances have no pool match; "
                f"CI on {n_matched} shared instances"
            )
        if n_matched > 0:
            (r0, r1), (c0, c1), _ = paired_ci(matched_p3, matched_p2)
            p3_ok, p3_cost = total(matched_p3)
            p2_ok, p2_cost = total(matched_p2)
            print(f"  {lname}: P3-ladder vs P2 on n={n_matched} shared instances")
            print(
                f"    P3-ladder: {p3_ok}/{n_matched} (${p3_cost / n_matched:.3f}/inst)  "
                f"P2: {p2_ok}/{n_matched} (${p2_cost / n_matched:.3f}/inst)"
            )
            print(
                f"    P3-ladder − P2 paired CI: "
                f"Δresolved [{r0:+.3f}, {r1:+.3f}], Δ$/inst [{c0:+.3f}, {c1:+.3f}]"
            )
            print(
                "    CONFOUND: P3 uses 2-model ladder; P2 uses 11-model pool. "
                "Use P3-pool vs P2 for the pool-controlled comparison."
            )
        print()


# ── self-check ─────────────────────────────────────────────────────────────────


def _self_check() -> None:
    # nocache cost must exceed withcache cost (caching saves money)
    nc_10 = nocache_cost(10, 1000, 500, 100, 1.0, 5.0)
    wc_10 = withcache_cost(10, 1000, 500, 100, 1.0, 5.0)
    assert nc_10 > wc_10, "uncached must cost more than cached"

    # longer session → higher nc/wc ratio
    nc_factor_10 = nc_10 / wc_10
    nc_factor_50 = nocache_cost(50, 1000, 500, 100, 1.0, 5.0) / withcache_cost(
        50, 1000, 500, 100, 1.0, 5.0
    )
    assert nc_factor_50 > nc_factor_10, "longer sessions benefit more from caching"

    # switch overhead scales with context size at tau
    sw5 = switch_overhead_usd(5, 15.0, 1000, 500)
    sw20 = switch_overhead_usd(20, 15.0, 1000, 500)
    assert sw20 > sw5, "later switch = larger context = higher overhead"

    row = {
        "id": "x__1",
        "repo": "x",
        "runs": [(True, 1.0, 10), (True, 5.0, 20)],
    }

    # below tau: serve cheap
    res, cwc = escalate_pin(row, tau=15.0)
    assert res is True and cwc == 1.0, f"below tau: expected cheap; got {cwc}"

    # above tau: escalate; cost = partial cheap + top
    res2, cwc2 = escalate_pin(row, tau=5.0)
    assert res2 is True, "above tau: should return top model result"
    expected = 1.0 * (5 / 10) + 5.0  # partial (5/10 of cheap) + top
    assert abs(cwc2 - expected) < 1e-9, f"above tau cost: {cwc2} != {expected}"

    # at equality (cheap_steps == tau): pre-reg says escalate (api_calls >= tau_pin)
    res3, cwc3 = escalate_pin(row, tau=10.0)
    assert res3 is True, "at equality (steps==tau): should escalate per pre-reg"
    expected3 = 1.0 * (10 / 10) + 5.0  # full cheap partial + top
    assert abs(cwc3 - expected3) < 1e-9, f"at equality cost: {cwc3} != {expected3}"

    # tau=-1: always escalate, zero partial cheap spend (actual_tau=0)
    _, cwc_neg1 = escalate_pin(row, tau=-1.0)
    assert abs(cwc_neg1 - 5.0) < 1e-9, (
        f"tau=-1: cost should be top only=5.0, got {cwc_neg1}"
    )

    # switch_overhead_usd: tau=0 boundary
    overhead_at_0 = switch_overhead_usd(0, 15.0, 1000, 300)
    assert abs(overhead_at_0 - 1000 * 15.0 / 1e6) < 1e-12, (
        "tau=0: overhead = I0 × price_top"
    )

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
