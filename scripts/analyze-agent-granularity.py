#!/usr/bin/env python3
"""Routing granularity for multi-turn coding agents: per-workload vs per-session vs mid-session.

Compares six routing granularities on n=500 SWE-bench Verified mini-SWE-agent instances
(same data source as analyze-agent-cascade.py; policies and decision rules pre-registered in
docs/benchmarks/agent-granularity-preregistration.md before this script ran).

Cache-aware cost model (key new contribution): within-session prefix caching at 1.25× write +
0.1× read; switching models mid-session resets the cache (top model pays full 1.0× input for
the accumulated prefix on its first call). Context growth estimated from a small trajectory
sample. Every policy's cost is reported with and without provider prompt caching.

Usage: analyze-agent-granularity.py [CACHE_DIR]    |   analyze-agent-granularity.py --self-check
"""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ── pinned data source ───────────────────────────────────────────────────────
PINNED = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
RAW = f"https://raw.githubusercontent.com/SWE-bench/experiments/{PINNED}/evaluation/verified"
TRAJ = "https://swe-bench-submissions.s3.amazonaws.com/bash-only/{s}/trajs/{i}/{i}.traj.json"

# ── ladders (fixed before results; same as analyze-agent-cascade.py for comparability) ──
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

# Pool for workload/per-session selection (11 models; same as analyze-agent-cascade.py POOL)
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
SELECT_EPS = 0.01  # workload selection tolerance (same as #239)

# ── price table: $/1M tokens for the leaderboard model suffix ─────────────────
# Used only for the cache-aware cost model; leaderboard costs are authoritative for observed spend.
# Matched to crates/firstpass-core/src/cost.rs where the model family aligns.
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

# Cache multipliers (same as crates/firstpass-core/src/cost.rs)
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.1
# Chars-per-token ratio for message length → token estimation.
# Calibrated on one haiku trajectory (code-heavy content ≈ 2.5 chars/token).
CHARS_PER_TOKEN = 2.5

# Context model fallback (from manually analysed astropy__astropy-12907 with haiku):
# I_0=initial context tokens (system + problem stmt), delta=tokens added per step.
FALLBACK_I0 = 2333
FALLBACK_DELTA = 574
FALLBACK_OUT = 219  # output tokens per step

CALIB_N = 20  # trajectories to fetch for context calibration

# Cross-fitting parameters (same as analyze-agent-cascade.py)
QUANTILES = [i / 10 for i in range(11)]
EPS_GRID = [0.0, 0.05, 0.1, 0.2, 0.3]

Run = tuple[bool, float, int]  # resolved, cost_usd, api_calls
Row = dict  # {"id", "repo", "runs": [Run]}


# ── data loading ─────────────────────────────────────────────────────────────


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


# ── context model: estimate I_0 and delta from trajectory sample ──────────────


def _traj_context(
    cache: Path, system: str, instance_id: str
) -> tuple[int, int, int] | None:
    """(I_0, step_delta, out_per_step) in tokens from one trajectory.

    Two formats:
    - OpenAI Responses API (mini-swe-agent ≥1.1 with GPT): messages contain response objects
      with a `usage` field carrying per-step token counts (authoritative).
    - Claude / tool-call format: messages have role-based structure; estimate from char lengths.
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

    # ── OpenAI Responses API format: messages contain response objects with usage ──
    response_usages = [
        m["usage"]
        for m in msgs
        if isinstance(m, dict) and "usage" in m and isinstance(m["usage"], dict)
    ]
    if response_usages:
        # Extract per-step input/output token counts
        input_toks = [u.get("input_tokens", 0) or 0 for u in response_usages]
        output_toks = [u.get("output_tokens", 0) or 0 for u in response_usages]
        if not input_toks:
            return None
        I_0 = input_toks[0]  # first call's input = initial context
        delta = (
            (input_toks[-1] - input_toks[0]) // max(1, len(input_toks) - 1)
            if len(input_toks) > 1
            else 0
        )
        out = sum(output_toks) // len(output_toks) if output_toks else 0
        return I_0, max(0, delta), out

    # ── Claude / role-based message format: estimate from char lengths ──
    def mlen(m: dict) -> int:
        c = m.get("content") or ""
        tc = m.get("tool_calls") or []
        chars = len(c) if isinstance(c, str) else len(json.dumps(c))
        chars += sum(len(json.dumps(t)) for t in tc)
        return int(chars / CHARS_PER_TOKEN)

    roles: dict[str, int] = {}
    for m in msgs:
        r = m.get("role", "?")
        roles[r] = roles.get(r, 0) + mlen(m)

    # Initial context = system + first user message
    I_0 = roles.get("system", 0) + roles.get("user", 0)
    # Step delta = (assistant tool_calls + tool results) / n_steps
    step_total = roles.get("assistant", 0) + roles.get("tool", 0)
    out_total = roles.get("assistant", 0)
    return I_0, step_total // n_steps, out_total // n_steps


def calibrate_context(cache: Path, system: str, ids: list[str]) -> tuple[int, int, int]:
    """Median (I_0, delta, out_per_step) from up to CALIB_N trajectories. Falls back to hardcoded."""
    sample = ids[:CALIB_N]

    def fetch(i: str) -> tuple[int, int, int] | None:
        return _traj_context(cache, system, i)

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(fetch, sample))

    valid = [r for r in results if r is not None]
    if not valid:
        print(
            f"  [context model] fallback to hardcoded defaults (no trajectories fetched for {system})"
        )
        return FALLBACK_I0, FALLBACK_DELTA, FALLBACK_OUT

    def median(xs: list[int]) -> int:
        s = sorted(xs)
        n = len(s)
        return s[n // 2]

    I_0 = median([r[0] for r in valid])
    delta = median([r[1] for r in valid])
    out = median([r[2] for r in valid])
    print(
        f"  [context model] calibrated from {len(valid)}/{len(sample)} trajectories: "
        f"I_0={I_0}, delta={delta}, out_per_step={out}"
    )
    return I_0, delta, out


def model_suffix(system: str) -> str:
    """Extract price-table key from leaderboard system name."""
    # e.g. "20260217_mini-v2.0.0_claude-4-5-haiku-high" → "claude-4-5-haiku-high"
    parts = system.split("_", 3)
    return parts[-1] if len(parts) >= 3 else system


def get_price(system: str) -> tuple[float, float]:
    """(price_in, price_out) in $/1M tokens. Returns (0,0) if unknown."""
    return MODEL_PRICES.get(model_suffix(system), (0.0, 0.0))


# ── analytical cost models ────────────────────────────────────────────────────


def nocache_cost(
    N: int, I0: int, delta: int, out: int, price_in: float, price_out: float
) -> float:
    """Estimated cost of N API calls with NO provider caching (flat token prices)."""
    total_in = sum(I0 + k * delta for k in range(N))
    total_out = N * out
    return (total_in * price_in + total_out * price_out) / 1e6


def withcache_cost(
    N: int, I0: int, delta: int, out: int, price_in: float, price_out: float
) -> float:
    """Estimated cost of N API calls WITH prefix caching (1.25× write, 0.1× read)."""
    if N == 0:
        return 0.0
    # Step 1: write initial context
    cost = I0 * CACHE_WRITE_MULT * price_in / 1e6
    cost += out * price_out / 1e6
    # Steps 2..N: read accumulated prefix + write new tokens
    for k in range(1, N):
        prefix = I0 + k * delta
        cost += (prefix * CACHE_READ_MULT + delta * CACHE_WRITE_MULT) * price_in / 1e6
        cost += out * price_out / 1e6
    return cost


def switch_overhead_usd(tau: int, price_top_in: float, I0: int, delta: int) -> float:
    """Extra cost when top model must re-read accumulated prefix at full uncached rate.
    Per pre-registration: top model's first call pays 1.0× input for the prefix.
    tau < 0 is treated as 0 (switch before any cheap model steps: context = I0 only)."""
    actual_steps = max(0, tau)
    context_at_tau = I0 + actual_steps * delta
    return context_at_tau * price_top_in / 1e6


def nc_factor(
    N: int, I0: int, delta: int, out: int, price_in: float, price_out: float
) -> float:
    """no_cache / with_cache ratio for a single-model session (>1 means caching saves money)."""
    wc = withcache_cost(N, I0, delta, out, price_in, price_out)
    nc = nocache_cost(N, I0, delta, out, price_in, price_out)
    return nc / wc if wc > 0 else 1.0


# ── policy helpers (shared with analyze-agent-cascade.py) ────────────────────


def total(outs: list[tuple[bool, float]]) -> tuple[int, float]:
    return sum(o for o, _ in outs), sum(c for _, c in outs)


def paired_ci(
    a: list[tuple[bool, float]],
    b: list[tuple[bool, float]],
    n_boot: int = 4000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Paired instance bootstrap of a − b: Δresolved-rate, Δ$/instance, Δ$/resolved (95% CI)."""
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
    """Per-call cascade: serve first rung whose step count passes gate, or last rung."""
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


def fit_taus(train: list[Row]) -> list[float]:
    """Per-rung step thresholds, cross-fit by step-count quantiles."""
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
    """Index of cheapest pool model within eps of best resolve rate on training set."""
    n_models = len(train[0])
    n = len(train)
    rates = [sum(r[k][0] for r in train) / n for k in range(n_models)]
    costs = [sum(r[k][1] for r in train) for k in range(n_models)]
    ok = [k for k in range(n_models) if rates[k] >= max(rates) - eps]
    return min(ok, key=lambda k: costs[k])


# ── escalate-and-pin policy ───────────────────────────────────────────────────


def escalate_pin(
    row: Row,
    tau: float,
    I0: int,
    delta: int,
    price_top_in: float,
) -> tuple[bool, float, float]:
    """(resolved, cost_with_cache, cost_no_cache).
    cost_with_cache includes switch overhead; cost_no_cache does not.
    """
    cheap_res, cheap_cost, cheap_steps = row["runs"][0]
    top_res, top_cost, top_steps = row["runs"][-1]
    # Cheap model's no-cache factor: scale observed cost up
    p_cheap_in, p_cheap_out = get_price(row.get("_cheap_sys", ""))
    if cheap_steps > 0 and p_cheap_in > 0:
        wc_cheap = withcache_cost(
            cheap_steps, I0, delta, FALLBACK_OUT, p_cheap_in, p_cheap_out
        )
        nc_cheap_factor = (
            nocache_cost(cheap_steps, I0, delta, FALLBACK_OUT, p_cheap_in, p_cheap_out)
            / wc_cheap
            if wc_cheap > 0
            else 1.0
        )
    else:
        nc_cheap_factor = 3.0  # conservative fallback

    p_top_in, p_top_out = get_price(row.get("_top_sys", ""))
    if top_steps > 0 and p_top_in > 0:
        wc_top = withcache_cost(top_steps, I0, delta, FALLBACK_OUT, p_top_in, p_top_out)
        nc_top_factor = (
            nocache_cost(top_steps, I0, delta, FALLBACK_OUT, p_top_in, p_top_out)
            / wc_top
            if wc_top > 0
            else 1.0
        )
    else:
        nc_top_factor = 3.0

    if cheap_steps <= tau:
        # No escalation: serve cheap model's outcome as-is
        cost_withcache = cheap_cost
        cost_nocache = cheap_cost * nc_cheap_factor
        return cheap_res, cost_withcache, cost_nocache

    # Escalation fires at step max(0, tau). tau=-1 means switch before step 0 (no cheap steps run).
    actual_tau = max(0, int(tau))
    partial_frac = actual_tau / cheap_steps if cheap_steps > 0 else 0.0
    partial_cheap_obs = cheap_cost * partial_frac
    # RESTART semantics: the top model's recorded run starts from scratch, so its recorded cost
    # already includes its own uncached first read (a cache write of the initial prompt). Adding
    # a separate "switch overhead" here would charge that read twice. A mid-run HANDOFF would pay
    # the overhead on the accumulated context instead, but its quality is not identifiable by
    # replay (The Replay Gap, arXiv 2608.08239), so it is not scored here.
    del price_top_in  # kept in the signature for the handoff cost-only analysis below

    cost_withcache = partial_cheap_obs + top_cost
    # Without caching: no switch overhead; scale each model's portion
    cost_nocache = partial_cheap_obs * nc_cheap_factor + top_cost * nc_top_factor
    return top_res, cost_withcache, cost_nocache


def fit_tau_pin(
    train: list[Row],
    I0: int,
    delta: int,
    price_top_in: float,
) -> float:
    """Cross-fit tau_pin for escalate-and-pin using cache-aware cost."""
    steps = sorted(r["runs"][0][2] for r in train)
    grid = sorted(
        {steps[min(int(q * (len(steps) - 1)), len(steps) - 1)] for q in QUANTILES}
        | {-1}
    )
    bar = sum(r["runs"][-1][0] for r in train)

    def run_policy(tau: float) -> tuple[int, float]:
        outs = [escalate_pin(r, tau, I0, delta, price_top_in)[:2] for r in train]
        return sum(o for o, _ in outs), sum(c for _, c in outs)

    scored = [(*run_policy(tau), tau) for tau in grid]
    feasible = [(cost, tau) for ok, cost, tau in scored if ok >= bar]
    return min(feasible)[1] if feasible else max(scored, key=lambda x: (x[0], -x[1]))[2]


# ── workload-level selection report (replicate #239) ─────────────────────────


def workload_selection(cache: Path) -> dict[str, list[tuple[bool, float]]]:
    """Policy P2: per-workload selection from 11-model pool. Returns two arms: selected, always-opus."""
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
    ref_idx = next(k for k, s in enumerate(POOL) if "opus-high" in s and "4-5" in s)
    ref = [r[ref_idx] for r in res]
    return {"pool-selection (cross-fit)": sel, "always-opus (pool ref)": ref}


# ── main analysis per ladder ──────────────────────────────────────────────────


def analyse_ladder(
    cache: Path,
    name: str,
    ladder: list[str],
    ctx: tuple[int, int, int],
) -> None:
    rows = rows_for(cache, ladder)
    n = len(rows)
    cheap_sys, top_sys = ladder[0], ladder[-1]
    _, price_top_in = get_price(top_sys), get_price(top_sys)[0]
    price_top_in = get_price(top_sys)[0]
    price_cheap_in, price_cheap_out = get_price(cheap_sys)
    I0, delta, out_per_step = ctx

    # Annotate rows with system names for escalate-and-pin
    for row in rows:
        row["_cheap_sys"] = model_suffix(cheap_sys)
        row["_top_sys"] = model_suffix(top_sys)

    # Cross-fit
    halves = [rows[0::2], rows[1::2]]
    fp: list[tuple[bool, float] | None] = [None] * n
    rp: list[tuple[bool, float] | None] = [None] * n
    ep_wc: list[tuple[bool, float] | None] = [None] * n  # escalate-and-pin with-cache
    ep_nc: list[tuple[bool, float] | None] = [None] * n  # escalate-and-pin no-cache
    fitted_cascade = []
    fitted_pin = []
    fitted_eps_val = []

    for h in (0, 1):
        train = halves[1 - h]
        taus = fit_taus(train)
        tau_pin = fit_tau_pin(train, I0, delta, price_top_in)
        eps_val = fit_eps(train)
        route = repo_router(train, eps_val)
        fitted_cascade.append(taus)
        fitted_pin.append(tau_pin)
        fitted_eps_val.append(eps_val)
        for j in range(h, n, 2):
            fp[j] = cascade(rows[j], taus)
            rp[j] = route(rows[j])
            res, cwc, cnc = escalate_pin(rows[j], tau_pin, I0, delta, price_top_in)
            ep_wc[j] = (res, cwc)
            ep_nc[j] = (res, cnc)

    # Compute no-cache scaling for single-model policies (for the nc table column)
    # Estimate average nc_factor from the cheap model across instances
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

    # Arms: (observed_cost_tuple, no_cache_cost_tuple)
    # For single-model policies: nc cost ≈ observed × nc_factor; wc cost = observed
    # For cascade: two independent sessions; both scaled by nc_factor (no cross-model effect)
    # For escalate-and-pin: ep_wc (includes switch overhead), ep_nc (no switch overhead)
    always_cheap = [(r["runs"][0][0], r["runs"][0][1]) for r in rows]
    always_top = [(r["runs"][-1][0], r["runs"][-1][1]) for r in rows]

    # Top-model nc_factor (for "always-top no-cache" estimate)
    p_top_in_full, p_top_out = get_price(top_sys)
    top_nc_factors = []
    for row in rows:
        steps = row["runs"][-1][2]
        if steps > 0 and p_top_in_full > 0:
            wc = withcache_cost(
                steps, I0, delta, out_per_step, p_top_in_full, p_top_out
            )
            nc = nocache_cost(steps, I0, delta, out_per_step, p_top_in_full, p_top_out)
            if wc > 0:
                top_nc_factors.append(nc / wc)
    avg_top_nc = sum(top_nc_factors) / len(top_nc_factors) if top_nc_factors else 3.0

    def scale_nc(
        arm: list[tuple[bool, float]], factor: float
    ) -> list[tuple[bool, float]]:
        return [(ok, cost * factor) for ok, cost in arm]

    arms_wc: dict[str, list[tuple[bool, float]]] = {
        "always-cheap": always_cheap,
        "always-top": always_top,
        "per-call cascade (cross-fit)": list(fp),  # type: ignore[arg-type]
        "per-session / repo (cross-fit)": list(rp),  # type: ignore[arg-type]
        "escalate-and-pin (cross-fit)": list(ep_wc),  # type: ignore[arg-type]
        "oracle-per-instance (bound)": [oracle_predictive(r) for r in rows],
        "oracle-cascade (bound)": [oracle_cascade(r) for r in rows],
    }
    # cascade nc: both cheap and top sessions are independent; scale observed cost
    cascade_nc = [
        (ok, cost * (avg_cheap_nc if cost < 0.5 else avg_top_nc))
        for ok, cost in fp  # type: ignore[union-attr]
    ]
    arms_nc: dict[str, list[tuple[bool, float]]] = {
        "always-cheap": scale_nc(always_cheap, avg_cheap_nc),
        "always-top": scale_nc(always_top, avg_top_nc),
        "per-call cascade (cross-fit)": cascade_nc,
        "per-session / repo (cross-fit)": scale_nc(list(rp), avg_cheap_nc),  # type: ignore[arg-type]
        "escalate-and-pin (cross-fit)": list(ep_nc),  # type: ignore[arg-type]
        "oracle-per-instance (bound)": scale_nc(
            [oracle_predictive(r) for r in rows], avg_cheap_nc
        ),
        "oracle-cascade (bound)": scale_nc(
            [oracle_cascade(r) for r in rows], avg_top_nc
        ),
    }

    top_wc = arms_wc["always-top"]

    print(f"## {name}  (n={n})\n")
    print(f"rungs: {model_suffix(cheap_sys)} → {model_suffix(top_sys)}")
    print(
        f"cascade tau per half: {fitted_cascade}  repo-eps per half: {fitted_eps_val}"
    )
    print(f"escalate-and-pin tau per half: {fitted_pin}")
    print(f"context model: I_0={I0}, delta={delta}, out/step={out_per_step} tokens")
    print(
        f"avg no-cache factor: cheap={avg_cheap_nc:.2f}×, top={avg_top_nc:.2f}× (observed cost with caching ÷ estimated without)"
    )
    print()
    print(
        "| arm | resolved | $/inst (observed, w/ caching) | $/inst (est. no caching) | $/resolved (w/ caching) | Δ resolved vs always-top | Δ $/inst vs always-top |"
    )
    print("|---|---|---|---|---|---|---|")
    for arm in arms_wc:
        wc = arms_wc[arm]
        nc = arms_nc[arm]
        ok_w, cost_w = total(wc)
        _, cost_n = total(nc)
        dr_ci = "— | —"
        if arm != "always-top":
            (r0, r1), (c0, c1), _ = paired_ci(wc, top_wc)
            dr_ci = f"[{r0:+.3f}, {r1:+.3f}] | [{c0:+.3f}, {c1:+.3f}]"
        print(
            f"| {arm} | {ok_w}/{n} ({ok_w / n:.3f}) | ${cost_w / n:.3f} | ${cost_n / n:.3f} | ${cost_w / ok_w:.3f} | {dr_ci} |"
        )
    print()

    # Switch overhead breakdown for escalate-and-pin
    n_switched = sum(1 for row in rows if row["runs"][0][2] > fitted_pin[0])
    avg_overhead = (
        sum(
            switch_overhead_usd(int(fitted_pin[0]), price_top_in, I0, delta)
            for row in rows
            if row["runs"][0][2] > fitted_pin[0]
        )
        / n_switched
        if n_switched > 0
        else 0.0
    )
    print(
        f"  escalate-and-pin: {n_switched}/{n} instances trigger escalation at tau={fitted_pin[0]}"
    )
    print(
        f"  [cost-only] a mid-run HANDOFF at this tau would re-read "
        f"{I0 + max(0, int(fitted_pin[0])) * delta:,} tokens uncached (${avg_overhead:.3f}/instance); "
        "not charged to escalate-by-restart, whose top-model run already includes it"
    )
    print(
        "  [validity] escalate-by-restart = cheap run to the trigger + the top model's own full run "
        "(exact, no stitching); per-call cascade quality is NOT identifiable by replay\n"
    )


# ── main report ───────────────────────────────────────────────────────────────


def report(cache: Path) -> None:
    print(
        f"# Routing granularity for multi-turn coding agents (SWE-bench Verified, data @{PINNED[:7]})\n"
    )
    print("## Validity caveat\n")
    print(
        "Trajectories are recorded runs; mid-session switches cannot be replayed. "
        "Escalate-and-pin post-switch outcome ≈ top model's full run (optimistic for quality, "
        "pessimistic for cost). Cache model estimated from trajectory sample (2.5 chars/token). "
        "See pre-registration doc for full validity discussion.\n"
    )

    for lname, ladder in LADDERS.items():
        cheap_sys = ladder[0]
        print(f"### Calibrating context model for {model_suffix(cheap_sys)} ...")
        data = load(cache, cheap_sys)
        ids = sorted(data.keys())
        ctx = calibrate_context(cache, cheap_sys, ids)
        print()
        analyse_ladder(cache, lname, ladder, ctx)

    # Workload-level selection (policy P2, replication of #239)
    print(
        "## Policy P2: workload-level selection from 11-model pool (cross-fit, n=500)\n"
    )
    wl = workload_selection(cache)
    top_ref = wl["always-opus (pool ref)"]
    n = len(top_ref)
    print(
        "| arm | resolved | $/instance | Δ resolved vs pool-reference | Δ $/inst vs pool-reference |"
    )
    print("|---|---|---|---|---|")
    for arm, outs in wl.items():
        ok, cost = total(outs)
        if arm == "always-opus (pool ref)":
            cis = "— | —"
        else:
            (r0, r1), (c0, c1), _ = paired_ci(outs, top_ref)
            cis = f"[{r0:+.3f}, {r1:+.3f}] | [{c0:+.3f}, {c1:+.3f}]"
        print(f"| {arm} | {ok}/{n} ({ok / n:.3f}) | ${cost / n:.3f} | {cis} |")
    print()


# ── self-check ────────────────────────────────────────────────────────────────


def _self_check() -> None:
    # no-cache cost grows quadratically with session length
    nc_10 = nocache_cost(10, 1000, 500, 100, 1.0, 5.0)
    wc_10 = withcache_cost(10, 1000, 500, 100, 1.0, 5.0)
    assert nc_10 > wc_10, "uncached must cost more than cached"
    # longer session → higher nc/wc ratio (caching becomes more valuable)
    nc_factor_10 = nc_10 / wc_10
    nc_factor_50 = nocache_cost(50, 1000, 500, 100, 1.0, 5.0) / withcache_cost(
        50, 1000, 500, 100, 1.0, 5.0
    )
    assert nc_factor_50 > nc_factor_10, "longer sessions benefit more from caching"
    # switch overhead scales with context size at tau
    sw5 = switch_overhead_usd(5, 15.0, 1000, 500)
    sw20 = switch_overhead_usd(20, 15.0, 1000, 500)
    assert sw20 > sw5, "later switch = larger context = higher overhead"
    # escalate-and-pin: no escalation below tau
    row = {
        "id": "x__1",
        "repo": "x",
        "runs": [(True, 1.0, 10), (True, 5.0, 20)],
        "_cheap_sys": "claude-4-5-haiku-high",
        "_top_sys": "claude-4-5-opus-high",
    }
    res, cwc, cnc = escalate_pin(row, tau=15.0, I0=1000, delta=300, price_top_in=15.0)
    assert res is True and cwc == 1.0, "below tau: serve cheap outcome at cheap cost"
    # escalate-and-pin: escalation fires at tau=5 (steps=10 > tau)
    res2, cwc2, cnc2 = escalate_pin(row, tau=5.0, I0=1000, delta=300, price_top_in=15.0)
    assert res2 is True and cwc2 > 5.0, "above tau: pay top model + overhead"
    overhead = switch_overhead_usd(5, 15.0, 1000, 300)
    # Restart semantics: partial cheap spend + the top model's full run; no separate overhead
    # (the top run's own first read is already in its recorded cost).
    expected_cwc = 1.0 * (5 / 10) + 5.0
    assert overhead > 0  # the handoff re-read is still computed, for the cost-only illustration
    assert abs(cwc2 - expected_cwc) < 1e-9, f"got {cwc2}, expected {expected_cwc}"
    # switch overhead is present in cwc2 but absent from cnc2 (no-cache has no cache-miss penalty)
    # cnc2 scales up the observed costs by the no-cache multiplier but does NOT add switch overhead
    no_switch_wc = 1.0 * (5 / 10) + 5.0  # partial_cheap + top, no overhead
    assert cwc2 > no_switch_wc, "switch overhead increases with-cache cost"
    # no-cache costs MORE than with-cache (caching saves money even with the switch overhead)
    assert cnc2 > cwc2, (
        "no-cache total > with-cache total (caching always saves, even with switch overhead)"
    )
    # switch overhead is the entire delta between cwc2 and no-switch cost
    assert abs((cwc2 - no_switch_wc) - overhead) < 1e-9, (
        "overhead is exactly the cache-miss cost"
    )
    # tau=-1 boundary: treated as tau=0, so overhead = I0 × price_top (no cheap steps accumulated)
    _, cwc_neg1, _ = escalate_pin(row, tau=-1.0, I0=1000, delta=300, price_top_in=15.0)
    overhead_at_0 = switch_overhead_usd(0, 15.0, 1000, 300)  # I0 × price_top
    assert overhead_at_0 == 1000 * 15.0 / 1e6, "tau=0 overhead = I0 × price_top"
    expected_tau_neg1 = 0.0 + overhead_at_0 + 5.0  # 0 partial cheap + overhead + top
    assert abs(cwc_neg1 - expected_tau_neg1) < 1e-9, (
        f"tau=-1: {cwc_neg1} vs {expected_tau_neg1}"
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
