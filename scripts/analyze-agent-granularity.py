#!/usr/bin/env python3
"""Routing granularity for multi-turn coding agents: per-workload vs per-session vs mid-session.

Compares routing granularities on n=500 SWE-bench Verified mini-SWE-agent instances.
Policies and decision rules pre-registered in docs/benchmarks/agent-granularity-preregistration.md.

Usage: analyze-agent-granularity.py [CACHE_DIR]  |  analyze-agent-granularity.py --self-check
"""

from __future__ import annotations

import json
import re
import os
import random
import sys
import tempfile
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
# Index of the quality-reference model (claude-4-5-opus-high) within POOL.
POOL_REF_IDX: int = next(k for k, s in enumerate(POOL) if "claude-4-5-opus-high" in s)

# Price table: $/1M tokens. Used for P4 switch-overhead and the handoff illustration.
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
CHARS_PER_TOKEN = 2.5  # role-based trajectory format only

FALLBACK_I0 = 2333
FALLBACK_DELTA = 574
FALLBACK_OUT = 219
CALIB_N = 20
CALIB_SEED = 42  # seeded random for calibration sample (not alphabetical)

QUANTILES = [i / 10 for i in range(11)]
EPS_GRID = [0.0, 0.05, 0.1, 0.2, 0.3]

Run = tuple[bool, float, int]  # resolved, cost_usd, api_calls
Row = dict  # {"id", "repo", "runs": [Run]}


# ── data loading ──────────────────────────────────────────────────────────────


def _atomic_write(path: Path, data: bytes) -> None:
    """Write data atomically via a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load(cache: Path, system: str) -> dict:
    path = cache / f"{system}.json"
    if not path.exists():
        with urllib.request.urlopen(
            f"{RAW}/{system}/per_instance_details.json", timeout=60
        ) as r:
            _atomic_write(path, r.read())
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


# ── context model ─────────────────────────────────────────────────────────────


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _fetch_traj(cache: Path, system: str, instance_id: str) -> bytes | None:
    """Fetch trajectory JSON, writing atomically and retrying once on decode failure."""
    # Names come from downloaded data and become cache file names and URL parts: refuse
    # anything that could escape the cache directory or the URL path.
    if not (_SAFE_NAME.match(system) and _SAFE_NAME.match(instance_id)) or ".." in instance_id:
        return None
    path = cache / f"{system}.ctx.{instance_id}.json"
    if path.exists():
        try:
            json.loads(path.read_text())
            return path.read_bytes()
        except (json.JSONDecodeError, OSError):
            path.unlink(missing_ok=True)
    # Fetch (or re-fetch after corrupt cache)
    try:
        url = TRAJ.format(s=system, i=instance_id)
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read()
    except OSError:
        return None
    _atomic_write(path, raw)
    return raw


def _traj_context(
    cache: Path, system: str, instance_id: str
) -> tuple[int, int, int] | None:
    """(I_0, step_delta, out_per_step) in tokens.

    I_0 = system prompt + first user message only.
    Later messages count toward per-step growth (delta).

    Two formats:
    - OpenAI Responses API: 'usage' field per response gives authoritative token counts.
    - Claude role-based: char-length estimate; I_0 from system + FIRST user message only.
    """
    raw = _fetch_traj(cache, system, instance_id)
    if raw is None:
        return None
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    msgs = d.get("messages") or d.get("history") or []
    if not msgs:
        return None
    n_steps = ((d.get("info") or {}).get("model_stats") or {}).get("api_calls") or 0
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
    """Median (I_0, delta, out_per_step) from CALIB_N seeded-random trajectory samples."""
    rng = random.Random(CALIB_SEED)
    sample = rng.sample(ids, min(CALIB_N, len(ids)))

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
        f"  [context model] seeded-random sample ({len(valid)}/{len(sample)}): "
        f"I_0={I_0}, delta={delta}, out_per_step={out}"
    )
    return I_0, delta, out


def model_suffix(system: str) -> str:
    parts = system.split("_", 3)
    return parts[-1] if len(parts) >= 3 else system


def get_price(system: str) -> tuple[float, float]:
    return MODEL_PRICES.get(model_suffix(system), (0.0, 0.0))


# ── cost model ────────────────────────────────────────────────────────────────


def nocache_cost(
    N: int, I0: int, delta: int, out: int, price_in: float, price_out: float
) -> float:
    """Estimated cost without provider caching (flat token pricing)."""
    total_in = sum(I0 + k * delta for k in range(N))
    return (total_in * price_in + N * out * price_out) / 1e6


def withcache_cost(
    N: int, I0: int, delta: int, out: int, price_in: float, price_out: float
) -> float:
    """Estimated cost with prefix caching (1.25× write, 0.1× read).

    At step k (k=0..N-1):
      k=0: write I0 tokens (no prior cache); output out tokens.
      k>=1: read cached prefix I0+(k-1)*delta at 0.1×; write delta new tokens at 1.25×.
    """
    if N == 0:
        return 0.0
    cost = I0 * CACHE_WRITE_MULT * price_in / 1e6 + out * price_out / 1e6
    for k in range(1, N):
        prefix = I0 + (k - 1) * delta  # cached prefix BEFORE this step's new tokens
        cost += (prefix * CACHE_READ_MULT + delta * CACHE_WRITE_MULT) * price_in / 1e6
        cost += out * price_out / 1e6
    return cost


def switch_overhead_usd(tau: int, price_top_in: float, I0: int, delta: int) -> float:
    """Cost for top model's uncached re-read of accumulated context at a mid-session switch.

    Pre-registration P4 cost model: the top model's first call pays the full (uncached) input
    price for the prefix accumulated by the cheap model.
    tau < 0 is treated as 0 (switch before any cheap step: context = I_0 only).
    """
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
    """Paired bootstrap CI a − b: [Δresolved-rate, Δ$/inst, Δ$/resolved] (95%)."""
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
        if not v:  # an arm with zero resolves has no $/resolved distribution
            out.append((float("nan"), float("nan")))
            continue
        v.sort()
        out.append((v[int(0.025 * len(v))], v[int(0.975 * len(v))]))
    return out


def cascade(row: Row, taus: list[float]) -> tuple[bool, float]:
    """Run-level cascade (P5): run cheap to completion; if api_calls <= tau serve cheap,
    else run top from scratch. Both are whole recorded runs; quality IS identifiable."""
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
    """P6 random: cheap or top with equal probability per instance (seeded)."""
    rng = random.Random(seed)
    return [
        (r["runs"][0][0], r["runs"][0][1])
        if rng.random() < 0.5
        else (r["runs"][-1][0], r["runs"][-1][1])
        for r in rows
    ]


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
    """Route each instance to the cheapest model within eps of the best resolve rate.

    For each repo (or overall if unseen): compute per-rung resolve rates and average costs
    on training instances, then pick the minimum-cost rung whose rate is within eps of max.
    This is the same selection rule as select_cheapest (cheapest within quality band).
    """
    n_rungs = len(train[0]["runs"])
    # per-repo: [resolved_sum, count, cost_sum] per rung
    stats: dict[str, list[list[float]]] = {}
    for r in train:
        s = stats.setdefault(r["repo"], [[0.0, 0.0, 0.0] for _ in range(n_rungs)])
        for k, (ok, cost, _) in enumerate(r["runs"]):
            s[k][0] += ok
            s[k][1] += 1.0
            s[k][2] += cost
    overall_rate = [
        sum(r["runs"][k][0] for r in train) / len(train) for k in range(n_rungs)
    ]
    overall_avg_cost = [
        sum(r["runs"][k][1] for r in train) / len(train) for k in range(n_rungs)
    ]

    def route(row: Row) -> tuple[bool, float]:
        s = stats.get(row["repo"])
        if s:
            rates = [s[k][0] / s[k][1] for k in range(n_rungs)]
            avg_costs = [s[k][2] / s[k][1] for k in range(n_rungs)]
        else:
            rates = overall_rate
            avg_costs = overall_avg_cost
        eligible = [k for k in range(n_rungs) if rates[k] >= max(rates) - eps]
        k = min(eligible, key=lambda k: avg_costs[k])
        ok, cost, _ = row["runs"][k]
        return ok, cost

    return route


def fit_eps(train: list[Row], ref_idx: int = -1) -> float:
    """Cross-fit eps: cheapest eps (from EPS_GRID) where repo router resolved >= ref model.

    ref_idx: index into row["runs"] of the quality reference model. Default -1 (last rung)
    is correct for 2-model ladders. For the 11-model pool pass POOL_REF_IDX (opus-high).
    """
    routers = {eps: repo_router(train, eps) for eps in EPS_GRID}
    bar = sum(r["runs"][ref_idx][0] for r in train)
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


# ── P4: escalate-and-pin, two variants ───────────────────────────────────────


def escalate_pin_prereg(
    row: Row, tau: float, I0: int, delta: int, price_top_in: float
) -> tuple[bool, float]:
    """P4 (pre-registration): partial_cheap + switch_overhead + top_cost.

    Pre-registration cost model: the top model's first call re-reads the accumulated
    prefix at the full uncached input rate (switch_overhead). This is the PRIMARY P4 arm.
    Trigger: escalate when cheap api_calls >= tau_pin.
    """
    cheap_res, cheap_cost, cheap_steps = row["runs"][0]
    top_res, top_cost, _ = row["runs"][-1]
    if cheap_steps < tau:
        return cheap_res, cheap_cost
    actual_tau = max(0, int(tau))
    if actual_tau == 0:
        # Escalating before the cheap model has taken a step is not a switch: the top model
        # just runs, and its recorded cost already includes its own first read.
        return top_res, top_cost
    partial_frac = actual_tau / cheap_steps if cheap_steps > 0 else 0.0
    overhead = switch_overhead_usd(actual_tau, price_top_in, I0, delta)
    return top_res, cheap_cost * partial_frac + overhead + top_cost


def escalate_pin_restart(row: Row, tau: float) -> tuple[bool, float]:
    """P4 restart-semantics (secondary): partial_cheap + top_cost.

    No separate switch overhead because the top model restarts from scratch and its
    full recorded run cost already includes its own first (uncached) read. Listed as
    a deviation in the results doc.
    """
    cheap_res, cheap_cost, cheap_steps = row["runs"][0]
    top_res, top_cost, _ = row["runs"][-1]
    if cheap_steps < tau:
        return cheap_res, cheap_cost
    actual_tau = max(0, int(tau))
    partial_frac = actual_tau / cheap_steps if cheap_steps > 0 else 0.0
    return top_res, cheap_cost * partial_frac + top_cost


def fit_tau_pin_prereg(
    train: list[Row], I0: int, delta: int, price_top_in: float
) -> float:
    """Cross-fit tau_pin for P4 pre-registration variant (switch overhead in objective)."""
    steps = sorted(r["runs"][0][2] for r in train)
    grid = sorted(
        {steps[min(int(q * (len(steps) - 1)), len(steps) - 1)] for q in QUANTILES}
        | {-1}
    )
    bar = sum(r["runs"][-1][0] for r in train)

    def run_policy(tau: float) -> tuple[int, float]:
        outs = [escalate_pin_prereg(r, tau, I0, delta, price_top_in) for r in train]
        return sum(o for o, _ in outs), sum(c for _, c in outs)

    scored = [(*run_policy(tau), tau) for tau in grid]
    feasible = [(cost, tau) for ok, cost, tau in scored if ok >= bar]
    return min(feasible)[1] if feasible else max(scored, key=lambda x: (x[0], -x[1]))[2]


def fit_tau_pin_restart(train: list[Row]) -> float:
    """Cross-fit tau_pin for P4 restart-semantics variant (no switch overhead)."""
    steps = sorted(r["runs"][0][2] for r in train)
    grid = sorted(
        {steps[min(int(q * (len(steps) - 1)), len(steps) - 1)] for q in QUANTILES}
        | {-1}
    )
    bar = sum(r["runs"][-1][0] for r in train)

    def run_policy(tau: float) -> tuple[int, float]:
        outs = [escalate_pin_restart(r, tau) for r in train]
        return sum(o for o, _ in outs), sum(c for _, c in outs)

    scored = [(*run_policy(tau), tau) for tau in grid]
    feasible = [(cost, tau) for ok, cost, tau in scored if ok >= bar]
    return min(feasible)[1] if feasible else max(scored, key=lambda x: (x[0], -x[1]))[2]


# ── pool helpers ──────────────────────────────────────────────────────────────


def load_pool_data(cache: Path) -> tuple[list[str], list[dict]]:
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
    ref = [r[POOL_REF_IDX] for r in res]
    return {"pool-selection (P2, cross-fit)": sel, "always-opus (pool ref)": ref}


def pool_session_selection(
    ids: list[str], data: list[dict]
) -> tuple[list[tuple[bool, float]], list[float]]:
    """P3-pool: per-session repo-level selection from 11-model pool (cross-fit).

    Uses POOL_REF_IDX as the quality bar for fit_eps (opus-high, not minimax).
    repo_router picks the cheapest eligible model, not the first in list order.
    """
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
        eps_val = fit_eps(train, ref_idx=POOL_REF_IDX)
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
    """Analyse one ladder. Returns (instance_ids, p3_outcomes) for cross-ladder CI."""
    rows = rows_for(cache, ladder)
    n = len(rows)
    cheap_sys, top_sys = ladder[0], ladder[-1]
    price_top_in = get_price(top_sys)[0]
    I0, delta, out_per_step = ctx

    # Informational: avg nc_factor for cheap model (not in table; pure-analytical estimate)
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
    ep_pre: list[tuple[bool, float] | None] = [None] * n
    ep_rst: list[tuple[bool, float] | None] = [None] * n
    fitted_cascade: list[list[float]] = []
    fitted_pin_pre: list[float] = []
    fitted_pin_rst: list[float] = []
    fitted_eps_val: list[float] = []

    for h in (0, 1):
        train = halves[1 - h]
        taus = fit_taus(train)
        tau_pre = fit_tau_pin_prereg(train, I0, delta, price_top_in)
        tau_rst = fit_tau_pin_restart(train)
        eps_val = fit_eps(train)  # 2-model ladder: ref_idx=-1 (top = last = correct)
        route = repo_router(train, eps_val)
        fitted_cascade.append(taus)
        fitted_pin_pre.append(tau_pre)
        fitted_pin_rst.append(tau_rst)
        fitted_eps_val.append(eps_val)
        for j in range(h, n, 2):
            fp[j] = cascade(rows[j], taus)
            rp[j] = route(rows[j])
            ep_pre[j] = escalate_pin_prereg(rows[j], tau_pre, I0, delta, price_top_in)
            ep_rst[j] = escalate_pin_restart(rows[j], tau_rst)

    print(f"## {name}  (n={n})\n")
    print(
        f"cascade tau per fold: {fitted_cascade}  repo-eps per fold: {fitted_eps_val}"
    )
    print(f"P4 tau per fold: prereg={fitted_pin_pre}  restart={fitted_pin_rst}")
    print(
        f"context model (cheap model, seeded-random sample): "
        f"I_0={I0}, delta={delta}, out/step={out_per_step}  (avg nc_factor={avg_cheap_nc:.2f}×)"
    )
    print()

    always_cheap = [(r["runs"][0][0], r["runs"][0][1]) for r in rows]
    always_top = [(r["runs"][-1][0], r["runs"][-1][1]) for r in rows]

    arms: dict[str, list[tuple[bool, float]]] = {
        "always-cheap [P1]": always_cheap,
        "always-top [P1]": always_top,
        "run-level cascade (cross-fit) [P5]": list(fp),  # type: ignore[arg-type]
        "per-session / repo (cross-fit) [P3]": list(rp),  # type: ignore[arg-type]
        "escalate-and-pin prereg (cross-fit) [P4]": list(ep_pre),  # type: ignore[arg-type]
        "  escalate-and-pin restart (secondary)": list(ep_rst),  # type: ignore[arg-type]
        "random (seed=42) [P6]": random_arm(rows),
        "oracle-per-instance (bound) [P6]": [oracle_predictive(r) for r in rows],
        "oracle-cascade (bound)": [oracle_cascade(r) for r in rows],
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
            f"{f'${cost_w / ok_w:.3f}' if ok_w else '—'} | {dr_ci} |"
        )
    print()

    # P4 breakdown
    for label, fitted_pin, ep_list in [
        ("prereg", fitted_pin_pre, ep_pre),
        ("restart", fitted_pin_rst, ep_rst),
    ]:
        n_sw = [0, 0]
        for j, row in enumerate(rows):
            fold = j % 2
            tau = fitted_pin[fold]
            if row["runs"][0][2] >= int(tau):
                n_sw[fold] += 1
        print(
            f"  P4 {label}: {n_sw[0]}/{n // 2} (fold 0, tau={fitted_pin[0]}) "
            f"+ {n_sw[1]}/{n - n // 2} (fold 1, tau={fitted_pin[1]}) "
            f"= {sum(n_sw)}/{n} trigger"
        )
        for fold_h, tau_p in enumerate(fitted_pin):
            context_toks = I0 + max(0, int(tau_p)) * delta
            overhead = switch_overhead_usd(int(tau_p), price_top_in, I0, delta)
            print(
                f"    [fold {fold_h}] context at switch = {context_toks:,} tokens, "
                f"switch overhead = ${overhead:.4f}/inst "
                f"({'charged to P4 prereg' if label == 'prereg' and int(tau_p) > 0 else 'not charged: no cheap step ran, so no switch' if label == 'prereg' else 'not charged to restart variant'})"
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
        "Whole-run policies are scored exactly from recorded runs. "
        "P4 post-switch quality ≈ top model's own full run (optimistic; see pre-registration). "
        "Context model calibrated from CALIB_N seeded-random cheap-model trajectories. "
        "P4 primary row follows the pre-registration cost model (switch overhead included). "
        "P4 secondary row uses restart semantics (no separate overhead). See results doc.\n"
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

    # Pool policies
    print("## Pool policies (P2 and P3-pool)\n")
    print("Loading pool data ...")
    pool_ids, pool_data = load_pool_data(cache)
    n_pool = len(pool_ids)
    print(
        f"  pool n={n_pool}, reference model index = {POOL_REF_IDX} ({model_suffix(POOL[POOL_REF_IDX])})"
    )

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

    (r0, r1), (c0, c1), _ = paired_ci(p3pool_outs, p2_sel)
    print(
        f"P3-pool vs P2 paired CI (P3-pool − P2): "
        f"Δresolved [{r0:+.3f}, {r1:+.3f}], Δ$/inst [{c0:+.3f}, {c1:+.3f}]"
    )
    print()

    # P3-ladder vs P2: match by instance ID
    print("## P3-ladder vs P2: cross-comparison (confounded: pool vs ladder)\n")
    for lname in LADDERS:
        lids = ladder_ids_map[lname]
        lp3 = ladder_p3_map[lname]
        lid_to_p3 = dict(zip(lids, lp3))
        matched_p3, matched_p2 = [], []
        for pid, p2_out in zip(pool_ids, p2_sel):
            if pid in lid_to_p3:
                matched_p3.append(lid_to_p3[pid])
                matched_p2.append(p2_out)
        n_matched = len(matched_p3)
        n_total = len(lids)
        if n_matched < n_total:
            print(
                f"  NOTE: {n_total - n_matched} ladder instances not in pool; CI on {n_matched}"
            )
        if n_matched > 0:
            (r0, r1), (c0, c1), _ = paired_ci(matched_p3, matched_p2)
            p3_ok, p3_cost = total(matched_p3)
            p2_ok, p2_cost = total(matched_p2)
            print(f"  {lname}: P3-ladder vs P2  n={n_matched}")
            print(
                f"    P3-ladder {p3_ok}/{n_matched} ${p3_cost / n_matched:.3f}/inst  "
                f"P2 {p2_ok}/{n_matched} ${p2_cost / n_matched:.3f}/inst"
            )
            print(
                f"    P3-ladder − P2: Δresolved [{r0:+.3f}, {r1:+.3f}], "
                f"Δ$/inst [{c0:+.3f}, {c1:+.3f}]"
            )
            print(
                "    CONFOUND: P3 uses 2-model ladder; P2 uses 11-model pool. "
                "Use P3-pool vs P2 for pool-controlled comparison."
            )
        print()


# ── self-check ────────────────────────────────────────────────────────────────


def _self_check() -> None:
    # withcache_cost: fixed prefix billing (k-1 not k)
    nc_10 = nocache_cost(10, 1000, 500, 100, 1.0, 5.0)
    wc_10 = withcache_cost(10, 1000, 500, 100, 1.0, 5.0)
    assert nc_10 > wc_10, "uncached must cost more than cached (10 steps)"
    nc_factor_10 = nc_10 / wc_10
    nc_factor_50 = nocache_cost(50, 1000, 500, 100, 1.0, 5.0) / withcache_cost(
        50, 1000, 500, 100, 1.0, 5.0
    )
    assert nc_factor_50 > nc_factor_10, "longer sessions benefit more from caching"
    # Verify k-1 fix: at k=1 the cached prefix is I0 (not I0+delta)
    wc_2 = withcache_cost(2, 1000, 500, 0, 1.0, 0.0)
    # step 0: write 1000 at 1.25 = 1250; step 1: read 1000 at 0.1=100, write 500 at 1.25=625 → 725
    assert abs(wc_2 - (1000 * 1.25 + 1000 * 0.1 + 500 * 1.25) / 1e6) < 1e-12, (
        f"withcache_cost k=1 prefix should be I0=1000, got wc_2={wc_2}"
    )

    # switch_overhead_usd
    sw5 = switch_overhead_usd(5, 15.0, 1000, 500)
    sw20 = switch_overhead_usd(20, 15.0, 1000, 500)
    assert sw20 > sw5, "later switch = larger context = higher overhead"
    assert abs(switch_overhead_usd(0, 15.0, 1000, 300) - 1000 * 15.0 / 1e6) < 1e-12

    row = {"id": "x__1", "repo": "x", "runs": [(True, 1.0, 10), (True, 5.0, 20)]}

    # escalate_pin_prereg: below tau
    res, cwc = escalate_pin_prereg(row, tau=15.0, I0=1000, delta=300, price_top_in=15.0)
    assert res is True and cwc == 1.0, (
        f"prereg below tau: expected cheap cost=1.0, got {cwc}"
    )

    # escalate_pin_prereg: above tau (includes switch overhead)
    res2, cwc2 = escalate_pin_prereg(
        row, tau=5.0, I0=1000, delta=300, price_top_in=15.0
    )
    assert res2 is True, "prereg above tau: top model result"
    overhead = switch_overhead_usd(5, 15.0, 1000, 300)  # (1000+5*300)*15/1e6 = 0.0375
    expected = 1.0 * (5 / 10) + overhead + 5.0
    assert abs(cwc2 - expected) < 1e-9, (
        f"prereg above tau: got {cwc2}, expected {expected}"
    )

    # escalate_pin_prereg: tau=-1 → no cheap step ran, so no switch and no overhead:
    # the top model's recorded run already includes its own first read.
    _, cwc_neg1 = escalate_pin_prereg(
        row, tau=-1.0, I0=1000, delta=300, price_top_in=15.0
    )
    expected_neg1 = 5.0
    assert abs(cwc_neg1 - expected_neg1) < 1e-9, (
        f"prereg tau=-1: {cwc_neg1} != {expected_neg1}"
    )

    # path-unsafe names never reach the filesystem or the URL
    assert _fetch_traj(Path("/nonexistent"), "sys", "../etc/passwd") is None
    assert _fetch_traj(Path("/nonexistent"), "../x", "astropy__astropy-1") is None
    # an arm with zero resolves does not crash the CI
    assert paired_ci([(False, 1.0)] * 4, [(True, 1.0)] * 4, n_boot=10)[2] != ()

    # escalate_pin_restart: no overhead; tau=-1 = top only
    _, cwc_rst = escalate_pin_restart(row, tau=-1.0)
    assert abs(cwc_rst - 5.0) < 1e-9, (
        f"restart tau=-1: cost should be top=5.0, got {cwc_rst}"
    )

    # escalate_pin_restart: at equality (steps==tau): should escalate
    res3, cwc3 = escalate_pin_restart(row, tau=10.0)
    assert res3 is True, "restart at tau==steps: should escalate"
    assert abs(cwc3 - (1.0 + 5.0)) < 1e-9, (
        f"restart at equality: full cheap + top = 6.0, got {cwc3}"
    )

    # repo_router: picks cheapest eligible, not first in list
    rr_train = [
        {"id": "rr__1", "repo": "rr", "runs": [(True, 3.0, 5), (True, 1.0, 3)]},
        {"id": "rr__2", "repo": "rr", "runs": [(True, 3.0, 5), (True, 1.0, 3)]},
    ]
    # both models resolve all; model 0 costs 3.0, model 1 costs 1.0
    # eps=0.0: both eligible (rate=1.0 >= 1.0-0.0); cheapest = model 1
    rr_route = repo_router(rr_train, eps=0.0)
    _, rr_cost = rr_route(rr_train[0])
    assert rr_cost == 1.0, (
        f"repo_router: tied rates must pick cheapest (1.0), got {rr_cost}"
    )
    assert rr_cost != 3.0, (
        "repo_router must not pick first-in-list when it is more expensive"
    )

    # fit_eps: uses ref_idx to set quality bar correctly
    # 5 instances, same repo; 3 models: good/expensive(0), mediocre/cheap(1), bad/last(2)
    # With ref_idx=0 (bar=5): only eps where router routes to model 0 is valid
    # With ref_idx=-1 (bar=0): any eps feasible; picks cheapest routing (model 1)
    fe_train = [
        {
            "id": f"fe__{i}",
            "repo": "fe",
            "runs": [
                (True, 5.0, 10),
                (i < 4, 0.5, 4),  # model 1: resolves 4/5
                (False, 0.1, 2),
            ],
        }
        for i in range(5)
    ]
    # rates: model 0=1.0, model 1=0.8, model 2=0.0
    # eps=0.2: models 0 and 1 eligible (0.8 >= 1.0-0.2); cheapest = model 1 → ok=4 < bar=5
    # With ref_idx=0: eps=0.2 infeasible; only eps<=0.1 route to model 0 (0.8 < 1.0-0.1=0.9)
    # With ref_idx=-1: bar=0; eps=0.2 feasible and cheaper than eps=0 → picks eps=0.2
    eps_strict = fit_eps(fe_train, ref_idx=0)
    eps_loose = fit_eps(fe_train, ref_idx=-1)
    assert eps_strict <= 0.1, (
        f"fit_eps ref_idx=0 (bar=5): must route to model 0, eps must be <=0.1, got {eps_strict}"
    )
    assert eps_loose >= 0.2, (
        f"fit_eps ref_idx=-1 (bar=0): can pick cheap model 1 at eps>=0.2, got {eps_loose}"
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
