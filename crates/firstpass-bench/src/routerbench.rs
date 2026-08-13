//! RouterBench's evaluation metric, ported to this harness (arXiv:2403.12031).
//!
//! # Why the metric and not the dataset
//!
//! RouterBench is the field's standard router benchmark, so the obvious move is to run Firstpass
//! on it. That does not work, for a structural reason worth stating precisely rather than
//! quietly working around:
//!
//! RouterBench ships **one stored response per model per prompt**, a lenient quality score, and
//! **no executable oracle**. That is exactly what a *pre-judgment* router needs — read the
//! prompt, pick a model, look up its score — and exactly what a *verification* router cannot
//! use, because there is nothing to run a real gate against. Using RouterBench's own quality
//! score as the gate would make gate and oracle the same measurement, which measures nothing.
//! Its MBPP prompts are additionally paraphrased rewrites of canonical MBPP (mean similarity
//! ~0.78 against the canonical text, with no recoverable index mapping), so the stored responses
//! cannot be re-scored against MBPP's real asserts either.
//!
//! So the comparison is inverted: rather than putting Firstpass on their data, this puts **their
//! metric on our data**. AIQ, the non-decreasing convex hull, and the Zero Router are computed
//! exactly as the paper defines them, over the measured matrix this harness already produces. A
//! RouterBench reader gets a number they recognise; nobody has to trust a join that cannot be
//! validated.
//!
//! # The bar
//!
//! RouterBench's own headline result is that **no learned router significantly outperformed the
//! Zero Router** — the parameter-free interpolation between the raw models — and that routers
//! were *worse* than it on MBPP specifically. The paper also reports that cascading routers beat
//! the Zero Router when the judge's error rate is **≤ 0.1**, degrading sharply past 0.2. Both are
//! pre-registered bars this module measures against, not ones invented after seeing the result.

use crate::coding_policy::RungOutcome;

/// A point in the cost-quality plane: mean USD per task against mean oracle-correct rate.
#[derive(Debug, Clone)]
pub struct Point {
    /// What this point is — a ladder rung, or a routing policy.
    pub label: String,
    /// Mean USD per task.
    pub cost: f64,
    /// Mean fraction of tasks answered correctly per the hidden oracle.
    pub quality: f64,
}

/// The full RouterBench-style view of one measured matrix.
#[derive(Debug, Clone)]
pub struct RouterBenchView {
    /// Every model and router placed in the cost-quality plane.
    pub points: Vec<Point>,
    /// Shared cost domain `[c_min, c_max]` every AIQ below is integrated over.
    pub domain: (f64, f64),
    /// AIQ of the Zero Router — the raw models' own hull. The bar a router must clear to matter.
    pub zero_router_aiq: f64,
    /// AIQ of first-pass interpolated against the raw models.
    pub first_pass_aiq: f64,
    /// AIQ of the oracle router (perfect foresight, cheapest rung that is actually correct).
    pub oracle_aiq: f64,
}

impl RouterBenchView {
    /// How much AIQ first-pass adds over the Zero Router. Positive ⇒ it cleared the bar that
    /// RouterBench reports no learned router cleared.
    #[must_use]
    pub fn lift_over_zero_router(&self) -> f64 {
        self.first_pass_aiq - self.zero_router_aiq
    }

    /// Fraction of the headroom between the Zero Router and perfect foresight that first-pass
    /// captures. Reported because raw AIQ deltas are small numbers whose size is hard to read:
    /// the oracle gap says how much was available to win in the first place.
    #[must_use]
    pub fn oracle_gap_closed(&self) -> Option<f64> {
        let headroom = self.oracle_aiq - self.zero_router_aiq;
        (headroom > 1e-12).then(|| self.lift_over_zero_router() / headroom)
    }
}

/// Mean of a slice, or 0 for an empty one.
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// The **upper** convex hull of a point set, made non-decreasing (the paper's `S_ndch`).
///
/// Two steps, in the paper's order. First the upper hull: because a router that dispatches to
/// `A` with probability `t` and `B` otherwise realises the whole segment `AB` (linearity of
/// expectation), every affine combination of measured points is achievable, so the reachable
/// region is their convex hull and interior points are strictly dominated. Then monotonicity: if
/// spending more buys no quality, the cheaper point's quality is carried rightward, since nobody
/// is forced to spend the extra money.
#[must_use]
pub fn non_decreasing_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut pts = points.to_vec();
    // Sort by cost, then by quality descending so the best point at a given cost comes first.
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Monotone chain, keeping only left turns — the upper hull.
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
    for p in pts {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            // b is below or on segment a->p ⇒ dominated by an affine combination, so drop it.
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            if cross >= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }

    // Carry the running-best quality rightward.
    let mut best = f64::NEG_INFINITY;
    for h in &mut hull {
        best = best.max(h.1);
        h.1 = best;
    }
    hull
}

/// Quality of a frontier at cost `c`, by linear interpolation between its two nearest points.
///
/// Outside the frontier the paper's extrapolation rules apply: below its cheapest point,
/// interpolate toward the null router at `(0, 0)` — you can always flip a coin and answer nothing;
/// above its dearest, quality is flat, since extra spend can be burned without buying anything.
fn quality_at(frontier: &[(f64, f64)], c: f64) -> f64 {
    let Some(&(first_c, first_q)) = frontier.first() else {
        return 0.0;
    };
    if c <= first_c {
        // Interpolate against the null router at the origin.
        if first_c <= 0.0 {
            return first_q;
        }
        return first_q * (c / first_c);
    }
    for w in frontier.windows(2) {
        let ((c0, q0), (c1, q1)) = (w[0], w[1]);
        if c <= c1 {
            if (c1 - c0).abs() < 1e-15 {
                return q0.max(q1);
            }
            let t = (c - c0) / (c1 - c0);
            return q0 + t * (q1 - q0);
        }
    }
    frontier.last().map_or(0.0, |&(_, q)| q)
}

/// Number of trapezoid slices used to integrate a frontier. The frontier is piecewise linear, so
/// this is exact for any resolution above its vertex count; 2048 is far above it and keeps the
/// number stable if the shape ever gains vertices.
const AIQ_SLICES: usize = 2048;

/// AIQ: the frontier's mean quality over `[c_min, c_max]`.
///
/// `AIQ(R) = 1/(c_max − c_min) · ∫ q̃_R(c) dc` — a normalised area, so it reads as an average
/// quality level and is comparable across routers **only when they share a cost domain**. That is
/// why the domain is a parameter rather than derived per-router.
#[must_use]
pub fn aiq(frontier: &[(f64, f64)], c_min: f64, c_max: f64) -> f64 {
    if frontier.is_empty() || c_max <= c_min {
        return 0.0;
    }
    let step = (c_max - c_min) / AIQ_SLICES as f64;
    let mut area = 0.0;
    for i in 0..AIQ_SLICES {
        let a = c_min + step * i as f64;
        let b = a + step;
        area += 0.5 * (quality_at(frontier, a) + quality_at(frontier, b)) * step;
    }
    area / (c_max - c_min)
}

/// Place a measured matrix in RouterBench's cost-quality plane and score every curve with AIQ.
///
/// The three curves are the paper's: the **Zero Router** (raw models only), **first-pass** added
/// to them, and the **oracle** router with perfect foresight.
#[must_use]
pub fn evaluate(matrix: &[Vec<RungOutcome>], ladder: &[String]) -> RouterBenchView {
    let mut points = Vec::new();

    // One point per ladder rung: what always serving that model costs and achieves.
    let mut model_pts = Vec::new();
    for (i, name) in ladder.iter().enumerate() {
        let costs: Vec<f64> = matrix
            .iter()
            .filter_map(|r| r.get(i))
            .map(|o| o.cost_usd)
            .collect();
        let quals: Vec<f64> = matrix
            .iter()
            .filter_map(|r| r.get(i))
            .map(|o| f64::from(u8::from(o.oracle_correct)))
            .collect();
        let p = (mean(&costs), mean(&quals));
        model_pts.push(p);
        points.push(Point {
            label: name.clone(),
            cost: p.0,
            quality: p.1,
        });
    }

    // first-pass: serve the cheapest rung whose gate passes, paying for every rung tried.
    let mut fp_cost = Vec::with_capacity(matrix.len());
    let mut fp_qual = Vec::with_capacity(matrix.len());
    // oracle: perfect foresight — cheapest rung that is actually correct, else the dearest tried.
    let mut or_cost = Vec::with_capacity(matrix.len());
    let mut or_qual = Vec::with_capacity(matrix.len());
    for row in matrix {
        let mut spent = 0.0;
        let mut served = row.last().is_some_and(|o| o.oracle_correct);
        for o in row {
            spent += o.cost_usd;
            if o.gate_full_pass {
                served = o.oracle_correct;
                break;
            }
        }
        fp_cost.push(spent);
        fp_qual.push(f64::from(u8::from(served)));

        match row.iter().find(|o| o.oracle_correct) {
            Some(o) => {
                or_cost.push(o.cost_usd);
                or_qual.push(1.0);
            }
            None => {
                // Nothing on the ladder solves it; the oracle still pays the cheapest rung rather
                // than pretending a free abstention was available.
                or_cost.push(row.first().map_or(0.0, |o| o.cost_usd));
                or_qual.push(0.0);
            }
        }
    }
    let fp = (mean(&fp_cost), mean(&fp_qual));
    let orc = (mean(&or_cost), mean(&or_qual));
    points.push(Point {
        label: "first-pass".to_owned(),
        cost: fp.0,
        quality: fp.1,
    });
    points.push(Point {
        label: "oracle".to_owned(),
        cost: orc.0,
        quality: orc.1,
    });

    // One shared domain for every AIQ, spanning every point so no curve is scored on a range it
    // was not measured over. Comparing AIQs computed on different domains is meaningless.
    let all: Vec<f64> = points.iter().map(|p| p.cost).collect();
    let c_min = all.iter().copied().fold(f64::INFINITY, f64::min);
    let c_max = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let zero = non_decreasing_hull(&model_pts);
    let mut with_fp = model_pts.clone();
    with_fp.push(fp);
    let mut with_or = model_pts;
    with_or.push(orc);

    RouterBenchView {
        points,
        domain: (c_min, c_max),
        zero_router_aiq: aiq(&zero, c_min, c_max),
        first_pass_aiq: aiq(&non_decreasing_hull(&with_fp), c_min, c_max),
        oracle_aiq: aiq(&non_decreasing_hull(&with_or), c_min, c_max),
    }
}

/// Render the view as Markdown, in the shape the paper's tables use.
#[must_use]
pub fn render(v: &RouterBenchView) -> String {
    let mut s = String::new();
    s.push_str("## RouterBench metric (arXiv:2403.12031) on this matrix\n\n");
    s.push_str(&format!(
        "Cost domain: ${:.5} .. ${:.5} per task. AIQ is mean quality over that domain.\n\n",
        v.domain.0, v.domain.1
    ));
    s.push_str("| point | $/task | quality |\n|---|---|---|\n");
    for p in &v.points {
        s.push_str(&format!(
            "| {} | ${:.5} | {:.4} |\n",
            p.label, p.cost, p.quality
        ));
    }
    s.push_str("\n| curve | AIQ |\n|---|---|\n");
    s.push_str(&format!(
        "| Zero Router (models only) | {:.4} |\n",
        v.zero_router_aiq
    ));
    s.push_str(&format!("| + first-pass | {:.4} |\n", v.first_pass_aiq));
    s.push_str(&format!(
        "| oracle (perfect foresight) | {:.4} |\n",
        v.oracle_aiq
    ));

    let lift = v.lift_over_zero_router();
    s.push_str(&format!("\n**AIQ lift over the Zero Router: {lift:+.4}**"));
    match v.oracle_gap_closed() {
        Some(f) => s.push_str(&format!(
            " — {:.0}% of the headroom to perfect foresight.\n",
            f * 100.0
        )),
        None => s.push_str(" — no headroom to close on this matrix.\n"),
    }
    s.push_str(
        "\nRouterBench reports that no learned router significantly beat the Zero Router, and that \
         routers were worse than it on MBPP specifically. A non-positive lift here means \
         first-pass does not beat that bar either, and should be reported as such.\n",
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(gate: bool, oracle: bool, cost: f64) -> RungOutcome {
        RungOutcome {
            gate_score: f64::from(u8::from(gate)),
            gate_full_pass: gate,
            oracle_correct: oracle,
            cost_usd: cost,
            judge_score: None,
        }
    }

    fn ladder() -> Vec<String> {
        vec!["cheap".to_owned(), "top".to_owned()]
    }

    /// A point strictly inside the hull is dominated by an affine combination of two others, so a
    /// probabilistic blend of them is at least as good — the paper's reason for discarding it.
    #[test]
    fn hull_drops_a_dominated_interior_point() {
        let hull = non_decreasing_hull(&[(0.0, 0.0), (1.0, 0.5), (2.0, 1.0)]);
        assert!(
            !hull.contains(&(1.0, 0.5)),
            "the midpoint lies on the segment and must be dropped, got {hull:?}"
        );
    }

    /// Paying more must never read as buying less: the frontier carries the cheaper point's
    /// quality rightward instead of sloping down.
    #[test]
    fn hull_is_non_decreasing_even_when_the_dearer_model_is_worse() {
        let hull = non_decreasing_hull(&[(1.0, 0.9), (2.0, 0.3)]);
        let qs: Vec<f64> = hull.iter().map(|p| p.1).collect();
        assert!(
            qs.windows(2).all(|w| w[1] >= w[0] - 1e-12),
            "quality must not decrease with cost, got {qs:?}"
        );
        assert!(
            (quality_at(&hull, 2.0) - 0.9).abs() < 1e-9,
            "the dearer point inherits the cheaper point's quality"
        );
    }

    /// AIQ of a flat frontier is that constant — the sanity check that the normalisation is a
    /// mean and not a raw area.
    #[test]
    fn aiq_of_a_flat_frontier_is_its_level() {
        let f = vec![(1.0, 0.75), (2.0, 0.75)];
        assert!(
            (aiq(&f, 1.0, 2.0) - 0.75).abs() < 1e-6,
            "got {}",
            aiq(&f, 1.0, 2.0)
        );
    }

    /// The oracle has perfect foresight, so it can never score below the models it chooses among.
    #[test]
    fn oracle_aiq_is_never_below_the_zero_router() {
        let matrix = vec![
            vec![o(true, true, 0.01), o(true, true, 0.10)],
            vec![o(false, false, 0.01), o(true, true, 0.10)],
            vec![o(true, true, 0.01), o(true, false, 0.10)],
        ];
        let v = evaluate(&matrix, &ladder());
        assert!(
            v.oracle_aiq >= v.zero_router_aiq - 1e-9,
            "oracle {} < zero {}",
            v.oracle_aiq,
            v.zero_router_aiq
        );
    }

    /// The shape the product claims: a gate that catches the cheap rung's errors buys quality the
    /// raw models' own interpolation cannot reach, so AIQ rises above the Zero Router.
    #[test]
    fn a_perfect_gate_lifts_aiq_above_the_zero_router() {
        // Cheap is right on 3 of 4 and its gate catches the 4th; top is always right but dear.
        let matrix = vec![
            vec![o(true, true, 0.01), o(true, true, 0.20)],
            vec![o(true, true, 0.01), o(true, true, 0.20)],
            vec![o(true, true, 0.01), o(true, true, 0.20)],
            vec![o(false, false, 0.01), o(true, true, 0.20)],
        ];
        let v = evaluate(&matrix, &ladder());
        assert!(
            v.lift_over_zero_router() > 0.0,
            "a gate that converts every failure must beat the parameter-free blend, got {:+.5}",
            v.lift_over_zero_router()
        );
    }

    /// The first-pass point must be plotted at what it actually spent, including every rung the
    /// gate rejected. Placing it at only the served rung's price would shift it left in the plane
    /// and manufacture AIQ lift out of unbilled escalations.
    ///
    /// Asserted on the cost directly rather than through the lift: a downstream lift assertion
    /// does not discriminate here, because a first-pass point sitting exactly on the top model is
    /// a non-win either way. (Found by mutation — dropping the accumulation left every other test
    /// in this module green.)
    #[test]
    fn the_first_pass_point_is_billed_for_rejected_rungs() {
        // One task serves cheap outright; one is rejected and escalates.
        let matrix = vec![
            vec![o(true, true, 0.01), o(true, true, 0.20)],
            vec![o(false, false, 0.01), o(true, true, 0.20)],
        ];
        let v = evaluate(&matrix, &ladder());
        let fp = v
            .points
            .iter()
            .find(|p| p.label == "first-pass")
            .expect("first-pass point");
        // (0.01) + (0.01 + 0.20) = 0.22 over 2 tasks.
        assert!(
            (fp.cost - 0.11).abs() < 1e-9,
            "escalation must bill both rungs: expected $0.11/task, got ${:.5}",
            fp.cost
        );
    }

    /// The honesty case, and the one that must not be quietly absent. A gate that rejects correct
    /// cheap answers spends the dear rung for nothing, so it must NOT clear the bar — otherwise
    /// the metric would flatter every cascade regardless of whether its gate works.
    #[test]
    fn a_gate_that_rejects_correct_answers_does_not_clear_the_bar() {
        // The cheap rung is right every time, but the gate rejects it every time, so first-pass
        // pays for both rungs on every task and buys nothing.
        let matrix = vec![
            vec![o(false, true, 0.01), o(true, true, 0.20)],
            vec![o(false, true, 0.01), o(true, true, 0.20)],
            vec![o(false, true, 0.01), o(true, true, 0.20)],
        ];
        let v = evaluate(&matrix, &ladder());
        assert!(
            v.lift_over_zero_router() <= 1e-9,
            "a purely wasteful gate must not read as a win, got {:+.5}",
            v.lift_over_zero_router()
        );
    }
}
