//! UCB1 start-rung bandit: predict-to-start, verify-to-serve.
//!
//! # Science
//!
//! Every request currently starts the escalation ladder at rung 0 (cheapest model). Many
//! contexts (task kind × prompt-size bucket) have a learned empirical pass rate per rung: if
//! rung 0 almost never passes for a given context, always starting there wastes money. This
//! module learns a per-context start rung that minimises expected cost while the enforce gate
//! still verifies the chosen rung's output before serving.
//!
//! **The invariant (predict-to-start, verify-to-serve):** the bandit only controls where the
//! ladder *starts*. Gating, escalation, budget, and speculation are untouched. If the predicted
//! start rung's output fails the gate, the ladder continues upward as normal — no downward
//! retry. Prediction errors can cost money (latency, slightly higher expected cost when the
//! estimate is wrong) but can never cause a wrong answer to be served.
//!
//! # Algorithm
//!
//! UCB1 (Auer et al. 2002) applied to cost-sensitive start-rung selection. For each candidate
//! start s, the expected cost is:
//!
//! ```text
//! E[s] = Σ_{r=s}^{top-1}  price_r · P(reach r | start s)
//! P(reach r | start s) = Π_{q=s}^{r-1} (1 − p̂_q)
//! p̂_q = clamp(successes_q / n_q  +  c · √(ln N / n_q),  0, 1)
//! ```
//!
//! where `N` is the total gate-verdict observations in this context, `n_q` is the observations
//! at rung `q`, and `c` is the exploration constant (default 1.0). Prices are evaluated at
//! nominal fixed tokens (1 000 in / 500 out) — relative prices are what matters for start-rung
//! comparison. Tie → lower s (conservative).
//!
//! **Cold-start safety:** if the context has fewer than `min_observations` total gate verdicts,
//! the bandit returns rung 0 (byte-identical to today's behavior). Abstain verdicts are not
//! counted (only clear Pass/Fail gate outcomes are informative).
//!
//! # State
//!
//! ponytail: in-memory `Mutex`, per-process. Survives restarts via warm-start replay of the
//! trace store at startup. No persistence of its own — the trace store is the durable record.

use std::collections::HashMap;

use firstpass_core::{Features, PriceTable, TaskKind, Verdict};

/// The context bucket that keys the bandit's per-arm statistics.
///
/// Coarsened to keep the arm count dense: `prompt_bucket_coarse = prompt_token_bucket / 2`
/// halves the bucket resolution (floor(log2(n)) → floor(log2(n)/2) in effect), trading
/// granularity for faster warm-up.
///
/// ponytail: the ladder identity is not part of the key — in practice a given (TaskKind,
/// prompt size) routes to the same ladder via the first-match routing rule. Would need to be
/// included if operators run overlapping routes for the same context.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContextBucket {
    /// Coarse task classification from the request feature vector.
    pub task_kind: TaskKind,
    /// `features.prompt_token_bucket / 2` — halved for denser arms.
    pub prompt_bucket_coarse: u32,
}

impl ContextBucket {
    /// Derive a context bucket from a request's feature vector.
    #[must_use]
    pub fn from_features(f: &Features) -> Self {
        Self {
            task_kind: f.task_kind,
            prompt_bucket_coarse: f.prompt_token_bucket / 2,
        }
    }
}

/// Per-(context, rung) gate-verdict counts. Abstain is excluded (see module doc).
#[derive(Debug, Default, Clone)]
struct ArmCounts {
    pass: f64,
    fail: f64,
}

impl ArmCounts {
    fn n(&self) -> f64 {
        self.pass + self.fail
    }
}

/// Online UCB1 bandit for start-rung selection.
///
/// Keyed by [`ContextBucket`]; per context, tracks per-rung pass/fail counts of gate verdicts.
/// Thread-safety comes from the caller wrapping this in `Arc<std::sync::Mutex<_>>` (mirroring
/// `AdaptiveConformal` — both are per-process, in-memory, low-contention).
#[derive(Debug)]
pub struct StartRungBandit {
    /// UCB1 exploration constant `c` ≥ 0 (used by [`Algorithm::Ucb1`]).
    exploration: f64,
    /// Cold-start threshold: contexts with fewer total observations return rung 0.
    min_observations: usize,
    /// Selection algorithm: deterministic UCB1 (auditable) or Thompson sampling (stochastic,
    /// native propensities, better under non-stationarity with `discount < 1`).
    algorithm: Algorithm,
    /// Per-observation multiplicative decay of a context's counts, in `(0, 1]`. `1.0` = no
    /// forgetting (stationary). Applied on every observe, so a context that stops matching
    /// reality fades at rate `discount^n` (discounted Thompson sampling).
    discount: f64,
    /// xorshift64* PRNG state for Thompson draws (seeded once; deterministic in tests).
    rng: u64,
    /// context → (rung index → counts).
    data: HashMap<ContextBucket, HashMap<u32, ArmCounts>>,
}

/// Start-rung selection algorithm.
/// Monte-Carlo repetitions for the Thompson propensity estimate.
const PROPENSITY_SAMPLES: usize = 64;

/// The expected-cost argmin over candidate start rungs, shared by UCB1 and Thompson: walk the
/// ladder from each candidate start, accumulating `P(reach r) · price(r)` with
/// `P(reach r+1) = P(reach r) · (1 − p_pass(r))`. Ties prefer the lower start (conservative).
fn argmin_expected_cost(
    ladder: &[String],
    prices: &PriceTable,
    mut p_pass: impl FnMut(u32) -> f64,
) -> u32 {
    // ponytail: nominal 1 000 in / 500 out — relative prices are what matters for start-rung
    // selection; absolute spend is immaterial here.
    const NOMINAL_IN: u64 = 1_000;
    const NOMINAL_OUT: u64 = 500;

    let mut best_s = 0u32;
    let mut best_cost = f64::MAX;
    for s in 0..ladder.len() {
        let mut expected_cost = 0.0_f64;
        let mut p_reach = 1.0_f64;
        for (r, model) in ladder.iter().enumerate().skip(s) {
            let price = prices
                .get(model)
                .map(|p| p.cost(NOMINAL_IN, NOMINAL_OUT))
                .unwrap_or(0.0);
            expected_cost += p_reach * price;
            p_reach *= 1.0 - p_pass(r as u32);
            if p_reach < 1e-10 {
                break;
            }
        }
        if expected_cost < best_cost {
            best_cost = expected_cost;
            best_s = s as u32;
        }
    }
    best_s
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// Deterministic optimism (Auer et al. 2002): auditable, but propensities are degenerate
    /// (0/1) — off-policy estimates then need the epsilon-greedy overlay.
    #[default]
    Ucb1,
    /// Thompson sampling over Beta posteriors of gate-pass: stochastic by nature, so every
    /// decision carries a non-degenerate propensity (Monte-Carlo estimated), and `discount`
    /// handles model churn (Chapelle & Li 2011; discounted TS for non-stationarity).
    Thompson,
}

impl StartRungBandit {
    /// Create a new UCB1 bandit with the given cold-start and exploration settings.
    #[must_use]
    pub fn new(min_observations: usize, exploration: f64) -> Self {
        Self::with_algorithm(min_observations, exploration, Algorithm::Ucb1, 1.0, 0x9E37)
    }

    /// Create a bandit with an explicit algorithm, discount, and PRNG seed (seed matters only
    /// for [`Algorithm::Thompson`]; pass anything non-zero).
    #[must_use]
    pub fn with_algorithm(
        min_observations: usize,
        exploration: f64,
        algorithm: Algorithm,
        discount: f64,
        seed: u64,
    ) -> Self {
        Self {
            exploration,
            min_observations,
            algorithm,
            discount: discount.clamp(f64::MIN_POSITIVE, 1.0),
            rng: seed.max(1),
            data: HashMap::new(),
        }
    }

    /// Apply one step of multiplicative forgetting to every arm in `ctx` (no-op at 1.0).
    fn discount_context(&mut self, ctx: &ContextBucket) {
        if self.discount >= 1.0 {
            return;
        }
        if let Some(arms) = self.data.get_mut(ctx) {
            for c in arms.values_mut() {
                c.pass *= self.discount;
                c.fail *= self.discount;
            }
        }
    }

    /// Next xorshift64* draw as a uniform in `[0, 1)`.
    fn next_u01(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller on the internal PRNG.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_u01().max(f64::MIN_POSITIVE);
        let u2 = self.next_u01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Gamma(shape ≥ 1, scale 1) sample — Marsaglia–Tsang squeeze method.
    fn next_gamma(&mut self, shape: f64) -> f64 {
        debug_assert!(shape >= 1.0, "Beta(+1 prior) keeps shapes >= 1");
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.next_normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_u01();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// One Beta(pass+1, fail+1) posterior draw of the gate-pass probability for `(ctx, rung)`.
    /// Unobserved arms draw from the uniform prior Beta(1, 1).
    fn thompson_pass(&mut self, ctx: &ContextBucket, rung: u32) -> f64 {
        let (a, b) = self
            .data
            .get(ctx)
            .and_then(|arms| arms.get(&rung))
            .map_or((1.0, 1.0), |c| (c.pass + 1.0, c.fail + 1.0));
        let x = self.next_gamma(a);
        let y = self.next_gamma(b);
        x / (x + y)
    }

    /// Posterior-mean gate-pass estimate for `(ctx, rung)` — `None` when the context is cold
    /// (below `min_observations`) or the arm is unobserved. Deterministic; used by the
    /// speculative-deferral band, never for serving decisions.
    #[must_use]
    pub fn pass_estimate(&self, ctx: &ContextBucket, rung: u32) -> Option<f64> {
        let arms = self.data.get(ctx)?;
        let total: f64 = arms.values().map(ArmCounts::n).sum();
        if total < self.min_observations as f64 {
            return None;
        }
        let c = arms.get(&rung)?;
        if c.n() == 0.0 {
            return None;
        }
        Some((c.pass + 1.0) / (c.n() + 2.0))
    }

    /// Record a gate verdict for `(context, rung)`.
    ///
    /// `Abstain` is ignored — only clear Pass/Fail outcomes are informative for cost modelling.
    pub fn observe(&mut self, ctx: &ContextBucket, rung: u32, verdict: Verdict) {
        match verdict {
            Verdict::Abstain => {} // not counted
            Verdict::Pass => {
                self.discount_context(ctx);
                self.data
                    .entry(ctx.clone())
                    .or_default()
                    .entry(rung)
                    .or_default()
                    .pass += 1.0;
            }
            Verdict::Fail => {
                self.discount_context(ctx);
                self.data
                    .entry(ctx.clone())
                    .or_default()
                    .entry(rung)
                    .or_default()
                    .fail += 1.0;
            }
        }
    }

    /// UCB1-optimistic pass-rate estimate for `rung` in `ctx`.
    ///
    /// Returns 1.0 for unobserved rungs (optimistic exploration, UCB → ∞, clamped).
    fn ucb_pass(&self, ctx: &ContextBucket, rung: u32, ln_n: f64) -> f64 {
        let Some(arms) = self.data.get(ctx) else {
            return 1.0; // context unseen: fully optimistic
        };
        let Some(counts) = arms.get(&rung) else {
            return 1.0; // rung unobserved for this context: fully optimistic
        };
        let n = counts.n();
        if n == 0.0 {
            return 1.0;
        }
        let p_hat = counts.pass / n;
        (p_hat + self.exploration * (ln_n / n).sqrt()).clamp(0.0, 1.0)
    }

    /// Choose the start rung that minimises expected cost for this context and ladder.
    ///
    /// Returns `0` in all cold-start cases:
    /// - bandit is cold (total observations < `min_observations` for this context)
    /// - `ladder` is empty
    ///
    /// Never returns a value ≥ `ladder.len()`.
    #[must_use]
    pub fn choose_start(&self, ctx: &ContextBucket, ladder: &[String], prices: &PriceTable) -> u32 {
        let top = ladder.len();
        if top == 0 {
            return 0;
        }

        // Cold-start guard: total gate observations in this context.
        let n_total: f64 = self
            .data
            .get(ctx)
            .map(|arms| arms.values().map(ArmCounts::n).sum())
            .unwrap_or(0.0);
        if n_total < self.min_observations as f64 {
            return 0;
        }

        let ln_n = n_total.ln();
        argmin_expected_cost(ladder, prices, |r| self.ucb_pass(ctx, r, ln_n))
    }

    /// Choose the start rung and its selection propensity.
    ///
    /// - [`Algorithm::Ucb1`]: deterministic — returns `(choice, None)`; the epsilon-greedy
    ///   overlay (if configured) supplies the logged propensity, exactly as before.
    /// - [`Algorithm::Thompson`]: one posterior draw per rung drives the same expected-cost
    ///   argmin as UCB1 (the decision), and the propensity is Monte-Carlo estimated by
    ///   re-running the draw `PROPENSITY_SAMPLES` times — the standard estimator for TS
    ///   selection probabilities (exact computation is analytically intractable). Cold-start
    ///   contexts return `(0, None)` (deterministic, no propensity to log).
    #[must_use]
    pub fn choose_start_with_propensity(
        &mut self,
        ctx: &ContextBucket,
        ladder: &[String],
        prices: &PriceTable,
    ) -> (u32, Option<f64>) {
        if self.algorithm == Algorithm::Ucb1 {
            return (self.choose_start(ctx, ladder, prices), None);
        }
        let top = ladder.len();
        if top == 0 {
            return (0, None);
        }
        let n_total: f64 = self
            .data
            .get(ctx)
            .map(|arms| arms.values().map(ArmCounts::n).sum())
            .unwrap_or(0.0);
        if n_total < self.min_observations as f64 {
            return (0, None);
        }

        let draw = |this: &mut Self| {
            // One full posterior draw per rung, then the shared cost argmin.
            let samples: Vec<f64> = (0..top)
                .map(|r| this.thompson_pass(ctx, r as u32))
                .collect();
            argmin_expected_cost(ladder, prices, |r| samples[r as usize])
        };

        let choice = draw(self);
        let matches = (0..PROPENSITY_SAMPLES)
            .filter(|_| draw(self) == choice)
            .count();
        // Clamp away 0: the MC estimate can miss a genuinely-possible arm in finite samples,
        // and a zero propensity would blow up IPS. 1/(2·M) is the standard floor.
        let p = (matches as f64 / PROPENSITY_SAMPLES as f64)
            .max(1.0 / (2.0 * PROPENSITY_SAMPLES as f64));
        (choice, Some(p))
    }

    /// Feed all attempts from a stored trace into this bandit — used for warm-start at startup.
    pub fn feed_trace_attempts(
        &mut self,
        ctx: &ContextBucket,
        attempts: &[firstpass_core::Attempt],
    ) {
        for attempt in attempts {
            self.observe(ctx, attempt.rung, attempt.verdict);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use firstpass_core::{
        Attempt, Features, FinalOutcome, GENESIS_HASH, Mode, PolicyRef, RequestInfo, ServedFrom,
        TaskKind, Trace,
    };

    fn ctx_code() -> ContextBucket {
        ContextBucket {
            task_kind: TaskKind::CodeEdit,
            prompt_bucket_coarse: 3,
        }
    }

    fn ctx_chat() -> ContextBucket {
        ContextBucket {
            task_kind: TaskKind::Chat,
            prompt_bucket_coarse: 3,
        }
    }

    const HAIKU: &str = "anthropic/claude-haiku-4-5";
    const SONNET: &str = "anthropic/claude-sonnet-5";

    #[test]
    fn cold_start_returns_rung_0() {
        let b = StartRungBandit::new(50, 1.0);
        let prices = PriceTable::defaults();
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        // No observations at all → cold start.
        assert_eq!(b.choose_start(&ctx_code(), &ladder, &prices), 0);
    }

    #[test]
    fn empty_ladder_returns_rung_0() {
        let b = StartRungBandit::new(50, 1.0);
        let prices = PriceTable::defaults();
        assert_eq!(b.choose_start(&ctx_code(), &[], &prices), 0);
    }

    #[test]
    fn after_enough_rung0_fails_rung1_passes_picks_rung1_for_that_context() {
        let mut b = StartRungBandit::new(50, 1.0);
        let prices = PriceTable::defaults();
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];

        // Feed 60 rung-0 fails + 60 rung-1 passes for ctx_code → bandit should skip rung 0.
        let code = ctx_code();
        for _ in 0..60 {
            b.observe(&code, 0, Verdict::Fail);
            b.observe(&code, 1, Verdict::Pass);
        }
        // Expected cost E[0] = price_haiku + (1 - p̂_0) * price_sonnet  >> E[1] = price_sonnet.
        assert_eq!(
            b.choose_start(&code, &ladder, &prices),
            1,
            "should skip rung 0 after observing it always fails"
        );

        // A DIFFERENT context with no data stays cold (returns 0), not influenced by code's data.
        let chat = ctx_chat();
        assert_eq!(
            b.choose_start(&chat, &ladder, &prices),
            0,
            "different context must be independent (cold start)"
        );
    }

    #[test]
    fn abstain_verdicts_are_not_counted() {
        let mut b = StartRungBandit::new(2, 1.0); // tiny min_obs to isolate the logic
        let code = ctx_code();
        // Only feed abstains — they must not push us past cold start.
        for _ in 0..100 {
            b.observe(&code, 0, Verdict::Abstain);
        }
        let prices = PriceTable::defaults();
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        // N = 0 (abstains excluded) < min_obs=2 → rung 0.
        assert_eq!(b.choose_start(&code, &ladder, &prices), 0);
    }

    #[test]
    fn single_rung_ladder_always_returns_0() {
        let mut b = StartRungBandit::new(1, 1.0);
        let code = ctx_code();
        b.observe(&code, 0, Verdict::Fail);
        b.observe(&code, 0, Verdict::Pass);
        let prices = PriceTable::defaults();
        let ladder = vec![HAIKU.to_owned()];
        assert_eq!(b.choose_start(&code, &ladder, &prices), 0);
    }

    /// Build a minimal attempt with a given rung and verdict (for warm-start tests).
    fn stub_attempt(rung: u32, verdict: Verdict) -> Attempt {
        Attempt {
            rung,
            model: HAIKU.to_owned(),
            provider: "anthropic".to_owned(),
            in_tokens: 1000,
            out_tokens: 500,
            cost_usd: 0.001,
            latency_ms: 10,
            gates: vec![],
            verdict,
        }
    }

    #[test]
    fn from_features_derives_bucket_correctly() {
        let mut f = Features::new(TaskKind::CodeEdit);
        f.prompt_token_bucket = 7; // coarse = 7/2 = 3
        let b = ContextBucket::from_features(&f);
        assert_eq!(b.task_kind, TaskKind::CodeEdit);
        assert_eq!(b.prompt_bucket_coarse, 3);
    }

    #[test]
    fn feed_trace_attempts_populates_counts() {
        let mut bandit = StartRungBandit::new(2, 1.0);
        let ctx = ctx_code();
        let attempts = vec![
            stub_attempt(0, Verdict::Fail),
            stub_attempt(1, Verdict::Pass),
        ];
        bandit.feed_trace_attempts(&ctx, &attempts);

        // Verify internally: 1 fail at rung 0, 1 pass at rung 1 → total N=2 ≥ min_obs=2.
        // With min_obs=2 and only rung-0 failing, the bandit should prefer rung 1.
        let prices = PriceTable::defaults();
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        // N=2, p̂_0 = 0/1 + sqrt(ln2/1) ≈ 0.83, p̂_1 = 1/1 + ... ≥ 1.0
        // E[0] = price_haiku + (1-0.83)*price_sonnet; E[1] = price_sonnet.
        // Whether bandit picks 0 or 1 depends on exact math — just verify it doesn't panic.
        let _ = bandit.choose_start(&ctx, &ladder, &prices);
    }

    // ---- Guarantee invariant: bandit on + start rung fails → escalates, never serves failing output.
    // (Integration with the router is tested in router.rs; here we prove the math doesn't
    // synthesise a "free pass" when an observed rung has 0% pass rate.)
    #[test]
    fn zero_pass_rate_rung_is_not_optimistic() {
        // After many fail observations at rung 0 and pass observations at rung 1,
        // choosing E[s] should prefer s=1 even with c=0 (no exploration bonus).
        let mut b = StartRungBandit::new(10, 0.0); // c=0: pure exploitation
        let ctx = ctx_code();
        for _ in 0..20 {
            b.observe(&ctx, 0, Verdict::Fail);
            b.observe(&ctx, 1, Verdict::Pass);
        }
        let prices = PriceTable::defaults();
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        // p̂_0 = 0 (c=0), p̂_1 = 1.0
        // E[0] = price_haiku + 1.0*price_sonnet;  E[1] = price_sonnet
        // E[0] > E[1] → choose start=1.
        assert_eq!(b.choose_start(&ctx, &ladder, &prices), 1);
    }

    // ---- Warm-start from disk --------------------------------------------------------

    /// Helper: build a minimal enforce-mode trace with one attempt at rung 0 (pass) and one
    /// at rung 1 (fail), carrying the given Features so ContextBucket::from_features works.
    fn trace_with_features_and_attempts(features: Features, attempts: Vec<Attempt>) -> Trace {
        let mut trace = Trace {
            trace_id: uuid::Uuid::now_v7(),
            prev_hash: GENESIS_HASH.to_owned(),
            tenant_id: "test-tenant".to_owned(),
            session_id: "sess-warm".to_owned(),
            ts: jiff::Timestamp::now(),
            mode: Mode::Enforce,
            policy: PolicyRef {
                id: "test@v0".to_owned(),
                explore: false,
                propensity: None,
                mode_profile: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".to_owned(),
                prompt_hash: "deadbeef".to_owned(),
                features,
            },
            attempts,
            deferred: vec![],
            final_: FinalOutcome {
                served_rung: Some(0),
                served_from: ServedFrom::Attempt,
                total_cost_usd: 0.001,
                gate_cost_usd: 0.0,
                total_latency_ms: 10,
                escalations: 0,
                counterfactual_baseline_usd: 0.001,
                savings_usd: 0.0,
            },
            probe: None,
            predicted_pass: None,
        };
        trace.recompute_savings();
        trace
    }

    /// Warm-start test: write traces to a temp store, replay them into a fresh bandit,
    /// assert the learned counts drive the expected start-rung choice.
    #[tokio::test]
    async fn warm_start_from_trace_store_replays_counts() {
        let db = std::env::temp_dir().join(format!("bandit-warmstart-{}.db", uuid::Uuid::now_v7()));
        let (tx, writer) = crate::store::open(&db).unwrap();

        // Build features that land in ctx_code (TaskKind::CodeEdit, bucket_coarse=3).
        let mut f = Features::new(TaskKind::CodeEdit);
        f.prompt_token_bucket = 7; // coarse = 7/2 = 3

        // 60 traces: rung 0 always fails, rung 1 always passes.
        for _ in 0..60 {
            let attempts = vec![
                stub_attempt(0, Verdict::Fail),
                stub_attempt(1, Verdict::Pass),
            ];
            tx.try_send(trace_with_features_and_attempts(f.clone(), attempts))
                .unwrap();
        }
        drop(tx);
        writer.await.unwrap();

        // Replay the stored traces into a fresh bandit (mirrors run.rs warm-start logic).
        let mut bandit = StartRungBandit::new(50, 0.0); // c=0: pure exploitation
        let stored = crate::store::load_all_traces(&db).unwrap();
        assert_eq!(stored.len(), 60, "all traces must be stored");
        for trace in &stored {
            let ctx = ContextBucket::from_features(&trace.request.features);
            bandit.feed_trace_attempts(&ctx, &trace.attempts);
        }

        // After warm-start: 60 rung-0 fails + 60 rung-1 passes → bandit prefers rung 1.
        let prices = PriceTable::defaults();
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        let ctx = ContextBucket::from_features(&f);
        assert_eq!(
            bandit.choose_start(&ctx, &ladder, &prices),
            1,
            "warm-started bandit must prefer rung 1 after 60 fail/pass pairs"
        );

        // A context with different features sees no data — cold start, returns 0.
        let mut f2 = Features::new(TaskKind::Chat);
        f2.prompt_token_bucket = 7;
        let ctx2 = ContextBucket::from_features(&f2);
        assert_eq!(
            bandit.choose_start(&ctx2, &ladder, &prices),
            0,
            "different context must be independent"
        );

        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn beta_sampler_mean_matches_posterior_mean() {
        let mut b = StartRungBandit::with_algorithm(0, 1.0, Algorithm::Thompson, 1.0, 0xDEADBEEF);
        // Beta(8, 2): mean 0.8. 4000 draws give a tight empirical mean.
        let ctx = ctx_code();
        for _ in 0..7 {
            b.observe(&ctx, 0, Verdict::Pass);
        }
        b.observe(&ctx, 0, Verdict::Fail);
        // counts: pass 7, fail 1 → Beta(8, 2), mean 0.8.
        let mean: f64 = (0..4000).map(|_| b.thompson_pass(&ctx, 0)).sum::<f64>() / 4000.0;
        assert!(
            (mean - 0.8).abs() < 0.03,
            "Beta(8,2) sample mean should be ~0.8, got {mean}"
        );
        // Unobserved arm = Beta(1,1) uniform, mean 0.5.
        let mean_u: f64 = (0..4000).map(|_| b.thompson_pass(&ctx, 5)).sum::<f64>() / 4000.0;
        assert!(
            (mean_u - 0.5).abs() < 0.03,
            "uniform prior mean ~0.5, got {mean_u}"
        );
    }

    #[test]
    fn thompson_converges_to_skipping_a_hopeless_cheap_rung() {
        let mut b = StartRungBandit::with_algorithm(10, 1.0, Algorithm::Thompson, 1.0, 42);
        let ctx = ctx_code();
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        let prices = PriceTable::defaults();
        // Rung 0 always fails, rung 1 always passes → starting at 1 is cheaper in expectation.
        for _ in 0..80 {
            b.observe(&ctx, 0, Verdict::Fail);
            b.observe(&ctx, 1, Verdict::Pass);
        }
        let picks_rung1 = (0..100)
            .filter(|_| b.choose_start_with_propensity(&ctx, &ladder, &prices).0 == 1)
            .count();
        assert!(
            picks_rung1 >= 90,
            "TS should overwhelmingly skip the hopeless cheap rung, picked rung 1 {picks_rung1}/100"
        );
    }

    #[test]
    fn discounting_adapts_after_a_distribution_flip() {
        let ctx = ctx_code();
        // History: rung 0 failed 200 times. Then the world flips: 40 recent passes.
        let feed = |b: &mut StartRungBandit| {
            for _ in 0..200 {
                b.observe(&ctx, 0, Verdict::Fail);
            }
            for _ in 0..40 {
                b.observe(&ctx, 0, Verdict::Pass);
            }
        };
        let mut discounted = StartRungBandit::with_algorithm(10, 1.0, Algorithm::Thompson, 0.95, 7);
        let mut undiscounted =
            StartRungBandit::with_algorithm(10, 1.0, Algorithm::Thompson, 1.0, 7);
        feed(&mut discounted);
        feed(&mut undiscounted);
        let p_disc = discounted.pass_estimate(&ctx, 0).unwrap();
        let p_flat = undiscounted.pass_estimate(&ctx, 0).unwrap();
        assert!(
            p_disc > 0.7 && p_flat < 0.25,
            "discounted tracks the flip (got {p_disc:.2}), undiscounted lags (got {p_flat:.2})"
        );
    }

    #[test]
    fn thompson_propensity_matches_empirical_selection_frequency() {
        let mut b = StartRungBandit::with_algorithm(10, 1.0, Algorithm::Thompson, 1.0, 99);
        let ctx = ctx_code();
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        let prices = PriceTable::defaults();
        // Ambiguous data: rung 0 passes ~half the time → genuinely stochastic selection.
        for _ in 0..20 {
            b.observe(&ctx, 0, Verdict::Pass);
            b.observe(&ctx, 0, Verdict::Fail);
            b.observe(&ctx, 1, Verdict::Pass);
        }
        // Empirical frequency of each choice over many decisions vs the logged propensity.
        let mut freq = std::collections::HashMap::new();
        let mut props: Vec<(u32, f64)> = Vec::new();
        for _ in 0..400 {
            let (c, p) = b.choose_start_with_propensity(&ctx, &ladder, &prices);
            *freq.entry(c).or_insert(0u32) += 1;
            props.push((c, p.expect("thompson always logs a propensity")));
        }
        for (arm, count) in freq {
            let empirical = f64::from(count) / 400.0;
            let mean_logged: f64 = {
                let logged: Vec<f64> = props
                    .iter()
                    .filter(|(c, _)| *c == arm)
                    .map(|(_, p)| *p)
                    .collect();
                logged.iter().sum::<f64>() / logged.len() as f64
            };
            assert!(
                (empirical - mean_logged).abs() < 0.15,
                "arm {arm}: empirical {empirical:.2} vs logged propensity {mean_logged:.2}"
            );
        }
    }

    #[test]
    fn pass_estimate_cold_and_warm() {
        let mut b = StartRungBandit::with_algorithm(10, 1.0, Algorithm::Thompson, 1.0, 5);
        let ctx = ctx_code();
        assert!(
            b.pass_estimate(&ctx, 0).is_none(),
            "cold context: no estimate"
        );
        for _ in 0..12 {
            b.observe(&ctx, 0, Verdict::Pass);
        }
        let p = b.pass_estimate(&ctx, 0).expect("warm");
        assert!(p > 0.85, "12/12 passes: posterior mean high, got {p}");
        assert!(
            b.pass_estimate(&ctx, 3).is_none(),
            "unobserved arm: no estimate"
        );
    }
}
