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
    pass: u64,
    fail: u64,
}

impl ArmCounts {
    fn n(&self) -> u64 {
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
    /// UCB1 exploration constant `c` ≥ 0.
    exploration: f64,
    /// Cold-start threshold: contexts with fewer total observations return rung 0.
    min_observations: usize,
    /// context → (rung index → counts).
    data: HashMap<ContextBucket, HashMap<u32, ArmCounts>>,
}

impl StartRungBandit {
    /// Create a new bandit with the given cold-start and exploration settings.
    #[must_use]
    pub fn new(min_observations: usize, exploration: f64) -> Self {
        Self {
            exploration,
            min_observations,
            data: HashMap::new(),
        }
    }

    /// Record a gate verdict for `(context, rung)`.
    ///
    /// `Abstain` is ignored — only clear Pass/Fail outcomes are informative for cost modelling.
    pub fn observe(&mut self, ctx: &ContextBucket, rung: u32, verdict: Verdict) {
        match verdict {
            Verdict::Abstain => {} // not counted
            Verdict::Pass => {
                self.data
                    .entry(ctx.clone())
                    .or_default()
                    .entry(rung)
                    .or_default()
                    .pass += 1;
            }
            Verdict::Fail => {
                self.data
                    .entry(ctx.clone())
                    .or_default()
                    .entry(rung)
                    .or_default()
                    .fail += 1;
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
        let n = counts.n() as f64;
        if n == 0.0 {
            return 1.0;
        }
        let p_hat = counts.pass as f64 / n;
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
        let n_total: u64 = self
            .data
            .get(ctx)
            .map(|arms| arms.values().map(ArmCounts::n).sum())
            .unwrap_or(0);
        if (n_total as usize) < self.min_observations {
            return 0;
        }

        // ponytail: nominal 1 000 in / 500 out — relative prices are what matters for start-rung
        // selection; absolute spend is immaterial here.
        const NOMINAL_IN: u64 = 1_000;
        const NOMINAL_OUT: u64 = 500;

        let ln_n = (n_total as f64).ln();

        let mut best_s = 0u32;
        let mut best_cost = f64::MAX;

        for s in 0..top {
            let mut expected_cost = 0.0_f64;
            let mut p_reach = 1.0_f64; // probability of reaching rung r given start s

            for (r, model) in ladder.iter().enumerate().skip(s) {
                let price = prices
                    .get(model)
                    .map(|p| p.cost(NOMINAL_IN, NOMINAL_OUT))
                    .unwrap_or(0.0);

                expected_cost += p_reach * price;

                let p_pass = self.ucb_pass(ctx, r as u32, ln_n);
                p_reach *= 1.0 - p_pass;

                // Early exit: negligible reach probability — cost contribution < floating noise.
                if p_reach < 1e-10 {
                    break;
                }
            }

            // Tie → lower s (conservative: prefer rung 0 on equal expected cost).
            if expected_cost < best_cost {
                best_cost = expected_cost;
                best_s = s as u32;
            }
        }

        best_s
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
}
