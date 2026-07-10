//! Simulation substrate: a deterministic RNG, a task suite, and simulated model + gate
//! backends behind the same traits a real provider/gate will implement.
//!
//! **Everything here is a labeled simulation.** It exists so the *proof methodology* — policies,
//! metrics, bootstrap CIs, conformal risk control, kill criterion — is correct and fully tested
//! before real-provider numbers exist. Swap [`SimBackend`]/[`SimGate`] for reqwest-backed
//! implementations of [`ModelBackend`]/[`Gate`] and the same harness produces real proof.
//! The simulation's assumptions (a cheaper model clears easier tasks; a gate has finite
//! precision/recall) mirror the real thesis; they do not pre-decide its outcome.

use firstpass_core::Verdict;

/// SplitMix64 — a tiny, dependency-free, fully deterministic PRNG. Used for both reproducible
/// per-item draws ([`hash01`]) and sequential sampling ([`Rng`], for the bootstrap).
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Map a `u64` to a uniform `f64` in `[0, 1)` using its top 53 bits.
#[inline]
fn to_unit(x: u64) -> f64 {
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// A reproducible uniform draw in `[0, 1)` keyed by a seed and two coordinates (e.g. task id,
/// model hash). Same inputs always yield the same draw — the basis of a reproducible benchmark.
#[must_use]
pub fn hash01(seed: u64, a: u64, b: u64) -> f64 {
    let mut s = seed
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
        .wrapping_add(a.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add(0x1234_5678_9ABC_DEF0);
    to_unit(splitmix64(&mut s))
}

/// A stateful deterministic RNG for sequential draws (bootstrap resampling).
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the RNG.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xA5A5_A5A5_5A5A_5A5A,
        }
    }

    /// Next uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        to_unit(splitmix64(&mut self.state))
    }

    /// Next index in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.unit() * n as f64) as usize % n
    }
}

/// Coarse hash of a model string, so per-model draws are stable across runs.
fn model_hash(model: &str) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64; // FNV-1a
    for b in model.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// A benchmark task with a hidden difficulty in `[0, 1]` (1 = hardest).
#[derive(Debug, Clone)]
pub struct Task {
    /// Stable id.
    pub id: u64,
    /// Hidden difficulty; higher means fewer models clear it.
    pub difficulty: f64,
    /// Approximate prompt size, drives input-token cost.
    pub prompt_tokens: u64,
    /// Real prompt text for a live run (`None` in simulation).
    pub prompt: Option<String>,
    /// Ground-truth expected answer for a live run, checked by the live gate (`None` in simulation).
    pub expected: Option<String>,
}

impl Task {
    /// A verifiable live task: a real prompt with a known expected answer. `difficulty` is set to a
    /// neutral 0.5 — a live task has no simulator-known difficulty to leak to the predictive baseline.
    #[must_use]
    pub fn verifiable(id: u64, prompt: impl Into<String>, expected: impl Into<String>) -> Self {
        let prompt = prompt.into();
        // Rough input estimate (~4 chars/token) for accounting before the API returns real usage;
        // the live backend overwrites in/out tokens with the provider's actual counts.
        let prompt_tokens = (prompt.len() / 4).max(1) as u64;
        Self {
            id,
            difficulty: 0.5,
            prompt_tokens,
            prompt: Some(prompt),
            expected: Some(expected.into()),
        }
    }
}

/// Generate a reproducible suite of `n` tasks with difficulty ~ U(0,1).
#[must_use]
pub fn task_suite(n: usize, seed: u64) -> Vec<Task> {
    (0..n as u64)
        .map(|id| Task {
            id,
            difficulty: hash01(seed, id, 1),
            prompt_tokens: 800 + (hash01(seed, id, 2) * 3200.0) as u64,
            prompt: None,
            expected: None,
        })
        .collect()
}

/// A ladder rung: a model and its (simulated) intrinsic strength in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct Rung {
    /// `provider/model` string (must exist in the price table).
    pub model: String,
    /// Intrinsic strength; stronger models clear harder tasks.
    pub strength: f64,
}

impl Rung {
    /// Convenience constructor.
    #[must_use]
    pub fn new(model: impl Into<String>, strength: f64) -> Self {
        Self {
            model: model.into(),
            strength,
        }
    }
}

/// The outcome of running a model on a task. `correct` is ground truth (only the simulator, or in
/// reality a downstream test, knows it); a policy must decide *without* reading it directly.
#[derive(Debug, Clone)]
pub struct Completion {
    /// Input tokens billed.
    pub in_tokens: u64,
    /// Output tokens billed.
    pub out_tokens: u64,
    /// Ground-truth correctness of the output.
    pub correct: bool,
    /// Simulated wall-clock latency.
    pub latency_ms: u64,
    /// The candidate's raw output text — `None` in simulation, `Some` on a live run so a live
    /// judge gate can grade it. Ground truth (`correct`) is computed separately and never shown
    /// to the judge.
    pub output: Option<String>,
}

/// A model backend: run a task on a rung, get a completion. Real impls call a provider.
pub trait ModelBackend {
    /// Execute `task` on `rung`.
    fn run(&self, task: &Task, rung: &Rung) -> Completion;
}

/// Deterministic simulated backend. Clearance probability falls with task difficulty and rises
/// with model strength; correctness is a reproducible draw against that probability.
#[derive(Debug, Clone)]
pub struct SimBackend {
    seed: u64,
}

impl SimBackend {
    /// Seed the backend.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Probability this rung produces a correct answer for this task.
    #[must_use]
    pub fn clearance(strength: f64, difficulty: f64) -> f64 {
        (strength - 0.45 * difficulty).clamp(0.02, 0.99)
    }
}

impl ModelBackend for SimBackend {
    fn run(&self, task: &Task, rung: &Rung) -> Completion {
        let mh = model_hash(&rung.model);
        let p = SimBackend::clearance(rung.strength, task.difficulty);
        let correct = hash01(self.seed, task.id, mh) < p;
        // Stronger models emit somewhat more (reasoning) tokens and take longer.
        let out_tokens = 300 + (rung.strength * 500.0) as u64;
        let latency_ms = 400 + (rung.strength * 1600.0) as u64;
        Completion {
            in_tokens: task.prompt_tokens,
            out_tokens,
            correct,
            latency_ms,
            output: None,
        }
    }
}

/// A gate's judgement of a completion.
#[derive(Debug, Clone)]
pub struct GateJudgement {
    /// pass / fail / abstain.
    pub verdict: Verdict,
    /// Continuous confidence in `[0, 1]`, correlated with true correctness — the signal
    /// conformal calibration thresholds on.
    pub score: f64,
    /// Marginal USD to run the gate.
    pub cost_usd: f64,
    /// Gate latency.
    pub ms: u64,
}

/// A gate: judge a completion against ground truth (imperfectly, like reality).
pub trait Gate {
    /// Judge `completion` for `task`.
    fn judge(&self, task: &Task, rung: &Rung, completion: &Completion) -> GateJudgement;
}

/// Deterministic simulated gate with configurable false-positive / false-negative rates and a
/// per-call cost. A perfect gate would make the whole problem trivial; a real one does not exist,
/// so the harness must study gate imperfection explicitly.
#[derive(Debug, Clone)]
pub struct SimGate {
    seed: u64,
    /// P(pass | truly incorrect) — a false accept (burns trust).
    pub fpr: f64,
    /// P(fail | truly correct) — a false reject (burns money via needless escalation).
    pub fnr: f64,
    /// Marginal cost per gate invocation.
    pub cost_usd: f64,
    /// Gate latency.
    pub ms: u64,
}

impl SimGate {
    /// Construct a gate with the given error rates and cost.
    #[must_use]
    pub fn new(seed: u64, fpr: f64, fnr: f64, cost_usd: f64) -> Self {
        Self {
            seed,
            fpr,
            fnr,
            cost_usd,
            ms: 2500,
        }
    }
}

impl Gate for SimGate {
    fn judge(&self, task: &Task, rung: &Rung, completion: &Completion) -> GateJudgement {
        let mh = model_hash(&rung.model);
        let flip = hash01(self.seed ^ 0xDEAD_BEEF, task.id, mh);
        // Verdict: start from ground truth, flip with configured error rates.
        let verdict = if completion.correct {
            if flip < self.fnr {
                Verdict::Fail
            } else {
                Verdict::Pass
            }
        } else if flip < self.fpr {
            Verdict::Pass
        } else {
            Verdict::Fail
        };
        // Score: correlated with true correctness plus noise, so higher score => more likely
        // correct (what conformal needs). Correct answers center at 0.72, incorrect at 0.30.
        let noise = (hash01(self.seed ^ 0x00C0_FFEE, task.id, mh) - 0.5) * 0.4;
        let base = if completion.correct { 0.72 } else { 0.30 };
        let score = (base + noise).clamp(0.0, 1.0);
        GateJudgement {
            verdict,
            score,
            cost_usd: self.cost_usd,
            ms: self.ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_are_reproducible() {
        assert_eq!(hash01(7, 1, 2), hash01(7, 1, 2));
        assert_ne!(hash01(7, 1, 2), hash01(7, 1, 3));
        assert_ne!(hash01(8, 1, 2), hash01(7, 1, 2));
    }

    #[test]
    fn clearance_falls_with_difficulty_and_rises_with_strength() {
        assert!(SimBackend::clearance(0.9, 0.1) > SimBackend::clearance(0.6, 0.1));
        assert!(SimBackend::clearance(0.9, 0.1) > SimBackend::clearance(0.9, 0.9));
        assert!((0.02..=0.99).contains(&SimBackend::clearance(0.9, 0.0)));
    }

    #[test]
    fn stronger_rung_clears_more_of_the_suite() {
        let suite = task_suite(2000, 42);
        let be = SimBackend::new(42);
        let weak = Rung::new("anthropic/claude-haiku-4-5", 0.62);
        let strong = Rung::new("anthropic/claude-opus-4-8", 0.93);
        let w = suite.iter().filter(|t| be.run(t, &weak).correct).count();
        let s = suite.iter().filter(|t| be.run(t, &strong).correct).count();
        assert!(s > w, "strong {s} should clear more than weak {w}");
    }

    #[test]
    fn gate_score_separates_correct_from_incorrect() {
        let suite = task_suite(3000, 5);
        let be = SimBackend::new(5);
        let gate = SimGate::new(9, 0.08, 0.10, 0.0);
        let rung = Rung::new("anthropic/claude-haiku-4-5", 0.62);
        let (mut sum_c, mut n_c, mut sum_i, mut n_i) = (0.0, 0, 0.0, 0);
        for t in &suite {
            let c = be.run(t, &rung);
            let j = gate.judge(t, &rung, &c);
            if c.correct {
                sum_c += j.score;
                n_c += 1;
            } else {
                sum_i += j.score;
                n_i += 1;
            }
        }
        // Mean score of correct outputs must exceed that of incorrect ones for conformal to work.
        assert!(sum_c / n_c as f64 > sum_i / n_i as f64 + 0.2);
    }
}
