//! Self-consistency uncertainty gate: measure a model's own confidence by resampling.
//!
//! **Research basis**
//! - Wang et al. 2022 ("Self-Consistency Improves Chain of Thought Reasoning in Language
//!   Models"): sampling multiple reasoning paths and taking the most-consistent answer
//!   significantly improves accuracy over greedy decoding. Answers a model reliably reproduces
//!   are far likelier correct.
//! - Farquhar et al., *Nature* 2024 ("Detecting Hallucinations in Large Language Models Using
//!   Semantic Entropy"): disagreement across samples — semantic entropy — flags hallucination.
//!   A continuous agreement score lets the conformal threshold machinery calibrate the serve
//!   cutoff against a target failure rate.
//!
//! **Mechanism**: call the same model `k` times on the original request (concurrently — serial
//! would multiply added latency by k), then score the candidate's agreement with each sample.
//! Per-sample agreement: (a) exact final-answer match (1.0 if the last number in the text, or
//! the whole normalized text, matches) or (b) token-set Jaccard similarity. Gate score is the
//! mean over all usable samples, clamped `[0, 1]`. Verdict = Pass iff score ≥ threshold.
//!
//! **maker == checker is ALLOWED** — unlike [`crate::judge::JudgeGate`], self-consistency is
//! definitionally self-referential: the model samples *itself*. This is not a bug; it is the
//! mechanism. The gate enforces no maker ≠ checker constraint.
//!
//! **Honest scope**
//! - ponytail: lexical agreement (final-number extraction + token-set Jaccard) is a cheap proxy
//!   for semantic clustering. Strongest on short, factual, or structured outputs (numbers,
//!   labels, code identifiers). Upgrade to entailment-based semantic clustering via a judge
//!   model for long-form or heavily paraphrastic outputs.
//! - Cost: k extra model calls per request — that is the honest price of measured confidence.
//!   Pair with a cheap model (e.g. haiku) on the first rung so the sampling cost stays low;
//!   expensive frontier models only bear escalation cost when cheaper rungs fail.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use firstpass_core::{GateResult, Score, Verdict, cost::PriceTable};

use crate::gate::Gate;
use crate::provider::{Auth, ModelRequest, ModelResponse, Provider};

/// A gate that scores the candidate by agreement with `k` fresh samples of the same model.
#[derive(Debug)]
pub struct ConsistencyGate {
    id: String,
    provider: Arc<dyn Provider>,
    /// `provider/model` used for resampling. May equal the candidate's model — expected.
    sample_model: String,
    auth: Auth,
    k: u32,
    threshold: f64,
    /// Prices the k sample calls so their cost lands on the receipt ([`GateResult::cost_usd`]).
    prices: PriceTable,
}

impl ConsistencyGate {
    /// Build a consistency gate. `provider` serves `sample_model`; `auth` carries credentials;
    /// `prices` prices the k sample calls onto the receipt.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        provider: Arc<dyn Provider>,
        sample_model: impl Into<String>,
        auth: Auth,
        k: u32,
        threshold: f64,
        prices: PriceTable,
    ) -> Self {
        Self {
            id: id.into(),
            provider,
            sample_model: sample_model.into(),
            auth,
            k,
            threshold,
            prices,
        }
    }
}

#[async_trait]
impl Gate for ConsistencyGate {
    fn id(&self) -> &str {
        &self.id
    }

    async fn evaluate(&self, req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        let start = Instant::now();

        // Build the sample request: same conversation, model swapped to the configured sampler.
        let mut sample_req = req.clone();
        sample_req.model = self.sample_model.clone();

        // Fire k sample calls concurrently — serial would multiply added latency by k.
        let mut set = tokio::task::JoinSet::new();
        for _ in 0..self.k {
            let provider = Arc::clone(&self.provider);
            let req_clone = sample_req.clone();
            let auth_clone = self.auth.clone();
            set.spawn(async move { provider.complete(&req_clone, &auth_clone).await });
        }

        let mut samples: Vec<String> = Vec::new();
        let mut sample_cost = 0.0_f64;
        while let Some(res) = set.join_next().await {
            // Provider error or task panic → drop this sample, score over the rest.
            if let Ok(Ok(sample)) = res {
                // Each sample is a real model call — price it onto the receipt so the router's
                // budget/savings math sees the true cost of measured confidence.
                sample_cost += self
                    .prices
                    .cost_usd(&self.sample_model, sample.in_tokens, sample.out_tokens)
                    .unwrap_or(0.0);
                samples.push(sample.text);
            }
        }

        let ms = elapsed_ms(start);

        // Zero usable samples → abstain; never fabricate a score.
        if samples.is_empty() {
            let mut r = GateResult::abstain(&self.id, "no_usable_samples", ms);
            r.evidence_ref = Some(format!("all {} sample calls failed", self.k));
            r.cost_usd = sample_cost;
            return r;
        }

        let candidate_norm = normalize(&resp.text);
        let candidate_answer = extract_final_answer(&candidate_norm);

        let total: f64 = samples
            .iter()
            .map(|s| {
                let snorm = normalize(s);
                let sanswer = extract_final_answer(&snorm);
                if candidate_answer == sanswer {
                    1.0_f64
                } else {
                    // ponytail: token-set Jaccard over full normalized text. Upgrade to
                    // entailment-based clustering via a judge model for semantic agreement on
                    // long-form or paraphrastic outputs.
                    jaccard(&candidate_norm, &snorm)
                }
            })
            .sum();

        let score_val = (total / samples.len() as f64).clamp(0.0, 1.0);
        let verdict = if score_val >= self.threshold {
            Verdict::Pass
        } else {
            Verdict::Fail
        };

        let mut r = GateResult::deterministic(&self.id, verdict, ms);
        r.score = Score::new(score_val).ok();
        r.cost_usd = sample_cost;
        r
    }
}

/// Normalize text for comparison: trim, lowercase, collapse internal whitespace to single spaces.
fn normalize(text: &str) -> String {
    text.trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract the final answer from normalized text: the last contiguous ASCII-digit sequence,
/// or the whole string when no digits are present.
///
/// This is the cheap "what number did the model settle on?" proxy from self-consistency
/// literature. Strongest on single-number answers (arithmetic, retrieval).
///
// ponytail: digit-only extraction. Upgrade to full expression parsing if answers include
// decimals, fractions, or labelled fields (e.g. "answer: 42.5").
fn extract_final_answer(normalized: &str) -> String {
    let mut last_num = String::new();
    let mut current_num = String::new();
    for c in normalized.chars() {
        if c.is_ascii_digit() {
            current_num.push(c);
        } else if !current_num.is_empty() {
            last_num.clone_from(&current_num);
            current_num.clear();
        }
    }
    if !current_num.is_empty() {
        last_num = current_num;
    }
    if last_num.is_empty() {
        normalized.to_owned()
    } else {
        last_num
    }
}

/// Token-set Jaccard similarity: `|A ∩ B| / |A ∪ B|`, tokens split on whitespace.
/// Two empty strings → 1.0 (identical); one empty vs one non-empty → 0.0.
fn jaccard(a: &str, b: &str) -> f64 {
    let tokens_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let tokens_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
    let inter = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.len() + tokens_b.len() - inter;
    if union == 0 {
        1.0 // both empty → identical
    } else {
        inter as f64 / union as f64
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ChatMessage, MockProvider, ProviderError};
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_req() -> ModelRequest {
        ModelRequest {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            system: None,
            messages: vec![ChatMessage::text("user", "What is 6 * 7?")],
            max_tokens: 64,
            tools: Value::Null,
        }
    }

    fn make_resp(model: &str, text: &str) -> ModelResponse {
        ModelResponse {
            model: model.to_owned(),
            text: text.to_owned(),
            in_tokens: 10,
            out_tokens: 10,
            raw: Value::Null,
        }
    }

    fn provider_returning(sample_model: &str, text: &str) -> Arc<dyn Provider> {
        let mut outcomes = HashMap::new();
        outcomes.insert(sample_model.to_owned(), Ok(make_resp(sample_model, text)));
        Arc::new(MockProvider::new(
            sample_model.split('/').next().unwrap_or("mock"),
            outcomes,
        ))
    }

    fn provider_erroring(sample_model: &str) -> Arc<dyn Provider> {
        let mut outcomes = HashMap::new();
        outcomes.insert(
            sample_model.to_owned(),
            Err(ProviderError::Transport("down".to_owned())),
        );
        Arc::new(MockProvider::new(
            sample_model.split('/').next().unwrap_or("mock"),
            outcomes,
        ))
    }

    /// Agreeing samples (same final answer "42", phrasing differs) → high score, Pass at 0.6.
    #[tokio::test]
    async fn agreeing_samples_pass_at_threshold_0_6() {
        // k=3 samples all return "The answer is 42" while candidate says "Therefore the answer
        // is 42." — final answers agree (both extract to "42") → score 1.0 → Pass.
        let gate = ConsistencyGate::new(
            "uncertainty",
            provider_returning("anthropic/sampler", "The answer is 42"),
            "anthropic/sampler",
            Auth::default(),
            3,
            0.6,
            PriceTable::default(),
        );
        let candidate = make_resp("anthropic/claude-haiku-4-5", "Therefore the answer is 42.");
        let result = gate.evaluate(&make_req(), &candidate).await;
        assert_eq!(result.verdict, Verdict::Pass, "final-answer match → Pass");
        let score = result
            .score
            .expect("score must be present on a non-abstain verdict");
        assert!(
            score.value() >= 0.9,
            "all samples agree → score near 1.0; got {score}"
        );
    }

    /// Disagreeing samples (different numbers) → low score, Fail at threshold 0.6.
    #[tokio::test]
    async fn disagreeing_samples_fail_at_threshold_0_6() {
        // k=3 samples return bare "7" while candidate says "The answer is 42".
        // Different final answers ("7" vs "42"); Jaccard("the answer is 42", "7") = 0/5 = 0.0,
        // well below threshold 0.6 → Fail.
        let gate = ConsistencyGate::new(
            "uncertainty",
            provider_returning("anthropic/sampler", "7"),
            "anthropic/sampler",
            Auth::default(),
            3,
            0.6,
            PriceTable::default(),
        );
        let candidate = make_resp("anthropic/claude-haiku-4-5", "The answer is 42");
        let result = gate.evaluate(&make_req(), &candidate).await;
        assert_eq!(
            result.verdict,
            Verdict::Fail,
            "different final answers → Fail"
        );
    }

    /// Exactly k calls are fired — confirming the concurrent fan-out.
    #[tokio::test]
    async fn exactly_k_sample_calls_are_fired() {
        let k: u32 = 3;
        let mut outcomes = HashMap::new();
        outcomes.insert(
            "anthropic/sampler".to_owned(),
            Ok(make_resp("anthropic/sampler", "42")),
        );
        let mock = MockProvider::new("anthropic", outcomes);
        let call_log = mock.call_log();
        let gate = ConsistencyGate::new(
            "uncertainty",
            Arc::new(mock),
            "anthropic/sampler",
            Auth::default(),
            k,
            0.5,
            PriceTable::default(),
        );
        gate.evaluate(&make_req(), &make_resp("anthropic/claude-haiku-4-5", "42"))
            .await;
        let log = call_log.lock().unwrap();
        assert_eq!(log.len(), k as usize, "exactly k calls must be fired");
        assert!(
            log.iter().all(|m| m == "anthropic/sampler"),
            "all calls target the configured sample model"
        );
    }

    /// All samples error → Abstain with reason "no_usable_samples".
    #[tokio::test]
    async fn all_samples_error_abstains() {
        let gate = ConsistencyGate::new(
            "uncertainty",
            provider_erroring("anthropic/sampler"),
            "anthropic/sampler",
            Auth::default(),
            3,
            0.6,
            PriceTable::default(),
        );
        let candidate = make_resp("anthropic/claude-haiku-4-5", "42");
        let result = gate.evaluate(&make_req(), &candidate).await;
        assert_eq!(result.verdict, Verdict::Abstain);
        assert_eq!(
            result.reason.as_deref(),
            Some("no_usable_samples"),
            "all-error abstain carries the expected reason"
        );
    }

    // ---- unit tests for the scoring helpers ----

    #[test]
    fn normalize_trims_lowercases_and_collapses_whitespace() {
        assert_eq!(normalize("  Hello   World  "), "hello world");
        assert_eq!(normalize("THE ANSWER IS 42"), "the answer is 42");
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn extract_final_answer_returns_last_digit_sequence() {
        assert_eq!(extract_final_answer("the answer is 42"), "42");
        assert_eq!(extract_final_answer("3 times 7 equals 21"), "21");
        // Trailing non-digit — still gets the last number.
        assert_eq!(extract_final_answer("result: 100 units"), "100");
        // No digits → whole string.
        assert_eq!(extract_final_answer("no numbers here"), "no numbers here");
        // Empty string.
        assert_eq!(extract_final_answer(""), "");
    }

    #[test]
    fn jaccard_similarity_boundary_values() {
        // Identical tokens → 1.0.
        assert!((jaccard("a b c", "a b c") - 1.0).abs() < 1e-9);
        // Disjoint tokens → 0.0.
        assert!((jaccard("a b", "c d") - 0.0).abs() < 1e-9);
        // Half overlap: {a,b,c} ∩ {b,c,d} = {b,c}, union = 4 → 0.5.
        let j = jaccard("a b c", "b c d");
        assert!((j - 0.5).abs() < 1e-9, "expected 0.5, got {j}");
        // Both empty → identical (1.0).
        assert!((jaccard("", "") - 1.0).abs() < 1e-9);
        // One empty → 0.0.
        assert!((jaccard("a b", "") - 0.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn sample_cost_lands_on_the_receipt() {
        use firstpass_core::cost::ModelPrice;
        // k=3 samples at 10 in / 10 out tokens each, priced 1.0/5.0 per Mtok.
        let prices = PriceTable::new().with_override(
            "anthropic/sampler",
            ModelPrice {
                input_per_mtok: 1.0,
                output_per_mtok: 5.0,
            },
        );
        let gate = ConsistencyGate::new(
            "uncertainty",
            provider_returning("anthropic/sampler", "42"),
            "anthropic/sampler",
            Auth::default(),
            3,
            0.5,
            prices,
        );
        let r = gate
            .evaluate(&make_req(), &make_resp("anthropic/sampler", "42"))
            .await;
        // 3 samples x (10/1e6*1.0 + 10/1e6*5.0) = 3 x 6e-5 = 1.8e-4.
        let expected = 3.0 * (10.0 / 1e6 * 1.0 + 10.0 / 1e6 * 5.0);
        assert!(
            (r.cost_usd - expected).abs() < 1e-12,
            "k sample calls priced onto the receipt: got {}, want {expected}",
            r.cost_usd
        );
    }
}
