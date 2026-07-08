//! Cost model: per-model token pricing and the counterfactual baseline (SPEC §9.1).
//!
//! The whole product claim is "cheapest model that passes," so cost math is load-bearing.
//! Two numbers matter per trace: what the escalation ladder actually spent, and the
//! **counterfactual baseline** — what always calling the top rung would have cost. Their
//! difference is the savings Firstpass proves.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Published price of a model, in USD per 1,000,000 tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelPrice {
    /// USD per 1M input (prompt) tokens.
    pub input_per_mtok: f64,
    /// USD per 1M output (completion) tokens.
    pub output_per_mtok: f64,
}

impl ModelPrice {
    /// Cost in USD of `input`/`output` tokens at this price.
    #[must_use]
    pub fn cost(&self, input: u64, output: u64) -> f64 {
        (input as f64 / 1e6) * self.input_per_mtok + (output as f64 / 1e6) * self.output_per_mtok
    }
}

/// A lookup from `provider/model` to [`ModelPrice`].
///
/// ponytail: the embedded [`PriceTable::defaults`] are a calibration knob, not gospel —
/// list prices drift and enterprise contracts differ. In prod, load overrides from config
/// via [`PriceTable::with_override`]; the defaults just make the common case work out of the box.
#[derive(Debug, Clone, Default)]
pub struct PriceTable {
    prices: HashMap<String, ModelPrice>,
}

impl PriceTable {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Built-in prices for the common frontier models (USD / 1M tokens, approximate list
    /// prices — a starting point to be overridden per deployment).
    #[must_use]
    pub fn defaults() -> Self {
        let mut prices = HashMap::new();
        let mut put = |k: &str, i: f64, o: f64| {
            prices.insert(
                k.to_owned(),
                ModelPrice {
                    input_per_mtok: i,
                    output_per_mtok: o,
                },
            );
        };
        // Anthropic
        put("anthropic/claude-haiku-4-5", 1.0, 5.0);
        put("anthropic/claude-sonnet-5", 3.0, 15.0);
        put("anthropic/claude-opus-4-8", 15.0, 75.0);
        // OpenAI
        put("openai/gpt-4.1-mini", 0.4, 1.6);
        put("openai/gpt-5.5", 5.0, 15.0);
        // Google
        put("google/gemini-3.1-flash", 0.35, 1.05);
        put("google/gemini-3.1-pro", 3.5, 10.5);
        Self { prices }
    }

    /// Insert or replace a model's price, returning `self` for chaining.
    #[must_use]
    pub fn with_override(mut self, model: impl Into<String>, price: ModelPrice) -> Self {
        self.prices.insert(model.into(), price);
        self
    }

    /// Look up a model's price by its `provider/model` key.
    #[must_use]
    pub fn get(&self, model: &str) -> Option<ModelPrice> {
        self.prices.get(model).copied()
    }

    /// Cost in USD of a call to `model` with the given token counts.
    ///
    /// # Errors
    /// Returns [`Error::UnknownModel`] if the model has no price entry.
    pub fn cost_usd(&self, model: &str, input: u64, output: u64) -> Result<f64> {
        self.get(model)
            .map(|p| p.cost(input, output))
            .ok_or_else(|| Error::UnknownModel(model.to_owned()))
    }

    /// The counterfactual baseline: what the request would have cost had it gone straight to
    /// `top_model` (the ladder's top rung) with the served token counts.
    ///
    /// This is an estimate — the top model might have emitted a different number of tokens —
    /// but token counts of served output are the fair, auditable proxy the trace records.
    ///
    /// # Errors
    /// Returns [`Error::UnknownModel`] if `top_model` has no price entry.
    pub fn baseline_usd(&self, top_model: &str, input: u64, output: u64) -> Result<f64> {
        self.cost_usd(top_model, input, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_math_is_correct() {
        let p = ModelPrice {
            input_per_mtok: 3.0,
            output_per_mtok: 15.0,
        };
        // 1000 in * $3/M + 500 out * $15/M = 0.003 + 0.0075 = 0.0105
        assert!((p.cost(1000, 500) - 0.0105).abs() < 1e-12);
        assert_eq!(p.cost(0, 0), 0.0);
    }

    #[test]
    fn table_lookup_and_unknown_model() {
        let t = PriceTable::defaults();
        assert!(t.cost_usd("anthropic/claude-haiku-4-5", 1000, 1000).is_ok());
        match t.cost_usd("acme/nope", 1, 1) {
            Err(Error::UnknownModel(m)) => assert_eq!(m, "acme/nope"),
            other => panic!("expected UnknownModel, got {other:?}"),
        }
    }

    #[test]
    fn baseline_exceeds_cheap_rung_for_same_tokens() {
        // The core value prop, as a math invariant: top rung costs more than the cheap rung.
        let t = PriceTable::defaults();
        let (i, o) = (2000, 800);
        let cheap = t.cost_usd("anthropic/claude-haiku-4-5", i, o).unwrap();
        let baseline = t.baseline_usd("anthropic/claude-opus-4-8", i, o).unwrap();
        assert!(baseline > cheap);
        assert!(baseline - cheap > 0.0); // there are savings to prove
    }

    #[test]
    fn overrides_win() {
        let t = PriceTable::new().with_override(
            "x/y",
            ModelPrice {
                input_per_mtok: 2.0,
                output_per_mtok: 2.0,
            },
        );
        assert_eq!(t.cost_usd("x/y", 1_000_000, 0).unwrap(), 2.0);
    }
}
