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

    /// Cost in USD including prompt-cache traffic, which bills at different rates than plain input.
    ///
    /// Cached prompt tokens are **not** free and **not** priced like ordinary input:
    /// - writing to the cache costs a premium ([`CACHE_WRITE_MULTIPLIER`]), because the provider
    ///   stores the prefix;
    /// - reading from it is heavily discounted ([`CACHE_READ_MULTIPLIER`]).
    ///
    /// Ignoring both — which is what counting `input_tokens` alone does — understates a cached
    /// request by orders of magnitude. A 190k-token cached prefix reports an `input_tokens` of
    /// about 20, so the receipt reads a fraction of a cent for a call that cost most of a dollar,
    /// and `[budget]` caps fed by that total never trip.
    #[must_use]
    pub fn cost_with_cache(
        &self,
        input: u64,
        cache_write: u64,
        cache_read: u64,
        output: u64,
    ) -> f64 {
        self.cost(input, output)
            + (cache_write as f64 / 1e6) * self.input_per_mtok * CACHE_WRITE_MULTIPLIER
            + (cache_read as f64 / 1e6) * self.input_per_mtok * CACHE_READ_MULTIPLIER
    }
}

/// Premium on input rate for tokens written to the provider's prompt cache.
///
/// Anthropic's published 5-minute-TTL rate. ponytail: one constant rather than a per-model column —
/// every provider that offers prompt caching currently prices it as a fixed multiple of its own
/// input rate, so a column would be the same number repeated. Move it into [`ModelPrice`] the day
/// one of them diverges.
pub const CACHE_WRITE_MULTIPLIER: f64 = 1.25;

/// Discount on input rate for tokens served from the provider's prompt cache (Anthropic: 0.1×).
pub const CACHE_READ_MULTIPLIER: f64 = 0.1;

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
        // Google. `gemini-3.1-flash`/`-pro` were priced here but never existed on the API —
        // `generateContent` 404s them — so the shipped Google ladder failed on its first call.
        // These two are GA per Google's model + pricing docs.
        // ponytail: Google prices Pro in tiers (>200k-token prompts cost more) and this table
        // is flat; these are the ≤200k rates. Pin your own with `[[price]]` if you run long
        // prompts on Pro.
        put("google/gemini-3.5-flash-lite", 0.3, 2.5);
        put("google/gemini-3.6-flash", 1.5, 7.5);
        put("google/gemini-2.5-pro", 1.25, 10.0);
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
    /// **Only correct for a response with no prompt-cache traffic.** Any path pricing a real
    /// provider response should call [`PriceTable::cost_usd_with_cache`] instead and pass the
    /// cache counters — this one silently prices a cached call at a small fraction of what it
    /// cost, which is how a ~$0.72 request once banked a $0.0001 receipt. Reserved for
    /// counterfactuals and synthetic token counts, where there is no cache traffic by
    /// construction.
    ///
    /// # Errors
    /// Returns [`Error::UnknownModel`] if the model has no price entry.
    pub fn cost_usd(&self, model: &str, input: u64, output: u64) -> Result<f64> {
        self.cost_usd_with_cache(model, input, 0, 0, output)
    }

    /// As [`PriceTable::cost_usd`], including prompt-cache traffic at its own rates.
    ///
    /// # Errors
    /// Returns [`Error::UnknownModel`] if the model has no price entry.
    pub fn cost_usd_with_cache(
        &self,
        model: &str,
        input: u64,
        cache_write: u64,
        cache_read: u64,
        output: u64,
    ) -> Result<f64> {
        self.get(model)
            .map(|p| p.cost_with_cache(input, cache_write, cache_read, output))
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
    /// A cached request must not be billed as if the cached prompt were free.
    ///
    /// This is the regression that mattered: usage was read as `input_tokens` alone, and a caller
    /// using prompt caching reports the bulk of its prompt in the cache counters. A 190k-token
    /// cached prefix arrives as an `input_tokens` of ~20, so the receipt recorded a fraction of a
    /// cent for a call that cost most of a dollar — in a product whose claim is a tamper-evident
    /// cost record, and with `[budget]` caps fed by that same total.
    #[test]
    fn a_cached_prompt_is_billed_not_ignored() {
        // sonnet-5: $3/1M input, $15/1M output.
        let p = super::ModelPrice {
            input_per_mtok: 3.0,
            output_per_mtok: 15.0,
        };

        // What a real cached Claude Code turn looks like: tiny uncached remainder, huge cache write.
        let naive = p.cost(20, 400);
        let real = p.cost_with_cache(20, 190_000, 0, 400);

        assert!(
            real > naive * 50.0,
            "cache-aware cost {real} must dwarf the naive {naive}; that gap is the bug"
        );
        // 20 in + 400 out + 190k written at 1.25x: 0.00006 + 0.006 + 0.7125
        assert!((real - 0.71856).abs() < 1e-5, "got {real}");
    }

    #[test]
    fn a_cache_read_is_cheap_but_never_free() {
        let p = super::ModelPrice {
            input_per_mtok: 3.0,
            output_per_mtok: 15.0,
        };
        // 190k read from cache at 0.1x = $0.057, versus $0.57 had it been uncached input.
        let read = p.cost_with_cache(0, 0, 190_000, 0);
        assert!((read - 0.057).abs() < 1e-6, "got {read}");
        assert!(read > 0.0, "a cache read is discounted, not free");
        let uncached = p.cost(190_000, 0);
        assert!(read < uncached, "{read} should undercut {uncached}");
    }

    #[test]
    fn zero_cache_traffic_bills_exactly_as_before() {
        // Every existing receipt and test must be unaffected: no cache traffic, no change.
        let p = super::ModelPrice {
            input_per_mtok: 3.0,
            output_per_mtok: 15.0,
        };
        assert!((p.cost_with_cache(1000, 0, 0, 500) - p.cost(1000, 500)).abs() < f64::EPSILON);
    }

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
