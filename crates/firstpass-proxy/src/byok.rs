//! BYOK metering for observe passthrough (#284).
//!
//! A caller that brings its own provider key is attributed by a SHA-256 of that key — the raw key
//! is never stored or logged. Each metered response emits a receipt hash-chained to the previous
//! receipt for the same key, and `[budget] per_day_usd` (when set) caps each key's spend per UTC
//! day: once spent, further passthrough requests on that key get a 402.

use std::collections::HashMap;
use std::sync::Mutex;

use axum::Json;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use firstpass_core::Trace;
use firstpass_core::hashchain::{Chained, GENESIS_HASH, record_hash, sha256_hex};
use serde::Serialize;

/// Stable attribution id for the caller's provider key, or `None` when the request carries none.
///
/// Checks the headers the passthrough forwards upstream: Anthropic `x-api-key`, Gemini
/// `x-goog-api-key`, then `Authorization` (OpenAI `Bearer`).
#[must_use]
pub fn key_hash(headers: &HeaderMap) -> Option<String> {
    ["x-api-key", "x-goog-api-key", "authorization"]
        .iter()
        .find_map(|h| headers.get(*h))
        .map(|v| {
            let raw = v.as_bytes();
            sha256_hex(raw.strip_prefix(b"Bearer ").unwrap_or(raw))
        })
}

/// One metered passthrough response for a key. Hash = `record_hash` over the canonical JSON of
/// these fields (including `prev_hash`), so the chain re-derives exactly like the trace chain.
#[derive(Debug, Clone, Serialize)]
pub struct Receipt {
    /// SHA-256 of the caller's key.
    pub key_hash: String,
    /// The audit trace this receipt meters.
    pub trace_id: String,
    /// UTC day number (`unix_seconds / 86400`) the spend counts against.
    pub day: i64,
    /// Model reported for the call (empty when unknown).
    pub model: String,
    /// Billable input tokens (uncached + cache read + cache write), from the upstream usage block.
    pub in_tokens: u64,
    /// Output tokens.
    pub out_tokens: u64,
    /// USD cost of this response.
    pub cost_usd: f64,
    /// This key's running spend for `day`, including this response.
    pub day_spent_usd: f64,
    /// Hash of this key's previous receipt, or [`GENESIS_HASH`].
    pub prev_hash: String,
}

impl Chained for Receipt {
    fn prev_hash(&self) -> &str {
        &self.prev_hash
    }
}

/// Per-key `(day, spent_usd, chain_head)`.
// ponytail: in-memory — spend and chain head reset on restart; seed from persisted receipts if
// a restart-proof daily cap is needed.
#[derive(Debug, Default)]
pub struct ByokMeter {
    inner: Mutex<HashMap<String, (i64, f64, String)>>,
}

impl ByokMeter {
    /// Whether `key` has already spent `cap_usd` or more today. A poisoned lock blocks (fail
    /// closed): this is a spend cap, and "budget available" is the unsafe direction.
    pub fn over_budget(&self, key: &str, cap_usd: f64, now: jiff::Timestamp) -> bool {
        let Ok(g) = self.inner.lock() else {
            return true;
        };
        g.get(key)
            .is_some_and(|(day, spent, _)| *day == crate::shadow::day_of(now) && *spent >= cap_usd)
    }

    /// Debit `trace`'s cost to `key`, extend its receipt chain, and return the receipt.
    pub fn record(&self, key: &str, trace: &Trace) -> Option<Receipt> {
        let day = crate::shadow::day_of(trace.ts);
        let mut g = self.inner.lock().ok()?;
        let e = g
            .entry(key.to_owned())
            .or_insert_with(|| (day, 0.0, GENESIS_HASH.to_owned()));
        if e.0 != day {
            e.0 = day;
            e.1 = 0.0;
        }
        let cost = trace.final_.total_cost_usd;
        if cost.is_finite() && cost > 0.0 {
            e.1 += cost;
        }
        let a = trace.attempts.first();
        let receipt = Receipt {
            key_hash: key.to_owned(),
            trace_id: trace.trace_id.to_string(),
            day,
            model: a.map(|a| a.model.clone()).unwrap_or_default(),
            in_tokens: a.map_or(0, firstpass_core::Attempt::billable_input),
            out_tokens: a.map_or(0, |a| a.out_tokens),
            cost_usd: cost,
            day_spent_usd: e.1,
            prev_hash: e.2.clone(),
        };
        e.2 = record_hash(&receipt).ok()?;
        Some(receipt)
    }
}

/// Meter a passthrough trace for `key` and emit its receipt as a structured event. Error traces
/// (no upstream usage) still get a receipt so the per-key chain has no gaps.
pub(crate) fn meter(meter: &ByokMeter, key: &str, trace: &Trace) {
    if let Some(r) = meter.record(key, trace) {
        let receipt = serde_json::to_string(&r).unwrap_or_default();
        tracing::info!(target: "firstpass::byok", %receipt, "byok receipt");
    }
}

/// The passthrough budget gate: the caller's key hash to meter against, or a 402 when that key
/// has spent `[budget] per_day_usd` today. Emits an over-budget event for every blocked request.
pub(crate) fn gate(
    meter: &ByokMeter,
    cap_usd: Option<f64>,
    headers: &HeaderMap,
) -> Result<Option<String>, Box<Response>> {
    let Some(key) = key_hash(headers) else {
        return Ok(None);
    };
    match cap_usd {
        Some(cap) if meter.over_budget(&key, cap, jiff::Timestamp::now()) => {
            metrics::counter!("firstpass_byok_over_budget_total").increment(1);
            tracing::warn!(target: "firstpass::byok", key_hash = %key, "byok over budget: blocked");
            let body = serde_json::json!({ "error": {
                "type": "budget_exceeded",
                "message": "this API key has used its daily budget; it resets at 00:00 UTC",
            }});
            Err(Box::new(
                (StatusCode::PAYMENT_REQUIRED, Json(body)).into_response(),
            ))
        }
        _ => Ok(Some(key)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_hash_is_stable_strips_bearer_and_never_echoes_the_key() {
        let mut a = HeaderMap::new();
        a.insert("x-api-key", "sk-test-123".parse().unwrap());
        let mut b = HeaderMap::new();
        b.insert("authorization", "Bearer sk-test-123".parse().unwrap());
        let h = key_hash(&a).unwrap();
        assert_eq!(h, key_hash(&b).unwrap());
        assert_eq!(h, sha256_hex(b"sk-test-123"));
        assert!(!h.contains("sk-test"));
        assert!(key_hash(&HeaderMap::new()).is_none());
    }
}
