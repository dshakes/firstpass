//! BYOK metering for observe passthrough (#284).
//!
//! A caller that brings its own provider key is attributed by a SHA-256 of the credential that
//! path actually forwards upstream — the raw key is never stored or logged. Each metered response
//! emits a receipt hash-chained to the previous receipt for the same key on the same UTC day, and
//! `[budget] per_day_usd` (when set) caps each key's spend per UTC day: once spent, further
//! passthrough requests on that key get a 402.

use std::collections::HashMap;
use std::sync::Mutex;

use axum::Json;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use firstpass_core::Trace;
use firstpass_core::hashchain::{Chained, GENESIS_HASH, record_hash, sha256_hex};
use serde::Serialize;

/// Credential headers per passthrough path, first present wins. Only headers that path forwards
/// upstream count, so a stray header cannot move the request onto another key's budget.
/// Anthropic authenticates with `x-api-key`, falling back to `Authorization: Bearer` (OAuth).
pub const ANTHROPIC_CREDS: &[&str] = &["x-api-key", "authorization"];
/// OpenAI(-compatible) authenticates with `Authorization: Bearer` only.
pub const OPENAI_CREDS: &[&str] = &["authorization"];
/// Gemini: only `x-goog-api-key` is forwarded (a `?key=` is moved there before forwarding).
pub const GEMINI_CREDS: &[&str] = &["x-goog-api-key"];

/// Distinct keys metered per UTC day. A new key beyond this is refused when a cap is configured
/// (fail closed: evicting would hand the evicted key a fresh budget) and left unmetered otherwise.
pub const MAX_KEYS_PER_DAY: usize = 10_000;

/// Stable attribution id for the credential in the first present header of `creds`, or `None`
/// when there is none. An RFC 6750 `Bearer` scheme is matched case-insensitively and surrounding
/// whitespace is trimmed, so `bearer  tok` and `Bearer tok` are the same key.
#[must_use]
pub fn key_hash(headers: &HeaderMap, creds: &[&str]) -> Option<String> {
    let v = creds.iter().find_map(|h| headers.get(*h))?;
    let v = String::from_utf8_lossy(v.as_bytes());
    let v = v.trim();
    let v = match v.split_once(char::is_whitespace) {
        Some((scheme, rest)) if scheme.eq_ignore_ascii_case("bearer") => rest.trim(),
        _ if v.eq_ignore_ascii_case("bearer") => "",
        _ => v,
    };
    (!v.is_empty()).then(|| sha256_hex(v.as_bytes()))
}

/// One metered passthrough response for a key. Hash = `record_hash` over the canonical JSON of
/// these fields (including `prev_hash`), so the chain re-derives exactly like the trace chain.
/// The chain is per `(key_hash, day)`: each key's first receipt of a UTC day links to genesis.
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
    /// Hash of this key's previous receipt today, or [`GENESIS_HASH`].
    pub prev_hash: String,
}

impl Chained for Receipt {
    fn prev_hash(&self) -> &str {
        &self.prev_hash
    }
}

/// An admitted request: the key it is metered against and the estimate reserved for it, which
/// [`ByokMeter::record`] releases and replaces with the actual cost.
#[derive(Debug, Clone)]
pub struct Admit {
    key: String,
    reserved_usd: f64,
}

#[derive(Debug)]
struct Entry {
    spent: f64,
    reserved: f64,
    last_cost: f64,
    head: String,
}

#[derive(Debug, Default)]
struct Ledger {
    day: i64,
    keys: HashMap<String, Entry>,
    last_cost: f64,
}

impl Ledger {
    /// A new UTC day drops every key: spend resets and the map cannot grow across days.
    fn roll(&mut self, today: i64) {
        if today != self.day {
            self.day = today;
            self.keys.clear();
        }
    }

    fn entry(&mut self, key: &str) -> &mut Entry {
        self.keys.entry(key.to_owned()).or_insert_with(|| Entry {
            spent: 0.0,
            reserved: 0.0,
            last_cost: 0.0,
            head: GENESIS_HASH.to_owned(),
        })
    }
}

/// Why a request was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The key's spend plus in-flight reservations reached the cap.
    Exhausted,
    /// [`MAX_KEYS_PER_DAY`] distinct keys are already metered today.
    TooManyKeys,
}

/// Per-key daily spend, in-flight reservations, and receipt-chain heads.
// ponytail: in-memory and receipts are log events, not stored rows — a restart resets today's
// spend and restarts each key's chain at genesis. Persist receipts and seed from today's rows on
// boot if the cap must survive restarts.
#[derive(Debug, Default)]
pub struct ByokMeter {
    inner: Mutex<Ledger>,
}

impl ByokMeter {
    /// Admit a request on `key`, reserving an estimate of its cost under the same lock that checks
    /// the cap, so concurrent requests cannot all pass a check none of them has paid for yet.
    ///
    /// The estimate is the key's last cost (else the last cost of any key). A key with nothing
    /// spent or reserved is always admitted; after that a request is admitted only if it fits.
    /// Ceiling: on a cold process (no cost seen yet) the estimate is 0, so the first concurrent
    /// wave on a key is not held back. A poisoned lock fails closed when a cap is set.
    ///
    /// # Errors
    /// [`Refusal`] when a cap is configured and the request must not be forwarded.
    pub fn admit(
        &self,
        key: String,
        cap_usd: Option<f64>,
        now: jiff::Timestamp,
    ) -> Result<Option<Admit>, Refusal> {
        let Ok(mut g) = self.inner.lock() else {
            return cap_usd.map_or(Ok(None), |_| Err(Refusal::Exhausted));
        };
        g.roll(crate::shadow::day_of(now));
        if !g.keys.contains_key(&key) && g.keys.len() >= MAX_KEYS_PER_DAY {
            return cap_usd.map_or(Ok(None), |_| Err(Refusal::TooManyKeys));
        }
        let global = g.last_cost;
        let e = g.entry(&key);
        let est = if e.last_cost > 0.0 {
            e.last_cost
        } else {
            global
        };
        if let Some(cap) = cap_usd {
            let committed = e.spent + e.reserved;
            if committed >= cap || (committed > 0.0 && committed + est > cap) {
                return Err(Refusal::Exhausted);
            }
        }
        e.reserved += est;
        Ok(Some(Admit {
            key,
            reserved_usd: est,
        }))
    }

    /// Release `admit`'s reservation, debit `trace`'s actual cost, extend the key's receipt chain,
    /// and return the receipt.
    pub fn record(&self, admit: &Admit, trace: &Trace) -> Option<Receipt> {
        let mut g = self.inner.lock().ok()?;
        g.roll(crate::shadow::day_of(trace.ts));
        let day = g.day;
        let cost = trace.final_.total_cost_usd;
        let cost = if cost.is_finite() && cost > 0.0 {
            cost
        } else {
            0.0
        };
        if cost > 0.0 {
            g.last_cost = cost;
        }
        let e = g.entry(&admit.key);
        e.reserved = (e.reserved - admit.reserved_usd).max(0.0);
        e.spent += cost;
        if cost > 0.0 {
            e.last_cost = cost;
        }
        let a = trace.attempts.first();
        let receipt = Receipt {
            key_hash: admit.key.clone(),
            trace_id: trace.trace_id.to_string(),
            day,
            model: a.map(|a| a.model.clone()).unwrap_or_default(),
            in_tokens: a.map_or(0, firstpass_core::Attempt::billable_input),
            out_tokens: a.map_or(0, |a| a.out_tokens),
            cost_usd: cost,
            day_spent_usd: e.spent,
            prev_hash: e.head.clone(),
        };
        e.head = record_hash(&receipt).ok()?;
        Some(receipt)
    }
}

/// Meter a passthrough trace and emit its receipt as a structured event. Error traces (no
/// upstream usage) still get a receipt, so the chain has no gaps and the reservation is released.
pub(crate) fn meter(meter: &ByokMeter, admit: &Admit, trace: &Trace) {
    if let Some(r) = meter.record(admit, trace) {
        let receipt = serde_json::to_string(&r).unwrap_or_default();
        tracing::info!(target: "firstpass::byok", %receipt, "byok receipt");
    }
}

/// The passthrough budget gate: the admission to meter against, or a 402 when the caller's key
/// (read from `creds`, the headers this path forwards) is out of today's `[budget] per_day_usd`.
/// Emits an over-budget event for every blocked request.
pub(crate) fn gate(
    meter: &ByokMeter,
    cap_usd: Option<f64>,
    headers: &HeaderMap,
    creds: &[&str],
) -> Result<Option<Admit>, Box<Response>> {
    let Some(key) = key_hash(headers, creds) else {
        return Ok(None);
    };
    let refusal = match meter.admit(key.clone(), cap_usd, jiff::Timestamp::now()) {
        Ok(admit) => return Ok(admit),
        Err(r) => r,
    };
    metrics::counter!("firstpass_byok_over_budget_total").increment(1);
    tracing::warn!(target: "firstpass::byok", key_hash = %key, ?refusal, "byok over budget: blocked");
    let message = match refusal {
        Refusal::Exhausted => "this API key has used its daily budget; it resets at 00:00 UTC",
        Refusal::TooManyKeys => {
            "too many distinct API keys metered today; new keys are accepted again at 00:00 UTC"
        }
    };
    let body = serde_json::json!({ "error": { "type": "budget_exceeded", "message": message } });
    Err(Box::new(
        (StatusCode::PAYMENT_REQUIRED, Json(body)).into_response(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, v.parse().unwrap());
        }
        h
    }

    #[test]
    fn key_hash_normalizes_bearer_case_and_whitespace_and_never_echoes_the_key() {
        let want = sha256_hex(b"sk-test-123");
        for v in [
            "Bearer sk-test-123",
            "bearer sk-test-123",
            "BEARER sk-test-123",
            "Bearer   sk-test-123  ",
            "  bearer\tsk-test-123",
        ] {
            assert_eq!(
                key_hash(&hdr(&[("authorization", v)]), OPENAI_CREDS).as_deref(),
                Some(want.as_str()),
                "{v:?}"
            );
        }
        let h = key_hash(&hdr(&[("x-api-key", "sk-test-123")]), ANTHROPIC_CREDS).unwrap();
        assert_eq!(h, want);
        assert!(!h.contains("sk-test"));
        assert!(key_hash(&HeaderMap::new(), ANTHROPIC_CREDS).is_none());
        assert!(key_hash(&hdr(&[("authorization", "Bearer  ")]), OPENAI_CREDS).is_none());
    }

    #[test]
    fn a_stray_header_does_not_change_the_metered_key() {
        let real = sha256_hex(b"real");
        // Gemini forwards only x-goog-api-key; OpenAI only Authorization.
        let h = hdr(&[("x-goog-api-key", "real"), ("x-api-key", "stray")]);
        assert_eq!(key_hash(&h, GEMINI_CREDS), Some(real.clone()));
        let h = hdr(&[("authorization", "Bearer real"), ("x-api-key", "stray")]);
        assert_eq!(key_hash(&h, OPENAI_CREDS), Some(real.clone()));
        // A stray credential the path does not forward is not a key at all.
        assert!(key_hash(&hdr(&[("x-api-key", "stray")]), OPENAI_CREDS).is_none());
        assert!(key_hash(&hdr(&[("authorization", "Bearer s")]), GEMINI_CREDS).is_none());
    }

    #[test]
    fn exhausted_key_stays_blocked_when_a_stray_header_is_added() {
        let meter = ByokMeter::default();
        let now = jiff::Timestamp::now();
        // The real key has already spent its whole cap today.
        meter.inner.lock().unwrap().roll(crate::shadow::day_of(now));
        meter
            .inner
            .lock()
            .unwrap()
            .entry(&sha256_hex(b"real"))
            .spent = 1.0;
        let h = hdr(&[
            ("authorization", "Bearer real"),
            ("x-api-key", "fresh-stray"),
        ]);
        let resp = gate(&meter, Some(1.0), &h, OPENAI_CREDS).unwrap_err();
        assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
    }

    #[test]
    fn distinct_keys_are_capped_per_day_and_reset_on_rollover() {
        let meter = ByokMeter::default();
        let now = jiff::Timestamp::now();
        for i in 0..MAX_KEYS_PER_DAY {
            meter.admit(format!("k{i}"), Some(1.0), now).unwrap();
        }
        assert_eq!(
            meter.admit("new".into(), Some(1.0), now).unwrap_err(),
            Refusal::TooManyKeys,
            "fail closed when full and a cap is set"
        );
        assert!(
            meter.admit("k0".into(), Some(1.0), now).is_ok(),
            "known key still admitted"
        );
        assert!(
            meter.admit("new".into(), None, now).unwrap().is_none(),
            "no cap: served unmetered rather than blocked"
        );
        let tomorrow = now + jiff::SignedDuration::from_hours(24);
        assert!(
            meter
                .admit("new".into(), Some(1.0), tomorrow)
                .unwrap()
                .is_some()
        );
        assert_eq!(
            meter.inner.lock().unwrap().keys.len(),
            1,
            "rollover drops old keys"
        );
    }
}
