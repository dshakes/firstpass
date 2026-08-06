//! A response cache that can only ever return an answer which **passed a gate**.
//!
//! ## Why this is not the same feature everyone else ships
//!
//! A conventional semantic or exact cache stores whatever the model said and replays it. It saves
//! money by skipping the model — and skipping the model is exactly how a gateway stops knowing
//! whether the thing it just served was any good. On a cache hit rate of 40%, a product whose claim
//! is "nothing ships until a real check passes" would be shipping 40% of its traffic unchecked.
//!
//! So insertion is the gate here, not lookup: an entry is written **only** when the request it came
//! from was served on a passing verdict. A failed answer, a best-effort serve after budget
//! exhaustion, an abstain — none of them are cacheable, because replaying one would launder an
//! unverified answer into a served one and no later reader could tell.
//!
//! The receipt records `served_from: cache` and `cache_source: <trace_id>`, so a hit is not the
//! word "cached" but a link to the decision whose gate passed. That is the difference between
//! asserting a saving and being able to prove what was served.
//!
//! ## What it is keyed on
//!
//! The salted prompt hash the router already computes, plus the ladder identity. Exact-match, not
//! semantic: two prompts that merely *look* similar can want different answers, and a verification
//! product guessing that they don't is the one shortcut it cannot take. Semantic matching would
//! need its own gate over the similarity decision, which is a bigger design than a cache.
//!
//! The ladder is part of the key because the same prompt under a different ladder is a different
//! question — an answer proven against one gate configuration says nothing about another.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use firstpass_core::{ServedFrom, Trace, Verdict};

use crate::provider::ModelResponse;

/// An answer that passed, kept for replay.
#[derive(Debug, Clone)]
pub struct Entry {
    /// The response that was served.
    ///
    /// Stored as the normalized [`ModelResponse`] rather than raw wire bytes, so a replay flows
    /// back out through the same dialect-shaping the live path uses. Caching bytes would mean an
    /// answer cached from an Anthropic caller could not be replayed to an OpenAI one, and the two
    /// paths would drift.
    pub response: ModelResponse,
    /// The decision that proved it — recorded on every replay's receipt.
    pub source: uuid::Uuid,
    /// What that decision cost, so a replay can report the spend it avoided rather than guess.
    pub original_cost_usd: f64,
    /// Insert time, for TTL.
    stored: Instant,
}

/// Key: tenant, then the salted prompt hash, then the ladder that answered it.
///
/// **The tenant is not optional.** The prompt salt is per-deployment, not per-tenant, so two
/// tenants sending an identical prompt produce an identical hash — a key without the tenant would
/// serve one customer's paid-for, verified answer to another. That is a cross-tenant data leak in
/// a multi-tenant proxy, not merely a wrong cache hit, and it is silent: the receipt would look
/// entirely normal.
fn key(tenant: &str, prompt_hash: &str, ladder: &[String]) -> String {
    format!("{tenant}|{prompt_hash}|{}", ladder.join(">"))
}

/// A bounded, in-process store of verified answers.
#[derive(Debug)]
pub struct VerifiedCache {
    entries: Mutex<HashMap<String, Entry>>,
    ttl: Duration,
    max_entries: usize,
}

impl VerifiedCache {
    /// Build a cache holding at most `max_entries` for `ttl`.
    #[must_use]
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl,
            max_entries,
        }
    }

    /// Look up a proven answer for this prompt under this ladder.
    #[must_use]
    pub fn get(
        &self,
        tenant: &str,
        prompt_hash: &str,
        ladder: &[String],
        now: Instant,
    ) -> Option<Entry> {
        let Ok(map) = self.entries.lock() else {
            // A poisoned lock must not take serving down. Missing the cache is always safe: the
            // request simply runs the ladder, which is the behaviour with caching off.
            return None;
        };
        map.get(&key(tenant, prompt_hash, ladder))
            .filter(|e| now.duration_since(e.stored) < self.ttl)
            .cloned()
    }

    /// Offer a finished decision to the cache.
    ///
    /// Returns whether it was stored. Everything about the safety of this feature is in the
    /// rejection rules below, so they are checked here — in one place, against the receipt — rather
    /// than trusted to each call site to have got right.
    pub fn offer(
        &self,
        trace: &Trace,
        response: &ModelResponse,
        ladder: &[String],
        now: Instant,
    ) -> bool {
        // 1. It must have been served from a real attempt. `BestAttempt` is the budget-exhausted
        //    fallback — served without a passing verdict, and caching it would replay an answer
        //    that never passed anything.
        if trace.final_.served_from != ServedFrom::Attempt {
            return false;
        }
        // 2. Never re-cache a cache hit. Harmless in itself, but it would relocate `cache_source`
        //    onto a replay rather than the decision that actually ran a gate, and the provenance
        //    chain is the feature.
        if trace.final_.cache_source.is_some() {
            return false;
        }
        // 3. The served rung's verdict must be an unambiguous Pass. An Abstain is "the gate could
        //    not tell", which is precisely the state that must not be frozen and replayed.
        let Some(served) = trace.final_.served_rung else {
            return false;
        };
        let passed = trace
            .attempts
            .iter()
            .any(|a| a.rung == served && a.verdict == Verdict::Pass);
        if !passed {
            return false;
        }
        // 4. A deferred verdict already on the receipt that says Fail retracts the proof.
        //
        //    This is rarely the path that matters: deferred verdicts usually arrive minutes later
        //    via `/v1/feedback`, long after insertion, so this check alone would be close to a
        //    no-op — a safety rule that never fires, which is worse than none because the comment
        //    claims protection it does not give. `retract()` below is the half that actually runs
        //    in production, called from the feedback handler.
        if trace.deferred.iter().any(|d| d.verdict == Verdict::Fail) {
            return false;
        }

        let Ok(mut map) = self.entries.lock() else {
            return false;
        };
        if map.len() >= self.max_entries {
            map.retain(|_, e| now.duration_since(e.stored) < self.ttl);
            if map.len() >= self.max_entries {
                // ponytail: drop the oldest half on the rare over-cap write rather than maintain an
                // LRU list on every read. A cache is an optimisation; its bookkeeping should not
                // cost more than the thing it saves.
                let mut by_age: Vec<(String, Instant)> =
                    map.iter().map(|(k, e)| (k.clone(), e.stored)).collect();
                by_age.sort_by_key(|(_, t)| *t);
                for (k, _) in by_age.into_iter().take(self.max_entries.max(2) / 2) {
                    map.remove(&k);
                }
            }
        }
        map.insert(
            // The tenant comes off the RECEIPT, not from a parameter a caller could pass wrong —
            // the same reason the pass-only rule is checked against the receipt.
            key(&trace.tenant_id, &trace.request.prompt_hash, ladder),
            Entry {
                response: response.clone(),
                source: trace.trace_id,
                original_cost_usd: trace.final_.total_cost_usd,
                stored: now,
            },
        );
        true
    }

    /// Evict every entry proven by `source`, because that proof has been retracted.
    ///
    /// Called when a deferred gate or downstream outcome reports a Fail for a decision. Without
    /// this, an answer the world has since disproven keeps being replayed until its TTL expires —
    /// and each replay carries a `cache_source` pointing at a receipt that now records a failure,
    /// so the audit trail would actively contradict what was served.
    ///
    /// Returns how many entries were dropped. A single decision can back several keys once the
    /// same prompt is seen under more than one ladder, so this scans by source rather than
    /// assuming one.
    pub fn retract(&self, source: uuid::Uuid) -> usize {
        let Ok(mut map) = self.entries.lock() else {
            return 0;
        };
        let before = map.len();
        map.retain(|_, e| e.source != source);
        before - map.len()
    }

    /// Entries currently held (tests and `/metrics`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Whether the cache holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use firstpass_core::{
        Attempt, DeferredVerdict, Features, FinalOutcome, GENESIS_HASH, GateResult, Mode,
        PolicyRef, RequestInfo, TaskKind,
    };

    fn resp(text: &str) -> ModelResponse {
        ModelResponse {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            text: text.to_owned(),
            in_tokens: 10,
            cache_write_tokens: 0,
            cache_read_tokens: 0,
            out_tokens: 5,
            raw: serde_json::Value::Null,
        }
    }

    fn ladder() -> Vec<String> {
        vec!["anthropic/claude-haiku-4-5".to_owned()]
    }

    fn trace(served_from: ServedFrom, verdict: Verdict) -> Trace {
        Trace {
            trace_id: uuid::Uuid::now_v7(),
            prev_hash: GENESIS_HASH.to_owned(),
            tenant_id: "t".to_owned(),
            session_id: "s".to_owned(),
            ts: jiff::Timestamp::UNIX_EPOCH,
            mode: Mode::Enforce,
            policy: PolicyRef {
                id: "static-ladder@v0".to_owned(),
                explore: false,
                propensity: None,
                mode_profile: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".to_owned(),
                prompt_hash: "hash-abc".to_owned(),
                features: Features::new(TaskKind::CodeEdit),
            },
            attempts: vec![Attempt {
                rung: 0,
                model: "anthropic/claude-haiku-4-5".to_owned(),
                provider: "anthropic".to_owned(),
                in_tokens: 10,
                cache_write_tokens: 0,
                cache_read_tokens: 0,
                out_tokens: 5,
                cost_usd: 0.01,
                latency_ms: 100,
                gates: vec![GateResult::deterministic("non-empty", verdict, 1)],
                verdict,
            }],
            final_: FinalOutcome {
                served_rung: Some(0),
                served_from,
                total_cost_usd: 0.01,
                gate_cost_usd: 0.0,
                total_latency_ms: 120,
                escalations: 0,
                counterfactual_baseline_usd: 0.05,
                savings_usd: 0.04,
                cache_source: None,
            },
            deferred: vec![],
            predicted_pass: None,
            probe: None,
            elastic: None,
            rollout: None,
            shadow: None,
            route_ix: None,
        }
    }

    #[test]
    fn a_passing_answer_is_stored_and_replayed_with_its_proof() {
        let c = VerifiedCache::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        let t = trace(ServedFrom::Attempt, Verdict::Pass);

        assert!(c.offer(&t, &resp("the answer"), &ladder(), now));

        let hit = c.get("t", "hash-abc", &ladder(), now).expect("hit");
        assert_eq!(hit.response.text, "the answer");
        // Not merely "cached" — the decision whose gate passed, so the proof is one lookup away.
        assert_eq!(hit.source, t.trace_id);
        assert!((hit.original_cost_usd - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn a_failed_answer_is_never_cached() {
        let c = VerifiedCache::new(Duration::from_secs(60), 100);
        let now = Instant::now();

        assert!(!c.offer(
            &trace(ServedFrom::Attempt, Verdict::Fail),
            &resp("bad"),
            &ladder(),
            now
        ));

        assert!(
            c.get("t", "hash-abc", &ladder(), now).is_none(),
            "must not be replayable"
        );
    }

    #[test]
    fn an_abstain_is_never_cached() {
        // "The gate could not tell" is exactly the state that must not be frozen and replayed —
        // every later hit would inherit an uncertainty nobody re-examines.
        let c = VerifiedCache::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        assert!(!c.offer(
            &trace(ServedFrom::Attempt, Verdict::Abstain),
            &resp("?"),
            &ladder(),
            now
        ));
        assert!(c.get("t", "hash-abc", &ladder(), now).is_none());
    }

    #[test]
    fn a_budget_exhausted_best_effort_serve_is_never_cached() {
        // BestAttempt is served WITHOUT a passing verdict. Caching it would launder an unverified
        // answer into a served one, and no later reader could tell the difference.
        let c = VerifiedCache::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        assert!(!c.offer(
            &trace(ServedFrom::BestAttempt, Verdict::Pass),
            &resp("x"),
            &ladder(),
            now
        ));
        assert!(c.get("t", "hash-abc", &ladder(), now).is_none());
    }

    #[test]
    fn a_later_failing_verdict_retracts_the_proof() {
        // A deferred gate (tests in CI, downstream feedback) that comes back Fail means the answer
        // was not good after all. It must not be sitting in the cache waiting to be served again.
        let c = VerifiedCache::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        let mut t = trace(ServedFrom::Attempt, Verdict::Pass);
        t.deferred.push(DeferredVerdict {
            gate_id: "tests".to_owned(),
            verdict: Verdict::Fail,
            score: None,
            reported_at: jiff::Timestamp::UNIX_EPOCH,
            reporter: "ci".to_owned(),
        });

        assert!(!c.offer(&t, &resp("looked fine"), &ladder(), now));
    }

    #[test]
    fn a_retraction_evicts_the_answer_it_disproved() {
        // Found in review: rule 4 only inspects the receipt at INSERT time, and deferred verdicts
        // arrive later via /v1/feedback — so on its own it is a no-op in production and a
        // disproven answer keeps being served until TTL. This is the half that actually runs.
        let c = VerifiedCache::new(Duration::from_secs(600), 100);
        let now = Instant::now();
        let t = trace(ServedFrom::Attempt, Verdict::Pass);
        assert!(c.offer(&t, &resp("looked fine"), &ladder(), now));
        assert!(c.get("t", "hash-abc", &ladder(), now).is_some());

        assert_eq!(c.retract(t.trace_id), 1);

        assert!(
            c.get("t", "hash-abc", &ladder(), now).is_none(),
            "an answer whose proof was retracted must stop being replayed immediately"
        );
    }

    #[test]
    fn retracting_an_unrelated_decision_evicts_nothing() {
        let c = VerifiedCache::new(Duration::from_secs(600), 100);
        let now = Instant::now();
        c.offer(
            &trace(ServedFrom::Attempt, Verdict::Pass),
            &resp("a"),
            &ladder(),
            now,
        );

        assert_eq!(c.retract(uuid::Uuid::now_v7()), 0);
        assert!(c.get("t", "hash-abc", &ladder(), now).is_some());
    }

    #[test]
    fn a_cache_hit_is_not_itself_re_cached() {
        // Otherwise `cache_source` would point at a replay instead of the decision that actually
        // ran a gate, and the provenance chain — the whole feature — would go one hop stale.
        let c = VerifiedCache::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        let mut t = trace(ServedFrom::Attempt, Verdict::Pass);
        t.final_.cache_source = Some(uuid::Uuid::now_v7());
        assert!(!c.offer(&t, &resp("replayed"), &ladder(), now));
    }

    #[test]
    fn one_tenant_can_never_replay_anothers_answer() {
        // Caught in review, and the worst bug in this feature: the key was
        // `prompt_hash | ladder`, with no tenant. The prompt salt is per-DEPLOYMENT, so two
        // tenants sending an identical prompt hash identically — tenant B would have been served
        // tenant A's paid-for, verified answer, with a receipt that looked entirely normal.
        let c = VerifiedCache::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        let mut t = trace(ServedFrom::Attempt, Verdict::Pass);
        t.tenant_id = "tenant-a".to_owned();
        assert!(c.offer(&t, &resp("a's private answer"), &ladder(), now));

        assert!(
            c.get("tenant-b", "hash-abc", &ladder(), now).is_none(),
            "cross-tenant replay — this is a data leak, not a cache miss"
        );
        assert!(
            c.get("tenant-a", "hash-abc", &ladder(), now).is_some(),
            "the owning tenant must still hit"
        );
    }

    #[test]
    fn the_cached_tenant_comes_from_the_receipt_not_the_caller() {
        // Reading it off the receipt means a call site cannot pass the wrong tenant and quietly
        // file an answer under someone else's key.
        let c = VerifiedCache::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        let mut t = trace(ServedFrom::Attempt, Verdict::Pass);
        t.tenant_id = "owner".to_owned();
        c.offer(&t, &resp("x"), &ladder(), now);

        assert!(c.get("owner", "hash-abc", &ladder(), now).is_some());
        assert!(c.get("t", "hash-abc", &ladder(), now).is_none());
    }

    #[test]
    fn a_different_ladder_is_a_different_question() {
        // An answer proven against one gate/ladder configuration says nothing about another.
        let c = VerifiedCache::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        c.offer(
            &trace(ServedFrom::Attempt, Verdict::Pass),
            &resp("a"),
            &ladder(),
            now,
        );

        let other = vec!["openai/gpt-5.5".to_owned()];
        assert!(c.get("t", "hash-abc", &other, now).is_none());
        assert!(c.get("t", "hash-abc", &ladder(), now).is_some());
    }

    #[test]
    fn an_entry_expires() {
        let c = VerifiedCache::new(Duration::from_secs(1), 100);
        let t0 = Instant::now();
        c.offer(
            &trace(ServedFrom::Attempt, Verdict::Pass),
            &resp("a"),
            &ladder(),
            t0,
        );

        assert!(c.get("t", "hash-abc", &ladder(), t0).is_some());
        assert!(
            c.get("t", "hash-abc", &ladder(), t0 + Duration::from_secs(2))
                .is_none(),
            "a stale proof is not a proof"
        );
    }

    #[test]
    fn the_cache_cannot_grow_without_bound() {
        let c = VerifiedCache::new(Duration::from_secs(600), 10);
        let now = Instant::now();
        for i in 0..100 {
            let mut t = trace(ServedFrom::Attempt, Verdict::Pass);
            t.request.prompt_hash = format!("h{i}");
            c.offer(&t, &resp("a"), &ladder(), now);
        }
        assert!(c.len() <= 10, "expected <= 10 entries, got {}", c.len());
    }
}
