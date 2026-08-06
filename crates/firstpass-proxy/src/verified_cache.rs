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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Insert time, for the in-process store's TTL.
    ///
    /// Skipped on the wire and re-stamped on read: an `Instant` is meaningless in another process,
    /// and expiry in the shared store is Redis's own TTL rather than a timestamp we compare. Two
    /// replicas comparing monotonic clocks would disagree about what is stale.
    #[serde(skip, default = "Instant::now")]
    stored: Instant,
}

/// Whether a finished decision may be cached at all.
///
/// **The single place these rules live.** They were duplicated nowhere and must stay that way: a
/// second backend re-deriving "was this proven?" from its own reading of the receipt is how one of
/// them eventually accepts something the other would reject, and the failure is silent — an
/// unverified answer served from cache looks exactly like a verified one.
///
/// Rejects, in order:
/// 1. **Not served from a real attempt.** `BestAttempt` is the budget-exhausted fallback, served
///    *without* a passing verdict; caching it would replay an answer that never passed anything.
/// 2. **A cache hit.** Harmless in itself, but it would relocate `cache_source` onto a replay
///    rather than the decision that ran a gate, letting the provenance chain go a hop stale.
/// 3. **A verdict that is not an unambiguous Pass.** An `Abstain` is "the gate could not tell",
///    which is precisely the state that must never be frozen and replayed.
/// 4. **A deferred Fail already on the receipt.** Rarely the path that matters — deferred verdicts
///    usually arrive later via `/v1/feedback`, which is what `retract` handles — but free to check.
#[must_use]
pub fn is_cacheable(trace: &Trace) -> bool {
    if trace.final_.served_from != ServedFrom::Attempt {
        return false;
    }
    if trace.final_.cache_source.is_some() {
        return false;
    }
    let Some(served) = trace.final_.served_rung else {
        return false;
    };
    if !trace
        .attempts
        .iter()
        .any(|a| a.rung == served && a.verdict == Verdict::Pass)
    {
        return false;
    }
    !trace.deferred.iter().any(|d| d.verdict == Verdict::Fail)
}

impl Entry {
    /// Build an entry from a decision and what it served.
    ///
    /// Takes the proof off the trace rather than as a parameter, for the same reason the tenant is
    /// read off the receipt: a call site cannot then file an answer under the wrong decision.
    #[must_use]
    pub fn new(response: ModelResponse, trace: &Trace, now: Instant) -> Self {
        Self {
            response,
            source: trace.trace_id,
            original_cost_usd: trace.final_.total_cost_usd,
            stored: now,
        }
    }
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

/// Where verified entries live.
///
/// Exists so the in-process store stays exactly what it was — the version four review rounds
/// hardened — while a shared L2 can sit behind it. A single-instance deployment keeps byte-identical
/// behaviour and no Redis dependency; a multi-replica one gets entries and, critically,
/// **retractions** that cross process boundaries.
///
/// Async because an L2 is a network hop. Blocking one inside an async proxy would stall the runtime
/// on a cache lookup, which is precisely the wrong place to lose latency.
#[async_trait::async_trait]
pub trait CacheStore: Send + Sync + std::fmt::Debug {
    /// Fetch a live entry, if one exists.
    async fn get(&self, key: &str, now: Instant) -> Option<Entry>;
    /// Store an entry under `key`, proven by `source`.
    ///
    /// `source` is passed separately because a backend has to index by it: retraction arrives
    /// naming a decision, not a key, and one decision can back several keys once the same prompt
    /// is seen under more than one ladder.
    async fn put(&self, key: &str, source: uuid::Uuid, entry: Entry, now: Instant);
    /// Drop every entry proven by `source`. Returns how many went.
    async fn retract(&self, source: uuid::Uuid) -> usize;
    /// Entries currently held, for tests and `/metrics`.
    async fn entry_count(&self) -> usize;
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
        if !is_cacheable(trace) {
            return false;
        }
        let mut entry = Entry {
            response: response.clone(),
            source: trace.trace_id,
            original_cost_usd: trace.final_.total_cost_usd,
            stored: now,
        };
        entry.stored = now;
        self.put_by_key(
            &key(&trace.tenant_id, &trace.request.prompt_hash, ladder),
            entry,
        );
        true
    }

    /// Fetch by a pre-built key. The key-level primitive behind [`CacheStore`]; the
    /// tenant/prompt/ladder composition lives in [`key`] so both stores agree on it.
    #[must_use]
    pub fn get_by_key(&self, key: &str, now: Instant) -> Option<Entry> {
        let Ok(map) = self.entries.lock() else {
            return None;
        };
        map.get(key)
            .filter(|e| now.duration_since(e.stored) < self.ttl)
            .cloned()
    }

    /// Store by a pre-built key, applying the same bound as `offer`.
    pub fn put_by_key(&self, key: &str, entry: Entry) {
        let Ok(mut map) = self.entries.lock() else {
            return;
        };
        let now = entry.stored;
        if map.len() >= self.max_entries {
            map.retain(|_, e| now.duration_since(e.stored) < self.ttl);
            if map.len() >= self.max_entries {
                let mut by_age: Vec<(String, Instant)> =
                    map.iter().map(|(k, e)| (k.clone(), e.stored)).collect();
                by_age.sort_by_key(|(_, t)| *t);
                for (k, _) in by_age.into_iter().take(self.max_entries.max(2) / 2) {
                    map.remove(&k);
                }
            }
        }
        map.insert(key.to_owned(), entry);
    }

    /// Compose the key both stores use, so an entry written by one is found by the other.
    #[must_use]
    pub fn compose_key(tenant: &str, prompt_hash: &str, ladder: &[String]) -> String {
        key(tenant, prompt_hash, ladder)
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

#[async_trait::async_trait]
impl CacheStore for VerifiedCache {
    // Delegates to the existing synchronous methods rather than reimplementing them. The
    // in-process path is the one four review rounds hardened — a parallel async copy would be free
    // to drift on exactly the rules that keep an unproven answer out.
    async fn get(&self, key: &str, now: Instant) -> Option<Entry> {
        self.get_by_key(key, now)
    }

    async fn put(&self, key: &str, _source: uuid::Uuid, entry: Entry, _now: Instant) {
        self.put_by_key(key, entry);
    }

    async fn retract(&self, source: uuid::Uuid) -> usize {
        Self::retract(self, source)
    }

    async fn entry_count(&self) -> usize {
        Self::len(self)
    }
}

/// A Redis-backed L2, so entries and retractions cross process boundaries.
///
/// Enabled by the `redis-cache` feature and a `redis_url`. Without it the cache is per-replica:
/// behind N instances a session's answer is cached N times over and the hit rate drops roughly by
/// N. Worse, a retraction only reaches the replica that received the feedback — the other N-1 keep
/// serving an answer the world has disproven, which is the multi-instance form of the bug this
/// module already had once.
///
/// ## Keys
///
/// - `fp:vc:<key>` → the serialized entry, with a Redis TTL so expiry is the server's job rather
///   than ours. A manual sweep across replicas would race; `SETEX` does not.
/// - `fp:vc:src:<uuid>` → a SET of the entry keys that decision proved, so `retract` can find them.
///   Given the same TTL, so the index cannot outlive what it points at and leak.
///
/// ## What it deliberately does not do
///
/// No local caching of L2 reads. A stale local copy would survive a retraction issued on another
/// replica — exactly the failure this exists to prevent — and a cache that can serve a disproven
/// answer is worse than no cache in a product whose claim is that it does not.
#[cfg(feature = "redis-cache")]
pub struct RedisStore {
    client: redis::aio::ConnectionManager,
    ttl_secs: u64,
}

// Hand-written: `ConnectionManager` is not `Debug`, and printing it would risk putting a
// credential-bearing URL into a log line anyway.
#[cfg(feature = "redis-cache")]
impl std::fmt::Debug for RedisStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisStore")
            .field("ttl_secs", &self.ttl_secs)
            .finish_non_exhaustive()
    }
}

/// How long to wait for Redis at startup before giving up.
///
/// Short on purpose: this runs before the listener binds, so every second here is a second the
/// proxy is not serving. A reachable Redis answers in milliseconds; one that needs ten seconds is
/// not one to put in the request path.
#[cfg(feature = "redis-cache")]
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[cfg(feature = "redis-cache")]
impl RedisStore {
    /// Connect to `url`.
    ///
    /// # Errors
    /// A malformed URL or an unreachable server. Surfaced at startup rather than swallowed: a
    /// cache that silently degrades to "always miss" looks like the feature is off, and an operator
    /// who configured Redis should be told it is not working.
    pub async fn connect(url: &str, ttl_secs: u64) -> Result<Self, String> {
        let client = redis::Client::open(url).map_err(|e| format!("redis url {url:?}: {e}"))?;
        // Bounded, because `ConnectionManager::new` retries a dead server indefinitely by default
        // — the proxy simply never finishes starting, with no error and no listening socket. A
        // hang is the worst of the three outcomes here: an operator can act on a refusal and can
        // live with a degraded cache, but "startup never returns" looks like a deploy that stalled
        // for unrelated reasons.
        let mgr = tokio::time::timeout(CONNECT_TIMEOUT, redis::aio::ConnectionManager::new(client))
            .await
            .map_err(|_| {
                format!(
                    "redis at {url:?} did not answer within {}s. The proxy refuses to start rather \
                 than run with a cache that silently stays per-replica — check the server is \
                 reachable, or remove redis_url to use the in-process cache.",
                    CONNECT_TIMEOUT.as_secs()
                )
            })?
            .map_err(|e| format!("redis connect {url:?}: {e}"))?;

        // `ConnectionManager::new` connects LAZILY: it returns Ok against a server that is not
        // running, the proxy logs "shared via redis", binds, and serves with a cache that never
        // works. That is worse than degrading to per-replica — it is non-functional while
        // announcing success, which is the exact failure this fail-fast exists to prevent.
        //
        // So prove the connection with a real round trip before claiming it.
        let mut probe = mgr.clone();
        let pong: Result<String, _> = tokio::time::timeout(
            CONNECT_TIMEOUT,
            redis::cmd("PING").query_async(&mut probe),
        )
        .await
        .map_err(|_| {
            format!(
                "redis at {url:?} did not answer PING within {}s. Refusing to start rather than \
                 run with a cache that reports itself as shared and never works — check the \
                 server is reachable, or remove redis_url to use the in-process cache.",
                CONNECT_TIMEOUT.as_secs()
            )
        })?;
        pong.map_err(|e| format!("redis at {url:?} rejected PING: {e}"))?;

        Ok(Self {
            client: mgr,
            ttl_secs,
        })
    }

    fn entry_key(key: &str) -> String {
        format!("fp:vc:{key}")
    }

    fn source_key(source: uuid::Uuid) -> String {
        format!("fp:vc:src:{source}")
    }
}

#[cfg(feature = "redis-cache")]
#[async_trait::async_trait]
impl CacheStore for RedisStore {
    async fn get(&self, key: &str, _now: Instant) -> Option<Entry> {
        let mut c = self.client.clone();
        let raw: Option<String> = redis::cmd("GET")
            .arg(Self::entry_key(key))
            .query_async(&mut c)
            .await
            .ok()?;
        // A malformed value is treated as a miss rather than an error: the request simply runs the
        // ladder, which is always safe. Failing the request because a cache entry did not
        // deserialize would turn an optimisation into an outage.
        serde_json::from_str(&raw?).ok()
    }

    async fn put(&self, key: &str, source: uuid::Uuid, entry: Entry, _now: Instant) {
        let Ok(body) = serde_json::to_string(&entry) else {
            return;
        };
        let mut c = self.client.clone();
        let ek = Self::entry_key(key);
        // Pipelined, not transactional: if the index write is lost the entry becomes unretractable
        // until TTL, so the TTL is the backstop that bounds that window. A MULTI would remove the
        // race but not the failure mode, and the entry TTL already caps the exposure.
        let _: Result<(), _> = redis::pipe()
            .cmd("SETEX")
            .arg(&ek)
            .arg(self.ttl_secs)
            .arg(body)
            .ignore()
            .cmd("SADD")
            .arg(Self::source_key(source))
            .arg(&ek)
            .ignore()
            .cmd("EXPIRE")
            .arg(Self::source_key(source))
            .arg(self.ttl_secs)
            .ignore()
            .query_async::<()>(&mut c)
            .await;
    }

    async fn retract(&self, source: uuid::Uuid) -> usize {
        let mut c = self.client.clone();
        let sk = Self::source_key(source);
        let keys: Vec<String> = redis::cmd("SMEMBERS")
            .arg(&sk)
            .query_async(&mut c)
            .await
            .unwrap_or_default();
        if keys.is_empty() {
            return 0;
        }
        let n: usize = redis::cmd("DEL")
            .arg(&keys)
            .query_async(&mut c)
            .await
            .unwrap_or(0);
        let _: Result<(), _> = redis::cmd("DEL").arg(&sk).query_async(&mut c).await;
        n
    }

    async fn entry_count(&self) -> usize {
        let mut c = self.client.clone();
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg("fp:vc:*")
            .query_async(&mut c)
            .await
            .unwrap_or_default();
        // ponytail: KEYS is O(n) and blocks Redis; acceptable because this is only read by
        // /metrics and tests, never on the request path. Switch to SCAN if it is ever called
        // per-request.
        keys.iter().filter(|k| !k.starts_with("fp:vc:src:")).count()
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

    #[tokio::test]
    async fn the_trait_and_the_concrete_api_agree_on_keys() {
        // The two stores must compose the key identically or a multi-replica deployment silently
        // never hits: one writes under a key the other never looks up, and the only symptom is a
        // hit rate that quietly stays at zero.
        let c = VerifiedCache::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        let t = trace(ServedFrom::Attempt, Verdict::Pass);
        assert!(c.offer(&t, &resp("via offer"), &ladder(), now));

        let k = VerifiedCache::compose_key("t", "hash-abc", &ladder());
        let via_trait = CacheStore::get(&c, &k, now)
            .await
            .expect("trait sees offer's entry");
        assert_eq!(via_trait.response.text, "via offer");
    }

    #[tokio::test]
    async fn the_trait_path_still_isolates_tenants() {
        // The leak found in review must not reappear through the new surface: `compose_key` is the
        // single place the tenant enters, and both stores go through it.
        let a = VerifiedCache::compose_key("tenant-a", "h", &ladder());
        let b = VerifiedCache::compose_key("tenant-b", "h", &ladder());
        assert_ne!(a, b, "tenant must be part of the key on the trait path too");
        assert!(a.starts_with("tenant-a|"), "{a}");
    }

    #[tokio::test]
    async fn an_entry_survives_a_serde_round_trip_without_its_local_clock() {
        // Redis carries entries between processes, where an Instant is meaningless. The proof
        // (source) and the answer must survive; `stored` is re-stamped, and expiry there is
        // Redis's TTL rather than a comparison of two machines' monotonic clocks.
        let e = Entry {
            response: resp("answer"),
            source: uuid::Uuid::now_v7(),
            original_cost_usd: 0.01,
            stored: Instant::now(),
        };
        let wire = serde_json::to_string(&e).expect("serializes");
        assert!(
            !wire.contains("stored"),
            "a local Instant must not go on the wire: {wire}"
        );

        let back: Entry = serde_json::from_str(&wire).expect("deserializes");
        assert_eq!(back.response.text, "answer");
        assert_eq!(back.source, e.source, "the PROOF must survive the crossing");
        assert!((back.original_cost_usd - 0.01).abs() < f64::EPSILON);
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
