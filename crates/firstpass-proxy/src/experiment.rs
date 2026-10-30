//! Live monitor for certified model selection (ADR 0014).
//!
//! A route with a `[route.challenger]` serves a randomized fraction of sessions from a second
//! ladder. This module owns the evidence: one [`Experiment`] per `(tenant, route)`, fed with the
//! cost of every served experiment request and the first task outcome of every session. Its two
//! jobs are the halt guard (the serving path asks [`ExperimentMonitor::status`] and stops using a
//! challenger that has been proven worse) and the report an operator reads before adopting one.
//!
//! ## What counts as an outcome
//!
//! Feedback with a `pass` or `fail` verdict counts, **whatever its `gate_id`**; `abstain` is
//! ignored, and only the first counted outcome per session is used. Nothing here can tell a CI
//! result from a human rating, so operators should post exactly one thing per session: the task
//! outcome the experiment is meant to compare. Posting several different signals under different
//! gate ids makes "first one wins" arbitrary.
//!
//! ## Configuration identity
//!
//! Every receipt carries a fingerprint of the experiment configuration (both ladders, margin,
//! alpha). Receipts whose fingerprint differs from the current config's are ignored and counted
//! (`ignored_stale`), so changing either ladder or the margin/alpha starts a fresh experiment
//! instead of pooling evidence about a different challenger.
//!
//! Tenants never share an experiment (ADR 0004): the key is `(tenant, route_ix)` and nothing here
//! reads across tenants.
//!
//! The same state is re-derivable from receipts. [`experiments_from_traces`] replays the stored
//! traces and deferred verdicts through the very same monitor, so a restart (or the
//! `firstpass experiment` CLI) lands on the decision the live process would have reached.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard, PoisonError};

use firstpass_core::{
    Arm, Config, Experiment, ExperimentReport, ExperimentStatus, Trace, Verdict, select,
};
use serde::Serialize;
use uuid::Uuid;

/// Upper bound on distinct sessions remembered per experiment, as 8-byte hashes (~8 MB per
/// million). Past it new sessions' outcomes are ignored rather than evicting old entries: evicting
/// would let a session count twice, which breaks the pairing argument, while ignoring only slows
/// the experiment down. A million sessions is far beyond what the test needs to decide.
const MAX_TRACKED_SESSIONS: usize = 1_000_000;

/// One experiment plus the sessions that already contributed an outcome.
#[derive(Debug)]
struct Entry {
    experiment: Experiment,
    /// Receipts seen from a previous experiment configuration and therefore not counted.
    ignored_stale: u64,
    /// Hashes of sessions whose outcome was already counted. A 64-bit collision merely drops one
    /// legitimate outcome, which costs power and never validity.
    sessions: HashSet<u64>,
    saturated_logged: bool,
}

fn session_hash(session_id: &str) -> u64 {
    // SHA-256, not `DefaultHasher`: std reserves the right to change its algorithm between
    // releases, and the dedup set rebuilt from receipts after an upgrade must agree with the one
    // the live path built before it.
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(session_id.as_bytes());
    u64::from_be_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
}

/// Gauge value for [`ExperimentStatus`].
fn status_code(s: ExperimentStatus) -> f64 {
    match s {
        ExperimentStatus::Undecided => 0.0,
        ExperimentStatus::Certified => 1.0,
        ExperimentStatus::Halted => 2.0,
    }
}

#[derive(Debug)]
struct Params {
    margin: f64,
    alpha: f64,
    fingerprint: String,
}

/// Per-`(tenant, route)` certified-selection state. Cheap to share behind an `Arc`.
#[derive(Debug, Default)]
pub struct ExperimentMonitor {
    /// `(margin, alpha)` per route index that has a challenger.
    params: HashMap<usize, Params>,
    /// Set when the trace store could not be read at startup: the history needed to know whether a
    /// challenger was already halted is missing, so no challenger is served (fail closed).
    unavailable: bool,
    /// tenant -> route -> entry. Nested so a lookup by `&str` never allocates on the request path.
    entries: Mutex<HashMap<String, HashMap<usize, Entry>>>,
}

impl ExperimentMonitor {
    /// A monitor for every route in `config` that has a challenger. Routes without one are ignored
    /// by every method, so a deployment with no experiments pays nothing.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let params = config
            .routes
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                r.challenger.as_ref().map(|c| {
                    (
                        i,
                        Params {
                            margin: c.margin,
                            alpha: c.alpha,
                            fingerprint: c.fingerprint(&r.ladder),
                        },
                    )
                })
            })
            .collect();
        Self {
            params,
            unavailable: false,
            entries: Mutex::default(),
        }
    }

    /// A monitor that refuses to serve any challenger. Used when the history could not be read.
    #[must_use]
    pub fn unavailable(config: &Config) -> Self {
        Self {
            unavailable: true,
            ..Self::from_config(config)
        }
    }

    /// The fingerprint receipts for `route_ix` must carry to count, if it has a challenger.
    #[must_use]
    pub fn fingerprint(&self, route_ix: usize) -> Option<&str> {
        self.params.get(&route_ix).map(|p| p.fingerprint.as_str())
    }

    /// Whether the serving path must NOT use the challenger for `(tenant, route_ix)`: the
    /// experiment is halted, or its history was unreadable at startup.
    #[must_use]
    pub fn challenger_blocked(&self, tenant: &str, route_ix: usize) -> bool {
        self.unavailable || self.status(tenant, route_ix) == ExperimentStatus::Halted
    }

    /// Whether any route runs an experiment.
    #[must_use]
    pub fn is_active(&self) -> bool {
        !self.params.is_empty()
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<String, HashMap<usize, Entry>>> {
        // A poisoned lock only means another thread panicked mid-update of plain counters; the
        // state is still usable, and the halt guard must not turn into a request-path panic.
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn entry<'a>(
        &self,
        map: &'a mut HashMap<String, HashMap<usize, Entry>>,
        tenant: &str,
        route_ix: usize,
    ) -> Option<&'a mut Entry> {
        let p = self.params.get(&route_ix)?;
        let (margin, alpha) = (p.margin, p.alpha);
        if !map.contains_key(tenant) {
            map.insert(tenant.to_owned(), HashMap::new());
        }
        let routes = map.get_mut(tenant)?;
        if let std::collections::hash_map::Entry::Vacant(v) = routes.entry(route_ix) {
            match Experiment::new(margin, alpha) {
                Ok(experiment) => {
                    v.insert(Entry {
                        experiment,
                        ignored_stale: 0,
                        sessions: HashSet::new(),
                        saturated_logged: false,
                    });
                }
                Err(e) => {
                    // Unreachable for a parsed config; fail open (no experiment) rather than panic.
                    tracing::warn!(route = route_ix, %e, "experiment parameters rejected");
                    return None;
                }
            }
        }
        routes.get_mut(&route_ix)
    }

    /// Where the experiment for `(tenant, route_ix)` stands. [`ExperimentStatus::Undecided`] when
    /// nothing has been observed or the route has no challenger. The serving path stops using the
    /// challenger when this is [`ExperimentStatus::Halted`].
    #[must_use]
    pub fn status(&self, tenant: &str, route_ix: usize) -> ExperimentStatus {
        self.lock()
            .get(tenant)
            .and_then(|r| r.get(&route_ix))
            .map_or(ExperimentStatus::Undecided, |e| e.experiment.status())
    }

    /// Record the cost of one served request of `arm`. Call with `trace.final_.total_cost_usd`.
    ///
    /// `fingerprint` is the receipt's; a mismatch with the current config counts the receipt in
    /// `ignored_stale` and records nothing.
    pub fn observe_cost(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        arm: Arm,
        usd: f64,
    ) {
        let current = self.fingerprint(route_ix) == Some(fingerprint);
        let mut map = self.lock();
        if let Some(e) = self.entry(&mut map, tenant, route_ix) {
            if current {
                e.experiment.observe_cost(arm, usd);
            } else {
                e.ignored_stale += 1;
            }
        }
    }

    /// Record a session's task outcome. Returns whether it was counted.
    ///
    /// Only the **first** clear outcome per session counts; later ones are ignored. The test pairs
    /// the k-th incumbent outcome with the k-th challenger outcome and treats sessions as
    /// independent draws, so several correlated outcomes from one long agent run would both
    /// overweight that session and break the independence the error bound rests on.
    ///
    /// `Pass` is 1.0, `Fail` is 0.0, `Abstain` carries no information and is ignored.
    pub fn observe_outcome(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        arm: Arm,
        session_id: &str,
        verdict: Verdict,
    ) -> bool {
        // A stale receipt is counted once, by `observe_cost`; here it is just not evidence.
        if self.fingerprint(route_ix) != Some(fingerprint) {
            return false;
        }
        let outcome = match verdict {
            Verdict::Pass => 1.0,
            Verdict::Fail => 0.0,
            Verdict::Abstain => return false,
        };
        let mut map = self.lock();
        let Some(e) = self.entry(&mut map, tenant, route_ix) else {
            return false;
        };
        let key = session_hash(session_id);
        if e.sessions.contains(&key) {
            return false;
        }
        if e.sessions.len() >= MAX_TRACKED_SESSIONS {
            if !e.saturated_logged {
                e.saturated_logged = true;
                tracing::warn!(
                    tenant,
                    route = route_ix,
                    "experiment session ledger is full; new sessions' outcomes are ignored"
                );
            }
            return false;
        }
        let before = e.experiment.status();
        if !e.experiment.observe_outcome(arm, outcome) {
            return false;
        }
        e.sessions.insert(key);
        let after = e.experiment.status();
        if before != after {
            match after {
                ExperimentStatus::Halted => tracing::error!(
                    tenant,
                    route = route_ix,
                    "EXPERIMENT HALTED: the challenger ladder is worse than the incumbent by more \
                     than the margin — it is no longer served"
                ),
                ExperimentStatus::Certified => tracing::info!(
                    tenant,
                    route = route_ix,
                    "experiment certified: the challenger is within the margin of the incumbent; \
                     adopting it is a human decision (`firstpass experiment`)"
                ),
                ExperimentStatus::Undecided => {}
            }
            Self::publish(&map, route_ix);
        }
        true
    }

    /// Export the route's status as `firstpass_experiment_status{route}` (0 undecided, 1 certified,
    /// 2 halted). The label is the route index — bounded by the config, never by traffic — and
    /// tenants are deliberately not a label (unbounded in a hosted deployment), so the value is the
    /// worst status across tenants: a halt anywhere is visible.
    fn publish(map: &HashMap<String, HashMap<usize, Entry>>, route_ix: usize) {
        let worst = map
            .values()
            .filter_map(|r| r.get(&route_ix))
            .map(|e| status_code(e.experiment.status()))
            .fold(0.0, f64::max);
        metrics::gauge!("firstpass_experiment_status", "route" => route_ix.to_string()).set(worst);
    }

    /// A snapshot of every experiment, ordered by `(tenant, route_ix)`.
    #[must_use]
    pub fn reports(&self) -> Vec<Snapshot> {
        let map = self.lock();
        let mut out: Vec<_> = map
            .iter()
            .flat_map(|(t, routes)| {
                routes.iter().map(move |(ix, e)| Snapshot {
                    tenant: t.clone(),
                    route_ix: *ix,
                    report: e.experiment.report(),
                    ignored_stale: e.ignored_stale,
                })
            })
            .collect();
        out.sort_by(|a, b| (&a.tenant, a.route_ix).cmp(&(&b.tenant, b.route_ix)));
        out
    }
}

/// One experiment as the monitor holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    /// Tenant the experiment belongs to.
    pub tenant: String,
    /// Index of the route in the config.
    pub route_ix: usize,
    /// The test's state and arm totals.
    pub report: ExperimentReport,
    /// Receipts from a previous experiment configuration that were ignored.
    pub ignored_stale: u64,
}

/// Whether a trace may be used for rung-indexed or threshold analyses (`ope`, `calibrate`).
/// Challenger-arm traces are excluded: their rungs index a different ladder and their gate scores
/// relate to correctness differently, so pooling them would silently mix two models' behaviour.
#[must_use]
pub fn is_incumbent_trace(t: &Trace) -> bool {
    t.experiment
        .as_ref()
        .is_none_or(|e| e.arm != Arm::Challenger)
}

/// Rebuild a monitor from receipts.
///
/// `traces` are in chain order; `deferred` are `(trace_id, verdict)` in the order the feedback
/// arrived (the store's insertion order). Costs are tallied per trace and outcomes are replayed in
/// arrival order with the same first-outcome-per-session rule as the live path. For sequential
/// feedback the result is exactly what the live monitor held. Under concurrent feedback the store's
/// insertion order and the monitor's lock order can differ, so the pairing — and with it
/// `decided_at_pair` and the e-values — may differ slightly from the live run. The replay is then
/// the authoritative record (it is what an auditor can re-derive), and both remain valid
/// e-processes because arm assignment is randomized independently of arrival order. Traces without an experiment record (other routes, other
/// eras) are skipped; deferred verdicts for unknown traces are ignored.
#[must_use]
pub fn monitor_from_traces(
    traces: &[Trace],
    deferred: &[(Uuid, Verdict)],
    config: &Config,
) -> ExperimentMonitor {
    let monitor = ExperimentMonitor::from_config(config);
    if !monitor.is_active() {
        return monitor;
    }
    let mut by_id: HashMap<Uuid, &Trace> = HashMap::new();
    for t in traces {
        let Some(rec) = &t.experiment else { continue };
        // Matches the feedback path, which attributes a trace with no recorded route to route 0.
        let route_ix = t.route_ix.map_or(0, |r| r as usize);
        monitor.observe_cost(
            &t.tenant_id,
            route_ix,
            &rec.fingerprint,
            rec.arm,
            t.final_.total_cost_usd,
        );
        by_id.insert(t.trace_id, t);
    }
    for (id, verdict) in deferred {
        let Some(t) = by_id.get(id) else { continue };
        let Some(rec) = &t.experiment else { continue };
        let route_ix = t.route_ix.map_or(0, |r| r as usize);
        monitor.observe_outcome(
            &t.tenant_id,
            route_ix,
            &rec.fingerprint,
            rec.arm,
            &t.session_id,
            *verdict,
        );
    }
    monitor
}

/// The monitor to serve with, rebuilt from the trace store at startup.
///
/// Only touches the store when some route has a challenger. **Fails closed on the challenger**: if
/// the store cannot be read, the proxy still starts but serves the incumbent only
/// ([`ExperimentMonitor::unavailable`]), because a challenger already proven worse must never be
/// served again just because its history could not be read.
#[must_use]
pub fn build_monitor(db_path: &str, config: &Config) -> ExperimentMonitor {
    let empty = ExperimentMonitor::from_config(config);
    if !empty.is_active() {
        return empty;
    }
    let loaded = crate::store::load_all_traces(db_path)
        .and_then(|t| crate::store::load_all_deferred(db_path).map(|d| (t, d)));
    match loaded {
        Ok((traces, deferred)) => {
            let m = monitor_from_traces(&traces, &deferred, config);
            for s in m.reports() {
                tracing::info!(
                    tenant = %s.tenant,
                    route = s.route_ix,
                    status = ?s.report.status,
                    pairs = s.report.pairs,
                    ignored_stale = s.ignored_stale,
                    "experiment restored from receipts"
                );
                ExperimentMonitor::publish(&m.lock(), s.route_ix);
            }
            m
        }
        Err(e) => {
            tracing::error!(
                %e,
                "could not read the trace store, so it is unknown whether a challenger was already \
                 halted: challenger arms are NOT served (incumbent only) until restart with a \
                 readable store"
            );
            ExperimentMonitor::unavailable(config)
        }
    }
}

/// Replay receipts into per-`(tenant, route)` snapshots. See [`monitor_from_traces`].
#[must_use]
pub fn experiments_from_traces(
    traces: &[Trace],
    deferred: &[(Uuid, Verdict)],
    config: &Config,
) -> Vec<Snapshot> {
    monitor_from_traces(traces, deferred, config).reports()
}

/// What an operator should do with an experiment's current state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Recommendation {
    /// Certified, and the challenger costs less per request: adopt it (by editing the config).
    Adopt,
    /// Certified, but the challenger is not measurably cheaper: nothing to gain.
    CertifiedNoSaving,
    /// Proven worse by more than the margin; no longer served.
    ChallengerStopped,
    /// Not enough evidence yet.
    KeepRunning,
}

/// The recommendation for a report. Cost is compared as dollars per served request on each arm;
/// missing cost data on either arm counts as "no saving" because adoption must not rest on a
/// number that was never measured.
#[must_use]
pub fn recommend(r: &ExperimentReport) -> Recommendation {
    match r.status {
        select::Status::Undecided => Recommendation::KeepRunning,
        select::Status::Halted => Recommendation::ChallengerStopped,
        select::Status::Certified => {
            match (
                r.challenger.cost_per_request(),
                r.incumbent.cost_per_request(),
            ) {
                (Some(c), Some(i)) if c < i => Recommendation::Adopt,
                _ => Recommendation::CertifiedNoSaving,
            }
        }
    }
}

/// One experiment as the CLI presents it.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentView {
    /// Tenant the experiment belongs to.
    pub tenant: String,
    /// Index of the route in the config.
    pub route_ix: usize,
    /// The test's state and arm totals.
    pub report: ExperimentReport,
    /// Receipts from a previous experiment configuration that were ignored.
    pub ignored_stale: u64,
    /// What to do about it.
    pub recommendation: Recommendation,
    /// The challenger ladder under test, so `adopt` is actionable without re-reading the config.
    pub challenger_ladder: Vec<String>,
}

/// Pair each report with its recommendation and the challenger ladder from `config`.
#[must_use]
pub fn views(reports: Vec<Snapshot>, config: &Config) -> Vec<ExperimentView> {
    reports
        .into_iter()
        .map(|s| ExperimentView {
            recommendation: recommend(&s.report),
            ignored_stale: s.ignored_stale,
            challenger_ladder: config
                .routes
                .get(s.route_ix)
                .and_then(|r| r.challenger.as_ref())
                .map(|c| c.ladder.clone())
                .unwrap_or_default(),
            tenant: s.tenant,
            route_ix: s.route_ix,
            report: s.report,
        })
        .collect()
}

fn fmt_opt(v: Option<f64>, f: impl Fn(f64) -> String) -> String {
    v.map_or_else(|| "n/a".to_owned(), f)
}

/// Human-readable report, one block per `(tenant, route)`.
#[must_use]
pub fn render(views: &[ExperimentView]) -> String {
    use std::fmt::Write as _;
    if views.is_empty() {
        return "no experiment data yet — the config has no [route.challenger], or no experiment \
                traffic has been recorded\n"
            .to_owned();
    }
    let mut out = String::new();
    for v in views {
        let r = &v.report;
        let _ = writeln!(
            out,
            "tenant {} · route {} · status: {:?}",
            v.tenant, v.route_ix, r.status
        );
        let _ = writeln!(
            out,
            "  pairs: {} (margin {}, alpha {})",
            r.pairs, r.margin, r.alpha
        );
        for (name, a) in [("incumbent ", &r.incumbent), ("challenger", &r.challenger)] {
            let _ = writeln!(
                out,
                "  {name}: success {} over {} sessions · ${} per request over {} requests",
                fmt_opt(a.success_rate(), |s| format!("{:.1}%", s * 100.0)),
                a.outcomes,
                fmt_opt(a.cost_per_request(), |c| format!("{c:.6}")),
                a.requests,
            );
        }
        let _ = writeln!(
            out,
            "  e-value to certify: {:.3} · e-value to halt: {:.3} (decide at {:.1})",
            r.e_certify,
            r.e_halt,
            1.0 / r.alpha
        );
        if v.ignored_stale > 0 {
            let _ = writeln!(
                out,
                "  {} receipts from a previous experiment configuration ignored",
                v.ignored_stale
            );
        }
        match v.recommendation {
            Recommendation::Adopt => {
                let ladder = v
                    .challenger_ladder
                    .iter()
                    .map(|m| format!("{m:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = writeln!(
                    out,
                    "  ADOPT: set this route's ladder to [{ladder}]\n\n    ladder = [{ladder}]\n"
                );
            }
            Recommendation::CertifiedNoSaving => {
                let _ = writeln!(out, "  certified, but no saving — keep incumbent");
            }
            Recommendation::ChallengerStopped => {
                let _ = writeln!(out, "  challenger stopped: worse by more than the margin");
            }
            Recommendation::KeepRunning => {
                let _ = writeln!(out, "  keep running");
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use firstpass_core::trace::ExperimentRecord;
    use firstpass_core::{
        Features, FinalOutcome, GENESIS_HASH, Mode, PolicyRef, RequestInfo, ServedFrom, TaskKind,
    };

    use super::*;

    const CFG: &str = "[[route]]\nmatch = {}\nmode = \"enforce\"\n\
        ladder = [\"anthropic/claude-sonnet-5\"]\n\
        [route.challenger]\nladder = [\"anthropic/claude-haiku-4-5\"]\npercent = 50.0\n";

    fn config() -> Config {
        Config::parse(CFG).expect("test config parses")
    }

    /// The fingerprint the current test config stamps on its receipts.
    fn fp() -> String {
        let c = config();
        c.routes[0]
            .challenger
            .as_ref()
            .expect("challenger")
            .fingerprint(&c.routes[0].ladder)
    }

    fn trace(tenant: &str, session: &str, arm: Arm, cost: f64) -> Trace {
        Trace {
            trace_id: Uuid::now_v7(),
            prev_hash: GENESIS_HASH.to_owned(),
            tenant_id: tenant.to_owned(),
            session_id: session.to_owned(),
            ts: jiff::Timestamp::now(),
            mode: Mode::Enforce,
            policy: PolicyRef {
                id: "static-ladder@v0".to_owned(),
                explore: false,
                propensity: None,
                mode_profile: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".to_owned(),
                prompt_hash: "h".to_owned(),
                features: Features::new(TaskKind::Other),
            },
            attempts: vec![],
            deferred: vec![],
            final_: FinalOutcome {
                served_rung: Some(0),
                served_from: ServedFrom::Attempt,
                total_cost_usd: cost,
                gate_cost_usd: 0.0,
                total_latency_ms: 1,
                escalations: 0,
                counterfactual_baseline_usd: cost,
                savings_usd: 0.0,
                cache_source: None,
            },
            probe: None,
            rollout: None,
            shadow: None,
            route_ix: Some(0),
            predicted_pass: None,
            decision_prior: None,
            elastic: None,
            risk: None,
            experiment: Some(ExperimentRecord {
                arm,
                percent: 50.0,
                bucket: 0,
                fingerprint: fp(),
            }),
        }
    }

    #[test]
    fn first_outcome_per_session_counts_and_abstain_is_ignored() {
        let m = ExperimentMonitor::from_config(&config());
        assert!(!m.observe_outcome("t", 0, &fp(), Arm::Incumbent, "s1", Verdict::Abstain));
        assert!(m.observe_outcome("t", 0, &fp(), Arm::Incumbent, "s1", Verdict::Pass));
        // Later outcomes from the same session are dropped, whatever they say.
        assert!(!m.observe_outcome("t", 0, &fp(), Arm::Incumbent, "s1", Verdict::Fail));
        assert!(m.observe_outcome("t", 0, &fp(), Arm::Incumbent, "s2", Verdict::Fail));
        let r = &m.reports()[0].report;
        assert_eq!(r.incumbent.outcomes, 2);
        assert!((r.incumbent.outcome_sum - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn routes_without_a_challenger_and_unknown_tenants_are_inert() {
        let m = ExperimentMonitor::from_config(&config());
        assert!(!m.observe_outcome("t", 7, &fp(), Arm::Incumbent, "s", Verdict::Pass));
        assert_eq!(m.status("nobody", 0), ExperimentStatus::Undecided);
        assert!(m.reports().is_empty());
        let off = ExperimentMonitor::default();
        assert!(!off.is_active());
        off.observe_cost("t", 0, &fp(), Arm::Incumbent, 1.0);
        assert!(off.reports().is_empty());
    }

    #[test]
    fn tenants_never_share_an_experiment() {
        let m = ExperimentMonitor::from_config(&config());
        for i in 0..400 {
            m.observe_outcome(
                "a",
                0,
                &fp(),
                Arm::Incumbent,
                &format!("i{i}"),
                Verdict::Pass,
            );
            m.observe_outcome(
                "a",
                0,
                &fp(),
                Arm::Challenger,
                &format!("c{i}"),
                Verdict::Fail,
            );
        }
        assert_eq!(m.status("a", 0), ExperimentStatus::Halted);
        assert_eq!(m.status("b", 0), ExperimentStatus::Undecided);
    }

    #[test]
    fn a_clearly_worse_challenger_halts() {
        let m = ExperimentMonitor::from_config(&config());
        let mut at = None;
        for i in 0..500 {
            m.observe_outcome(
                "t",
                0,
                &fp(),
                Arm::Incumbent,
                &format!("i{i}"),
                Verdict::Pass,
            );
            m.observe_outcome(
                "t",
                0,
                &fp(),
                Arm::Challenger,
                &format!("c{i}"),
                Verdict::Fail,
            );
            if at.is_none() && m.status("t", 0) == ExperimentStatus::Halted {
                at = Some(i);
            }
        }
        assert!(at.is_some(), "never halted");
        assert_eq!(m.status("t", 0), ExperimentStatus::Halted);
    }

    #[test]
    fn replay_matches_live_with_session_dedup_and_halt() {
        let cfg = config();
        let mut traces = Vec::new();
        let mut deferred = Vec::new();
        for i in 0..400 {
            let inc = trace("t", &format!("inc{i}"), Arm::Incumbent, 0.10);
            let chl = trace("t", &format!("chl{i}"), Arm::Challenger, 0.02);
            deferred.push((inc.trace_id, Verdict::Pass));
            deferred.push((chl.trace_id, Verdict::Fail));
            // A second trace in the same session whose outcome contradicts the first: ignored.
            let again = trace("t", &format!("chl{i}"), Arm::Challenger, 0.02);
            deferred.push((again.trace_id, Verdict::Pass));
            traces.extend([inc, chl, again]);
        }
        let reports = experiments_from_traces(&traces, &deferred, &cfg);
        assert_eq!(reports.len(), 1);
        let snap = &reports[0];
        let r = &snap.report;
        assert_eq!((snap.tenant.as_str(), snap.route_ix), ("t", 0));
        assert_eq!(snap.ignored_stale, 0);
        assert_eq!(r.status, select::Status::Halted);
        // Dedup: one counted outcome per session, and the counted one is the first (Fail = 0).
        assert_eq!(r.challenger.outcomes, 400);
        assert!(r.challenger.outcome_sum.abs() < f64::EPSILON);
        // Cost is per request, not per session: the extra trace in each session still cost money.
        assert_eq!(r.challenger.requests, 800);
        assert!((r.challenger.cost_per_request().unwrap_or(0.0) - 0.02).abs() < 1e-12);
        assert!((r.incumbent.cost_per_request().unwrap_or(0.0) - 0.10).abs() < 1e-12);
        assert_eq!(recommend(r), Recommendation::ChallengerStopped);

        // The live monitor, fed the same events, reaches the same decision.
        let live = ExperimentMonitor::from_config(&cfg);
        for t in &traces {
            let rec = t.experiment.as_ref().expect("recorded");
            live.observe_cost(
                &t.tenant_id,
                0,
                &rec.fingerprint,
                rec.arm,
                t.final_.total_cost_usd,
            );
        }
        for (id, v) in &deferred {
            let t = traces.iter().find(|t| t.trace_id == *id).expect("trace");
            let rec = t.experiment.as_ref().expect("recorded");
            live.observe_outcome("t", 0, &rec.fingerprint, rec.arm, &t.session_id, *v);
        }
        assert_eq!(live.reports()[0], *snap);
        // Replaying the same receipts under the same config is exactly reproducible.
        assert_eq!(experiments_from_traces(&traces, &deferred, &cfg), reports);
    }

    #[test]
    fn replay_ignores_unrecorded_traces_and_unknown_deferred() {
        let cfg = config();
        let mut plain = trace("t", "s", Arm::Incumbent, 0.1);
        plain.experiment = None;
        let deferred = [
            (plain.trace_id, Verdict::Pass),
            (Uuid::now_v7(), Verdict::Pass),
        ];
        assert!(experiments_from_traces(&[plain], &deferred, &cfg).is_empty());
        // A config with no challenger rebuilds nothing at all.
        let bare = Config::parse(
            "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-sonnet-5\"]\n",
        )
        .expect("parses");
        let t = trace("t", "s", Arm::Incumbent, 0.1);
        assert!(experiments_from_traces(&[t], &[], &bare).is_empty());
    }

    fn view_with(status: select::Status, inc_cost: f64, ch_cost: f64) -> ExperimentView {
        let mut exp = Experiment::new(0.05, 0.05).expect("valid");
        exp.observe_cost(Arm::Incumbent, inc_cost);
        exp.observe_cost(Arm::Challenger, ch_cost);
        let mut report = exp.report();
        report.status = status;
        ExperimentView {
            tenant: "t".to_owned(),
            route_ix: 0,
            recommendation: recommend(&report),
            ignored_stale: 0,
            report,
            challenger_ladder: vec!["anthropic/claude-haiku-4-5".to_owned()],
        }
    }

    #[test]
    fn recommendation_and_rendering_cover_every_state() {
        let adopt = view_with(select::Status::Certified, 0.10, 0.02);
        assert_eq!(adopt.recommendation, Recommendation::Adopt);
        let text = render(&[adopt]);
        assert!(
            text.contains("ADOPT: set this route's ladder to [\"anthropic/claude-haiku-4-5\"]"),
            "{text}"
        );
        assert!(text.contains("ladder = [\"anthropic/claude-haiku-4-5\"]"));

        let dear = view_with(select::Status::Certified, 0.02, 0.10);
        assert_eq!(dear.recommendation, Recommendation::CertifiedNoSaving);
        assert!(render(&[dear]).contains("certified, but no saving — keep incumbent"));

        let halted = view_with(select::Status::Halted, 0.10, 0.02);
        assert!(render(&[halted]).contains("challenger stopped: worse by more than the margin"));

        let open = view_with(select::Status::Undecided, 0.10, 0.02);
        assert_eq!(open.recommendation, Recommendation::KeepRunning);
        assert!(render(&[open]).contains("keep running"));
        assert!(render(&[]).contains("no experiment data"));
    }

    #[test]
    fn certified_without_cost_data_is_not_adopted() {
        let exp = Experiment::new(0.05, 0.05).expect("valid");
        let mut report = exp.report();
        report.status = select::Status::Certified;
        assert_eq!(recommend(&report), Recommendation::CertifiedNoSaving);
    }

    /// Changing the challenger ladder (or margin/alpha) starts a fresh experiment: receipts from
    /// the old configuration are ignored and counted, never pooled.
    #[test]
    fn changing_the_challenger_ladder_resets_the_experiment() {
        let old = config();
        let mut traces = Vec::new();
        let mut deferred = Vec::new();
        for i in 0..300 {
            let inc = trace("t", &format!("inc{i}"), Arm::Incumbent, 0.10);
            let chl = trace("t", &format!("chl{i}"), Arm::Challenger, 0.02);
            deferred.push((inc.trace_id, Verdict::Pass));
            deferred.push((chl.trace_id, Verdict::Fail));
            traces.extend([inc, chl]);
        }
        let before = experiments_from_traces(&traces, &deferred, &old);
        assert_eq!(before[0].report.status, select::Status::Halted);
        assert_eq!(before[0].ignored_stale, 0);

        let new =
            Config::parse(&CFG.replace("claude-haiku-4-5", "claude-opus-4-8")).expect("parses");
        let after = experiments_from_traces(&traces, &deferred, &new);
        assert_eq!(after[0].report.status, select::Status::Undecided);
        assert_eq!(after[0].report.pairs, 0);
        assert_eq!(after[0].report.challenger.requests, 0);
        assert_eq!(after[0].ignored_stale, 600);
        let text = render(&views(after, &new));
        assert!(
            text.contains("600 receipts from a previous experiment configuration ignored"),
            "{text}"
        );

        // margin and alpha are part of the identity too.
        let margin = Config::parse(&format!("{CFG}margin = 0.04\n")).expect("parses");
        assert_eq!(
            experiments_from_traces(&traces, &deferred, &margin)[0].ignored_stale,
            600
        );
        // percent is not: ramping the split keeps the evidence.
        let ramped = Config::parse(&CFG.replace("50.0", "30.0")).expect("parses");
        assert_eq!(
            experiments_from_traces(&traces, &deferred, &ramped)[0].report,
            before[0].report
        );
    }

    #[test]
    fn challenger_traces_are_not_incumbent_traces() {
        assert!(is_incumbent_trace(&trace("t", "s", Arm::Incumbent, 0.1)));
        assert!(!is_incumbent_trace(&trace("t", "s", Arm::Challenger, 0.1)));
        let mut plain = trace("t", "s", Arm::Challenger, 0.1);
        plain.experiment = None;
        assert!(is_incumbent_trace(&plain));
    }

    /// An unreadable store must never re-open a halted challenger: the monitor is built blocked.
    #[tokio::test]
    async fn unreadable_store_fails_closed_on_the_challenger() {
        let cfg = config();
        let ok = ExperimentMonitor::from_config(&cfg);
        assert!(!ok.challenger_blocked("t", 0));
        let blocked = ExperimentMonitor::unavailable(&cfg);
        assert!(blocked.challenger_blocked("t", 0));
        assert!(blocked.challenger_blocked("any-other-tenant", 0));

        // A path that is a directory cannot be opened as a database.
        let dir = std::env::temp_dir();
        let m = build_monitor(&dir.to_string_lossy(), &cfg);
        assert!(
            m.challenger_blocked("t", 0),
            "unreadable store served a challenger"
        );

        // A readable (fresh) store is not blocked.
        let db = dir.join(format!("firstpass-exp-build-{}.db", Uuid::now_v7()));
        let (tx, handle) = crate::store::open(&db).expect("open");
        drop(tx);
        handle.await.expect("writer");
        let m = build_monitor(&db.to_string_lossy(), &cfg);
        assert!(!m.challenger_blocked("t", 0));
        let _ = std::fs::remove_file(&db);

        // No challenger configured: nothing to block, store never touched.
        let bare = Config::parse(
            "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-sonnet-5\"]\n",
        )
        .expect("parses");
        assert!(!build_monitor(&dir.to_string_lossy(), &bare).challenger_blocked("t", 0));
    }
}
