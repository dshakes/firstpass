//! Routing configuration (SPEC §8.4) — declarative, agent-first.
//!
//! Config is the policy an operator (or agent) hands Firstpass: an ordered list of routes,
//! each matching a slice of traffic to a mode, a model ladder, and gates; plus budget caps
//! and escalation rules. The first matching route wins, so specific routes go first and a
//! bare `match = {}` catch-all goes last.

use crate::error::{Error, Result};
use crate::features::{Features, TaskKind};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Serving mode for a route (SPEC glossary).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Gate before serving; escalate on fail. The output is proven before the caller sees it.
    Enforce,
    /// Serve immediately, gate asynchronously — verdicts feed learning only. Zero added latency,
    /// zero risk; the default for onboarding unfamiliar traffic.
    Observe,
}

/// Top-level configuration document.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Ordered routes; first match wins.
    #[serde(rename = "route", default)]
    pub routes: Vec<Route>,
    /// Spend caps.
    #[serde(default)]
    pub budget: Budget,
    /// Escalation limits and promotion rules.
    #[serde(default)]
    pub escalation: Escalation,
}

/// One routing rule.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Route {
    /// The traffic this route claims. An empty match (`{}`) matches everything.
    #[serde(rename = "match", default)]
    pub match_: Match,
    /// Serving mode.
    pub mode: Mode,
    /// Model ladder, cheapest first, as `provider/model` strings.
    #[serde(default)]
    pub ladder: Vec<String>,
    /// Inline gates run before serving (enforce mode).
    #[serde(default)]
    pub gates: Vec<String>,
    /// Gates run asynchronously after serving; their verdicts attach to the trace as a
    /// learning signal and never block the response.
    #[serde(default)]
    pub deferred_gates: Vec<String>,
}

/// Predicate over a request's [`Features`]. Every present field is an AND-constraint; absent
/// fields are wildcards.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Match {
    /// Require a specific calling agent.
    #[serde(default)]
    pub agent: Option<String>,
    /// Require the subagent to be one of these (or this one, if a bare string).
    #[serde(default)]
    pub subagent: Option<StringOrVec>,
    /// Require a specific task kind.
    #[serde(default)]
    pub task_kind: Option<TaskKind>,
    /// Require the language to be one of these (or this one, if a bare string).
    #[serde(default)]
    pub language: Option<StringOrVec>,
}

impl Match {
    /// Whether `f` satisfies every present constraint.
    #[must_use]
    pub fn matches(&self, f: &Features) -> bool {
        if let Some(agent) = &self.agent
            && f.agent.as_deref() != Some(agent.as_str())
        {
            return false;
        }
        if let Some(subs) = &self.subagent {
            match &f.subagent {
                Some(s) if subs.contains(s) => {}
                _ => return false,
            }
        }
        if let Some(tk) = self.task_kind
            && f.task_kind != tk
        {
            return false;
        }
        if let Some(langs) = &self.language {
            match &f.language {
                Some(l) if langs.contains(l) => {}
                _ => return false,
            }
        }
        true
    }
}

/// A field that accepts either a single string or a list of strings in TOML/JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StringOrVec {
    /// A single value.
    One(String),
    /// A set of values.
    Many(Vec<String>),
}

impl StringOrVec {
    /// Whether `needle` is contained in this value.
    #[must_use]
    pub fn contains(&self, needle: &str) -> bool {
        match self {
            StringOrVec::One(s) => s == needle,
            StringOrVec::Many(v) => v.iter().any(|s| s == needle),
        }
    }
}

/// What to do when a spend cap is hit (SPEC §8.4 — never brick the customer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnExhausted {
    /// Serve the best attempt seen so far (default).
    #[default]
    ServeBestAttempt,
    /// Return a structured error instead of serving.
    Error,
}

/// Spend caps. `None` means "no cap".
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    /// Max USD per single request (across all rungs + gates).
    #[serde(default)]
    pub per_request_usd: Option<f64>,
    /// Max USD per session.
    #[serde(default)]
    pub per_session_usd: Option<f64>,
    /// Max USD per day.
    #[serde(default)]
    pub per_day_usd: Option<f64>,
    /// Behaviour when a cap is reached.
    #[serde(default)]
    pub on_exhausted: OnExhausted,
}

/// Escalation limits and session-level promotion.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Escalation {
    /// Hard ceiling on rungs climbed within one request.
    #[serde(default = "default_max_rungs")]
    pub max_rungs_per_request: u32,
    /// Optionally start higher for the rest of a session that keeps failing.
    #[serde(default)]
    pub session_promotion: Option<SessionPromotion>,
}

const fn default_max_rungs() -> u32 {
    3
}

impl Default for Escalation {
    fn default() -> Self {
        Self {
            max_rungs_per_request: default_max_rungs(),
            session_promotion: None,
        }
    }
}

/// "After N failures within a window, promote this session's starting rung."
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionPromotion {
    /// Failure count that triggers promotion.
    pub after_failures: u32,
    /// Sliding window, e.g. `"30m"`.
    pub window: String,
}

impl SessionPromotion {
    /// Parse [`SessionPromotion::window`] into a [`Duration`].
    ///
    /// # Errors
    /// Returns [`Error::BadDuration`] if the window is not `<int><unit>` with unit in `s`/`m`/`h`/`d`.
    pub fn window_duration(&self) -> Result<Duration> {
        parse_window(&self.window)
    }
}

/// Parse a compact duration like `30m`, `2h`, `90s`, `1d`.
fn parse_window(s: &str) -> Result<Duration> {
    let s = s.trim();
    let split = s
        .find(|c: char| c.is_ascii_alphabetic())
        .ok_or_else(|| Error::BadDuration(s.to_owned()))?;
    let (num, unit) = s.split_at(split);
    let n: u64 = num
        .trim()
        .parse()
        .map_err(|_| Error::BadDuration(s.to_owned()))?;
    let secs = match unit {
        "s" => n,
        "m" => n.saturating_mul(60),
        "h" => n.saturating_mul(3600),
        "d" => n.saturating_mul(86_400),
        _ => return Err(Error::BadDuration(s.to_owned())),
    };
    Ok(Duration::from_secs(secs))
}

/// A parsed `provider/model` reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    /// Provider segment, e.g. `anthropic`.
    pub provider: String,
    /// Model segment, e.g. `claude-haiku-4-5`.
    pub model: String,
}

impl ModelRef {
    /// Parse a `provider/model` string.
    ///
    /// # Errors
    /// Returns [`Error::BadModelRef`] if the input is not exactly one `provider/model` pair.
    pub fn parse(s: &str) -> Result<Self> {
        match s.split_once('/') {
            Some((p, m)) if !p.is_empty() && !m.is_empty() && !m.contains('/') => Ok(Self {
                provider: p.to_owned(),
                model: m.to_owned(),
            }),
            _ => Err(Error::BadModelRef(s.to_owned())),
        }
    }
}

impl Config {
    /// Parse a TOML configuration document.
    ///
    /// # Errors
    /// Returns [`Error::Config`] on invalid TOML or unknown fields.
    pub fn parse(toml_str: &str) -> Result<Self> {
        Ok(toml::from_str(toml_str)?)
    }

    /// The first route whose match claims `f`, or `None` if no route matches.
    #[must_use]
    pub fn route_for(&self, f: &Features) -> Option<&Route> {
        self.routes.iter().find(|r| r.match_.matches(f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::Features;

    const SPEC_CONFIG: &str = r#"
[[route]]
match = { agent = "claude-code", subagent = ["test-runner", "explore"] }
mode  = "enforce"
ladder = ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]
gates  = ["schema", "judge-diff"]

[[route]]
match = { task_kind = "code_edit" }
mode  = "enforce"
ladder = ["anthropic/claude-sonnet-5", "anthropic/claude-opus-4-8"]
gates  = ["patch-applies", "lint-diff", "judge-diff"]
deferred_gates = ["compiles", "tests"]

[[route]]
match = {}
mode  = "observe"
ladder = ["anthropic/claude-opus-4-8"]

[budget]
per_request_usd = 0.50
per_session_usd = 10.00
per_day_usd     = 250.00
on_exhausted    = "serve_best_attempt"

[escalation]
max_rungs_per_request = 3
session_promotion = { after_failures = 3, window = "30m" }
"#;

    #[test]
    fn parses_the_spec_example() {
        let c = Config::parse(SPEC_CONFIG).unwrap();
        assert_eq!(c.routes.len(), 3);
        assert_eq!(c.routes[0].mode, Mode::Enforce);
        assert_eq!(
            c.routes[0].ladder,
            ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]
        );
        assert_eq!(c.routes[1].deferred_gates, ["compiles", "tests"]);
        assert_eq!(c.budget.per_request_usd, Some(0.50));
        assert_eq!(c.budget.on_exhausted, OnExhausted::ServeBestAttempt);
        assert_eq!(c.escalation.max_rungs_per_request, 3);
        let sp = c.escalation.session_promotion.as_ref().unwrap();
        assert_eq!(sp.after_failures, 3);
        assert_eq!(sp.window_duration().unwrap(), Duration::from_secs(1800));
    }

    #[test]
    fn first_matching_route_wins() {
        let c = Config::parse(SPEC_CONFIG).unwrap();

        // claude-code / test-runner -> route 0
        let mut f = Features::new(TaskKind::Explore);
        f.agent = Some("claude-code".into());
        f.subagent = Some("test-runner".into());
        let r = c.route_for(&f).unwrap();
        assert_eq!(r.ladder[0], "anthropic/claude-haiku-4-5");

        // a code_edit with no agent -> route 1 (route 0's agent constraint fails)
        let f2 = Features::new(TaskKind::CodeEdit);
        let r2 = c.route_for(&f2).unwrap();
        assert_eq!(r2.gates, ["patch-applies", "lint-diff", "judge-diff"]);

        // anything else -> catch-all observe route
        let f3 = Features::new(TaskKind::Chat);
        let r3 = c.route_for(&f3).unwrap();
        assert_eq!(r3.mode, Mode::Observe);
    }

    #[test]
    fn shipped_example_config_parses() {
        // The repo ships `firstpass.example.toml` for users to copy. If it drifts
        // from the schema, this fails in CI rather than at a user's first run.
        let toml = include_str!("../../../firstpass.example.toml");
        let c = Config::parse(toml).expect("firstpass.example.toml must parse");
        assert_eq!(c.routes.len(), 3);
        assert_eq!(c.routes[0].mode, Mode::Enforce);
    }

    #[test]
    fn empty_match_is_wildcard() {
        let m = Match::default();
        assert!(m.matches(&Features::new(TaskKind::Other)));
    }

    #[test]
    fn subagent_list_membership() {
        let c = Config::parse(SPEC_CONFIG).unwrap();
        let route0 = &c.routes[0];
        let mut f = Features::new(TaskKind::Other);
        f.agent = Some("claude-code".into());
        f.subagent = Some("docs-writer".into()); // not in [test-runner, explore]
        assert!(!route0.match_.matches(&f));
        f.subagent = Some("explore".into());
        assert!(route0.match_.matches(&f));
    }

    #[test]
    fn model_ref_parsing() {
        let m = ModelRef::parse("anthropic/claude-haiku-4-5").unwrap();
        assert_eq!(m.provider, "anthropic");
        assert_eq!(m.model, "claude-haiku-4-5");
        assert!(ModelRef::parse("no-slash").is_err());
        assert!(ModelRef::parse("/model").is_err());
        assert!(ModelRef::parse("a/b/c").is_err());
    }

    #[test]
    fn window_parsing_units_and_errors() {
        assert_eq!(parse_window("90s").unwrap(), Duration::from_secs(90));
        assert_eq!(parse_window("30m").unwrap(), Duration::from_secs(1800));
        assert_eq!(parse_window("2h").unwrap(), Duration::from_secs(7200));
        assert_eq!(parse_window("1d").unwrap(), Duration::from_secs(86_400));
        assert!(parse_window("30x").is_err());
        assert!(parse_window("abc").is_err());
    }

    #[test]
    fn empty_config_defaults() {
        let c = Config::parse("").unwrap();
        assert!(c.routes.is_empty());
        assert_eq!(c.escalation.max_rungs_per_request, 3);
        assert_eq!(c.budget.on_exhausted, OnExhausted::ServeBestAttempt);
    }
}
