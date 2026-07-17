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
    /// User-defined subprocess gates (SPEC §8.1), referenced by `id` from a route's `gates` /
    /// `deferred_gates`. Declared as `[[gate]]` sections in TOML.
    #[serde(rename = "gate", default)]
    pub gate_defs: Vec<GateDef>,
    /// Extra model providers a ladder can route to, beyond the built-in `anthropic` / `openai`.
    /// Any OpenAI-compatible endpoint (Groq, Together, Fireworks, DeepSeek, Mistral, xAI,
    /// OpenRouter, Ollama, vLLM, Azure, …) is one `[[provider]]` entry — no rebuild. Declared as
    /// `[[provider]]` sections; a ladder rung is then `<id>/<model>`.
    #[serde(rename = "provider", default)]
    pub providers: Vec<ProviderDef>,
}

/// The wire API a provider speaks. `anthropic` = Messages API; `openai` = Chat Completions API
/// (the de-facto standard that nearly every hosted and open-source model host implements);
/// `gemini` = Google's Generative Language API (`generateContent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dialect {
    /// Anthropic Messages API (`POST /v1/messages`).
    Anthropic,
    /// OpenAI Chat Completions API (`POST /v1/chat/completions`).
    Openai,
    /// Google Gemini Generative Language API (`POST /v1beta/models/<model>:generateContent`),
    /// authenticated with an API key in the `x-goog-api-key` header.
    Gemini,
}

/// How a provider call is credentialed — orthogonal to [`Dialect`] (ADR 0006): dialect shapes the
/// request/response body, auth scheme shapes how the request is signed/credentialed. Default
/// (`api_key_env` / BYOK header) is unchanged for every existing `[[provider]]` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthScheme {
    /// API key in a header (`x-api-key`, `Authorization: Bearer`, or `x-goog-api-key`) — today's
    /// behavior for `anthropic` / `openai` / `gemini`.
    #[default]
    ApiKey,
    /// AWS SigV4 request signing (Bedrock) — credentials from `AWS_ACCESS_KEY_ID` /
    /// `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`, region-scoped.
    AwsSigv4,
    /// GCP OAuth2 bearer token (Vertex AI) — minted and cached by `gcp_auth` from
    /// `GOOGLE_APPLICATION_CREDENTIALS` or the ambient environment.
    GcpOauth,
}

/// A model provider a ladder can route to. Declared as `[[provider]]` in TOML; referenced from a
/// ladder as `<id>/<model>` (e.g. `groq/llama-3.3-70b-versatile`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderDef {
    /// Ladder prefix for this provider, e.g. `"groq"`. Overrides a built-in of the same id.
    pub id: String,
    /// Which wire API it speaks.
    pub dialect: Dialect,
    /// Base URL, e.g. `"https://api.groq.com/openai"` or `"http://localhost:11434"` for Ollama.
    /// Unused for `aws_sigv4/gcp_oauth` auth (those construct the URL from `region`/`project`).
    #[serde(default)]
    pub base_url: String,
    /// Env var the API key is read from at call time, e.g. `"GROQ_API_KEY"`. Omit for a keyless
    /// endpoint (local Ollama / vLLM). Per-request BYOK headers still apply to the built-in
    /// `anthropic` / `openai` providers.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// How this provider's requests are credentialed. `api_key` (default) is byte-identical to
    /// today; `aws_sigv4` / `gcp_oauth` are the Bedrock/Vertex auth schemes (ADR 0006).
    #[serde(default)]
    pub auth: AuthScheme,
    /// Cloud region, e.g. `"us-east-1"` — required for `aws_sigv4` (Bedrock) and `gcp_oauth`
    /// (Vertex).
    #[serde(default)]
    pub region: Option<String>,
    /// GCP project id — required for `gcp_oauth` (Vertex).
    #[serde(default)]
    pub project: Option<String>,
}

/// A user-defined gate (SPEC §8.1). Exactly one kind per definition:
/// - **subprocess** (`cmd`): any executable that reads the candidate as JSON on **stdin** (never
///   argv — injection-resistant) and emits `{"verdict":"pass|fail|abstain", ...}` on stdout.
/// - **judge** (`judge`): a native LLM-judge gate that grades the candidate against a rubric.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateDef {
    /// The id a route references this gate by (must be unique and not shadow a built-in gate id).
    pub id: String,
    /// Subprocess command: program first, then its args — e.g. `["pytest", "-q"]`. Set this **or**
    /// `judge`, not both.
    #[serde(default)]
    pub cmd: Vec<String>,
    /// Hard timeout in milliseconds for a subprocess gate; it abstains (`timeout`) if the process
    /// runs longer.
    #[serde(default = "default_gate_timeout_ms")]
    pub timeout_ms: u64,
    /// LLM-judge configuration. Set this **or** `cmd`, not both.
    #[serde(default)]
    pub judge: Option<JudgeDef>,
}

/// Configuration for a native LLM-judge gate (SPEC §8.3): a separate model grades the candidate
/// against a rubric. The runner enforces maker ≠ checker (a model never grades its own output) and
/// treats the candidate as data, not instructions.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeDef {
    /// The judge model, as `provider/model` (must differ from the candidate's model at runtime).
    pub model: String,
    /// Pass iff the judge's score ≥ this threshold, in `[0, 1]`.
    #[serde(default = "default_judge_threshold")]
    pub threshold: f64,
    /// What "good" means for this route — handed to the judge as the grading rubric.
    #[serde(default)]
    pub rubric: Option<String>,
}

/// Default subprocess-gate timeout: 30s. Long enough for a test suite, short enough to bound the
/// enforce-path tail.
fn default_gate_timeout_ms() -> u64 {
    30_000
}

/// Default judge pass threshold.
fn default_judge_threshold() -> f64 {
    0.7
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
    /// Prefetch depth: fire this many rungs ahead concurrently to trade wasted spend for latency.
    /// `0` (default) is serial — one call at a time. The served result is identical either way.
    #[serde(default)]
    pub speculation: u32,
    /// Calibrated conformal serve threshold: a rung is served iff its aggregate gate score is
    /// `>=` this value (SPEC §10.1). `None` (default) keeps the original rule — serve iff the
    /// aggregate gate verdict is `Pass` — byte-identical to today.
    #[serde(default)]
    pub serve_threshold: Option<f64>,
    /// Online/adaptive conformal (Gibbs-Candès ACI): when set, the serve threshold is tracked
    /// **live** from deferred feedback instead of held fixed, so served-failure stays at target under
    /// distribution shift. `None` (default) uses the fixed `serve_threshold` above — byte-identical.
    #[serde(default)]
    pub adaptive: Option<AdaptiveConfig>,
    /// Opt-in: route tool-calling / multimodal requests through enforce instead of falling back to
    /// observe (ADR 0005). `false` (default) is byte-identical to today — such requests pass through
    /// un-gated. Turn on **only after** verifying enforce faithfully round-trips your tool workload;
    /// content fidelity is preserved either way, but escalating a live tool turn is operator-gated.
    #[serde(default)]
    pub enforce_structured: bool,
}

/// Config for online/adaptive conformal serving ([`crate::conformal::AdaptiveConformal`]).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveConfig {
    /// Target served-failure rate the online loop holds to.
    pub alpha: f64,
    /// Step size (default 0.02) — larger tracks shift faster but noisier.
    #[serde(default = "default_adaptive_gamma")]
    pub gamma: f64,
}

fn default_adaptive_gamma() -> f64 {
    0.02
}

const fn default_max_rungs() -> u32 {
    3
}

impl Default for Escalation {
    fn default() -> Self {
        Self {
            max_rungs_per_request: default_max_rungs(),
            session_promotion: None,
            speculation: 0,
            serve_threshold: None,
            adaptive: None,
            enforce_structured: false,
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
    /// Returns [`Error::Config`] on invalid TOML or unknown fields, or [`Error::InvalidConfig`] on
    /// an invalid gate definition (not exactly one of `cmd`/`judge`, a judge threshold outside
    /// `[0,1]`, or a duplicate/blank `id`).
    pub fn parse(toml_str: &str) -> Result<Self> {
        let config: Self = toml::from_str(toml_str)?;
        let mut seen = std::collections::HashSet::new();
        for def in &config.gate_defs {
            if def.id.trim().is_empty() {
                return Err(Error::InvalidConfig("gate id must not be empty".to_owned()));
            }
            // Exactly one kind: a subprocess `cmd` or a `judge`, never both, never neither.
            if def.cmd.is_empty() == def.judge.is_none() {
                return Err(Error::InvalidConfig(format!(
                    "gate {:?} must set exactly one of `cmd` or `judge`",
                    def.id
                )));
            }
            if let Some(judge) = &def.judge
                && !(0.0..=1.0).contains(&judge.threshold)
            {
                return Err(Error::InvalidConfig(format!(
                    "gate {:?} judge threshold {} is outside [0, 1]",
                    def.id, judge.threshold
                )));
            }
            if !seen.insert(def.id.as_str()) {
                return Err(Error::InvalidConfig(format!(
                    "duplicate gate id {:?}",
                    def.id
                )));
            }
        }
        Ok(config)
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
    fn parses_gate_definitions() {
        let toml = r#"
[[route]]
match = {}
mode  = "enforce"
ladder = ["anthropic/claude-haiku-4-5"]
gates  = ["my-tests"]

[[gate]]
id  = "my-tests"
cmd = ["pytest", "-q"]

[[gate]]
id         = "judge"
cmd        = ["bash", "-c", "./judge.sh"]
timeout_ms = 60000
"#;
        let c = Config::parse(toml).unwrap();
        assert_eq!(c.gate_defs.len(), 2);
        assert_eq!(c.gate_defs[0].id, "my-tests");
        assert_eq!(c.gate_defs[0].cmd, ["pytest", "-q"]);
        assert_eq!(c.gate_defs[0].timeout_ms, 30_000, "default timeout applies");
        assert_eq!(c.gate_defs[1].timeout_ms, 60_000);
    }

    #[test]
    fn rejects_invalid_gate_definitions() {
        let empty_cmd = "[[gate]]\nid = \"g\"\ncmd = []\n";
        assert!(matches!(
            Config::parse(empty_cmd),
            Err(Error::InvalidConfig(_))
        ));

        let dup = "[[gate]]\nid = \"g\"\ncmd = [\"a\"]\n[[gate]]\nid = \"g\"\ncmd = [\"b\"]\n";
        assert!(matches!(Config::parse(dup), Err(Error::InvalidConfig(_))));
    }

    #[test]
    fn parses_adaptive_conformal_config() {
        let c = Config::parse("[escalation.adaptive]\nalpha = 0.1\n").unwrap();
        let a = c
            .escalation
            .adaptive
            .expect("[escalation.adaptive] should parse");
        assert!((a.alpha - 0.1).abs() < 1e-9);
        assert!((a.gamma - 0.02).abs() < 1e-9, "gamma defaults to 0.02");
        // Absent => None (fixed-threshold serving, default byte-identical behavior).
        assert!(Config::parse("").unwrap().escalation.adaptive.is_none());
    }

    #[test]
    fn parses_provider_entries_and_ladders_can_reference_them() {
        let c = Config::parse(
            r#"
[[provider]]
id = "groq"
dialect = "openai"
base_url = "https://api.groq.com/openai"
api_key_env = "GROQ_API_KEY"

[[provider]]
id = "ollama"
dialect = "openai"
base_url = "http://localhost:11434"

[[route]]
match = {}
mode = "enforce"
ladder = ["groq/llama-3.3-70b-versatile", "anthropic/claude-sonnet-5"]
"#,
        )
        .unwrap();
        assert_eq!(c.providers.len(), 2);
        let groq = &c.providers[0];
        assert_eq!(groq.id, "groq");
        assert_eq!(groq.dialect, Dialect::Openai);
        assert_eq!(groq.base_url, "https://api.groq.com/openai");
        assert_eq!(groq.api_key_env.as_deref(), Some("GROQ_API_KEY"));
        // A keyless local endpoint (Ollama) parses with no api_key_env.
        assert_eq!(c.providers[1].id, "ollama");
        assert!(c.providers[1].api_key_env.is_none());
        // Absent => no extra providers (built-ins only).
        assert!(Config::parse("").unwrap().providers.is_empty());
    }

    #[test]
    fn parses_aws_sigv4_provider_and_defaults_auth_to_api_key() {
        let c = Config::parse(
            r#"
[[provider]]
id = "bedrock"
dialect = "anthropic"
auth = "aws_sigv4"
region = "us-east-1"
"#,
        )
        .unwrap();
        let bedrock = &c.providers[0];
        assert_eq!(bedrock.auth, AuthScheme::AwsSigv4);
        assert_eq!(bedrock.region.as_deref(), Some("us-east-1"));
        assert!(bedrock.project.is_none());

        // Omitting `auth` on an existing provider entry defaults to ApiKey (I1: no behavior
        // change for today's `[[provider]]` entries).
        let c2 = Config::parse(
            r#"
[[provider]]
id = "groq"
dialect = "openai"
base_url = "https://api.groq.com/openai"
"#,
        )
        .unwrap();
        assert_eq!(c2.providers[0].auth, AuthScheme::ApiKey);
    }

    #[test]
    fn all_documented_provider_shapes_parse() {
        // The exact `[[provider]]` shapes shown in the README / usage page / firstpass.example.toml.
        // This is the guard that the documented provider instructions stay valid — one entry per
        // dialect/auth combination we tell users to write.
        let c = Config::parse(
            r#"
[[provider]]
id = "groq"
dialect = "openai"
base_url = "https://api.groq.com/openai"
api_key_env = "GROQ_API_KEY"

[[provider]]
id = "ollama"
dialect = "openai"
base_url = "http://localhost:11434"

[[provider]]
id = "gemini"
dialect = "gemini"
base_url = "https://generativelanguage.googleapis.com"
api_key_env = "GEMINI_API_KEY"

[[provider]]
id = "bedrock"
dialect = "anthropic"
auth = "aws_sigv4"
region = "us-east-1"

[[provider]]
id = "vertex"
dialect = "anthropic"
auth = "gcp_oauth"
region = "us-east5"
project = "my-gcp-project"
"#,
        )
        .expect("every documented provider shape must parse");
        assert_eq!(c.providers.len(), 5);

        let gemini = c.providers.iter().find(|p| p.id == "gemini").unwrap();
        assert_eq!(gemini.dialect, Dialect::Gemini);
        assert_eq!(gemini.auth, AuthScheme::ApiKey);

        let vertex = c.providers.iter().find(|p| p.id == "vertex").unwrap();
        assert_eq!(vertex.auth, AuthScheme::GcpOauth);
        assert_eq!(vertex.region.as_deref(), Some("us-east5"));
        assert_eq!(vertex.project.as_deref(), Some("my-gcp-project"));
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
