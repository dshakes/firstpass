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

/// Named routing-mode presets: coherent (accuracy/cost/latency) constraint bundles layered
/// on top of the existing escalation knobs. A mode is a **resolver** over existing config,
/// not a new engine — it overrides only the knobs it has an opinion on; everything else
/// comes from the route/global config unchanged.
///
/// `Balanced` (the default) is a strict no-op: with no mode set anywhere, behaviour is
/// byte-identical to existing behaviour. All other modes are opt-in overlays.
///
/// Set per-request via the `x-firstpass-mode` header, per-route via `routing_mode = "..."`,
/// or globally via the `FIRSTPASS_MODE_PROFILE` env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RoutingMode {
    /// Forward without routing or gating; verdicts are asynchronous learning signals only.
    /// Maps to the existing observe passthrough path. Zero added latency, zero quality proof.
    Observe,
    /// Prefer cheapest start; no speculative prefetch.
    /// Tradeoff: lowest token spend; can serve lower quality on hard or heavy traffic.
    Cost,
    /// Today's default — bandit start, configured thresholds, configured speculation.
    /// **This preset is a strict no-op: byte-identical to existing behaviour when applied.**
    #[default]
    Balanced,
    /// One extra escalation rung allowed; serial (no speculative prefetch waste).
    /// Tradeoff: higher quality ceiling at higher cost; still bounded by gate verification.
    Quality,
    /// `speculation = 1` always: always prefetch one rung ahead to cut p95 latency.
    /// Tradeoff: pays 1 wasted speculative call when the cheap rung passes.
    Latency,
    /// Start at the top (most expensive) ladder rung; verification as insurance.
    /// Tradeoff: highest quality, highest cost per call; bandit is bypassed; savings are minimal.
    Max,
}

impl RoutingMode {
    /// Lowercase string name, for use in headers and trace fields.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::Cost => "cost",
            Self::Balanced => "balanced",
            Self::Quality => "quality",
            Self::Latency => "latency",
            Self::Max => "max",
        }
    }

    /// All named modes in display order, for discovery surfaces (capabilities, MCP, CLI).
    pub const ALL: &'static [Self] = &[
        Self::Observe,
        Self::Cost,
        Self::Balanced,
        Self::Quality,
        Self::Latency,
        Self::Max,
    ];

    /// The knob overrides this mode applies on top of the route/global escalation config.
    /// `None` fields are left unchanged — modes only override where they have an opinion.
    ///
    /// **`Balanced` returns all `None`/`false`: it is a strict no-op by invariant.**
    #[must_use]
    pub fn preset(self) -> ModePreset {
        match self {
            // Observe: handled as a passthrough redirect before EnforceCtx; preset is unused.
            Self::Observe => ModePreset {
                speculation: None,
                max_rungs_delta: None,
                start_at_top: false,
                description: "shadow mode: forward without gating; verdicts are async learning signals only",
                tradeoff: "zero added latency, zero quality proof; use to observe before enforcing",
            },
            // Cost: kill speculative spend; let the bandit start cheap.
            Self::Cost => ModePreset {
                speculation: Some(0),
                max_rungs_delta: None,
                start_at_top: false,
                description: "no speculative prefetch; bandit start on cheapest rung",
                tradeoff: "lowest token spend; can serve lower quality on hard or heavy traffic",
            },
            // Balanced: all None — strict no-op so existing behaviour is preserved.
            Self::Balanced => ModePreset {
                speculation: None,
                max_rungs_delta: None,
                start_at_top: false,
                description: "bandit start, configured thresholds and speculation (default — no behavioral change)",
                tradeoff: "today's tuned balance; add a mode only when you need a different point",
            },
            // Quality: one more rung; no speculative waste.
            Self::Quality => ModePreset {
                speculation: Some(0),
                max_rungs_delta: Some(1),
                start_at_top: false,
                description: "one extra escalation rung; serial (no speculative waste)",
                tradeoff: "higher quality ceiling at higher cost; still bounded by gate verification",
            },
            // Latency: speculation=1 always; pay 1 wasted call to cut p95.
            Self::Latency => ModePreset {
                speculation: Some(1),
                max_rungs_delta: None,
                start_at_top: false,
                description: "speculation=1: always prefetch one rung ahead to cut p95 latency",
                tradeoff: "pays 1 wasted speculative call when cheap rung passes; speculation_band is overridden",
            },
            // Max: jump to top rung immediately; verification as insurance.
            Self::Max => ModePreset {
                speculation: Some(0),
                max_rungs_delta: None,
                start_at_top: true,
                description: "start at the top (most expensive) ladder rung; verification as insurance",
                tradeoff: "highest quality, highest cost per call; bandit bypassed; savings minimal",
            },
        }
    }
}

/// Knob overrides a [`RoutingMode`] applies on top of the route/global escalation config.
/// Only non-`None` fields override; absent fields leave the config value unchanged.
///
/// `Balanced` returns all `None`/`false` — it is a strict no-op.
#[derive(Debug, Clone, Copy)]
pub struct ModePreset {
    /// Override for `escalation.speculation`. `None` = no override (config value unchanged).
    pub speculation: Option<u32>,
    /// Additive delta applied to `max_rungs_per_request`. `None` = no change.
    /// Applied as `(current + delta).max(1)`.
    pub max_rungs_delta: Option<i32>,
    /// When `true`, start at the last ladder rung (top quality), bypassing the bandit.
    pub start_at_top: bool,
    /// One-line description of what this mode targets.
    pub description: &'static str,
    /// Honest tradeoff: what you gain and what you give up.
    pub tradeoff: &'static str,
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
    /// Per-deployment price overrides, replacing the built-in defaults for the named models.
    /// List prices drift and enterprise contracts differ — savings math is only honest when the
    /// operator can pin THEIR prices. Declared as `[[price]]` sections in TOML.
    #[serde(rename = "price", default)]
    pub price_defs: Vec<PriceDef>,
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

/// A per-deployment price override for one model (`provider/model`), USD per 1M tokens.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PriceDef {
    /// The ladder id this price applies to, e.g. `"anthropic/claude-haiku-4-5"`.
    pub model: String,
    /// USD per 1M input (prompt) tokens.
    pub input_per_mtok: f64,
    /// USD per 1M output (completion) tokens.
    pub output_per_mtok: f64,
}

/// A user-defined gate (SPEC §8.1). Exactly one kind per definition:
/// - **subprocess** (`cmd`): any executable that reads the candidate as JSON on **stdin** (never
///   argv — injection-resistant) and emits `{"verdict":"pass|fail|abstain", ...}` on stdout.
/// - **judge** (`judge`): a native LLM-judge gate that grades the candidate against a rubric.
/// - **consistency** (`consistency`): a self-consistency uncertainty gate that scores the
///   candidate by agreement with k fresh samples of the same model (Wang et al. 2022).
/// - **schema** (`schema`): validates the candidate (parsed as JSON) against a JSON-Schema
///   subset (top-level `type` / `required` / per-property `type`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateDef {
    /// The id a route references this gate by (must be unique and not shadow a built-in gate id).
    pub id: String,
    /// Subprocess command: program first, then its args — e.g. `["pytest", "-q"]`. Set this **or**
    /// `judge` / `consistency` / `schema`, not both.
    #[serde(default)]
    pub cmd: Vec<String>,
    /// Hard timeout in milliseconds for a subprocess gate; it abstains (`timeout`) if the process
    /// runs longer.
    #[serde(default = "default_gate_timeout_ms")]
    pub timeout_ms: u64,
    /// LLM-judge configuration. Set this **or** `cmd` / `consistency` / `schema`, not both.
    #[serde(default)]
    pub judge: Option<JudgeDef>,
    /// Self-consistency configuration. Set this **or** `cmd` / `judge` / `schema`, not both.
    #[serde(default)]
    pub consistency: Option<ConsistencyDef>,
    /// JSON-Schema (subset) the candidate must satisfy. Set this **or** `cmd` / `judge` /
    /// `consistency`, not both.
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
    /// What an **abstain** from this gate means for serving (§7.2). `fail_open` (default): an
    /// abstaining gate never blocks serving — availability over strictness, today's behavior.
    /// `fail_closed`: an abstain blocks serving exactly like a `Fail` — strictness over
    /// availability, for gates whose silence must never be mistaken for approval.
    #[serde(default)]
    pub on_abstain: AbstainPolicy,
}

/// Per-gate abstain policy (§7.2): what happens to serving when the gate can't produce a verdict.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstainPolicy {
    /// An abstain never blocks serving (the historical behavior).
    #[default]
    FailOpen,
    /// An abstain blocks serving exactly like a `Fail`.
    FailClosed,
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

/// Configuration for a self-consistency uncertainty gate: resample the original request `k` times
/// on the same model and score the candidate by agreement with the fresh samples.
///
/// Research basis: Wang et al. 2022 (self-consistency) and Farquhar et al., *Nature* 2024
/// (semantic entropy). Answers a model reliably reproduces are far likelier correct; disagreement
/// flags hallucination. This produces a continuous confidence score the conformal threshold
/// machinery can calibrate.
///
/// **maker == checker is intentional** — unlike a judge gate, self-consistency is definitionally
/// self-referential. This is the mechanism, not a bug.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsistencyDef {
    /// The model used for resampling, as `provider/model`. May equal the candidate's model —
    /// self-consistency is self-referential by design.
    pub model: String,
    /// Number of resample calls. Must be in `[2, 8]`; defaults to `3`.
    #[serde(default = "default_consistency_k")]
    pub k: u32,
    /// Pass iff the mean agreement score ≥ this threshold, in `[0, 1]`.
    pub threshold: f64,
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

/// Default self-consistency k (resample count).
fn default_consistency_k() -> u32 {
    3
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
    /// Per-route routing-mode preset. Overrides `FIRSTPASS_MODE_PROFILE`; overridden by the
    /// per-request `x-firstpass-mode` header. Absent (default `None`) falls through to the
    /// global default, then `Balanced` — byte-identical to existing behaviour.
    #[serde(default)]
    pub routing_mode: Option<RoutingMode>,
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
    /// Route tool-calling / multimodal requests through enforce (ADR 0005). **Default `true`**:
    /// agent traffic — the target workload — is gated out of the box. A per-request fidelity
    /// guard still applies: structured content only routes when every ladder rung's provider
    /// carries it verbatim (Anthropic-dialect today); otherwise the request falls back to
    /// transparent observe passthrough rather than risking corruption. Set `false` to restore
    /// the pre-ADR-0005 behavior (structured requests always pass through un-gated).
    #[serde(default = "default_enforce_structured")]
    pub enforce_structured: bool,
    /// UCB1 start-rung bandit: learn which rung to START the ladder on per request context, to
    /// cut expected cost by skipping rungs that almost always fail for this context. `None`
    /// (default) starts every request at rung 0 — byte-identical to today. Prediction may only
    /// choose where the ladder STARTS; gating, escalation, and serving are untouched.
    #[serde(default)]
    pub bandit: Option<BanditConfig>,
    /// Speculative-deferral band: when set (and the bandit has a warm gate-pass estimate for
    /// the chosen start rung), speculative prefetch fires **only** when that estimate falls
    /// inside `[low, high]` — the marginal zone where the next rung is *probably but not
    /// certainly* needed, which is where parallel spend buys the most latency (speculative
    /// cascades). Outside the band (confident pass or confident fail-through) requests run
    /// serial and the speculative tokens are saved. `None` (default) = `speculation` applies
    /// unconditionally, byte-identical to today. No bandit / cold context ⇒ band inapplicable
    /// ⇒ configured `speculation` applies.
    #[serde(default)]
    pub speculation_band: Option<[f64; 2]>,
    /// Epsilon-greedy start-rung exploration: randomise a fraction of start-rung choices and
    /// record propensities so IPS/SNIPS off-policy estimates are valid. `None` (default) →
    /// deterministic policy — byte-identical to today. See [`ExplorationConfig`].
    #[serde(default)]
    pub exploration: Option<ExplorationConfig>,
    /// Shadow probe (ADR 0008 Phase 1): measure the k-sample gate-pass-count signal on a
    /// sampled fraction of enforce requests without changing serving or the served cost.
    /// `None` (default) = off = zero extra provider calls = byte-identical to today.
    #[serde(default)]
    pub probe: Option<ProbeConfig>,
    /// Per-query gate-pass predictor (ADR 0008 Phase 2): a learned `P(gate-pass | rung,
    /// features)` model, trained online from receipts and recorded on the receipt in shadow.
    /// `None` (default) = off = byte-identical to today (no predictor, no `predicted_pass`).
    #[serde(default)]
    pub predictor: Option<PredictorConfig>,
}

/// Per-query gate-pass predictor configuration (ADR 0008 Phase 2). The model is an online
/// logistic regression (`firstpass_core::predictor::PassPredictor`) trained from the trace
/// store. Its prediction is recorded on the receipt in **shadow** — it does not change routing
/// in this phase; whether it is good enough to act on is decided offline (`firstpass
/// predictor-eval`).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictorConfig {
    /// SGD learning rate, in `(0, 1]`. Validated by [`Config::parse`].
    #[serde(default = "default_predictor_lr")]
    pub lr: f64,
    /// L2 shrinkage on the weights (bias excluded), `>= 0`. Validated by [`Config::parse`].
    #[serde(default = "default_predictor_l2")]
    pub l2: f64,
}

fn default_predictor_lr() -> f64 {
    0.05
}

fn default_predictor_l2() -> f64 {
    1e-4
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

/// Epsilon-greedy start-rung exploration: a fraction `epsilon` of requests start at a
/// uniformly-random rung so that propensities are logged and IPS/SNIPS off-policy estimates
/// are valid (Horvitz-Thompson 1952; SNIPS: Swaminathan & Joachims 2015). The bandit still
/// observes all gate verdicts — including from exploration draws — so learning is uninterrupted.
///
/// Every request under this policy records `policy.propensity` in the trace: the probability the
/// logging policy had of choosing the start rung it actually chose. Old traces (before this field
/// was added) serialize byte-identically (propensity is omitted when `None`).
///
/// Absent (default) → deterministic policy — no exploration, no propensity recorded,
/// byte-identical to today.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExplorationConfig {
    /// Fraction of requests routed to a uniformly-random start rung instead of the greedy
    /// choice. Must be finite and in `(0, 0.5]`; validated by [`Config::parse`].
    /// A small `epsilon` (0.05–0.1) is usually enough: it keeps learning alive and makes
    /// IPS/SNIPS estimates valid at low cost in exploration waste.
    pub epsilon: f64,
}

/// Shadow-probe configuration (ADR 0008 Phase 1): measure the validated k-sample
/// gate-pass-count signal on a sampled fraction of requests WITHOUT changing serving.
///
/// **Cost caveat**: `sample_rate × k` extra model calls (at the `start_rung` model) per
/// sampled request — measurement only, nothing served changes. Default absent = off = zero
/// extra provider calls, byte-identical to today. Enable only when you want to collect the
/// regime signal for offline analysis.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeConfig {
    /// Number of parallel probe samples per request. Must be in `[2, 8]`; validated by
    /// [`Config::parse`]. Each sample is one model call at the `start_rung` model.
    pub k: u32,
    /// Fraction of requests that trigger the probe, in `[0, 1]`. `0.0` = never; `1.0` = always.
    /// Must be finite; validated by [`Config::parse`].
    pub sample_rate: f64,
}

/// Config for the UCB1 start-rung bandit (`firstpass_proxy::bandit::StartRungBandit`).
///
/// Absent (`None`) → start every request at rung 0 (byte-identical to today).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BanditConfig {
    /// Minimum total gate-verdict observations in a context bucket before the bandit's
    /// prediction kicks in. Below this, every request starts at rung 0 (cold-start safety).
    #[serde(default = "default_bandit_min_observations")]
    pub min_observations: usize,
    /// UCB1 exploration constant `c` (Auer et al. 2002). Higher values explore more; 1.0 is the
    /// theoretical default. Must be finite and `>= 0`; validated by [`Config::parse`].
    #[serde(default = "default_bandit_exploration")]
    pub exploration: f64,
    /// Selection algorithm. `"ucb1"` (default) is deterministic and auditable; `"thompson"`
    /// samples Beta posteriors — stochastic by nature, so every decision logs a non-degenerate
    /// propensity (clean IPS/SNIPS/DR off-policy estimates without the epsilon overlay) and
    /// pairs with `discount` for non-stationary traffic.
    #[serde(default)]
    pub algorithm: BanditAlgorithm,
    /// Per-observation multiplicative decay of a context's counts, in `(0, 1]`. `1.0`
    /// (default) = no forgetting. `0.99` adapts to model churn / workload drift at the cost
    /// of a slightly noisier estimate. Validated by [`Config::parse`].
    #[serde(default = "default_bandit_discount")]
    pub discount: f64,
}

/// Start-rung bandit selection algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BanditAlgorithm {
    /// Deterministic UCB1 (Auer et al. 2002).
    #[default]
    Ucb1,
    /// Thompson sampling with Beta posteriors (Chapelle & Li 2011).
    Thompson,
}

fn default_bandit_discount() -> f64 {
    1.0
}

fn default_bandit_min_observations() -> usize {
    50
}

fn default_bandit_exploration() -> f64 {
    1.0
}

const fn default_max_rungs() -> u32 {
    3
}

/// Structured (tools/images) enforce is on by default — the fidelity guard in the proxy keeps
/// it safe (verbatim-carry rungs only).
fn default_enforce_structured() -> bool {
    true
}

impl Default for Escalation {
    fn default() -> Self {
        Self {
            max_rungs_per_request: default_max_rungs(),
            session_promotion: None,
            speculation: 0,
            serve_threshold: None,
            adaptive: None,
            enforce_structured: default_enforce_structured(),
            speculation_band: None,
            bandit: None,
            exploration: None,
            probe: None,
            predictor: None,
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
            // Exactly one kind: `cmd`, `judge`, `consistency`, or `schema` — never more, never none.
            let kinds_set = [
                !def.cmd.is_empty(),
                def.judge.is_some(),
                def.consistency.is_some(),
                def.schema.is_some(),
            ]
            .iter()
            .filter(|&&b| b)
            .count();
            if kinds_set != 1 {
                return Err(Error::InvalidConfig(format!(
                    "gate {:?} must set exactly one of `cmd`, `judge`, `consistency`, or `schema`",
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
            if let Some(c) = &def.consistency {
                if !(2..=8).contains(&c.k) {
                    return Err(Error::InvalidConfig(format!(
                        "gate {:?} consistency k {} is outside [2, 8]",
                        def.id, c.k
                    )));
                }
                if !(0.0..=1.0).contains(&c.threshold) {
                    return Err(Error::InvalidConfig(format!(
                        "gate {:?} consistency threshold {} is outside [0, 1]",
                        def.id, c.threshold
                    )));
                }
            }
            if !seen.insert(def.id.as_str()) {
                return Err(Error::InvalidConfig(format!(
                    "duplicate gate id {:?}",
                    def.id
                )));
            }
        }
        for price in &config.price_defs {
            if price.model.trim().is_empty() {
                return Err(Error::InvalidConfig(
                    "price model must not be empty".to_owned(),
                ));
            }
            let ok = |v: f64| v.is_finite() && v >= 0.0;
            if !ok(price.input_per_mtok) || !ok(price.output_per_mtok) {
                return Err(Error::InvalidConfig(format!(
                    "price for {:?} must be finite and >= 0",
                    price.model
                )));
            }
        }
        if let Some(b) = &config.escalation.bandit {
            if !b.exploration.is_finite() || b.exploration < 0.0 {
                return Err(Error::InvalidConfig(format!(
                    "bandit.exploration must be finite and >= 0, got {}",
                    b.exploration
                )));
            }
            if !b.discount.is_finite() || b.discount <= 0.0 || b.discount > 1.0 {
                return Err(Error::InvalidConfig(format!(
                    "bandit.discount must be in (0, 1], got {}",
                    b.discount
                )));
            }
        }
        if let Some([lo, hi]) = config.escalation.speculation_band {
            let ok = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
            if !ok(lo) || !ok(hi) || lo > hi {
                return Err(Error::InvalidConfig(format!(
                    "escalation.speculation_band must satisfy 0 <= low <= high <= 1, got [{lo}, {hi}]"
                )));
            }
        }
        if let Some(exp) = &config.escalation.exploration
            && (!exp.epsilon.is_finite() || exp.epsilon <= 0.0 || exp.epsilon > 0.5)
        {
            return Err(Error::InvalidConfig(format!(
                "escalation.exploration.epsilon must be finite and in (0, 0.5], got {}",
                exp.epsilon
            )));
        }
        if let Some(probe) = &config.escalation.probe {
            if !(2..=8).contains(&probe.k) {
                return Err(Error::InvalidConfig(format!(
                    "escalation.probe.k must be in [2, 8], got {}",
                    probe.k
                )));
            }
            if !probe.sample_rate.is_finite() || !(0.0..=1.0).contains(&probe.sample_rate) {
                return Err(Error::InvalidConfig(format!(
                    "escalation.probe.sample_rate must be finite and in [0, 1], got {}",
                    probe.sample_rate
                )));
            }
        }
        if let Some(pred) = &config.escalation.predictor {
            if !pred.lr.is_finite() || !(0.0..=1.0).contains(&pred.lr) || pred.lr == 0.0 {
                return Err(Error::InvalidConfig(format!(
                    "escalation.predictor.lr must be finite and in (0, 1], got {}",
                    pred.lr
                )));
            }
            if !pred.l2.is_finite() || pred.l2 < 0.0 {
                return Err(Error::InvalidConfig(format!(
                    "escalation.predictor.l2 must be finite and >= 0, got {}",
                    pred.l2
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
    fn parses_consistency_gate_definition() {
        let toml = r#"
[[gate]]
id          = "uncertainty"
consistency = { model = "anthropic/claude-haiku-4-5", k = 5, threshold = 0.6 }
"#;
        let c = Config::parse(toml).unwrap();
        assert_eq!(c.gate_defs.len(), 1);
        let cons = c.gate_defs[0].consistency.as_ref().unwrap();
        assert_eq!(cons.model, "anthropic/claude-haiku-4-5");
        assert_eq!(cons.k, 5);
        assert!((cons.threshold - 0.6).abs() < 1e-9);
    }

    #[test]
    fn consistency_k_defaults_to_3() {
        let toml = "[[gate]]\nid = \"u\"\nconsistency = { model = \"anthropic/claude-haiku-4-5\", threshold = 0.7 }\n";
        let c = Config::parse(toml).unwrap();
        assert_eq!(c.gate_defs[0].consistency.as_ref().unwrap().k, 3);
    }

    #[test]
    fn rejects_exactly_one_of_violations_for_consistency() {
        // cmd + consistency — two kinds set
        let both = "[[gate]]\nid = \"g\"\ncmd = [\"x\"]\nconsistency = { model = \"a/b\", threshold = 0.5 }\n";
        assert!(matches!(Config::parse(both), Err(Error::InvalidConfig(_))));

        // consistency k out of bounds
        let bad_k =
            "[[gate]]\nid = \"g\"\nconsistency = { model = \"a/b\", k = 1, threshold = 0.5 }\n";
        assert!(matches!(Config::parse(bad_k), Err(Error::InvalidConfig(_))));

        let bad_k2 =
            "[[gate]]\nid = \"g\"\nconsistency = { model = \"a/b\", k = 9, threshold = 0.5 }\n";
        assert!(matches!(
            Config::parse(bad_k2),
            Err(Error::InvalidConfig(_))
        ));

        // consistency threshold out of bounds
        let bad_thresh =
            "[[gate]]\nid = \"g\"\nconsistency = { model = \"a/b\", threshold = 1.1 }\n";
        assert!(matches!(
            Config::parse(bad_thresh),
            Err(Error::InvalidConfig(_))
        ));
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

    // ── ExplorationConfig parse / validation ──────────────────────────────────

    #[test]
    fn parses_exploration_config() {
        let c = Config::parse("[escalation.exploration]\nepsilon = 0.1\n").unwrap();
        let exp = c
            .escalation
            .exploration
            .expect("[escalation.exploration] should parse");
        assert!((exp.epsilon - 0.1).abs() < 1e-12);
        // Absent => None (deterministic policy, byte-identical behavior).
        assert!(Config::parse("").unwrap().escalation.exploration.is_none());
    }

    #[test]
    fn exploration_epsilon_boundary_valid() {
        // Upper bound 0.5 is allowed.
        let c = Config::parse("[escalation.exploration]\nepsilon = 0.5\n").unwrap();
        assert!((c.escalation.exploration.unwrap().epsilon - 0.5).abs() < 1e-12);
    }

    #[test]
    fn exploration_epsilon_above_half_rejected() {
        let bad = "[escalation.exploration]\nepsilon = 0.51\n";
        assert!(
            matches!(Config::parse(bad), Err(Error::InvalidConfig(_))),
            "epsilon > 0.5 must be rejected"
        );
    }

    #[test]
    fn exploration_epsilon_zero_rejected() {
        let bad = "[escalation.exploration]\nepsilon = 0.0\n";
        assert!(
            matches!(Config::parse(bad), Err(Error::InvalidConfig(_))),
            "epsilon = 0 must be rejected (must be strictly positive)"
        );
    }

    #[test]
    fn exploration_epsilon_negative_rejected() {
        let bad = "[escalation.exploration]\nepsilon = -0.1\n";
        assert!(
            matches!(Config::parse(bad), Err(Error::InvalidConfig(_))),
            "epsilon < 0 must be rejected"
        );
    }

    #[test]
    fn gate_def_schema_and_on_abstain_parse() {
        let toml = r#"
[[route]]
match = {}
mode = "enforce"
ladder = ["anthropic/claude-haiku-4-5"]
gates = ["extract-shape"]

[[gate]]
id = "extract-shape"
schema = { type = "object", required = ["name"] }
on_abstain = "fail_closed"
"#;
        let config = Config::parse(toml).expect("schema gate def must parse");
        let def = &config.gate_defs[0];
        assert_eq!(def.id, "extract-shape");
        let schema = def.schema.as_ref().expect("schema captured");
        assert_eq!(schema["type"], "object");
        assert_eq!(def.on_abstain, AbstainPolicy::FailClosed);
    }

    #[test]
    fn gate_def_on_abstain_defaults_fail_open() {
        let toml = r#"
[[route]]
match = {}
mode = "enforce"
ladder = ["anthropic/claude-haiku-4-5"]

[[gate]]
id = "tests"
cmd = ["true"]
"#;
        let config = Config::parse(toml).expect("parse");
        assert_eq!(config.gate_defs[0].on_abstain, AbstainPolicy::FailOpen);
    }

    #[test]
    fn gate_def_rejects_schema_plus_cmd() {
        let toml = r#"
[[route]]
match = {}
mode = "enforce"
ladder = ["anthropic/claude-haiku-4-5"]

[[gate]]
id = "both"
cmd = ["true"]
schema = { type = "object" }
"#;
        let err = Config::parse(toml).unwrap_err();
        assert!(
            err.to_string().contains("exactly one"),
            "two kinds must be rejected: {err}"
        );
    }

    #[test]
    fn price_overrides_parse_and_validate() {
        let toml = r#"
[[route]]
match = {}
mode = "observe"
ladder = ["anthropic/claude-haiku-4-5"]

[[price]]
model = "anthropic/claude-haiku-4-5"
input_per_mtok = 0.8
output_per_mtok = 4.0
"#;
        let config = Config::parse(toml).expect("price override must parse");
        assert_eq!(config.price_defs[0].model, "anthropic/claude-haiku-4-5");
        assert!((config.price_defs[0].input_per_mtok - 0.8).abs() < 1e-12);

        let bad = toml.replace("input_per_mtok = 0.8", "input_per_mtok = -1.0");
        assert!(Config::parse(&bad).is_err(), "negative price rejected");
    }

    // ── RoutingMode / ModePreset ──────────────────────────────────────────────

    /// The Balanced preset must be a strict no-op: all overrides None/false.
    /// This is the invariant that guarantees byte-identical behaviour when no mode is set.
    #[test]
    fn balanced_preset_is_strict_noop() {
        let p = RoutingMode::Balanced.preset();
        assert!(
            p.speculation.is_none(),
            "Balanced must not override speculation"
        );
        assert!(
            p.max_rungs_delta.is_none(),
            "Balanced must not override max_rungs"
        );
        assert!(!p.start_at_top, "Balanced must not set start_at_top");
    }

    #[test]
    fn cost_preset_disables_speculation() {
        let p = RoutingMode::Cost.preset();
        assert_eq!(p.speculation, Some(0));
        assert!(p.max_rungs_delta.is_none());
        assert!(!p.start_at_top);
    }

    #[test]
    fn quality_preset_bumps_max_rungs_and_disables_speculation() {
        let p = RoutingMode::Quality.preset();
        assert_eq!(p.max_rungs_delta, Some(1));
        assert_eq!(p.speculation, Some(0));
        assert!(!p.start_at_top);
    }

    #[test]
    fn latency_preset_enables_speculation() {
        let p = RoutingMode::Latency.preset();
        assert_eq!(p.speculation, Some(1));
        assert!(p.max_rungs_delta.is_none());
        assert!(!p.start_at_top);
    }

    #[test]
    fn max_preset_sets_start_at_top_and_disables_speculation() {
        let p = RoutingMode::Max.preset();
        assert!(p.start_at_top);
        assert_eq!(p.speculation, Some(0));
        assert!(p.max_rungs_delta.is_none());
    }

    #[test]
    fn routing_mode_as_str_roundtrips() {
        for mode in RoutingMode::ALL {
            let s = mode.as_str();
            let back: RoutingMode = serde_json::from_str(&format!("\"{s}\"")).unwrap();
            assert_eq!(*mode, back, "as_str/serde roundtrip for {s}");
        }
    }

    #[test]
    fn routing_mode_defaults_to_balanced() {
        assert_eq!(RoutingMode::default(), RoutingMode::Balanced);
    }

    #[test]
    fn route_routing_mode_parses_and_defaults_to_none() {
        // Absent → None (byte-identical, no behavioral change).
        let no_mode = Config::parse(
            "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-haiku-4-5\"]\n",
        )
        .unwrap();
        assert_eq!(no_mode.routes[0].routing_mode, None);

        // Explicit cost mode.
        let with_mode = Config::parse(
            "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-haiku-4-5\"]\nrouting_mode = \"cost\"\n",
        )
        .unwrap();
        assert_eq!(with_mode.routes[0].routing_mode, Some(RoutingMode::Cost));
    }

    #[test]
    fn all_routing_modes_have_non_empty_description_and_tradeoff() {
        for mode in RoutingMode::ALL {
            let p = mode.preset();
            assert!(
                !p.description.is_empty(),
                "mode {} has empty description",
                mode.as_str()
            );
            assert!(
                !p.tradeoff.is_empty(),
                "mode {} has empty tradeoff",
                mode.as_str()
            );
        }
    }

    #[test]
    fn bandit_thompson_and_band_parse_and_validate() {
        let toml = r#"
[[route]]
match = {}
mode = "enforce"
ladder = ["anthropic/claude-haiku-4-5"]

[escalation]
speculation = 1
speculation_band = [0.3, 0.7]

[escalation.bandit]
algorithm = "thompson"
discount = 0.98
"#;
        let config = Config::parse(toml).expect("thompson + band must parse");
        let b = config.escalation.bandit.as_ref().unwrap();
        assert_eq!(b.algorithm, BanditAlgorithm::Thompson);
        assert!((b.discount - 0.98).abs() < 1e-12);
        assert_eq!(config.escalation.speculation_band, Some([0.3, 0.7]));

        // Defaults: ucb1, discount 1.0, no band.
        let plain = Config::parse(
            "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-haiku-4-5\"]\n[escalation.bandit]\n",
        )
        .unwrap();
        let b = plain.escalation.bandit.as_ref().unwrap();
        assert_eq!(b.algorithm, BanditAlgorithm::Ucb1);
        assert!((b.discount - 1.0).abs() < 1e-12);

        // Bad discount and bad band are rejected.
        assert!(Config::parse(&toml.replace("discount = 0.98", "discount = 0.0")).is_err());
        assert!(
            Config::parse(&toml.replace(
                "speculation_band = [0.3, 0.7]",
                "speculation_band = [0.9, 0.2]"
            ))
            .is_err()
        );
    }

    // ── ProbeConfig parse / validation ───────────────────────────────────────

    #[test]
    fn probe_absent_defaults_to_none() {
        // Default off = byte-identical to today.
        let c = Config::parse("").unwrap();
        assert!(c.escalation.probe.is_none());
    }

    #[test]
    fn parses_valid_probe_config() {
        let c = Config::parse("[escalation.probe]\nk = 5\nsample_rate = 0.1\n").unwrap();
        let p = c.escalation.probe.expect("[escalation.probe] should parse");
        assert_eq!(p.k, 5);
        assert!((p.sample_rate - 0.1).abs() < 1e-12);
    }

    #[test]
    fn probe_sample_rate_boundaries_accepted() {
        // 0.0 (never probe) and 1.0 (always probe) are both valid.
        let c0 = Config::parse("[escalation.probe]\nk = 2\nsample_rate = 0.0\n").unwrap();
        assert!((c0.escalation.probe.unwrap().sample_rate - 0.0).abs() < 1e-12);
        let c1 = Config::parse("[escalation.probe]\nk = 8\nsample_rate = 1.0\n").unwrap();
        assert!((c1.escalation.probe.unwrap().sample_rate - 1.0).abs() < 1e-12);
    }

    #[test]
    fn probe_rejects_k_below_2() {
        let bad = "[escalation.probe]\nk = 1\nsample_rate = 0.5\n";
        assert!(
            matches!(Config::parse(bad), Err(Error::InvalidConfig(_))),
            "k=1 must be rejected"
        );
    }

    #[test]
    fn probe_rejects_k_above_8() {
        let bad = "[escalation.probe]\nk = 9\nsample_rate = 0.5\n";
        assert!(
            matches!(Config::parse(bad), Err(Error::InvalidConfig(_))),
            "k=9 must be rejected"
        );
    }

    #[test]
    fn probe_rejects_sample_rate_above_1() {
        let bad = "[escalation.probe]\nk = 5\nsample_rate = 1.5\n";
        assert!(
            matches!(Config::parse(bad), Err(Error::InvalidConfig(_))),
            "sample_rate=1.5 must be rejected"
        );
    }

    #[test]
    fn probe_rejects_negative_sample_rate() {
        let bad = "[escalation.probe]\nk = 5\nsample_rate = -0.1\n";
        assert!(
            matches!(Config::parse(bad), Err(Error::InvalidConfig(_))),
            "negative sample_rate must be rejected"
        );
    }

    #[test]
    fn predictor_config_parses_and_validates() {
        let base = "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-haiku-4-5\"]\n[escalation.predictor]\nlr = 0.05\nl2 = 0.001\n";
        let cfg = Config::parse(base).expect("valid predictor must parse");
        let pred = cfg.escalation.predictor.unwrap();
        assert!((pred.lr - 0.05).abs() < 1e-12 && (pred.l2 - 0.001).abs() < 1e-12);
        // defaults when omitted
        let d = Config::parse("[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-haiku-4-5\"]\n[escalation.predictor]\n").unwrap();
        assert!(d.escalation.predictor.is_some());
        // bad lr / l2 rejected
        assert!(Config::parse(&base.replace("lr = 0.05", "lr = 0.0")).is_err());
        assert!(Config::parse(&base.replace("lr = 0.05", "lr = 1.5")).is_err());
        assert!(Config::parse(&base.replace("l2 = 0.001", "l2 = -1.0")).is_err());
    }
}
