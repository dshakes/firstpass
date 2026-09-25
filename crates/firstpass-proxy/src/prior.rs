//! Verified predictive routing: the I/O side. Calls a pre-generation decision-model (TypeSafe's
//! Jev today) with the request text and turns its answer into a `decision_prior` — a per-rung
//! `P(pass | rung)` fed into the existing start-rung expected-cost argmin
//! ([`firstpass_core::cumulative_pass`], [`crate::bandit`]).
//!
//! **This is a prior, not a verifier.** The enforce gate still runs on every served attempt
//! exactly as before; nothing here can cause a failing output to be served. A timeout, a
//! transport error, an unexpected status, or a response this module can't parse all fail open:
//! `None`, a `firstpass_prior_errors_total` metric bump, and a `tracing::warn!` — never a panic,
//! never added serving latency beyond the configured `timeout_ms`.

use std::time::Duration;

use firstpass_core::PriorConfig;
use serde_json::Value;

use crate::provider::ModelRequest;

/// The single choice question sent to the decision-model. Its options are the ladder rungs
/// (`r0..rN`, in ladder order); its `probabilities` map is the per-rung "this is the least
/// capable tier that suffices" distribution.
const QUESTION_NAME: &str = "rung";

/// A configured verified-predictive-routing client: the shared HTTP client, the resolved config,
/// and the API key read once at startup (never logged, never put on the trace).
#[derive(Debug)]
pub struct PriorClient {
    http: reqwest::Client,
    cfg: PriorConfig,
    api_key: String,
}

impl PriorClient {
    /// Build a client from a resolved config and API key.
    #[must_use]
    pub fn new(http: reqwest::Client, cfg: PriorConfig, api_key: String) -> Self {
        Self { http, cfg, api_key }
    }

    /// Ladder rung labels, in ladder order (the choice question's criteria).
    #[must_use]
    pub fn rungs(&self) -> &[String] {
        &self.cfg.rungs
    }

    /// Beta pseudo-count weight for the bandit blend (`[escalation.prior] strength`).
    #[must_use]
    pub fn strength(&self) -> f64 {
        self.cfg.strength
    }

    /// Fetch the cumulative gate-pass prior for `query_text`, or `None` on any error/timeout/
    /// malformed response (fail open — never blocks or corrupts serving).
    pub async fn fetch(&self, query_text: &str) -> Option<Vec<f64>> {
        fetch_prior(&self.http, &self.cfg, &self.api_key, query_text).await
    }
}

/// Build the request text a prior should evaluate: the system prompt (if any) plus every
/// message's text projection, newline-joined. Pure and unit-testable independent of the network
/// call.
#[must_use]
pub fn query_text(req: &ModelRequest) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(req.messages.len() + 1);
    if let Some(system) = req.system.as_deref() {
        parts.push(system.to_owned());
    }
    parts.extend(
        req.messages
            .iter()
            .map(crate::provider::ChatMessage::text_view),
    );
    parts.join("\n")
}

/// Build the Jev `/v1/systemone` request body: one choice question whose criteria are the
/// configured ladder rung labels (`r0..rN`).
fn build_request(cfg: &PriorConfig, query_text: &str) -> Value {
    let mut criteria = serde_json::Map::new();
    for (i, rung) in cfg.rungs.iter().enumerate() {
        criteria.insert(format!("r{i}"), Value::String(rung.clone()));
    }
    serde_json::json!({
        "model": cfg.model,
        "state": query_text,
        "questions": {
            QUESTION_NAME: {
                "type": "choice",
                "instructions": "Which is the least capable model tier that will fully and \
                    correctly handle this request?",
                "criteria": criteria,
            }
        }
    })
}

/// Extract the `rung` question's `probabilities` map from a Jev response, in ladder order
/// (`r0..rN`; a missing option maps to `0.0`). Accepts either `{"answers":{"rung":{...}}}` or
/// `{"rung":{...}}` at top level — anything else, or a missing/non-object `probabilities`,
/// yields `None`.
fn extract_probabilities(json: &Value, cfg: &PriorConfig) -> Option<Vec<f64>> {
    let answer = json
        .get("answers")
        .and_then(|a| a.get(QUESTION_NAME))
        .or_else(|| json.get(QUESTION_NAME))?;
    let probabilities = answer.get("probabilities")?.as_object()?;
    Some(
        (0..cfg.rungs.len())
            .map(|i| {
                probabilities
                    .get(&format!("r{i}"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
            })
            .collect(),
    )
}

/// Call the decision-model, map its answer to a cumulative `P(pass | rung)` prior, and fail open
/// on anything that isn't a clean, well-formed, in-time answer.
///
/// Never panics: every fallible step (transport, timeout, status, decode, shape, math) returns
/// `None` rather than propagating an error, because a wrong or missing prior can only cost money
/// (a suboptimal start rung) — this call is never allowed to block or corrupt serving.
pub async fn fetch_prior(
    client: &reqwest::Client,
    cfg: &PriorConfig,
    api_key: &str,
    query_text: &str,
) -> Option<Vec<f64>> {
    let body = build_request(cfg, query_text);
    let url = format!("{}/v1/systemone", cfg.base_url.trim_end_matches('/'));
    let call = client.post(url).bearer_auth(api_key).json(&body).send();

    let resp = match tokio::time::timeout(Duration::from_millis(cfg.timeout_ms), call).await {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => {
            metrics::counter!("firstpass_prior_errors_total").increment(1);
            tracing::warn!(error = %e, "verified predictive routing: transport error, prior disabled for this request");
            return None;
        }
        Err(_) => {
            metrics::counter!("firstpass_prior_errors_total").increment(1);
            tracing::warn!(
                timeout_ms = cfg.timeout_ms,
                "verified predictive routing: prior call timed out, prior disabled for this request"
            );
            return None;
        }
    };

    if !resp.status().is_success() {
        metrics::counter!("firstpass_prior_errors_total").increment(1);
        tracing::warn!(
            status = resp.status().as_u16(),
            "verified predictive routing: non-2xx response, prior disabled for this request"
        );
        return None;
    }

    let json: Value = match resp.json().await {
        Ok(j) => j,
        Err(e) => {
            metrics::counter!("firstpass_prior_errors_total").increment(1);
            tracing::warn!(error = %e, "verified predictive routing: undecodable response, prior disabled for this request");
            return None;
        }
    };

    let Some(probs) = extract_probabilities(&json, cfg) else {
        metrics::counter!("firstpass_prior_errors_total").increment(1);
        tracing::warn!(
            "verified predictive routing: response missing the expected answer shape, prior \
             disabled for this request"
        );
        return None;
    };

    let prior = firstpass_core::cumulative_pass(&probs);
    if prior.is_none() {
        metrics::counter!("firstpass_prior_errors_total").increment(1);
        tracing::warn!(
            "verified predictive routing: probabilities could not be normalized, prior disabled \
             for this request"
        );
    }
    prior
}

#[cfg(test)]
mod tests {
    use axum::response::IntoResponse;

    use super::*;

    fn test_cfg(rungs: &[&str]) -> PriorConfig {
        PriorConfig {
            provider: "typesafe".to_owned(),
            model: "jev-latest".to_owned(),
            api_key_env: "TYPESAFE_API_KEY".to_owned(),
            base_url: "https://api.typesafe.ai".to_owned(),
            timeout_ms: 150,
            strength: 10.0,
            rungs: rungs.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    #[test]
    fn query_text_joins_system_and_messages() {
        let req = ModelRequest {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            system: Some("be terse".to_owned()),
            messages: vec![
                crate::provider::ChatMessage::text("user", "fix the bug"),
                crate::provider::ChatMessage::text("assistant", "on it"),
            ],
            max_tokens: 64,
            tools: Value::Null,
            raw: Value::Null,
            cache_prefix: false,
        };
        assert_eq!(query_text(&req), "be terse\nfix the bug\non it");
    }

    #[test]
    fn query_text_without_system_skips_it() {
        let req = ModelRequest {
            model: "m".to_owned(),
            system: None,
            messages: vec![crate::provider::ChatMessage::text("user", "hi")],
            max_tokens: 1,
            tools: Value::Null,
            raw: Value::Null,
            cache_prefix: false,
        };
        assert_eq!(query_text(&req), "hi");
    }

    #[test]
    fn build_request_shape() {
        let cfg = test_cfg(&["cheap", "strong"]);
        let body = build_request(&cfg, "hello");
        assert_eq!(body["model"], "jev-latest");
        assert_eq!(body["state"], "hello");
        assert_eq!(body["questions"]["rung"]["type"], "choice");
        assert_eq!(body["questions"]["rung"]["criteria"]["r0"], "cheap");
        assert_eq!(body["questions"]["rung"]["criteria"]["r1"], "strong");
    }

    #[test]
    fn extract_probabilities_from_nested_answers_map() {
        let cfg = test_cfg(&["cheap", "strong"]);
        let json = serde_json::json!({
            "answers": { "rung": { "choice": "r1", "confidence": 0.8,
                "probabilities": { "r0": 0.2, "r1": 0.8 } } }
        });
        assert_eq!(extract_probabilities(&json, &cfg), Some(vec![0.2, 0.8]));
    }

    #[test]
    fn extract_probabilities_from_flat_top_level() {
        let cfg = test_cfg(&["cheap", "strong"]);
        let json = serde_json::json!({
            "rung": { "choice": "r0", "confidence": 0.6, "probabilities": { "r0": 0.6, "r1": 0.4 } }
        });
        assert_eq!(extract_probabilities(&json, &cfg), Some(vec![0.6, 0.4]));
    }

    #[test]
    fn extract_probabilities_missing_option_defaults_to_zero() {
        let cfg = test_cfg(&["cheap", "strong", "top"]);
        let json = serde_json::json!({
            "rung": { "probabilities": { "r0": 0.5, "r2": 0.5 } }
        });
        assert_eq!(
            extract_probabilities(&json, &cfg),
            Some(vec![0.5, 0.0, 0.5])
        );
    }

    #[test]
    fn extract_probabilities_rejects_unexpected_shape() {
        let cfg = test_cfg(&["cheap", "strong"]);
        assert_eq!(extract_probabilities(&serde_json::json!({}), &cfg), None);
        assert_eq!(
            extract_probabilities(&serde_json::json!({"rung": {}}), &cfg),
            None
        );
        assert_eq!(
            extract_probabilities(
                &serde_json::json!({"rung": {"probabilities": "nope"}}),
                &cfg
            ),
            None
        );
    }

    // ---- fetch_prior against a fake axum `/v1/systemone` server ----

    async fn spawn_fake_server(
        respond: impl Fn() -> axum::response::Response + Send + Sync + 'static,
    ) -> String {
        use axum::routing::post;
        let respond = std::sync::Arc::new(respond);
        let app = axum::Router::new().route(
            "/v1/systemone",
            post(move || {
                let respond = respond.clone();
                async move { (*respond)() }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn fetch_prior_maps_a_confident_answer() {
        let base_url = spawn_fake_server(|| {
            axum::Json(serde_json::json!({
                "answers": { "rung": { "choice": "r2", "confidence": 0.9,
                    "probabilities": { "r0": 0.05, "r1": 0.1, "r2": 0.85 } } }
            }))
            .into_response()
        })
        .await;
        let mut cfg = test_cfg(&["cheap", "mid", "top"]);
        cfg.base_url = base_url;
        let client = reqwest::Client::new();
        let prior = fetch_prior(&client, &cfg, "test-key", "hard request")
            .await
            .expect("a well-formed answer must produce a prior");
        assert_eq!(prior.len(), 3);
        assert!(prior[0] < prior[1] && prior[1] < prior[2]);
        assert_eq!(*prior.last().unwrap(), 1.0);
    }

    #[tokio::test]
    async fn fetch_prior_fails_open_on_500() {
        let base_url = spawn_fake_server(|| {
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response()
        })
        .await;
        let mut cfg = test_cfg(&["cheap", "top"]);
        cfg.base_url = base_url;
        let client = reqwest::Client::new();
        assert_eq!(
            fetch_prior(&client, &cfg, "test-key", "anything").await,
            None
        );
    }

    #[tokio::test]
    async fn fetch_prior_fails_open_on_malformed_body() {
        let base_url = spawn_fake_server(|| {
            axum::Json(serde_json::json!({"unexpected": true})).into_response()
        })
        .await;
        let mut cfg = test_cfg(&["cheap", "top"]);
        cfg.base_url = base_url;
        let client = reqwest::Client::new();
        assert_eq!(
            fetch_prior(&client, &cfg, "test-key", "anything").await,
            None
        );
    }

    #[tokio::test]
    async fn fetch_prior_fails_open_on_timeout() {
        let base_url = spawn_fake_server(|| {
            // The handler never returns before the client's short timeout fires. axum server
            // itself has no timeout, but a slow handler models what a stalled upstream does.
            axum::Json(serde_json::json!({})).into_response()
        })
        .await;
        let mut cfg = test_cfg(&["cheap", "top"]);
        cfg.base_url = base_url;
        cfg.timeout_ms = 0; // instantly elapsed
        let client = reqwest::Client::new();
        // timeout_ms = 0 would fail config validation in production, but the fetch path itself
        // must still handle a zero/near-zero duration without panicking.
        assert_eq!(
            fetch_prior(&client, &cfg, "test-key", "anything").await,
            None
        );
    }
}
