//! `firstpass kiosk` — one command, no questions, no prior state, always ends in a receipt.
//!
//! `onboard` asks three questions and wires your shell; `demo` proves the loop with no keys at
//! all. The kiosk is the single-press button between them: look at the environment, pick the
//! provider whose key is already there, write a config, stand the real proxy up, put one real
//! request through it, and print the audit receipt. No prompts, so it behaves identically for a
//! human at a terminal and for an agent with no tty — which is the point, since an agent that
//! blocks on a question has failed to onboard.
//!
//! **It never dead-ends.** With no provider key, it does not print instructions and quit; it falls
//! through to [`crate::demo`], whose local upstream needs no key, so pressing the button always
//! produces a working, inspectable receipt. "Nothing happened, here is what to install" is the
//! failure mode a kiosk exists to avoid.
//!
//! Config generation is [`crate::onboard::render_config`], not a second template. This repo has
//! already paid for two implementations of the same config drifting apart.

use crate::onboard::{LadderChoice, Provider, Shape, render_config};
use firstpass_core::{GENESIS_HASH, Mode, Trace, verify_chain};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Anything that can go wrong. The kiosk reports plainly rather than panicking — it is the same
/// binary operators run in production.
type Fail = Box<dyn std::error::Error>;

/// The provider the kiosk will open its ladder on, and the env var its key came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pick {
    /// Which provider.
    pub provider: Provider,
    /// The environment variable the key was found in.
    pub key_env: &'static str,
}

/// Every provider the kiosk can start from, in priority order, with the key it needs.
const CANDIDATES: [(Provider, &str); 3] = [
    (Provider::Anthropic, "ANTHROPIC_API_KEY"),
    (Provider::OpenAi, "OPENAI_API_KEY"),
    (Provider::Google, "GEMINI_API_KEY"),
];

/// Pick the first provider whose key is present and non-blank. `None` means no key at all, which
/// is a supported state, not an error — the caller falls through to the keyless demo.
///
/// A blank value counts as absent: `export ANTHROPIC_API_KEY=` in a stale shell profile is common,
/// and treating it as present would route the kiosk into a guaranteed 401.
pub fn pick_provider(env: impl Fn(&str) -> Option<String>) -> Option<Pick> {
    CANDIDATES.iter().find_map(|&(provider, key_env)| {
        env(key_env)
            .filter(|v| !v.trim().is_empty())
            .map(|_| Pick { provider, key_env })
    })
}

/// The config the kiosk runs: the picked provider's ladder, JSON-shaped so a real gate does real
/// work, in enforce mode so the receipt shows an actual routing decision rather than a passthrough.
#[must_use]
pub fn kiosk_config(pick: Pick) -> String {
    render_config(&LadderChoice {
        provider: pick.provider,
        shape: Shape::Json,
        mode: Mode::Enforce,
    })
}

/// Print the audit receipt for one trace: every rung tried, what the gate said, what it cost, and
/// what it saved against always calling the top of the ladder.
pub fn print_receipt(trace: &Trace) {
    println!("\x1b[1m── audit receipt ──────────────────────────────────\x1b[0m");
    for a in &trace.attempts {
        let verdict = match a.verdict {
            firstpass_core::Verdict::Pass => "\x1b[32mPASS\x1b[0m",
            firstpass_core::Verdict::Fail => "\x1b[31mFAIL\x1b[0m",
            firstpass_core::Verdict::Abstain => "\x1b[33mABSTAIN\x1b[0m",
        };
        println!(
            "  rung {} · {:<28} · {verdict} · ${:.4}",
            a.rung, a.model, a.cost_usd
        );
    }
    let f = &trace.final_;
    println!("  ─────────────────────────────────────────────────");
    println!("  total     ${:.4}", f.total_cost_usd);
    println!(
        "  baseline  ${:.4}   (always top-tier)",
        f.counterfactual_baseline_usd
    );
    let pct = if f.counterfactual_baseline_usd > 0.0 {
        f.savings_usd / f.counterfactual_baseline_usd * 100.0
    } else {
        0.0
    };
    println!(
        "  \x1b[32mSAVED     ${:.4}   ({pct:.0}% cheaper at proven quality)\x1b[0m",
        f.savings_usd
    );
    println!("  trace_id  {}", trace.trace_id);
    println!(
        "  chain     {}\n",
        if verify_chain(std::slice::from_ref(trace), &trace.prev_hash).is_ok() {
            "verified ✓"
        } else {
            "BROKEN"
        }
    );
}

/// Run the kiosk end to end.
///
/// # Errors
/// A real provider call failed, the proxy could not bind, or no receipt was recorded. A *missing
/// key* is not an error — it routes to the keyless demo.
pub async fn run() -> Result<(), Fail> {
    let Some(pick) = pick_provider(|k| std::env::var(k).ok()) else {
        println!("\n\x1b[1mFirstpass kiosk\x1b[0m");
        println!(
            "no provider key in the environment (looked for {}).",
            CANDIDATES
                .iter()
                .map(|(_, k)| *k)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("running the keyless demo instead — same proxy, same receipt, local upstream.\n");
        return crate::demo::run().await;
    };

    // The proxy runs in-process here, so its logs are the operator's only window into why a call
    // failed. Without this the kiosk swallows the provider's actual error message.
    crate::run::init_tracing();

    println!("\n\x1b[1mFirstpass kiosk\x1b[0m");
    println!(
        "found {} — routing one real request through a real enforce ladder\n",
        pick.key_env
    );

    // Write the config next to the operator, but never over the top of one they already have:
    // clobbering a tuned routing policy to run a demo would be an unforgivable trade.
    let path = PathBuf::from("firstpass.toml");
    let toml = kiosk_config(pick);
    let config_note = if path.exists() {
        "firstpass.toml already exists — using it unchanged".to_owned()
    } else {
        std::fs::write(&path, &toml)?;
        format!("wrote {}", path.display())
    };
    println!("config   : {config_note}");
    let toml = std::fs::read_to_string(&path)?;

    let db = std::env::temp_dir().join(format!("firstpass-kiosk-{}.db", uuid::Uuid::now_v7()));
    let (proxy, _writer) = spawn(&toml, &db).await?;
    println!("proxy    : {proxy}\n");

    // Exactly the request a caller would make, in the Anthropic wire format the agent tools speak.
    let key = std::env::var(pick.key_env).unwrap_or_default();
    let client = reqwest::Client::new();
    let served: Value = client
        .post(format!("{proxy}/v1/messages"))
        .header("x-api-key", &key)
        .header("authorization", format!("Bearer {key}"))
        .json(&json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 256,
            "messages": [{
                "role": "user",
                "content": "Reply with ONLY a JSON object with keys \"id\" (string) and \"total\" (number). No prose, no code fence.",
            }],
        }))
        .send()
        .await?
        .json()
        .await?;

    if served.get("error").is_some() {
        // The proxy answers callers with a deliberately generic `engine_error` — an upstream body
        // can echo request content, so it is logged rather than returned. Here the proxy is
        // in-process and the operator IS the caller, so the log line above this one already
        // carries the provider's own words (`provider rejected the call ... body=...`). Point at
        // it: "no rung could serve a valid response" is true and useless when the real answer is
        // "that key was rejected".
        return Err(format!(
            "the {} call failed — see the `provider rejected the call` line above for the \
             provider's own message (a rejected key is the usual cause). `firstpass doctor` \
             checks the key and config.",
            pick.key_env
        )
        .into());
    }
    println!("served   : {}", served["content"][0]["text"]);
    println!("model    : {}\n", served["model"]);

    let trace = wait_for_trace(&db).await.ok_or("no receipt was recorded")?;
    print_receipt(&trace);

    let all = crate::store::load_all_traces(&db)?;
    println!(
        "chain    : {}",
        if verify_chain(&all, GENESIS_HASH).is_ok() {
            "\x1b[32mverified ✓\x1b[0m — re-derivable by anyone holding the log"
        } else {
            "BROKEN"
        }
    );
    println!("panel    : {proxy}/panel");
    println!("\nkeep it: FIRSTPASS_CONFIG=./firstpass.toml firstpass up");
    println!("route your agent: export ANTHROPIC_BASE_URL={proxy}");
    println!("undo everything : unset ANTHROPIC_BASE_URL\n");

    let _ = std::fs::remove_file(&db);
    Ok(())
}

/// The agent-native form of the kiosk: no prose, no colours, no prompts — one structured document
/// describing what was found, what config would run, whether the pipeline actually works, and the
/// exact environment variables to route and to leave.
///
/// The self-test is deliberately **keyless**: it exercises the whole path (route → gate →
/// escalate → serve → receipt → chain) against a local upstream, so an agent can confirm the
/// install is sound without spending the operator's money on a call it did not ask for. Whether a
/// provider key exists is reported as a fact, not proven by billing it.
///
/// Every value is machine-readable. An agent that has to regex a paragraph has not been onboarded.
pub async fn handshake() -> Value {
    let mut doc = handshake_static();
    doc["selftest"] = selftest().await;
    doc
}

/// The part of the handshake that needs no runtime: what keys exist, what config would run, and
/// how to route and unroute. Split out so the MCP server — whose dispatch is synchronous — can
/// answer the same question without blocking a runtime thread to do it.
#[must_use]
pub fn handshake_static() -> Value {
    let pick = pick_provider(|k| std::env::var(k).ok());
    let detected: Vec<&str> = CANDIDATES
        .iter()
        .filter(|(_, k)| std::env::var(k).is_ok_and(|v| !v.trim().is_empty()))
        .map(|(_, k)| *k)
        .collect();
    let selftest = json!({
        "ok": null,
        "reason": "not run here — `firstpass handshake` runs the keyless self-test",
    });
    json!({
        "product": "firstpass",
        "version": env!("CARGO_PKG_VERSION"),
        "provider": {
            "picked": pick.map(|p| p.provider.id()),
            "key_env": pick.map(|p| p.key_env),
            "keys_present": detected,
        },
        "config": {
            "path": "firstpass.toml",
            "exists": std::path::Path::new("firstpass.toml").exists(),
            // The config is RETURNED, not written: a handshake reports, it does not mutate the
            // working tree behind the agent's back. `firstpass kiosk` is the command that writes.
            "toml": pick.map(kiosk_config),
        },
        "selftest": selftest,
        "route_your_agent": { "env": { "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080" } },
        "offboard": { "unset": ["ANTHROPIC_BASE_URL"] },
        "next": [
            "firstpass kiosk    — write the config and put one real request through it",
            "firstpass up       — run the proxy",
            "firstpass doctor   — validate config, key, and gate binaries",
        ],
    })
}

/// Exercise the whole path — route, gate, escalate, serve, receipt, chain — against a local
/// upstream. Keyless on purpose: an agent must be able to confirm the install is sound without
/// spending the operator's money on a call nobody asked for.
async fn selftest() -> Value {
    if let Ok(upstream) = crate::demo::spawn_upstream().await {
        let db =
            std::env::temp_dir().join(format!("firstpass-handshake-{}.db", uuid::Uuid::now_v7()));
        if let Ok((proxy, _w)) = crate::demo::spawn_proxy(&upstream, &db).await {
            let served = reqwest::Client::new()
                .post(format!("{proxy}/v1/messages"))
                .header("x-api-key", "keyless-selftest")
                .json(&json!({
                    "model": "claude-haiku-4-5",
                    "max_tokens": 64,
                    "messages": [{ "role": "user", "content": "selftest" }],
                }))
                .send()
                .await;
            let ok_http = served.is_ok_and(|r| r.status().is_success());
            let trace = wait_for_trace(&db).await;
            let chain_ok = trace
                .as_ref()
                .is_some_and(|t| verify_chain(std::slice::from_ref(t), &t.prev_hash).is_ok());
            let out = json!({
                "ok": ok_http && trace.is_some() && chain_ok,
                "keyless": true,
                "served": ok_http,
                "receipt": trace.is_some(),
                "escalated": trace.as_ref().is_some_and(|t| t.attempts.len() > 1),
                "chain_verified": chain_ok,
            });
            let _ = std::fs::remove_file(&db);
            return out;
        }
        let _ = std::fs::remove_file(&db);
    }
    json!({ "ok": false, "keyless": true, "reason": "could not stand up the local self-test" })
}

/// Stand up the real proxy on an ephemeral port with `toml` as its routing config.
async fn spawn(toml: &str, db: &Path) -> Result<(String, tokio::task::JoinHandle<()>), Fail> {
    let toml = toml.to_owned();
    let db_str = db.to_string_lossy().into_owned();
    let config = crate::ProxyConfig::from_lookup(move |k| match k {
        "FIRSTPASS_CONFIG_TOML" => Some(toml.clone()),
        "FIRSTPASS_MODE" => Some("enforce".to_owned()),
        "FIRSTPASS_DB" => Some(db_str.clone()),
        _ => None,
    })?;
    let provider_defs = config
        .routing
        .as_ref()
        .map(|r| r.providers.as_slice())
        .unwrap_or_default();
    let providers = crate::provider::ProviderRegistry::from_config(
        provider_defs,
        &config.upstream_anthropic,
        &config.upstream_openai,
    );
    let (traces, writer) = crate::store::open(db)?;
    let state = crate::proxy::AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers,
        gate_health: Arc::new(crate::gate::GateHealthRegistry::new()),
        shadow_ledger: Arc::new(crate::shadow::ShadowLedger::new()),
        guardrails: Arc::new(crate::guard::GuardrailRegistry::new()),
        traces,
        adaptive: None,
        bandit: None,
        predictor: None,
        tenant_rate_limiter: None,
        spill: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let app = crate::app(state)?;
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok((format!("http://{addr}"), writer))
}

/// Traces are written off the hot path, so the receipt lands a beat after the response.
async fn wait_for_trace(db: &Path) -> Option<Trace> {
    for _ in 0..50 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if let Ok(traces) = crate::store::load_all_traces(db)
            && let Some(t) = traces.into_iter().next()
        {
            return Some(t);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_first_provider_whose_key_is_present() {
        let pick = pick_provider(|k| (k == "OPENAI_API_KEY").then(|| "sk-x".to_owned()))
            .expect("openai key must be found");
        assert_eq!(pick.provider, Provider::OpenAi);
        assert_eq!(pick.key_env, "OPENAI_API_KEY");

        // Anthropic wins when both are set — it is first in priority order.
        let both = pick_provider(|k| {
            matches!(k, "OPENAI_API_KEY" | "ANTHROPIC_API_KEY").then(|| "sk-x".to_owned())
        })
        .expect("a key must be found");
        assert_eq!(both.provider, Provider::Anthropic);

        assert_eq!(
            pick_provider(|_| None),
            None,
            "no key at all is a supported state, not an error"
        );
    }

    /// A leftover `export ANTHROPIC_API_KEY=` in a shell profile is common. Treating it as a key
    /// would send the kiosk into a guaranteed 401 instead of the demo that always works.
    #[test]
    fn a_blank_key_counts_as_absent() {
        assert_eq!(
            pick_provider(|k| (k == "ANTHROPIC_API_KEY").then(|| "   ".to_owned())),
            None
        );
    }

    /// The handshake reports; it does not act. An agent calling it to *find out* whether it is
    /// set up must not discover that asking rewrote the working tree.
    #[test]
    fn handshake_is_machine_readable_and_writes_nothing() {
        let before = std::path::Path::new("firstpass.toml").exists();
        let doc = handshake_static();

        // Every field an agent needs is a value, not prose it would have to parse.
        assert_eq!(doc["product"], "firstpass");
        assert!(doc["version"].is_string());
        assert!(doc["provider"]["keys_present"].is_array());
        assert!(doc["config"]["exists"].is_boolean());
        assert!(doc["route_your_agent"]["env"]["ANTHROPIC_BASE_URL"].is_string());
        // Offboarding stays one env var — the invariant the whole product rests on.
        assert_eq!(doc["offboard"]["unset"][0], "ANTHROPIC_BASE_URL");
        // The static form must not claim a self-test it did not run.
        assert!(doc["selftest"]["ok"].is_null());

        assert_eq!(
            std::path::Path::new("firstpass.toml").exists(),
            before,
            "handshake must not create or remove config"
        );
    }

    /// The kiosk must emit config the parser actually accepts, for every provider it can pick —
    /// a config that fails to parse would turn the one-press button into a stack trace.
    #[test]
    fn every_pickable_provider_yields_a_parseable_enforcing_config() {
        for (provider, key_env) in CANDIDATES {
            let toml = kiosk_config(Pick { provider, key_env });
            let cfg = firstpass_core::Config::parse(&toml)
                .unwrap_or_else(|e| panic!("{provider:?} kiosk config must parse: {e}"));
            assert_eq!(
                cfg.routes[0].mode,
                Mode::Enforce,
                "{provider:?} must enforce, or the receipt shows no decision"
            );
            // Parse enforces that every rung is priced, so the receipt cannot report a $0.00
            // spend it never measured — assert the ladder is actually populated.
            assert!(
                cfg.routes[0].ladder.len() >= 2,
                "{provider:?} needs a rung to escalate to"
            );
        }
    }
}
