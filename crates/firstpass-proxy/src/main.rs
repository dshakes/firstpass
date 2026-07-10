//! Firstpass proxy binary: loads config, opens the trace store, and serves the observe- and
//! enforce-mode HTTP proxy until Ctrl-C.

use firstpass_proxy::{ProxyConfig, run};

/// Usage text for `--help`. The proxy is configured entirely through the environment
/// (12-factor), so `--help` doubles as the config reference — there are no subcommands.
const HELP: &str = "\
firstpass-proxy — drop-in, Anthropic-compatible LLM proxy that routes to the cheapest
model that provably passes your gate, and writes a tamper-evident receipt for every call.

USAGE:
    firstpass-proxy [--help] [--version]
    <configured via environment variables; then point your agent's ANTHROPIC_BASE_URL at it>

ENVIRONMENT:
    FIRSTPASS_MODE                observe (default) | enforce
    FIRSTPASS_BIND               listen address           [default 127.0.0.1:8080]
    FIRSTPASS_CONFIG             path to firstpass.toml (routes, ladders, gates)
    FIRSTPASS_DB                 trace store path         [default firstpass.db]
    FIRSTPASS_UPSTREAM_ANTHROPIC upstream base URL        [default https://api.anthropic.com]
    FIRSTPASS_UPSTREAM_OPENAI    upstream base URL        [default https://api.openai.com]
    FIRSTPASS_TENANT             tenant id for the trace  [default default]
    FIRSTPASS_PROMPT_SALT        salt for prompt hashing in traces
    RUST_LOG                     tracing filter           [default info]

QUICKSTART:
    firstpass-proxy                                   # observe mode, no behavior change
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8080   # point your agent at it
    # offboard anytime:  unset ANTHROPIC_BASE_URL

DOCS:  https://dshakes.github.io/firstpass  ·  SPEC: https://github.com/dshakes/firstpass/blob/main/SPEC.md";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Zero-dependency arg handling: the real interface is env vars, so we only field
    // the two flags every CLI is expected to answer.
    if let Some(flag) = std::env::args().nth(1) {
        match flag.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            "--version" | "-V" => {
                println!("firstpass-proxy {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            other => {
                eprintln!("firstpass-proxy: unrecognized argument `{other}`\n\n{HELP}");
                std::process::exit(2);
            }
        }
    }

    run::init_tracing();
    let config = ProxyConfig::from_env()?;
    run::serve(config).await
}
