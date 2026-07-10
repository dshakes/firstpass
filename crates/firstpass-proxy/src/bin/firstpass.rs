//! `firstpass` — the unified CLI (SPEC §7.3/§7.4): start the proxy, validate a setup, or read the
//! audit trail. Every subcommand reads the same environment as the server, so onboarding is one
//! `base_url` swap and offboarding is one `unset`.

use firstpass_proxy::{ProxyConfig, cli, load_all_traces, run};

const HELP: &str = "\
firstpass — the cheapest model that provably passes, with a receipt for every call.

USAGE:
    firstpass up                  start the proxy (serves until Ctrl-C)
    firstpass doctor              validate config, provider key, and gate binaries
    firstpass trace [--limit N]   print recent audit traces as JSON lines (default 20)
    firstpass mcp                 serve MCP over stdio (agent reads traces + submits feedback)
    firstpass --help | --version

ENVIRONMENT (shared by every subcommand):
    FIRSTPASS_MODE=observe|enforce   FIRSTPASS_BIND=127.0.0.1:8080
    FIRSTPASS_CONFIG=./firstpass.toml (or FIRSTPASS_CONFIG_TOML=<inline>)
    FIRSTPASS_DB=./firstpass.db       RUST_LOG=info

Point your agent at it:  export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
Offboard any time:       unset ANTHROPIC_BASE_URL";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str).unwrap_or("--help") {
        "up" => {
            run::init_tracing();
            run::serve(ProxyConfig::from_env()?).await
        }
        "doctor" => cmd_doctor(),
        "trace" => cmd_trace(&args),
        "mcp" => {
            // Synchronous stdio server; run it off the async runtime so nothing else contends.
            let db = ProxyConfig::from_env()?.db_path;
            tokio::task::spawn_blocking(move || firstpass_proxy::mcp::serve_stdio(&db)).await??;
            Ok(())
        }
        "--help" | "-h" | "help" => {
            println!("{HELP}");
            Ok(())
        }
        "--version" | "-V" => {
            println!("firstpass {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => {
            eprintln!("firstpass: unknown command `{other}`\n\n{HELP}");
            std::process::exit(2);
        }
    }
}

/// `firstpass doctor` — a config error is itself a finding, so report it and exit non-zero rather
/// than bubbling a raw error.
fn cmd_doctor() -> Result<(), Box<dyn std::error::Error>> {
    match ProxyConfig::from_env() {
        Ok(config) => {
            let report = cli::doctor(&config, |k| std::env::var(k).ok());
            print!("{}", report.render());
            if report.healthy() {
                Ok(())
            } else {
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("✗ config: {e}\n\nnot healthy — fix the config error above.");
            std::process::exit(1);
        }
    }
}

/// `firstpass trace [--limit N]` — read recent audit records from the store.
fn cmd_trace(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let limit = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or(20);
    let config = ProxyConfig::from_env()?;
    let traces = load_all_traces(std::path::Path::new(&config.db_path)).unwrap_or_default();
    println!("{}", cli::format_traces(&traces, limit));
    Ok(())
}
