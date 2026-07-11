//! `firstpass` — the unified CLI (SPEC §7.3/§7.4): start the proxy, validate a setup, or read the
//! audit trail. Every subcommand reads the same environment as the server, so onboarding is one
//! `base_url` swap and offboarding is one `unset`.

use firstpass_proxy::calibrate::calibrate_from_store;
use firstpass_proxy::{ProxyConfig, cli, run, store};

const HELP: &str = "\
firstpass — the cheapest model that provably passes, with a receipt for every call.

USAGE:
    firstpass up                  start the proxy (serves until Ctrl-C)
    firstpass doctor              validate config, provider key, and gate binaries
    firstpass trace [--limit N]   print recent audit traces as JSON lines (default 20)
    firstpass calibrate [--alpha A] [--delta D] [--min-n N]
                                   recalibrate the serving threshold from deferred feedback
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
        "calibrate" => cmd_calibrate(&args),
        "mcp" => {
            // Synchronous stdio server; run it off the async runtime so nothing else contends.
            // Scoped to a single tenant (ADR 0004 §D3): `--tenant <id>` or the configured default.
            let config = ProxyConfig::from_env()?;
            let tenant = tenant_arg(&args, &config);
            let db = config.db_path;
            tokio::task::spawn_blocking(move || firstpass_proxy::mcp::serve_stdio(&db, &tenant))
                .await??;
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

/// The tenant a read-side CLI scopes to: `--tenant <id>` if given, else the configured default
/// (ADR 0004 §D3). Keeps single-operator use zero-config while never silently reading across tenants.
fn tenant_arg(args: &[String], config: &ProxyConfig) -> String {
    args.iter()
        .position(|a| a == "--tenant")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| config.tenant_id.clone())
}

/// `firstpass trace [--limit N] [--tenant ID]` — read recent audit records, scoped to one tenant.
fn cmd_trace(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let limit = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or(20);
    let config = ProxyConfig::from_env()?;
    let tenant = tenant_arg(args, &config);
    let traces = store::load_tenant_traces(std::path::Path::new(&config.db_path), &tenant)
        .unwrap_or_default();
    println!("{}", cli::format_traces(&traces, limit));
    Ok(())
}

/// `firstpass calibrate [--alpha A] [--delta D] [--min-n N]` — recalibrate the conformal serving
/// threshold from real deferred feedback recorded in the trace store. An empty or not-yet-created
/// store reports 0 pairs (feasible: false) and exits 0, like `trace`; only a genuine read error on
/// an existing trace's feedback bubbles up non-zero.
fn cmd_calibrate(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let flag = |name: &str, default: f64| -> f64 {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(default)
    };
    let alpha = flag("--alpha", 0.1);
    let delta = flag("--delta", 0.05);
    let min_n = flag("--min-n", 30.0) as usize;

    let config = ProxyConfig::from_env()?;
    let tenant = tenant_arg(args, &config);
    let report = calibrate_from_store(&config.db_path, &tenant, alpha, delta, min_n)?;
    print!("{}", report.render());
    // Infeasible is a valid finding (not enough clean feedback yet, or the gate is too weak), not a
    // failure — the report says `feasible: false`. Only a store read error (the `?` above) exits
    // non-zero, so scripting `firstpass calibrate` for its output stays reliable.
    Ok(())
}
