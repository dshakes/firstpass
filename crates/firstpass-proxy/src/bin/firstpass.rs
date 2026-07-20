//! `firstpass` — the unified CLI (SPEC §7.3/§7.4): start the proxy, validate a setup, or read the
//! audit trail. Every subcommand reads the same environment as the server, so onboarding is one
//! `base_url` swap and offboarding is one `unset`.

use firstpass_proxy::calibrate::{calibrate_from_store, calibrate_from_store_ltt};
use firstpass_proxy::ope::{CandidatePolicy, ips_from_store, ope_from_store};
use firstpass_proxy::{ProxyConfig, cli, run, store};

const HELP: &str = "\
firstpass — the cheapest model that provably passes, with a receipt for every call.

USAGE:
    firstpass onboard [--apply]   agentic setup: detect env, start proxy, route your agent, verify
    firstpass offboard            undo it: strip the rc line, stop the proxy, print the unset
    firstpass up                  start the proxy (serves until Ctrl-C)
    firstpass doctor              validate config, provider key, and gate binaries
    firstpass trace [--limit N]   print recent audit traces as JSON lines (default 20)
    firstpass savings [--json]    spend vs the always-top counterfactual, from your own receipts
    firstpass evals [--json]      per-gate verdict rates + escalation + serve-by-rung, from receipts
    firstpass explain <trace-id>  [--json]  why one routing decision went the way it did
    firstpass export [--out F]    write the sealed receipt log as JSONL (hand to an auditor)
    firstpass verify [--file F] [--json]
                                   independently re-derive the receipt hash chain; exit 1 if broken
    firstpass calibrate [--alpha A] [--delta D] [--min-n N] [--method conformal|ltt]
                                   recalibrate the serving threshold from deferred feedback
                                   (default method: conformal; ltt = Learn-then-Test / RCPS)
    firstpass ope --config <candidate.toml> [--db <path>] [--tenant <id>]
                                   evaluate a candidate policy against logged traffic before enforcing
    firstpass ope --start-rung N [--db <path>] [--tenant <id>]
                                   IPS/SNIPS/DR estimate for a fixed start-rung (requires propensity-logged traffic)
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
        "onboard" => cmd_onboard(args.iter().any(|a| a == "--apply")),
        "offboard" => cmd_offboard(),
        "doctor" => cmd_doctor(),
        "trace" => cmd_trace(&args),
        "savings" => cmd_savings(&args),
        "evals" => cmd_evals(&args),
        "explain" => cmd_explain(&args),
        "export" => cmd_export(&args),
        "verify" => cmd_verify(&args),
        "calibrate" => cmd_calibrate(&args),
        "ope" => cmd_ope(&args),
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

/// `firstpass onboard [--apply]` — agentic setup: detect the environment, plan the exact steps,
/// execute them under `--apply` (dry run otherwise), and verify end-to-end.
fn cmd_onboard(apply: bool) -> Result<(), Box<dyn std::error::Error>> {
    use firstpass_proxy::onboard;
    let env = onboard::detect(
        |k| std::env::var(k).ok(),
        |bin| {
            std::env::var("PATH")
                .is_ok_and(|p| std::env::split_paths(&p).any(|d| d.join(bin).is_file()))
        },
        || {
            let bind = std::env::var("FIRSTPASS_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
            std::net::TcpStream::connect_timeout(
                &bind
                    .parse()
                    .unwrap_or_else(|_| ([127, 0, 0, 1], 8080).into()),
                std::time::Duration::from_millis(400),
            )
            .is_ok()
        },
    );
    let home = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()));
    let (rc, _) = onboard::shell_wiring(&env.shell, &home, &env.bind);
    let steps = onboard::plan(&env, &home, onboard::rc_wired(&rc));
    print!("{}", onboard::render(&env, &steps, apply));
    if apply {
        print!("\n{}", onboard::execute(&env, &steps)?);
    }
    Ok(())
}

/// `firstpass offboard` — the mirror: strip the marked rc line(s), stop the proxy onboard started,
/// print the one command for this shell.
fn cmd_offboard() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()));
    print!("{}", firstpass_proxy::onboard::offboard(&home)?);
    Ok(())
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

/// `firstpass savings [--json] [--tenant ID]` — aggregate spend vs the always-top counterfactual
/// from the trace store: the operator's own measured number, not a marketing claim. Empty or
/// missing store prints the zero state and exits 0, like `trace`.
fn cmd_savings(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let config = ProxyConfig::from_env()?;
    let tenant = tenant_arg(args, &config);
    let traces = store::load_tenant_traces(std::path::Path::new(&config.db_path), &tenant)
        .unwrap_or_default();
    let summary = cli::summarize_savings(&traces);
    if args.iter().any(|a| a == "--json") {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("{}", cli::format_savings(&summary));
    }
    Ok(())
}

/// `firstpass evals [--json] [--tenant ID]` — the live eval suite computed from receipts:
/// per-gate pass/fail/abstain, escalation count, and which rung serves. Empty store → zero
/// state, exit 0, like `savings`.
fn cmd_evals(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let config = ProxyConfig::from_env()?;
    let tenant = tenant_arg(args, &config);
    let traces = store::load_tenant_traces(std::path::Path::new(&config.db_path), &tenant)
        .unwrap_or_default();
    let summary = cli::summarize_evals(&traces);
    if args.iter().any(|a| a == "--json") {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("{}", cli::format_evals(&summary));
    }
    Ok(())
}

/// `firstpass explain <trace-id> [--json] [--tenant ID]` — explain a single routing decision
/// from its sealed receipt: served model, per-rung verdicts, escalations, cost vs the always-top
/// baseline, and the savings. Unknown id exits 1.
fn cmd_explain(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let trace_id = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .ok_or("usage: firstpass explain <trace-id> [--json]")?;
    let config = ProxyConfig::from_env()?;
    let tenant = tenant_arg(args, &config);
    match store::load_trace_view(std::path::Path::new(&config.db_path), &tenant, trace_id)? {
        Some(trace) => {
            let ex = cli::explain_trace(&trace);
            if args.iter().any(|a| a == "--json") {
                println!("{}", serde_json::to_string_pretty(&ex)?);
            } else {
                println!("{}", ex.summary);
                for a in &ex.attempts {
                    let gates: Vec<String> =
                        a.gates.iter().map(|(g, v)| format!("{g}={v}")).collect();
                    println!(
                        "  rung {} {} → {} [{}]",
                        a.rung,
                        a.model,
                        a.verdict,
                        gates.join(", ")
                    );
                }
            }
            Ok(())
        }
        None => {
            eprintln!("unknown trace_id {trace_id:?}");
            std::process::exit(1);
        }
    }
}

/// `firstpass export [--out FILE]` — write the operator-wide sealed receipt log as JSONL
/// (one receipt per line, in chain order), to a file or stdout. This is the artifact an
/// operator hands an external auditor; it carries only the hashed bodies, never the deferred
/// verdicts. Empty/missing store writes nothing and exits 0.
fn cmd_export(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let config = ProxyConfig::from_env()?;
    // Operator-wide (the chain spans all tenants — a per-tenant view can't be chain-verified).
    let traces = store::load_all_traces(std::path::Path::new(&config.db_path)).unwrap_or_default();
    let jsonl = cli::export_receipts_jsonl(&traces);
    match args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
    {
        Some(path) => {
            std::fs::write(path, jsonl)?;
            eprintln!("exported {} receipts to {path}", traces.len());
        }
        None => print!("{jsonl}"),
    }
    Ok(())
}

/// `firstpass verify [--file FILE] [--json]` — independently re-derive the receipt hash chain
/// from genesis and report whether it is intact. With `--file` it verifies an exported JSONL
/// log (the auditor's path: no proxy, no database, no trust); without it, the local store.
/// **Exits 1 if the chain is broken** so it drops straight into CI / compliance gates.
fn cmd_verify(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let traces = match args
        .iter()
        .position(|a| a == "--file")
        .and_then(|i| args.get(i + 1))
    {
        Some(path) => {
            let text = std::fs::read_to_string(path)?;
            cli::parse_receipt_jsonl(&text).map_err(|e| format!("{path}: {e}"))?
        }
        None => {
            let config = ProxyConfig::from_env()?;
            store::load_all_traces(std::path::Path::new(&config.db_path)).unwrap_or_default()
        }
    };
    let report = cli::verify_receipts(&traces);
    if args.iter().any(|a| a == "--json") {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if report.valid {
        println!(
            "OK: {} receipts, hash chain intact from genesis",
            report.receipts
        );
    } else {
        println!(
            "TAMPERED: chain broke at receipt {} — {}",
            report.broken_at.map_or("?".to_owned(), |i| i.to_string()),
            report.detail.as_deref().unwrap_or("unknown")
        );
    }
    if !report.valid {
        std::process::exit(1);
    }
    Ok(())
}

/// `firstpass calibrate [--alpha A] [--delta D] [--min-n N] [--method conformal|ltt]` —
/// recalibrate the serving threshold from real deferred feedback recorded in the trace store.
/// An empty or not-yet-created store reports 0 pairs (feasible: false) and exits 0, like
/// `trace`; only a genuine store read error exits non-zero.
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
    let method = args
        .iter()
        .position(|a| a == "--method")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("conformal");

    let config = ProxyConfig::from_env()?;
    let tenant = tenant_arg(args, &config);

    // Infeasible is a valid finding (not enough clean feedback yet, or a weak gate) — the
    // report says `feasible: false`. Only a store read error bubbles up as non-zero exit.
    match method {
        "ltt" => {
            let report = calibrate_from_store_ltt(&config.db_path, &tenant, alpha, delta, min_n)?;
            print!("{}", report.render());
        }
        _ => {
            let report = calibrate_from_store(&config.db_path, &tenant, alpha, delta, min_n)?;
            print!("{}", report.render());
        }
    }
    Ok(())
}

/// `firstpass ope` — off-policy evaluation in two modes:
///
/// - `--config <candidate.toml>`: direct-method replay (ladder / threshold changes).
/// - `--start-rung N`: IPS/SNIPS estimate for a fixed start rung (requires propensity-logged
///   traffic from `[escalation.exploration]`). Mutually exclusive with `--config`.
///
/// Empty/missing store is treated as zero traces and exits 0, matching `calibrate`.
fn cmd_ope(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    // Env config provides defaults for db and tenant; --db / --tenant override them.
    let env_config = ProxyConfig::from_env()?;
    let db_path = args
        .iter()
        .position(|a| a == "--db")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| env_config.db_path.clone());
    let tenant = tenant_arg(args, &env_config);

    // --start-rung N: IPS/SNIPS path (takes precedence over --config when both given).
    if let Some(n_str) = args
        .iter()
        .position(|a| a == "--start-rung")
        .and_then(|i| args.get(i + 1))
    {
        let start_rung: u32 = n_str.parse().map_err(|e| {
            format!("firstpass ope: --start-rung must be a non-negative integer: {e}")
        })?;
        let report = ips_from_store(&db_path, &tenant, start_rung)?;
        print!("{}", report.render());
        return Ok(());
    }

    // --config <candidate.toml>: direct-method replay path (required if no --start-rung).
    let config_path = args
        .iter()
        .position(|a| a == "--config")
        .and_then(|i| args.get(i + 1))
        .ok_or_else(|| {
            eprintln!(
                "firstpass ope: one of --config <candidate.toml> or --start-rung N is required"
            );
            std::process::exit(1);
            // Unreachable, but satisfies the type: exit(1) never returns.
            #[allow(unreachable_code)]
            "unreachable"
        })?;

    let toml = std::fs::read_to_string(config_path)
        .map_err(|e| format!("firstpass ope: cannot read --config {config_path:?}: {e}"))?;
    let policy = CandidatePolicy::from_toml(&toml)
        .map_err(|e| format!("firstpass ope: invalid candidate config: {e}"))?;

    let report = ope_from_store(&db_path, &tenant, &policy)?;
    print!("{}", report.render());
    Ok(())
}
