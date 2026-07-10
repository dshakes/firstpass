//! `firstpass-bench` — run the M0 proof harness and print the report.
//!
//! Usage:
//!   firstpass-bench            # simulated report (Markdown) to stdout
//!   firstpass-bench --json     # machine-readable JSON report
//!   firstpass-bench --live     # REAL providers: needs ANTHROPIC_API_KEY (BYOK); costs real tokens
//!   firstpass-bench --sandbox-selfcheck  # prove the code-exec sandbox isolates, then exit (ADR 0002)

use firstpass_bench::sandbox::establish_sandbox;
use firstpass_bench::{BenchConfig, run_benchmark, run_benchmark_live};

/// Container image for the sandbox self-check (needs `python3` + busybox `base64`/`timeout`).
const SANDBOX_IMAGE: &str = "python:3.12-alpine";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json = args.iter().any(|a| a == "--json");
    let live = args.iter().any(|a| a == "--live");

    // Operator's isolation proof surface (ADR 0002 §D3): establish the sandbox and run the isolation
    // probes against the REAL runtime. Fails closed — a breach or missing runtime exits non-zero.
    if args.iter().any(|a| a == "--sandbox-selfcheck") {
        match establish_sandbox(SANDBOX_IMAGE) {
            Ok(sb) => {
                println!(
                    "sandbox OK: isolation proven (runtime tier: {})",
                    sb.runtime()
                );
            }
            Err(e) => {
                eprintln!("sandbox FAILED (candidate code must NOT run): {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let cfg = BenchConfig::default();

    let report = if live {
        let key = match std::env::var("ANTHROPIC_API_KEY") {
            Ok(k) if !k.is_empty() => k,
            _ => {
                eprintln!(
                    "--live needs ANTHROPIC_API_KEY set (BYOK; used only to call Anthropic, never stored)"
                );
                std::process::exit(2);
            }
        };
        match run_benchmark_live(&cfg, key) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("live run failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        run_benchmark(&cfg)
    };

    if json {
        match report.to_json() {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("failed to serialize report: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.to_markdown());
    }
}
