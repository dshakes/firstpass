//! `firstpass-bench` — run the M0 proof harness and print the report.
//!
//! Usage:
//!   firstpass-bench            # simulated report (Markdown) to stdout
//!   firstpass-bench --json     # machine-readable JSON report
//!   firstpass-bench --live     # REAL providers: needs ANTHROPIC_API_KEY (BYOK); costs real tokens

use firstpass_bench::{BenchConfig, run_benchmark, run_benchmark_live};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json = args.iter().any(|a| a == "--json");
    let live = args.iter().any(|a| a == "--live");
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
