//! `firstpass-bench` — run the M0 proof harness and print the report.
//!
//! Usage:
//!   firstpass-bench            # Markdown report to stdout
//!   firstpass-bench --json     # machine-readable JSON report

use firstpass_bench::{BenchConfig, run_benchmark};

fn main() {
    let json = std::env::args().any(|a| a == "--json");
    let report = run_benchmark(&BenchConfig::default());

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
