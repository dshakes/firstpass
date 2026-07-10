//! `firstpass-bench` — run the M0 proof harness and print the report.
//!
//! Usage:
//!   firstpass-bench            # simulated report (Markdown) to stdout
//!   firstpass-bench --json     # machine-readable JSON report
//!   firstpass-bench --live     # REAL providers: needs ANTHROPIC_API_KEY (BYOK); costs real tokens
//!   firstpass-bench --sandbox-selfcheck  # prove the code-exec sandbox isolates, then exit (ADR 0002)
//!   firstpass-bench --coding       # coding-with-tests benchmark, MOCK solver in the sandbox (no spend)
//!   firstpass-bench --coding-live  # coding-with-tests with a LIVE candidate model (needs ANTHROPIC_API_KEY)

use firstpass_bench::coding::{
    CandidateSolver, CodingReport, LiveSolver, coding_suite, mock_solutions, run_coding_benchmark,
};
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

    // Coding-with-tests benchmark (Batch 3b): candidate writes code → run visible (gate) + hidden
    // (oracle) in the fail-closed sandbox → real gate error + conformal. `--coding` uses the mock
    // solver (no model spend); `--coding-live` uses a live candidate model.
    if args.iter().any(|a| a == "--coding" || a == "--coding-live") {
        let live_coding = args.iter().any(|a| a == "--coding-live");
        let sb = match establish_sandbox(SANDBOX_IMAGE) {
            Ok(sb) => sb,
            Err(e) => {
                eprintln!("cannot run coding benchmark — sandbox not established: {e}");
                std::process::exit(1);
            }
        };
        let cfg = BenchConfig::default();
        let solver: Box<dyn CandidateSolver> = if live_coding {
            let key = match std::env::var("ANTHROPIC_API_KEY") {
                Ok(k) if !k.is_empty() => k,
                _ => {
                    eprintln!("--coding-live needs ANTHROPIC_API_KEY set (BYOK)");
                    std::process::exit(2);
                }
            };
            Box::new(LiveSolver::new(key, "claude-haiku-4-5".to_owned()))
        } else {
            Box::new(mock_solutions())
        };
        match run_coding_benchmark(
            &coding_suite(),
            solver.as_ref(),
            sb.as_ref(),
            cfg.conformal_alpha,
            cfg.conformal_delta,
            50, // min served samples for a conformal bound (matches the arithmetic path)
        ) {
            Ok(report) => print_coding(&report),
            Err(e) => {
                eprintln!("coding benchmark failed: {e}");
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

/// Print a coding-with-tests report: the gate's real error, the served-failure it induces, and the
/// conformal bound (with an honest note when the demo suite is too small to be feasible).
fn print_coding(r: &CodingReport) {
    println!("# Coding-with-tests benchmark (Batch 3b)\n");
    println!("- sandbox runtime tier: {}", r.runtime_tier);
    println!("- tasks: {}", r.n);
    println!(
        "- candidate oracle pass-rate: {:.1}%",
        r.oracle_pass_rate * 100.0
    );
    println!(
        "- gate false-accept rate (P(gate pass | incorrect)): {:.1}%  <- real gate error (arithmetic had 0)",
        r.gate_false_accept_rate * 100.0
    );
    println!(
        "- gate false-reject rate (P(gate fail | correct)):  {:.1}%",
        r.gate_false_reject_rate * 100.0
    );
    println!(
        "- served-failure if you serve every gate-pass:      {:.1}%",
        r.served_failure_rate * 100.0
    );
    println!(
        "- conformal: threshold {:.3}, served {:.0}%, calib-risk {:.1}%, feasible={}",
        r.conformal.threshold,
        r.conformal.served_frac * 100.0,
        r.conformal.calib_risk * 100.0,
        r.conformal.feasible
    );
    if !r.conformal.feasible {
        println!(
            "  (infeasible: this demo suite is tiny — a real conformal bound needs a much larger live run)"
        );
    }
    println!("\nper-task (gate_pass / oracle_correct):");
    for o in &r.outcomes {
        println!(
            "  {:<16} gate={:<5} oracle={}",
            o.id, o.gate_pass, o.oracle_correct
        );
    }
}
