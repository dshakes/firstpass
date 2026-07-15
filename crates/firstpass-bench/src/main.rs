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
    CandidateSolver, CodingReport, GeneratedSolver, Judge, LiveJudge, LiveSolver, coding_suite,
    generated_coding_suite, mock_solutions, run_coding_benchmark, run_coding_benchmark_judged,
};
use firstpass_bench::dataset::load_mbpp_jsonl;
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

    // Coding-with-tests benchmark (Batch 3b/3c): candidate writes code → run visible (gate) + hidden
    // (oracle) in the fail-closed sandbox → real gate error + conformal on the continuous score.
    // `--coding` = offline solver (no model spend); `--coding-live` = live candidate model.
    // FIRSTPASS_CODING_N=<n> swaps the 3-task demo suite for the scalable generated suite of n tasks
    // (needed for a feasible conformal bound).
    if args.iter().any(|a| a == "--coding" || a == "--coding-live") {
        let live_coding = args.iter().any(|a| a == "--coding-live");
        let dataset_path = std::env::var("FIRSTPASS_CODING_DATASET").ok();
        if dataset_path.is_some() && !live_coding {
            eprintln!(
                "FIRSTPASS_CODING_DATASET needs --coding-live (plain --coding has no live solver to grade a real dataset)"
            );
            std::process::exit(2);
        }
        let sb = match establish_sandbox(SANDBOX_IMAGE) {
            Ok(sb) => sb,
            Err(e) => {
                eprintln!("cannot run coding benchmark — sandbox not established: {e}");
                std::process::exit(1);
            }
        };
        let cfg = BenchConfig::default();
        let n_gen = std::env::var("FIRSTPASS_CODING_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let tasks = match &dataset_path {
            Some(path) => match load_mbpp_jsonl(path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("failed to load coding dataset {path}: {e}");
                    std::process::exit(1);
                }
            },
            None => match n_gen {
                Some(n) => generated_coding_suite(n),
                None => coding_suite(),
            },
        };
        // Offline solver must match the suite: GeneratedSolver for the generated suite, mock for demo.
        let solver: Box<dyn CandidateSolver> = if live_coding {
            let key = match std::env::var("ANTHROPIC_API_KEY") {
                Ok(k) if !k.is_empty() => k,
                _ => {
                    eprintln!("--coding-live needs ANTHROPIC_API_KEY set (BYOK)");
                    std::process::exit(2);
                }
            };
            Box::new(LiveSolver::new(key, "claude-haiku-4-5".to_owned()))
        } else if n_gen.is_some() {
            Box::new(GeneratedSolver::new(cfg.backend_seed))
        } else {
            Box::new(mock_solutions())
        };
        // Optional judge gate: FIRSTPASS_CODING_JUDGE=<model> adds a continuous, correctness-aware
        // score so conformal can separate full-visible-pass false-accepts (the test-only gate can't).
        // FIRSTPASS_CODING_JUDGE_SAMPLES=<n> = self-consistency: score is the YES-verdict frequency
        // over n samples (default 5).
        let judge: Option<Box<dyn Judge>> = std::env::var("FIRSTPASS_CODING_JUDGE")
            .ok()
            .filter(|m| !m.is_empty())
            .and_then(|model| {
                let jkey = std::env::var("ANTHROPIC_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty())?;
                let samples = std::env::var("FIRSTPASS_CODING_JUDGE_SAMPLES")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5); // self-consistency: verdict frequency over 5 samples
                Some(Box::new(LiveJudge::new(jkey, model, samples)) as Box<dyn Judge>)
            });
        let result = match &judge {
            Some(j) => run_coding_benchmark_judged(
                &tasks,
                solver.as_ref(),
                j.as_ref(),
                sb.as_ref(),
                cfg.conformal_alpha,
                cfg.conformal_delta,
                50,
            ),
            None => run_coding_benchmark(
                &tasks,
                solver.as_ref(),
                sb.as_ref(),
                cfg.conformal_alpha,
                cfg.conformal_delta,
                50, // min served samples for a conformal bound (matches the arithmetic path)
            ),
        };
        match result {
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
/// conformal bound on the continuous score (with an honest note when the suite is too small).
fn print_coding(r: &CodingReport) {
    println!("# Coding-with-tests benchmark (Batch 3b/3c)\n");
    println!("- sandbox runtime tier: {}", r.runtime_tier);
    println!("- tasks: {}", r.n);
    println!(
        "- candidate oracle pass-rate: {:.1}%",
        r.oracle_pass_rate * 100.0
    );
    println!(
        "- gate false-accept rate (P(full visible pass | incorrect)): {:.1}%  <- real gate error (arithmetic had 0)",
        r.gate_false_accept_rate * 100.0
    );
    println!(
        "- gate false-reject rate (P(not full pass | correct)):       {:.1}%",
        r.gate_false_reject_rate * 100.0
    );
    println!(
        "- served-failure if you serve every full-visible-pass:       {:.1}%",
        r.served_failure_full_pass * 100.0
    );
    println!(
        "- conformal (continuous score): threshold {:.3}, served {:.0}%, calib-risk {:.1}%, feasible={}",
        r.conformal.threshold,
        r.conformal.served_frac * 100.0,
        r.conformal.calib_risk * 100.0,
        r.conformal.feasible
    );
    println!(
        "- served-failure at the conformal threshold:                 {:.1}%",
        r.served_failure_at_threshold * 100.0
    );
    if !r.conformal.feasible {
        println!(
            "  (test-only gate: full-pass false-accepts score 1.0 = correct answers, so no threshold\n   separates them. Add a judge gate: FIRSTPASS_CODING_JUDGE=claude-sonnet-5)"
        );
    }
    if let (Some(cc), Some(sf)) = (r.conformal_combined.as_ref(), r.served_failure_combined) {
        println!(
            "- conformal (COMBINED score = test + judge): threshold {:.3}, served {:.0}%, calib-risk {:.1}%, feasible={}",
            cc.threshold,
            cc.served_frac * 100.0,
            cc.calib_risk * 100.0,
            cc.feasible
        );
        println!(
            "- served-failure at the combined threshold:                  {:.1}%",
            sf * 100.0
        );
        if cc.feasible {
            println!(
                "  ^ FEASIBLE distribution-free bound — the judge separates full-pass false-accepts\n    the test-only gate cannot. This is the guarantee arithmetic could not earn."
            );
        }
    }
    let shown = r.outcomes.len().min(12);
    println!("\nper-task (gate_score / oracle_correct), first {shown}:");
    for o in r.outcomes.iter().take(shown) {
        println!(
            "  {:<22} score={:.2} oracle={}",
            o.id, o.gate_score, o.oracle_correct
        );
    }
}
