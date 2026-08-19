//! `firstpass-bench` — run the M0 proof harness and print the report.
//!
//! Usage:
//!   firstpass-bench            # simulated report (Markdown) to stdout
//!   firstpass-bench --json     # machine-readable JSON report
//!   firstpass-bench --live     # REAL providers: needs ANTHROPIC_API_KEY (BYOK); costs real tokens
//!   firstpass-bench --sandbox-selfcheck  # prove the code-exec sandbox isolates, then exit (ADR 0002)
//!   firstpass-bench --coding       # coding-with-tests benchmark, MOCK solver in the sandbox (no spend)
//!   firstpass-bench --coding-live  # coding-with-tests with a LIVE candidate model (needs ANTHROPIC_API_KEY)
//!   firstpass-bench --replay <checkpoint.jsonl>  # re-study a paid run offline: no key, no sandbox, no spend
//!   firstpass-bench --multiturn-selfcheck  # prove the multi-turn harness + pre-registered bar work, no spend

use firstpass_bench::coding::{
    CandidateSolver, CodingReport, GeneratedSolver, Judge, LiveJudge, LiveSolver, coding_suite,
    generated_coding_suite, mock_solutions, run_coding_benchmark, run_coding_benchmark_judged,
};
use firstpass_bench::coding_policy::Rung;
use firstpass_bench::dataset::load_coding_dataset;
use firstpass_bench::sandbox::establish_sandbox;
use firstpass_bench::{BenchConfig, run_benchmark, run_benchmark_live};

/// Container image for the sandbox self-check (needs `python3` + busybox `base64`/`timeout`).
/// Default sandbox image. Override with `FIRSTPASS_SANDBOX_IMAGE` — BigCodeBench tasks import
/// numpy/pandas/etc, and the sandbox has no network to install them, so those runs need an image
/// that already carries the dataset's libraries.
const DEFAULT_SANDBOX_IMAGE: &str = "python:3.12-alpine";

/// The sandbox image to run candidates in.
fn sandbox_image() -> String {
    std::env::var("FIRSTPASS_SANDBOX_IMAGE").unwrap_or_else(|_| DEFAULT_SANDBOX_IMAGE.to_owned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json = args.iter().any(|a| a == "--json");
    let live = args.iter().any(|a| a == "--live");

    // Prove the multi-turn harness and its pre-registered bar are wired and can BOTH pass and fail
    // (ADR 0012). Runs on synthetic fixtures — no key, no sandbox, no spend — so it verifies the
    // apparatus, never the feature. A real number needs paid provider calls, which is the
    // operator's spending decision to make, not a benchmark's to trigger.
    if args.iter().any(|a| a == "--swe-agentic") {
        swe_agentic(json);
        return;
    }

    if args.iter().any(|a| a == "--agentic-multiturn") {
        agentic_multiturn(json);
        return;
    }

    if args.iter().any(|a| a == "--multiturn-selfcheck") {
        multiturn_selfcheck(json);
        return;
    }

    // Operator's isolation proof surface (ADR 0002 §D3): establish the sandbox and run the isolation
    // probes against the REAL runtime. Fails closed — a breach or missing runtime exits non-zero.
    if args.iter().any(|a| a == "--sandbox-selfcheck") {
        match establish_sandbox(&sandbox_image()) {
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
    // Probe-signal validation study (ADR 0008 go/no-go): draw k cheap samples per task, measure
    // whether self-consistency uncertainty predicts the oracle outcome. Needs a dataset + key + Docker.
    //   FIRSTPASS_CODING_DATASET=mbpp.jsonl FIRSTPASS_PROBE_K=5 firstpass-bench --probe-study
    // Start-rung ablation (offline, no spend): is the learned start rung smart, or just safe?
    // FIRSTPASS_ABLATION_N=<n> (default 8000) tasks; sweeps context->difficulty signal levels.
    // Elastic-verification offline validation (ADR 0008 Phase 3): does skipping the expensive
    // gate on the confident majority save cost while holding served-failure <= alpha (held-out)?
    if args.iter().any(|a| a == "--elastic") {
        let n = std::env::var("FIRSTPASS_ELASTIC_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10000);
        let r = firstpass_bench::elastic::run_elastic_validation(n, 0.10, 0.05, 1);
        println!("{}", firstpass_bench::elastic::render(&r));
        return;
    }

    // Re-study an already-paid measurement. A checkpoint written by --coding-policy holds the
    // per-task, per-rung outcomes, so every policy question after the run is arithmetic over that
    // file — no key, no sandbox, no spend. This is what makes the published numbers reproducible
    // by a reviewer who has neither our credentials nor our budget.
    if let Some(i) = args.iter().position(|a| a == "--replay") {
        let Some(path) = args.get(i + 1).filter(|p| !p.starts_with("--")) else {
            eprintln!("--replay needs a checkpoint path: --replay <checkpoint.jsonl>");
            std::process::exit(2);
        };
        // A file may hold several ladders; naming one is how you pick between experiments.
        let want: Option<Vec<String>> = std::env::var("FIRSTPASS_CODING_LADDER")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.split(',').map(|m| m.trim().to_owned()).collect());
        match firstpass_bench::coding_policy::load_matrix(
            std::path::Path::new(path),
            want.as_deref(),
        ) {
            Ok((matrix, ladder)) => {
                eprintln!(
                    "replaying {} tasks x {} rungs from {path} (offline, no spend)",
                    matrix.len(),
                    ladder.len()
                );
                // The runtime tier is a property of the run that produced the file, not of this
                // replay, so it is labelled as such rather than guessed.
                let study = firstpass_bench::coding_policy::replay(&matrix, &ladder, "replay");
                println!("{}", study.render());
                // The same matrix, scored the way the field's standard benchmark scores routers,
                // so the result is legible to a RouterBench reader without a second measurement.
                let rb = firstpass_bench::routerbench::evaluate(&matrix, &ladder);
                println!("\n{}", firstpass_bench::routerbench::render(&rb));
                // What the gate does when it is allowed to be wrong, and whether treating its
                // verdict as evidence rather than truth would have done better.
                let sw = firstpass_bench::sweep::sweep(&matrix, 0.10, 0.05, 50);
                println!("\n{}", firstpass_bench::sweep::render(&sw));
                let mv = firstpass_bench::metaverify::study(&matrix);
                println!("\n{}", firstpass_bench::metaverify::render(&mv));
                // Whether spending the cheap attempt was the right call in the first place.
                let ca = firstpass_bench::costaware::study(&matrix);
                println!("\n{}", firstpass_bench::costaware::render(&ca));
            }
            Err(e) => {
                eprintln!("cannot replay {path}: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // VRBench reference participant: run the verified cascade over the task set and WRITE a
    // submission. A ladder of one model yields a single-shot ungated submission, which is exactly
    // an AIQ baseline — so baselines come from this same path rather than being asserted.
    //   FIRSTPASS_CODING_LADDER=a,b firstpass-bench --vrbench-run tasks.jsonl out.jsonl
    if let Some(i) = args.iter().position(|a| a == "--vrbench-run") {
        let (Some(tasks_path), Some(out_path)) = (args.get(i + 1), args.get(i + 2)) else {
            eprintln!("usage: firstpass-bench --vrbench-run <tasks.jsonl> <out-submission.jsonl>");
            std::process::exit(2);
        };
        let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        let openai_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        let ladder: Vec<String> = std::env::var("FIRSTPASS_CODING_LADDER")
            .unwrap_or_else(|_| "anthropic/claude-haiku-4-5,anthropic/claude-sonnet-5".to_owned())
            .split(',')
            .map(|m| m.trim().to_owned())
            .collect();
        let text = std::fs::read_to_string(tasks_path).unwrap_or_else(|e| {
            eprintln!("cannot read {tasks_path}: {e}");
            std::process::exit(1);
        });
        let tasks = firstpass_bench::vrbench::parse_tasks(&text).unwrap_or_else(|e| {
            eprintln!("task file rejected: {e}");
            std::process::exit(1);
        });
        let solvers: Vec<firstpass_bench::coding::LiveSolver> = ladder
            .iter()
            .map(|m| {
                firstpass_bench::coding::LiveSolver::for_ladder_id(m, &key, openai_key.as_deref())
                    .unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(2);
                    })
            })
            .collect();
        let refs: Vec<(String, &dyn firstpass_bench::coding::CandidateSolver)> = ladder
            .iter()
            .cloned()
            .zip(
                solvers
                    .iter()
                    .map(|s| s as &dyn firstpass_bench::coding::CandidateSolver),
            )
            .collect();
        let sb = match establish_sandbox(&sandbox_image()) {
            Ok(sb) => sb,
            Err(e) => {
                eprintln!("cannot run — sandbox not established: {e}");
                std::process::exit(1);
            }
        };
        eprintln!(
            "running {} tasks over {} rung(s): {}",
            tasks.len(),
            ladder.len(),
            ladder.join(" -> ")
        );
        let prices = firstpass_core::PriceTable::defaults();
        let limits = firstpass_bench::sandbox::Limits::default();
        match firstpass_bench::vrbench::run_reference_router(
            &tasks,
            &refs,
            sb.as_ref(),
            &prices,
            &limits,
            // The output file IS the checkpoint: an interrupted run resumes from what it already
            // wrote, so an upstream 529 costs the task in flight rather than the whole run.
            Some(std::path::Path::new(out_path)),
        ) {
            Ok(subs) => {
                let mut out = String::new();
                for s in &subs {
                    match serde_json::to_string(s) {
                        Ok(l) => {
                            out.push_str(&l);
                            out.push('\n');
                        }
                        Err(e) => {
                            eprintln!("cannot serialise submission: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                if let Err(e) = std::fs::write(out_path, out) {
                    eprintln!("cannot write {out_path}: {e}");
                    std::process::exit(1);
                }
                let spend: f64 = subs.iter().map(|s| s.cost_usd).sum();
                eprintln!(
                    "wrote {} submissions to {out_path} (spend ${spend:.4})",
                    subs.len()
                );
            }
            Err(e) => {
                eprintln!("run failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // VRBench: score a THIRD-PARTY router's submission against the task set (specs/vrbench-v1.md).
    // Needs the sandbox (hidden cases execute untrusted model output) but no API key — the router
    // already made its calls; this only runs the oracle.
    if let Some(i) = args.iter().position(|a| a == "--vrbench") {
        let (Some(tasks_path), Some(sub_path)) = (args.get(i + 1), args.get(i + 2)) else {
            eprintln!("usage: firstpass-bench --vrbench <tasks.jsonl> <submission.jsonl>");
            std::process::exit(2);
        };
        let read = |p: &str| {
            std::fs::read_to_string(p).unwrap_or_else(|e| {
                eprintln!("cannot read {p}: {e}");
                std::process::exit(1);
            })
        };
        let tasks = firstpass_bench::vrbench::parse_tasks(&read(tasks_path)).unwrap_or_else(|e| {
            eprintln!("task file rejected: {e}");
            std::process::exit(1);
        });
        // Validation happens before any scoring: a submission that under-reports cost is rejected
        // outright rather than scored and caveated.
        let subs =
            firstpass_bench::vrbench::parse_submissions(&read(sub_path)).unwrap_or_else(|e| {
                eprintln!("submission rejected: {e}");
                std::process::exit(1);
            });
        let sb = match establish_sandbox(&sandbox_image()) {
            Ok(sb) => sb,
            Err(e) => {
                eprintln!(
                    "cannot score — sandbox not established: {e}\nhidden cases execute \
                     model-generated code and will not be run on the host"
                );
                std::process::exit(1);
            }
        };
        let limits = firstpass_bench::sandbox::Limits::default();
        match firstpass_bench::vrbench::score_all(sb.as_ref(), &tasks, &subs, &limits) {
            Ok(scored) => {
                let rep = firstpass_bench::vrbench::report(&scored);
                println!("{}", firstpass_bench::vrbench::render(&rep));
            }
            Err(e) => {
                eprintln!("scoring failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.iter().any(|a| a == "--ablation") {
        let n = std::env::var("FIRSTPASS_ABLATION_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8000);
        println!("{}", firstpass_bench::ablation::run_ablation_sweep(n, 1));
        return;
    }

    if args.iter().any(|a| a == "--probe-study") {
        let key = match std::env::var("ANTHROPIC_API_KEY") {
            Ok(k) if !k.is_empty() => k,
            _ => {
                eprintln!("--probe-study needs ANTHROPIC_API_KEY set (BYOK)");
                std::process::exit(2);
            }
        };
        let dataset_path = std::env::var("FIRSTPASS_CODING_DATASET").ok();
        let k: u32 = std::env::var("FIRSTPASS_PROBE_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let sb = match establish_sandbox(&sandbox_image()) {
            Ok(sb) => sb,
            Err(e) => {
                eprintln!("cannot run probe study — sandbox not established: {e}");
                std::process::exit(1);
            }
        };
        let n_gen = std::env::var("FIRSTPASS_CODING_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let tasks = match &dataset_path {
            Some(path) => match load_coding_dataset(path) {
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
        let solver = LiveSolver::new(key, "claude-haiku-4-5".to_owned());
        match firstpass_bench::coding::run_probe_study(&tasks, &solver, sb.as_ref(), k) {
            Ok(report) => {
                println!("{}", report.render());
                // Emit the per-task points as JSONL on stderr for the committed artifact / ROC.
                for pt in &report.points {
                    if let Ok(line) = serde_json::to_string(pt) {
                        eprintln!("{line}");
                    }
                }
            }
            Err(e) => {
                eprintln!("probe study failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // The policy study: does the first-pass LOGIC beat the baselines on real coding tasks? This
    // is a different question from `--coding`, which measures how good the gate is. Solves each
    // task once per rung and replays every policy over that one matrix, so the cost is
    // tasks x rungs regardless of how many policies are compared.
    if args.iter().any(|a| a == "--coding-policy") {
        let key = match std::env::var("ANTHROPIC_API_KEY") {
            Ok(k) if !k.is_empty() => k,
            _ => {
                eprintln!("--coding-policy needs ANTHROPIC_API_KEY set (BYOK)");
                std::process::exit(2);
            }
        };
        let dataset_path = match std::env::var("FIRSTPASS_CODING_DATASET") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!(
                    "--coding-policy needs FIRSTPASS_CODING_DATASET=<path.jsonl> (see scripts/fetch-coding-dataset.py)"
                );
                std::process::exit(2);
            }
        };
        let tasks = match load_coding_dataset(&dataset_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("failed to load coding dataset {dataset_path}: {e}");
                std::process::exit(1);
            }
        };
        let sb = match establish_sandbox(&sandbox_image()) {
            Ok(sb) => sb,
            Err(e) => {
                eprintln!("cannot run policy study — sandbox not established: {e}");
                std::process::exit(1);
            }
        };
        // Ladder, cheapest first. Overridable so the study is not welded to one pair of models.
        let ladder: Vec<String> = std::env::var("FIRSTPASS_CODING_LADDER")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.split(',').map(|m| m.trim().to_owned()).collect())
            .unwrap_or_else(|| {
                vec![
                    "anthropic/claude-haiku-4-5".to_owned(),
                    "anthropic/claude-sonnet-5".to_owned(),
                ]
            });
        // Provider comes from the ladder id's prefix, so a ladder may mix vendors
        // (e.g. openai/gpt-4.1-mini -> anthropic/claude-opus-4-8). Selection happens here, where
        // the prefix still exists — stripping it first is what previously made the OpenAI branch
        // unreachable.
        let openai_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        let solvers: Vec<LiveSolver> = ladder
            .iter()
            .map(|m| {
                LiveSolver::for_ladder_id(m, &key, openai_key.as_deref()).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2);
                })
            })
            .collect();
        let rungs: Vec<Rung> = ladder
            .iter()
            .zip(solvers.iter())
            .map(|(model, solver)| Rung {
                model: model.clone(),
                solver,
            })
            .collect();
        let prices = firstpass_core::cost::PriceTable::defaults();
        let limits = firstpass_bench::sandbox::Limits::default();
        eprintln!(
            "measuring {} tasks x {} rungs = {} live calls...",
            tasks.len(),
            rungs.len(),
            tasks.len() * rungs.len()
        );
        // FIRSTPASS_CODING_JUDGE=<model> adds a second opinion the tests do not have. The first
        // pilot's finding was that a visible-test prefix passes the cheap rung's wrong answers,
        // so this decides whether first-pass loses on the workload or merely on the gate.
        let judge: Option<LiveJudge> = std::env::var("FIRSTPASS_CODING_JUDGE")
            .ok()
            .filter(|m| !m.is_empty())
            .map(|model| {
                let samples = std::env::var("FIRSTPASS_CODING_JUDGE_SAMPLES")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1);
                LiveJudge::new(key.clone(), model, samples)
            });
        if judge.is_some() {
            eprintln!("judge enabled — one extra call per candidate");
        }
        // Checkpoint beside the dataset. A ~2000-call run WILL be interrupted eventually; without
        // this, every call already paid for goes with it.
        let ckpt = std::env::var("FIRSTPASS_CODING_CHECKPOINT")
            .unwrap_or_else(|_| format!("{dataset_path}.checkpoint.jsonl"));
        eprintln!("checkpoint: {ckpt} (delete it to force a clean re-measure)");
        match firstpass_bench::coding_policy::measure_resumable(
            &tasks,
            &rungs,
            sb.as_ref(),
            &prices,
            &limits,
            judge
                .as_ref()
                .map(|j| j as &dyn firstpass_bench::coding::Judge),
            Some(std::path::Path::new(&ckpt)),
        ) {
            Ok(matrix) => {
                let study = firstpass_bench::coding_policy::replay(&matrix, &ladder, sb.runtime());
                println!("{}", study.render());
            }
            Err(e) => {
                eprintln!("policy study aborted: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.iter().any(|a| a == "--coding" || a == "--coding-live") {
        let live_coding = args.iter().any(|a| a == "--coding-live");
        let dataset_path = std::env::var("FIRSTPASS_CODING_DATASET").ok();
        if dataset_path.is_some() && !live_coding {
            eprintln!(
                "FIRSTPASS_CODING_DATASET needs --coding-live (plain --coding has no live solver to grade a real dataset)"
            );
            std::process::exit(2);
        }
        let sb = match establish_sandbox(&sandbox_image()) {
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
            Some(path) => match load_coding_dataset(path) {
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

/// `--multiturn-selfcheck`: prove the multi-turn harness and its pre-registered bar are correctly
/// wired, using synthetic fixtures.
///
/// This verifies the **apparatus**, not the feature. Two arms are run deliberately: one where the
/// trajectory signal is genuinely informative (must SHIP) and one where it is misleading (must be
/// KILLED). A harness that can only ever return one verdict proves nothing, and the arm that must
/// fail is the one that makes the arm that passes worth reading.
///
/// Exits non-zero if either arm returns the wrong verdict, so CI catches a harness that has quietly
/// stopped discriminating.
fn multiturn_selfcheck(json: bool) {
    use firstpass_bench::coding_policy::RungOutcome;
    use firstpass_bench::multiturn::{MultiTurnTask, PreRegistered, Turn, evaluate};
    use firstpass_core::features::TrajectorySignals;

    let rung = |cost: f64, pass: bool, correct: bool| RungOutcome {
        gate_score: if pass { 1.0 } else { 0.0 },
        gate_full_pass: pass,
        oracle_correct: correct,
        cost_usd: cost,
        judge_score: None,
    };
    let stuck = TrajectorySignals {
        tool_errors: 8,
        tool_results: 10,
        assistant_turns: 12,
        repeated_tool_calls: 3,
    };
    let healthy = TrajectorySignals {
        tool_errors: 0,
        tool_results: 4,
        assistant_turns: 2,
        repeated_tool_calls: 0,
    };

    // Arm A — the signal predicts difficulty: struggling conversations really do precede turns the
    // cheap rung cannot handle.
    let informative: Vec<MultiTurnTask> = (0..40)
        .map(|i| MultiTurnTask {
            id: format!("informative-{i}"),
            turns: vec![
                Turn {
                    signals: healthy,
                    rungs: vec![rung(0.001, true, true), rung(0.010, true, true)],
                },
                Turn {
                    signals: stuck,
                    rungs: vec![rung(0.001, false, false), rung(0.010, true, true)],
                },
            ],
        })
        .collect();

    // Arm B — the signal is noise: every turn is easy, but half the conversations look stuck.
    // Acting on it buys the expensive rung for nothing.
    let misleading: Vec<MultiTurnTask> = (0..40)
        .map(|i| MultiTurnTask {
            id: format!("misleading-{i}"),
            turns: vec![
                Turn {
                    signals: stuck,
                    rungs: vec![rung(0.001, true, true), rung(0.010, true, true)],
                },
                Turn {
                    signals: healthy,
                    rungs: vec![rung(0.001, true, true), rung(0.010, true, true)],
                },
            ],
        })
        .collect();

    let a = evaluate(&informative, PreRegistered::default());
    let b = evaluate(&misleading, PreRegistered::default());

    if json {
        let doc = serde_json::json!({
            "informative_arm": a,
            "misleading_arm": b,
            "harness_ok": a.ships && !b.ships,
        });
        println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
    } else {
        println!("# Multi-turn harness self-check (synthetic fixtures, no spend)\n");
        println!("This proves the APPARATUS works. It does not measure the feature — that needs");
        println!("paid provider calls against a real multi-turn workload.\n");
        println!("Informative signal (must SHIP): {}", a.verdict);
        println!("Misleading signal (must be KILLED): {}", b.verdict);
        println!(
            "\nHarness discriminates: {}",
            if a.ships && !b.ships {
                "YES — the bar can both pass and fail"
            } else {
                "NO — the criterion is not discriminating, treat every result as suspect"
            }
        );
    }

    if !a.ships || b.ships {
        eprintln!(
            "multiturn-selfcheck FAILED: informative_ships={} misleading_ships={} \
             (expected true/false)",
            a.ships, b.ships
        );
        std::process::exit(1);
    }
}

/// `--agentic-multiturn`: run MBPP as real multi-turn agentic conversations and score the
/// trajectory-informed router against the pre-registered bar (ADR 0012).
///
/// This SPENDS REAL MONEY. Every turn is a paid model call, so:
/// - `FIRSTPASS_AGENTIC_BUDGET_USD` is a hard ceiling, checked before each task; the run stops
///   cleanly rather than overshooting, and what was already bought is kept.
/// - Every finished task is checkpointed immediately, and a resumed run skips it. A crash at task
///   900 of 974 must not destroy the first 899 — that has happened here before.
fn agentic_multiturn(json: bool) {
    use firstpass_bench::agentic::{AgenticRung, RecordedTask, run_task};
    use firstpass_bench::multiturn::{PreRegistered, evaluate};
    use std::io::Write as _;

    let Ok(dataset) = std::env::var("FIRSTPASS_CODING_DATASET") else {
        eprintln!("--agentic-multiturn needs FIRSTPASS_CODING_DATASET=<path.jsonl>");
        std::process::exit(2);
    };
    let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
        eprintln!("--agentic-multiturn needs ANTHROPIC_API_KEY (this run costs real money)");
        std::process::exit(2);
    };
    let budget: f64 = std::env::var("FIRSTPASS_AGENTIC_BUDGET_USD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);
    let limit: usize = std::env::var("FIRSTPASS_AGENTIC_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(974);
    let max_turns: usize = std::env::var("FIRSTPASS_AGENTIC_TURNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let ckpt_path = std::env::var("FIRSTPASS_AGENTIC_CHECKPOINT")
        .unwrap_or_else(|_| "agentic-multiturn.checkpoint.jsonl".to_owned());

    let tasks = match load_coding_dataset(&dataset) {
        Ok(mut t) => {
            t.truncate(limit);
            t
        }
        Err(e) => {
            eprintln!("cannot load {dataset}: {e}");
            std::process::exit(1);
        }
    };
    let sb = match establish_sandbox(&sandbox_image()) {
        Ok(sb) => sb,
        Err(e) => {
            eprintln!("sandbox FAILED (candidate code must NOT run): {e}");
            std::process::exit(1);
        }
    };

    // Per-million prices from the core table, converted to per-token.
    let rungs = vec![
        AgenticRung {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            bare: "claude-haiku-4-5".to_owned(),
            in_price: 1.0 / 1_000_000.0,
            out_price: 5.0 / 1_000_000.0,
        },
        AgenticRung {
            model: "anthropic/claude-sonnet-5".to_owned(),
            bare: "claude-sonnet-5".to_owned(),
            in_price: 3.0 / 1_000_000.0,
            out_price: 15.0 / 1_000_000.0,
        },
    ];

    // Resume: anything already bought is not bought again.
    let mut done: Vec<RecordedTask> = std::fs::read_to_string(&ckpt_path)
        .map(|t| {
            t.lines()
                .filter_map(|l| serde_json::from_str::<RecordedTask>(l).ok())
                .collect()
        })
        .unwrap_or_default();
    let already: std::collections::HashSet<String> = done.iter().map(|t| t.id.clone()).collect();
    if !already.is_empty() {
        eprintln!(
            "resuming: {} tasks already recorded in {ckpt_path}",
            already.len()
        );
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .unwrap_or_default();
    let limits = firstpass_bench::sandbox::Limits::default();
    let mut spent = 0.0f64;
    let mut stopped_early = false;

    for (i, task) in tasks.iter().enumerate() {
        if already.contains(&task.id) {
            continue;
        }
        if spent >= budget {
            eprintln!(
                "BUDGET REACHED: ${spent:.2} of ${budget:.2} after {i} tasks — stopping cleanly. \
                 Re-run with a higher FIRSTPASS_AGENTIC_BUDGET_USD to continue from the checkpoint."
            );
            stopped_early = true;
            break;
        }
        let call = |bare: &str, msgs: &serde_json::Value, budget: u32| {
            firstpass_bench::live::anthropic_call_messages(
                &client,
                "https://api.anthropic.com",
                &api_key,
                bare,
                Some("You are a Python coding assistant. Output ONLY code, no prose."),
                msgs,
                budget,
            )
        };
        match run_task(task, &rungs, sb.as_ref(), &limits, max_turns, &call) {
            Ok(run) => {
                spent += run.spent_usd;
                // Append BEFORE anything else can fail: a task that was paid for is recorded.
                if let (Ok(line), Ok(mut f)) = (
                    serde_json::to_string(&run.recorded),
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&ckpt_path),
                ) {
                    let _ = writeln!(f, "{line}");
                }
                done.push(run.recorded);
                if i % 25 == 0 {
                    eprintln!("[{i}/{}] spent ${spent:.2} of ${budget:.2}", tasks.len());
                }
            }
            Err(e) => {
                eprintln!(
                    "task {} failed: {e} — stopping (spent ${spent:.2})",
                    task.id
                );
                stopped_early = true;
                break;
            }
        }
    }

    let mt: Vec<_> = done.iter().map(RecordedTask::to_multiturn).collect();
    let result = evaluate(&mt, PreRegistered::default());

    if json {
        let doc = serde_json::json!({
            "tasks": done.len(),
            "spent_usd": spent,
            "stopped_early": stopped_early,
            "result": result,
        });
        println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
    } else {
        println!("# Agentic multi-turn MBPP — real trajectories, real spend\n");
        println!("tasks recorded: {}", done.len());
        println!("turns scored:   {}", result.n_turns);
        println!("spend this run: ${spent:.2} of ${budget:.2}");
        if stopped_early {
            println!("**stopped early** — the numbers below cover only what was bought.");
        }
        println!("\n| policy | success | $/success |");
        println!("|---|---|---|");
        println!(
            "| baseline (always start cheap) | {:.3} | ${:.5} |",
            result.baseline_success, result.baseline_cost_per_success
        );
        println!(
            "| trajectory-informed start | {:.3} | ${:.5} |",
            result.trajectory_success, result.trajectory_cost_per_success
        );
        println!(
            "\npaired quality delta: {:+.4}  CI [{:.4}, {:.4}]",
            result.delta_success, result.delta_success_ci.lo, result.delta_success_ci.hi
        );
        println!(
            "cost improvement:     {:+.1}%",
            result.cost_improvement * 100.0
        );
        println!("\n## Pre-registered verdict\n\n{}", result.verdict);
    }
}

/// `--swe-agentic`: run SWE-bench instances as real multi-turn repair sessions and score the
/// trajectory-informed router against the pre-registered bar (ADR 0012).
///
/// This is the workload MBPP could not provide. There, conversation depth capped at 2 and two of
/// four difficulty levels never fired; a repository task genuinely runs long, so the full hint range
/// is reachable and the trajectory signal has something to be a signal about.
///
/// SPENDS REAL MONEY and needs multi-GB eval images pulled in advance (no network at eval time).
fn swe_agentic(json: bool) {
    use firstpass_bench::agentic::RecordedTask;
    use firstpass_bench::multiturn::{PreRegistered, evaluate};
    use firstpass_bench::swe_agentic::{SweRung, run_instance, should_stop_after_failures};
    use firstpass_bench::swebench::{SweLimits, load_swebench_jsonl};
    use std::io::Write as _;

    let Ok(dataset) = std::env::var("FIRSTPASS_SWE_DATASET") else {
        eprintln!("--swe-agentic needs FIRSTPASS_SWE_DATASET=<path.jsonl>");
        eprintln!("  scripts/fetch-coding-dataset.py --dataset swebench --limit N --out swe.jsonl");
        std::process::exit(2);
    };
    let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
        eprintln!("--swe-agentic needs ANTHROPIC_API_KEY (this run costs real money)");
        std::process::exit(2);
    };
    let budget: f64 = std::env::var("FIRSTPASS_SWE_BUDGET_USD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);
    let max_turns: usize = std::env::var("FIRSTPASS_SWE_TURNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let limit: usize = std::env::var("FIRSTPASS_SWE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let ckpt_path = std::env::var("FIRSTPASS_SWE_CHECKPOINT")
        .unwrap_or_else(|_| "swe-agentic.checkpoint.jsonl".to_owned());

    let mut instances = match load_swebench_jsonl(&dataset) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot load {dataset}: {e}");
            std::process::exit(1);
        }
    };
    instances.truncate(limit);

    let rungs = vec![
        SweRung {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            bare: "claude-haiku-4-5".to_owned(),
            in_price: 1.0 / 1_000_000.0,
            out_price: 5.0 / 1_000_000.0,
        },
        SweRung {
            model: "anthropic/claude-sonnet-5".to_owned(),
            bare: "claude-sonnet-5".to_owned(),
            in_price: 3.0 / 1_000_000.0,
            out_price: 15.0 / 1_000_000.0,
        },
    ];

    let mut done: Vec<RecordedTask> = std::fs::read_to_string(&ckpt_path)
        .map(|t| {
            t.lines()
                .filter_map(|l| serde_json::from_str::<RecordedTask>(l).ok())
                .collect()
        })
        .unwrap_or_default();
    let already: std::collections::HashSet<String> = done.iter().map(|t| t.id.clone()).collect();
    if !already.is_empty() {
        eprintln!("resuming: {} instances already recorded", already.len());
    }

    let client = reqwest::blocking::Client::builder()
        // 300s was survivable while the token ladder topped out at 8192. Raising it to 32768
        // (see swe_agentic, the truncated-thinking fix) made the deadline reachable: at roughly
        // 60 output tokens/sec a full 32768-token response needs ~9 minutes, so a thinking-heavy
        // instance timed out deterministically and — because one failure aborted the run — wedged
        // the whole benchmark at instance 13. A cap that is smaller than the output it must carry
        // is not a safety limit, it is a guaranteed failure.
        .timeout(std::time::Duration::from_secs(900))
        .build()
        .unwrap_or_default();
    /// Enough consecutive failures to mean the provider is down or the config is wrong, rather
    /// than one awkward instance. Isolated failures are skipped; a run of them stops the harness.
    const MAX_CONSECUTIVE_FAILURES: usize = 3;
    let mut consecutive_failures = 0usize;
    let mut skipped: Vec<(String, String)> = Vec::new();

    let limits = SweLimits::default();
    let mut spent = 0.0f64;
    let mut stopped_early = false;

    for (i, inst) in instances.iter().enumerate() {
        if already.contains(&inst.instance_id) {
            continue;
        }
        if spent >= budget {
            eprintln!(
                "BUDGET REACHED: ${spent:.2} of ${budget:.2} after {i} instances — stopping."
            );
            stopped_early = true;
            break;
        }
        let call = |bare: &str, msgs: &serde_json::Value, tokens: u32| {
            firstpass_bench::live::anthropic_call_messages(
                &client,
                "https://api.anthropic.com",
                &api_key,
                bare,
                Some("You are an expert software engineer. Output ONLY a unified diff."),
                msgs,
                tokens,
            )
        };
        eprintln!(
            "[{i}/{}] {} (spent ${spent:.2})",
            instances.len(),
            inst.instance_id
        );
        // Charged whether the instance succeeds or fails: a failed instance still burned
        // provider tokens, and skipping it must not make that spend invisible to the cap.
        let mut inst_spent = 0.0f64;
        match run_instance(inst, &rungs, &limits, max_turns, &call, &mut inst_spent) {
            Ok(run) => {
                spent += inst_spent;
                if let (Ok(line), Ok(mut f)) = (
                    serde_json::to_string(&run.recorded),
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&ckpt_path),
                ) {
                    let _ = writeln!(f, "{line}");
                }
                consecutive_failures = 0;
                done.push(run.recorded);
            }
            Err(e) => {
                // Skip the instance, do not abort the benchmark. One instance timed out
                // deterministically and killed a 50-instance run twice at instance 13 — the
                // second time spending $0.00 before dying, so the retry bought nothing. A single
                // poison-pill instance must not decide how much of the sample gets measured.
                //
                // Consecutive failures still stop the run: if everything is failing, that is an
                // outage or a misconfiguration and continuing just burns money. Isolated
                // failures are counted and reported, because a silently shortened sample is the
                // dishonest version of this fix.
                spent += inst_spent;
                consecutive_failures += 1;
                skipped.push((inst.instance_id.clone(), e.to_string()));
                eprintln!(
                    "instance {} failed: {e} — SKIPPING ({consecutive_failures} in a row, \
                     {} skipped total, spent ${spent:.2})",
                    inst.instance_id,
                    skipped.len()
                );
                if should_stop_after_failures(consecutive_failures, MAX_CONSECUTIVE_FAILURES) {
                    eprintln!(
                        "{consecutive_failures} consecutive failures — stopping (spent ${spent:.2})"
                    );
                    stopped_early = true;
                    break;
                }
                continue;
            }
        }
    }

    let mt: Vec<_> = done.iter().map(RecordedTask::to_multiturn).collect();
    let result = evaluate(&mt, PreRegistered::default());
    let depths: Vec<u32> = done
        .iter()
        .flat_map(|t| t.turns.iter().map(|x| x.assistant_turns))
        .collect();
    let max_depth = depths.iter().copied().max().unwrap_or(0);

    if json {
        let doc = serde_json::json!({
            "instances": done.len(),
            "spent_usd": spent,
            "stopped_early": stopped_early,
            "skipped": skipped
                .iter()
                .map(|(id, e)| serde_json::json!({ "instance_id": id, "error": e }))
                .collect::<Vec<_>>(),
            "max_depth": max_depth,
            "result": result,
        });
        println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
    } else {
        println!("# Agentic SWE-bench — repository-scale multi-turn repair\n");
        println!("instances: {}", done.len());
        println!("turns:     {}", result.n_turns);
        println!("max depth: {max_depth}  (MBPP capped at 2; `High` needs >= 8)");
        println!("spend:     ${spent:.2} of ${budget:.2}");
        if stopped_early {
            println!("\n**stopped early** — numbers cover only what was bought.");
        }
        if !skipped.is_empty() {
            // Disclose the shortened sample. An unreported skip turns "50 instances" into
            // "however many happened to succeed" while the header still claims 50, which is the
            // dishonest failure mode this whole harness exists to avoid.
            println!(
                "\n**{} instance(s) skipped** after repeated provider failures — the sample below \
                 is smaller than requested, and these are not counted as failures because they \
                 were never measured:",
                skipped.len()
            );
            for (id, e) in &skipped {
                println!("- `{id}`: {e}");
            }
        }
        println!("\n| policy | success | $/success |");
        println!("|---|---|---|");
        println!(
            "| baseline (always start cheap) | {:.3} | ${:.5} |",
            result.baseline_success, result.baseline_cost_per_success
        );
        println!(
            "| trajectory-informed start | {:.3} | ${:.5} |",
            result.trajectory_success, result.trajectory_cost_per_success
        );
        println!(
            "\npaired quality delta: {:+.4}  CI [{:.4}, {:.4}]",
            result.delta_success, result.delta_success_ci.lo, result.delta_success_ci.hi
        );
        println!(
            "cost improvement:     {:+.1}%",
            result.cost_improvement * 100.0
        );
        println!("\n## Pre-registered verdict\n\n{}", result.verdict);
    }
}
