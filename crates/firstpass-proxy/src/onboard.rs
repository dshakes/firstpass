//! `firstpass onboard` — fully agentic onboarding (SPEC §0.2: the primary user is an agent, and
//! the human deserves the same one-command experience). Detect the environment, plan the exact
//! steps, execute them under `--apply`, and verify end-to-end — so "install firstpass and route
//! my agent through it" is one command, not a doc to follow.
//!
//! Structure: [`detect`] and [`plan`] are pure (injected lookups, no I/O) so the decision logic is
//! unit-tested offline; [`execute`] performs the side effects (spawn proxy, append one marked line
//! to the shell rc, probe `/healthz`) and is deliberately thin. Without `--apply` the command is a
//! dry run that prints the plan — agentic, but never surprising.

use std::io::Write as _;
use std::path::PathBuf;

/// Marker comment appended with the env line so re-running onboard is idempotent and offboarding
/// is greppable.
pub const RC_MARKER: &str = "# added by `firstpass onboard`";

/// What onboarding discovered about this machine. Everything is injected-lookup driven so tests
/// construct arbitrary environments without touching the real one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Environment {
    /// User's login shell binary name (`zsh` / `bash` / `fish` / other).
    pub shell: String,
    /// The proxy is already answering `/healthz` on the target port.
    pub proxy_running: bool,
    /// `ANTHROPIC_BASE_URL` already points at the target proxy.
    pub already_routed: bool,
    /// `ANTHROPIC_API_KEY` is set (needed for enforce-mode upstream calls; observe passes BYOK).
    pub has_api_key: bool,
    /// The `claude` CLI (Claude Code) is on PATH — we can offer MCP wiring.
    pub has_claude_cli: bool,
    /// Target bind, e.g. `127.0.0.1:8080`.
    pub bind: String,
}

/// Which provider the ladder opens on. Only `anthropic` and `openai` are built into the provider
/// registry; the other two must carry a `[[provider]]` block or the config will not resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// Built in.
    Anthropic,
    /// Built in.
    OpenAi,
    /// Gemini dialect; needs a `[[provider]]` block and `GEMINI_API_KEY`.
    Google,
    /// A local OpenAI-compatible server (Ollama), escalating to a frontier rung.
    Local,
}

impl Provider {
    /// Every choice, in prompt order.
    pub const ALL: [Self; 4] = [Self::Anthropic, Self::OpenAi, Self::Google, Self::Local];

    /// Stable id accepted by `--provider`.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Google => "google",
            Self::Local => "local",
        }
    }

    /// One-line description shown next to the option when prompting.
    #[must_use]
    pub const fn blurb(self) -> &'static str {
        match self {
            Self::Anthropic => "Claude — haiku opens, sonnet catches (built in)",
            Self::OpenAi => "GPT — 4.1-mini opens, 5.5 catches (built in)",
            Self::Google => "Gemini — flash opens, pro catches (needs GEMINI_API_KEY)",
            Self::Local => "Ollama on localhost, escalating to Claude sonnet",
        }
    }

    /// Cheapest-first ladder. Every id here is priced by the built-in table, so `firstpass savings`
    /// is meaningful out of the box (the local rung is the exception — it is free to run).
    #[must_use]
    pub const fn ladder(self) -> [&'static str; 2] {
        match self {
            Self::Anthropic => ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"],
            Self::OpenAi => ["openai/gpt-4.1-mini", "openai/gpt-5.5"],
            Self::Google => ["google/gemini-3.1-flash", "google/gemini-3.1-pro"],
            Self::Local => ["ollama/qwen2.5-coder:7b", "anthropic/claude-sonnet-5"],
        }
    }

    /// A model for the judge gate that is deliberately NOT on the ladder — the runner enforces
    /// maker != checker, so a judge drawn from the ladder would be rejected.
    #[must_use]
    pub const fn judge_model(self) -> &'static str {
        match self {
            Self::Anthropic | Self::Local => "anthropic/claude-opus-4-8",
            Self::OpenAi | Self::Google => "anthropic/claude-haiku-4-5",
        }
    }

    /// The `[[provider]]` block this choice requires, if it is not built in.
    #[must_use]
    pub const fn provider_block(self) -> Option<&'static str> {
        match self {
            Self::Anthropic | Self::OpenAi => None,
            Self::Google => Some(
                "[[provider]]                # only anthropic + openai are built in\n\
                 id          = \"google\"\n\
                 dialect     = \"gemini\"\n\
                 base_url    = \"https://generativelanguage.googleapis.com\"\n\
                 api_key_env = \"GEMINI_API_KEY\"\n",
            ),
            Self::Local => Some(
                "[[provider]]                # local rung; escalates to a frontier model\n\
                 id       = \"ollama\"\n\
                 dialect  = \"openai\"\n\
                 base_url = \"http://localhost:11434\"   # keyless\n",
            ),
        }
    }
}

/// What the model is being asked to produce. This is what actually picks the gate — the whole
/// product rests on gating the real output, so it is the one question worth asking carefully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Structured output — checked against a schema.
    Json,
    /// Code — checked by running the caller's tests.
    Code,
    /// Prose — graded by a judge on a different model.
    Prose,
    /// Mixed traffic — scored by k-sample self-consistency.
    Mixed,
}

impl Shape {
    /// Every choice, in prompt order.
    pub const ALL: [Self; 4] = [Self::Json, Self::Code, Self::Prose, Self::Mixed];

    /// Stable id accepted by `--shape`.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Code => "code",
            Self::Prose => "prose",
            Self::Mixed => "mixed",
        }
    }

    /// One-line description shown next to the option when prompting.
    #[must_use]
    pub const fn blurb(self) -> &'static str {
        match self {
            Self::Json => "JSON / API responses — schema gate, cheapest proof there is",
            Self::Code => "Code — your own test suite is the gate",
            Self::Prose => "Prose — an LLM judge grades it (maker != checker)",
            Self::Mixed => "Mixed — k-sample self-consistency scores agreement",
        }
    }

    /// Gate ids the route names. `non-empty` and `json-valid` are built in; the rest are declared
    /// by [`Self::gate_block`] below, because a route naming an undeclared gate will not resolve.
    #[must_use]
    pub const fn gates(self) -> [&'static str; 2] {
        match self {
            Self::Json => ["json-valid", "extract-shape"],
            Self::Code => ["json-valid", "unit-tests"],
            Self::Prose => ["non-empty", "judge"],
            Self::Mixed => ["non-empty", "uncertainty"],
        }
    }

    /// The `[[gate]]` block defining this shape's non-built-in gate.
    #[must_use]
    pub fn gate_block(self, provider: Provider) -> String {
        match self {
            Self::Json => "[[gate]]\n\
                 id         = \"extract-shape\"\n\
                 schema     = { type = \"object\", required = [\"id\", \"total\"] }\n\
                 on_abstain = \"fail_closed\"\n"
                .to_owned(),
            // Deliberately a placeholder, not `cargo test` / `npm test`: a gate is not just any
            // command. It must read the candidate as JSON on stdin and print
            // {"verdict":"pass|fail|abstain"} on stdout, so naming a real test runner here would
            // look wired up while silently abstaining on every call. `doctor` flags this until
            // it is replaced, which is the intended nudge.
            Self::Code => {
                "[[gate]]\n\
                 # REPLACE ME. A gate reads the candidate as JSON on stdin and prints\n\
                 #   {\"verdict\":\"pass|fail|abstain\", \"score\"?: 0.0-1.0, \"reason\"?: \"...\"}\n\
                 # on stdout. Wrap your real test command in that contract — a bare `cargo test`\n\
                 # or `npm test` will not work, it would abstain on every request.\n\
                 # `firstpass doctor` fails on this line until you point it at your wrapper.\n\
                 id  = \"unit-tests\"\n\
                 cmd = [\"your-test-runner\", \"--from-stdin\"]\n"
                    .to_owned()
            }
            Self::Prose => format!(
                "[[gate]]                    # judge sits OUTSIDE the ladder: maker != checker\n\
                 id    = \"judge\"\n\
                 judge = {{ model = \"{}\", threshold = 0.7, rubric = \"The response fully and \
                 correctly resolves the request, with no errors.\" }}\n",
                provider.judge_model()
            ),
            Self::Mixed => format!(
                "[[gate]]                    # k samples; agreement becomes the score\n\
                 id          = \"uncertainty\"\n\
                 consistency = {{ model = \"{}\", k = 3, threshold = 0.6 }}\n",
                provider.ladder()[0]
            ),
        }
    }
}

/// The three answers that decide a starting ladder. Everything else in the generated file follows
/// from these, which is why onboarding asks exactly three questions and not a wizard's worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LadderChoice {
    /// Which provider the ladder opens on.
    pub provider: Provider,
    /// What the output is, which picks the gate.
    pub shape: Shape,
    /// `observe` forwards unchanged; `enforce` serves from the cheap rung once a gate passes.
    pub mode: firstpass_core::Mode,
}

impl Default for LadderChoice {
    /// The answer set used when nothing is on a terminal to ask: Claude, JSON, and observe — the
    /// combination that changes no behavior at all.
    fn default() -> Self {
        Self {
            provider: Provider::Anthropic,
            shape: Shape::Json,
            mode: firstpass_core::Mode::Observe,
        }
    }
}

/// Render a complete, runnable `firstpass.toml` for these answers — including the `[[provider]]`
/// and `[[gate]]` blocks the choice requires. Emitting a fragment would be worse than emitting
/// nothing: only `anthropic`/`openai` are built-in providers and only `non-empty`/`json-valid`
/// are built-in gates, so anything else must be declared here or the file will not resolve.
#[must_use]
pub fn render_config(choice: &LadderChoice) -> String {
    let enforce = choice.mode == firstpass_core::Mode::Enforce;
    let mode = if enforce { "enforce" } else { "observe" };
    let quoted = |xs: [&str; 2]| -> String { format!("\"{}\", \"{}\"", xs[0], xs[1]) };
    let mut out = format!(
        "# firstpass.toml — written by `firstpass onboard`. Re-run onboard to regenerate.\n\
         #   FIRSTPASS_MODE={mode} FIRSTPASS_CONFIG=./firstpass.toml firstpass up\n\n"
    );
    if let Some(block) = choice.provider.provider_block() {
        out.push_str(block);
        out.push('\n');
    }
    out.push_str(&format!(
        "[[route]]                   # routes match top to bottom; first match wins\n\
         match  = {{}}                 # everything\n\
         mode   = \"{mode}\"\n\
         ladder = [{}]\n\
         gates  = [{}]\n\n",
        quoted(choice.provider.ladder()),
        quoted(choice.shape.gates()),
    ));
    out.push_str(&choice.shape.gate_block(choice.provider));
    if enforce {
        out.push_str(
            "\n[escalation]\nmax_rungs_per_request = 2   # one rung up, never a runaway\n",
        );
    } else {
        out.push_str(
            "\n# observe: every request is forwarded unchanged and a receipt is written off\n\
             # the hot path. Nothing routes differently until mode = \"enforce\".\n",
        );
    }
    out
}

/// Render every answer combination as a JavaScript asset for the documentation site.
///
/// The docs used to carry their own copy of this template in `docs/assets/docs.js`. Two
/// implementations of the same config format drift silently — and a docs page that emits config
/// the binary would reject is worse than a page with no builder at all. So Rust owns the format
/// and the site consumes what Rust produced; `docs_presets_are_current` fails CI if this output
/// and the committed asset disagree.
#[must_use]
pub fn presets_js() -> String {
    use firstpass_core::Mode;
    let mut out = String::from(
        "// GENERATED by `cargo test -p firstpass-proxy presets` — do not edit by hand.\n\
         // Source of truth: crates/firstpass-proxy/src/onboard.rs (render_config).\n\
         // Regenerate: FIRSTPASS_WRITE_PRESETS=1 cargo test -p firstpass-proxy presets\n\
         window.FP_LADDER_PRESETS = {\n",
    );
    for provider in Provider::ALL {
        out.push_str(&format!("  {:?}: {{\n", provider.id()));
        for shape in Shape::ALL {
            out.push_str(&format!("    {:?}: {{\n", shape.id()));
            for (mode, key) in [(Mode::Observe, "observe"), (Mode::Enforce, "enforce")] {
                let toml = render_config(&LadderChoice {
                    provider,
                    shape,
                    mode,
                });
                out.push_str(&format!("      {key:?}: {},\n", js_string(&toml)));
            }
            out.push_str("    },\n");
        }
        out.push_str("  },\n");
    }
    out.push_str("};\n");
    out
}

/// Encode a string as a JS double-quoted literal. Hand-rolled rather than pulling in a
/// serialiser: the payload is our own generated TOML, and the escape set is small and fixed.
fn js_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => {}
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// One onboarding step: what it is, and whether it's already satisfied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Write the generated routing config. Never overwrites an existing file — the caller checks.
    WriteConfig {
        /// Where the file goes (`./firstpass.toml`).
        path: PathBuf,
        /// Full rendered contents.
        toml: String,
    },
    /// Spawn `firstpass up` detached (observe mode), logging to `firstpass-proxy.log`.
    StartProxy {
        /// Config to hand the child via `FIRSTPASS_CONFIG`. The proxy has no default config path,
        /// so a written file is inert unless it is passed explicitly.
        config: Option<PathBuf>,
    },
    /// Append the marked `ANTHROPIC_BASE_URL` export to the shell rc file.
    WireShell {
        /// Rc file the line goes into.
        rc: PathBuf,
        /// The exact line (shell-dialect aware).
        line: String,
    },
    /// Print the `claude mcp add` command so Claude Code can query traces/savings as tools.
    /// (Printed, not executed — mutating another tool's config uninvited isn't onboarding.)
    SuggestClaudeMcp,
    /// Probe `/healthz` and `/v1/capabilities` and report what's routed.
    Verify,
    /// Nothing to do — with the reason shown to the user.
    AlreadyDone(&'static str),
}

/// Detect the environment via injected lookups: `env(key)` for env vars, `on_path(bin)` for PATH
/// probes, `healthz()` for a live proxy probe. Pure decision logic; all I/O behind the closures.
pub fn detect(
    env: impl Fn(&str) -> Option<String>,
    on_path: impl Fn(&str) -> bool,
    healthz: impl Fn() -> bool,
) -> Environment {
    let bind = env("FIRSTPASS_BIND").unwrap_or_else(|| "127.0.0.1:8080".to_owned());
    let base_url = format!("http://{bind}");
    let shell = env("SHELL")
        .and_then(|s| s.rsplit('/').next().map(str::to_owned))
        .unwrap_or_else(|| "sh".to_owned());
    Environment {
        shell,
        proxy_running: healthz(),
        already_routed: env("ANTHROPIC_BASE_URL").is_some_and(|v| v == base_url),
        has_api_key: env("ANTHROPIC_API_KEY").is_some(),
        has_claude_cli: on_path("claude"),
        bind,
    }
}

/// The shell-dialect line that routes agents through the proxy, plus the rc file it belongs in.
/// `home` is injected so tests never touch a real home directory.
#[must_use]
pub fn shell_wiring(shell: &str, home: &std::path::Path, bind: &str) -> (PathBuf, String) {
    let url = format!("http://{bind}");
    match shell {
        "fish" => (
            home.join(".config/fish/config.fish"),
            format!("set -gx ANTHROPIC_BASE_URL {url}  {RC_MARKER}"),
        ),
        "bash" => (
            home.join(".bashrc"),
            format!("export ANTHROPIC_BASE_URL={url}  {RC_MARKER}"),
        ),
        // zsh and anything else POSIX-ish: default to ~/.zshrc only for zsh, else ~/.profile.
        "zsh" => (
            home.join(".zshrc"),
            format!("export ANTHROPIC_BASE_URL={url}  {RC_MARKER}"),
        ),
        _ => (
            home.join(".profile"),
            format!("export ANTHROPIC_BASE_URL={url}  {RC_MARKER}"),
        ),
    }
}

/// A config the caller wants written: where it goes and the answers behind it. `None` means the
/// step is skipped — either a `firstpass.toml` is already there or the user opted out.
#[derive(Debug, Clone)]
pub struct ConfigPlan {
    /// Destination, normally `./firstpass.toml`.
    pub path: PathBuf,
    /// The three answers.
    pub choice: LadderChoice,
}

/// Build the ordered plan for this environment. `rc_already_wired` is whether the rc file already
/// carries the marker line (checked by the caller, injected here to stay pure). `config` is the
/// routing file to generate, if any — it is written *before* the proxy starts so the child can be
/// handed it, since a config the proxy never loads is worse than no config at all.
#[must_use]
pub fn plan(
    env: &Environment,
    home: &std::path::Path,
    rc_already_wired: bool,
    config: Option<&ConfigPlan>,
) -> Vec<Step> {
    let mut steps = Vec::new();
    if let Some(c) = config {
        steps.push(Step::WriteConfig {
            path: c.path.clone(),
            toml: render_config(&c.choice),
        });
    }
    if env.proxy_running {
        steps.push(Step::AlreadyDone("proxy already answering /healthz"));
    } else {
        steps.push(Step::StartProxy {
            config: config.map(|c| c.path.clone()),
        });
    }
    if env.already_routed || rc_already_wired {
        steps.push(Step::AlreadyDone("ANTHROPIC_BASE_URL already wired"));
    } else {
        let (rc, line) = shell_wiring(&env.shell, home, &env.bind);
        steps.push(Step::WireShell { rc, line });
    }
    if env.has_claude_cli {
        steps.push(Step::SuggestClaudeMcp);
    }
    steps.push(Step::Verify);
    steps
}

/// Render the plan for the dry run (default) or as the running commentary under `--apply`.
#[must_use]
pub fn render(env: &Environment, steps: &[Step], apply: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "detected: shell={} · proxy_running={} · routed={} · api_key={} · claude_cli={}\n\n",
        env.shell, env.proxy_running, env.already_routed, env.has_api_key, env.has_claude_cli
    ));
    for (i, s) in steps.iter().enumerate() {
        let n = i + 1;
        match s {
            Step::WriteConfig { path, toml } => {
                out.push_str(&format!("{n}. write {} —\n", path.display()));
                for line in toml.lines() {
                    out.push_str(&format!("     {line}\n"));
                }
            }
            Step::StartProxy { config } => {
                out.push_str(&format!(
                    "{n}. start the proxy — `firstpass up` (observe mode: watches, changes nothing), log → firstpass-proxy.log\n"
                ));
                if let Some(c) = config {
                    out.push_str(&format!(
                        "     with FIRSTPASS_CONFIG={} — the proxy has no default config path\n",
                        c.display()
                    ));
                }
            }
            Step::WireShell { rc, line } => out.push_str(&format!(
                "{n}. route your agents — append to {}:\n     {line}\n",
                rc.display()
            )),
            Step::SuggestClaudeMcp => out.push_str(&format!(
                "{n}. (optional) let Claude Code query receipts as tools:\n     claude mcp add firstpass -- firstpass mcp\n"
            )),
            Step::Verify => out.push_str(&format!(
                "{n}. verify — probe /healthz and /v1/capabilities, report what's routed\n"
            )),
            Step::AlreadyDone(why) => out.push_str(&format!("{n}. ✓ {why}\n")),
        }
    }
    if !env.has_api_key {
        out.push_str(
            "\nnote: ANTHROPIC_API_KEY is not set — observe mode passes your agent's own key \
             through (BYOK), so this only matters for enforce mode.\n",
        );
    }
    if !apply {
        out.push_str(
            "\ndry run — nothing changed. Re-run with `firstpass onboard --apply` to execute.\n",
        );
    }
    out
}

/// Execute the side-effectful steps. Returns a human report. Failures are reported per step —
/// onboarding never half-dies silently.
///
/// # Errors
/// Only on I/O failures writing the rc file; every other issue is reported in the returned text.
pub fn execute(env: &Environment, steps: &[Step]) -> Result<String, std::io::Error> {
    let mut out = String::new();
    for s in steps {
        match s {
            Step::WriteConfig { path, toml } => {
                // Never clobber a config the operator already has; the caller only plans this
                // step when the path is free, but re-check rather than trust the gap between.
                if path.exists() {
                    out.push_str(&format!(
                        "✓ {} already present — left untouched\n",
                        path.display()
                    ));
                } else {
                    std::fs::write(path, toml)?;
                    out.push_str(&format!("✓ wrote {}\n", path.display()));
                }
            }
            Step::StartProxy { config } => {
                let log = std::fs::File::create("firstpass-proxy.log")?;
                let exe = std::env::current_exe()?;
                let mut cmd = std::process::Command::new(exe);
                cmd.arg("up");
                if let Some(c) = config {
                    // A written config is inert unless it is named: `ProxyConfig::from_env` reads
                    // `FIRSTPASS_CONFIG` and has no default path to fall back on.
                    cmd.env("FIRSTPASS_CONFIG", c);
                }
                let child = cmd
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::from(log.try_clone()?))
                    .stderr(std::process::Stdio::from(log))
                    .spawn();
                match child {
                    Ok(c) => {
                        // Pidfile makes `firstpass offboard` able to stop what onboard started.
                        let _ = std::fs::write("firstpass-proxy.pid", c.id().to_string());
                        out.push_str(&format!(
                            "✓ proxy started (pid {}, observe mode) — log: firstpass-proxy.log\n",
                            c.id()
                        ));
                    }
                    Err(e) => out.push_str(&format!("✗ could not start proxy: {e}\n")),
                }
            }
            Step::WireShell { rc, line } => {
                if let Some(parent) = rc.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(rc)?;
                writeln!(f, "{line}")?;
                out.push_str(&format!(
                    "✓ wired {} — takes effect in new shells; for this one:\n     {}\n",
                    rc.display(),
                    line.trim_end_matches(RC_MARKER).trim_end()
                ));
            }
            Step::SuggestClaudeMcp => {
                out.push_str("→ optional: claude mcp add firstpass -- firstpass mcp\n");
            }
            Step::Verify => {
                let url = format!("http://{}/healthz", env.bind);
                let ok = wait_healthz(&url, std::time::Duration::from_secs(6));
                if ok {
                    out.push_str(&format!(
                        "✓ verified — proxy healthy at http://{} · capabilities: http://{}/v1/capabilities\n",
                        env.bind, env.bind
                    ));
                } else {
                    out.push_str(&format!(
                        "✗ proxy not answering http://{} after 6s — check firstpass-proxy.log\n",
                        env.bind
                    ));
                }
            }
            Step::AlreadyDone(why) => out.push_str(&format!("✓ {why}\n")),
        }
    }
    out.push_str("\noffboard any time: unset ANTHROPIC_BASE_URL (and remove the marked rc line)\n");
    Ok(out)
}

/// Poll `/healthz` until it answers 200 or the deadline passes. Blocking + std-only (the CLI calls
/// this off the async runtime): a plain TCP connect + minimal HTTP/1.1 GET, no client dependency.
fn wait_healthz(url: &str, deadline: std::time::Duration) -> bool {
    let Some(addr) = url
        .strip_prefix("http://")
        .and_then(|r| r.split('/').next())
        .map(str::to_owned)
    else {
        return false;
    };
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        if let Ok(mut s) = std::net::TcpStream::connect(&addr) {
            let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(500)));
            let req = format!("GET /healthz HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
            if s.write_all(req.as_bytes()).is_ok() {
                let mut buf = [0u8; 64];
                use std::io::Read as _;
                if let Ok(n) = s.read(&mut buf)
                    && n > 0
                    && String::from_utf8_lossy(&buf[..n]).contains("200")
                {
                    return true;
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    false
}

/// Whether the rc file already carries the onboard marker (idempotence check for [`plan`]).
#[must_use]
pub fn rc_wired(rc: &std::path::Path) -> bool {
    std::fs::read_to_string(rc).is_ok_and(|s| s.contains(RC_MARKER))
}

/// Remove every marked onboard line from `rc`. Returns whether anything was removed. Only lines
/// carrying [`RC_MARKER`] are touched — nothing else in the user's rc is ever rewritten.
///
/// # Errors
/// Only on I/O failures reading/writing the rc file.
pub fn offboard_rc(rc: &std::path::Path) -> Result<bool, std::io::Error> {
    let Ok(content) = std::fs::read_to_string(rc) else {
        return Ok(false); // no file → nothing to offboard
    };
    if !content.contains(RC_MARKER) {
        return Ok(false);
    }
    let kept: Vec<&str> = content.lines().filter(|l| !l.contains(RC_MARKER)).collect();
    std::fs::write(rc, kept.join("\n") + "\n")?;
    Ok(true)
}

/// `firstpass offboard` — the exact mirror of onboard: strip the marked line from every candidate
/// rc file, stop the proxy onboard started (pidfile), and print the one command for this shell.
/// Fully idempotent; reports what it found either way.
///
/// # Errors
/// Only on rc-file I/O failures.
pub fn offboard(home: &std::path::Path) -> Result<String, std::io::Error> {
    let mut out = String::new();
    for rc in [
        home.join(".zshrc"),
        home.join(".bashrc"),
        home.join(".profile"),
        home.join(".config/fish/config.fish"),
    ] {
        if offboard_rc(&rc)? {
            out.push_str(&format!("✓ removed firstpass line from {}\n", rc.display()));
        }
    }
    // Stop the proxy onboard spawned, if its pidfile is present.
    if let Ok(pid) = std::fs::read_to_string("firstpass-proxy.pid") {
        let pid = pid.trim().to_owned();
        #[cfg(unix)]
        {
            let killed = std::process::Command::new("kill")
                .arg(&pid)
                .status()
                .is_ok_and(|s| s.success());
            if killed {
                out.push_str(&format!("✓ stopped proxy (pid {pid})\n"));
            } else {
                out.push_str(&format!(
                    "→ proxy pid {pid} not running (already stopped)\n"
                ));
            }
        }
        let _ = std::fs::remove_file("firstpass-proxy.pid");
    }
    if out.is_empty() {
        out.push_str("nothing to offboard — no marked rc lines, no pidfile.\n");
    }
    out.push_str("for this shell: unset ANTHROPIC_BASE_URL\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(a, _)| *a == k)
                .map(|(_, v)| (*v).to_owned())
        }
    }

    #[test]
    fn detect_reads_shell_routing_and_tools() {
        let e = detect(
            env_of(&[
                ("SHELL", "/bin/zsh"),
                ("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080"),
                ("ANTHROPIC_API_KEY", "sk-x"),
            ]),
            |bin| bin == "claude",
            || true,
        );
        assert_eq!(e.shell, "zsh");
        assert!(e.proxy_running && e.already_routed && e.has_api_key && e.has_claude_cli);
        assert_eq!(e.bind, "127.0.0.1:8080");
    }

    #[test]
    fn detect_respects_custom_bind_and_mismatched_base_url() {
        let e = detect(
            env_of(&[
                ("FIRSTPASS_BIND", "127.0.0.1:9999"),
                ("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080"), // points elsewhere
            ]),
            |_| false,
            || false,
        );
        assert_eq!(e.bind, "127.0.0.1:9999");
        assert!(
            !e.already_routed,
            "routed to a different port is not routed"
        );
    }

    #[test]
    fn shell_wiring_speaks_each_dialect() {
        let home = std::path::Path::new("/home/u");
        let (rc, line) = shell_wiring("fish", home, "127.0.0.1:8080");
        assert!(rc.ends_with(".config/fish/config.fish"));
        assert!(line.starts_with("set -gx ANTHROPIC_BASE_URL http://127.0.0.1:8080"));

        let (rc, line) = shell_wiring("zsh", home, "127.0.0.1:8080");
        assert!(rc.ends_with(".zshrc"));
        assert!(line.starts_with("export ANTHROPIC_BASE_URL="));

        let (rc, _) = shell_wiring("dash", home, "127.0.0.1:8080");
        assert!(
            rc.ends_with(".profile"),
            "unknown shells fall back to .profile"
        );
    }

    #[test]
    fn plan_covers_fresh_machine_and_is_idempotent_when_done() {
        let home = std::path::Path::new("/home/u");
        let fresh = Environment {
            shell: "zsh".into(),
            proxy_running: false,
            already_routed: false,
            has_api_key: false,
            has_claude_cli: true,
            bind: "127.0.0.1:8080".into(),
        };
        let steps = plan(&fresh, home, false, None);
        assert!(matches!(steps[0], Step::StartProxy { .. }));
        assert!(matches!(steps[1], Step::WireShell { .. }));
        assert!(matches!(steps[2], Step::SuggestClaudeMcp));
        assert!(matches!(steps.last(), Some(Step::Verify)));

        // Everything already set up → only AlreadyDone + Verify, nothing mutating.
        let done = Environment {
            proxy_running: true,
            already_routed: true,
            has_claude_cli: false,
            ..fresh
        };
        let steps = plan(&done, home, true, None);
        assert!(
            steps
                .iter()
                .all(|s| matches!(s, Step::AlreadyDone(_) | Step::Verify))
        );
    }

    #[test]
    fn render_dry_run_says_nothing_changed_and_flags_missing_key() {
        let home = std::path::Path::new("/home/u");
        let e = Environment {
            shell: "bash".into(),
            proxy_running: false,
            already_routed: false,
            has_api_key: false,
            has_claude_cli: false,
            bind: "127.0.0.1:8080".into(),
        };
        let text = render(&e, &plan(&e, home, false, None), false);
        assert!(text.contains("dry run — nothing changed"));
        assert!(text.contains("ANTHROPIC_API_KEY is not set"));
        assert!(text.contains(".bashrc"));
    }

    #[test]
    fn rc_wired_detects_the_marker_and_execute_appends_it_once() {
        let dir = std::env::temp_dir().join(format!("fp-onboard-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let rc = dir.join(".zshrc");
        assert!(!rc_wired(&rc), "missing file is not wired");

        let e = Environment {
            shell: "zsh".into(),
            proxy_running: true, // no spawn in this test
            already_routed: false,
            has_api_key: true,
            has_claude_cli: false,
            bind: "127.0.0.1:1".into(), // verify step will fail fast — that's fine, we assert wiring
        };
        let (rc_path, line) = shell_wiring("zsh", &dir, &e.bind);
        let steps = vec![Step::WireShell {
            rc: rc_path.clone(),
            line,
        }];
        let report = execute(&e, &steps).unwrap();
        assert!(report.contains("✓ wired"));
        assert!(rc_wired(&rc_path), "marker written");
        // Re-planning with the marker present must not wire again (idempotent onboarding).
        let steps = plan(&e, &dir, rc_wired(&rc_path), None);
        assert!(!steps.iter().any(|s| matches!(s, Step::WireShell { .. })));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn offboard_removes_only_the_marked_line_and_is_idempotent() {
        let dir = std::env::temp_dir().join(format!("fp-offboard-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let rc = dir.join(".zshrc");
        std::fs::write(
            &rc,
            format!("alias ll='ls -l'\nexport ANTHROPIC_BASE_URL=http://127.0.0.1:8080  {RC_MARKER}\nexport EDITOR=vim\n"),
        )
        .unwrap();

        assert!(offboard_rc(&rc).unwrap(), "marked line removed");
        let after = std::fs::read_to_string(&rc).unwrap();
        assert!(!after.contains(RC_MARKER));
        assert!(
            after.contains("alias ll") && after.contains("EDITOR=vim"),
            "user lines untouched"
        );
        assert!(!offboard_rc(&rc).unwrap(), "second offboard is a no-op");

        // The full offboard reports the rc removal and the unset line.
        std::fs::write(&rc, format!("x  {RC_MARKER}\n")).unwrap();
        let report = offboard(&dir).unwrap();
        assert!(report.contains("removed firstpass line"));
        assert!(report.contains("unset ANTHROPIC_BASE_URL"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The whole point of generating a config is that it runs. Every answer combination must
    /// survive the real parser — including the `[[provider]]`/`[[gate]]` blocks each one implies,
    /// since only `anthropic`/`openai` and `non-empty`/`json-valid` are built in.
    #[test]
    fn every_generated_config_parses_and_resolves() {
        use firstpass_core::Mode;
        let mut n = 0;
        for provider in Provider::ALL {
            for shape in Shape::ALL {
                for mode in [Mode::Observe, Mode::Enforce] {
                    n += 1;
                    let choice = LadderChoice {
                        provider,
                        shape,
                        mode,
                    };
                    let toml = render_config(&choice);
                    let cfg = firstpass_core::Config::parse(&toml).unwrap_or_else(|e| {
                        panic!(
                            "{}/{}/{mode:?} did not parse: {e}\n{toml}",
                            provider.id(),
                            shape.id()
                        )
                    });
                    let route = &cfg.routes[0];
                    assert_eq!(
                        route.mode, mode,
                        "route mode is what actually gates enforcement"
                    );
                    assert_eq!(
                        route.ladder.len(),
                        2,
                        "a ladder needs somewhere to escalate to"
                    );

                    // Every gate the route names must be built in or declared here.
                    for g in &route.gates {
                        let builtin = g == "non-empty" || g == "json-valid";
                        assert!(
                            builtin || cfg.gate_defs.iter().any(|d| &d.id == g),
                            "{}/{}: gate {g:?} is neither built in nor declared",
                            provider.id(),
                            shape.id()
                        );
                    }
                    // Same for every provider a rung names.
                    for rung in &route.ladder {
                        let pid = rung.split('/').next().unwrap();
                        let builtin = pid == "anthropic" || pid == "openai";
                        assert!(
                            builtin || cfg.providers.iter().any(|d| d.id == pid),
                            "{}/{}: provider {pid:?} is neither built in nor declared",
                            provider.id(),
                            shape.id()
                        );
                    }
                    // maker != checker: a judge drawn from the ladder would be rejected at runtime.
                    if let Some(j) = cfg.gate_defs.iter().find_map(|d| d.judge.as_ref()) {
                        assert!(
                            !route.ladder.contains(&j.model),
                            "{}: judge {} is on its own ladder",
                            provider.id(),
                            j.model
                        );
                    }
                    // Observe must stay inert — an escalation cap implies enforcement.
                    assert_eq!(
                        toml.contains("[escalation]"),
                        mode == Mode::Enforce,
                        "escalation block should track enforce only"
                    );
                }
            }
        }
        assert_eq!(n, 32, "4 providers x 4 shapes x 2 modes");
    }

    /// A written config the proxy never loads is worse than no config: `ProxyConfig::from_env`
    /// reads `FIRSTPASS_CONFIG` and has no default path, so the plan must carry it to the child.
    #[test]
    fn planned_config_is_written_before_the_proxy_starts_and_is_handed_to_it() {
        let home = std::path::Path::new("/home/u");
        let env = Environment {
            shell: "zsh".into(),
            proxy_running: false,
            already_routed: false,
            has_api_key: true,
            has_claude_cli: false,
            bind: "127.0.0.1:8080".into(),
        };
        let cfg = ConfigPlan {
            path: PathBuf::from("firstpass.toml"),
            choice: LadderChoice::default(),
        };
        let steps = plan(&env, home, false, Some(&cfg));
        assert!(
            matches!(&steps[0], Step::WriteConfig { path, .. } if path == &cfg.path),
            "config is written first, before anything reads it"
        );
        assert!(
            matches!(&steps[1], Step::StartProxy { config: Some(p) } if p == &cfg.path),
            "the spawned proxy is handed the config explicitly"
        );

        // Without a config the plan is exactly what it always was.
        let steps = plan(&env, home, false, None);
        assert!(matches!(steps[0], Step::StartProxy { config: None }));
        assert!(!steps.iter().any(|s| matches!(s, Step::WriteConfig { .. })));
    }

    /// Re-running onboard must never clobber a routing file the operator already edited.
    #[test]
    fn write_config_refuses_to_overwrite_an_existing_file() {
        let dir = std::env::temp_dir().join(format!("fp-cfg-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("firstpass.toml");
        std::fs::write(&path, "# hand-tuned, do not touch\n").unwrap();
        let env = Environment {
            shell: "zsh".into(),
            proxy_running: true, // no spawn
            already_routed: true,
            has_api_key: true,
            has_claude_cli: false,
            bind: "127.0.0.1:1".into(),
        };
        let steps = vec![Step::WriteConfig {
            path: path.clone(),
            toml: render_config(&LadderChoice::default()),
        }];
        let report = execute(&env, &steps).unwrap();
        assert!(report.contains("already present"));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "# hand-tuned, do not touch\n",
            "existing config survived untouched"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The dry run has to show the actual file, or "onboard asks then writes" is unverifiable
    /// before the fact.
    #[test]
    fn dry_run_shows_the_config_it_would_write() {
        let home = std::path::Path::new("/home/u");
        let env = Environment {
            shell: "bash".into(),
            proxy_running: false,
            already_routed: false,
            has_api_key: true,
            has_claude_cli: false,
            bind: "127.0.0.1:8080".into(),
        };
        let cfg = ConfigPlan {
            path: PathBuf::from("firstpass.toml"),
            choice: LadderChoice {
                provider: Provider::Local,
                shape: Shape::Code,
                mode: firstpass_core::Mode::Enforce,
            },
        };
        let text = render(&env, &plan(&env, home, false, Some(&cfg)), false);
        assert!(text.contains("write firstpass.toml"));
        assert!(text.contains("ollama"), "provider block is shown");
        assert!(text.contains("unit-tests"), "gate block is shown");
        assert!(
            text.contains("FIRSTPASS_CONFIG="),
            "explains how the proxy finds it"
        );
        assert!(text.contains("dry run — nothing changed"));
    }

    /// The documentation site must not carry a second implementation of the config format.
    /// This pins the committed asset to what Rust generates; if they diverge, CI fails here
    /// rather than the site quietly shipping config the binary would reject.
    #[test]
    fn docs_presets_are_current() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/assets/ladder-presets.js");
        let generated = presets_js();

        // Regeneration is an explicit opt-in so a normal test run can never rewrite the repo.
        if std::env::var("FIRSTPASS_WRITE_PRESETS").is_ok() {
            std::fs::write(&path, &generated).expect("write presets");
            return;
        }

        let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e}\nregenerate with:\n  \
                 FIRSTPASS_WRITE_PRESETS=1 cargo test -p firstpass-proxy presets",
                path.display()
            )
        });
        assert_eq!(
            committed.replace("\r\n", "\n"),
            generated,
            "docs/assets/ladder-presets.js is stale — the docs builder and `firstpass onboard` \
             would emit different config.\nregenerate with:\n  \
             FIRSTPASS_WRITE_PRESETS=1 cargo test -p firstpass-proxy presets"
        );
    }

    /// Every preset the site can hand a user has to parse, for the same reason onboard's output
    /// does — the builder's whole value is that its output runs.
    #[test]
    fn every_docs_preset_parses() {
        let js = presets_js();
        let mut n = 0;
        for raw in js.split("\": \"").skip(1) {
            let lit = &raw[..raw.find("\",\n").unwrap_or(raw.len())];
            let toml = lit
                .replace("\\n", "\n")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
            if !toml.contains("[[route]]") {
                continue;
            }
            n += 1;
            firstpass_core::Config::parse(&toml)
                .unwrap_or_else(|e| panic!("preset #{n} does not parse: {e}\n{toml}"));
        }
        assert_eq!(n, 32, "expected all 32 presets to be checked, saw {n}");
    }
}
