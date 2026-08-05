//! `firstpass launch <agent> [args…]` — start a coding agent already pointed at this proxy.
//!
//! The env-var swap is one line, so a subcommand that only exported it would be a convenience and
//! nothing more. What it actually buys is the **precondition**: pointing an agent at a proxy that
//! is not listening produces a connection error from inside the agent's own error surface, which
//! reads like the agent is broken rather than like the proxy is down. So this refuses to launch
//! until `/healthz` answers, and says which of the two is wrong.
//!
//! It also refuses when something answers that is **not** Firstpass. A 200 on `/healthz` only
//! proves the port is held; whatever else is listening on 8080 during development will happily
//! answer it. Handing an agent that address would route every request past the gate to a stranger
//! that may well reply — a failure that looks like success, which is the worse one.
//!
//! Uses `exec` on unix rather than spawning a child: the agent inherits this process's PID and
//! terminal directly, so job control, `Ctrl-C`, and raw-mode TUIs behave exactly as they do when
//! the agent is run by hand. A wrapper process in between breaks all three.

use std::collections::HashMap;
use std::process::Command;

/// A launchable agent: which binary to run, and which base-url variable points it at a proxy.
struct Agent {
    /// The `firstpass launch <name>` name.
    name: &'static str,
    /// The executable to run, if it is not the same as `name`.
    bin: &'static str,
    /// Base-url env vars to set. More than one when the agent can talk to several vendors and we
    /// want every one of them routed — a half-routed agent silently bypasses the gate.
    vars: &'static [&'static str],
}

const AGENTS: &[Agent] = &[
    Agent {
        name: "claude",
        bin: "claude",
        vars: &["ANTHROPIC_BASE_URL"],
    },
    Agent {
        name: "codex",
        bin: "codex",
        vars: &["OPENAI_BASE_URL"],
    },
    // Anything OpenAI-compatible: `firstpass launch openai -- <cmd> [args…]`.
    Agent {
        name: "openai",
        bin: "",
        vars: &["OPENAI_BASE_URL"],
    },
];

/// Print a multi-line diagnostic and exit non-zero.
///
/// Returning `Box<dyn Error>` from `main` renders it with `Debug`, which turns every newline in a
/// message into a literal `\n` — fine for a one-line error, unreadable for guidance that includes
/// a command to run. These messages exist to be followed, so they are printed as written.
fn fail(msg: &str) -> ! {
    eprintln!("firstpass launch: {msg}");
    std::process::exit(2)
}

/// Resolve the proxy's URL from the same variable the server binds on, so `launch` and `up` cannot
/// disagree about where it is.
#[must_use]
pub fn base_url() -> String {
    url_for_bind(&std::env::var("FIRSTPASS_BIND").unwrap_or_else(|_| "127.0.0.1:8080".to_owned()))
}

/// The client-reachable URL for a bind address. Pure, so the wildcard case is testable without
/// touching process environment.
fn url_for_bind(bind: &str) -> String {
    // A wildcard bind is not an address a client can connect to.
    let host = bind
        .strip_prefix("0.0.0.0")
        .map_or_else(|| bind.to_owned(), |port| format!("127.0.0.1{port}"));
    format!("http://{host}")
}

/// Split `args` (everything after `launch`) into the agent name and the argv to hand the agent.
/// A bare `--` separates our arguments from the agent's, so `firstpass launch claude -- -p 'hi'`
/// passes `-p hi` through untouched.
fn split(args: &[String]) -> (Option<&str>, Vec<String>) {
    let mut rest = args.iter();
    let name = rest.next().map(String::as_str);
    let passthrough: Vec<String> = rest
        .skip_while(|a| a.as_str() == "--")
        .map(Clone::clone)
        .collect();
    (name, passthrough)
}

/// The env the agent should run with: every base-url var this agent needs, pointed at `url`.
fn env_for(agent: &Agent, url: &str) -> HashMap<String, String> {
    agent
        .vars
        .iter()
        .map(|v| ((*v).to_owned(), url.to_owned()))
        .collect()
}

/// What is listening at `url`.
#[derive(Debug, PartialEq, Eq)]
enum Listener {
    /// Firstpass answered.
    Firstpass,
    /// Something answered, but it is not Firstpass. Routing an agent into it would send real
    /// traffic to a stranger — a worse outcome than nothing listening, because it can look like it
    /// is working.
    Foreign,
    /// Nothing reachable.
    None,
}

/// Decide from a `/healthz` body whether we are talking to Firstpass. Split out so the
/// discrimination is testable without a live port.
fn classify(status_ok: bool, body: Option<&serde_json::Value>) -> Listener {
    if !status_ok {
        return Listener::None;
    }
    match body.and_then(|b| b.get("service")).and_then(|s| s.as_str()) {
        Some("firstpass") => Listener::Firstpass,
        _ => Listener::Foreign,
    }
}

/// Probe `url` and say what is there.
async fn probe(url: &str) -> Listener {
    let Ok(client) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    else {
        return Listener::None;
    };
    let Ok(resp) = client.get(format!("{url}/healthz")).send().await else {
        return Listener::None;
    };
    let ok = resp.status().is_success();
    let body = resp.json::<serde_json::Value>().await.ok();
    classify(ok, body.as_ref())
}

/// Run `firstpass launch …`.
///
/// # Errors
/// When no agent is named, the agent is unknown, the proxy is not listening, or the agent binary
/// cannot be executed.
pub async fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (name, passthrough) = split(args);
    let Some(name) = name else {
        let names: Vec<&str> = AGENTS.iter().map(|a| a.name).collect();
        fail(&format!(
            "firstpass launch <agent> [-- args…]\n  agents: {}\n\n\
             `openai` needs an explicit command: firstpass launch openai -- <cmd> [args…]",
            names.join(", ")
        ));
    };
    let Some(agent) = AGENTS.iter().find(|a| a.name == name) else {
        let names: Vec<&str> = AGENTS.iter().map(|a| a.name).collect();
        fail(&format!(
            "unknown agent {name:?} (known: {})",
            names.join(", ")
        ));
    };

    // The binary is the agent's own, unless this is the generic form, where the caller supplies it
    // after `--`.
    let (bin, argv) = if agent.bin.is_empty() {
        let Some((first, rest)) = passthrough.split_first() else {
            fail(&format!(
                "`firstpass launch {name}` needs a command:\n\n    firstpass launch {name} -- <cmd> [args…]"
            ));
        };
        (first.clone(), rest.to_vec())
    } else {
        (agent.bin.to_owned(), passthrough)
    };

    let url = base_url();
    match probe(&url).await {
        Listener::Firstpass => {}
        Listener::None => {
            fail(&format!(
                "no proxy is listening at {url} — start one first:\n\n    firstpass up\n\n\
                 (or `firstpass kiosk` to configure and start it in one press).\n\n\
                 Launching into a proxy that is not up produces a connection error from inside \
                 {bin}, which reads like {bin} is broken."
            ));
        }
        Listener::Foreign => {
            fail(&format!(
                "something is listening at {url}, but it is not firstpass.\n\n\
                 Refusing to launch {bin} against it: every request would go to that service \
                 instead of through the gate, and it may well answer, so nothing would look \
                 wrong.\n\n\
                 Free the port, or point firstpass elsewhere and use the same address here:\n\n    \
                 FIRSTPASS_BIND=127.0.0.1:8099 firstpass up\n    \
                 FIRSTPASS_BIND=127.0.0.1:8099 firstpass launch {name}"
            ));
        }
    }

    let env = env_for(agent, &url);
    for (k, v) in &env {
        eprintln!("firstpass: {k}={v}");
    }
    eprintln!("firstpass: exec {bin}");

    let mut cmd = Command::new(&bin);
    cmd.args(&argv).envs(&env);

    #[cfg(unix)]
    {
        // Replaces this process: the agent gets the terminal and PID directly, so Ctrl-C and TUI
        // raw mode behave as if it had been run by hand. Only returns on failure.
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        Err(format!("could not exec {bin}: {err}. Is it installed and on PATH?").into())
    }
    #[cfg(not(unix))]
    {
        let status = cmd
            .status()
            .map_err(|e| format!("could not run {bin}: {e}. Is it installed and on PATH?"))?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wildcard_bind_becomes_a_connectable_loopback_address() {
        // 0.0.0.0 is a bind address, not somewhere a client can connect to. Handing it to an agent
        // verbatim fails on some stacks and is never what was meant.
        assert_eq!(url_for_bind("0.0.0.0:9090"), "http://127.0.0.1:9090");
        assert_eq!(url_for_bind("127.0.0.1:7777"), "http://127.0.0.1:7777");
    }

    #[test]
    fn a_foreign_service_holding_the_port_is_not_mistaken_for_firstpass() {
        // Found by running this for real: an unrelated dev server was on :8080, answered /healthz
        // 200, and the first version of this check launched an agent straight into it. Every
        // request would have bypassed the gate while appearing to work.
        let foreign = serde_json::json!({ "status": "ok", "llmMode": "api" });
        assert_eq!(classify(true, Some(&foreign)), Listener::Foreign);

        let ours =
            serde_json::json!({ "status": "ok", "service": "firstpass", "version": "0.3.0" });
        assert_eq!(classify(true, Some(&ours)), Listener::Firstpass);

        // A body-less or unhealthy answer is not proof of anything.
        assert_eq!(classify(false, Some(&ours)), Listener::None);
        assert_eq!(classify(true, None), Listener::Foreign);
    }

    #[test]
    fn a_bare_double_dash_separates_our_args_from_the_agents() {
        let args: Vec<String> = ["claude", "--", "-p", "hi"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let (name, rest) = split(&args);
        assert_eq!(name, Some("claude"));
        assert_eq!(rest, vec!["-p".to_owned(), "hi".to_owned()]);
    }

    #[test]
    fn agent_args_survive_without_a_separator_too() {
        let args: Vec<String> = ["claude", "-p", "hi"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let (_, rest) = split(&args);
        assert_eq!(rest, vec!["-p".to_owned(), "hi".to_owned()]);
    }

    #[test]
    fn each_agent_routes_the_vendor_it_actually_speaks() {
        // A wrong variable here is the worst failure mode this module has: the agent starts fine,
        // talks straight to the vendor, and every call silently bypasses the gate.
        let claude = AGENTS.iter().find(|a| a.name == "claude").expect("claude");
        assert_eq!(
            env_for(claude, "http://127.0.0.1:8080").get("ANTHROPIC_BASE_URL"),
            Some(&"http://127.0.0.1:8080".to_owned())
        );
        let codex = AGENTS.iter().find(|a| a.name == "codex").expect("codex");
        assert_eq!(
            env_for(codex, "http://127.0.0.1:8080").get("OPENAI_BASE_URL"),
            Some(&"http://127.0.0.1:8080".to_owned())
        );
    }
}
