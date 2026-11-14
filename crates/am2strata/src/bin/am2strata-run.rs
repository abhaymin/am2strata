//! am2strata-run: the launcher (it replaced run.py; am2Strata has no Python).
//!
//!   cargo run -q --release --bin am2strata-run -- serve --port 8001          in the foreground
//!   cargo run -q --release --bin am2strata-run -- serve --take-over --port 8000
//!                                       stop the PolyStrata service while it runs (the GPU fits one engine)
//!   cargo run -q --release --bin am2strata-run -- temp-service --port 8000
//!                                       a temporary user service in place of PolyStrata's, which comes back when
//!                                       it stops (however it stops: a crash, stop-temp-service, a reboot)
//!   cargo run -q --release --bin am2strata-run -- stop-temp-service
//!   cargo run -q --release --bin am2strata-run -- install --port 8000
//!                                       make am2Strata the service that starts at boot in place of strata-orca
//!   cargo run -q --release --bin am2strata-run -- uninstall      give the boot service back to strata-orca
//!   cargo run -q --release --bin am2strata-run -- test           the whole test suite (no GPU)
//!   cargo run -q --release --bin am2strata-run -- pull unsloth/Qwen3.6-35B-A3B-GGUF            list its quantizations
//!   cargo run -q --release --bin am2strata-run -- pull unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_XL
//!                                       download (token optional: --token, HF_TOKEN or the hf login file), check, and
//!                                       write its run config when an engine here runs its architecture
//!
//! Everything after `--` on serve / temp-service / install goes to the am2strata server (`am2strata --help`).

#[path = "../agents.rs"]
mod agents;

use clap::{Args, Parser, Subcommand};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

const SERVICE: &str = "strata-orca";
const TEMP_UNIT: &str = "am2strata";
const BOOT_UNIT: &str = "am2strata-boot";

#[derive(Parser)]
#[command(name = "am2strata-run", about = "Run am2Strata: in the foreground, as a temporary service, or as the boot service")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// run in the foreground (Ctrl+C stops it)
    Serve {
        #[command(flatten)]
        s: ServerArgs,
        /// stop the PolyStrata service while this runs, start it again after
        #[arg(long)]
        take_over: bool,
    },
    /// run as the temporary user service am2strata in place of PolyStrata's (restored when it stops)
    TempService {
        #[command(flatten)]
        s: ServerArgs,
    },
    /// stop the temporary service; PolyStrata's starts again
    StopTempService,
    /// make am2Strata the service that starts at boot, in place of strata-orca
    Install {
        #[command(flatten)]
        s: ServerArgs,
    },
    /// remove am2Strata's boot service and enable strata-orca again
    Uninstall,
    /// run the whole test suite (no GPU)
    Test,
    /// download a model from Hugging Face and write its run config (no --quant: list what the repository offers)
    Pull(PullArgs),
    /// coding agents on am2Strata, this PC's or another's: list | start <agent> | setup <agent> | reset <agent> | show <agent>
    Agent(AgentArgs),
    /// run a coding agent on am2Strata once, as `unsloth start <agent>` does: `agentrun opencode`,
    /// `agentrun claude --model tiel --continue`; --model, --small-model, --url, --api-key and --dry-run are
    /// am2Strata's, every other argument goes to the agent
    #[command(disable_help_flag = true)]
    Agentrun {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 0..)]
        args: Vec<String>,
    },
    /// a native pack for the strata engine from a model's GGUF (Orca, Qwen3.8 Flash Next): PolyStrata's tools/iq_pack.py
    IqPack {
        /// the model's shard 1
        #[arg(long)]
        gguf: PathBuf,
        /// the pack folder (e.g. ~/Development/Strata-data/packs/orca-iq3_xxs)
        #[arg(long)]
        out: PathBuf,
        /// a Q2_0 canonical pack whose dense.bin holds the shared float tensors
        #[arg(long)]
        base: Option<PathBuf>,
        /// rewrite the index only
        #[arg(long)]
        skip_experts: bool,
        /// dequantize small non-native projections to BF16 for ordinary GGUFs (rounds them; experts and the PLE table stay)
        #[arg(long)]
        compat_bf16: bool,
        /// also write experts.bin (the engine otherwise reads the experts from the GGUF itself)
        #[arg(long)]
        experts_bin: bool,
    },
    /// the MTP tensors from the Qwen3.8 Flash Next BF16 checkpoint by range requests: PolyStrata's tools/mtp_fetch.py
    MtpFetch {
        /// inventory (headers only), fetch (the tensors, resumable) or verify (against the pinned SHA-256; exit 3 when bad)
        #[arg(value_parser = ["inventory", "fetch", "verify"])]
        cmd: String,
        #[arg(long)]
        out: PathBuf,
        /// only tensors whose name contains this
        #[arg(long)]
        only: Option<String>,
    },
    /// the MTP draft block as a GGUF from the fetched tensors: PolyStrata's tools/mtp_pack.py
    MtpPack {
        /// the folder mtp-fetch wrote
        #[arg(long)]
        src: PathBuf,
        #[arg(long, default_value = "q2_0", value_parser = ["q2_0", "q4_0", "q8_0"])]
        experts: String,
        #[arg(long)]
        out: PathBuf,
        /// experts per tensor used for the error report
        #[arg(long, default_value_t = 8)]
        check_experts: usize,
    },
    /// the Workshop through the running server (its jobs show in the web app's Workshop tab):
    /// convert <hf-folder> | quantize <gguf> --type q4_k | merge <base> <adapter> | strata <gguf> [--name N]
    /// | jobs | job <id> | cancel <id> | files
    Workshop {
        action: String,
        /// convert/quantize: the input; merge: the base model; job/cancel: the job id
        first: Option<String>,
        /// merge: the adapter folder
        second: Option<String>,
        #[arg(long)]
        out: Option<String>,
        /// quantize: q8_0, q6_k, q5_k, q4_k, q4_0 or f16
        #[arg(long = "type")]
        ty: Option<String>,
        /// convert: f16 or f32
        #[arg(long)]
        outtype: Option<String>,
        /// convert: name the BPE pre-tokenizer when its hash is unknown
        #[arg(long)]
        pre_tokenizer: Option<String>,
        /// strata: the model's name (default: from the file) and display name; --overwrite replaces one of that name
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        display_name: Option<String>,
        #[arg(long)]
        overwrite: bool,
        /// follow the job until it ends
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        url: Option<String>,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long, default_value = "~/Development/PolyStrata")]
        models_dir: String,
    },
    /// the Computer tab from a terminal: status | screenshot <file.png> | take | give | copy | paste <text>
    /// | type <text> | open <app>  (the sandbox desktop computer use works on)
    Computer {
        action: String,
        /// screenshot: the file; paste/type: the text; open: the app
        arg: Option<String>,
        #[arg(long)]
        url: Option<String>,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long, default_value = "~/Development/PolyStrata")]
        models_dir: String,
    },
    /// HTTPS on the running server, applied at once and kept in its run config:
    /// status | self-signed | tailscale | load --cert <pem> --key <pem> | renew | off
    Tls {
        action: String,
        /// load: the certificate (PEM, with its chain)
        #[arg(long)]
        cert: Option<PathBuf>,
        /// load: its private key (PEM)
        #[arg(long)]
        key: Option<PathBuf>,
        /// the HTTPS (and wss) port (default 8001)
        #[arg(long)]
        port: Option<u16>,
        /// what plain HTTP does then: keep (default: http:// and ws:// keep working), local-only or redirect
        #[arg(long)]
        http: Option<String>,
        /// the server (default: this PC's, from its run configs; or AM2STRATA_URL)
        #[arg(long)]
        url: Option<String>,
        /// its API key (default: from the run configs; or AM2STRATA_API_KEY)
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long, default_value = "~/Development/PolyStrata")]
        models_dir: String,
    },
    /// what upstream (PolyStrata, Strata) changed since am2Strata's ports: tools/upstream.json against GitHub, read-only
    UpstreamCheck {
        /// the manifest (default: tools/upstream.json in this checkout)
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// download an image (later: video, voice) model for the media engine into <data-dir>/media, e.g. z-image-turbo:q8
    MediaPull {
        model: String,
        #[arg(long, default_value = "~/Development/Strata-data")]
        data_dir: PathBuf,
    },
    /// rewrite a GGUF with one tensor per expert (Mixtral 8x7B files from before April 2024) with the experts stacked,
    /// as llama.cpp reads them now; the weights are copied byte for byte
    GgufStackExperts {
        #[arg(long = "in")]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// wait until nvidia-smi sees every NVIDIA card on the PCI bus (a GPU on USB4 comes up after the boot); never fails
    WaitGpus {
        #[arg(long, default_value_t = 150)]
        timeout: u64,
    },
    /// a STRP expert profile (*.prof) as strata-poly's "layer expert count" text
    ProfileToCounts {
        #[arg(long = "in")]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// the MTP draft layer's runtime files (experts.bin, dense.bin, dense.txt) from the MTP GGUF: tools/mtp_rt.py
    MtpRt {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Args)]
struct PullArgs {
    /// the repository, e.g. unsloth/Qwen3.6-35B-A3B-GGUF (a https://huggingface.co/... address works too)
    repo: String,
    /// the quantization, e.g. Q4_K_M or UD-Q6_K_XL (case does not matter)
    #[arg(long)]
    quant: Option<String>,
    /// an exact file path in the repository instead of --quant (repeat for several)
    #[arg(long)]
    file: Vec<String>,
    #[arg(long, default_value = "main")]
    revision: String,
    /// access token for gated or private repositories (else HF_TOKEN, else the `hf auth login` file; none: public only)
    #[arg(long)]
    token: Option<String>,
    /// do not fetch the picture encoder (mmproj) even when the repository has one
    #[arg(long)]
    no_mmproj: bool,
    /// where the model files and packs go (the folder the installed models use)
    #[arg(long, default_value = "~/Development/Strata-data")]
    data_dir: String,
    /// the PolyStrata folder: engines, and where the run config is written (both servers list it)
    #[arg(long, default_value = "~/Development/PolyStrata")]
    models_dir: String,
    /// the context to configure (default: the model's own, from its GGUF)
    #[arg(long)]
    context: Option<u64>,
    /// download only; write no run config
    #[arg(long)]
    no_config: bool,
    /// the folder under <data-dir>/models for the files (default: the repository's name and the quantization)
    #[arg(long)]
    dir: Option<String>,
}

#[derive(Args)]
struct AgentArgs {
    /// list, start (one run, nothing of yours changes), setup (its config, backed up), reset (the backup back),
    /// show (what to enter, and what start would set)
    action: String,
    /// claude, codex, opencode, pi, droid, qwen, aider, goose, omp, crush, hermes, openclaw, continue, cline,
    /// kilocode, zed, vscode (agy and cursor-agent take no custom endpoint)
    agent: Option<String>,
    /// another am2Strata, e.g. http://192.168.1.20:8000 (default: this PC's, from the run configs)
    #[arg(long, env = "AM2STRATA_URL")]
    url: Option<String>,
    /// that server's API key (default: this PC's run config's)
    #[arg(long, env = "AM2STRATA_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    /// the model the agent uses (default: the one the server has loaded)
    #[arg(long, short)]
    model: Option<String>,
    /// a small, fast model for background work (Claude Code's haiku; default: --model)
    #[arg(long)]
    small_model: Option<String>,
    /// the server's tools this run asks for, within the key's client profile: web, memory, image, video, speech,
    /// music, mcp:<server> (sent as the x-strata-tools header by Claude Code, Codex and OpenCode)
    #[arg(long)]
    tools: Option<String>,
    /// the PolyStrata folder, for this PC's address and key
    #[arg(long, default_value = "~/Development/PolyStrata")]
    models_dir: String,
    /// print what would run or be written, and stop
    #[arg(long)]
    dry_run: bool,
    /// passed to the agent as they are (after --)
    #[arg(last = true)]
    rest: Vec<String>,
}

#[derive(Args, Clone)]
struct ServerArgs {
    /// the PolyStrata folder whose models and run configs are served
    #[arg(long, default_value = "~/Development/PolyStrata")]
    models_dir: String,
    /// model loaded at start (id or a unique part of it)
    #[arg(long)]
    model: Option<String>,
    #[arg(long, default_value_t = 8001)]
    port: u16,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    /// do not keep the last 100 API calls for /api-monitor (on by default)
    #[arg(long)]
    no_api_monitor: bool,
    /// passed to the server as they are
    #[arg(last = true)]
    rest: Vec<String>,
}

fn say(msg: impl AsRef<str>) {
    println!("[am2strata] {}", msg.as_ref());
}

/// The am2strata folder (this crate is crates/am2strata in it).
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the am2strata folder")
}

fn binary() -> PathBuf {
    root().join("target/release/am2strata")
}

fn expand(p: &str) -> String {
    match p.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_default()),
        None => p.to_string(),
    }
}

fn run(cmd: &mut Command) -> bool {
    cmd.status().map(|s| s.success()).unwrap_or(false)
}

fn systemctl(args: &[&str]) -> bool {
    run(Command::new("systemctl").arg("--user").args(args))
}

fn quiet_systemctl(args: &[&str]) -> bool {
    run(Command::new("systemctl").arg("--user").args(args).stdout(Stdio::null()).stderr(Stdio::null()))
}

fn active(unit: &str) -> bool {
    quiet_systemctl(&["is-active", "--quiet", unit])
}

/// Build the server when its source changed (cargo decides; a no-op when it is current).
fn build() -> bool {
    say("building am2strata (cargo build --release) ...");
    run(Command::new("cargo").args(["build", "--release", "-q", "-p", "am2strata"]).current_dir(root()))
}

fn server_cmd(s: &ServerArgs) -> Vec<String> {
    let mut c = vec![binary().display().to_string(), "--models-dir".into(), expand(&s.models_dir), "--port".into(),
                     s.port.to_string(), "--host".into(), s.host.clone()];
    if let Some(m) = &s.model {
        c.extend(["--model".into(), m.clone()]);
    }
    if !s.no_api_monitor {
        c.push("--api-monitor".into());
    }
    c.extend(s.rest.iter().cloned());
    c
}

/// The server resolves the configs and the model name, and exits: a typo fails here, before a running server
/// is stopped.
fn check(s: &ServerArgs) -> bool {
    let mut c = server_cmd(s);
    c.push("--check".into());
    match Command::new(&c[0]).args(&c[1..]).env("RUST_LOG", "warn").output() {
        Ok(o) if o.status.success() => {
            say(format!("model: {}", String::from_utf8_lossy(&o.stdout).trim()));
            true
        }
        Ok(o) => {
            say(format!("not started, nothing stopped: {}", String::from_utf8_lossy(&o.stderr).trim().trim_start_matches("Error: ")));
            false
        }
        Err(e) => {
            say(format!("could not run {}: {e}", c[0]));
            false
        }
    }
}

/// GET /health over a plain socket (the launcher needs no HTTP client): Some(body) once the server answers 200.
fn health(host: &str, port: u16) -> Option<String> {
    let host = if host == "0.0.0.0" || host.is_empty() { "127.0.0.1" } else { host };
    let mut s = std::net::TcpStream::connect_timeout(&format!("{host}:{port}").parse().ok()?, Duration::from_secs(2)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(3))).ok()?;
    write!(s, "GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n").ok()?;
    let mut buf = String::new();
    s.read_to_string(&mut buf).ok()?;
    let (head, body) = buf.split_once("\r\n\r\n")?;
    head.starts_with("HTTP/1.1 200").then(|| body.to_string())
}

/// Wait for /health while `alive()` holds (up to 15 minutes: a resident-RAM load takes minutes).
fn wait_ready(s: &ServerArgs, alive: impl Fn() -> bool) -> bool {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(900) {
        if !alive() {
            return false;
        }
        if let Some(body) = health(&s.host, s.port) {
            let v: serde_json::Value = serde_json::from_str(body.trim_start_matches(|c: char| c.is_ascii_hexdigit() || c == '\r' || c == '\n'))
                .unwrap_or_default();
            say(format!("ready: {} (images: {}, video: {})", v["model"].as_str().unwrap_or("?"), v["images"], v["video"]));
            say(format!("web app: http://127.0.0.1:{0}/   API: http://127.0.0.1:{0}/v1   API monitor: /api-monitor", s.port));
            return true;
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    false
}

fn free_gpu() {
    if active(SERVICE) {
        say(format!("systemctl --user stop {SERVICE}"));
        systemctl(&["stop", SERVICE]);
        std::thread::sleep(Duration::from_secs(3)); // the engine gives its GPU memory back
    }
}

fn serve(s: &ServerArgs, take_over: bool) -> ExitCode {
    if !build() {
        return ExitCode::FAILURE;
    }
    let restore = take_over && active(SERVICE);
    if restore {
        free_gpu();
    }
    let cmd = server_cmd(s);
    say(cmd.join(" "));
    let mut child = match Command::new(&cmd[0]).args(&cmd[1..]).current_dir(root()).spawn() {
        Ok(c) => c,
        Err(e) => {
            say(format!("could not start {}: {e}", cmd[0]));
            return ExitCode::FAILURE;
        }
    };
    // Ctrl+C reaches the server too (same process group): it stops its engine and image encoder itself
    let pid = child.id();
    let alive = || Path::new(&format!("/proc/{pid}")).exists();
    wait_ready(s, alive);
    let code = child.wait().map(|s| s.code().unwrap_or(1)).unwrap_or(1);
    if restore {
        say(format!("systemctl --user start {SERVICE}"));
        systemctl(&["start", SERVICE]);
    }
    ExitCode::from(code as u8)
}

fn temp_service(s: &ServerArgs) -> ExitCode {
    if !build() || !check(s) {
        return ExitCode::FAILURE;
    }
    if active(TEMP_UNIT) {
        // a restart: stop the running one and wait until systemd has unloaded it (a transient unit's name is busy
        // until then, and systemd-run would fail)
        say(format!("{TEMP_UNIT} is running: restarting it"));
        systemctl(&["stop", TEMP_UNIT]);
        for _ in 0..100 {
            if !quiet_systemctl(&["cat", TEMP_UNIT]) {
                break;
            }
            quiet_systemctl(&["reset-failed", TEMP_UNIT]);
            std::thread::sleep(Duration::from_millis(200));
        }
    }
    quiet_systemctl(&["reset-failed", TEMP_UNIT]);
    if active(BOOT_UNIT) {
        say(format!("{BOOT_UNIT} (am2Strata's boot service) is running already"));
        return ExitCode::FAILURE;
    }
    free_gpu();
    let root = root().display().to_string();
    let mut unit = vec!["systemd-run".to_string(), "--user".into(), format!("--unit={TEMP_UNIT}"),
                        "--description=am2Strata (temporary)".into(), format!("--working-directory={root}"),
                        format!("--property=Conflicts={SERVICE}.service"),
                        format!("--property=ExecStopPost=/usr/bin/systemctl --user start {SERVICE}.service"),
                        "--property=KillSignal=SIGINT".into(), "--property=TimeoutStopSec=90".into()];
    unit.extend(server_cmd(s));
    say(unit.join(" "));
    if !run(Command::new(&unit[0]).args(&unit[1..])) {
        systemctl(&["start", SERVICE]);
        return ExitCode::FAILURE;
    }
    say(format!("logs: journalctl --user -u {TEMP_UNIT} -f    stop: am2strata-run stop-temp-service"));
    if !wait_ready(s, || active(TEMP_UNIT)) {
        say(format!("{TEMP_UNIT} did not become ready; see journalctl --user -u {TEMP_UNIT}"));
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn unit_dir() -> PathBuf {
    PathBuf::from(expand("~/.config/systemd/user"))
}

/// The boot service: the same place in the system as strata-orca (port, configs, models, linger), am2Strata's binary.
fn install(s: &ServerArgs) -> ExitCode {
    if !build() || !check(s) {
        return ExitCode::FAILURE;
    }
    let cmd = server_cmd(s).iter().map(|a| if a.contains(' ') { format!("\"{a}\"") } else { a.clone() }).collect::<Vec<_>>().join(" ");
    let text = format!(
        "[Unit]\nDescription=am2Strata server (port {port}); replaces {SERVICE} while installed\nAfter=network-online.target\n\
         Conflicts={SERVICE}.service\n\n[Service]\nType=simple\nWorkingDirectory={root}\nExecStartPre={wait} wait-gpus\n\
         TimeoutStartSec=240\nExecStart={cmd}\nKillSignal=SIGINT\n\
         Restart=on-failure\nRestartSec=15\nTimeoutStopSec=90\n\n[Install]\nWantedBy=default.target\n",
        port = s.port, root = root().display(), wait = binary().with_file_name("am2strata-run").display());
    let path = unit_dir().join(format!("{BOOT_UNIT}.service"));
    if let Err(e) = std::fs::create_dir_all(unit_dir()).and_then(|_| std::fs::write(&path, text)) {
        say(format!("could not write {}: {e}", path.display()));
        return ExitCode::FAILURE;
    }
    say(format!("wrote {}", path.display()));
    if active(TEMP_UNIT) {
        systemctl(&["stop", TEMP_UNIT]);
    }
    systemctl(&["daemon-reload"]);
    systemctl(&["disable", SERVICE]);
    systemctl(&["stop", SERVICE]);
    std::thread::sleep(Duration::from_secs(3));
    // restart, not `enable --now`: a boot unit already running would keep the old binary and unit
    systemctl(&["enable", BOOT_UNIT]);
    systemctl(&["restart", BOOT_UNIT]);
    if !wait_ready(s, || active(BOOT_UNIT)) {
        say(format!("{BOOT_UNIT} did not become ready; see journalctl --user -u {BOOT_UNIT}. `am2strata-run uninstall` restores {SERVICE}"));
        return ExitCode::FAILURE;
    }
    say(format!("am2Strata now starts at boot ({BOOT_UNIT}); {SERVICE} is disabled. Undo: am2strata-run uninstall"));
    ExitCode::SUCCESS
}

/// The boot service starts after this, so a split config finds both cards; a card that never comes is the server's to
/// report (it runs on the others), so this always succeeds.
fn wait_gpus(timeout: u64) -> ExitCode {
    let cards = || std::fs::read_dir("/sys/bus/pci/devices").into_iter().flatten().flatten().filter(|d| {
        let read = |n: &str| std::fs::read_to_string(d.path().join(n)).unwrap_or_default();
        read("vendor").trim() == "0x10de" && read("class").starts_with("0x03")
    }).count();
    let seen = || Command::new("nvidia-smi").arg("-L").output().map(|o| String::from_utf8_lossy(&o.stdout).lines().filter(|l| l.starts_with("GPU ")).count()).unwrap_or(0);
    let t0 = std::time::Instant::now();
    loop {
        let (want, have) = (cards(), seen());
        if have >= want {
            say(format!("{have} of {want} GPU(s) ready"));
            return ExitCode::SUCCESS;
        }
        if t0.elapsed().as_secs() >= timeout {
            say(format!("only {have} of {want} GPU(s) after {timeout} s; going on without the others"));
            return ExitCode::SUCCESS;
        }
        std::thread::sleep(Duration::from_secs(3));
    }
}

fn uninstall() -> ExitCode {
    systemctl(&["disable", "--now", BOOT_UNIT]);
    let _ = std::fs::remove_file(unit_dir().join(format!("{BOOT_UNIT}.service")));
    systemctl(&["daemon-reload"]);
    systemctl(&["enable", "--now", SERVICE]);
    say(format!("{SERVICE} starts at boot again and is starting now"));
    ExitCode::SUCCESS
}

fn pull(p: PullArgs) -> ExitCode {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    match rt.block_on(pull_async(p)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            say(format!("pull failed: {e}"));
            ExitCode::FAILURE
        }
    }
}

async fn pull_async(p: PullArgs) -> Result<(), String> {
    use strata_hub::human;
    use strata_hub::pull::{Event, Outcome, Spec};
    let repo = strata_hub::pull::repo_name(&p.repo);
    let token = strata_hub::token(p.token.as_deref());
    say(format!("{repo} on {} ({})", strata_hub::endpoint(), if token.is_some() { "with a token" } else { "no token: public files only" }));
    if p.quant.is_none() && p.file.is_empty() {
        let qs = strata_hub::pull::list(&repo, token).await?;
        if qs.is_empty() {
            return Err(format!("{repo} has no GGUF files (am2Strata runs GGUF models)"));
        }
        say("quantizations (pass one with --quant):");
        for (q, size, n) in qs {
            println!("  {q:<14} {:>10}{}", human(size), if n > 1 { format!("  ({n} files)") } else { String::new() });
        }
        return Ok(());
    }
    let spec = Spec { repo, revision: p.revision.clone(), quant: p.quant.clone(), files: p.file.clone(), token, mmproj: !p.no_mmproj,
                      data_dir: PathBuf::from(expand(&p.data_dir)), models_dir: PathBuf::from(expand(&p.models_dir)),
                      context: p.context, write_config: !p.no_config, dir: p.dir.clone() };
    let t0 = Instant::now();
    let mut on = |e: Event| match e {
        Event::Note(n) => say(n),
        Event::Progress { file, done, total } => {
            let s = t0.elapsed().as_secs_f64().max(0.1);
            print!("\r  {file}: {} / {} ({:.0}%, {:.1} MB/s)   ", human(done), human(total), done as f64 * 100.0 / total.max(1) as f64,
                   done as f64 / s / 1048576.0);
            let _ = std::io::stdout().flush();
        }
        Event::FileDone { file, checked } => println!("\r  {file}: done, SHA-256 {}                    ", if checked { "checked" } else { "not published" }),
    };
    match strata_hub::pull::run(&spec, &mut on).await? {
        Outcome::Configured { id, config } => {
            say(format!("run config: {}", config.display()));
            say(format!("ready: load it from the web app's model picker, or POST /api/models/load {{\"model\": \"{id}\"}}"));
        }
        Outcome::Downloaded { dir, why: Some(why) } => say(format!("downloaded, but no run config was written: {why}. The files stay in {}", dir.display())),
        Outcome::Downloaded { dir, why: None } => say(format!("downloaded to {}", dir.display())),
    }
    Ok(())
}

fn main() -> ExitCode {
    // a closed pipe (`am2strata agent list | head`) ends the program quietly, as other command-line tools do; Rust's
    // default ignores SIGPIPE, and the next print then panicked with "failed printing to stdout: Broken pipe"
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    match Cli::parse().cmd {
        Cmd::Serve { s, take_over } => serve(&s, take_over),
        Cmd::TempService { s } => temp_service(&s),
        Cmd::StopTempService => {
            say(format!("systemctl --user stop {TEMP_UNIT}  ({SERVICE} starts again)"));
            if systemctl(&["stop", TEMP_UNIT]) { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        Cmd::Install { s } => install(&s),
        Cmd::Uninstall => uninstall(),
        Cmd::Pull(p) => pull(p),
        Cmd::Agent(a) => agent(a),
        Cmd::Agentrun { args } => match agentrun_args(args) {
            Ok(a) => agent(a),
            Err(e) => { eprintln!("{e}"); ExitCode::from(2) }
        },
        Cmd::IqPack { gguf, out, base, skip_experts, compat_bf16, experts_bin } => done(strata_pack::iq_pack::run(&strata_pack::iq_pack::Options {
            gguf: expand_path(&gguf), out: expand_path(&out), base: base.map(|b| expand_path(&b)), skip_experts, compat_bf16, experts_bin })),
        Cmd::MtpFetch { cmd, out, only } => {
            let out = expand_path(&out);
            let mut src = strata_pack::mtp_fetch::Source::from_env();
            match cmd.as_str() {
                "verify" => {
                    let bad = strata_pack::mtp_fetch::verify(&src, &out);
                    for n in &bad {
                        eprintln!("MTP tensor missing or corrupt: {n}");
                    }
                    if bad.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(strata_pack::mtp_fetch::BAD as u8) }
                }
                "inventory" => done(src.resolve().and_then(|_| strata_pack::mtp_fetch::inventory(&src, &out).map(|_| ()))),
                _ => {
                    strata_pack::mtp_fetch::exit_with_parent();
                    let lock = strata_pack::mtp_fetch::lock_folder(&out, std::time::Duration::from_secs(900));
                    done(lock.and_then(|_l| src.resolve().and_then(|_| strata_pack::mtp_fetch::fetch(&src, &out, only.as_deref()))))
                }
            }
        }
        Cmd::MtpPack { src, experts, out, check_experts } => done(strata_pack::mtp_pack::run(&strata_pack::mtp_pack::Options {
            src: expand_path(&src), experts, out: expand_path(&out), check_experts })),
        Cmd::UpstreamCheck { manifest } => upstream_check(manifest),
        Cmd::Computer { action, arg, url, api_key, models_dir } => computer_cmd(&action, arg, url, api_key, &models_dir),
        Cmd::Workshop { action, first, second, out, ty, outtype, pre_tokenizer, name, display_name, overwrite, wait, url, api_key, models_dir } =>
            workshop_cmd(&action, first, second, out, ty, outtype, pre_tokenizer, (name, display_name, overwrite), wait, url, api_key, &models_dir),
        Cmd::Tls { action, cert, key, port, http, url, api_key, models_dir } => tls_cmd(&action, cert, key, port, http, url, api_key, &models_dir),
        Cmd::MediaPull { model, data_dir } => media_pull(&model, &expand_path(&data_dir)),
        Cmd::GgufStackExperts { input, out } => done(strata_pack::stack_experts::run(&strata_pack::stack_experts::Options {
            input: expand_path(&input), output: expand_path(&out) })),
        Cmd::WaitGpus { timeout } => wait_gpus(timeout),
        Cmd::ProfileToCounts { input, out } => done(std::fs::read(expand_path(&input)).map_err(|e| e.to_string())
            .and_then(|b| strata_hub::pull::strp_to_counts(&b))
            .and_then(|t| std::fs::write(expand_path(&out), t).map_err(|e| e.to_string()))),
        Cmd::MtpRt { gguf, out } => done(strata_pack::mtp_rt::run(&strata_pack::mtp_rt::Options { gguf: expand_path(&gguf), out: expand_path(&out) })),
        Cmd::Test => {
            if run(Command::new("cargo").args(["test", "--release"]).current_dir(root())) { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
    }
}

fn done(r: Result<(), String>) -> ExitCode {
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn expand_path(p: &std::path::Path) -> PathBuf {
    PathBuf::from(expand(&p.to_string_lossy()))
}

/// The media engine's pull: its catalog's files for `model`, into <data-dir>/media (resumable, SHA-256 checked).
/// `upstream-check`: per repository of tools/upstream.json, how far its branch is from the commit am2Strata ported
/// (PolyStrata: also the local checkout), Strata's newest tag against the version PolyStrata carries, and per ported
/// file whether upstream's blob changed since. Read-only: GitHub's API (GITHUB_TOKEN if set) and `git` in the checkout.
fn upstream_check(manifest: Option<PathBuf>) -> ExitCode {
    let path = manifest.map(|p| expand_path(&p)).unwrap_or_else(|| root().join("tools/upstream.json"));
    let m: serde_json::Value = match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string())) {
        Ok(v) => v,
        Err(e) => { eprintln!("{}: {e}", path.display()); return ExitCode::from(2); }
    };
    let client = reqwest::blocking::Client::builder().user_agent("am2strata-upstream-check").timeout(std::time::Duration::from_secs(60))
        .build().expect("http client");
    let gh = |url: &str| -> Result<serde_json::Value, String> {
        let mut req = client.get(format!("https://api.github.com/{url}")).header("Accept", "application/vnd.github+json");
        if let Ok(t) = std::env::var("GITHUB_TOKEN") {
            req = req.bearer_auth(t);
        }
        let r = req.send().map_err(|e| e.to_string())?;
        if !r.status().is_success() {
            return Err(format!("GitHub {} for {url}", r.status()));
        }
        r.json().map_err(|e| e.to_string())
    };
    let git = |dir: &Path, args: &[&str]| std::process::Command::new("git").arg("-C").arg(dir).args(args).output().ok()
        .filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let mut drift = 0;
    let mut trees: std::collections::HashMap<String, std::collections::HashMap<String, String>> = Default::default();
    for (key, r) in m["repos"].as_object().into_iter().flatten() {
        let repo = r["github"].as_str().unwrap_or("");
        let reference = r["ref"].as_str().unwrap_or("main");
        println!("== {key} ({repo})");
        if let Some(at) = r["at"].as_str() {
            match gh(&format!("repos/{repo}/compare/{at}...{reference}")) {
                Ok(c) => {
                    let ahead = c["ahead_by"].as_u64().unwrap_or(0);
                    drift += (ahead > 0) as u32;
                    println!("  {reference} is {ahead} commit(s) past the port ({})", &at[..8.min(at.len())]);
                    for x in c["commits"].as_array().into_iter().flatten().rev().take(8) {
                        println!("    {} {}", &x["sha"].as_str().unwrap_or("")[..8], x["commit"]["message"].as_str().unwrap_or("").lines().next().unwrap_or(""));
                    }
                }
                Err(e) => println!("  {e}"),
            }
        }
        if let Some(co) = r["checkout"].as_str() {
            let dir = PathBuf::from(strata_engine::expand_str(co));
            match git(&dir, &["rev-parse", "HEAD"]) {
                Some(head) => {
                    let behind = gh(&format!("repos/{repo}/compare/{head}...{reference}")).ok().and_then(|c| c["ahead_by"].as_u64());
                    let version = std::fs::read_to_string(dir.join("CMakeLists.txt")).ok()
                        .and_then(|t| t.lines().find(|l| l.contains("project(strata VERSION")).map(|l| l.split_whitespace().nth(2).unwrap_or("").to_string()));
                    println!("  local checkout {} at {} - {} commit(s) behind {reference}; carries Strata {}", dir.display(), &head[..8],
                             behind.map(|b| b.to_string()).unwrap_or("?".into()), version.unwrap_or("?".into()));
                }
                None => println!("  local checkout {}: not a git checkout", dir.display()),
            }
        }
        // a tagged repository is compared at its newest tag: what a port from it would take now
        let mut compare_at = reference.to_string();
        if r["latest_tags"] == true {
            if let Ok(t) = gh(&format!("repos/{repo}/tags?per_page=3")) {
                let tags: Vec<&str> = t.as_array().into_iter().flatten().filter_map(|x| x["name"].as_str()).collect();
                println!("  newest tags: {} (ported from: {reference})", tags.join(", "));
                if let Some(newest) = tags.first().filter(|t| **t != reference) {
                    drift += 1;
                    compare_at = newest.to_string();
                }
            }
        }
        println!("  files compared at {compare_at}");
        match gh(&format!("repos/{repo}/git/trees/{compare_at}?recursive=1")) {
            Ok(t) => {
                trees.insert(key.clone(), t["tree"].as_array().into_iter().flatten()
                    .filter_map(|x| Some((x["path"].as_str()?.to_string(), x["sha"].as_str()?.to_string()))).collect());
            }
            Err(e) => println!("  tree: {e}"),
        }
    }
    println!("== ported files");
    for f in m["files"].as_array().into_iter().flatten() {
        let (repo, p, blob) = (f["repo"].as_str().unwrap_or(""), f["path"].as_str().unwrap_or(""), f["blob"].as_str().unwrap_or(""));
        let now = trees.get(repo).and_then(|t| t.get(p));
        let state = match now {
            None => "gone or not read".to_string(),
            Some(b) if b == blob => "current".to_string(),
            Some(b) => {
                drift += 1;
                format!("CHANGED ({} -> {})", &blob[..8.min(blob.len())], &b[..8])
            }
        };
        println!("  {:<11} {:<28} {:<22} -> {}", repo, p, state,
                 f["rust"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default());
    }
    println!("{}", if drift == 0 { "nothing to port" } else { "upstream has moved: see above (after porting, update tools/upstream.json)" });
    ExitCode::SUCCESS
}

fn media_pull(model: &str, data_dir: &Path) -> ExitCode {
    let exe = root().join("crates/strata-media/target/release/am2strata-media");
    if !exe.is_file() {
        say("the media engine is not built: cargo build --release --manifest-path crates/strata-media/Cargo.toml --features cuda");
        return ExitCode::FAILURE;
    }
    let home = data_dir.join("media");
    let _ = std::fs::create_dir_all(&home);
    say(format!("pulling {model} into {}", home.display()));
    let child = Command::new(&exe).env("MOLD_HOME", &home).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn();
    let Ok(mut child) = child else {
        say(format!("could not start {}", exe.display()));
        return ExitCode::FAILURE;
    };
    let mut stdin = child.stdin.take().unwrap();
    let _ = writeln!(stdin, "{}", serde_json::json!({"op": "pull", "model": model}));
    drop(stdin);
    let out = std::io::BufReader::new(child.stdout.take().unwrap());
    let mut ok = false;
    for line in std::io::BufRead::lines(out).map_while(Result::ok) {
        let v: serde_json::Value = serde_json::from_str(&line).unwrap_or_default();
        match v["event"].as_str() {
            Some("pulled") => {
                ok = true;
                say(format!("{model} is downloaded"));
            }
            Some("error") => say(format!("not downloaded: {}", v["message"].as_str().unwrap_or(""))),
            _ => {}
        }
    }
    let _ = child.wait();
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

/// `am2strata-run workshop ...`: the server's /v1/workshop API, as the Workshop tab uses it.
#[allow(clippy::too_many_arguments)]
fn workshop_cmd(action: &str, first: Option<String>, second: Option<String>, out: Option<String>, ty: Option<String>,
                outtype: Option<String>, pre: Option<String>, install: (Option<String>, Option<String>, bool), wait: bool,
                url: Option<String>, api_key: Option<String>,
                models_dir: &str) -> ExitCode {
    let (local_url, local_key) = local_server(models_dir);
    let url = url.or_else(|| std::env::var("AM2STRATA_URL").ok()).unwrap_or(local_url).trim_end_matches('/').to_string();
    let key = api_key.or_else(|| std::env::var("AM2STRATA_API_KEY").ok()).or(local_key);
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(60)).build().unwrap();
    let send = |rb: reqwest::blocking::RequestBuilder| -> Result<serde_json::Value, String> {
        let rb = match &key { Some(k) => rb.bearer_auth(k), None => rb };
        let r = rb.send().map_err(|e| format!("{url} does not answer ({e}); is the server running?"))?;
        let ok = r.status().is_success();
        let j: serde_json::Value = r.json().unwrap_or_default();
        if ok { Ok(j) } else { Err(j["error"]["message"].as_str().unwrap_or("failed").to_string()) }
    };
    // a relative path is the caller's: make it absolute before the server sees it
    let abs = |p: String| -> String {
        let e = expand_path(Path::new(&p));
        if e.is_absolute() { e.display().to_string() } else { std::env::current_dir().map(|d| d.join(&e)).unwrap_or(e).display().to_string() }
    };
    let show = |j: &serde_json::Value| {
        let res = j["result"]["output"].as_str().map(|o| format!(" -> {o}")).unwrap_or_default();
        println!("#{} {} {}{}", j["id"], j["kind"].as_str().unwrap_or(""), j["state"].as_str().unwrap_or(""), res);
        if let Some(e) = j["error"].as_str() { println!("  error: {e}"); }
    };
    let result = (|| -> Result<(), String> {
        let body = match action {
            "convert" => serde_json::json!({"kind": "convert", "input": abs(first.clone().ok_or("convert <hf-folder>")?),
                                            "output": out.clone().map(abs), "outtype": outtype, "pre_tokenizer": pre}),
            "quantize" => serde_json::json!({"kind": "quantize", "input": abs(first.clone().ok_or("quantize <gguf> --type q4_k")?),
                                             "type": ty.clone().ok_or("quantize needs --type (q8_0, q6_k, q5_k, q4_k, q4_0, f16)")?,
                                             "output": out.clone().map(abs)}),
            "merge" => serde_json::json!({"kind": "merge", "base": abs(first.clone().ok_or("merge <base-folder> <adapter-folder>")?),
                                          "adapter": abs(second.clone().ok_or("merge <base-folder> <adapter-folder>")?),
                                          "output": out.clone().map(abs)}),
            "strata" => serde_json::json!({"kind": "strata", "input": abs(first.clone().ok_or("strata <gguf> [--name N]")?),
                                           "name": install.0.clone(), "display_name": install.1.clone(), "overwrite": install.2}),
            "jobs" => {
                let j = send(client.get(format!("{url}/v1/workshop/jobs")))?;
                for x in j["jobs"].as_array().into_iter().flatten() { show(x); }
                return Ok(());
            }
            "files" => {
                let j = send(client.get(format!("{url}/v1/workshop/files")))?;
                println!("{}", serde_json::to_string_pretty(&j).unwrap_or_default());
                return Ok(());
            }
            "job" | "cancel" => {
                let id = first.clone().ok_or(format!("{action} <job id>"))?;
                let j = if action == "job" { send(client.get(format!("{url}/v1/workshop/jobs/{id}")))? }
                        else { send(client.delete(format!("{url}/v1/workshop/jobs/{id}")))? };
                show(&j);
                for l in j["log"].as_array().into_iter().flatten() { println!("  {}", l.as_str().unwrap_or("")); }
                return Ok(());
            }
            other => return Err(format!("unknown action {other:?}: convert, quantize, merge, strata, jobs, job, cancel, files")),
        };
        let mut body = body;
        if let Some(o) = body.as_object_mut() { o.retain(|_, v| !v.is_null()); }
        let j = send(client.post(format!("{url}/v1/workshop/jobs")).json(&body))?;
        show(&j);
        if wait {
            let id = j["id"].clone();
            let mut seen = 0usize;
            loop {
                std::thread::sleep(Duration::from_millis(800));
                let j = send(client.get(format!("{url}/v1/workshop/jobs/{id}")))?;
                let log = j["log"].as_array().cloned().unwrap_or_default();
                for l in log.iter().skip(seen) { println!("  {}", l.as_str().unwrap_or("")); }
                seen = log.len();
                if !matches!(j["state"].as_str(), Some("queued" | "running")) {
                    show(&j);
                    return if j["state"] == "done" { Ok(()) } else { Err(format!("job {id} {}", j["state"].as_str().unwrap_or(""))) };
                }
            }
        }
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("workshop: {e}"); ExitCode::from(1) }
    }
}

/// `am2strata-run computer ...`: the Computer tab's REST side (/api/computer/*).
fn computer_cmd(action: &str, arg: Option<String>, url: Option<String>, api_key: Option<String>, models_dir: &str) -> ExitCode {
    let (local_url, local_key) = local_server(models_dir);
    let url = url.or_else(|| std::env::var("AM2STRATA_URL").ok()).unwrap_or(local_url).trim_end_matches('/').to_string();
    let key = api_key.or_else(|| std::env::var("AM2STRATA_API_KEY").ok()).or(local_key);
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(60)).build().unwrap();
    let auth = |rb: reqwest::blocking::RequestBuilder| match &key { Some(k) => rb.bearer_auth(k), None => rb };
    let need = |what: &str| arg.clone().ok_or_else(|| format!("computer {action} needs {what}"));
    let result = (|| -> Result<(), String> {
        let rb = match action {
            "status" => client.get(format!("{url}/api/computer/status")),
            "screenshot" => {
                let file = need("a file name (e.g. desktop.png)")?;
                let r = auth(client.get(format!("{url}/api/computer/screenshot"))).send().map_err(|e| format!("{url}: {e}"))?;
                if !r.status().is_success() {
                    let j: serde_json::Value = r.json().unwrap_or_default();
                    return Err(j["error"]["message"].as_str().unwrap_or("failed").to_string());
                }
                let bytes = r.bytes().map_err(|e| e.to_string())?;
                std::fs::write(&file, &bytes).map_err(|e| format!("{file}: {e}"))?;
                println!("saved {file} ({} KB)", bytes.len() / 1024);
                return Ok(());
            }
            "take" | "give" => client.post(format!("{url}/api/computer/control")).json(&serde_json::json!({"action": action})),
            "copy" => client.get(format!("{url}/api/computer/clipboard")),
            "paste" => client.post(format!("{url}/api/computer/clipboard")).json(&serde_json::json!({"text": need("the text")?})),
            "type" => client.post(format!("{url}/api/computer/type")).json(&serde_json::json!({"text": need("the text")?})),
            "open" => client.post(format!("{url}/api/computer/open")).json(&serde_json::json!({"app": need("an app name")?})),
            other => return Err(format!("unknown action {other:?}: status, screenshot, take, give, copy, paste, type, open")),
        };
        let r = auth(rb).send().map_err(|e| format!("{url} does not answer ({e}); is the server running?"))?;
        let ok = r.status().is_success();
        let j: serde_json::Value = r.json().unwrap_or_default();
        if !ok {
            return Err(j["error"]["message"].as_str().unwrap_or("failed").to_string());
        }
        match action {
            "copy" => print!("{}", j["text"].as_str().unwrap_or("")),
            "status" | "take" | "give" => println!("computer use: {}, mode {}, {}{}",
                if j["on"] == true { "on" } else { "off" }, j["mode"].as_str().unwrap_or("?"),
                if j["operator_in_control"] == true { "you have control" } else { "am2Strata may act" },
                j["apps"].as_array().filter(|a| !a.is_empty()).map(|a| format!(", apps: {}", a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", "))).unwrap_or_default()),
            _ => println!("done"),
        }
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("computer: {e}"); ExitCode::from(1) }
    }
}

/// `am2strata-run tls ...`: the same /api/tls the web app's Settings › Secure connection uses.
#[allow(clippy::too_many_arguments)]
fn tls_cmd(action: &str, cert: Option<PathBuf>, key_file: Option<PathBuf>, port: Option<u16>, http: Option<String>,
           url: Option<String>, api_key: Option<String>, models_dir: &str) -> ExitCode {
    let (local_url, local_key) = local_server(models_dir);
    let url = url.or_else(|| std::env::var("AM2STRATA_URL").ok()).unwrap_or(local_url).trim_end_matches('/').to_string();
    let api_key = api_key.or_else(|| std::env::var("AM2STRATA_API_KEY").ok()).or(local_key);
    let mut body = match action {
        "status" => None,
        "self-signed" | "tailscale" | "renew" | "off" => Some(serde_json::json!({"action": action})),
        "load" => {
            let (Some(c), Some(k)) = (cert, key_file) else {
                eprintln!("tls load needs --cert <certificate.pem> and --key <private-key.pem>");
                return ExitCode::from(2);
            };
            let read = |p: &Path| std::fs::read_to_string(expand_path(p)).map_err(|e| format!("{}: {e}", p.display()));
            match (read(&c), read(&k)) {
                (Ok(cp), Ok(kp)) => Some(serde_json::json!({"action": "load", "cert_pem": cp, "key_pem": kp})),
                (Err(e), _) | (_, Err(e)) => { eprintln!("tls load: {e}"); return ExitCode::from(2); }
            }
        }
        other => {
            eprintln!("tls: unknown action {other:?} (status, self-signed, tailscale, load, renew, off)");
            return ExitCode::from(2);
        }
    };
    if let Some(b) = body.as_mut() {
        if let Some(p) = port { b["https_port"] = serde_json::json!(p); }
        if let Some(h) = &http { b["http"] = serde_json::json!(h); }
    }
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(120)).build().unwrap();
    let mut rb = match &body {
        None => client.get(format!("{url}/api/tls")),
        Some(b) => client.post(format!("{url}/api/tls")).json(b),
    };
    if let Some(k) = &api_key {
        rb = rb.bearer_auth(k);
    }
    let r = match rb.send() {
        Ok(r) => r,
        Err(e) => { eprintln!("tls: {url} does not answer ({e}); is the server running?"); return ExitCode::from(1); }
    };
    let ok = r.status().is_success();
    let j: serde_json::Value = r.json().unwrap_or_default();
    if !ok {
        eprintln!("tls {action}: {}", j["error"]["message"].as_str().unwrap_or("failed"));
        return ExitCode::from(1);
    }
    print_tls(&j);
    if let Some(e) = j["saved_error"].as_str() { eprintln!("warning: {e}"); }
    ExitCode::SUCCESS
}

fn print_tls(j: &serde_json::Value) {
    let s = |v: &serde_json::Value| v.as_str().unwrap_or("").to_string();
    if j["on"] != true {
        println!("HTTPS: off (plain HTTP on port {})", j["api_port"]);
    } else {
        println!("HTTPS: on, {} certificate, port {} (wss on the same port)", s(&j["mode"]), j["https_port"]);
        println!("  plain HTTP: {}", s(&j["http"]));
        println!("  HSTS: {}", s(&j["hsts"]));
        let c = &j["certificate"];
        println!("  certificate: {}{}", s(&c["subject"]), if c["self_signed"] == true { " (self-signed)".into() } else { format!(", issued by {}", s(&c["issuer"])) });
        println!("  names: {}", c["names"].as_array().map(|a| a.iter().map(|x| s(x)).collect::<Vec<_>>().join(", ")).unwrap_or_default());
        println!("  valid until: {} ({} days)", s(&c["not_after"]), c["expires_in_days"]);
        println!("  SHA-256: {}", s(&c["sha256"]));
        for u in j["urls"].as_array().into_iter().flatten() { println!("  open: {}", s(u)); }
    }
    match j["tailscale"].as_object() {
        Some(t) => println!("Tailscale: {} {}", t.get("name").and_then(|v| v.as_str()).unwrap_or("(no MagicDNS name)"),
                            t.get("ips").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default()),
        None => println!("Tailscale: not running or not logged in"),
    }
    if let Some(e) = j["last_error"].as_str() { println!("last error: {e}"); }
    if let Some(p) = j["saved"].as_str() { println!("kept in {p}"); }
}

/// This PC's server from its run configs: the address (127.0.0.1 and the port) and the key.
fn local_server(models_dir: &str) -> (String, Option<String>) {
    let dir = expand_path(Path::new(models_dir));
    let mut configs: Vec<PathBuf> = std::fs::read_dir(&dir).into_iter().flatten().flatten().map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("strata-") && n.ends_with(".json"))).collect();
    configs.sort();
    let cfg: serde_json::Value = configs.first().and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let port = cfg["port"].as_u64().unwrap_or(8080);
    (format!("http://127.0.0.1:{port}"), cfg["api_key"].as_str().filter(|k| !k.is_empty()).map(str::to_string))
}

/// The model the server has loaded (else its first), as /api/models or /v1/models says.
fn server_model(url: &str, key: Option<&str>) -> Result<String, String> {
    let get = |path: &str| -> Result<serde_json::Value, String> {
        let mut rb = reqwest::blocking::Client::new().get(format!("{url}{path}")).timeout(Duration::from_secs(10));
        if let Some(k) = key {
            rb = rb.bearer_auth(k);
        }
        let r = rb.send().map_err(|e| format!("{url} does not answer: {e}"))?;
        if r.status().as_u16() == 401 {
            return Err(format!("{url} wants an API key: --api-key or {}", agents::KEY_VAR));
        }
        r.json().map_err(|e| e.to_string())
    };
    if let Ok(v) = get("/api/models") {
        let list = v["models"].as_array().cloned().unwrap_or_default();
        if let Some(m) = list.iter().find(|m| m["current"] == true).or(list.first()).and_then(|m| m["id"].as_str()) {
            return Ok(m.to_string());
        }
    }
    let v = get("/v1/models")?;
    v["data"][0]["id"].as_str().map(str::to_string).ok_or_else(|| format!("{url} lists no model"))
}

/// The agent's command on PATH (on Windows also its .cmd or .exe, as npm installs them).
fn find_exe(exe: &str) -> Option<PathBuf> {
    let exts: &[&str] = if cfg!(windows) { &[".exe", ".cmd", ".bat", ""] } else { &[""] };
    std::env::split_paths(&std::env::var_os("PATH")?).flat_map(|d| exts.iter().map(move |e| d.join(format!("{exe}{e}"))))
        .find(|p| p.is_file())
}

fn agent(a: AgentArgs) -> ExitCode {
    if a.action == "list" {
        for g in agents::AGENTS {
            let what = match g.support {
                agents::Support::Full => "start, setup".to_string(),
                agents::Support::Start => "start".into(),
                agents::Support::Setup => "setup".into(),
                agents::Support::Guide => "guide (show)".into(),
                agents::Support::No(why) => format!("not supported: {why}"),
            };
            let have = if find_exe(g.exe).is_some() { "installed" } else { "" };
            println!("{:<13} {:<24} {:<10} {:<14} {have}", g.name, g.label, g.api, what);
        }
        return ExitCode::SUCCESS;
    }
    let Some(name) = a.agent.as_deref() else {
        eprintln!("which agent? (am2strata agent list)");
        return ExitCode::from(2);
    };
    let Some(info) = agents::find(name) else {
        eprintln!("no agent called {name:?} (am2strata agent list)");
        return ExitCode::from(2);
    };
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from).unwrap_or_default();
    if a.action == "reset" {
        return match agents::reset(info.name, &home) {
            Ok(v) if v.is_empty() => { say(format!("nothing of {} was set up by am2strata", info.label)); ExitCode::SUCCESS }
            Ok(v) => { for p in v { say(format!("restored {}", p.display())); } ExitCode::SUCCESS }
            Err(e) => { eprintln!("{e}"); ExitCode::FAILURE }
        };
    }
    let (local_url, local_key) = local_server(&a.models_dir);
    let url = a.url.clone().unwrap_or(local_url).trim_end_matches('/').to_string();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        eprintln!("--url is an address such as http://192.168.1.20:8000");
        return ExitCode::from(2);
    }
    let key = a.api_key.clone().filter(|k| !k.is_empty()).or(if a.url.is_none() { local_key } else { None });
    let model = match a.model.clone() {
        Some(m) => m,
        None => match server_model(&url, key.as_deref()) {
            Ok(m) => m,
            Err(e) => { eprintln!("{e}; or name one with --model"); return ExitCode::FAILURE; }
        },
    };
    let t = agents::Target { url, key, model, small: a.small_model.clone(), tools: a.tools.clone().filter(|t| !t.trim().is_empty()) };
    match a.action.as_str() {
        "show" => {
            println!("{} on {} with {} ({}):\n", info.label, t.url, t.model, info.checked);
            println!("{}", agents::guide(info.name, &t).unwrap_or_default());
            ExitCode::SUCCESS
        }
        "setup" => match agents::setup(info.name, &t, &home) {
            Ok(changes) => {
                for c in &changes {
                    if a.dry_run {
                        println!("--- {}\n{}", c.path.display(), c.after);
                    } else {
                        say(format!("{} {} (backup: {}.am2strata-backup; `agent reset {}` restores it)",
                                    if c.before.is_some() { "updated" } else { "wrote" }, c.path.display(), c.path.display(), info.name));
                    }
                }
                if a.dry_run {
                    return ExitCode::SUCCESS;
                }
                if let Err(e) = agents::apply(&changes) {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
                say(format!("{} now uses {} on {}", info.label, t.model, t.url));
                if t.key.is_some() {
                    say(format!("it reads the key from {} in your shell: set it in your shell profile", agents::KEY_VAR));
                }
                ExitCode::SUCCESS
            }
            Err(e) => { eprintln!("{e}"); ExitCode::FAILURE }
        },
        "start" => {
            let tmp = std::env::temp_dir().join(format!("am2strata-agent-{}-{}", info.name, std::process::id()));
            let plan = match agents::start(info.name, &t, &tmp) {
                Ok(p) => p,
                Err(e) => { eprintln!("{e}"); return ExitCode::FAILURE; }
            };
            if a.dry_run {
                for (k, v) in &plan.env {
                    let shown = if k == agents::KEY_VAR || k.contains("KEY") || k.contains("TOKEN") { "***" } else { v };
                    println!("{k}={shown}");
                }
                for (p, _) in &plan.files {
                    println!("(file) {}", p.display());
                }
                println!("{} {}", plan.exe, plan.args.iter().chain(&a.rest).cloned().collect::<Vec<_>>().join(" "));
                return ExitCode::SUCCESS;
            }
            let Some(exe) = find_exe(&plan.exe) else {
                eprintln!("{} is not installed ({} is not on PATH)", info.label, plan.exe);
                return ExitCode::from(127);
            };
            for (p, body) in &plan.files {
                if let Some(d) = p.parent() {
                    let _ = std::fs::create_dir_all(d);
                }
                if let Err(e) = std::fs::write(p, body) {
                    eprintln!("{}: {e}", p.display());
                    return ExitCode::FAILURE;
                }
            }
            for n in &plan.notes {
                say(n);
            }
            say(format!("{} on {} with {}", info.label, t.url, t.model));
            let mut cmd = if exe.extension().is_some_and(|e| e == "cmd" || e == "bat") {
                let mut c = Command::new("cmd");
                c.arg("/C").arg(&exe);
                c
            } else {
                Command::new(&exe)
            };
            let status = cmd.args(&plan.args).args(&a.rest).envs(plan.env.iter().map(|(k, v)| (k, v))).status();
            let _ = std::fs::remove_dir_all(&tmp);
            match status {
                Ok(s) => ExitCode::from(s.code().unwrap_or(1).clamp(0, 255) as u8),
                Err(e) => { eprintln!("{}: {e}", exe.display()); ExitCode::FAILURE }
            }
        }
        other => {
            eprintln!("{other}: list, start, setup, reset or show");
            ExitCode::from(2)
        }
    }
}

/// `agentrun <agent> [options] [agent arguments]`: am2Strata's own options wherever they are, the rest (and all after
/// `--`) for the agent, in their order.
fn agentrun_args(args: Vec<String>) -> Result<AgentArgs, String> {
    const USAGE: &str = "usage: am2strata agentrun <agent> [--model M] [--small-model M] [--url U] [--api-key K] [--tools web,image] [--dry-run] [agent arguments]\n\
                         (am2strata agent list shows the agents)";
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let mut a = AgentArgs { action: "start".into(), agent: None, url: env("AM2STRATA_URL"), api_key: env("AM2STRATA_API_KEY"),
                            model: None, small_model: None, tools: None, models_dir: "~/Development/PolyStrata".into(), dry_run: false, rest: Vec::new() };
    let mut it = args.into_iter();
    while let Some(x) = it.next() {
        let (flag, inline) = match x.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (x.clone(), None),
        };
        let mut value = |name: &str| inline.clone().or_else(|| it.next()).ok_or_else(|| format!("{name} needs a value\n{USAGE}"));
        match flag.as_str() {
            "--" => { a.rest.extend(it.by_ref()); }
            "--model" | "-m" => a.model = Some(value("--model")?),
            "--small-model" => a.small_model = Some(value("--small-model")?),
            "--tools" => a.tools = Some(value("--tools")?),
            "--url" => a.url = Some(value("--url")?),
            "--api-key" => a.api_key = Some(value("--api-key")?),
            "--models-dir" => a.models_dir = value("--models-dir")?,
            "--dry-run" => a.dry_run = true,
            "-h" | "--help" if a.agent.is_none() => return Err(USAGE.into()),
            _ if a.agent.is_none() && !x.starts_with('-') => a.agent = Some(x),
            _ => a.rest.push(x),
        }
    }
    if a.agent.is_none() {
        return Err(USAGE.into());
    }
    Ok(a)
}

#[cfg(test)]
mod agentrun_tests {
    use super::*;

    #[test]
    fn its_own_options_anywhere_and_the_rest_for_the_agent() {
        let v = |s: &str| s.split(' ').map(str::to_string).collect::<Vec<_>>();
        let a = agentrun_args(v("claude --model tiel --continue -p hi --url=http://h:8000")).unwrap();
        assert_eq!((a.agent.as_deref(), a.model.as_deref(), a.url.as_deref()), (Some("claude"), Some("tiel"), Some("http://h:8000")));
        assert_eq!(a.rest, v("--continue -p hi"));
        let a = agentrun_args(v("-m x opencode -- run --model y")).unwrap();
        assert_eq!((a.agent.as_deref(), a.model.as_deref(), a.rest.clone()), (Some("opencode"), Some("x"), v("run --model y")));
        assert!(agentrun_args(v("--model")).is_err() && agentrun_args(Vec::new()).is_err());
    }
}
