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
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
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
         Conflicts={SERVICE}.service\n\n[Service]\nType=simple\nWorkingDirectory={root}\nExecStart={cmd}\nKillSignal=SIGINT\n\
         Restart=on-failure\nRestartSec=15\nTimeoutStopSec=90\n\n[Install]\nWantedBy=default.target\n",
        port = s.port, root = root().display());
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
    systemctl(&["enable", "--now", BOOT_UNIT]);
    if !wait_ready(s, || active(BOOT_UNIT)) {
        say(format!("{BOOT_UNIT} did not become ready; see journalctl --user -u {BOOT_UNIT}. `am2strata-run uninstall` restores {SERVICE}"));
        return ExitCode::FAILURE;
    }
    say(format!("am2Strata now starts at boot ({BOOT_UNIT}); {SERVICE} is disabled. Undo: am2strata-run uninstall"));
    ExitCode::SUCCESS
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
                "inventory" => {
                    src.resolve();
                    done(strata_pack::mtp_fetch::inventory(&src, &out).map(|_| ()))
                }
                _ => {
                    src.resolve();
                    done(strata_pack::mtp_fetch::fetch(&src, &out, only.as_deref()))
                }
            }
        }
        Cmd::MtpPack { src, experts, out, check_experts } => done(strata_pack::mtp_pack::run(&strata_pack::mtp_pack::Options {
            src: expand_path(&src), experts, out: expand_path(&out), check_experts })),
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
