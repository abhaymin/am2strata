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
    if !build() {
        return ExitCode::FAILURE;
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
    if !build() {
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
    use strata_hub::{human, Hub};
    let repo = p.repo.trim_start_matches("https://huggingface.co/").trim_start_matches("hf.co/").trim_end_matches('/').to_string();
    let token = strata_hub::token(p.token.as_deref());
    say(format!("{repo} on {} ({})", strata_hub::endpoint(), if token.is_some() { "with a token" } else { "no token: public files only" }));
    let hub = Hub::new(token).map_err(|e| e.to_string())?;
    let files = hub.files(&repo, &p.revision).await.map_err(|e| e.to_string())?;
    let chosen = if !p.file.is_empty() {
        p.file.iter().map(|want| files.iter().find(|f| &f.path == want).cloned().ok_or(format!("{repo} has no file {want}")))
            .collect::<Result<Vec<_>, _>>()?
    } else if let Some(q) = &p.quant {
        strata_hub::select(&files, q, !p.no_mmproj).map_err(|e| e.to_string())?
    } else {
        let qs = strata_hub::quants(&files);
        if qs.is_empty() {
            return Err(format!("{repo} has no GGUF files (am2Strata runs GGUF models)"));
        }
        say("quantizations (pass one with --quant):");
        for (q, size, n) in qs {
            println!("  {q:<14} {:>10}{}", human(size), if n > 1 { format!("  ({n} files)") } else { String::new() });
        }
        return Ok(());
    };
    let total: u64 = chosen.iter().map(|f| f.size).sum();
    let tag = strata_hub::quant_tag(&chosen[0].path).or_else(|| p.quant.as_ref().map(|q| q.to_uppercase())).unwrap_or_else(|| "model".into());
    let base = repo.rsplit('/').next().unwrap_or(&repo).trim_end_matches("-GGUF").trim_end_matches("-gguf");
    let dir = PathBuf::from(expand(&p.data_dir)).join("models").join(format!("{base}-{tag}"));
    say(format!("{} file(s), {} -> {}", chosen.len(), human(total), dir.display()));
    let have: u64 = chosen.iter().map(|f| std::fs::metadata(strata_hub::local_path(&dir, f)).map(|m| m.len()).unwrap_or(0)).sum();
    if let Some(free) = strata_hub::free_space(&dir) {
        if free + have < total + (1 << 30) {
            return Err(format!("not enough disk space: {} needed, {} free", human(total - have.min(total)), human(free)));
        }
    }
    for f in &chosen {
        let dest = strata_hub::local_path(&dir, f);
        let name = f.path.rsplit('/').next().unwrap_or(&f.path).to_string();
        let t0 = Instant::now();
        let mut show = |done: u64, all: u64| {
            let s = t0.elapsed().as_secs_f64().max(0.1);
            print!("\r  {name}: {} / {} ({:.0}%, {:.1} MB/s)   ", human(done), human(all), done as f64 * 100.0 / all.max(1) as f64,
                   done as f64 / s / 1048576.0);
            let _ = std::io::stdout().flush();
        };
        hub.download(&repo, &p.revision, f, &dest, &mut show).await.map_err(|e| e.to_string())?;
        println!("\r  {name}: done, SHA-256 {}                    ", if f.sha256.is_some() { "checked" } else { "not published" });
    }
    if p.no_config {
        return Ok(());
    }
    let models: Vec<&strata_hub::RepoFile> = chosen.iter().filter(|f| !f.path.to_lowercase().contains("mmproj")).collect();
    let first = strata_hub::local_path(&dir, models[0]);
    let mmproj = chosen.iter().find(|f| f.path.to_lowercase().contains("mmproj")).map(|f| strata_hub::local_path(&dir, f));
    let g = strata_hub::config::inspect(&first)?;
    let arch = g.architecture().unwrap_or("?").to_string();
    say(format!("architecture {arch}, context {}, {} experts", g.arch_u64("context_length").map(|c| c.to_string()).unwrap_or("?".into()),
                g.arch_u64("expert_count").map(|c| c.to_string()).unwrap_or("no".into())));
    if let Err(why) = strata_hub::config::supported(&arch) {
        say(format!("downloaded, but no run config was written: {why}. The files stay in {}", dir.display()));
        return Ok(());
    }
    let file_name = first.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    let id = strata_hub::config::id_of(file_name);
    let display = format!("{} {tag}", g.str("general.name").filter(|n| !n.is_empty()).unwrap_or(base));
    let w = strata_hub::config::write(Path::new(&expand(&p.models_dir)), &PathBuf::from(expand(&p.data_dir)).join("packs"),
                                      &first, &g, mmproj.as_deref(), p.context, &id, &display)?;
    say(format!("run config: {} (tokenizer {})", w.config.display(), w.tokenizer.display()));
    say(format!("ready: load it from the web app's model picker, or POST /api/models/load {{\"model\": \"{id}\"}}"));
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
        Cmd::Test => {
            if run(Command::new("cargo").args(["test", "--release"]).current_dir(root())) { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
    }
}
