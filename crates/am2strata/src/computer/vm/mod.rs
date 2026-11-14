//! The sandbox VM: a small Debian 13 virtual machine am2Strata runs itself (QEMU + KVM, no root, no Docker), for
//! computer use with no limits inside it — the agent is root there and may install or remove anything (apt, Flathub,
//! a .deb, an AppImage), while the host is untouched: its own kernel, its own disk (a copy-on-write layer over the
//! Debian cloud image), user-mode networking with nothing reachable from outside, and no GPU.
//!
//! **Light on the host.** QEMU runs in a cgroup scope with `CPUWeight=idle` (it only gets CPU the inference engine is
//! not using), a hard `MemoryMax`, the idle I/O class and nice 19. The guest hands freed memory back (virtio-balloon
//! free-page reporting) and swaps to compressed RAM. It starts only when needed, pauses its CPUs after
//! `idle_pause_s` without use, and after `idle_stop_s` is saved to a file and quit (0 RAM, 0 CPU). It refuses to start
//! when it would leave the host less than `host_reserve_mb` free (a large model stays loaded).
//!
//! **Fast.** After the first boot (about 15 s, plus a few minutes of one-time setup), a start restores the saved
//! running VM in about half a second: the desktop and its open apps are as they were.
//!
//! **The screen** is TigerVNC's Xvnc in the guest (no virtual GPU), reached over an SSH-forwarded Unix socket in a
//! private runtime folder (rfb.rs). Commands run over SSH with our own key, checking the VM's host key, which we made.

pub mod api;
pub mod backend;
pub mod guest;
pub mod qmp;
pub mod rfb;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const IMAGE_FILE: &str = "debian-13-genericcloud-amd64.qcow2";
pub const IMAGE_URL: &str = "https://cloud.debian.org/images/cloud/trixie/latest/debian-13-genericcloud-amd64.qcow2";
pub const SUMS_URL: &str = "https://cloud.debian.org/images/cloud/trixie/latest/SHA512SUMS";
/// The device set the saved state belongs to: a restore needs the same one.
const MACHINE: &str = "q35-ovmf-virtio-balloon-net-blk-seed-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VmConfig {
    pub cpus: u32,
    /// the most RAM the VM may use (it gives back what it does not use)
    pub memory_mb: u32,
    pub disk_gb: u32,
    pub width: u16,
    pub height: u16,
    /// pause the VM's CPUs after this long unused (0: never)
    pub idle_pause_s: u64,
    /// save it to disk and quit after this long unused (0: never)
    pub idle_stop_s: u64,
    /// never start when it would leave the host less RAM than this
    pub host_reserve_mb: u32,
    /// block ads and trackers in the VM (every program, and uBlock in the browsers)
    pub block_ads: bool,
    /// the lists: "pihole" (Pi-hole's default list) and/or "adguard" (the AdGuard DNS filter)
    pub ads_lists: Vec<String>,
    /// domains whose ads are let through while blocking (a site a job must see as it is)
    pub ads_allow: Vec<String>,
}

impl Default for VmConfig {
    fn default() -> Self {
        VmConfig { cpus: 2, memory_mb: 2048, disk_gb: 20, width: 1280, height: 800, idle_pause_s: 120, idle_stop_s: 900, host_reserve_mb: 3072,
                   block_ads: false, ads_lists: vec!["pihole".into()], ads_allow: vec![] }
    }
}

impl VmConfig {
    /// Clamped to what makes sense; `changes` from the API or the CLI are merged over `self`.
    pub fn merged(&self, changes: &Value) -> Result<VmConfig, String> {
        let mut v = serde_json::to_value(self).unwrap();
        for (k, val) in changes.as_object().ok_or("settings are a JSON object")? {
            if v.get(k).is_none() {
                return Err(format!("unknown setting {k:?}"));
            }
            v[k] = val.clone();
        }
        let mut c: VmConfig = serde_json::from_value(v).map_err(|e| e.to_string())?;
        c.cpus = c.cpus.clamp(1, 16);
        c.memory_mb = c.memory_mb.clamp(768, 65536);
        c.disk_gb = c.disk_gb.clamp(8, 1024);
        c.width = c.width.clamp(800, 3840);
        c.height = c.height.clamp(600, 2160);
        c.host_reserve_mb = c.host_reserve_mb.clamp(512, 65536);
        Ok(c)
    }

    fn fingerprint(&self) -> String {
        format!("{MACHINE} cpus={} mem={}", self.cpus, self.memory_mb)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Phase { Off, Saved, Starting, Setup, Running, Paused, Stopping }

impl Phase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Phase::Off => "off", Phase::Saved => "saved", Phase::Starting => "starting", Phase::Setup => "setting-up",
            Phase::Running => "running", Phase::Paused => "paused", Phase::Stopping => "stopping",
        }
    }
}

/// A long operation (start, install, download, …) with its log, for the tab and the CLI to follow.
#[derive(Clone, Debug, Serialize)]
pub struct Task {
    pub id: u64,
    pub what: String,
    pub state: &'static str,
    pub log: Vec<String>,
    pub error: Option<String>,
    pub result: Option<String>,
    pub started: u64,
    pub finished: Option<u64>,
}

struct Running {
    child: Child,
    ssh_port: u16,
    qmp: qmp::Qmp,
}

pub struct Vm {
    dir: PathBuf,
    rt: PathBuf,
    cfg: Mutex<VmConfig>,
    phase: Mutex<Phase>,
    run: Mutex<Option<Running>>,
    /// lifecycle steps (start, stop, pause, …) one at a time
    op: Mutex<()>,
    tunnel: Mutex<Option<Child>>,
    screen: Mutex<Option<Arc<rfb::Rfb>>>,
    last_used: Mutex<Instant>,
    /// open Computer-tab viewers: a watched VM is not paused
    pub viewers: AtomicUsize,
    tasks: Mutex<Vec<Task>>,
    next_task: Mutex<u64>,
    busy: Mutex<bool>,
    cpu_sample: Mutex<Option<(Instant, u64)>>,
    last_error: Mutex<Option<String>>,
}

static VM: std::sync::OnceLock<Arc<Vm>> = std::sync::OnceLock::new();

pub fn get() -> Option<Arc<Vm>> {
    VM.get().cloned()
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// The private folder for the VM's sockets: short (a Unix socket path is at most 107 bytes) and only ours.
fn runtime_dir() -> PathBuf {
    let base = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).filter(|p| p.is_dir())
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/am2strata-{}", unsafe { libc::getuid() })));
    base.join("am2strata-vm")
}

pub fn mem_available_mb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    s.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb / 1024)
}

fn find_first(paths: &[&str]) -> Option<PathBuf> {
    paths.iter().map(PathBuf::from).find(|p| p.is_file())
}

fn ovmf() -> Option<(PathBuf, PathBuf)> {
    let code = find_first(&["/usr/share/OVMF/OVMF_CODE_4M.fd", "/usr/share/OVMF/OVMF_CODE.fd", "/usr/share/edk2/ovmf/OVMF_CODE.fd",
                            "/usr/share/edk2/x64/OVMF_CODE.4m.fd", "/usr/share/qemu/edk2-x86_64-code.fd"])?;
    let vars = find_first(&["/usr/share/OVMF/OVMF_VARS_4M.fd", "/usr/share/OVMF/OVMF_VARS.fd", "/usr/share/edk2/ovmf/OVMF_VARS.fd",
                            "/usr/share/edk2/x64/OVMF_VARS.4m.fd", "/usr/share/qemu/edk2-i386-vars.fd"])?;
    Some((code, vars))
}

/// What this PC lacks to run the VM (empty: nothing), each with how to get it.
pub fn missing_requirements() -> Vec<String> {
    let mut m = vec![];
    for (tool, pkg) in [("qemu-system-x86_64", "qemu-system-x86"), ("qemu-img", "qemu-utils"), ("ssh", "openssh-client"), ("ssh-keygen", "openssh-client")] {
        if !crate::computer::backend::tool_available(tool) {
            m.push(format!("{tool} (sudo apt install {pkg})"));
        }
    }
    if ovmf().is_none() {
        m.push("UEFI firmware for VMs (sudo apt install ovmf)".into());
    }
    let kvm = std::ffi::CString::new("/dev/kvm").unwrap();
    if unsafe { libc::access(kvm.as_ptr(), libc::R_OK | libc::W_OK) } != 0 {
        m.push("access to /dev/kvm (sudo usermod -aG kvm $USER, then sign in again; and virtualization on in the BIOS)".into());
    }
    m
}

/// This PC's time zone (the VM's clock shows the same time).
fn host_timezone() -> String {
    std::fs::read_to_string("/etc/timezone").ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
        .or_else(|| std::fs::read_link("/etc/localtime").ok()
            .and_then(|p| p.to_str().and_then(|s| s.split("zoneinfo/").nth(1)).map(str::to_string)))
        .unwrap_or_else(|| "UTC".into())
}

fn free_port() -> Result<u16, String> {
    let l = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    Ok(l.local_addr().map_err(|e| e.to_string())?.port())
}

/// The QEMU command line (without the program). Pure, so it is unit-tested.
pub fn qemu_args(cfg: &VmConfig, vm_dir: &Path, rt: &Path, code: &Path, ssh_port: u16, restore: bool) -> Vec<String> {
    let p = |f: &str| vm_dir.join(f).display().to_string();
    let mut a: Vec<String> = vec![
        "-name".into(), "am2strata-sandbox,process=am2strata-vm".into(),
        "-machine".into(), "q35,accel=kvm".into(), "-cpu".into(), "host".into(),
        "-smp".into(), cfg.cpus.to_string(), "-m".into(), cfg.memory_mb.to_string(),
        "-nodefaults".into(), "-vga".into(), "none".into(), "-display".into(), "none".into(),
        "-drive".into(), format!("if=pflash,format=raw,readonly=on,file={}", code.display()),
        "-drive".into(), format!("if=pflash,format=raw,file={}", p("vars.fd")),
        "-device".into(), "virtio-balloon-pci,free-page-reporting=on".into(),
        "-drive".into(), format!("if=virtio,file={},format=qcow2,discard=unmap", p("disk.qcow2")),
        "-blockdev".into(), format!("driver=vvfat,node-name=seed,dir={},label=CIDATA,read-only=on", p("seed")),
        "-device".into(), "virtio-blk-pci,drive=seed".into(),
        "-netdev".into(), format!("user,id=n0,hostfwd=tcp:127.0.0.1:{ssh_port}-:22"),
        "-device".into(), "virtio-net-pci,netdev=n0".into(),
        "-qmp".into(), format!("unix:{},server=on,wait=off", rt.join("qmp.sock").display()),
        "-serial".into(), format!("file:{}", p("serial.log")),
    ];
    if restore {
        a.extend(["-incoming".into(), format!("file:{}", p("state.bin"))]);
    }
    a
}

impl Vm {
    pub fn init(data_dir: &Path) -> Arc<Vm> {
        let dir = data_dir.join("sandbox");
        let cfg = std::fs::read(dir.join("vm.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        let vm = Arc::new(Vm {
            phase: Mutex::new(if dir.join("vm/state.bin").exists() { Phase::Saved } else { Phase::Off }),
            dir, rt: runtime_dir(), cfg: Mutex::new(cfg), run: Mutex::new(None), op: Mutex::new(()),
            tunnel: Mutex::new(None), screen: Mutex::new(None), last_used: Mutex::new(Instant::now()),
            viewers: AtomicUsize::new(0), tasks: Mutex::new(vec![]), next_task: Mutex::new(1), busy: Mutex::new(false),
            cpu_sample: Mutex::new(None), last_error: Mutex::new(None),
        });
        let vm = VM.get_or_init(|| vm).clone();
        let w = Arc::downgrade(&vm);
        std::thread::Builder::new().name("vm-idle".into()).spawn(move || loop {
            std::thread::sleep(Duration::from_secs(10));
            let Some(vm) = w.upgrade() else { return };
            vm.idle_check();
        }).ok();
        vm
    }

    pub fn config(&self) -> VmConfig { self.cfg.lock().unwrap().clone() }

    fn vm_dir(&self) -> PathBuf { self.dir.join("vm") }
    fn image(&self) -> PathBuf { self.dir.join("base").join(IMAGE_FILE) }
    fn key(&self) -> PathBuf { self.vm_dir().join("keys/id_ed25519") }

    pub fn phase(&self) -> Phase { self.phase.lock().unwrap().clone() }
    fn set_phase(&self, p: Phase) { *self.phase.lock().unwrap() = p; }

    pub fn touch(&self) { *self.last_used.lock().unwrap() = Instant::now(); }

    pub fn set_config(&self, changes: &Value) -> Result<(VmConfig, bool), String> {
        let new = self.config().merged(changes)?;
        std::fs::create_dir_all(&self.dir).map_err(|e| e.to_string())?;
        std::fs::write(self.dir.join("vm.json"), serde_json::to_vec_pretty(&new).unwrap()).map_err(|e| e.to_string())?;
        let old = std::mem::replace(&mut *self.cfg.lock().unwrap(), new.clone());
        // CPUs and memory apply at the next start; a new size re-runs the desktop setup then
        let restart = old.cpus != new.cpus || old.memory_mb != new.memory_mb || old.width != new.width || old.height != new.height;
        if old.width != new.width || old.height != new.height {
            let _ = std::fs::remove_file(self.vm_dir().join("provisioned"));
        }
        if old.disk_gb < new.disk_gb && self.vm_dir().join("disk.qcow2").exists() && self.run.lock().unwrap().is_none() {
            let _ = Command::new("qemu-img").args(["resize", "-q"]).arg(self.vm_dir().join("disk.qcow2")).arg(format!("{}G", new.disk_gb)).status();
        }
        Ok((new, restart))
    }

    // ------------------------------------------------------------------------------------------------ SSH

    fn ssh_base(&self, port: u16) -> Command {
        let mut c = Command::new("ssh");
        let kh = self.vm_dir().join("keys/known_hosts");
        c.arg("-F").arg("/dev/null").arg("-i").arg(self.key()).arg("-p").arg(port.to_string())
            .args(["-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "HostKeyAlias=am2strata-vm",
                   "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3", "-o", "LogLevel=ERROR"])
            .arg("-o").arg(format!("UserKnownHostsFile={}", kh.display()))
            .env_remove("SSH_AUTH_SOCK").stdin(Stdio::null());
        c
    }

    fn port(&self) -> Result<u16, String> {
        self.run.lock().unwrap().as_ref().map(|r| r.ssh_port).ok_or_else(|| "the sandbox VM is not running".to_string())
    }

    /// Run `script` in the VM (as root when `root`), feeding `input` on stdin when given; each output line goes to
    /// `on_line`. Returns the exit code and the output (its last 64 KB).
    pub fn sh(&self, script: &str, root: bool, input: Option<&[u8]>, timeout: Duration, on_line: &dyn Fn(&str)) -> Result<(i32, String), String> {
        let port = self.port()?;
        self.touch();
        let mut c = self.ssh_base(port);
        c.arg(format!("{}@127.0.0.1", guest::USER));
        let wrapped = format!("export DISPLAY=:1 XDG_RUNTIME_DIR=/run/user/$(id -u {u}) DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u {u})/bus; {{\n{}\n}} 2>&1",
                              script, u = guest::USER);
        let remote = if root { format!("sudo -E bash -c {}", guest::shq(&wrapped)) } else { format!("bash -c {}", guest::shq(&wrapped)) };
        c.arg(remote).stdout(Stdio::piped()).stderr(Stdio::piped());
        if input.is_some() { c.stdin(Stdio::piped()); }
        let mut child = c.spawn().map_err(|e| format!("ssh: {e}"))?;
        if let Some(i) = input {
            let mut w = child.stdin.take().unwrap();
            let data = i.to_vec();
            std::thread::spawn(move || { let _ = w.write_all(&data); });
        }
        let out = child.stdout.take().unwrap();
        let err = child.stderr.take().unwrap();
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        // stderr too (the remote shell's own errors, ssh's), so a failure is never silent
        let tx2 = tx.clone();
        let err_reader = std::thread::spawn(move || {
            for line in BufReader::new(err).lines() {
                let Ok(l) = line else { break };
                if tx2.send(l).is_err() { break; }
            }
        });
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                let Ok(l) = line else { break };
                if tx.send(l).is_err() { break; }
            }
        });
        let t0 = Instant::now();
        let mut text = String::new();
        let status = loop {
            while let Ok(l) = rx.try_recv() {
                on_line(&l);
                text.push_str(&l);
                text.push('\n');
                if text.len() > 128 * 1024 { text.drain(..text.len() - 64 * 1024); }
            }
            match child.try_wait().map_err(|e| e.to_string())? {
                Some(s) => break s,
                None if t0.elapsed() > timeout => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("the command ran longer than {} s and was stopped", timeout.as_secs()));
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        };
        let _ = reader.join();
        let _ = err_reader.join();
        while let Ok(l) = rx.try_recv() {
            on_line(&l);
            text.push_str(&l);
            text.push('\n');
        }
        let code = status.code().unwrap_or(-1);
        if code == 255 && text.lines().count() <= 3 {
            return Err(format!("ssh to the VM: {}", text.trim()));
        }
        Ok((code, text))
    }

    /// A command's output, failing when it exits non-zero.
    pub fn sh_ok(&self, script: &str, root: bool) -> Result<String, String> {
        let (code, out) = self.sh(script, root, None, Duration::from_secs(60), &|_| {})?;
        if code == 0 { Ok(out) } else { Err(format!("exit {code}: {}", out.trim())) }
    }

    fn wait_ssh(&self, limit: Duration) -> Result<(), String> {
        let t0 = Instant::now();
        let port = self.port()?;
        loop {
            let ok = self.ssh_base(port).arg(format!("{}@127.0.0.1", guest::USER)).arg("true")
                .stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
            if ok {
                return Ok(());
            }
            if let Some(r) = self.run.lock().unwrap().as_mut() {
                if let Ok(Some(s)) = r.child.try_wait() {
                    return Err(format!("the VM stopped while starting ({s}); see {}", self.vm_dir().join("qemu.log").display()));
                }
            }
            if t0.elapsed() > limit {
                return Err("the VM did not answer over SSH in time".into());
            }
            std::thread::sleep(Duration::from_millis(150));
        }
    }

    // ------------------------------------------------------------------------------------------ lifecycle

    fn ssh_keygen(path: &Path, comment: &str) -> Result<(), String> {
        if path.exists() {
            return Ok(());
        }
        let ok = Command::new("ssh-keygen").args(["-q", "-t", "ed25519", "-N", "", "-C", comment, "-f"]).arg(path)
            .stdin(Stdio::null()).stdout(Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
        if ok { Ok(()) } else { Err(format!("ssh-keygen could not make {}", path.display())) }
    }

    /// The VM's own files: a copy-on-write disk over the base image, the firmware's variables, our SSH keys (ours and
    /// the VM's host key) and the cloud-init seed.
    fn prepare(&self, cfg: &VmConfig, vars: &Path) -> Result<(), String> {
        let d = self.vm_dir();
        for sub in ["keys", "seed"] {
            std::fs::create_dir_all(d.join(sub)).map_err(|e| e.to_string())?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700));
        }
        Self::ssh_keygen(&d.join("keys/id_ed25519"), "am2strata")?;
        Self::ssh_keygen(&d.join("keys/host_ed25519"), "am2strata-vm")?;
        let read = |f: &str| std::fs::read_to_string(d.join("keys").join(f)).map_err(|e| e.to_string());
        let host_pub = read("host_ed25519.pub")?;
        let host_pub = host_pub.split_whitespace().take(2).collect::<Vec<_>>().join(" ");
        std::fs::write(d.join("keys/known_hosts"), format!("am2strata-vm {host_pub}\n")).map_err(|e| e.to_string())?;
        if !d.join("seed/user-data").exists() {
            std::fs::write(d.join("seed/user-data"), guest::user_data(&read("id_ed25519.pub")?, &read("host_ed25519")?, &host_pub)).map_err(|e| e.to_string())?;
            std::fs::write(d.join("seed/meta-data"), guest::meta_data(&format!("am2-{}", uuid::Uuid::new_v4().simple()))).map_err(|e| e.to_string())?;
        }
        if !d.join("vars.fd").exists() {
            std::fs::copy(vars, d.join("vars.fd")).map_err(|e| format!("the firmware variables: {e}"))?;
        }
        if !d.join("disk.qcow2").exists() {
            let ok = Command::new("qemu-img").args(["create", "-q", "-f", "qcow2", "-F", "qcow2", "-b"]).arg(self.image())
                .arg(d.join("disk.qcow2")).arg(format!("{}G", cfg.disk_gb)).status().map(|s| s.success()).unwrap_or(false);
            if !ok {
                return Err("qemu-img could not make the VM's disk".into());
            }
        }
        Ok(())
    }

    fn spawn_qemu(&self, cfg: &VmConfig, code: &Path, port: u16, restore: bool) -> Result<Child, String> {
        let args = qemu_args(cfg, &self.vm_dir(), &self.rt, code, port, restore);
        let log = std::fs::File::create(self.vm_dir().join("qemu.log")).map_err(|e| e.to_string())?;
        // a cgroup scope: only spare CPU (CPUWeight=idle), a hard memory ceiling, low I/O weight
        let scoped = crate::computer::backend::tool_available("systemd-run")
            && Command::new("systemd-run").args(["--user", "--scope", "-q", "true"]).stdout(Stdio::null()).stderr(Stdio::null())
                .status().map(|s| s.success()).unwrap_or(false);
        let mut c = if scoped {
            let mut c = Command::new("systemd-run");
            c.args(["--user", "--scope", "-q", "--collect", "-p", "CPUWeight=idle", "-p", "IOWeight=10"])
                .arg("-p").arg(format!("MemoryMax={}M", cfg.memory_mb + 768))
                .arg("--unit").arg(format!("am2strata-vm-{}", std::process::id()))
                .arg("--").arg("qemu-system-x86_64");
            c
        } else {
            Command::new("qemu-system-x86_64")
        };
        c.args(&args).stdin(Stdio::null()).stdout(log.try_clone().map_err(|e| e.to_string())?).stderr(log);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            c.process_group(0);
            unsafe {
                c.pre_exec(|| {
                    // the lowest CPU and I/O priority, inherited by QEMU and all its threads
                    libc::setpriority(libc::PRIO_PROCESS, 0, 19);
                    let p = libc::sched_param { sched_priority: 0 };
                    libc::sched_setscheduler(0, libc::SCHED_IDLE, &p);
                    libc::syscall(libc::SYS_ioprio_set, 1, 0, 3 << 13);
                    Ok(())
                });
            }
        }
        c.spawn().map_err(|e| format!("starting QEMU: {e}"))
    }

    fn kill_running(&self) {
        *self.screen.lock().unwrap() = None;
        if let Some(mut t) = self.tunnel.lock().unwrap().take() {
            let _ = t.kill();
            let _ = t.wait();
        }
        if let Some(mut r) = self.run.lock().unwrap().take() {
            let _ = r.child.kill();
            let _ = r.child.wait();
        }
        let _ = std::fs::remove_file(self.rt.join("qmp.sock"));
    }

    /// Start (or resume, or restore) the VM and make sure its desktop is set up. Blocking.
    pub fn start(&self, log: &dyn Fn(String)) -> Result<(), String> {
        let _op = self.op.lock().unwrap();
        self.touch();
        match self.phase() {
            Phase::Running if self.alive() => return Ok(()),
            Phase::Paused if self.alive() => {
                self.qmp(|q| q.cmd("cont", None).map(|_| ()))?;
                self.set_phase(Phase::Running);
                log("resumed".into());
                return Ok(());
            }
            _ => {}
        }
        self.kill_running();
        let missing = missing_requirements();
        if !missing.is_empty() {
            return Err(format!("this PC needs: {}", missing.join("; ")));
        }
        if !self.image().exists() {
            return Err("the Debian image is not downloaded yet: use \"Download the Debian image\" (342 MB) first".into());
        }
        let cfg = self.config();
        if let Some(avail) = mem_available_mb() {
            let need = cfg.memory_mb as u64 + cfg.host_reserve_mb as u64;
            if avail < need {
                return Err(format!("not enough free memory: {avail} MB free, and the VM (up to {} MB) must leave {} MB for the rest of this PC \
                                    (a loaded model). Lower the VM's memory or unload a model", cfg.memory_mb, cfg.host_reserve_mb));
            }
        }
        let (code, vars) = ovmf().ok_or("no UEFI firmware")?;
        std::fs::create_dir_all(&self.rt).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&self.rt, std::fs::Permissions::from_mode(0o700));
        }
        self.prepare(&cfg, &vars)?;
        self.set_phase(Phase::Starting);
        let state = self.vm_dir().join("state.bin");
        let fp = std::fs::read_to_string(self.vm_dir().join("state.fingerprint")).unwrap_or_default();
        let can_restore = state.exists() && fp.trim() == cfg.fingerprint();
        if state.exists() && !can_restore {
            log("the VM's CPUs or memory changed since it was saved: starting it fresh (open apps are closed)".into());
            let _ = std::fs::remove_file(&state);
        }
        let r = self.boot(&cfg, &code, can_restore, log).or_else(|e| {
            if can_restore {
                log(format!("restoring the saved VM failed ({e}); starting it fresh"));
                self.kill_running();
                let _ = std::fs::remove_file(&state);
                self.boot(&cfg, &code, false, log)
            } else {
                Err(e)
            }
        });
        if let Err(e) = r {
            self.kill_running();
            self.set_phase(if state.exists() { Phase::Saved } else { Phase::Off });
            *self.last_error.lock().unwrap() = Some(e.clone());
            return Err(e);
        }
        let done = std::fs::read_to_string(self.vm_dir().join("provisioned")).ok().and_then(|s| s.trim().parse::<u32>().ok());
        if done != Some(guest::PROVISION_VERSION) {
            self.set_phase(Phase::Setup);
            log("setting up the desktop (once; a few minutes): Xvnc, openbox, Flatpak with Flathub, AppImage support".into());
            let (rc, out) = self.sh(&guest::provision_script(cfg.width, cfg.height, &host_timezone()), true, None, Duration::from_secs(1800), &|l| log(l.to_string()))?;
            if rc != 0 {
                self.set_phase(Phase::Running);
                return Err(format!("the desktop setup failed (exit {rc}): {}", out.lines().rev().take(5).collect::<Vec<_>>().join(" | ")));
            }
            std::fs::write(self.vm_dir().join("provisioned"), guest::PROVISION_VERSION.to_string()).ok();
            if cfg.block_ads {
                if let Ok(s) = guest::ads_script(true, &cfg.ads_lists, &cfg.ads_allow) {
                    let (_, out) = self.sh(&s, true, None, Duration::from_secs(300), &|_| {})?;
                    log(out.trim().to_string());
                }
            }
        }
        self.set_phase(Phase::Running);
        *self.last_error.lock().unwrap() = None;
        self.touch();
        Ok(())
    }

    fn boot(&self, cfg: &VmConfig, code: &Path, restore: bool, log: &dyn Fn(String)) -> Result<(), String> {
        let t0 = Instant::now();
        let _ = std::fs::remove_file(self.rt.join("qmp.sock"));
        let port = free_port()?;
        let child = self.spawn_qemu(cfg, code, port, restore)?;
        let q = qmp::Qmp::connect(&self.rt.join("qmp.sock"), Duration::from_secs(15));
        let mut q = match q {
            Ok(q) => q,
            Err(e) => {
                let mut child = child;
                let _ = child.kill();
                let log_text = std::fs::read_to_string(self.vm_dir().join("qemu.log")).unwrap_or_default();
                return Err(format!("{e}: {}", log_text.lines().last().unwrap_or("")));
            }
        };
        if restore {
            q.wait_migration(Duration::from_secs(120))?;
            q.cmd("cont", None)?;
        }
        *self.run.lock().unwrap() = Some(Running { child, ssh_port: port, qmp: q });
        self.wait_ssh(Duration::from_secs(if restore { 30 } else { 180 }))?;
        if restore {
            // consumed: the disk moves on from here, so this state must never be restored again
            let _ = std::fs::remove_file(self.vm_dir().join("state.bin"));
            let _ = self.sh(&format!("date -u -s @{} >/dev/null", now()), true, None, Duration::from_secs(10), &|_| {});
            log(format!("restored the saved VM in {:.1} s", t0.elapsed().as_secs_f64()));
        } else {
            log(format!("booted in {:.1} s", t0.elapsed().as_secs_f64()));
        }
        Ok(())
    }

    fn alive(&self) -> bool {
        self.run.lock().unwrap().as_mut().is_some_and(|r| matches!(r.child.try_wait(), Ok(None)))
    }

    fn qmp<T>(&self, f: impl FnOnce(&mut qmp::Qmp) -> Result<T, String>) -> Result<T, String> {
        let mut g = self.run.lock().unwrap();
        let r = g.as_mut().ok_or("the sandbox VM is not running")?;
        f(&mut r.qmp)
    }

    pub fn pause(&self) -> Result<(), String> {
        let _op = self.op.lock().unwrap();
        if self.phase() != Phase::Running {
            return Ok(());
        }
        *self.screen.lock().unwrap() = None;
        self.qmp(|q| q.cmd("stop", None).map(|_| ()))?;
        self.set_phase(Phase::Paused);
        Ok(())
    }

    /// Save the running VM to disk and quit (the next start restores it in about a second); on failure, shut it
    /// down cleanly instead.
    pub fn stop(&self, log: &dyn Fn(String)) -> Result<(), String> {
        let _op = self.op.lock().unwrap();
        if !self.alive() {
            self.kill_running();
            return Ok(());
        }
        self.set_phase(Phase::Stopping);
        *self.screen.lock().unwrap() = None;
        if let Some(mut t) = self.tunnel.lock().unwrap().take() {
            let _ = t.kill();
            let _ = t.wait();
        }
        // the guest's caches are dropped first, so the saved state is small
        let _ = self.sh("sync; echo 3 > /proc/sys/vm/drop_caches", true, None, Duration::from_secs(20), &|_| {});
        let t0 = Instant::now();
        let state = self.vm_dir().join("state.bin");
        let saved = self.qmp(|q| q.save_to(&state));
        match saved {
            Ok(()) => {
                std::fs::write(self.vm_dir().join("state.fingerprint"), self.config().fingerprint()).ok();
                let _ = self.qmp(|q| q.cmd("quit", None));
                self.kill_running();
                self.set_phase(Phase::Saved);
                log(format!("saved in {:.1} s ({} MB); the next start restores it", t0.elapsed().as_secs_f64(),
                            std::fs::metadata(&state).map(|m| m.len() >> 20).unwrap_or(0)));
                Ok(())
            }
            Err(e) => {
                log(format!("saving failed ({e}); shutting it down cleanly instead"));
                let _ = std::fs::remove_file(&state);
                let _ = self.qmp(|q| q.cmd("cont", None));
                drop(_op);
                self.shutdown_inner(log)
            }
        }
    }

    /// A clean power-off (open apps are closed; the next start boots).
    pub fn shutdown(&self, log: &dyn Fn(String)) -> Result<(), String> {
        let _op = self.op.lock().unwrap();
        self.shutdown_inner(log)
    }

    fn shutdown_inner(&self, log: &dyn Fn(String)) -> Result<(), String> {
        if self.alive() {
            self.set_phase(Phase::Stopping);
            let _ = self.qmp(|q| q.cmd("system_powerdown", None));
            let t0 = Instant::now();
            while self.alive() && t0.elapsed() < Duration::from_secs(60) {
                std::thread::sleep(Duration::from_millis(200));
            }
            log(format!("shut down in {:.1} s", t0.elapsed().as_secs_f64()));
        }
        self.kill_running();
        let _ = std::fs::remove_file(self.vm_dir().join("state.bin"));
        self.set_phase(Phase::Off);
        Ok(())
    }

    /// Throw the VM away (its disk, apps, files and keys); the next start makes a new one from the base image.
    pub fn reset(&self) -> Result<(), String> {
        let _op = self.op.lock().unwrap();
        self.kill_running();
        let d = self.vm_dir();
        if d.exists() {
            std::fs::remove_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        self.set_phase(Phase::Off);
        Ok(())
    }

    /// Stop at server exit: save it, so it comes back as it was.
    pub fn shutdown_for_exit(&self) {
        if self.alive() {
            let _ = self.stop(&|m| tracing::info!("sandbox VM: {m}"));
        }
    }

    fn idle_check(&self) {
        let cfg = self.config();
        let idle = self.last_used.lock().unwrap().elapsed();
        if self.viewers.load(Ordering::SeqCst) > 0 || *self.busy.lock().unwrap() {
            return;
        }
        let phase = self.phase();
        if cfg.idle_stop_s > 0 && idle > Duration::from_secs(cfg.idle_stop_s) && matches!(phase, Phase::Running | Phase::Paused) {
            tracing::info!("sandbox VM: unused for {} s; saving it and stopping", idle.as_secs());
            if phase == Phase::Paused {
                let _ = self.qmp(|q| q.cmd("cont", None));
                self.set_phase(Phase::Running);
            }
            let _ = self.stop(&|m| tracing::info!("sandbox VM: {m}"));
        } else if cfg.idle_pause_s > 0 && idle > Duration::from_secs(cfg.idle_pause_s) && phase == Phase::Running {
            tracing::info!("sandbox VM: unused for {} s; pausing its CPUs", idle.as_secs());
            let _ = self.pause();
        }
    }

    /// Running (started, resumed or restored as needed): what every use of the VM calls first.
    pub fn ensure(&self) -> Result<(), String> {
        if self.phase() == Phase::Running && self.alive() {
            self.touch();
            return Ok(());
        }
        self.start(&|m| tracing::info!("sandbox VM: {m}"))
    }

    // ------------------------------------------------------------------------------------------- screen

    /// The VM's screen (connecting the SSH tunnel to its Xvnc when needed).
    pub fn screen(&self) -> Result<Arc<rfb::Rfb>, String> {
        self.ensure()?;
        if let Some(s) = self.screen.lock().unwrap().as_ref().filter(|s| s.alive()) {
            return Ok(s.clone());
        }
        let sock = self.rt.join("rfb.sock");
        let port = self.port()?;
        let mut tunnel = self.tunnel.lock().unwrap();
        let live = tunnel.as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None)));
        if !live || !sock.exists() {
            if let Some(mut t) = tunnel.take() {
                let _ = t.kill();
                let _ = t.wait();
            }
            let _ = std::fs::remove_file(&sock);
            let child = self.ssh_base(port)
                .args(["-N", "-o", "ExitOnForwardFailure=yes", "-o", "StreamLocalBindMask=0177", "-L"])
                .arg(format!("{}:127.0.0.1:5901", sock.display()))
                .arg(format!("{}@127.0.0.1", guest::USER)).stdout(Stdio::null()).stderr(Stdio::null())
                .spawn().map_err(|e| format!("ssh tunnel: {e}"))?;
            *tunnel = Some(child);
        }
        drop(tunnel);
        let t0 = Instant::now();
        let rfb = loop {
            match sock.exists().then(|| rfb::Rfb::connect(&sock, Duration::from_millis(100))) {
                Some(Ok(r)) => break r,
                other if t0.elapsed() > Duration::from_secs(15) => {
                    let why = match other { Some(Err(e)) => e, _ => "the tunnel did not open".into() };
                    return Err(format!("the VM's screen: {why} (is the desktop set up? try Start again)"));
                }
                _ => std::thread::sleep(Duration::from_millis(100)),
            }
        };
        *self.screen.lock().unwrap() = Some(rfb.clone());
        Ok(rfb)
    }

    // ---------------------------------------------------------------------------------------------- apps

    pub fn launch(&self, command: &str) -> Result<(), String> {
        self.ensure()?;
        self.sh_ok(&format!("am2-run sh -c {}", guest::shq(command)), false).map(|_| ())
    }

    pub fn open_app(&self, name: &str) -> Result<(), String> {
        let cmd = guest::app(name).map(|a| a.launch).filter(|l| !l.is_empty()).map(str::to_string).unwrap_or_else(|| name.to_string());
        self.launch(&cmd)
    }

    pub fn catalog(&self) -> Result<Value, String> {
        let mut installed = HashMap::new();
        if self.phase() == Phase::Paused {
            self.ensure()?; // resuming takes milliseconds
        }
        if self.phase() == Phase::Running && self.alive() {
            for l in self.sh_ok(&guest::catalog_check_script(), false)?.lines() {
                if let Some((id, v)) = l.split_once(' ') {
                    installed.insert(id.to_string(), v.trim() == "1");
                }
            }
        }
        Ok(guest::catalog_json(&installed))
    }

    /// Block or allow ads in the VM (applied now, kept for a new VM); what the VM says it did.
    pub fn set_ads(&self, block: bool, lists: &[String], allow: &[String]) -> Result<String, String> {
        let lists: Vec<String> = if lists.is_empty() { self.config().ads_lists } else { lists.to_vec() };
        let script = guest::ads_script(block, &lists, allow)?;
        self.ensure()?;
        let (code, out) = self.sh(&script, true, None, Duration::from_secs(300), &|_| {})?;
        if code != 0 {
            return Err(format!("exit {code}: {}", out.trim()));
        }
        self.set_config(&json!({"block_ads": block, "ads_lists": lists, "ads_allow": if block { allow.to_vec() } else { vec![] }}))?;
        Ok(out.trim().to_string())
    }

    /// What the VM itself says: (blocking, its lists, the domains let through). None when it is not running.
    pub fn ads_state(&self) -> Option<(bool, Vec<String>, Vec<String>)> {
        if self.phase() != Phase::Running || !self.alive() {
            return None;
        }
        let out = self.sh_ok("am2-ads status", false).ok()?;
        let out = out.trim();
        let field = |k: &str| -> Vec<String> {
            out.split_whitespace().find_map(|w| w.strip_prefix(k)).unwrap_or("").split(',').filter(|x| !x.is_empty()).map(str::to_string).collect()
        };
        Some((out.starts_with("blocked"), field("lists="), field("allow=")))
    }

    pub fn clipboard_get(&self) -> Result<String, String> {
        self.ensure()?;
        let (code, out) = self.sh("xclip -selection clipboard -o", false, None, Duration::from_secs(10), &|_| {})?;
        Ok(if code == 0 { out.strip_suffix('\n').unwrap_or(&out).to_string() } else { String::new() })
    }

    pub fn clipboard_set(&self, text: &str) -> Result<(), String> {
        self.ensure()?;
        // xclip stays to serve the selection; it is detached so the command returns
        let (code, out) = self.sh("f=$(mktemp); cat > \"$f\"; setsid -f sh -c \"xclip -selection clipboard -i < '$f'; rm -f '$f'\" >/dev/null 2>&1",
                                  false, Some(text.as_bytes()), Duration::from_secs(10), &|_| {})?;
        if code == 0 { Ok(()) } else { Err(out) }
    }

    pub fn type_text(&self, text: &str) -> Result<(), String> {
        self.ensure()?;
        let (code, out) = self.sh("xdotool type --clearmodifiers --delay 3 --file -", false, Some(text.as_bytes()), Duration::from_secs(120), &|_| {})?;
        if code == 0 { Ok(()) } else { Err(out) }
    }

    // ---------------------------------------------------------------------------------------------- tasks

    /// Run `f` as the one background task (a second is refused while one runs); its id.
    pub fn task(self: &Arc<Self>, what: &str, f: impl FnOnce(&Arc<Vm>, &dyn Fn(String)) -> Result<String, String> + Send + 'static) -> Result<u64, String> {
        {
            let mut b = self.busy.lock().unwrap();
            if *b {
                let cur = self.tasks.lock().unwrap().iter().rev().find(|t| t.state == "running").map(|t| t.what.clone()).unwrap_or_default();
                return Err(format!("the VM is busy ({cur}); try again when it is done"));
            }
            *b = true;
        }
        let id = { let mut n = self.next_task.lock().unwrap(); let id = *n; *n += 1; id };
        {
            let mut tasks = self.tasks.lock().unwrap();
            tasks.push(Task { id, what: what.into(), state: "running", log: vec![], error: None, result: None, started: now(), finished: None });
            if tasks.len() > 20 { tasks.remove(0); }
        }
        let me = self.clone();
        std::thread::Builder::new().name("vm-task".into()).spawn(move || {
            let m2 = me.clone();
            let log = move |l: String| {
                m2.touch();
                if let Some(t) = m2.tasks.lock().unwrap().iter_mut().find(|t| t.id == id) {
                    t.log.push(l);
                    if t.log.len() > 400 { t.log.drain(..100); }
                }
            };
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&me, &log)))
                .unwrap_or_else(|_| Err("the task crashed".into()));
            if let Some(t) = me.tasks.lock().unwrap().iter_mut().find(|t| t.id == id) {
                t.finished = Some(now());
                match r {
                    Ok(res) => { t.state = "done"; t.result = Some(res); }
                    Err(e) => { t.state = "failed"; t.error = Some(e); }
                }
            }
            *me.busy.lock().unwrap() = false;
            me.touch();
        }).map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn task_view(&self, id: u64) -> Option<Task> {
        self.tasks.lock().unwrap().iter().find(|t| t.id == id).cloned()
    }

    /// Install or remove a catalog app, or a package by kind (apt, flatpak, deb, appimage). Blocking.
    pub fn install(&self, app_or_kind: &str, target: &str, install: bool, log: &dyn Fn(String)) -> Result<String, String> {
        self.ensure()?;
        let script = match guest::app(app_or_kind) {
            Some(a) => guest::app_script(a, install),
            None => guest::package_script(app_or_kind, target, install)?,
        };
        let (code, out) = self.sh(&script, true, None, Duration::from_secs(3600), &|l| log(l.to_string()))?;
        let _ = self.sh("sync; echo 3 > /proc/sys/vm/drop_caches", true, None, Duration::from_secs(20), &|_| {});
        if code == 0 {
            Ok(format!("{} {}", if install { "installed" } else { "removed" }, if target.is_empty() { app_or_kind } else { target }))
        } else {
            Err(format!("exit {code}: {}", out.lines().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n")))
        }
    }

    /// Download the Debian cloud image and check it against Debian's SHA512SUMS. Blocking.
    pub fn download_image(&self, log: &dyn Fn(String)) -> Result<String, String> {
        use sha2::Digest;
        let dst = self.image();
        if dst.exists() {
            return Ok("the image is already here".into());
        }
        std::fs::create_dir_all(dst.parent().unwrap()).map_err(|e| e.to_string())?;
        let client = reqwest::blocking::Client::builder().timeout(None).connect_timeout(Duration::from_secs(20)).build().map_err(|e| e.to_string())?;
        let sums = client.get(SUMS_URL).send().and_then(|r| r.error_for_status()).and_then(|r| r.text()).map_err(|e| format!("{SUMS_URL}: {e}"))?;
        let want = sums.lines().find(|l| l.ends_with(&format!("  {IMAGE_FILE}"))).and_then(|l| l.split_whitespace().next())
            .ok_or("Debian's SHA512SUMS does not list the image")?.to_lowercase();
        let mut resp = client.get(IMAGE_URL).send().and_then(|r| r.error_for_status()).map_err(|e| format!("{IMAGE_URL}: {e}"))?;
        let total = resp.content_length().unwrap_or(0);
        let part = dst.with_extension("qcow2.part");
        let mut f = std::fs::File::create(&part).map_err(|e| e.to_string())?;
        let mut h = sha2::Sha512::new();
        let (mut got, mut buf, mut shown) = (0u64, vec![0u8; 1 << 20], 0u64);
        loop {
            let n = resp.read(&mut buf).map_err(|e| format!("downloading: {e}"))?;
            if n == 0 { break; }
            h.update(&buf[..n]);
            f.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            got += n as u64;
            if total > 0 && got * 20 / total > shown {
                shown = got * 20 / total;
                log(format!("{} of {} MB", got >> 20, total >> 20));
            }
        }
        let have = h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();
        if have != want {
            let _ = std::fs::remove_file(&part);
            return Err("the download does not match Debian's checksum; it was deleted".into());
        }
        std::fs::rename(&part, &dst).map_err(|e| e.to_string())?;
        Ok(format!("downloaded and verified {IMAGE_FILE} ({} MB)", got >> 20))
    }

    // --------------------------------------------------------------------------------------------- status

    fn usage(&self) -> Value {
        let pid = self.run.lock().unwrap().as_ref().map(|r| r.child.id());
        let Some(pid) = pid else { return json!(null) };
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
        let rss_mb = status.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok()).map(|kb| kb / 1024);
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
        let ticks = stat.rsplit(')').next().map(|r| r.split_whitespace().collect::<Vec<_>>())
            .and_then(|f| Some(f.get(11)?.parse::<u64>().ok()? + f.get(12)?.parse::<u64>().ok()?));
        let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as f64;
        let cpu = ticks.and_then(|t| {
            let mut s = self.cpu_sample.lock().unwrap();
            let prev = s.replace((Instant::now(), t));
            prev.map(|(at, pt)| (t.saturating_sub(pt) as f64 / hz) / at.elapsed().as_secs_f64().max(0.1) * 100.0)
        });
        json!({"pid": pid, "rss_mb": rss_mb, "cpu_percent": cpu.map(|c| (c * 10.0).round() / 10.0)})
    }

    pub fn status(&self) -> Value {
        let d = self.vm_dir();
        let disk = std::fs::metadata(d.join("disk.qcow2")).ok().map(|m| {
            #[cfg(unix)]
            { use std::os::unix::fs::MetadataExt; m.blocks() * 512 >> 20 }
            #[cfg(not(unix))]
            { m.len() >> 20 }
        });
        let running_task = self.tasks.lock().unwrap().iter().rev().find(|t| t.state == "running").cloned();
        let last_task = self.tasks.lock().unwrap().last().cloned();
        json!({
            "phase": self.phase().as_str(), "config": self.config(), "image": self.image().exists(),
            "image_mb": 326, "requirements_missing": missing_requirements(), "provisioned": d.join("provisioned").exists(),
            "saved_state": d.join("state.bin").exists(), "disk_used_mb": disk, "host": self.usage(),
            "host_free_mb": mem_available_mb(), "idle_s": self.last_used.lock().unwrap().elapsed().as_secs(),
            "viewers": self.viewers.load(Ordering::SeqCst), "task": running_task.or(last_task),
            "error": self.last_error.lock().unwrap().clone(), "folder": self.dir.display().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qemu_runs_minimal_and_private() {
        let cfg = VmConfig::default();
        let a = qemu_args(&cfg, Path::new("/d/vm"), Path::new("/run/user/1/am2strata-vm"), Path::new("/fw/code.fd"), 2222, false);
        let s = a.join(" ");
        assert!(s.contains("-machine q35,accel=kvm") && s.contains("-smp 2") && s.contains("-m 2048"), "{s}");
        assert!(s.contains("-vga none") && s.contains("-display none") && s.contains("-nodefaults"), "no GPU, no window: {s}");
        assert!(s.contains("free-page-reporting=on"), "memory goes back to the host");
        assert!(s.contains("hostfwd=tcp:127.0.0.1:2222-:22"), "SSH only on this PC's loopback");
        assert!(!s.contains("-vnc") && !s.contains("0.0.0.0"), "nothing listens on the network");
        assert!(!s.contains("-incoming"));
        let r = qemu_args(&cfg, Path::new("/d/vm"), Path::new("/r"), Path::new("/fw/code.fd"), 2222, true).join(" ");
        assert!(r.ends_with("-incoming file:/d/vm/state.bin"), "{r}");
    }

    #[test]
    fn settings_are_checked_and_clamped() {
        let c = VmConfig::default();
        let n = c.merged(&json!({"memory_mb": 100, "cpus": 64, "idle_pause_s": 0})).unwrap();
        assert_eq!((n.memory_mb, n.cpus, n.idle_pause_s), (768, 16, 0));
        assert!(c.merged(&json!({"gpu": true})).is_err(), "unknown settings are refused");
        assert!(c.merged(&json!({"cpus": "two"})).is_err());
        assert_ne!(c.fingerprint(), n.fingerprint(), "a restore needs the same CPUs and memory");
    }
}
