//! `am2strata-run doctor`: which optional features this PC is ready for, and exactly what each one still needs.
//! The installers (deploy/release/install.sh, install.ps1) print this report, so they and the product agree.
//!
//! Nothing is installed or changed here. `report` is pure: it takes what was found on the PC (`Host`) and returns
//! the features, so the tests give it any system. `Host::probe` is the only part that looks at the real one.

use std::path::{Path, PathBuf};

/// What a feature can still need: a program (looked up on PATH) or a file, and the package that brings it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Need {
    /// the program's name, or a short name for the file
    pub what: &'static str,
    pub apt: &'static str,
    pub dnf: &'static str,
    pub pacman: &'static str,
    pub brew: &'static str,
    pub winget: &'static str,
}

const fn need(what: &'static str, apt: &'static str, dnf: &'static str, pacman: &'static str, brew: &'static str, winget: &'static str) -> Need {
    Need { what, apt, dnf, pacman, brew, winget }
}

const PDFTOTEXT: Need = need("pdftotext", "poppler-utils", "poppler-utils", "poppler", "poppler", "oschwartz10612.Poppler");
const PDFTOPPM: Need = need("pdftoppm", "poppler-utils", "poppler-utils", "poppler", "poppler", "oschwartz10612.Poppler");
const FFMPEG: Need = need("ffmpeg", "ffmpeg", "ffmpeg-free", "ffmpeg", "ffmpeg", "Gyan.FFmpeg");
const XVFB: Need = need("Xvfb", "xvfb", "xorg-x11-server-Xvfb", "xorg-server-xvfb", "", "");
const XDOTOOL: Need = need("xdotool", "xdotool", "xdotool", "xdotool", "", "");
const XCLIP: Need = need("xclip", "xclip", "xclip", "xclip", "", "");
const BWRAP: Need = need("bwrap", "bubblewrap", "bubblewrap", "bubblewrap", "", "");
const QEMU: Need = need("qemu-system-x86_64", "qemu-system-x86", "qemu-system-x86", "qemu-system-x86", "", "");
const QEMU_IMG: Need = need("qemu-img", "qemu-utils", "qemu-img", "qemu-img", "", "");
const SSH: Need = need("ssh", "openssh-client", "openssh-clients", "openssh", "", "");
const SSH_KEYGEN: Need = need("ssh-keygen", "openssh-client", "openssh-clients", "openssh", "", "");
/// the VM's UEFI firmware: a file, not a program (`Host::ovmf`)
const OVMF: Need = need("OVMF (UEFI firmware)", "ovmf", "edk2-ovmf", "edk2-ovmf", "", "");
/// any Chrome, Chromium or Edge, or the headless shell the installer fetches (`Host::browser`). No package: the
/// name differs per distribution (and Ubuntu's is a snap, which the sandbox desktop cannot show).
const BROWSER: Need = need("Chrome, Chromium or Edge", "", "", "", "", "");

impl Need {
    /// The package for a package manager ("" when it has none: the feature is not for that system).
    pub fn package(&self, manager: &str) -> &'static str {
        match manager {
            "apt" | "apt-get" => self.apt,
            "dnf" => self.dnf,
            "pacman" => self.pacman,
            "brew" => self.brew,
            "winget" => self.winget,
            _ => "",
        }
    }
}

/// What was found on this PC.
pub struct Host {
    /// "linux", "macos" or "windows" (`std::env::consts::OS`)
    pub os: String,
    /// "x86_64" or "aarch64"
    pub arch: String,
    /// programs found, by `Need::what` (on PATH, or the copy a download bundles)
    pub tools: Vec<&'static str>,
    /// am2strata-workshop is beside this program
    pub workshop: bool,
    /// a browser the browser tool can start
    pub browser: bool,
    /// the VM's UEFI firmware is installed
    pub ovmf: bool,
    /// /dev/kvm opens for reading and writing
    pub kvm: bool,
    /// the first package manager found: apt, dnf, pacman, brew or winget ("" for none)
    pub manager: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum State {
    Ready,
    Missing,
    /// not on this operating system or processor at all
    NotHere,
}

#[derive(Debug)]
pub struct Feature {
    pub name: &'static str,
    pub state: State,
    /// why it is ready, or why it is not for this system
    pub note: String,
    /// programs and files still needed
    pub missing: Vec<Need>,
    /// things no package fixes (a group to join, a BIOS setting), as a sentence each
    pub steps: Vec<String>,
}

fn feature(name: &'static str, host: &Host, needs: &[Need], ready: &str) -> Feature {
    let missing: Vec<Need> = needs.iter().copied().filter(|n| !host.has(n)).collect();
    let state = if missing.is_empty() { State::Ready } else { State::Missing };
    Feature { name, state, note: if missing.is_empty() { ready.to_string() } else { String::new() }, missing, steps: Vec::new() }
}

fn not_here(name: &'static str, why: &str) -> Feature {
    Feature { name, state: State::NotHere, note: why.to_string(), missing: Vec::new(), steps: Vec::new() }
}

impl Host {
    fn has(&self, n: &Need) -> bool {
        if *n == OVMF {
            self.ovmf
        } else if *n == BROWSER {
            self.browser
        } else {
            self.tools.contains(&n.what)
        }
    }
}

/// Every optional feature, in the order the report prints them.
pub fn report(host: &Host) -> Vec<Feature> {
    let linux = host.os == "linux";
    let mut out = vec![Feature { name: "HTTPS", state: State::Ready, note: "built in (am2strata-run tls self-signed)".into(), missing: Vec::new(), steps: Vec::new() }];
    out.push(if host.workshop {
        Feature { name: "Workshop", state: State::Ready, note: "included (convert, quantize, merge, training data)".into(), missing: Vec::new(), steps: Vec::new() }
    } else {
        Feature { name: "Workshop", state: State::Missing, note: "am2strata-workshop is not beside this program: take the full download".into(), missing: Vec::new(), steps: Vec::new() }
    });
    out.push(feature("PDFs in chats", host, &[PDFTOTEXT, PDFTOPPM], if host.os == "windows" { "Poppler found (the download carries it)" } else { "poppler found" }));
    out.push(feature("Video from API clients", host, &[FFMPEG], "ffmpeg found"));
    let mut browser = feature("Browser tool", host, &[BROWSER], "a browser found");
    if browser.state == State::Missing {
        browser.steps.push("install one (the installer fetches a headless Chrome where there is one)".into());
    }
    out.push(browser);
    out.push(if linux {
        feature("Python in the canvas", host, &[BWRAP], "bubblewrap found")
    } else {
        not_here("Python in the canvas", "Linux only (its sandbox is bubblewrap)")
    });
    out.push(if linux {
        feature("Computer tab (sandbox desktop)", host, &[XVFB, XDOTOOL, XCLIP], "Xvfb, xdotool and xclip found")
    } else {
        not_here("Computer tab (sandbox desktop)", "Linux only (it is a virtual X display)")
    });
    out.push(if linux && host.arch == "x86_64" {
        let mut f = feature("Sandbox VM", host, &[QEMU, QEMU_IMG, OVMF, SSH, SSH_KEYGEN], "QEMU, OVMF, ssh and /dev/kvm ready");
        if !host.kvm {
            f.state = State::Missing;
            f.note.clear();
            f.steps.push("access to /dev/kvm: sudo usermod -aG kvm $USER, then sign in again (and virtualization on in the BIOS)".into());
        }
        f
    } else {
        not_here("Sandbox VM", "Linux on x86_64 only (QEMU with KVM)")
    });
    out
}

/// The packages `manager` installs for everything missing, each once, in the report's order.
pub fn packages(features: &[Feature], manager: &str) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for n in features.iter().flat_map(|f| &f.missing) {
        let p = n.package(manager);
        if !p.is_empty() && !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// The command that installs `pkgs` with `manager` (None: nothing to install, or no such manager).
pub fn install_command(manager: &str, pkgs: &[&str]) -> Option<String> {
    if pkgs.is_empty() {
        return None;
    }
    let list = pkgs.join(" ");
    Some(match manager {
        "apt" | "apt-get" => format!("sudo apt-get install -y {list}"),
        "dnf" => format!("sudo dnf install -y {list}"),
        "pacman" => format!("sudo pacman -S --needed --noconfirm {list}"),
        "brew" => format!("brew install {list}"),
        "winget" => pkgs.iter().map(|p| format!("winget install -e --id {p}")).collect::<Vec<_>>().join("; "),
        _ => return None,
    })
}

/// The report as text.
pub fn render(host: &Host, features: &[Feature], version: &str) -> String {
    let mut s = format!("am2Strata {version} on {} {}\n\n", host.os, host.arch);
    for f in features {
        let state = match f.state { State::Ready => "ready", State::Missing => "missing", State::NotHere => "not here" };
        let mut detail = f.note.clone();
        if !f.missing.is_empty() {
            detail = format!("needs {}", f.missing.iter().map(|n| n.what).collect::<Vec<_>>().join(", "));
        }
        for step in &f.steps {
            if !detail.is_empty() {
                detail.push_str("; ");
            }
            detail.push_str(step);
        }
        s.push_str(&format!("  {state:<9} {:<31} {detail}\n", f.name));
    }
    let pkgs = packages(features, &host.manager);
    s.push('\n');
    if let Some(cmd) = install_command(&host.manager, &pkgs) {
        s.push_str(&format!("To add what is missing:\n  {cmd}\n"));
    } else if features.iter().any(|f| !f.missing.is_empty()) {
        s.push_str(match host.os.as_str() {
            "macos" => "To add what is missing, install Homebrew (https://brew.sh), then run this again.\n",
            "windows" => "To add what is missing, install it and put its folder on PATH (winget, when this PC has it, does both).\n",
            _ => "To add what is missing, install it with this system's package manager.\n",
        });
    } else if features.iter().all(|f| f.state != State::Missing) {
        s.push_str("Nothing is missing for the features this system supports.\n");
    }
    s
}

fn on_path(names: &[String]) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|d| names.iter().map(|n| d.join(n)).find(|p| p.is_file()))
}

fn exe_names(name: &str) -> Vec<String> {
    if cfg!(windows) { vec![format!("{name}.exe"), format!("{name}.cmd"), format!("{name}.bat")] } else { vec![name.to_string()] }
}

/// As browser.rs's find_chrome: the headless shell the installers fetch, else an installed Chrome, Chromium or Edge.
fn browser(data_dir: Option<&Path>) -> bool {
    let fetched = data_dir.and_then(|d| std::fs::read_dir(d.join("browser")).ok()).is_some_and(|rd| {
        rd.flatten().any(|e| e.file_name().to_string_lossy().starts_with("chrome-headless-shell-"))
    });
    let names: Vec<String> = if cfg!(windows) { vec!["chrome.exe".into(), "msedge.exe".into()] }
        else { ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable"].map(String::from).to_vec() };
    let mut known: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        for var in ["ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"] {
            if let Some(base) = std::env::var_os(var).map(PathBuf::from) {
                known.push(base.join("Google/Chrome/Application/chrome.exe"));
                known.push(base.join("Microsoft/Edge/Application/msedge.exe"));
            }
        }
    } else if cfg!(target_os = "macos") {
        for app in ["Google Chrome.app/Contents/MacOS/Google Chrome", "Chromium.app/Contents/MacOS/Chromium", "Microsoft Edge.app/Contents/MacOS/Microsoft Edge"] {
            known.push(Path::new("/Applications").join(app));
        }
    }
    fetched || on_path(&names).is_some() || known.iter().any(|p| p.is_file())
}

impl Host {
    /// Look at this PC. `data_dir`: the server's data folder (where the installer puts the browser), when known.
    pub fn probe(data_dir: Option<&Path>) -> Host {
        let here = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf));
        let beside = |name: &str| here.as_ref().is_some_and(|d| d.join(format!("{name}{}", std::env::consts::EXE_SUFFIX)).is_file());
        // Poppler: the copy a download bundles counts (media.rs looks in poppler/bin beside or above this folder)
        let bundled = |name: &str| here.as_ref().is_some_and(|d| {
            let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
            d.join("poppler/bin").join(&file).is_file() || d.join("../poppler/bin").join(&file).is_file()
        });
        let all = [PDFTOTEXT, PDFTOPPM, FFMPEG, XVFB, XDOTOOL, XCLIP, QEMU, QEMU_IMG, SSH, SSH_KEYGEN];
        let mut tools: Vec<&'static str> = all.iter().filter(|n| on_path(&exe_names(n.what)).is_some() || bundled(n.what)).map(|n| n.what).collect();
        // code_run.rs starts exactly this path
        if Path::new("/usr/bin/bwrap").is_file() {
            tools.push(BWRAP.what);
        }
        let firmware = |names: &[&str]| names.iter().any(|p| Path::new(p).is_file());
        let ovmf = firmware(&["/usr/share/OVMF/OVMF_CODE_4M.fd", "/usr/share/OVMF/OVMF_CODE.fd", "/usr/share/edk2/ovmf/OVMF_CODE.fd",
                              "/usr/share/edk2/x64/OVMF_CODE.4m.fd", "/usr/share/qemu/edk2-x86_64-code.fd"])
            && firmware(&["/usr/share/OVMF/OVMF_VARS_4M.fd", "/usr/share/OVMF/OVMF_VARS.fd", "/usr/share/edk2/ovmf/OVMF_VARS.fd",
                          "/usr/share/edk2/x64/OVMF_VARS.4m.fd", "/usr/share/qemu/edk2-i386-vars.fd"]);
        let managers: &[&str] = match std::env::consts::OS {
            "linux" => &["apt-get", "dnf", "pacman"],
            "macos" => &["brew"],
            "windows" => &["winget"],
            _ => &[],
        };
        let manager = managers.iter().find(|m| on_path(&exe_names(m)).is_some()).map(|m| m.trim_end_matches("-get").to_string()).unwrap_or_default();
        Host {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            tools,
            workshop: beside("am2strata-workshop"),
            browser: browser(data_dir),
            ovmf,
            kvm: cfg!(target_os = "linux") && std::fs::OpenOptions::new().read(true).write(true).open("/dev/kvm").is_ok(),
            manager,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(os: &str, arch: &str, tools: &[&'static str], manager: &str) -> Host {
        Host { os: os.into(), arch: arch.into(), tools: tools.to_vec(), workshop: true, browser: true, ovmf: false, kvm: false, manager: manager.into() }
    }

    fn state(fs: &[Feature], name: &str) -> State {
        fs.iter().find(|f| f.name == name).unwrap().state
    }

    #[test]
    fn a_bare_linux_pc_lists_each_package_once_in_its_managers_names() {
        let h = host("linux", "x86_64", &["ssh", "ssh-keygen"], "apt");
        let fs = report(&h);
        assert_eq!(state(&fs, "HTTPS"), State::Ready);
        assert_eq!(state(&fs, "Workshop"), State::Ready);
        assert_eq!(state(&fs, "Sandbox VM"), State::Missing);
        assert_eq!(packages(&fs, "apt"), ["poppler-utils", "ffmpeg", "bubblewrap", "xvfb", "xdotool", "xclip", "qemu-system-x86", "qemu-utils", "ovmf"]);
        assert_eq!(packages(&fs, "dnf"), ["poppler-utils", "ffmpeg-free", "bubblewrap", "xorg-x11-server-Xvfb", "xdotool", "xclip", "qemu-system-x86", "qemu-img", "edk2-ovmf"]);
        assert_eq!(packages(&fs, "pacman"), ["poppler", "ffmpeg", "bubblewrap", "xorg-server-xvfb", "xdotool", "xclip", "qemu-system-x86", "qemu-img", "edk2-ovmf"]);
        let text = render(&h, &fs, "0.0.0");
        assert!(text.contains("sudo apt-get install -y poppler-utils ffmpeg bubblewrap xvfb"), "{text}");
        assert!(text.contains("missing   Sandbox VM") && text.contains("sudo usermod -aG kvm $USER"), "{text}");
    }

    #[test]
    fn a_complete_linux_pc_needs_nothing_and_kvm_alone_keeps_the_vm_missing() {
        let all = ["pdftotext", "pdftoppm", "ffmpeg", "Xvfb", "xdotool", "xclip", "bwrap", "qemu-system-x86_64", "qemu-img", "ssh", "ssh-keygen"];
        let mut h = host("linux", "x86_64", &all, "dnf");
        h.ovmf = true;
        let fs = report(&h);
        assert_eq!(state(&fs, "Sandbox VM"), State::Missing, "no /dev/kvm access yet");
        assert!(packages(&fs, "dnf").is_empty() && install_command("dnf", &packages(&fs, "dnf")).is_none());
        h.kvm = true;
        let fs = report(&h);
        assert!(fs.iter().all(|f| f.state == State::Ready));
        assert!(render(&h, &fs, "0.0.0").contains("Nothing is missing"));
    }

    #[test]
    fn linux_only_features_say_so_elsewhere() {
        for (os, arch, manager) in [("macos", "aarch64", "brew"), ("macos", "x86_64", ""), ("windows", "aarch64", "winget"), ("windows", "x86_64", "")] {
            let h = host(os, arch, &[], manager);
            let fs = report(&h);
            for name in ["Python in the canvas", "Computer tab (sandbox desktop)", "Sandbox VM"] {
                assert_eq!(state(&fs, name), State::NotHere, "{name} on {os}");
            }
            assert_eq!(state(&fs, "PDFs in chats"), State::Missing);
            // only what this system can install: no Linux package names
            let want: &[&str] = match manager { "brew" => &["poppler", "ffmpeg"], "winget" => &["oschwartz10612.Poppler", "Gyan.FFmpeg"], _ => &[] };
            assert_eq!(packages(&fs, manager), want);
            assert!(render(&h, &fs, "0.0.0").contains("Linux only"));
        }
        // the VM is QEMU with KVM for x86_64 guests: not on arm64 Linux, where the sandbox desktop still works
        let h = host("linux", "aarch64", &["Xvfb", "xdotool", "xclip"], "apt");
        let fs = report(&h);
        assert_eq!(state(&fs, "Sandbox VM"), State::NotHere);
        assert_eq!(state(&fs, "Computer tab (sandbox desktop)"), State::Ready);
    }

    #[test]
    fn a_missing_browser_names_no_package() {
        let mut h = host("linux", "x86_64", &[], "apt");
        h.browser = false;
        let fs = report(&h);
        assert_eq!(state(&fs, "Browser tool"), State::Missing);
        assert!(!packages(&fs, "apt").iter().any(|p| p.contains("chrom")));
        assert!(render(&h, &fs, "0.0.0").contains("needs Chrome, Chromium or Edge; install one ("));
    }

    #[test]
    fn install_commands() {
        assert_eq!(install_command("brew", &["poppler", "ffmpeg"]).unwrap(), "brew install poppler ffmpeg");
        assert_eq!(install_command("pacman", &["poppler"]).unwrap(), "sudo pacman -S --needed --noconfirm poppler");
        assert_eq!(install_command("winget", &["Gyan.FFmpeg"]).unwrap(), "winget install -e --id Gyan.FFmpeg");
        assert!(install_command("apt", &[]).is_none() && install_command("", &["x"]).is_none());
    }
}
