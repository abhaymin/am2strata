//! This machine's RAM, in MiB, on Linux (/proc/meminfo), macOS (hw.memsize, host_statistics64) and Windows
//! (GlobalMemoryStatusEx). The one place RAM is read: am2Strata's server (mem.rs) and the Rust engine use it.

/// Total RAM this PC has (MiB; 0 when it cannot tell): /proc/meminfo here, hw.memsize on macOS, GlobalMemoryStatusEx on Windows.
pub fn total_mib() -> u64 {
    #[cfg(target_os = "linux")]
    { linux_mem("MemTotal:") }
    #[cfg(target_os = "macos")]
    { macos_total_mib() }
    #[cfg(windows)]
    { windows_total_mib() }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    { 0 }
}

/// RAM the OS says is free right now (MiB; 0 when it cannot tell): the same source as total_mib().
pub fn available_mib() -> u64 {
    #[cfg(target_os = "linux")]
    { linux_mem("MemAvailable:") }
    #[cfg(target_os = "macos")]
    { macos_available_mib() }
    #[cfg(windows)]
    { windows_available_mib() }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    { 0 }
}

/// /proc/meminfo's number for `key`, in kB (the unit the line reports). Pure: a test can feed any string.
pub fn meminfo_kb(text: &str, key: &str) -> Option<u64> {
    text.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse::<u64>().ok()
}

/// /proc/meminfo reports kB (KiB): MiB is that / 1024.
pub fn kb_to_mib(kb: u64) -> u64 { kb / 1024 }

/// Total or available RAM in MiB on Linux (as today: /proc/meminfo). 0 when it cannot read the file.
#[cfg(target_os = "linux")]
fn linux_mem(key: &str) -> u64 {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|t| meminfo_kb(&t, key).map(kb_to_mib))
        .unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn macos_total_mib() -> u64 {
    let mut v: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    if unsafe { libc::sysctlbyname(c"hw.memsize".as_ptr(), &mut v as *mut u64 as *mut libc::c_void, &mut len, std::ptr::null_mut(), 0) == 0 } {
        v / (1024 * 1024) // hw.memsize is in bytes
    } else {
        0
    }
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn macos_available_mib() -> u64 {
    unsafe {
        let host = libc::mach_host_self();
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        if libc::host_statistics64(host, libc::HOST_VM_INFO64, &mut stats as *mut _ as libc::host_info64_t, &mut count) != 0 {
            return 0;
        }
        let pages = stats.free_count as u64 + stats.inactive_count as u64 + stats.purgeable_count as u64 + stats.speculative_count as u64;
        pages * (libc::sysconf(libc::_SC_PAGESIZE) as u64) / (1024 * 1024)
    }
}

#[cfg(windows)]
fn windows_total_mib() -> u64 { global_memory().map(|s| s.0 / (1024 * 1024)).unwrap_or(0) }

#[cfg(windows)]
fn windows_available_mib() -> u64 { global_memory().map(|s| s.1 / (1024 * 1024)).unwrap_or(0) }

#[cfg(windows)]
fn global_memory() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    unsafe {
        let mut info: MEMORYSTATUSEX = std::mem::zeroed();
        info.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut info) == 0 { return None; }
        Some((info.ullTotalPhys, info.ullAvailPhys))
    }
}

