//! A minimal VNC (RFB 3.8) client for the VM's desktop: TigerVNC's Xvnc in the guest, reached through an SSH-forwarded
//! Unix socket in a private folder (no password, no port open on the network).
//!
//! The screen is kept in memory as the server sends changed rectangles (Raw encoding, plus the DesktopSize
//! pseudo-encoding when the guest resizes). A reader thread asks for the next incremental update no more often than
//! every `min_gap`, so an unchanged screen costs nothing and a busy one at most ~10 updates a second. Pointer and key
//! events go the other way; keys are X keysyms, which Xvnc maps to its own keyboard.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// The screen as the server last sent it: `bgrx` is 4 bytes a pixel (blue, green, red, unused).
pub struct Frame {
    pub width: u16,
    pub height: u16,
    pub bgrx: Vec<u8>,
}

impl Frame {
    pub fn rgb(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.width as usize * self.height as usize * 3);
        for p in self.bgrx.chunks_exact(4) {
            out.extend_from_slice(&[p[2], p[1], p[0]]);
        }
        out
    }

    pub fn rgba(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bgrx.len());
        for p in self.bgrx.chunks_exact(4) {
            out.extend_from_slice(&[p[2], p[1], p[0], 255]);
        }
        out
    }
}

struct Shared {
    frame: Mutex<Frame>,
    /// bumped on every applied update, so a viewer sends a frame only when it changed
    version: AtomicU64,
    changed: Condvar,
    dead: AtomicBool,
    error: Mutex<Option<String>>,
}

pub struct Rfb {
    writer: Mutex<UnixStream>,
    shared: Arc<Shared>,
    buttons: Mutex<u8>,
}

fn read_exact(s: &mut impl Read, n: usize) -> std::io::Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    s.read_exact(&mut b)?;
    Ok(b)
}

fn be16(b: &[u8]) -> u16 { u16::from_be_bytes([b[0], b[1]]) }
fn be32(b: &[u8]) -> u32 { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) }

/// Raw encoding, the DesktopSize pseudo-encoding.
const ENC_RAW: i32 = 0;
const ENC_DESKTOP_SIZE: i32 = -223;

impl Rfb {
    /// Handshake (RFB 3.8, security None), ask for 32-bit BGRX pixels, and start the reader thread.
    pub fn connect(sock: &Path, min_gap: Duration) -> Result<Arc<Rfb>, String> {
        let mut s = UnixStream::connect(sock).map_err(|e| format!("the VM's screen ({}): {e}", sock.display()))?;
        s.set_read_timeout(Some(Duration::from_secs(10))).ok();
        let e = |what: &str, err: std::io::Error| format!("VNC {what}: {err}");
        let version = read_exact(&mut s, 12).map_err(|x| e("version", x))?;
        if !version.starts_with(b"RFB 003.") {
            return Err(format!("not a VNC server: {:?}", String::from_utf8_lossy(&version)));
        }
        s.write_all(b"RFB 003.008\n").map_err(|x| e("version", x))?;
        let n = read_exact(&mut s, 1).map_err(|x| e("security", x))?[0] as usize;
        if n == 0 {
            let len = be32(&read_exact(&mut s, 4).map_err(|x| e("security", x))?) as usize;
            let why = read_exact(&mut s, len.min(4096)).unwrap_or_default();
            return Err(format!("the VM's screen refused the connection: {}", String::from_utf8_lossy(&why)));
        }
        let types = read_exact(&mut s, n).map_err(|x| e("security", x))?;
        if !types.contains(&1) {
            return Err(format!("the VM's screen wants a password (security types {types:?})"));
        }
        s.write_all(&[1]).map_err(|x| e("security", x))?;
        if be32(&read_exact(&mut s, 4).map_err(|x| e("security", x))?) != 0 {
            return Err("the VM's screen refused the connection".into());
        }
        s.write_all(&[1]).map_err(|x| e("init", x))?; // shared: other viewers stay connected
        let init = read_exact(&mut s, 24).map_err(|x| e("init", x))?;
        let (w, h) = (be16(&init[0..2]), be16(&init[2..4]));
        let name_len = be32(&init[20..24]) as usize;
        read_exact(&mut s, name_len.min(1 << 16)).map_err(|x| e("init", x))?;
        // SetPixelFormat: 32 bpp, depth 24, little-endian, true colour, max 255, red 16, green 8, blue 0 (BGRX bytes)
        let mut pf = vec![0u8, 0, 0, 0, 32, 24, 0, 1];
        pf.extend(255u16.to_be_bytes());
        pf.extend(255u16.to_be_bytes());
        pf.extend(255u16.to_be_bytes());
        pf.extend([16, 8, 0, 0, 0, 0]);
        // SetEncodings
        let mut enc = vec![2u8, 0];
        enc.extend(2u16.to_be_bytes());
        enc.extend(ENC_RAW.to_be_bytes());
        enc.extend(ENC_DESKTOP_SIZE.to_be_bytes());
        s.write_all(&pf).and_then(|_| s.write_all(&enc)).map_err(|x| e("setup", x))?;
        let shared = Arc::new(Shared {
            frame: Mutex::new(Frame { width: w, height: h, bgrx: vec![0; w as usize * h as usize * 4] }),
            version: AtomicU64::new(0),
            changed: Condvar::new(),
            dead: AtomicBool::new(false),
            error: Mutex::new(None),
        });
        let writer = s.try_clone().map_err(|x| x.to_string())?;
        let rfb = Arc::new(Rfb { writer: Mutex::new(writer), shared: shared.clone(), buttons: Mutex::new(0) });
        rfb.request(false, w, h)?;
        s.set_read_timeout(None).ok();
        let me = rfb.clone();
        std::thread::Builder::new().name("vm-rfb".into()).spawn(move || {
            let r = me.read_loop(s, min_gap);
            *me.shared.error.lock().unwrap() = Some(r.err().unwrap_or_else(|| "closed".into()));
            me.shared.dead.store(true, Ordering::SeqCst);
            me.shared.changed.notify_all();
        }).map_err(|x| x.to_string())?;
        Ok(rfb)
    }

    fn request(&self, incremental: bool, w: u16, h: u16) -> Result<(), String> {
        let mut m = vec![3u8, incremental as u8, 0, 0, 0, 0];
        m.extend(w.to_be_bytes());
        m.extend(h.to_be_bytes());
        self.writer.lock().unwrap().write_all(&m).map_err(|e| format!("VNC: {e}"))
    }

    fn read_loop(&self, mut s: UnixStream, min_gap: Duration) -> Result<(), String> {
        let io = |e: std::io::Error| format!("VNC: {e}");
        loop {
            let kind = read_exact(&mut s, 1).map_err(io)?[0];
            match kind {
                0 => {
                    let hdr = read_exact(&mut s, 3).map_err(io)?;
                    let rects = be16(&hdr[1..3]);
                    for _ in 0..rects {
                        let r = read_exact(&mut s, 12).map_err(io)?;
                        let (x, y, w, h) = (be16(&r[0..2]) as usize, be16(&r[2..4]) as usize, be16(&r[4..6]) as usize, be16(&r[6..8]) as usize);
                        let enc = i32::from_be_bytes([r[8], r[9], r[10], r[11]]);
                        match enc {
                            ENC_RAW => {
                                let px = read_exact(&mut s, w * h * 4).map_err(io)?;
                                let mut f = self.shared.frame.lock().unwrap();
                                let fw = f.width as usize;
                                for row in 0..h {
                                    let (yy, dst) = (y + row, (y + row) * fw * 4 + x * 4);
                                    if yy >= f.height as usize || x + w > fw { continue; }
                                    f.bgrx[dst..dst + w * 4].copy_from_slice(&px[row * w * 4..(row + 1) * w * 4]);
                                }
                            }
                            ENC_DESKTOP_SIZE => {
                                let mut f = self.shared.frame.lock().unwrap();
                                *f = Frame { width: w as u16, height: h as u16, bgrx: vec![0; w * h * 4] };
                            }
                            other => return Err(format!("VNC: unexpected encoding {other}")),
                        }
                    }
                    self.shared.version.fetch_add(1, Ordering::SeqCst);
                    self.shared.changed.notify_all();
                    std::thread::sleep(min_gap);
                    let (w, h) = { let f = self.shared.frame.lock().unwrap(); (f.width, f.height) };
                    self.request(true, w, h)?;
                }
                1 => {
                    // colour map entries (not used in true colour): skip
                    let hdr = read_exact(&mut s, 5).map_err(io)?;
                    read_exact(&mut s, be16(&hdr[3..5]) as usize * 6).map_err(io)?;
                }
                2 => {} // bell
                3 => {
                    // server cut text: the clipboard goes through xclip instead (UTF-8), so it is read and dropped
                    let hdr = read_exact(&mut s, 7).map_err(io)?;
                    let len = be32(&hdr[3..7]) as i32;
                    read_exact(&mut s, len.unsigned_abs().min(1 << 24) as usize).map_err(io)?;
                }
                other => return Err(format!("VNC: unknown message {other}")),
            }
        }
    }

    pub fn alive(&self) -> bool { !self.shared.dead.load(Ordering::SeqCst) }

    pub fn error(&self) -> Option<String> { self.shared.error.lock().unwrap().clone() }

    pub fn version(&self) -> u64 { self.shared.version.load(Ordering::SeqCst) }

    /// Wait until the screen changed past `seen` (or `limit` passes); the new version.
    pub fn wait_change(&self, seen: u64, limit: Duration) -> u64 {
        let f = self.shared.frame.lock().unwrap();
        let _ = self.shared.changed.wait_timeout_while(f, limit, |_| self.version() <= seen && self.alive());
        self.version()
    }

    /// The screen now (waiting for the first full frame after connecting).
    pub fn frame<T>(&self, f: impl FnOnce(&Frame) -> T) -> Result<T, String> {
        if self.version() == 0 {
            self.wait_change(0, Duration::from_secs(5));
        }
        if !self.alive() {
            return Err(format!("the VM's screen is closed: {}", self.error().unwrap_or_default()));
        }
        Ok(f(&self.shared.frame.lock().unwrap()))
    }

    pub fn size(&self) -> (u16, u16) {
        let f = self.shared.frame.lock().unwrap();
        (f.width, f.height)
    }

    fn pointer_event(&self, mask: u8, x: i32, y: i32) -> Result<(), String> {
        let (w, h) = self.size();
        let (x, y) = (x.clamp(0, w as i32 - 1) as u16, y.clamp(0, h as i32 - 1) as u16);
        let mut m = vec![5u8, mask];
        m.extend(x.to_be_bytes());
        m.extend(y.to_be_bytes());
        self.writer.lock().unwrap().write_all(&m).map_err(|e| format!("VNC: {e}"))
    }

    pub fn move_to(&self, x: i32, y: i32) -> Result<(), String> {
        let mask = *self.buttons.lock().unwrap();
        self.pointer_event(mask, x, y)
    }

    /// `button` is X's numbering: 1 left, 2 middle, 3 right, 4/5 wheel up/down, 6/7 wheel left/right.
    pub fn button(&self, x: i32, y: i32, button: u8, down: bool) -> Result<(), String> {
        let bit = 1u8 << (button.clamp(1, 7) - 1);
        let mask = {
            let mut b = self.buttons.lock().unwrap();
            if down { *b |= bit } else { *b &= !bit }
            *b
        };
        self.pointer_event(mask, x, y)
    }

    pub fn click(&self, x: i32, y: i32, button: u8) -> Result<(), String> {
        self.move_to(x, y)?;
        self.button(x, y, button, true)?;
        self.button(x, y, button, false)
    }

    pub fn key(&self, keysym: u32, down: bool) -> Result<(), String> {
        let mut m = vec![4u8, down as u8, 0, 0];
        m.extend(keysym.to_be_bytes());
        self.writer.lock().unwrap().write_all(&m).map_err(|e| format!("VNC: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny RFB server in a thread: a 4x2 screen, one update that paints the top-left pixel red, then records the
    /// client's input messages.
    fn serve(sock: std::path::PathBuf) -> std::thread::JoinHandle<Vec<Vec<u8>>> {
        let l = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            c.write_all(b"RFB 003.008\n").unwrap();
            read_exact(&mut c, 12).unwrap();
            c.write_all(&[1, 1]).unwrap();
            assert_eq!(read_exact(&mut c, 1).unwrap(), [1]);
            c.write_all(&0u32.to_be_bytes()).unwrap();
            read_exact(&mut c, 1).unwrap();
            let mut init = vec![0, 4, 0, 2];
            init.extend([0u8; 16]);
            init.extend(4u32.to_be_bytes());
            init.extend(b"test");
            c.write_all(&init).unwrap();
            read_exact(&mut c, 20).unwrap(); // SetPixelFormat
            read_exact(&mut c, 12).unwrap(); // SetEncodings (2)
            assert_eq!(read_exact(&mut c, 10).unwrap()[0..2], [3, 0], "a full first request");
            // one rect: (0,0) 1x1 raw, pure red in BGRX
            let mut u = vec![0u8, 0, 0, 1];
            u.extend([0, 0, 0, 0, 0, 1, 0, 1]);
            u.extend(0i32.to_be_bytes());
            u.extend([0, 0, 255, 0]);
            c.write_all(&u).unwrap();
            // then, in any order: the next incremental update request (type 3) and the input — pointer (type 5, 6 bytes)
            // and key (type 4, 8 bytes) events
            let (mut got, mut incremental) = (vec![], false);
            while got.len() < 5 {
                let t = read_exact(&mut c, 1).unwrap()[0];
                let n = match t { 3 => 10, 5 => 6, 4 => 8, other => panic!("message type {other}") };
                let mut m = vec![t];
                m.extend(read_exact(&mut c, n - 1).unwrap());
                if t == 3 { assert_eq!(m[1], 1, "later requests are incremental"); incremental = true; } else { got.push(m); }
            }
            assert!(incremental || read_exact(&mut c, 10).unwrap()[0..2] == [3, 1], "the client asks for the next update");
            got
        })
    }

    #[test]
    fn reads_updates_and_sends_pointer_and_keys() {
        let dir = std::env::temp_dir().join(format!("am2-rfb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("s");
        let _ = std::fs::remove_file(&sock);
        let server = serve(sock.clone());
        std::thread::sleep(Duration::from_millis(50));
        let rfb = Rfb::connect(&sock, Duration::from_millis(1)).unwrap();
        let rgb = rfb.frame(|f| (f.width, f.height, f.rgb())).unwrap();
        assert_eq!((rgb.0, rgb.1), (4, 2));
        assert_eq!(&rgb.2[0..3], &[255, 0, 0], "the painted pixel is red");
        assert_eq!(&rgb.2[3..6], &[0, 0, 0]);
        rfb.click(3, 1, 1).unwrap();
        rfb.key(0xff0d, true).unwrap();
        rfb.key(0xff0d, false).unwrap();
        let got = server.join().unwrap();
        assert_eq!(got[0], [5, 0, 0, 3, 0, 1], "move");
        assert_eq!(got[1], [5, 1, 0, 3, 0, 1], "left down");
        assert_eq!(got[2], [5, 0, 0, 3, 0, 1], "left up");
        assert_eq!(got[3], [4, 1, 0, 0, 0, 0, 0xff, 0x0d], "Enter down");
        assert_eq!(got[4], [4, 0, 0, 0, 0, 0, 0xff, 0x0d]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
