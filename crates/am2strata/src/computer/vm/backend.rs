//! The computer-use backend for the sandbox VM: the same tools as the local sandbox (screenshot, click, type, keys,
//! scroll, open an app), acting on the VM's desktop — the screen and the mouse through VNC, typing, keys and windows
//! through xdotool in the guest.

use std::sync::Arc;

use crate::computer::backend::{Backend, Buffer, Button, Key, Window, WindowId};

use super::{guest, Vm};

pub struct VmBackend {
    pub vm: Arc<Vm>,
}

impl Backend for VmBackend {
    fn screenshot(&self) -> Result<Buffer, String> {
        let s = self.vm.screen()?;
        s.frame(|f| Buffer { width: f.width as u32, height: f.height as u32, rgba: f.rgba() })
    }

    fn list_windows(&self) -> Result<Vec<Window>, String> {
        self.vm.ensure()?;
        Ok(crate::computer::backend::parse_wininfo(&self.vm.sh_ok("xwininfo -root -tree", false)?))
    }

    fn focus(&self, id: WindowId) -> Result<(), String> {
        self.vm.ensure()?;
        self.vm.sh_ok(&format!("xdotool windowactivate --sync 0x{id:x}"), false).map(|_| ())
    }

    fn click(&self, x: i32, y: i32, button: Button) -> Result<(), String> {
        let b = match button { Button::Left => 1, Button::Middle => 2, Button::Right => 3 };
        self.vm.screen()?.click(x, y, b)
    }

    fn type_text(&self, text: &str) -> Result<(), String> {
        self.vm.type_text(text)
    }

    fn press_keys(&self, keys: &[Key]) -> Result<(), String> {
        let seq = crate::computer::backend::keys_to_xdotool(keys);
        if seq.is_empty() {
            return Ok(());
        }
        self.vm.ensure()?;
        self.vm.sh_ok(&format!("xdotool key --clearmodifiers {}", guest::shq(&seq)), false).map(|_| ())
    }

    fn scroll(&self, dx: i32, dy: i32) -> Result<(), String> {
        let s = self.vm.screen()?;
        let (w, h) = s.size();
        let (cx, cy) = (w as i32 / 2, h as i32 / 2);
        // a screen is about 10 wheel notches
        for (b, n) in [(if dy < 0 { 4 } else { 5 }, dy.unsigned_abs() * 10), (if dx < 0 { 6 } else { 7 }, dx.unsigned_abs() * 10)] {
            for _ in 0..n.min(100) {
                s.button(cx, cy, b, true)?;
                s.button(cx, cy, b, false)?;
            }
        }
        Ok(())
    }

    fn open_app(&self, name: &str) -> Result<(), String> {
        self.vm.open_app(name)
    }
}
