//! `read_exact_at` on every platform: the standard library has it per operating system, under different names.

use std::fs::File;
use std::io;

pub trait ReadAt {
    fn read_exact_at(&self, buf: &mut [u8], at: u64) -> io::Result<()>;
}

impl ReadAt for File {
    #[cfg(unix)]
    fn read_exact_at(&self, buf: &mut [u8], at: u64) -> io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, buf, at)
    }

    #[cfg(windows)]
    fn read_exact_at(&self, mut buf: &mut [u8], mut at: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match std::os::windows::fs::FileExt::seek_read(self, buf, at) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    buf = &mut buf[n..];
                    at += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}
