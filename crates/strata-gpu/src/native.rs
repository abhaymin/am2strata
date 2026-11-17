//! The native Metal backend (macOS): the same operations as the CubeCL kernels, with MSL kernels (native.metal) compiled
//! at run time, and every kernel of a token encoded into ONE command buffer that is committed only when a result is read.
//! CubeCL's wgpu path pays about 35-40 us per dispatch on an M4 and its products reach 15-40 GB/s; here a dispatch is
//! an encoder call and the products use SIMD-group sums.
//!
//! Picked for `metal:*` devices when AM2_METAL_NATIVE=1 (until it has passed the parity gate, the CubeCL path stays the
//! default; AM2_METAL_NATIVE=0 keeps it explicitly).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

const SOURCE: &str = include_str!("native.metal");

type Dev = ProtocolObject<dyn MTLDevice>;
type MBuffer = ProtocolObject<dyn MTLBuffer>;
type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// A buffer on the device (shared storage: on Apple silicon the CPU and GPU see the same memory).
pub struct MBuf {
    pub(crate) buf: Retained<MBuffer>,
    pub(crate) bytes: usize,
}

// Metal buffers may be used from any thread; the engine only uses them from one at a time.
unsafe impl Send for MBuf {}
unsafe impl Sync for MBuf {}

struct Open {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

pub struct Native {
    dev: Retained<Dev>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pso: HashMap<&'static str, Pso>,
    open: Mutex<Option<Open>>,
    /// device-written temporaries, reused once nothing else holds them
    pool: Mutex<Vec<Arc<MBuf>>>,
}

unsafe impl Send for Native {}
unsafe impl Sync for Native {}

#[repr(C)]
struct MvArgs { rows: u32, cols: u32, row_bytes: u32, kind: u32 }
#[repr(C)]
struct NormArgs { n: u32, eps: f32 }
#[repr(C)]
struct NArgs { n: u32, at: u32 }
#[repr(C)]
struct RopeArgs { hd: u32, n_rot: u32, neox: u32 }
#[repr(C)]
struct AttnArgs { n: u32, n_kv: u32, group: u32, hd: u32, scale: f32 }

/// Rows a threadgroup of `mv` takes (NSG * NR in native.metal) and its threads (NSG * 32).
const MV_ROWS: u32 = 8;
const MV_THREADS: usize = 64;

fn size(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize { width: w, height: h, depth: d }
}

/// Whether to use the native backend for `metal:*`.
pub fn wanted() -> bool {
    std::env::var("AM2_METAL_NATIVE").map(|v| v != "0").unwrap_or(false)
}

impl Native {
    pub fn open() -> Result<Native, String> {
        let dev = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = dev.newCommandQueue().ok_or("no Metal command queue")?;
        let lib = dev.newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|e| format!("native Metal kernels did not compile: {}", e.localizedDescription()))?;
        let mut pso = HashMap::new();
        for name in ["mv", "rms_norm", "add_into", "silu_gate", "kv_append", "rope", "attend"] {
            let f = lib.newFunctionWithName(&NSString::from_str(name)).ok_or_else(|| format!("no kernel {name}"))?;
            let p = dev.newComputePipelineStateWithFunction_error(&f).map_err(|e| format!("{name}: {}", e.localizedDescription()))?;
            pso.insert(name, p);
        }
        Ok(Native { dev, queue, pso, open: Mutex::new(None), pool: Mutex::new(Vec::new()) })
    }

    pub fn name(&self) -> String {
        self.dev.name().to_string()
    }

    fn new_buffer(&self, bytes: usize) -> Retained<MBuffer> {
        self.dev.newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared).expect("Metal buffer")
    }

    /// A buffer holding `data` (a new one: the CPU writes it now, so it must not be one a queued kernel still reads).
    pub fn upload(&self, data: &[u8]) -> Arc<MBuf> {
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap_or(NonNull::dangling());
        let buf = if data.is_empty() {
            self.new_buffer(16)
        } else {
            unsafe { self.dev.newBufferWithBytes_length_options(ptr, data.len(), MTLResourceOptions::StorageModeShared) }
                .expect("Metal buffer")
        };
        Arc::new(MBuf { buf, bytes: data.len() })
    }

    /// A buffer of `bytes` the device will write (persistent: KV caches, scratch).
    pub fn empty(&self, bytes: usize) -> Arc<MBuf> {
        Arc::new(MBuf { buf: self.new_buffer(bytes), bytes })
    }

    /// A temporary of `bytes` for a kernel's output: one from the pool that nothing holds any more, or a new one.
    fn temp(&self, bytes: usize) -> Arc<MBuf> {
        let mut pool = self.pool.lock().unwrap();
        if let Some(b) = pool.iter().find(|b| b.bytes == bytes && Arc::strong_count(b) == 1) {
            return b.clone();
        }
        let b = Arc::new(MBuf { buf: self.new_buffer(bytes), bytes });
        pool.push(b.clone());
        b
    }

    /// Encodes one dispatch into the open command buffer (opening one when none is).
    fn encode(&self, pso: &str, bufs: &[&MBuf], args: &[u8], groups: MTLSize, threads: MTLSize) {
        let mut open = self.open.lock().unwrap();
        if open.is_none() {
            let cmd = self.queue.commandBuffer().expect("Metal command buffer");
            let enc = cmd.computeCommandEncoder().expect("Metal compute encoder");
            *open = Some(Open { cmd, enc });
        }
        let enc = &open.as_ref().unwrap().enc;
        enc.setComputePipelineState(&self.pso[pso]);
        unsafe {
            for (i, b) in bufs.iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(&b.buf), 0, i);
            }
            enc.setBytes_length_atIndex(NonNull::new(args.as_ptr() as *mut c_void).unwrap(), args.len(), bufs.len());
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
    }

    /// Commits what is queued and waits for it.
    pub fn sync(&self) {
        if let Some(o) = self.open.lock().unwrap().take() {
            o.enc.endEncoding();
            o.cmd.commit();
            o.cmd.waitUntilCompleted();
        }
    }

    /// Waits for the device and reads `n` f32 from a buffer.
    pub fn read(&self, b: &MBuf, n: usize) -> Vec<f32> {
        self.sync();
        let p = b.buf.contents().as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(p, n.min(b.bytes / 4)) }.to_vec()
    }

    fn args<T>(a: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts(a as *const T as *const u8, std::mem::size_of::<T>()) }
    }

    /// y = W x for `tokens` vectors, queued.
    pub fn mv(&self, w: &MBuf, rows: u32, cols: u32, row_bytes: u32, kind: u32, x: &MBuf, tokens: u32) -> Arc<MBuf> {
        let y = self.temp(tokens as usize * rows as usize * 4);
        let a = MvArgs { rows, cols, row_bytes, kind };
        self.encode("mv", &[w, x, &y], Self::args(&a), size(rows.div_ceil(MV_ROWS) as usize, tokens as usize, 1), size(MV_THREADS, 1, 1));
        y
    }

    pub fn rms_norm(&self, x: &MBuf, w: &MBuf, n: u32, tokens: u32, eps: f32) -> Arc<MBuf> {
        let y = self.temp(tokens as usize * n as usize * 4);
        let a = NormArgs { n, eps };
        self.encode("rms_norm", &[x, w, &y], Self::args(&a), size(tokens as usize, 1, 1), size(1024, 1, 1));
        y
    }

    fn elementwise(&self, pso: &str, x: &MBuf, y: &MBuf, n: u32, at: u32) {
        let a = NArgs { n, at };
        self.encode(pso, &[x, y], Self::args(&a), size(n.div_ceil(256) as usize, 1, 1), size(256, 1, 1));
    }

    pub fn add_into(&self, x: &MBuf, y: &MBuf, n: u32) {
        self.elementwise("add_into", x, y, n, 0)
    }

    pub fn silu_gate(&self, gate: &MBuf, up: &MBuf, n: u32) {
        self.elementwise("silu_gate", gate, up, n, 0)
    }

    pub fn kv_append(&self, x: &MBuf, cache: &MBuf, n: u32, at: u32) {
        self.elementwise("kv_append", x, cache, n, at)
    }

    pub fn rope(&self, tab: &MBuf, x: &MBuf, heads: u32, hd: u32, n_rot: u32, neox: bool) {
        let a = RopeArgs { hd, n_rot, neox: neox as u32 };
        self.encode("rope", &[tab, x], Self::args(&a), size(heads as usize, 1, 1), size(((n_rot / 2).max(1)).next_multiple_of(32) as usize, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attend(&self, q: &MBuf, k: &MBuf, v: &MBuf, s: &MBuf, n: u32, n_head: u32, n_kv: u32, hd: u32, scale: f32) -> Arc<MBuf> {
        let o = self.temp((n_head * hd) as usize * 4);
        let a = AttnArgs { n, n_kv, group: n_head / n_kv, hd, scale };
        self.encode("attend", &[q, k, v, s, &o], Self::args(&a), size(n_head as usize, 1, 1), size(256, 1, 1));
        o
    }
}
