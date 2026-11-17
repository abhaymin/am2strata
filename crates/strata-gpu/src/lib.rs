//! strata-gpu: am2Strata's GPU layer, in Rust (CubeCL) - see docs/GPU-PLAN.md.
//!
//! - [`list`] the GPUs every runtime here can use: CUDA (NVIDIA), HIP (AMD ROCm) and wgpu (Vulkan: NVIDIA, AMD and
//!   Intel, integrated ones included), and the CPU runtime;
//! - [`Gpu::open`] one of them by name ("cuda:0", "hip:0", "vulkan:discrete", "vulkan:integrated", "cpu");
//! - weights uploaded in their GGUF encoding ([`Gpu::upload`]) and multiplied there ([`Gpu::matvec`]).
//!
//! One source of kernels (kernels.rs) for every backend; the C/C++ under it is only the vendors' runtimes.

pub mod kernels;
#[cfg(target_os = "macos")]
pub mod native;

use cubecl::prelude::*;
use cubecl::server::Handle;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The runtimes, as `Gpu` holds them.
enum Client {
    #[cfg(feature = "cuda")]
    Cuda(ComputeClient<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "hip")]
    Hip(ComputeClient<cubecl::hip::HipRuntime>),
    #[cfg(feature = "wgpu")]
    Wgpu(ComputeClient<cubecl::wgpu::WgpuRuntime>),
    Cpu(ComputeClient<cubecl::cpu::CpuRuntime>),
    /// the native Metal backend (native.rs), for `metal:*` when AM2_METAL_NATIVE=1
    #[cfg(target_os = "macos")]
    Metal(native::Native),
}

/// Where a buffer or a matrix lives: a CubeCL handle, or a native Metal buffer.
#[derive(Clone)]
enum Mem {
    Cube(Handle),
    #[cfg(target_os = "macos")]
    Mtl(std::sync::Arc<native::MBuf>),
}

impl Mem {
    fn cube(&self) -> Handle {
        match self {
            Mem::Cube(h) => h.clone(),
            #[cfg(target_os = "macos")]
            Mem::Mtl(_) => panic!("a native Metal buffer given to a CubeCL kernel"),
        }
    }

    #[cfg(target_os = "macos")]
    fn mtl(&self) -> &native::MBuf {
        match self {
            Mem::Mtl(b) => b,
            Mem::Cube(_) => panic!("a CubeCL buffer given to a native Metal kernel"),
        }
    }
}

/// A device: its name as `Gpu::open` takes it, and what it is.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: &'static str,
    pub detail: String,
}

/// One weight matrix on the device, in its GGUF encoding.
pub struct Weights {
    handle: Mem,
    pub rows: u32,
    pub cols: u32,
    row_bytes: u32,
    geometry: (u32, u32, u32),
    pub bytes: usize,
    /// whether matvec_q4k's shared memory (x, its group sums and the cube's partial sums) fits this device: Metal
    /// allows 32 KiB, so a 9216-column Q4_K matrix (a 3B model's ffn_down) must take the plain kernel there
    wide_fits: bool,
}

/// Values on the device, so that a chain of kernels needs no trip through the CPU.
#[derive(Clone)]
pub struct Buf {
    handle: Mem,
    pub len: usize,
}

impl Buf {
    fn arg<R: Runtime>(&self) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.handle.cube(), self.len) }
    }
}

pub struct Gpu {
    client: Client,
    pub name: String,
}

macro_rules! with_client {
    ($self:expr, $c:ident, $R:ident, $body:expr) => {
        match &$self.client {
            #[cfg(feature = "cuda")]
            Client::Cuda($c) => { #[allow(dead_code)] type $R = cubecl::cuda::CudaRuntime; $body }
            #[cfg(feature = "hip")]
            Client::Hip($c) => { #[allow(dead_code)] type $R = cubecl::hip::HipRuntime; $body }
            #[cfg(feature = "wgpu")]
            Client::Wgpu($c) => { #[allow(dead_code)] type $R = cubecl::wgpu::WgpuRuntime; $body }
            Client::Cpu($c) => { #[allow(dead_code)] type $R = cubecl::cpu::CpuRuntime; $body }
            #[cfg(target_os = "macos")]
            Client::Metal(_) => unreachable!("native Metal is handled before with_client!"),
        }
    };
}

/// Runs `$body` with `$n` the native Metal backend when that is this device's client, and returns its value.
macro_rules! native {
    ($self:expr, $n:ident, $body:expr) => {
        #[cfg(target_os = "macos")]
        if let Client::Metal($n) = &$self.client {
            return $body;
        }
    };
}

/// From this many rows a Q4_K product takes `matvec_q4k` (a variable so a test can force either kernel).
pub static WIDE_ROWS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(kernels::WIDE_ROWS);

/// Queues y = W x for `tokens` vectors with the kernel that suits the matrix.
macro_rules! product {
    ($c:expr, $R:ident, $w:expr, $hx:expr, $n:expr, $hy:expr, $tokens:expr) => {{
        let w = $w;
        let (be, bb, k) = w.geometry;
        let wide = k == kernels::kind::Q4_K && w.wide_fits && w.rows >= WIDE_ROWS.load(std::sync::atomic::Ordering::Relaxed);
        // cubes over x and y of the grid (each axis has a limit of 65535 on some backends)
        let cubes = w.rows.div_ceil(if wide { 32 } else { kernels::ROWS_PER_CUBE });
        let (cx, cy) = (cubes.min(65535), cubes.div_ceil(cubes.min(65535)));
        let count = CubeCount::Static(cx, cy, $tokens);
        unsafe {
            let y = ArrayArg::from_raw_parts($hy, $tokens as usize * w.rows as usize);
            let (wa, xa) = (ArrayArg::from_raw_parts(w.handle.cube(), w.bytes.div_ceil(4)), ArrayArg::from_raw_parts($hx, $n));
            if wide {
                kernels::matvec_q4k::launch_unchecked::<$R>($c, count, CubeDim::new_1d(kernels::WIDE_CUBE), wa, xa, y, w.rows, w.row_bytes, w.cols);
            } else {
                kernels::matvec::launch_unchecked::<$R>($c, count, CubeDim::new_1d(kernels::CUBE), wa, xa, y, w.rows, w.cols, w.row_bytes, be, bb, k);
            }
        }
    }};
}

/// A runtime that works here: a tiny kernel runs and gives the right answer (a missing driver or runtime library
/// panics inside CubeCL; that is caught and the device left out).
fn works<R: Runtime>(dev: R::Device) -> Option<ComputeClient<R>> {
    let prev = std::panic::take_hook();
    let loud = std::env::var_os("STRATA_GPU_DEBUG").is_some();
    std::panic::set_hook(Box::new(move |i| if loud { eprintln!("strata-gpu: {i}") }));
    let r = catch_unwind(AssertUnwindSafe(|| {
        let client = R::client(&dev);
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let w: Vec<u32> = (0..32).map(|_| 1.0f32.to_bits()).collect();
        let (hw, hx, hy) = (client.create_from_slice(u32::as_bytes(&w)), client.create_from_slice(f32::as_bytes(&x)), client.empty(4));
        unsafe {
            kernels::matvec::launch_unchecked::<R>(&client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(kernels::CUBE),
                ArrayArg::from_raw_parts(hw, 32), ArrayArg::from_raw_parts(hx, 32), ArrayArg::from_raw_parts(hy.clone(), 1),
                1u32, 32u32, 128u32, 32, 128, kernels::kind::F32);
        }
        let y = f32::from_bytes(&client.read_one_unchecked(hy))[0];
        if loud { eprintln!("strata-gpu: self-check gave {y}") }
        (y == 496.0).then_some(client)
    }));
    std::panic::set_hook(prev);
    r.ok().flatten()
}

/// What the wgpu runtime's name is on this platform (findings item 2): Metal on macOS/iOS, where wgpu uses it, else
/// Vulkan. Device names are "{backend}:{subtype}", so `--list-devices` suggests exactly what "--devices" should take.
#[cfg(feature = "wgpu")]
fn wgpu_backend() -> &'static str {
    match std::env::consts::OS { "macos" | "ios" => "metal", _ => "vulkan" }
}

/// Every device a runtime here can use.
pub fn list() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    for i in 0..8 {
        if works::<cubecl::cuda::CudaRuntime>(cubecl::cuda::CudaDevice { index: i }).is_none() {
            break;
        }
        out.push(DeviceInfo { name: format!("cuda:{i}"), backend: "cuda", detail: "NVIDIA (CUDA)".into() });
    }
    #[cfg(feature = "hip")]
    for i in 0..8 {
        if works::<cubecl::hip::HipRuntime>(cubecl::hip::AmdDevice { index: i }).is_none() {
            break;
        }
        out.push(DeviceInfo { name: format!("hip:{i}"), backend: "hip", detail: "AMD (ROCm/HIP)".into() });
    }
    #[cfg(feature = "wgpu")]
    {
        use cubecl::wgpu::WgpuDevice;
        let backend = wgpu_backend();
        let kind = |s: &str| format!("{backend}:{s}");
        for (n, d, what) in [(kind("discrete"), WgpuDevice::DiscreteGpu(0), "a discrete GPU"),
                             (kind("integrated"), WgpuDevice::IntegratedGpu(0), "an integrated GPU")] {
            if works::<cubecl::wgpu::WgpuRuntime>(d).is_some() {
                out.push(DeviceInfo { name: n, backend, detail: what.into() });
            }
        }
    }
    out.push(DeviceInfo { name: "cpu".into(), backend: "cpu", detail: "the CPU (CubeCL's CPU runtime)".into() });
    out
}

impl Gpu {
    /// Open a device by its name from [`list`].
    pub fn open(name: &str) -> Result<Gpu, String> {
        let (kind, idx) = name.split_once(':').unwrap_or((name, "0"));
        let i: usize = idx.parse().unwrap_or(0);
        let client = match kind {
            #[cfg(feature = "cuda")]
            "cuda" => works::<cubecl::cuda::CudaRuntime>(cubecl::cuda::CudaDevice { index: i }).map(Client::Cuda),
            #[cfg(feature = "hip")]
            "hip" | "rocm" => works::<cubecl::hip::HipRuntime>(cubecl::hip::AmdDevice { index: i }).map(Client::Hip),
            #[cfg(feature = "wgpu")]
            // "metal" is what macOS names it (see wgpu_backend); "vulkan"/"wgpu" stay accepted for older configs.
            #[cfg(target_os = "macos")]
            "metal" if native::wanted() => {
                return native::Native::open().map(|n| Gpu { client: Client::Metal(n), name: name.to_string() });
            }
            "metal" | "vulkan" | "wgpu" => {
                use cubecl::wgpu::WgpuDevice;
                let d = match idx {
                    "integrated" | "igpu" => WgpuDevice::IntegratedGpu(0),
                    "discrete" | "dgpu" => WgpuDevice::DiscreteGpu(0),
                    _ => WgpuDevice::DiscreteGpu(i),
                };
                works::<cubecl::wgpu::WgpuRuntime>(d).map(Client::Wgpu)
            }
            "cpu" => works::<cubecl::cpu::CpuRuntime>(cubecl::cpu::CpuDevice).map(Client::Cpu),
            _ => return Err(format!("unknown device {name:?}: cuda:N, hip:N, metal:discrete, metal:integrated (vulkan still accepted) or cpu")),
        };
        client.map(|client| Gpu { client, name: name.to_string() }).ok_or_else(|| format!("{name} is not usable here"))
    }

    /// Upload a `rows` x `cols` matrix stored as GGUF type `ty` (its bytes as they are in the file).
    pub fn upload(&self, ty: &str, data: &[u8], rows: usize, cols: usize) -> Result<Weights, String> {
        let geometry = kernels::geometry(ty).ok_or_else(|| format!("no GPU kernel for {ty} weights yet"))?;
        let (be, bb, _) = geometry;
        if cols % be as usize != 0 {
            return Err(format!("{cols} columns are not whole {ty} blocks"));
        }
        let row_bytes = cols / be as usize * bb as usize;
        if data.len() != rows * row_bytes {
            return Err(format!("{} bytes for a {rows}x{cols} {ty} matrix (want {})", data.len(), rows * row_bytes));
        }
        native!(self, n, Ok(Weights { handle: Mem::Mtl(n.upload(data)), rows: rows as u32, cols: cols as u32, row_bytes: row_bytes as u32,
                                      geometry, bytes: data.len(), wide_fits: false }));
        // whole u32 words: the tail padded with zeros
        let mut words = vec![0u32; data.len().div_ceil(4)];
        for (i, c) in data.chunks(4).enumerate() {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            words[i] = u32::from_le_bytes(b);
        }
        let handle = Mem::Cube(with_client!(self, c, R, c.create_from_slice(u32::as_bytes(&words))));
        let shared = with_client!(self, c, R, c.properties().hardware.max_shared_memory_size);
        let wide_fits = kernels::wide_shared_bytes(cols as u32) <= shared;
        Ok(Weights { handle, rows: rows as u32, cols: cols as u32, row_bytes: row_bytes as u32, geometry, bytes: data.len(), wide_fits })
    }

    /// The products of several matrices with the same input (q, k, v; gate and up): one upload of x, the kernels
    /// launched back to back, then the results read (only the first read waits for the device).
    pub fn matvec_many(&self, ws: &[&Weights], x: &[f32]) -> Vec<Vec<f32>> {
        if ws.is_empty() {
            return Vec::new();
        }
        let cols = ws[0].cols;
        debug_assert!(ws.iter().all(|w| w.cols == cols));
        let tokens = (x.len() / cols as usize).max(1) as u32;
        native!(self, n, {
            let hx = self.buffer(x);
            let outs: Vec<Buf> = ws.iter().map(|w| self.mv_tokens(w, &hx, tokens)).collect();
            let _ = n;
            outs.iter().map(|b| self.read(b)).collect()
        });
        with_client!(self, c, R, {
            let hx = c.create_from_slice(f32::as_bytes(x));
            let outs: Vec<_> = ws.iter().map(|w| {
                let hy = c.empty(tokens as usize * w.rows as usize * 4);
                product!(c, R, *w, hx.clone(), x.len(), hy.clone(), tokens);
                hy
            }).collect();
            outs.into_iter().map(|hy| f32::from_bytes(&c.read_one_unchecked(hy)).to_vec()).collect()
        })
    }

    /// A buffer of f32 on the device.
    pub fn buffer(&self, v: &[f32]) -> Buf {
        native!(self, n, Buf { handle: Mem::Mtl(n.upload(f32::as_bytes(v))), len: v.len() });
        Buf { handle: Mem::Cube(with_client!(self, c, R, c.create_from_slice(f32::as_bytes(v)))), len: v.len() }
    }

    /// An uninitialized buffer of `len` f32 on the device.
    pub fn scratch(&self, len: usize) -> Buf {
        native!(self, n, Buf { handle: Mem::Mtl(n.empty(len * 4)), len });
        Buf { handle: Mem::Cube(with_client!(self, c, R, c.empty(len * 4))), len }
    }

    /// Waits for the device and reads a buffer back. Only this waits: every call below just queues its kernel.
    pub fn read(&self, b: &Buf) -> Vec<f32> {
        native!(self, n, n.read(b.handle.mtl(), b.len));
        with_client!(self, c, R, f32::from_bytes(&c.read_one_unchecked(b.handle.cube())).to_vec())
    }

    /// y = W x, queued; y stays on the device.
    pub fn mv(&self, w: &Weights, x: &Buf) -> Buf {
        self.mv_tokens(w, x, 1)
    }

    /// y = W x for `tokens` vectors of `w.cols` values in x, queued; y (tokens * rows) stays on the device.
    pub fn mv_tokens(&self, w: &Weights, x: &Buf, tokens: u32) -> Buf {
        native!(self, n, Buf { handle: Mem::Mtl(n.mv(w.handle.mtl(), w.rows, w.cols, w.row_bytes, w.geometry.2, x.handle.mtl(), tokens)),
                               len: tokens as usize * w.rows as usize });
        with_client!(self, c, R, {
            let hy = c.empty(tokens as usize * w.rows as usize * 4);
            product!(c, R, w, x.handle.cube(), x.len, hy.clone(), tokens);
            Buf { handle: Mem::Cube(hy), len: tokens as usize * w.rows as usize }
        })
    }

    /// y = x * w / sqrt(mean(x^2) + eps), queued.
    pub fn rms_norm(&self, x: &Buf, w: &Buf, eps: f32) -> Buf {
        native!(self, n, Buf { handle: Mem::Mtl(n.rms_norm(x.handle.mtl(), w.handle.mtl(), w.len as u32, (x.len / w.len) as u32, eps)), len: x.len });
        let y = self.scratch(x.len);
        with_client!(self, c, R, unsafe {
            kernels::rms_norm::launch_unchecked::<R>(c, CubeCount::Static(1, 1, 1), CubeDim::new_1d(kernels::CUBE), x.arg(), w.arg(), y.arg(), x.len as u32, eps);
        });
        y
    }

    /// y += x, queued.
    pub fn add_into(&self, x: &Buf, y: &Buf) {
        native!(self, m, m.add_into(x.handle.mtl(), y.handle.mtl(), x.len as u32));
        let n = x.len as u32;
        with_client!(self, c, R, unsafe {
            kernels::add_into::launch_unchecked::<R>(c, CubeCount::Static(n.div_ceil(kernels::CUBE), 1, 1), CubeDim::new_1d(kernels::CUBE), x.arg(), y.arg(), n);
        });
    }

    /// up *= silu(gate), queued.
    pub fn silu_gate(&self, gate: &Buf, up: &Buf) {
        native!(self, m, m.silu_gate(gate.handle.mtl(), up.handle.mtl(), up.len as u32));
        let n = up.len as u32;
        with_client!(self, c, R, unsafe {
            kernels::silu_gate::launch_unchecked::<R>(c, CubeCount::Static(n.div_ceil(kernels::CUBE), 1, 1), CubeDim::new_1d(kernels::CUBE), gate.arg(), up.arg(), n);
        });
    }

    /// Rotates x, `hd` values per head, by a (cos, sin) table of `n_rot / 2` pairs, queued.
    pub fn rope(&self, tab: &Buf, x: &Buf, hd: u32, n_rot: u32, neox: bool) {
        let heads = x.len as u32 / hd;
        native!(self, m, m.rope(tab.handle.mtl(), x.handle.mtl(), heads, hd, n_rot, neox));
        with_client!(self, c, R, unsafe {
            kernels::rope::launch_unchecked::<R>(c, CubeCount::Static(heads, 1, 1), CubeDim::new_1d(kernels::CUBE), tab.arg(), x.arg(), hd, n_rot, neox as u32);
        });
    }

    /// cache[at..] = x, queued.
    pub fn kv_append(&self, x: &Buf, cache: &Buf, at: usize) {
        native!(self, m, m.kv_append(x.handle.mtl(), cache.handle.mtl(), x.len as u32, at as u32));
        let n = x.len as u32;
        with_client!(self, c, R, unsafe {
            kernels::kv_append::launch_unchecked::<R>(c, CubeCount::Static(n.div_ceil(kernels::CUBE), 1, 1), CubeDim::new_1d(kernels::CUBE), x.arg(), cache.arg(), at as u32, n);
        });
    }

    /// Attention of q (`n_head` heads of `hd`) over `n` positions of K and V (`n_kv` heads each), queued; -> n_head * hd
    /// values. `s` is scratch for n_head * n scores.
    #[allow(clippy::too_many_arguments)]
    pub fn attend(&self, q: &Buf, k: &Buf, v: &Buf, s: &Buf, n: u32, n_head: u32, n_kv: u32, hd: u32, scale: f32) -> Buf {
        native!(self, m, Buf { handle: Mem::Mtl(m.attend(q.handle.mtl(), k.handle.mtl(), v.handle.mtl(), s.handle.mtl(), n, n_head, n_kv, hd, scale)),
                               len: (n_head * hd) as usize });
        let o = self.scratch((n_head * hd) as usize);
        let group = n_head / n_kv;
        with_client!(self, c, R, unsafe {
            kernels::attn_scores::launch_unchecked::<R>(c, CubeCount::Static(n, n_head, 1), CubeDim::new_1d(kernels::CUBE), q.arg(), k.arg(), s.arg(), n, n_kv, group, hd, scale);
            kernels::attn_softmax::launch_unchecked::<R>(c, CubeCount::Static(n_head, 1, 1), CubeDim::new_1d(kernels::CUBE), s.arg(), n);
            kernels::attn_blend::launch_unchecked::<R>(c, CubeCount::Static(n_head, 1, 1), CubeDim::new_1d(kernels::CUBE), s.arg(), v.arg(), o.arg(), n, n_kv, group, hd);
        });
        o
    }

    /// Whether this device runs the native Metal backend (which has the chunk kernels below).
    pub fn is_native(&self) -> bool {
        #[cfg(target_os = "macos")]
        if let Client::Metal(_) = &self.client {
            return true;
        }
        false
    }

    /// RoPE over a chunk of `tokens` tokens (native Metal only): `tab` holds each token's (cos, sin) pairs in turn.
    pub fn rope_tokens(&self, tab: &Buf, x: &Buf, hd: u32, n_rot: u32, neox: bool, tokens: u32) {
        native!(self, m, m.rope_tokens(tab.handle.mtl(), x.handle.mtl(), x.len as u32 / hd / tokens, tokens, hd, n_rot, neox));
        panic!("rope_tokens needs the native Metal backend")
    }

    /// x[at .. at + n] as a buffer of its own (native Metal only).
    pub fn row(&self, x: &Buf, at: usize, n: usize) -> Buf {
        native!(self, m, Buf { handle: Mem::Mtl(m.copy_from(x.handle.mtl(), at as u32, n as u32)), len: n });
        panic!("row needs the native Metal backend")
    }

    /// Causal attention for a chunk (native Metal only): `tokens` queries at positions base.., K and V already appended.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_tokens(&self, q: &Buf, k: &Buf, v: &Buf, scales: &Buf, base: u32, tokens: u32, n_head: u32, n_kv: u32, hd: u32) -> Buf {
        native!(self, m, Buf { handle: Mem::Mtl(m.attend_tokens(q.handle.mtl(), k.handle.mtl(), v.handle.mtl(), scales.handle.mtl(), base, tokens,
                                                                n_head, n_kv, hd)), len: (tokens * n_head * hd) as usize });
        panic!("attend_tokens needs the native Metal backend")
    }

    /// y = W x for `x.len() / cols` vectors at once (rows of y in the same order).
    pub fn matvec(&self, w: &Weights, x: &[f32]) -> Vec<f32> {
        let tokens = (x.len() / w.cols as usize).max(1) as u32;
        native!(self, m, { let _ = m; let hx = self.buffer(x); let y = self.mv_tokens(w, &hx, tokens); self.read(&y) });
        with_client!(self, c, R, {
            let hx = c.create_from_slice(f32::as_bytes(x));
            let hy = c.empty(tokens as usize * w.rows as usize * 4);
            product!(c, R, w, hx, x.len(), hy.clone(), tokens);
            f32::from_bytes(&c.read_one_unchecked(hy)).to_vec()
        })
    }
}
