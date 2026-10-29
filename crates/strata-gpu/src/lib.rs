//! strata-gpu: am2Strata's GPU layer, in Rust (CubeCL) - see docs/GPU-PLAN.md.
//!
//! - [`list`] the GPUs every runtime here can use: CUDA (NVIDIA), HIP (AMD ROCm) and wgpu (Vulkan: NVIDIA, AMD and
//!   Intel, integrated ones included), and the CPU runtime;
//! - [`Gpu::open`] one of them by name ("cuda:0", "hip:0", "vulkan:discrete", "vulkan:integrated", "cpu");
//! - weights uploaded in their GGUF encoding ([`Gpu::upload`]) and multiplied there ([`Gpu::matvec`]).
//!
//! One source of kernels (kernels.rs) for every backend; the C/C++ under it is only the vendors' runtimes.

pub mod kernels;

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
    handle: Handle,
    pub rows: u32,
    pub cols: u32,
    row_bytes: u32,
    geometry: (u32, u32, u32),
    pub bytes: usize,
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
        }
    };
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
        for (n, d, what) in [("vulkan:discrete", WgpuDevice::DiscreteGpu(0), "a discrete GPU (Vulkan)"),
                             ("vulkan:integrated", WgpuDevice::IntegratedGpu(0), "an integrated GPU (Vulkan)")] {
            if works::<cubecl::wgpu::WgpuRuntime>(d).is_some() {
                out.push(DeviceInfo { name: n.into(), backend: "vulkan", detail: what.into() });
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
            "vulkan" | "wgpu" => {
                use cubecl::wgpu::WgpuDevice;
                let d = match idx {
                    "integrated" | "igpu" => WgpuDevice::IntegratedGpu(0),
                    "discrete" | "dgpu" => WgpuDevice::DiscreteGpu(0),
                    _ => WgpuDevice::DiscreteGpu(i),
                };
                works::<cubecl::wgpu::WgpuRuntime>(d).map(Client::Wgpu)
            }
            "cpu" => works::<cubecl::cpu::CpuRuntime>(cubecl::cpu::CpuDevice).map(Client::Cpu),
            _ => return Err(format!("unknown device {name:?}: cuda:N, hip:N, vulkan:discrete, vulkan:integrated or cpu")),
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
        // whole u32 words: the tail padded with zeros
        let mut words = vec![0u32; data.len().div_ceil(4)];
        for (i, c) in data.chunks(4).enumerate() {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            words[i] = u32::from_le_bytes(b);
        }
        let handle = with_client!(self, c, R, c.create_from_slice(u32::as_bytes(&words)));
        Ok(Weights { handle, rows: rows as u32, cols: cols as u32, row_bytes: row_bytes as u32, geometry, bytes: data.len() })
    }

    /// y = W x for `x.len() / cols` vectors at once (rows of y in the same order).
    pub fn matvec(&self, w: &Weights, x: &[f32]) -> Vec<f32> {
        let tokens = (x.len() / w.cols as usize).max(1) as u32;
        let (be, bb, k) = w.geometry;
        // cubes over x and y of the grid (each axis has a limit of 65535 on some backends)
        let cubes = w.rows.div_ceil(kernels::ROWS_PER_CUBE);
        let cx = cubes.min(65535);
        let cy = cubes.div_ceil(cx);
        with_client!(self, c, R, {
            let hx = c.create_from_slice(f32::as_bytes(x));
            let hy = c.empty(tokens as usize * w.rows as usize * 4);
            unsafe {
                kernels::matvec::launch_unchecked::<R>(c, CubeCount::Static(cx, cy, tokens), CubeDim::new_1d(kernels::CUBE),
                    ArrayArg::from_raw_parts(w.handle.clone(), w.bytes.div_ceil(4)), ArrayArg::from_raw_parts(hx, x.len()),
                    ArrayArg::from_raw_parts(hy.clone(), tokens as usize * w.rows as usize),
                    w.rows, w.cols, w.row_bytes, be, bb, k);
            }
            f32::from_bytes(&c.read_one_unchecked(hy)).to_vec()
        })
    }
}
