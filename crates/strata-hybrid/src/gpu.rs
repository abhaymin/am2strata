//! Weights on GPUs (docs/GPU-PLAN.md phases 3 and 4): `--devices cuda:0=12,hip:0=6` puts matrices on those devices
//! within those budgets (GiB), in their GGUF encoding, and the products of those matrices are computed there by
//! strata-gpu's kernels (CubeCL: CUDA, ROCm/HIP, Vulkan). The rest stays on the CPU, memory-mapped as before.
//!
//! The order is the strata placement's: first what every token uses (the output projection, then each layer's
//! attention, SSM, dense, shared-expert and latent matrices), then the routed experts as the expert profile ranks
//! them (the hottest first; experts it does not rank follow, layer by layer). Devices fill in the order given, so the
//! fastest one should come first; an integrated GPU (it shares the system RAM) comes after a discrete one.

use crate::model::{Model, QMat};
use crate::pack::Profile;
use std::sync::{Arc, OnceLock};

static DEVICES: OnceLock<Vec<strata_gpu::Gpu>> = OnceLock::new();

/// A matrix's copy on a device.
#[derive(Clone)]
pub struct OnDevice {
    dev: usize,
    w: Arc<strata_gpu::Weights>,
}

impl OnDevice {
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        DEVICES.get().expect("devices opened")[self.dev].matvec(&self.w, x)
    }

    /// Several matrices on the same device with the same input, in one submission.
    pub fn matvec_many(group: &[&OnDevice], x: &[f32]) -> Vec<Vec<f32>> {
        let ws: Vec<&strata_gpu::Weights> = group.iter().map(|g| g.w.as_ref()).collect();
        DEVICES.get().expect("devices opened")[group[0].dev].matvec_many(&ws, x)
    }

    pub fn device(&self) -> usize {
        self.dev
    }

    pub(crate) fn weights(&self) -> &strata_gpu::Weights {
        &self.w
    }
}

/// An open device, for code that chains kernels on it (graph.rs).
pub(crate) fn device(d: usize) -> &'static strata_gpu::Gpu {
    &DEVICES.get().expect("devices opened")[d]
}

/// `cuda:0=12,hip:0=6` -> [("cuda:0", 12 GiB), ("hip:0", 6 GiB)]; a device without `=` gets 1 TiB (all it takes).
pub fn parse(spec: &str) -> Result<Vec<(String, u64)>, String> {
    spec.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| {
        let (name, gib) = s.split_once('=').unwrap_or((s, "1024"));
        let g: f64 = gib.parse().map_err(|_| format!("--devices: {s:?}: the budget is GiB, e.g. cuda:0=12"))?;
        Ok((name.to_string(), (g * (1u64 << 30) as f64) as u64))
    }).collect()
}

/// Upload `q` to device `d` unless it does not fit `left` (or no kernel takes its type); true when placed.
fn put(gpus: &[strata_gpu::Gpu], d: usize, q: &mut QMat, left: &mut u64) -> Result<bool, String> {
    let size = q.data.len() as u64;
    if size > *left || strata_gpu::kernels::geometry(q.ty()).is_none() {
        return Ok(false);
    }
    let w = gpus[d].upload(q.ty(), q.data, q.rows, q.cols)?;
    q.gpu = Some(OnDevice { dev: d, w: Arc::new(w) });
    *left -= size;
    Ok(true)
}

/// Place the model's matrices on the devices; -> a summary for the engine's INFO line.
pub fn place(m: &mut Model, devices: &[(String, u64)], profile: Option<&Profile>) -> Result<String, String> {
    let gpus: Vec<strata_gpu::Gpu> = devices.iter().map(|(n, _)| strata_gpu::Gpu::open(n)).collect::<Result<_, _>>()?;
    let ne = m.profile_experts;
    let n_layers = m.layers.len();
    // the experts' order: the profile's ranking, then the rest layer by layer
    let mut order: Vec<(usize, usize)> = profile.map(|p| p.ranked.iter().map(|&(l, e)| (l as usize, e as usize)).collect()).unwrap_or_default();
    let mut seen = vec![false; n_layers * ne.max(1)];
    order.iter().for_each(|&(l, e)| if l * ne + e < seen.len() { seen[l * ne + e] = true });
    let n_exp: Vec<(usize, usize)> = m.moes().map(|x| (x.layer, x.n_expert)).collect();
    for (l, n) in n_exp {
        for e in 0..n {
            if !seen[l * ne + e] {
                order.push((l, e));
            }
        }
    }
    let (every, mut experts) = m.matrices_mut();
    let mut left: Vec<u64> = devices.iter().map(|(_, b)| *b).collect();
    let mut placed = vec![(0usize, 0u64); gpus.len()];
    let (mut cpu_n, mut cpu_bytes) = (0usize, 0u64);
    let mut d = 0usize;
    let mut take = |q: &mut QMat, d: &mut usize| -> Result<(), String> {
        while *d < gpus.len() {
            let before = left[*d];
            if put(&gpus, *d, q, &mut left[*d])? {
                placed[*d].0 += 1;
                placed[*d].1 += before - left[*d];
                return Ok(());
            }
            // a type with no kernel stays on the CPU without moving on to the next device
            if strata_gpu::kernels::geometry(q.ty()).is_none() {
                break;
            }
            *d += 1;
        }
        cpu_n += 1;
        cpu_bytes += q.data.len() as u64;
        Ok(())
    };
    for q in every {
        take(q, &mut d)?;
    }
    let mut n_experts_on = 0usize;
    for key in order {
        if let Some(qs) = experts.remove(&key) {
            let mut all = true;
            for q in qs {
                let was = d;
                take(q, &mut d)?;
                all &= d < gpus.len() || was < gpus.len();
            }
            if all && d < gpus.len() {
                n_experts_on += 1;
            }
        }
    }
    let _ = DEVICES.set(gpus);
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let per: Vec<String> = devices.iter().zip(&placed).map(|((n, _), (k, b))| format!("{n}:{k}:{:.2}GiB", gib(*b))).collect();
    Ok(format!("devices={} experts_on_devices={n_experts_on} cpu_matrices={cpu_n} cpu_gib={:.2}", per.join(","), gib(cpu_bytes)))
}
