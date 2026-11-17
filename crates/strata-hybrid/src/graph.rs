//! A whole token on the device (docs/ENGINE-PLAN.md phase 1): the norms, RoPE, attention over a KV cache that stays on the
//! device, the feed-forwards and the additions run as one queue of kernels, and only the logits are read back. The CPU
//! path crosses to the device and back for every product, which is what limited it.
//!
//! This covers dense attention models (`llama`, `mistral3`) with every matrix on one device. Activations are not rounded to
//! 8 bits here as the CPU path rounds them (ggml's way), so a logit can differ a little more; the parity gate decides.

use crate::model::{Block, Model, QMat, State};
use strata_gpu::{Buf, Gpu, Weights};

struct Attn<'a> {
    norm: Buf,
    q: &'a Weights,
    k: &'a Weights,
    v: &'a Weights,
    o: &'a Weights,
    kc: Buf,
    vc: Buf,
    s: Buf,
    n_head: u32,
    n_kv: u32,
    hd: u32,
}

struct Ffn<'a> {
    norm: Buf,
    gate: &'a Weights,
    up: &'a Weights,
    down: &'a Weights,
}

pub struct Graph<'a> {
    g: &'static Gpu,
    layers: Vec<(Attn<'a>, Ffn<'a>)>,
    out_norm: Buf,
    output: &'a Weights,
    n_rot: u32,
    neox: bool,
}

fn on_device<'a>(q: &'a QMat, dev: &mut Option<usize>) -> Result<&'a Weights, String> {
    let d = q.gpu.as_ref().ok_or("--graph needs every matrix on a device: give --devices room for the whole model")?;
    match *dev {
        Some(x) if x != d.device() => Err("--graph needs every matrix on one device".to_string()),
        _ => {
            *dev = Some(d.device());
            Ok(d.weights())
        }
    }
}

impl<'a> Graph<'a> {
    pub fn new(m: &'a Model, max_ctx: usize) -> Result<Graph<'a>, String> {
        let mut dev = None;
        let rope = m.rope.as_ref().ok_or("--graph needs a model with RoPE")?;
        let output = on_device(&m.output, &mut dev)?;
        let mut raw = Vec::new();
        for l in &m.layers {
            let [(n1, Block::Attn(a)), (n2, Block::Dense(f))] = &l.parts[..] else {
                return Err("--graph runs dense attention layers only so far".into());
            };
            if a.q_b.is_some() || a.k_b.is_some() || a.v_b.is_some() || a.o_b.is_some() || a.q_norm.is_some() || a.k_norm.is_some() || !a.rope {
                return Err("--graph: this model's attention has parts the device path does not have yet".into());
            }
            let gate = f.gate.as_ref().ok_or("--graph needs gated feed-forwards")?;
            raw.push((n1, a, on_device(&a.q, &mut dev)?, on_device(&a.k, &mut dev)?, on_device(&a.v, &mut dev)?, on_device(&a.o, &mut dev)?,
                      n2, on_device(gate, &mut dev)?, on_device(&f.up, &mut dev)?, on_device(&f.down, &mut dev)?));
        }
        let dev = dev.ok_or("--graph: no layers")?;
        // the KV caches and score scratch for max_ctx tokens count against the device's --devices budget. All three are
        // f32 on the device (Gpu::scratch reserves len * 4 bytes; the f16 rounding of K and V is the CPU path's), so 4 bytes.
        let kv: u64 = raw.iter().map(|r| (2 * r.1.n_kv * r.1.head_dim + r.1.n_head) as u64 * max_ctx as u64 * 4).sum();
        if let Some(room) = crate::gpu::ROOM.get().and_then(|r| r.get(dev)) {
            if kv > *room {
                let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                return Err(format!("the KV cache for {max_ctx} tokens needs {:.2} GiB on the device and its --devices budget has {:.2} GiB left \
                                    (a smaller --max-context or a larger budget)", gib(kv), gib(*room)));
            }
        }
        let g = crate::gpu::device(dev);
        let layers = raw.into_iter().map(|(n1, a, q, k, v, o, n2, gate, up, down)| {
            let row = a.n_kv * a.head_dim;
            (Attn { norm: g.buffer(n1), q, k, v, o, kc: g.scratch(row * max_ctx), vc: g.scratch(row * max_ctx), s: g.scratch(a.n_head * max_ctx),
                    n_head: a.n_head as u32, n_kv: a.n_kv as u32, hd: a.head_dim as u32 },
             Ffn { norm: g.buffer(n2), gate, up, down })
        }).collect();
        Ok(Graph { g, layers, out_norm: g.buffer(&m.out_norm), output, n_rot: rope.n_dims as u32, neox: rope.neox })
    }

    /// AM2_GPU_BENCH=1: how long each kind of kernel takes on this device, each queued `n` times before one read (so the
    /// time per call is the device's plus the queueing, without a wait per call). Prints one line per kernel to stderr.
    pub fn bench(&self, m: &Model) {
        let g = self.g;
        let n = 50;
        let time = |what: &str, bytes: usize, f: &dyn Fn() -> Buf| {
            let warm = f();
            g.read(&warm);
            let t = std::time::Instant::now();
            let mut last = f();
            for _ in 1..n {
                last = f();
            }
            g.read(&last);
            let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            let gbs = if bytes > 0 { format!(" {:.1} GB/s", bytes as f64 / us / 1e3) } else { String::new() };
            eprintln!("GPU_BENCH {what:<28} {us:9.1} us/call{gbs}");
        };
        let (a, f) = &self.layers[0];
        let dim = m.embd.row(0).len();
        let x = g.buffer(&vec![0.01f32; dim]);
        for (name, w) in [("attn_q", a.q), ("attn_k", a.k), ("attn_v", a.v), ("attn_o", a.o), ("ffn_gate", f.gate), ("ffn_up", f.up), ("ffn_down", f.down), ("output", self.output)] {
            let xin = g.buffer(&vec![0.01f32; w.cols as usize]);
            time(&format!("mv {name} {}x{}", w.rows, w.cols), w.bytes, &|| g.mv(w, &xin));
        }
        time("rms_norm", 0, &|| g.rms_norm(&x, &a.norm, m.eps));
        time("add_into", 0, &|| { g.add_into(&x, &x); x.clone() });
        let up = g.scratch(f.up.rows as usize);
        time("silu_gate", 0, &|| { g.silu_gate(&up, &up); up.clone() });
        let rope = m.rope.as_ref().expect("rope");
        let tab = g.buffer(&rope.table(5));
        let q = g.scratch(a.q.rows as usize);
        time("rope q", 0, &|| { g.rope(&tab, &q, a.hd, self.n_rot, self.neox); q.clone() });
        let k = g.scratch(a.k.rows as usize);
        time("kv_append", 0, &|| { g.kv_append(&k, &a.kc, 0); k.clone() });
        for ctx in [16u32, 128, 1024] {
            time(&format!("attend n={ctx}"), 0, &|| g.attend(&q, &a.kc, &a.vc, &a.s, ctx, a.n_head, a.n_kv, a.hd, 0.1));
        }
        time("buffer upload (rope table)", 0, &|| g.buffer(&rope.table(5)));
        let mut st = m.new_state();
        let t = std::time::Instant::now();
        for i in 0..20 {
            self.forward(m, &mut st, 1000 + i);
        }
        eprintln!("GPU_BENCH whole token (forward + read)  {:9.1} us/token, {} layers", t.elapsed().as_secs_f64() * 1e6 / 20.0, self.layers.len());
    }

    /// One token: the logits of the next one. The state only counts positions: K and V are on the device.
    pub fn forward(&self, m: &Model, st: &mut State, token: u32) -> Vec<f32> {
        let g = self.g;
        let pos = st.pos;
        let rope = m.rope.as_ref().expect("rope");
        let tab = g.buffer(&rope.table(pos));
        let x = g.buffer(&m.embd.row(token as usize));
        let ts = m.temp_scale_at(pos);
        for (a, f) in &self.layers {
            let h = g.rms_norm(&x, &a.norm, m.eps);
            let (q, k, v) = (g.mv(a.q, &h), g.mv(a.k, &h), g.mv(a.v, &h));
            g.rope(&tab, &q, a.hd, self.n_rot, self.neox);
            g.rope(&tab, &k, a.hd, self.n_rot, self.neox);
            let row = (a.n_kv * a.hd) as usize;
            g.kv_append(&k, &a.kc, pos * row);
            g.kv_append(&v, &a.vc, pos * row);
            let att = g.attend(&q, &a.kc, &a.vc, &a.s, pos as u32 + 1, a.n_head, a.n_kv, a.hd, ts / (a.hd as f32).sqrt());
            g.add_into(&g.mv(a.o, &att), &x);
            let h = g.rms_norm(&x, &f.norm, m.eps);
            let (up, gate) = (g.mv(f.up, &h), g.mv(f.gate, &h));
            g.silu_gate(&gate, &up);
            g.add_into(&g.mv(f.down, &up), &x);
        }
        st.pos += 1;
        g.read(&g.mv(self.output, &g.rms_norm(&x, &self.out_norm, m.eps)))
    }
}
