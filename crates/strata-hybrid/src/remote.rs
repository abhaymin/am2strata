//! Memory across machines (docs/CLUSTER-PLAN.md, "Phase 2, revised"): another machine's RAM as a tier between this
//! machine's RAM and its disk. Whichever machine runs the model is the master; any paired machine can lend memory as the
//! worker: Mac and PC either way round, Mac and Mac, PC and PC.
//!
//! - **The worker** runs `am2strata-hybrid --serve-experts ADDR --model M --identity node-tls.json --allow-pin PIN`: an
//!   expert host. It loads the same GGUF (checked by a fingerprint), keeps the experts a master assigns it in its own
//!   memory, and computes them on request.
//! - **The master** runs `--remote-experts HOST:PORT --remote-pin PIN --identity node-tls.json` with `--resident-experts`:
//!   after its own RAM tier is placed, the next-hottest experts (by the profile) go to the worker, up to the memory the
//!   worker reports, and only experts that would otherwise be read from this machine's disk.
//! - **Exact answers:** the worker returns each expert's output separately and the master adds them in routing order,
//!   with the same row functions on both sides. The integer kernels give the same bits on every CPU (AVX2, NEON, plain;
//!   cpu_q8's tests), so the tokens are identical with and without the remote tier, whichever machines are paired.
//! - **Encrypted and paired only:** mutual TLS 1.3 with the cluster's node certificates and pins (the ones pairing
//!   swapped); the host lets in only the pins it is given. There is no plain-text mode.
//! - **Failure:** a worker that stops answering is marked down and its experts are computed here from the file again: the
//!   same tokens, slower.
//!
//! Wire format (little-endian on both kinds of machine): a frame is `u32 header length, JSON header, u32 data length,
//! data`, the data being f32 values. Requests: `hello`, `assign {experts: [[layer, expert], ...]}`, `compute {layer,
//! jobs: [{e, n}, ...]}` with each job's n input vectors; the answer carries each job's n output vectors in order.

use crate::model::{Model, Moe, QMat};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{DigitallySignedStruct, DistinguishedName, SignatureScheme};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---- identity and pins (the cluster's node-tls.json: {"cert": base64 DER, "key": base64 PKCS#8 DER}) ----

pub struct Identity {
    cert: Vec<u8>,
    key: Vec<u8>,
    pub pin: String,
}

pub fn pin_of(der: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(der).iter().map(|b| format!("{b:02x}")).collect()
}

fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

impl Identity {
    pub fn load(path: &std::path::Path) -> Result<Identity, String> {
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;
        let v: Value = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let cert = v["cert"].as_str().and_then(|c| b64.decode(c).ok()).ok_or("the identity has no cert")?;
        let key = v["key"].as_str().and_then(|c| b64.decode(c).ok()).ok_or("the identity has no key")?;
        let pin = pin_of(&cert);
        Ok(Identity { cert, key, pin })
    }

    /// A new self-signed identity in the cluster's format (for tests and for machines without am2Strata's server).
    pub fn make(path: &std::path::Path, name: &str) -> Result<Identity, String> {
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;
        let ck = rcgen::generate_simple_self_signed(vec!["am2strata-node".to_string(), name.to_string()]).map_err(|e| e.to_string())?;
        let (cert, key) = (ck.cert.der().to_vec(), ck.key_pair.serialize_der());
        std::fs::write(path, json!({"cert": b64.encode(&cert), "key": b64.encode(&key)}).to_string()).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        let pin = pin_of(&cert);
        Ok(Identity { cert, key, pin })
    }

    fn chain(&self) -> Vec<CertificateDer<'static>> {
        vec![CertificateDer::from(self.cert.clone())]
    }

    fn private_key(&self) -> PrivateKeyDer<'static> {
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(self.key.clone()))
    }
}

fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

#[derive(Debug)]
struct PinnedServer {
    pin: String,
    p: Arc<CryptoProvider>,
}

impl ServerCertVerifier for PinnedServer {
    fn verify_server_cert(&self, end: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: &ServerName<'_>, _: &[u8], _: UnixTime)
                          -> Result<ServerCertVerified, rustls::Error> {
        if same(&pin_of(end), &self.pin) {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General("the expert host's certificate is not the paired one".into()))
        }
    }
    fn verify_tls12_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(m, c, d, &self.p.signature_verification_algorithms)
    }
    fn verify_tls13_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(m, c, d, &self.p.signature_verification_algorithms)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.p.signature_verification_algorithms.supported_schemes()
    }
}

#[derive(Debug)]
struct PinnedClients {
    pins: Vec<String>,
    p: Arc<CryptoProvider>,
}

impl ClientCertVerifier for PinnedClients {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }
    fn verify_client_cert(&self, end: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: UnixTime) -> Result<ClientCertVerified, rustls::Error> {
        let pin = pin_of(end);
        if self.pins.iter().any(|p| same(p, &pin)) {
            Ok(ClientCertVerified::assertion())
        } else {
            Err(rustls::Error::General("this master is not paired with the expert host".into()))
        }
    }
    fn verify_tls12_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(m, c, d, &self.p.signature_verification_algorithms)
    }
    fn verify_tls13_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(m, c, d, &self.p.signature_verification_algorithms)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.p.signature_verification_algorithms.supported_schemes()
    }
}

// ---- frames ----

fn write_frame<W: Write>(w: &mut W, header: &Value, data: &[f32]) -> std::io::Result<()> {
    let h = header.to_string().into_bytes();
    let mut buf = Vec::with_capacity(8 + h.len() + data.len() * 4);
    buf.extend_from_slice(&(h.len() as u32).to_le_bytes());
    buf.extend_from_slice(&h);
    buf.extend_from_slice(&((data.len() * 4) as u32).to_le_bytes());
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&buf)?;
    w.flush()
}

fn read_frame<R: Read>(r: &mut R) -> std::io::Result<(Value, Vec<f32>)> {
    let mut n = [0u8; 4];
    r.read_exact(&mut n)?;
    let hl = u32::from_le_bytes(n) as usize;
    if hl > 16 << 20 {
        return Err(std::io::Error::other("header too large"));
    }
    let mut h = vec![0u8; hl];
    r.read_exact(&mut h)?;
    r.read_exact(&mut n)?;
    let dl = u32::from_le_bytes(n) as usize;
    if dl % 4 != 0 || dl > 1 << 30 {
        return Err(std::io::Error::other("bad data length"));
    }
    let mut d = vec![0u8; dl];
    r.read_exact(&mut d)?;
    let header = serde_json::from_slice(&h).map_err(std::io::Error::other)?;
    Ok((header, d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()))
}

/// What both sides check they have loaded: the file's size and a SHA-256 of its first 8 MiB (the GGUF header and the
/// tensor index are in there).
pub fn fingerprint(path: &std::path::Path) -> Result<String, String> {
    use sha2::Digest;
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    let mut buf = vec![0u8; (8u64 << 20).min(len) as usize];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    let d = sha2::Sha256::digest(&buf);
    Ok(format!("{len}-{}", d.iter().take(16).map(|b| format!("{b:02x}")).collect::<String>()))
}

// ---- the expert host (the worker) ----

/// An expert's three matrices as held by the host: its own copies in its own memory.
struct Held {
    up: QMat,
    gate: Option<QMat>,
    down: QMat,
    /// the bytes the three matrices point into; freed when the last Arc<Held> goes (a compute in flight keeps it)
    _mem: Vec<Box<[u8]>>,
    size: u64,
}

impl Held {
    fn copy(m: &Moe, e: usize) -> Held {
        let mut mem = Vec::new();
        let mut take = |q: &QMat| {
            let b: Box<[u8]> = q.data.into();
            // SAFETY: the box lives in this Held next to the matrix that points into it, never moves (a Box's heap
            // data does not move with the Box) and is dropped with it; no QMat leaves the Held.
            let data: &'static [u8] = unsafe { std::slice::from_raw_parts(b.as_ptr(), b.len()) };
            mem.push(b);
            q.with_data(data, q.ty()).expect("same shape")
        };
        let up = take(&m.up_exps.0[e]);
        let gate = m.gate_exps.as_ref().map(|g| take(&g.0[e]));
        let down = take(&m.down_exps.0[e]);
        let size = mem.iter().map(|b| b.len() as u64).sum();
        Held { up, gate, down, _mem: mem, size }
    }
}

/// Serves `model`'s experts to the masters whose pins are allowed, until the process ends.
pub fn serve(model: &'static Model, model_id: String, addr: &str, me: Identity, allow: Vec<String>, budget: u64) -> Result<(), String> {
    let p = provider();
    let cfg = rustls::ServerConfig::builder_with_provider(p.clone()).with_protocol_versions(&[&rustls::version::TLS13]).map_err(|e| e.to_string())?
        .with_client_cert_verifier(Arc::new(PinnedClients { pins: allow, p: p.clone() }))
        .with_single_cert(me.chain(), me.private_key()).map_err(|e| e.to_string())?;
    let cfg = Arc::new(cfg);
    let listener = TcpListener::bind(addr).map_err(|e| format!("--serve-experts {addr}: {e}"))?;
    eprintln!("expert host on {addr}, model {model_id}, memory for experts {:.2} GiB, pin {}…", budget as f64 / (1u64 << 30) as f64, &me.pin[..16]);
    let held: Arc<Mutex<HashMap<(usize, usize), Arc<Held>>>> = Arc::new(Mutex::new(HashMap::new()));
    let used = Arc::new(AtomicU64::new(0));
    let moes: Arc<HashMap<usize, &'static Moe>> = Arc::new(model.moes().map(|m| (m.layer, m)).collect());
    for conn in listener.incoming() {
        let Ok(tcp) = conn else { continue };
        let _ = tcp.set_nodelay(true);
        let (cfg, held, used, moes, model_id) = (cfg.clone(), held.clone(), used.clone(), moes.clone(), model_id.clone());
        std::thread::spawn(move || {
            let peer = tcp.peer_addr().map(|a| a.to_string()).unwrap_or_default();
            let Ok(sc) = rustls::ServerConnection::new(cfg) else { return };
            let mut s = rustls::StreamOwned::new(sc, tcp);
            eprintln!("expert host: master {peer} connected");
            loop {
                let (h, data) = match read_frame(&mut s) {
                    Ok(x) => x,
                    Err(_) => break,
                };
                let (reply, out) = handle(&h, &data, model, &model_id, &moes, &held, &used, budget);
                if write_frame(&mut s, &reply, &out).is_err() {
                    break;
                }
            }
            eprintln!("expert host: master {peer} left");
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle(h: &Value, data: &[f32], model: &Model, model_id: &str, moes: &HashMap<usize, &Moe>,
          held: &Mutex<HashMap<(usize, usize), Arc<Held>>>, used: &AtomicU64, budget: u64) -> (Value, Vec<f32>) {
    let err = |e: String| (json!({"ok": false, "error": e}), Vec::new());
    match h["op"].as_str().unwrap_or("") {
        // the whole budget: experts already held (from this master's last session) cost nothing to assign again
        "hello" => (json!({"ok": true, "model": model_id, "budget_bytes": budget, "used_bytes": used.load(Ordering::Relaxed),
                          "layers": model.layers.len()}), Vec::new()),
        // the master's whole wanted list, hottest first: experts no longer wanted are let go, then the wanted ones are
        // taken in order while they fit (already held ones cost nothing new); the answer says how many of the list are
        // held, a prefix, so a master whose order differs from the last one's still gets the most it can
        "assign" => {
            let mut want = Vec::new();
            for pair in h["experts"].as_array().cloned().unwrap_or_default() {
                let (l, e) = (pair[0].as_u64().unwrap_or(u64::MAX) as usize, pair[1].as_u64().unwrap_or(u64::MAX) as usize);
                let Some(m) = moes.get(&l) else { return err(format!("layer {l} has no experts")) };
                if e >= m.n_expert {
                    return err(format!("layer {l} has no expert {e}"));
                }
                want.push((l, e));
            }
            let mut map = held.lock().unwrap();
            let keep: std::collections::HashSet<(usize, usize)> = want.iter().copied().collect();
            map.retain(|k, _| keep.contains(k));
            let mut bytes: u64 = map.values().map(|x| x.size).sum();
            let mut taken = 0usize;
            for &(l, e) in &want {
                if !map.contains_key(&(l, e)) {
                    let m = moes[&l];
                    let size = (m.up_exps.0[e].data.len() + m.down_exps.0[e].data.len() + m.gate_exps.as_ref().map_or(0, |g| g.0[e].data.len())) as u64;
                    if bytes + size > budget {
                        break;
                    }
                    map.insert((l, e), Arc::new(Held::copy(m, e)));
                    bytes += size;
                }
                taken += 1;
            }
            // anything after the prefix that happened to be held already is let go too: the master will not send it
            let prefix: std::collections::HashSet<(usize, usize)> = want[..taken].iter().copied().collect();
            map.retain(|k, _| prefix.contains(k));
            used.store(map.values().map(|x| x.size).sum(), Ordering::Relaxed);
            (json!({"ok": true, "taken": taken, "bytes": used.load(Ordering::Relaxed)}), Vec::new())
        }
        "compute" => {
            let l = h["layer"].as_u64().unwrap_or(u64::MAX) as usize;
            let Some(m) = moes.get(&l) else { return err(format!("layer {l} has no experts")) };
            let cols = m.up_exps.0[0].cols;
            let mut at = 0usize;
            let mut out = Vec::new();
            for job in h["jobs"].as_array().cloned().unwrap_or_default() {
                let (e, n) = (job["e"].as_u64().unwrap_or(u64::MAX) as usize, job["n"].as_u64().unwrap_or(0) as usize);
                let Some(x) = held.lock().unwrap().get(&(l, e)).cloned() else { return err(format!("expert {l}/{e} is not held here")) };
                if at + n * cols > data.len() {
                    return err("the inputs are shorter than the jobs".into());
                }
                let xs: Vec<Vec<f32>> = (0..n).map(|i| data[at + i * cols..at + (i + 1) * cols].to_vec()).collect();
                at += n * cols;
                for o in m.expert_forward(&x.up, x.gate.as_ref(), &x.down, &xs) {
                    out.extend_from_slice(&o);
                }
            }
            (json!({"ok": true}), out)
        }
        other => err(format!("unknown op {other:?}")),
    }
}

// ---- the master's side ----

type Stream = rustls::StreamOwned<rustls::ClientConnection, TcpStream>;

/// The master's link to one expert host: the experts it holds, and the connection.
pub struct Remote {
    pub addr: String,
    held: Vec<bool>,
    per_layer: usize,
    conn: Mutex<Option<Stream>>,
    alive: AtomicBool,
    pub n_held: usize,
    pub bytes: u64,
    calls: AtomicU64,
    us: AtomicU64,
    failed: AtomicU64,
    /// expert uses (one expert for one token) answered by the host
    uses: AtomicU64,
}

/// A request sent and not yet answered (the master computes its own experts meanwhile).
pub struct Pending {
    t0: Instant,
    shapes: Vec<(usize, usize)>,
}

impl Remote {
    /// Connects, checks the host has the same model, and asks how much memory it lends.
    pub fn connect(addr: &str, pin: &str, me: &Identity, model_id: &str, n_layers: usize, per_layer: usize) -> Result<(Remote, u64), String> {
        let p = provider();
        let cfg = rustls::ClientConfig::builder_with_provider(p.clone()).with_protocol_versions(&[&rustls::version::TLS13]).map_err(|e| e.to_string())?
            .dangerous().with_custom_certificate_verifier(Arc::new(PinnedServer { pin: pin.to_string(), p }))
            .with_client_auth_cert(me.chain(), me.private_key()).map_err(|e| e.to_string())?;
        let tcp = TcpStream::connect(addr).map_err(|e| format!("--remote-experts {addr}: {e}"))?;
        let _ = tcp.set_nodelay(true);
        let _ = tcp.set_read_timeout(Some(Duration::from_secs(20)));
        let cc = rustls::ClientConnection::new(Arc::new(cfg), ServerName::try_from("am2strata-node").map_err(|e| e.to_string())?.to_owned())
            .map_err(|e| e.to_string())?;
        let mut s = rustls::StreamOwned::new(cc, tcp);
        write_frame(&mut s, &json!({"op": "hello"}), &[]).map_err(|e| format!("{addr}: {e}"))?;
        let (h, _) = read_frame(&mut s).map_err(|e| format!("{addr}: {e} (is this master's pin allowed on the host?)"))?;
        if h["model"].as_str() != Some(model_id) {
            return Err(format!("{addr} serves another model ({}), this is {model_id}", h["model"].as_str().unwrap_or("?")));
        }
        let free = h["budget_bytes"].as_u64().unwrap_or(0);
        Ok((Remote { addr: addr.to_string(), held: vec![false; n_layers * per_layer], per_layer, conn: Mutex::new(Some(s)),
                     alive: AtomicBool::new(true), n_held: 0, bytes: 0, calls: AtomicU64::new(0), us: AtomicU64::new(0), failed: AtomicU64::new(0),
                     uses: AtomicU64::new(0) },
            free))
    }

    /// Hands the host this master's wanted experts, hottest first, in one request: it lets go of what is no longer
    /// wanted and takes a prefix of the list as far as its memory goes. Only the experts it took are marked held.
    pub fn assign(&mut self, model: &Model, experts: &[(usize, usize)]) -> Result<(), String> {
        let mut guard = self.conn.lock().unwrap();
        let s = guard.as_mut().ok_or("not connected")?;
        let list: Vec<[usize; 2]> = experts.iter().map(|&(l, e)| [l, e]).collect();
        write_frame(s, &json!({"op": "assign", "experts": list}), &[]).map_err(|e| e.to_string())?;
        let (h, _) = read_frame(s).map_err(|e| e.to_string())?;
        if h["ok"] != true {
            return Err(h["error"].as_str().unwrap_or("assign refused").to_string());
        }
        let taken = (h["taken"].as_u64().unwrap_or(0) as usize).min(experts.len());
        for &(l, e) in &experts[..taken] {
            self.held[l * self.per_layer + e] = true;
            self.n_held += 1;
            let m = model.moes().find(|m| m.layer == l).expect("an MoE layer");
            self.bytes += (m.up_exps.0[e].data.len() + m.down_exps.0[e].data.len() + m.gate_exps.as_ref().map_or(0, |g| g.0[e].data.len())) as u64;
        }
        Ok(())
    }

    /// Whether the host holds this expert and is answering.
    pub fn holds(&self, layer: usize, e: usize) -> bool {
        self.alive.load(Ordering::Relaxed) && self.held.get(layer * self.per_layer + e).copied().unwrap_or(false)
    }

    /// Sends one layer's remote work: for each expert, its input vectors. None when the host is down.
    pub fn start(&self, layer: usize, jobs: &[(usize, Vec<&[f32]>)]) -> Option<Pending> {
        if !self.alive.load(Ordering::Relaxed) || jobs.is_empty() {
            return None;
        }
        let t0 = Instant::now();
        let mut data = Vec::new();
        let mut list = Vec::new();
        let mut shapes = Vec::new();
        for (e, xs) in jobs {
            list.push(json!({"e": e, "n": xs.len()}));
            shapes.push((*e, xs.len()));
            for x in xs {
                data.extend_from_slice(x);
            }
        }
        let mut guard = self.conn.lock().unwrap();
        let s = guard.as_mut()?;
        if write_frame(s, &json!({"op": "compute", "layer": layer, "jobs": list}), &data).is_err() {
            drop(guard);
            self.down("send failed");
            return None;
        }
        Some(Pending { t0, shapes })
    }

    /// The answer to `start`: for each job, its output vectors (each `rows` long). None when the host failed (the caller
    /// then computes those experts itself).
    pub fn finish(&self, p: Pending, rows: usize) -> Option<Vec<Vec<Vec<f32>>>> {
        let mut guard = self.conn.lock().unwrap();
        let s = guard.as_mut()?;
        let (h, data) = match read_frame(s) {
            Ok(x) => x,
            Err(e) => {
                drop(guard);
                self.down(&e.to_string());
                return None;
            }
        };
        let total: usize = p.shapes.iter().map(|(_, n)| n * rows).sum();
        if h["ok"] != true || data.len() != total {
            drop(guard);
            self.down(h["error"].as_str().unwrap_or("a short answer"));
            return None;
        }
        let mut at = 0;
        let out = p.shapes.iter().map(|&(_, n)| (0..n).map(|_| { let v = data[at..at + rows].to_vec(); at += rows; v }).collect()).collect();
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.uses.fetch_add(p.shapes.iter().map(|&(_, n)| n as u64).sum(), Ordering::Relaxed);
        self.us.fetch_add(p.t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        Some(out)
    }

    fn down(&self, why: &str) {
        if self.alive.swap(false, Ordering::Relaxed) {
            eprintln!("remote experts on {}: down ({why}); its experts are computed here from now on", self.addr);
        }
        self.failed.fetch_add(1, Ordering::Relaxed);
        *self.conn.lock().unwrap() = None;
    }

    /// "remote_uses=U remote_calls=N remote_avg_us=M remote_alive=true" since the last call (for the INFO line after a
    /// request): U expert uses (one expert for one token) were served by the host, in N round trips.
    pub fn take_stats(&self) -> String {
        let (c, us, u) = (self.calls.swap(0, Ordering::Relaxed), self.us.swap(0, Ordering::Relaxed), self.uses.swap(0, Ordering::Relaxed));
        format!("remote_uses={u} remote_calls={c} remote_avg_us={} remote_alive={}", if c > 0 { us / c } else { 0 }, self.alive.load(Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame reads back as written: the header and the f32 values bit for bit (both machines are little-endian on
    /// the wire, whatever their CPU), and a frame that claims too much is refused rather than allocated.
    #[test]
    fn frames_round_trip_and_bad_lengths_are_refused() {
        let vals: Vec<f32> = vec![0.0, -0.0, 1.5, f32::MIN_POSITIVE, -3.25e-7, 65504.0, f32::NAN];
        let mut buf = Vec::new();
        write_frame(&mut buf, &json!({"op": "compute", "layer": 3, "jobs": [{"e": 7, "n": 1}]}), &vals).unwrap();
        let (h, back) = read_frame(&mut &buf[..]).unwrap();
        assert_eq!(h["layer"], 3);
        assert_eq!(back.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        let mut huge = (64u32 << 20).to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        assert!(read_frame(&mut &huge[..]).is_err(), "a 64 MiB header is refused");
        let mut odd = Vec::new();
        write_frame(&mut odd, &json!({}), &[]).unwrap();
        let n = odd.len();
        odd[n - 4..].copy_from_slice(&3u32.to_le_bytes());
        odd.extend_from_slice(&[0, 0, 0]);
        assert!(read_frame(&mut &odd[..]).is_err(), "data that is not whole f32 values is refused");
    }

    /// Pins: an identity's pin is the SHA-256 of its certificate, and the comparison is exact.
    #[test]
    fn identities_and_pins() {
        let dir = std::env::temp_dir().join(format!("am2-remote-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = Identity::make(&dir.join("a.json"), "a").unwrap();
        let b = Identity::make(&dir.join("b.json"), "b").unwrap();
        assert_eq!(a.pin.len(), 64);
        assert_ne!(a.pin, b.pin);
        assert_eq!(Identity::load(&dir.join("a.json")).unwrap().pin, a.pin, "the file reads back the same identity");
        assert!(same(&a.pin, &a.pin) && !same(&a.pin, &b.pin) && !same(&a.pin, &a.pin[..63]));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
