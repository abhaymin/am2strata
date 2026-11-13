//! Cluster traffic encrypted (CLUSTER-PLAN: "end-to-end encrypted and mutually authenticated ... TLS with the
//! pinned keys"): mutual TLS 1.3 between a master and its paired workers.
//!
//! - **Every node has a TLS certificate**, self-signed, made once and kept in `<data>/node-tls.json` (0600). Its
//!   SHA-256 (of the certificate's DER) is its pin.
//! - **Pairing swaps the pins**, over the same request the 6-digit code proves: the master sends its pin, the worker
//!   answers with its own and its TLS port. Each side keeps the other's pin (cluster-workers.json /
//!   cluster-masters.json, 0600).
//! - **A worker serves a second listener for cluster traffic** (`"cluster": {"role": "worker", "tls_port": N}`,
//!   default the API port + 443): TLS 1.3 only, a client certificate required, and only a certificate whose pin a
//!   paired master gave is let in. Requests on it carry the peer's pin (`x-am2strata-peer`, set here and removed from
//!   anything that came in from outside), and a master's key works only with that master's own certificate.
//! - **The master** talks to such a worker only over that listener, checking the worker's certificate against the pin
//!   it was given (no certificate authority: the pin is the trust) and showing its own certificate.
//! - **Why TLS and not Noise:** rustls is already in the build (reqwest), audited and fast, and HTTP over it needs
//!   nothing new on either side; a Noise channel would need its own framing for HTTP and streaming.

use axum::http::HeaderMap;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{DigitallySignedStruct, DistinguishedName, SignatureScheme};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

/// The header that carries a TLS peer's pin to the handlers (only ever set by this listener).
pub const PEER_HEADER: &str = "x-am2strata-peer";

/// This node's certificate and its private key (PKCS#8 DER).
#[derive(Clone)]
pub struct Identity {
    pub cert: Vec<u8>,
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

/// A pin as given by the other side: 64 hex digits.
pub fn valid_pin(p: &str) -> bool {
    p.len() == 64 && p.bytes().all(|c| c.is_ascii_hexdigit())
}

impl Identity {
    /// The node's certificate from `<data>/node-tls.json`, made on first use (0600).
    pub fn load_or_make(data_dir: &Path, name: &str) -> Result<Identity, String> {
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;
        let path = data_dir.join("node-tls.json");
        if let Some(v) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
            if let (Some(c), Some(k)) = (v["cert"].as_str().and_then(|c| b64.decode(c).ok()), v["key"].as_str().and_then(|k| b64.decode(k).ok())) {
                let pin = pin_of(&c);
                return Ok(Identity { cert: c, key: k, pin });
            }
        }
        let names = vec!["am2strata-node".to_string(), name.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').take(60).collect()];
        let ck = rcgen::generate_simple_self_signed(names.into_iter().filter(|n| !n.is_empty()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
        let (cert, key) = (ck.cert.der().to_vec(), ck.key_pair.serialize_der());
        crate::cluster::write_private(&path, &json!({"cert": b64.encode(&cert), "key": b64.encode(&key)})).map_err(|e| e.to_string())?;
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

/// The signature checks both verifiers share: the handshake is signed by the key of the certificate presented.
fn tls12(p: &CryptoProvider, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
    rustls::crypto::verify_tls12_signature(m, c, d, &p.signature_verification_algorithms)
}

fn tls13(p: &CryptoProvider, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
    rustls::crypto::verify_tls13_signature(m, c, d, &p.signature_verification_algorithms)
}

/// The master's check of a worker: its certificate is the one pinned at pairing.
#[derive(Debug)]
struct PinnedServer {
    pin: String,
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for PinnedServer {
    fn verify_server_cert(&self, end: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: &ServerName<'_>, _: &[u8], _: UnixTime)
                          -> Result<ServerCertVerified, rustls::Error> {
        if same(&pin_of(end), &self.pin) {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General(format!("the worker's certificate is not the one paired (pin {}…, expected {}…)",
                                               &pin_of(end)[..12], &self.pin[..12.min(self.pin.len())])))
        }
    }
    fn verify_tls12_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        tls12(&self.provider, m, c, d)
    }
    fn verify_tls13_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        tls13(&self.provider, m, c, d)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}

/// The worker's check of a master: its certificate's pin is one a paired master gave.
pub struct PinnedClients {
    pins: Box<dyn Fn() -> Vec<String> + Send + Sync>,
    provider: Arc<CryptoProvider>,
}

impl std::fmt::Debug for PinnedClients {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PinnedClients")
    }
}

impl ClientCertVerifier for PinnedClients {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }
    fn verify_client_cert(&self, end: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: UnixTime) -> Result<ClientCertVerified, rustls::Error> {
        let p = pin_of(end);
        if (self.pins)().iter().any(|k| same(k, &p)) {
            Ok(ClientCertVerified::assertion())
        } else {
            Err(rustls::Error::General("this certificate belongs to no master this worker paired with".into()))
        }
    }
    fn verify_tls12_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        tls12(&self.provider, m, c, d)
    }
    fn verify_tls13_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        tls13(&self.provider, m, c, d)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}

/// The master's HTTP client for a worker: TLS 1.3 to the pinned certificate, showing this node's own.
pub fn client(me: &Identity, worker_pin: &str) -> Result<reqwest::Client, String> {
    let provider = provider();
    let mut cfg = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13]).map_err(|e| e.to_string())?
        .dangerous().with_custom_certificate_verifier(Arc::new(PinnedServer { pin: worker_pin.to_string(), provider }))
        .with_client_auth_cert(me.chain(), me.private_key()).map_err(|e| e.to_string())?;
    cfg.alpn_protocols = vec![b"http/1.1".to_vec()];
    reqwest::Client::builder().use_preconfigured_tls(cfg).connect_timeout(std::time::Duration::from_secs(5)).build().map_err(|e| e.to_string())
}

/// The worker's listener for cluster traffic: TLS 1.3, a paired master's certificate required, each request handed to
/// `app` with the peer's pin (PEER_HEADER) and its address.
pub async fn serve(me: Identity, pins: Box<dyn Fn() -> Vec<String> + Send + Sync>, app: axum::Router, host: String, port: u16)
                   -> Result<(), String> {
    let provider = provider();
    let verifier = Arc::new(PinnedClients { pins, provider: provider.clone() });
    let mut cfg = rustls::ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13]).map_err(|e| e.to_string())?
        .with_client_cert_verifier(verifier)
        .with_single_cert(me.chain(), me.private_key()).map_err(|e| e.to_string())?;
    cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(cfg));
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await.map_err(|e| format!("bind {host}:{port}: {e}"))?;
    tracing::info!("cluster: encrypted listener for paired masters on {host}:{port} (TLS 1.3, pinned certificates)");
    loop {
        let Ok((tcp, addr)) = listener.accept().await else { continue };
        let (acceptor, app) = (acceptor.clone(), app.clone());
        tokio::spawn(async move {
            let tls = match acceptor.accept(tcp).await {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("cluster: refused a TLS connection from {addr}: {e}");
                    return;
                }
            };
            let peer = tls.get_ref().1.peer_certificates().and_then(|c| c.first()).map(|c| pin_of(c));
            let svc = app.layer(axum::Extension(Peer(peer))).layer(axum::Extension(axum::extract::ConnectInfo(addr)));
            let svc = hyper_util::service::TowerToHyperService::new(svc);
            let _ = hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                .serve_connection(hyper_util::rt::TokioIo::new(tls), svc).await;
        });
    }
}

/// A TLS peer's pin, put on each request by `serve`.
#[derive(Clone)]
pub struct Peer(pub Option<String>);

/// The layer (outermost) that keeps PEER_HEADER honest: removed from every request, and set only for one that came
/// through the TLS listener.
pub async fn peer_header(mut req: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    req.headers_mut().remove(PEER_HEADER);
    if let Some(Peer(Some(p))) = req.extensions().get::<Peer>().cloned() {
        if let Ok(v) = p.parse() {
            req.headers_mut().insert(PEER_HEADER, v);
        }
    }
    next.run(req).await
}

/// The peer's pin of this request, when it came through the TLS listener.
pub fn peer_of(headers: &HeaderMap) -> Option<&str> {
    headers.get(PEER_HEADER).and_then(|v| v.to_str().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_keeps_one_certificate_and_its_pin() {
        let d = std::env::temp_dir().join(format!("tls-id-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let a = Identity::load_or_make(&d, "pc2").unwrap();
        let b = Identity::load_or_make(&d, "pc2").unwrap();
        assert_eq!(a.pin, b.pin, "kept, not made again");
        assert!(valid_pin(&a.pin) && !valid_pin("abc"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(d.join("node-tls.json")).unwrap().permissions().mode() & 0o777, 0o600);
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A worker with certificate A, and masters that pinned A or something else, with or without a certificate the
    /// worker knows: only the right pair talks.
    #[tokio::test]
    async fn only_pinned_certificates_talk() {
        let dir = |n: &str| { let d = std::env::temp_dir().join(format!("tls-{n}-{}", std::process::id())); std::fs::create_dir_all(&d).unwrap(); d };
        let (dw, dm, dx) = (dir("w"), dir("m"), dir("x"));
        let worker = Identity::load_or_make(&dw, "w").unwrap();
        let master = Identity::load_or_make(&dm, "m").unwrap();
        let stranger = Identity::load_or_make(&dx, "x").unwrap();
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let app = axum::Router::new().route("/ping", axum::routing::get(|h: HeaderMap| async move {
            peer_of(&h).unwrap_or("none").to_string()
        })).layer(axum::middleware::from_fn(peer_header));
        let mpin = master.pin.clone();
        tokio::spawn(serve(worker.clone(), Box::new(move || vec![mpin.clone()]), app, "127.0.0.1".into(), port));
        // the listener is up once it takes a TCP connection
        for _ in 0..100 {
            if tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let url = format!("https://127.0.0.1:{port}/ping");
        // the paired master, pinning the worker: in, and the handler sees the master's pin
        let ok = client(&master, &worker.pin).unwrap().get(&url).send().await.unwrap().text().await.unwrap();
        assert_eq!(ok, master.pin);
        // the master pinning another certificate: it refuses the worker
        let e = client(&master, &stranger.pin).unwrap().get(&url).send().await.unwrap_err();
        assert!(format!("{e:?}").contains("not the one paired"), "{e:?}");
        // a certificate the worker never paired: refused
        assert!(client(&stranger, &worker.pin).unwrap().get(&url).send().await.is_err());
        // no certificate at all: refused
        let bare = reqwest::Client::builder().danger_accept_invalid_certs(true).build().unwrap();
        assert!(bare.get(&url).send().await.is_err());
        // cleanup only: on macOS a folder can already be gone here, which is not what this test is about
        for d in [dw, dm, dx] {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}
