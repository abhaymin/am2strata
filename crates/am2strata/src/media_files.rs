//! The media library: every image, video and sound the server makes, and every one sent to it to edit, transcribe or
//! check, kept once by its content so it can be used again by id (`mf_` and 16 hex digits of its SHA-256).
//!
//! - **Where:** `<data dir>/media-files/<id>.<ext>`, and `index.json` beside them with what is known of each file.
//! - **Reuse:** any media input that takes base64, a data: URL or a URL also takes a file id (`resolve`).
//! - **Room:** `media_files_max_gib` in the run config (default 10); past it the least recently used files go.
//! - **Age:** a file made or sent more than `media_files_max_age_days` ago (default 90) goes, unless it is pinned.
//!   A pinned file stays until it is unpinned or deleted, and is never the one dropped for room.
//! - **Who sees what:** the main key sees every file; a client key sees and uses the files it sent or had made.
//!
//! Routes (main.rs): GET/POST /v1/media/files, GET/DELETE /v1/media/files/{id}, GET /v1/media/files/{id}/content.

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

/// The server's library (main.rs sets it at start; the monitor and the media endpoints add to it).
pub static FILES: OnceLock<Arc<Library>> = OnceLock::new();

/// The owner name of the main key's files.
pub const MAIN: &str = "main";

pub struct Library {
    dir: PathBuf,
    max_bytes: u64,
    /// files older than this (seconds since they were first kept) go, unless pinned; 0: no limit
    max_age_s: f64,
    index: Mutex<BTreeMap<String, Value>>,
}

/// A file's id: `mf_` and the first 16 hex digits of its SHA-256.
pub fn id_of(bytes: &[u8]) -> String {
    let h = Sha256::digest(bytes);
    format!("mf_{}", h.iter().take(8).map(|b| format!("{b:02x}")).collect::<String>())
}

/// Whether `s` is a file id (and nothing else).
pub fn is_id(s: &str) -> bool {
    s.len() == 19 && s.starts_with("mf_") && s[3..].bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// The kind and extension of a file from its first bytes (what was sent is not trusted for this).
pub fn sniff(bytes: &[u8]) -> (&'static str, &'static str) {
    let b = |at: usize, s: &[u8]| bytes.len() >= at + s.len() && &bytes[at..at + s.len()] == s;
    if b(0, b"\x89PNG") { ("image", "png") }
    else if b(0, b"\xff\xd8\xff") { ("image", "jpg") }
    else if b(0, b"GIF8") { ("image", "gif") }
    else if b(0, b"RIFF") && b(8, b"WEBP") { ("image", "webp") }
    else if b(0, b"RIFF") && b(8, b"WAVE") { ("audio", "wav") }
    else if b(4, b"ftyp") { if b(8, b"M4A ") { ("audio", "m4a") } else { ("video", "mp4") } }
    else if b(0, b"\x1aE\xdf\xa3") { ("video", "webm") }
    else if b(0, b"ID3") || (bytes.len() > 1 && bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0) { ("audio", "mp3") }
    else if b(0, b"OggS") { ("audio", "ogg") }
    else if b(0, b"fLaC") { ("audio", "flac") }
    else { ("file", "bin") }
}

/// Now in seconds, to the millisecond: the order of use matters within a second.
fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as f64 / 1000.0).unwrap_or(0.0)
}

fn mime(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png", "jpg" => "image/jpeg", "gif" => "image/gif", "webp" => "image/webp",
        "wav" => "audio/wav", "mp3" => "audio/mpeg", "ogg" => "audio/ogg", "flac" => "audio/flac", "m4a" => "audio/mp4",
        "mp4" => "video/mp4", "webm" => "video/webm", _ => "application/octet-stream",
    }
}

impl Library {
    pub fn open(dir: PathBuf, max_gib: f64) -> Library {
        Self::open_with_age(dir, max_gib, 90.0)
    }

    pub fn open_with_age(dir: PathBuf, max_gib: f64, max_age_days: f64) -> Library {
        let index = std::fs::read(dir.join("index.json")).ok().and_then(|b| serde_json::from_slice::<BTreeMap<String, Value>>(&b).ok())
            .unwrap_or_default();
        // entries whose file is gone (removed by hand) are forgotten
        let index = index.into_iter().filter(|(_, m)| dir.join(m["file"].as_str().unwrap_or("")).is_file()).collect();
        let l = Library { dir, max_bytes: (max_gib.max(0.0) * (1u64 << 30) as f64) as u64, max_age_s: max_age_days.max(0.0) * 86400.0,
                          index: Mutex::new(index) };
        l.expire();
        l
    }

    fn save(&self, index: &BTreeMap<String, Value>) {
        let _ = std::fs::create_dir_all(&self.dir);
        let tmp = self.dir.join("index.json.tmp");
        if std::fs::write(&tmp, serde_json::to_vec(index).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(&tmp, self.dir.join("index.json"));
        }
    }

    /// Keeps `bytes` (once: the same content is the same file) -> its id. `source` is "generated" or "uploaded";
    /// `meta` adds what is known (model, prompt, seed, name); `owner` may see and use it (None: added later by `own`).
    pub fn put(&self, bytes: &[u8], source: &str, meta: Value, owner: Option<&str>) -> Option<String> {
        if bytes.is_empty() || self.max_bytes == 0 {
            return None;
        }
        let id = id_of(bytes);
        let (kind, ext) = sniff(bytes);
        let now = now();
        let mut idx = self.index.lock().unwrap();
        let entry = idx.entry(id.clone()).or_insert_with(|| json!({"id": id, "file": format!("{id}.{ext}"), "kind": kind, "mime": mime(ext),
                                                                  "bytes": bytes.len(), "source": source, "created": now, "owners": []}));
        let path = self.dir.join(entry["file"].as_str().unwrap_or_default());
        if !path.is_file() {
            let _ = std::fs::create_dir_all(&self.dir);
            if std::fs::write(&path, bytes).is_err() {
                idx.remove(&id);
                return None;
            }
        }
        entry["last_used"] = json!(now);
        // what was learnt about it (a generated file keeps its first prompt and seed; an upload gains a name)
        if let Some(m) = meta.as_object() {
            for (k, v) in m {
                if !v.is_null() && entry.get(k).is_none_or(Value::is_null) {
                    entry[k] = v.clone();
                }
            }
        }
        if let Some(o) = owner {
            add_owner(entry, o);
        }
        self.evict(&mut idx, &id);
        self.save(&idx);
        drop(idx);
        self.expire();
        Some(id)
    }

    /// `owner` may also see and use the file `id`.
    pub fn own(&self, id: &str, owner: &str) {
        let mut idx = self.index.lock().unwrap();
        if let Some(e) = idx.get_mut(id) {
            add_owner(e, owner);
            self.save(&idx);
        }
    }

    /// Least recently used first, until the library fits; never `keep` (the file just added) nor a pinned one.
    fn evict(&self, idx: &mut BTreeMap<String, Value>, keep: &str) {
        let total = |idx: &BTreeMap<String, Value>| idx.values().map(|m| m["bytes"].as_u64().unwrap_or(0)).sum::<u64>();
        while total(idx) > self.max_bytes {
            let Some(old) = idx.iter().filter(|(k, e)| *k != keep && e["pinned"] != true)
                .min_by(|a, b| a.1["last_used"].as_f64().unwrap_or(0.0).total_cmp(&b.1["last_used"].as_f64().unwrap_or(0.0)))
                .map(|(k, _)| k.clone()) else { break };
            if let Some(m) = idx.remove(&old) {
                let _ = std::fs::remove_file(self.dir.join(m["file"].as_str().unwrap_or("")));
            }
        }
    }

    /// Removes the files older than the age limit that are not pinned. -> how many went.
    pub fn expire(&self) -> usize {
        if self.max_age_s <= 0.0 {
            return 0;
        }
        let cut = now() - self.max_age_s;
        let mut idx = self.index.lock().unwrap();
        let old: Vec<String> = idx.iter().filter(|(_, e)| e["pinned"] != true && e["created"].as_f64().unwrap_or(0.0) < cut)
            .map(|(k, _)| k.clone()).collect();
        for k in &old {
            if let Some(m) = idx.remove(k) {
                let _ = std::fs::remove_file(self.dir.join(m["file"].as_str().unwrap_or("")));
            }
        }
        if !old.is_empty() {
            self.save(&idx);
        }
        old.len()
    }

    /// Pins (kept past the age limit and never dropped for room) or unpins the file `id`. -> its entry.
    pub fn pin(&self, id: &str, owner: &str, pinned: bool) -> Option<Value> {
        let mut idx = self.index.lock().unwrap();
        let e = idx.get_mut(id).filter(|e| may(e, owner))?;
        e["pinned"] = json!(pinned);
        let out = e.clone();
        self.save(&idx);
        Some(out)
    }

    pub fn max_age_days(&self) -> f64 {
        self.max_age_s / 86400.0
    }

    /// The file `id` and its entry, when `owner` may use it (the main key may use any); its last use is now.
    pub fn get(&self, id: &str, owner: &str) -> Option<(Value, Vec<u8>)> {
        let mut idx = self.index.lock().unwrap();
        let e = idx.get_mut(id).filter(|e| may(e, owner))?;
        let bytes = std::fs::read(self.dir.join(e["file"].as_str().unwrap_or(""))).ok()?;
        e["last_used"] = json!(now());
        let out = e.clone();
        drop(idx);
        Some((out, bytes))
    }

    pub fn entry(&self, id: &str, owner: &str) -> Option<Value> {
        self.index.lock().unwrap().get(id).filter(|e| may(e, owner)).cloned()
    }

    /// Newest first: `kind` (image, video, audio) and `source` (generated, uploaded) filter.
    pub fn list(&self, owner: &str, kind: Option<&str>, source: Option<&str>) -> Vec<Value> {
        self.expire();
        let mut v: Vec<Value> = self.index.lock().unwrap().values()
            .filter(|e| may(e, owner) && kind.is_none_or(|k| e["kind"] == k) && source.is_none_or(|s| e["source"] == s)).cloned().collect();
        v.sort_by(|a, b| b["created"].as_f64().unwrap_or(0.0).total_cmp(&a["created"].as_f64().unwrap_or(0.0)));
        v
    }

    /// Removes the file `id` (the main key, or a client that alone owns it; one of several owners only lets go of it).
    pub fn delete(&self, id: &str, owner: &str) -> bool {
        let mut idx = self.index.lock().unwrap();
        let Some(e) = idx.get_mut(id).filter(|e| may(e, owner)) else { return false };
        if owner != MAIN {
            if let Some(o) = e["owners"].as_array_mut() {
                o.retain(|x| x != owner);
                if !o.is_empty() {
                    self.save(&idx);
                    return true;
                }
            }
        }
        if let Some(m) = idx.remove(id) {
            let _ = std::fs::remove_file(self.dir.join(m["file"].as_str().unwrap_or("")));
        }
        self.save(&idx);
        true
    }

    pub fn usage(&self) -> (usize, u64, u64) {
        let idx = self.index.lock().unwrap();
        (idx.len(), idx.values().map(|m| m["bytes"].as_u64().unwrap_or(0)).sum(), self.max_bytes)
    }
}

fn add_owner(e: &mut Value, owner: &str) {
    if !e["owners"].is_array() {
        e["owners"] = json!([]);
    }
    let o = e["owners"].as_array_mut().unwrap();
    if !o.iter().any(|x| x == owner) {
        o.push(json!(owner));
    }
}

/// The main key sees everything; a client key the files it owns.
fn may(e: &Value, owner: &str) -> bool {
    owner == MAIN || e["owners"].as_array().is_some_and(|o| o.iter().any(|x| x == owner))
}

/// Keeps a file the server made (the monitor calls this for every media job's output) -> its id.
pub fn keep_generated(bytes: &[u8], meta: Value) -> Option<String> {
    FILES.get()?.put(bytes, "generated", meta, None)
}

/// Keeps a file a request sent -> its id.
pub fn keep_uploaded(bytes: &[u8], owner: &str, meta: Value) -> Option<String> {
    FILES.get()?.put(bytes, "uploaded", meta, Some(owner))
}

/// A media input that is a file id: its bytes, when `owner` may use it. None when `src` is no file id.
pub fn resolve(src: &str, owner: &str) -> Option<Result<Vec<u8>, String>> {
    let src = src.trim();
    if !is_id(src) {
        return None;
    }
    Some(match FILES.get() {
        None => Err("the media library is off on this server".to_string()),
        Some(l) => l.get(src, owner).map(|(_, b)| b).ok_or_else(|| format!("no media file {src} (or not one this key may use)")),
    })
}

/// Who a request is, for the library: "main" for the main key, a client profile's name, the cluster's master.
pub fn owner_of(s: &crate::AppState, headers: &HeaderMap) -> Option<String> {
    match crate::api_access(s, headers)? {
        None => Some(MAIN.to_string()),
        Some(p) => Some(format!("client:{}", p.name)),
    }
}

// ---------------------------------------------------------------------------------------------------------------- HTTP

fn lib() -> Result<Arc<Library>, Response> {
    FILES.get().cloned().ok_or_else(|| crate::error(StatusCode::NOT_FOUND, "invalid_request_error", "the media library is off on this server"))
}

fn who(s: &crate::AppState, headers: &HeaderMap) -> Result<String, Response> {
    owner_of(s, headers).ok_or_else(|| crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required"))
}

/// GET /v1/media/files?kind=image&source=generated&limit=50&offset=0
pub async fn list(State(s): State<crate::AppState>, headers: HeaderMap, Query(q): Query<BTreeMap<String, String>>) -> Response {
    let (owner, l) = match (who(&s, &headers), lib()) { (Ok(o), Ok(l)) => (o, l), (Err(e), _) | (_, Err(e)) => return e };
    let all = l.list(&owner, q.get("kind").map(String::as_str), q.get("source").map(String::as_str));
    let limit = q.get("limit").and_then(|v| v.parse::<usize>().ok()).unwrap_or(100).clamp(1, 1000);
    let offset = q.get("offset").and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
    let (n, used, max) = l.usage();
    let page: Vec<Value> = all.iter().skip(offset).take(limit).map(public).collect();
    Json(json!({"object": "list", "data": page, "total": all.len(), "max_age_days": l.max_age_days(),
                "library": if owner == MAIN { json!({"files": n, "bytes": used, "max_bytes": max}) } else { Value::Null }})).into_response()
}

/// What a client is told of an entry (not its file name on disk, nor other owners).
fn public(e: &Value) -> Value {
    let mut v = e.clone();
    if let Some(o) = v.as_object_mut() {
        o.remove("file");
        o.remove("owners");
        o.insert("object".into(), json!("media.file"));
    }
    v
}

/// POST /v1/media/files: JSON {"data": base64 or a data: URL, "name"?} or multipart `file` -> the entry (its id).
pub async fn upload(State(s): State<crate::AppState>, req: axum::extract::Request) -> Response {
    let headers = req.headers().clone();
    let (owner, l) = match (who(&s, &headers), lib()) { (Ok(o), Ok(l)) => (o, l), (Err(e), _) | (_, Err(e)) => return e };
    let bad = |m: &str| crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", m);
    let ctype = headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
    let (bytes, name) = if ctype.starts_with("multipart/form-data") {
        let Ok(mut mp) = axum::extract::Multipart::from_request(req, &s).await else { return bad("multipart/form-data could not be read") };
        let mut got = None;
        while let Ok(Some(f)) = mp.next_field().await {
            if f.name() == Some("file") {
                let name = f.file_name().map(str::to_string);
                match f.bytes().await {
                    Ok(b) => got = Some((b.to_vec(), name)),
                    Err(e) => return bad(&format!("the file could not be read: {e}")),
                }
            }
        }
        match got { Some(g) => g, None => return bad("send the file as the multipart field `file`") }
    } else {
        let Ok(Json(v)) = Json::<Value>::from_request(req, &s).await else { return bad("send JSON {\"data\": base64 or a data: URL}, or multipart `file`") };
        let raw = v["data"].as_str().unwrap_or("");
        let b64 = raw.split_once(";base64,").map(|(_, b)| b).unwrap_or(raw);
        match base64::engine::general_purpose::STANDARD.decode(b64.trim()) {
            Ok(b) if !b.is_empty() => (b, v["name"].as_str().map(str::to_string)),
            _ => return bad("\"data\": base64 or a data: URL"),
        }
    };
    if bytes.len() > 512 << 20 {
        return bad("a file of at most 512 MiB");
    }
    match l.put(&bytes, "uploaded", json!({"name": name}), Some(&owner)) {
        Some(id) => Json(public(&l.entry(&id, &owner).unwrap_or(json!({"id": id})))).into_response(),
        None => crate::error(StatusCode::INSUFFICIENT_STORAGE, "server_error", "the file could not be kept (the library is off or the disk is full)"),
    }
}

use axum::extract::FromRequest;

/// GET /v1/media/files/{id}
pub async fn get(State(s): State<crate::AppState>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    let (owner, l) = match (who(&s, &headers), lib()) { (Ok(o), Ok(l)) => (o, l), (Err(e), _) | (_, Err(e)) => return e };
    match l.entry(&id, &owner) {
        Some(e) => Json(public(&e)).into_response(),
        None => crate::error(StatusCode::NOT_FOUND, "invalid_request_error", format!("no media file {id}")),
    }
}

/// GET /v1/media/files/{id}/content: the file itself, with its type.
pub async fn content(State(s): State<crate::AppState>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    let (owner, l) = match (who(&s, &headers), lib()) { (Ok(o), Ok(l)) => (o, l), (Err(e), _) | (_, Err(e)) => return e };
    match l.get(&id, &owner) {
        Some((e, bytes)) => {
            let mut r = bytes.into_response();
            if let Ok(v) = e["mime"].as_str().unwrap_or("application/octet-stream").parse() {
                r.headers_mut().insert("content-type", v);
            }
            if let Ok(v) = "private, max-age=31536000, immutable".parse() {
                r.headers_mut().insert("cache-control", v);
            }
            r
        }
        None => crate::error(StatusCode::NOT_FOUND, "invalid_request_error", format!("no media file {id}")),
    }
}

/// PATCH /v1/media/files/{id} {"pinned": true|false}: a pinned file stays past the age limit and is never dropped for room.
pub async fn patch(State(s): State<crate::AppState>, headers: HeaderMap, Path(id): Path<String>, Json(req): Json<Value>) -> Response {
    let (owner, l) = match (who(&s, &headers), lib()) { (Ok(o), Ok(l)) => (o, l), (Err(e), _) | (_, Err(e)) => return e };
    let Some(pinned) = req["pinned"].as_bool() else {
        return crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"pinned\": true} or {\"pinned\": false}");
    };
    match l.pin(&id, &owner, pinned) {
        Some(e) => Json(public(&e)).into_response(),
        None => crate::error(StatusCode::NOT_FOUND, "invalid_request_error", format!("no media file {id}")),
    }
}

/// DELETE /v1/media/files/{id}
pub async fn delete(State(s): State<crate::AppState>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    let (owner, l) = match (who(&s, &headers), lib()) { (Ok(o), Ok(l)) => (o, l), (Err(e), _) | (_, Err(e)) => return e };
    if l.delete(&id, &owner) {
        Json(json!({"id": id, "object": "media.file", "deleted": true})).into_response()
    } else {
        crate::error(StatusCode::NOT_FOUND, "invalid_request_error", format!("no media file {id}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("am2-mf-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_file_is_kept_once_used_by_id_and_seen_only_by_its_owners() {
        let d = tmp();
        let l = Library::open(d.clone(), 1.0);
        let png = b"\x89PNG\r\n\x1a\nfake image".to_vec();
        let id = l.put(&png, "uploaded", json!({"name": "a.png"}), Some("client:a")).unwrap();
        assert!(is_id(&id) && id == id_of(&png));
        assert_eq!(l.put(&png, "uploaded", json!({}), Some("client:b")).as_deref(), Some(id.as_str()), "the same content is the same file");
        assert_eq!(l.entry(&id, "client:a").unwrap()["kind"], "image");
        assert!(l.get(&id, "client:c").is_none(), "another client neither sees nor uses it");
        assert_eq!(l.get(&id, MAIN).unwrap().1, png);
        assert_eq!(l.list("client:b", Some("image"), None).len(), 1);
        // one owner letting go keeps it for the other; the main key removes it
        assert!(l.delete(&id, "client:a") && l.entry(&id, "client:b").is_some() && l.entry(&id, "client:a").is_none());
        assert!(l.delete(&id, MAIN) && l.entry(&id, MAIN).is_none());
        // the index survives a restart
        let id2 = l.put(b"RIFF\0\0\0\0WAVEdata", "generated", json!({"model": "m"}), None).unwrap();
        assert_eq!(Library::open(d.clone(), 1.0).entry(&id2, MAIN).unwrap()["model"], "m");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_least_recently_used_files_go_when_the_library_is_full() {
        let d = tmp();
        let l = Library::open(d.clone(), 2500.0 / (1u64 << 30) as f64);
        let a = l.put(&[1u8; 1000], "uploaded", json!({}), Some(MAIN)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = l.put(&[2u8; 1000], "uploaded", json!({}), Some(MAIN)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        l.get(&a, MAIN);   // a used again: b is now the oldest
        std::thread::sleep(std::time::Duration::from_millis(5));
        let c = l.put(&[3u8; 1000], "uploaded", json!({}), Some(MAIN)).unwrap();
        assert!(l.entry(&a, MAIN).is_some() && l.entry(&c, MAIN).is_some() && l.entry(&b, MAIN).is_none());
        assert_eq!(std::fs::read_dir(&d).unwrap().filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "bin")).count(), 2);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn old_files_go_unless_pinned() {
        let d = tmp();
        let l = Library::open_with_age(d.clone(), 1.0, 90.0);
        let a = l.put(&[1u8; 10], "uploaded", json!({}), Some(MAIN)).unwrap();
        let b = l.put(&[2u8; 10], "uploaded", json!({}), Some(MAIN)).unwrap();
        l.pin(&b, MAIN, true).unwrap();
        // both made 100 days ago
        for id in [&a, &b] {
            l.index.lock().unwrap().get_mut(id.as_str()).unwrap()["created"] = json!(now() - 100.0 * 86400.0);
        }
        assert_eq!(l.expire(), 1);
        assert!(l.entry(&a, MAIN).is_none() && l.entry(&b, MAIN).is_some(), "the pinned one stays");
        l.pin(&b, MAIN, false);
        assert_eq!(l.list(MAIN, None, None).len(), 0, "unpinned, it goes at the next look");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn files_are_told_by_their_first_bytes() {
        assert_eq!(sniff(b"\xff\xd8\xff\xe0"), ("image", "jpg"));
        assert_eq!(sniff(b"\0\0\0\x20ftypisom"), ("video", "mp4"));
        assert_eq!(sniff(b"RIFF\0\0\0\0WAVE"), ("audio", "wav"));
        assert!(!is_id("mf_0123") && is_id("mf_0123456789abcdef") && !is_id("mf_0123456789ABCDEF"));
    }
}
