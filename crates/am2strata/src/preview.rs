//! The canvas's sub-browser: a page from a chat (HTML, a diagram) gets a real address, /preview/<id>, so it can be
//! reloaded, kept in the canvas's history and opened in its own browser tab. Every preview is served with a
//! `Content-Security-Policy: sandbox` header: its origin is opaque, so even in its own tab it cannot read this
//! server's API or the web app's storage. The id is the capability (128 random bits); the last 64 are kept in memory.

use crate::{authorized, error, foreign_origin, not_own_page, AppState};
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::Mutex;

const KEEP: usize = 64;
const MAX_BYTES: usize = 8 << 20;
pub const SANDBOX: &str = "sandbox allow-scripts allow-forms allow-modals allow-popups allow-downloads";

#[derive(Default)]
pub struct Previews(Mutex<VecDeque<(String, String)>>);

impl Previews {
    fn put(&self, html: String) -> String {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let mut q = self.0.lock().unwrap();
        q.push_back((id.clone(), html));
        while q.len() > KEEP {
            q.pop_front();
        }
        id
    }
    fn get(&self, id: &str) -> Option<String> {
        self.0.lock().unwrap().iter().find(|(i, _)| i == id).map(|(_, h)| h.clone())
    }
}

pub async fn create(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if foreign_origin(&s, &headers) {
        return not_own_page("previews can be made");
    }
    let Some(html) = req.get("html").and_then(Value::as_str) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"html\": \"...\"}");
    };
    if html.len() > MAX_BYTES {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", "a preview is at most 8 MB");
    }
    let id = s.previews.put(html.to_string());
    Json(json!({"id": id, "url": format!("/preview/{id}")})).into_response()
}

pub async fn show(State(s): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(html) = s.previews.get(&id) else {
        return (StatusCode::NOT_FOUND, [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                "<!doctype html><title>Gone</title><p style=\"font-family:sans-serif\">This preview is no longer kept \
                 (am2Strata keeps the last 64 and forgets them when it restarts). Open it again from the chat.</p>").into_response();
    };
    let mut r = (StatusCode::OK, html).into_response();
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
    h.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(SANDBOX));
    h.insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    h.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    r
}
