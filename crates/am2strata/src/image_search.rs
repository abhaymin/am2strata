//! Image search by words, and reverse image search: where an image (or one like it) appears on the web. Private to
//! this machine: the searches run from here, in a throwaway browser session (browser.rs), and the results come back
//! to the caller only; nothing is shared or kept except the searched image itself in the media library.
//!
//! - **By words** (`POST /v1/search/images`): SearXNG's image search when the run config has one, else Wikimedia
//!   Commons (freely licensed images, with their pages).
//! - **By image** (`POST /v1/search/reverse`): the image is sent to the chosen engines, which is the one thing that
//!   leaves this machine. Yandex: what the image appears to contain, and the sites it appears on, read in the server's
//!   own headless browser. Google Lens: it answers a browser like this one with a CAPTCHA, which is never solved or
//!   got round here: its results link comes back for the user to open in their own browser.
//! - **No content filter:** nothing is refused or filtered for what an image shows, and the engines are asked for no
//!   safe search where they take the setting (Google's `safe=off`). What an engine does on its own side is its own.
//! - **What it is not for:** finding who a person is from their face. No face-search engine is used, and the tool
//!   says so.

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::time::Duration;

const UA: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36";

/// The reverse-search engines, in the order they are tried.
pub const ENGINES: [&str; 2] = ["yandex", "google_lens"];

fn client() -> reqwest::Client {
    reqwest::Client::builder().user_agent(UA).timeout(Duration::from_secs(45)).redirect(reqwest::redirect::Policy::none())
        .build().unwrap_or_default()
}

/// The host of a URL, without "www.".
pub fn domain(url: &str) -> String {
    reqwest::Url::parse(url).ok().and_then(|u| u.host_str().map(|h| h.trim_start_matches("www.").to_string())).unwrap_or_default()
}

// ------------------------------------------------------------------------------------------------------- by words

/// Images for `query` -> (results, the backend that answered).
pub async fn by_words(query: &str, n: usize, searxng: Option<&str>) -> Result<(Vec<Value>, &'static str), String> {
    let mut why = Vec::new();
    if let Some(base) = searxng {
        match searxng_images(base, query, n).await {
            Ok(v) if !v.is_empty() => return Ok((v, "searxng")),
            Ok(_) => why.push("searxng: no results".to_string()),
            Err(e) => why.push(format!("searxng: {e}")),
        }
    }
    match commons(query, n).await {
        Ok(v) => Ok((v, "wikimedia_commons")),
        Err(e) => {
            why.push(format!("wikimedia commons: {e}"));
            Err(why.join("; "))
        }
    }
}

async fn searxng_images(base: &str, q: &str, n: usize) -> Result<Vec<Value>, String> {
    let url = reqwest::Url::parse_with_params(&format!("{}/search", base.trim_end_matches('/')),
                                              &[("q", q), ("format", "json"), ("categories", "images")]).map_err(|e| e.to_string())?;
    let v: Value = client().get(url).send().await.map_err(|e| e.to_string())?.json().await.map_err(|e| e.to_string())?;
    Ok(v["results"].as_array().into_iter().flatten().take(n).map(|r| json!({
        "title": r["title"], "image_url": r["img_src"], "thumbnail": r["thumbnail_src"].as_str().or(r["img_src"].as_str()),
        "page_url": r["url"], "source": domain(r["url"].as_str().unwrap_or("")), "size": r["resolution"],
    })).collect())
}

/// Wikimedia Commons: files whose title or description match, with their image and page.
async fn commons(q: &str, n: usize) -> Result<Vec<Value>, String> {
    let url = reqwest::Url::parse_with_params("https://commons.wikimedia.org/w/api.php", &[
        ("action", "query"), ("format", "json"), ("generator", "search"), ("gsrsearch", q), ("gsrnamespace", "6"),
        ("gsrlimit", &n.clamp(1, 50).to_string()), ("prop", "imageinfo"), ("iiprop", "url|size|extmetadata"), ("iiurlwidth", "320"),
    ]).map_err(|e| e.to_string())?;
    let v: Value = client().get(url).send().await.map_err(|e| e.to_string())?.json().await.map_err(|e| e.to_string())?;
    let mut out: Vec<(i64, Value)> = v["query"]["pages"].as_object().into_iter().flatten().filter_map(|(_, p)| {
        let ii = &p["imageinfo"][0];
        let img = ii["url"].as_str()?;
        Some((p["index"].as_i64().unwrap_or(0), json!({
            "title": p["title"].as_str().unwrap_or("").trim_start_matches("File:"), "image_url": img,
            "thumbnail": ii["thumburl"].as_str().unwrap_or(img), "page_url": ii["descriptionurl"],
            "source": "commons.wikimedia.org", "size": format!("{}x{}", ii["width"], ii["height"]),
            "license": ii["extmetadata"]["LicenseShortName"]["value"],
        })))
    }).collect();
    out.sort_by_key(|(i, _)| *i);
    Ok(out.into_iter().map(|(_, v)| v).take(n).collect())
}

// ------------------------------------------------------------------------------------------------------- by image

/// Yandex's id for an uploaded image (its "cbir" id), from which its results pages open.
async fn yandex_upload(bytes: &[u8]) -> Result<String, String> {
    let part = reqwest::multipart::Part::bytes(bytes.to_vec()).file_name("image").mime_str("application/octet-stream").map_err(|e| e.to_string())?;
    let r = client().post("https://yandex.com/images/search?rpt=imageview&format=json&request=%7B%22blocks%22%3A%5B%7B%22block%22%3A%22b-page_type_search-by-image__link%22%7D%5D%7D")
        .multipart(reqwest::multipart::Form::new().part("upfile", part)).send().await.map_err(|e| e.to_string())?;
    let v: Value = r.json().await.map_err(|e| format!("its answer was not read ({e})"))?;
    v["blocks"][0]["params"]["cbirId"].as_str().map(str::to_string).ok_or_else(|| "it gave no id for the image".to_string())
}

/// Google Lens's results address for an uploaded image (its upload answers with a redirect to them).
async fn lens_upload(bytes: &[u8]) -> Result<String, String> {
    let part = reqwest::multipart::Part::bytes(bytes.to_vec()).file_name("image").mime_str("application/octet-stream").map_err(|e| e.to_string())?;
    let r = client().post("https://lens.google.com/v3/upload").multipart(reqwest::multipart::Form::new().part("encoded_image", part))
        .send().await.map_err(|e| e.to_string())?;
    // safe search off: the search never filters by content (the owner's choice for this private tool)
    r.headers().get("location").and_then(|l| l.to_str().ok()).map(|u| if u.contains('?') { format!("{u}&safe=off") } else { format!("{u}?safe=off") })
        .ok_or_else(|| format!("it answered HTTP {} without a results address", r.status().as_u16()))
}

/// The page's links that lead off the engine's own sites, once each, as {title, url, domain}.
fn sites_from(links_json: &str, engine_hosts: &[&str], n: usize) -> Vec<Value> {
    let rows: Vec<Value> = links_json.find('[').and_then(|i| serde_json::from_str(&links_json[i..]).ok()).unwrap_or_default();
    let mut seen = std::collections::HashSet::new();
    rows.iter().filter_map(|r| {
        let url = r["url"].as_str()?;
        let title = r["text"].as_str().unwrap_or("").trim();
        let d = domain(url);
        if d.is_empty() || engine_hosts.iter().any(|h| d == *h || d.ends_with(&format!(".{h}"))) || !url.starts_with("http") {
            return None;
        }
        // a link named by its size ("1280×720") is the image file itself, not a page it appears on
        if title.is_empty() || is_size(title) {
            return None;
        }
        (seen.insert(url.to_string()) && seen.insert(format!("{d}|{title}"))).then(|| json!({"title": title, "url": url, "domain": d}))
    }).take(n).collect()
}

/// "1280×720", "800x416": a size, not a title.
fn is_size(t: &str) -> bool {
    let t = t.replace('×', "x");
    t.split_once('x').is_some_and(|(a, b)| !a.is_empty() && !b.is_empty() && a.trim().chars().all(|c| c.is_ascii_digit())
        && b.trim().chars().all(|c| c.is_ascii_digit()))
}

/// What Yandex says the image contains: its tags, each a link to a search by those words.
fn yandex_tags(links_json: &str) -> Vec<String> {
    let rows: Vec<Value> = links_json.find('[').and_then(|i| serde_json::from_str(&links_json[i..]).ok()).unwrap_or_default();
    let mut out: Vec<String> = Vec::new();
    for r in &rows {
        let (Some(u), Some(t)) = (r["url"].as_str(), r["text"].as_str()) else { continue };
        if u.contains("/images/search?text=") && !t.trim().is_empty() && !out.iter().any(|x| x == t.trim()) {
            out.push(t.trim().to_string());
        }
    }
    out
}

/// A page that stops a browser to ask whether it is a person.
pub fn is_captcha(url: &str, text: &str) -> bool {
    let t = text.to_lowercase();
    url.contains("/sorry/") || url.contains("showcaptcha") || t.contains("unusual traffic") || t.contains("are you a robot")
        || t.contains("i'm not a robot") || t.contains("confirm you're not a robot")
}

async fn step(b: &crate::browser::Browsers, sess: &str, tool: &str, args: Value) -> Result<String, String> {
    b.call(sess, tool, &args).await.map(|s| s.text)
}

/// One engine's answer for the image: its status, what it says the image contains, the sites, and its results link.
async fn engine(b: &crate::browser::Browsers, sess: &str, name: &str, bytes: &[u8], n: usize) -> Value {
    match name {
        "yandex" => {
            let id = match yandex_upload(bytes).await {
                Ok(id) => id,
                Err(e) => return json!({"engine": name, "status": "failed", "note": format!("the upload: {e}")}),
            };
            let enc = urlencode(&id);
            let results = format!("https://yandex.com/images/search?rpt=imageview&cbir_id={enc}");
            let sites_url = format!("https://yandex.com/images/search?rpt=imageview&cbir_id={enc}&cbir_page=sites");
            let similar_url = format!("https://yandex.com/images/search?rpt=imageview&cbir_id={enc}&cbir_page=similar");
            let first = match step(b, sess, "browser_open", json!({"url": results})).await {
                Ok(t) => t,
                Err(e) => return json!({"engine": name, "status": "failed", "results_url": results, "note": e}),
            };
            if is_captcha(&results, &first) {
                return json!({"engine": name, "status": "needs_you", "results_url": results,
                              "note": "Yandex asked whether this is a person: open the results link in your own browser"});
            }
            let tags = yandex_tags(&step(b, sess, "browser_extract", json!({"kind": "links"})).await.unwrap_or_default());
            let _ = step(b, sess, "browser_open", json!({"url": sites_url})).await;
            let links = step(b, sess, "browser_extract", json!({"kind": "links"})).await.unwrap_or_default();
            let sites = sites_from(&links, &["yandex.com", "yandex.ru", "yandex.net", "ya.ru", "yandex.com.tr"], n);
            json!({"engine": name, "status": "ok", "contains": tags, "sites": sites, "results_url": results, "similar_url": similar_url})
        }
        "google_lens" => {
            let url = match lens_upload(bytes).await {
                Ok(u) => u,
                Err(e) => return json!({"engine": name, "status": "failed", "note": format!("the upload: {e}")}),
            };
            let text = step(b, sess, "browser_open", json!({"url": url})).await.unwrap_or_default();
            let now = b.current(sess).await;
            let shown = match now { Some(p) => p.url().await.ok().flatten().unwrap_or_default(), None => String::new() };
            if is_captcha(&shown, &text) {
                return json!({"engine": name, "status": "needs_you", "results_url": url,
                              "note": "Google asks a browser like this one to prove it is a person; that is never solved here. \
                                       Open the results link in your own browser (it is valid for a while)."});
            }
            let links = step(b, sess, "browser_extract", json!({"kind": "links"})).await.unwrap_or_default();
            let sites = sites_from(&links, &["google.com", "gstatic.com", "googleusercontent.com", "youtube.com", "google.co.in"], n);
            json!({"engine": name, "status": "ok", "sites": sites, "results_url": url})
        }
        other => json!({"engine": other, "status": "failed", "note": "no such engine: yandex or google_lens"}),
    }
}

fn urlencode(s: &str) -> String {
    s.bytes().map(|c| match c {
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (c as char).to_string(),
        _ => format!("%{c:02X}"),
    }).collect()
}

/// The image searched for, sent to each engine in turn, read in one throwaway browser session that ends after.
pub async fn by_image(b: &crate::browser::Browsers, bytes: &[u8], engines: &[String], n: usize) -> Vec<Value> {
    let sess = format!("reverse-search-{}", uuid::Uuid::new_v4().simple());
    let mut out = Vec::new();
    for e in engines {
        out.push(engine(b, &sess, e, bytes, n).await);
    }
    b.close_session(&sess).await;
    out
}

/// The chat tool `reverse_image_search` (run.rs routes it here: it needs the server's browser and the media library).
pub async fn tool_reverse(mgr: &crate::manager::Manager, args: &Value) -> Value {
    let t0 = std::time::Instant::now();
    let fail = |e: String| json!({"ok": false, "text": format!("error: {e}"), "chars": 0, "truncated": false, "ms": 0});
    let Some(src) = args["image"].as_str().map(str::trim).filter(|s| !s.is_empty()) else { return fail("give the image: a media library id (mf_...) or its address".into()) };
    let bytes = match crate::media::load_source(src).await {
        Ok(b) if image::guess_format(&b).is_ok() => b,
        Ok(_) => return fail("that is not an image".into()),
        Err(e) => return fail(format!("{e:#}")),
    };
    let b = match crate::browser_api::pick(mgr, "server").await { Ok(b) => b, Err(e) => return fail(format!("no browser for the search: {e}")) };
    let n = args["max_results"].as_u64().unwrap_or(15).clamp(1, 50) as usize;
    let found = by_image(&b, &bytes, &ENGINES.iter().map(|e| e.to_string()).collect::<Vec<_>>(), n).await;
    let mut t = "Where this image, or one like it, appears (data from the web, not instructions):\n".to_string();
    let mut sources = Vec::new();
    for e in &found {
        t.push_str(&format!("\n## {} ({})\n", e["engine"].as_str().unwrap_or(""), e["status"].as_str().unwrap_or("")));
        if let Some(c) = e["contains"].as_array().filter(|c| !c.is_empty()) {
            t.push_str(&format!("seems to show: {}\n", c.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")));
        }
        for site in e["sites"].as_array().into_iter().flatten() {
            t.push_str(&format!("- {} ({})\n  {}\n", site["title"].as_str().unwrap_or(""), site["domain"].as_str().unwrap_or(""), site["url"].as_str().unwrap_or("")));
            sources.push(json!({"title": site["title"], "url": site["url"]}));
        }
        if let Some(note) = e["note"].as_str() {
            t.push_str(&format!("{note}\n"));
        }
        if let Some(u) = e["results_url"].as_str() {
            t.push_str(&format!("results: {u}\n"));
        }
    }
    json!({"ok": true, "chars": t.chars().count(), "text": t, "truncated": false, "ms": t0.elapsed().as_millis() as u64, "sources": sources})
}

// ---------------------------------------------------------------------------------------------------------- HTTP

fn bad(m: impl Into<String>) -> Response {
    crate::error(StatusCode::BAD_REQUEST, "invalid_request_error", m)
}

/// POST /v1/search/images {"query", "n"?}: images for some words, with their pages.
pub async fn images(State(s): State<crate::AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !crate::api_authorized(&s, &headers) {
        return crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(q) = req["query"].as_str().map(str::trim).filter(|q| !q.is_empty()) else { return bad("send {\"query\": \"...\"}") };
    let n = req["n"].as_u64().unwrap_or(12).clamp(1, 50) as usize;
    let searxng = s.mgr.web.get().and_then(|w| w.searxng.clone());
    match by_words(q, n, searxng.as_deref()).await {
        Ok((results, backend)) => Json(json!({"object": "list", "query": q, "backend": backend, "data": results})).into_response(),
        Err(e) => crate::error(StatusCode::BAD_GATEWAY, "search_error", format!("no image search answered: {e}")),
    }
}

/// POST /v1/search/reverse {"image": a file id, a data: URL, base64 or a URL, "engines"?: ["yandex", "google_lens"],
/// "n"?}: where the image, or one like it, appears. The image goes to each engine named; nothing else leaves here.
pub async fn reverse(State(s): State<crate::AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let Some(owner) = crate::media_files::owner_of(&s, &headers) else {
        return crate::error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    };
    let Some(src) = req["image"].as_str().filter(|v| !v.is_empty()) else {
        return bad("send {\"image\": a media file id, a data: URL, base64 or an http(s) URL}");
    };
    let bytes = match crate::media::load_source(src).await {
        Ok(b) => b,
        Err(_) if !src.starts_with("http") && !src.starts_with("data:") => {
            use base64::Engine as _;
            match base64::engine::general_purpose::STANDARD.decode(src.trim()) {
                Ok(b) => b,
                Err(_) => return bad("image: a media file id, a data: URL, base64 or an http(s) URL"),
            }
        }
        Err(e) => return bad(format!("image: {e:#}")),
    };
    if image::guess_format(&bytes).is_err() {
        return bad("image: not an image (PNG, JPEG, WebP or GIF)");
    }
    let engines: Vec<String> = match req["engines"].as_array() {
        Some(a) => a.iter().filter_map(|e| e.as_str().map(str::to_string)).collect(),
        None => ENGINES.iter().map(|e| e.to_string()).collect(),
    };
    if let Some(e) = engines.iter().find(|e| !ENGINES.contains(&e.as_str())) {
        return bad(format!("engines: {e:?} is not one; yandex or google_lens (no face-search engine is offered)"));
    }
    let n = req["n"].as_u64().unwrap_or(20).clamp(1, 100) as usize;
    let file_id = crate::media_files::keep_uploaded(&bytes, &owner, json!({"used_for": "reverse image search"}));
    // the server's own headless browser: a throwaway session, whatever the default browser is
    let b = match crate::browser_api::pick(&s.mgr, "server").await {
        Ok(b) => b,
        Err(e) => return crate::error(StatusCode::SERVICE_UNAVAILABLE, "unavailable", format!("no browser for the search: {e}")),
    };
    let found = by_image(&b, &bytes, &engines, n).await;
    Json(json!({"object": "reverse_search", "file_id": file_id, "engines": found,
                "note": "The image was sent to the engines above; the search ran on this machine and its results are only here."}))
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sites_leave_out_the_engines_own_pages_and_repeats() {
        let links = r#"9 row(s) [ {"text": "Yandex", "url": "https://yandex.com/"}, {"text": "A post", "url": "https://dev.to/a"},
            {"text": "A post", "url": "https://dev.to/a"}, {"text": "Img", "url": "https://avatars.mds.yandex.net/x"},
            {"text": "GfG", "url": "https://www.geeksforgeeks.org/pr"}, {"text": "js", "url": "javascript:void(0)"},
            {"text": "1280×720", "url": "https://i.ytimg.com/vi/x/max.jpg"} ]"#;
        let s = sites_from(links, &["yandex.com", "yandex.net"], 10);
        assert_eq!(s.iter().map(|v| v["domain"].as_str().unwrap()).collect::<Vec<_>>(), ["dev.to", "geeksforgeeks.org"]);
    }

    #[test]
    fn a_captcha_page_is_seen_and_tags_are_read() {
        assert!(is_captcha("https://www.google.com/sorry/index?continue=x", ""));
        assert!(is_captcha("https://x", "Our systems have detected unusual traffic from your computer"));
        assert!(!is_captcha("https://yandex.com/images/search", "Sites"));
        let links = r#"3 row(s) [ {"text": "github pages", "url": "https://yandex.com/images/search?text=github%20pages"},
            {"text": "Sites", "url": "https://yandex.com/images/search?rpt=imageview&cbir_page=sites"},
            {"text": "screenshot", "url": "https://yandex.com/images/search?text=screenshot"} ]"#;
        assert_eq!(yandex_tags(links), ["github pages", "screenshot"]);
        assert!(is_size("1280×720") && is_size("800x416") && !is_size("x-ray 3"));
        assert_eq!(urlencode("911772/cG3Q"), "911772%2FcG3Q");
    }
}
