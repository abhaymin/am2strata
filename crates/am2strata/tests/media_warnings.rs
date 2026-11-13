//! PR #2 compose_video review warnings (round 2), tested end to end against the real server:
//!  - warning 3: submit_compose refuses a missing/empty audio with a clear 400;
//!  - warning 4: submit_thumbnail rejects an empty image and unusable lines up front instead of only null;
//!  - warning 2: a limited key queues one compose (or thumbnail) job, then is refused by media_jobs_per_hour.

mod common;

use common::*;
use serde_json::{json, Value};
use std::time::Duration;

/// POST `path` for the bearer token `key` (the shared helpers only ever use the main KEY).
fn post_as(base: &str, key: &str, path: &str, body: &Value) -> (u16, Value) {
    let r = reqwest::blocking::Client::new()
        .post(format!("{base}{path}"))
        .bearer_auth(key)
        .json(body)
        .timeout(Duration::from_secs(60))
        .send()
        .expect("request");
    (r.status().as_u16(), r.json().unwrap_or(Value::Null))
}

/// A client profile named "limited", whose key may queue just one media job an hour.
fn limited_key_model(extra: Value) -> Model<'static> {
    let mut c = json!({"clients": {"limited": {"key": "limit-key-0123456789abcd",
        "limits": {"media_jobs_per_hour": 1}}}});
    if let (Some(o), Some(e)) = (c.as_object_mut(), extra.as_object()) {
        o.extend(e.clone());
    }
    Model { id: "mock-a", vision: true, arch: None, context: 8192, extra: c }
}

#[test]
fn compose_without_audio_is_refused_with_a_clear_400() {
    let s = Server::start(&[Model::default()], &[], json!({}));
    // shots present but audio missing: nothing to compose to, refused up front rather than queued and later failed
    let (c, r) = post_as(&s.base, KEY, "/v1/media/videos/compose", &json!({"shots": [{"image": "mj-1"}]}));
    assert_eq!(c, 400, "{r}");
    assert!(r["error"]["message"].as_str().unwrap_or("").contains("audio"), "clear about audio: {r}");
    // an empty audio string is refused the same way
    let (c, r) = post_as(&s.base, KEY, "/v1/media/videos/compose", &json!({"audio": "", "shots": [{"image": "mj-1"}]}));
    assert_eq!(c, 400, "{r}");
    assert!(r["error"]["message"].as_str().unwrap_or("").contains("audio"), "{r}");
}

#[test]
fn thumbnail_rejects_bad_image_and_missing_lines_up_front() {
    let s = Server::start(&[Model::default()], &[], json!({}));
    // an empty image (not just null) is refused like a missing one
    let (c, r) = post_as(&s.base, KEY, "/v1/media/thumbnails", &json!({"image": "", "lines": [["Hi", "white"]]}));
    assert_eq!(c, 400, "{r}");
    assert!(r["error"]["message"].as_str().unwrap_or("").contains("image"), "{r}");
    // an image absent entirely (the old check only caught null) is refused too
    let (c, r) = post_as(&s.base, KEY, "/v1/media/thumbnails", &json!({"lines": [["Hi", "white"]]}));
    assert_eq!(c, 400, "{r}");
    assert!(r["error"]["message"].as_str().unwrap_or("").contains("image"), "{r}");
    // a present image but only empty word rows is refused like missing lines
    let (c, r) = post_as(&s.base, KEY, "/v1/media/thumbnails", &json!({"image": "x", "lines": [["   ", "white"]]}));
    assert_eq!(c, 400, "{r}");
    assert!(r["error"]["message"].as_str().unwrap_or("").contains("lines"), "{r}");
    // no lines at all is refused as well
    let (c, r) = post_as(&s.base, KEY, "/v1/media/thumbnails", &json!({"image": "x"}));
    assert_eq!(c, 400, "{r}");
    assert!(r["error"]["message"].as_str().unwrap_or("").contains("lines"), "{r}");
}

#[test]
fn a_limited_key_queues_one_then_hits_the_hourly_cap() {
    let s = Server::start(&[limited_key_model(json!({}))], &[], json!({}));
    let compose = json!({"audio": "mj-0", "shots": [{"image": "mj-1"}]});
    // the key queues its first compose job; the second (same hour) is refused by media_jobs_per_hour
    let (c1, r1) = post_as(&s.base, "limit-key-0123456789abcd", "/v1/media/videos/compose", &compose);
    assert_eq!(c1, 200, "the first compose is queued: {r1}");
    let (c2, r2) = post_as(&s.base, "limit-key-0123456789abcd", "/v1/media/videos/compose", &compose);
    assert_eq!(c2, 400, "the second is refused by the limit: {r2}");
    assert!(r2["error"]["message"].as_str().unwrap_or("").contains("this hour"), "about the hourly cap: {r2}");

    // a thumbnail from the same exhausted key is refused the same way
    let thumb = json!({"image": "x", "lines": [["Hi", "white"]]});
    let (c3, r3) = post_as(&s.base, "limit-key-0123456789abcd", "/v1/media/thumbnails", &thumb);
    assert_eq!(c3, 400, "the thumbnail is also refused by the limit: {r3}");
    assert!(r3["error"]["message"].as_str().unwrap_or("").contains("this hour"), "{r3}");

    // a different key has no such limit and queues freely (no cross-talk between keys)
    let (c4, r4) = post_as(&s.base, KEY, "/v1/media/videos/compose", &compose);
    assert_eq!(c4, 200, "the unlimited main key is unaffected: {r4}");
}
