//! POST /v1/audio/music: a song with vocals from a style and lyrics (ACE-Step 1.5, in the media engine, in Rust).
//! OpenAI has no such endpoint; the fields follow /v1/audio/speech's spirit:
//! {"style" (or "caption" / "prompt"), "lyrics", "language", "duration" (s, 10 to 600), "seed", "bpm", "key",
//!  "time_signature", "steps", "planner" (default true: ACE-Step's song planner sets the song's structure first),
//!  "response_format": "wav"} -> audio/wav (48 kHz stereo).
//! A GPU job like an image: the text model stops for it and loads again with the next chat.

use crate::{error, AppState};
use axum::{body::Body, extract::State, http::{header, HeaderMap, StatusCode}, response::Response, Json};
use serde_json::{json, Value};

pub const MODEL: &str = "ace-step-1.5-turbo";

/// Genres by name, each a style ACE-Step understands (instruments, voice, mood). `genre` picks one; `style` adds to it
/// or replaces it. Indian genres first, then the world's.
pub const GENRES: &[(&str, &str, &str)] = &[
    ("bollywood", "Bollywood", "bollywood pop, tabla, dholak, sitar and synth, catchy hook, cheerful"),
    ("festive", "Festive (Diwali, Holi)", "festive indian celebration song, dhol, shehnai, tabla and claps, joyful, energetic"),
    ("bhajan", "Bhajan (devotional)", "devotional bhajan, harmonium, tabla, manjira, peaceful, spiritual"),
    ("ghazal", "Ghazal", "ghazal, soft male vocal, harmonium, tabla and sarangi, romantic, slow"),
    ("qawwali", "Qawwali", "qawwali, chorus claps, harmonium and tabla, rising energy, sufi"),
    ("sufi", "Sufi", "sufi song, rubab, harmonium and tabla, soulful, mystical"),
    ("hindustani", "Hindustani classical", "hindustani classical vocal, tanpura drone, tabla, raga, meditative"),
    ("carnatic", "Carnatic fusion", "carnatic fusion, veena, mridangam, violin and electronic beats"),
    ("malayalam-melody", "Malayalam film melody", "malayalam film melody, violin and flute, gentle, nostalgic"),
    ("tamil-kuthu", "Tamil kuthu", "tamil kuthu, thappu drums, nadaswaram, fast, high energy, dance"),
    ("bhangra", "Punjabi bhangra", "punjabi bhangra, dhol, tumbi, energetic, danceable"),
    ("indie-pop", "Indie pop", "warm indie pop, acoustic guitar, soft drums, hopeful"),
    ("pop", "Pop", "modern pop, synths, punchy drums, catchy chorus, bright"),
    ("ballad", "Ballad", "soulful ballad, piano and strings, slow, emotional"),
    ("rock", "Rock", "energetic rock anthem, distorted guitars, big drums"),
    ("metal", "Metal", "heavy metal, fast double kick drums, distorted riffs, powerful"),
    ("hip-hop", "Hip hop", "hip hop, boom bap drums, bass, rap vocal, confident"),
    ("rnb", "R&B", "smooth r&b, soft keys, groove, sensual vocal"),
    ("jazz", "Jazz", "smooth jazz, saxophone, upright bass, brushed drums, late night"),
    ("blues", "Blues", "electric blues, slide guitar, shuffle rhythm, gritty"),
    ("country", "Country", "country, acoustic guitar, banjo, fiddle, storytelling"),
    ("folk", "Folk", "acoustic folk, guitar and harmonica, warm, storytelling"),
    ("reggae", "Reggae", "reggae, offbeat guitar skank, deep bass, relaxed"),
    ("latin", "Latin", "latin pop, guitar, congas and brass, sunny, danceable"),
    ("kpop", "K-pop", "k-pop, polished synths, dance beat, bright vocals"),
    ("edm", "EDM", "upbeat edm, synth leads, four-on-the-floor, festival drop"),
    ("house", "House", "deep house, groovy bassline, four-on-the-floor, warm pads"),
    ("lofi", "Lo-fi", "lo-fi hip hop, mellow, vinyl crackle, soft keys, relaxed"),
    ("ambient", "Ambient", "ambient, evolving pads, soft textures, calm, spacious"),
    ("cinematic", "Cinematic orchestral", "cinematic orchestral score, strings, brass, choir, epic, building"),
    ("piano", "Solo piano", "solo piano, expressive, gentle, intimate"),
    ("classical", "Western classical", "classical string quartet, elegant, flowing"),
];

/// The style for a request: the genre's, with the request's own style added (or the style alone).
pub fn style_for(genre: Option<&str>, style: Option<&str>) -> Result<String, String> {
    let g = match genre.map(|g| g.trim().to_lowercase()).filter(|g| !g.is_empty()) {
        Some(g) => match GENRES.iter().find(|x| x.0 == g || x.1.to_lowercase() == g) {
            Some(x) => Some(x.2),
            None => return Err(format!("genre {g:?}: one of {}", GENRES.iter().map(|x| x.0).collect::<Vec<_>>().join(", "))),
        },
        None => None,
    };
    match (g, style.map(str::trim).filter(|s| !s.is_empty())) {
        (Some(g), Some(s)) => Ok(format!("{g}, {s}")),
        (Some(g), None) => Ok(g.to_string()),
        (None, Some(s)) => Ok(s.to_string()),
        (None, None) => Err("style or genre: describe the song (genre, instruments, voice, mood), or name a genre".into()),
    }
}

/// GET /api/media/genres: the genres by name with their styles.
pub async fn genres(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !crate::api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    use axum::response::IntoResponse;
    Json(json!({"genres": GENRES.iter().map(|g| json!({"id": g.0, "name": g.1, "style": g.2})).collect::<Vec<_>>()})).into_response()
}

/// How much of `lyrics` a transcript holds: the words they share in order over the lyrics' words (0 to 1). Structure
/// tags ([verse]) do not count.
pub fn lyrics_heard(lyrics: &str, heard: &str) -> f64 {
    let words = |t: &str| -> Vec<String> {
        let mut out = String::new();
        let mut depth = 0;
        for c in t.to_lowercase().chars() {
            match c { '[' => depth += 1, ']' => depth -= 1, _ if depth > 0 => {}, c if c.is_alphanumeric() => out.push(c), _ => out.push(' ') }
        }
        out.split_whitespace().map(str::to_string).collect()
    };
    let (a, b) = (words(lyrics), words(heard));
    if a.is_empty() {
        return 1.0;
    }
    let mut prev = vec![0usize; b.len() + 1];
    for x in &a {
        let mut cur = vec![0usize; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { cur[j].max(prev[j + 1]) };
        }
        prev = cur;
    }
    prev[b.len()] as f64 / a.len() as f64
}

fn bad(msg: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

pub async fn music(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    if !crate::api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available", "the media engine is not built");
    }
    let text = |k: &str| req[k].as_str().map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
    let style = match style_for(req["genre"].as_str(), text("style").or_else(|| text("caption")).or_else(|| text("prompt")).as_deref()) {
        Ok(s) => s,
        Err(e) => return bad(e),
    };
    if style.chars().count() > 512 {
        return bad("style: at most 512 characters");
    }
    let lyrics = text("lyrics").unwrap_or_else(|| "[Instrumental]".into());
    if lyrics.chars().count() > 4096 {
        return bad("lyrics: at most 4096 characters");
    }
    let duration = req["duration"].as_u64().unwrap_or(60);
    if !(10..=600).contains(&duration) {
        return bad("duration: 10 to 600 seconds (10 minutes)");
    }
    if let Some(f) = req["response_format"].as_str().filter(|f| *f != "wav") {
        return bad(format!("response_format {f:?}: this server makes wav only"));
    }
    let list = match s.media.list().await {
        Ok(l) => l,
        Err(e) => return error(StatusCode::BAD_GATEWAY, "media_error", format!("{e:#}")),
    };
    let Some(m) = list["models"].as_array().and_then(|a| a.iter().find(|m| m["name"] == MODEL).cloned()) else {
        return bad(format!("{MODEL} is not in this media engine's catalog: update the media engine"));
    };
    if m["downloaded"] != true {
        return bad(format!("{MODEL} is not downloaded: download it in Settings > Models > Media models > Music ({} GB)", m["size_gb"]));
    }
    let language = text("language").map(|l| l.to_lowercase())
        .unwrap_or_else(|| crate::langdetect::detect(&lyrics).to_string());
    let seed = req["seed"].as_u64().unwrap_or_else(|| (crate::now() ^ (std::process::id() as u64) << 20) % 4_294_967_295);
    let steps = req["steps"].as_u64().unwrap_or(8).clamp(1, 20);
    let mut cmd = json!({"op": "song", "caption": style, "lyrics": lyrics, "language": language, "duration": duration,
                         "seed": seed, "steps": steps, "device": "cuda"});
    // "planner": false makes the song without ACE-Step's song planner (faster, lyrics followed less often);
    // its sampling can be set with planner_temperature, planner_cfg and planner_top_p
    // a continuation of the song before (LONG-MEDIA-PLAN: ACE-Step's repaint past the source's end): "continue_from"
    // is a WAV (base64) of that song's end; the answer starts x-strata-replaces-s seconds before that end
    for k in ["bpm", "key", "time_signature", "planner", "planner_temperature", "planner_cfg", "planner_top_p",
              "continue_from", "context_seconds", "timbre_from_source"] {
        if !req[k].is_null() {
            cmd[k] = req[k].clone();
        }
    }
    let t0 = std::time::Instant::now();
    let label = s.media_jobs.label(&req["strata_label"]);
    let _gpu = match s.mgr.gpu_exclusive_as(!s.media.keep_text_model, "media", &label).await {
        Ok(g) => g,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, "busy", format!("{e:#}")),
    };
    let mon = s.mgr.monitor.clone();
    let job = mon.media_begin("song", MODEL);
    let _row = crate::monitor::MediaRow::new(&mon, job);
    // takes: a song is made again with other seeds until the speech-to-text model hears enough of its lyrics (even
    // with the song planner, sung words vary with the seed), the best take kept
    let instrumental = lyrics.trim().eq_ignore_ascii_case("[instrumental]");
    let takes = if instrumental { 1 } else { req["takes"].as_u64().unwrap_or(1).clamp(1, 4) };
    let stt = s.media.stt_model();
    let mut res: Result<Value, anyhow::Error> = Err(anyhow::anyhow!("no take"));
    let mut best = -1.0f64;
    let mut heard_note = String::new();
    for t in 0..takes {
        let mut c = cmd.clone();
        c["seed"] = json!(seed.wrapping_add(t * 7919) % 4_294_967_295);
        let r = s.media.call(c.clone(), |ev| {
            if ev["event"] == "step" {
                mon.media_step(job, ev["step"].as_u64().unwrap_or(0) + t * steps, steps * takes);
            }
        }).await;
        let Ok(v) = r else {
            if t == 0 { res = r; }
            break;
        };
        if takes == 1 {
            res = Ok(v);
            break;
        }
        let heard = s.media.call(json!({"op": "transcribe", "model": stt, "audio": v["data"], "language": language,
                                         "translate": false, "device": "cpu"}), |_| {}).await;
        let score = heard.map(|h| lyrics_heard(&lyrics, h["text"].as_str().unwrap_or(""))).unwrap_or(0.0);
        if score > best {
            best = score;
            heard_note = format!(", {:.0}% of the lyrics heard (take {} of {takes})", score * 100.0, t + 1);
            res = Ok(v);
        }
        if score >= 0.75 {
            break;
        }
    }
    let _ = s.media.call(json!({"op": "unload_audio"}), |_| {}).await;
    if let Err(e) = s.media.call(json!({"op": "unload"}), |_| {}).await {
        tracing::warn!("the media engine did not unload: {e:#}");
    }
    let v = match res {
        Ok(v) => v,
        Err(e) => {
            mon.media_end(job, "error", &format!("{e:#}"));
            return error(StatusCode::BAD_GATEWAY, "music_error", format!("{e:#}"));
        }
    };
    let wav = {
        use base64::Engine as _;
        match base64::engine::general_purpose::STANDARD.decode(v["data"].as_str().unwrap_or("")) {
            Ok(b) => b,
            Err(_) => return error(StatusCode::BAD_GATEWAY, "music_error", "the media engine returned no audio"),
        }
    };
    mon.media_file_add(job, "wav", wav.clone());
    let seed = v["seed"].as_u64().unwrap_or(seed);
    mon.media_end(job, "stop", &format!("{duration} s {}, {language}, seed {seed}{heard_note}", if instrumental { "music" } else { "song" }));
    Response::builder().header(header::CONTENT_TYPE, "audio/wav")
        .header("x-strata-seconds", duration.to_string()).header("x-strata-model", MODEL)
        .header("x-strata-language", language).header("x-strata-seed", seed.to_string())
        .header("x-strata-ms", t0.elapsed().as_millis().to_string())
        .header("x-strata-lyrics-heard", if best >= 0.0 { format!("{best:.2}") } else { "-".into() })
        // the planner's tempo, key and time signature, for a song asked without them
        .header("x-strata-planned", v["planned"].to_string())
        .header("x-strata-replaces-s", v["replaces_s"].as_f64().map(|r| r.to_string()).unwrap_or_else(|| "0".into()))
        .body(Body::from(wav)).unwrap()
}
