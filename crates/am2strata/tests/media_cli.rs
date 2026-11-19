//! `am2strata-run media <action>` against a running server plus strata-mock-media (docs/GAPS-PLAN B, docs/MEDIA.md).
//! Every action the CLI wraps goes through the real HTTP surface: image writes a PNG, speak and song write WAVs (RIFF),
//! transcribe prints its text, check exits 0 for a good take and non-zero for a clipped one, an unknown job id is
//! refused with the server's message, and answering for a file without --out is refused. All of it against one server
//! so the CLI is exercised exactly as a user runs it; the mock engine records every command to $MOCK_MEDIA_RECORD.

mod common;

use common::*;
use serde_json::json;

fn run(server: &Server, args: &[&str]) -> (bool, String, String) {
    // `media`'s value is the last token and follows `--` (clap `first = true`), so every option comes before it;
    // the global --url/--api-key go first and are not per-action.
    let mut full: Vec<String> = vec![
        "media".into(), "--models-dir".into(), server.dir.to_string_lossy().into_owned(),
        "--url".into(), server.base.clone(), "--api-key".into(), KEY.into(),
    ];
    for a in args { full.push(a.to_string()); }
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_am2strata-run")).args(&full).output().expect("am2strata-run media");
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

/// A 16-bit mono PCM WAV from [-1, 1] samples — enough for the check's pure-Rust reader.
fn wav_16(rate: u32, flat: &[f32]) -> Vec<u8> {
    let mut data = Vec::new();
    for &s in flat { data.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()); }
    let mut out = Vec::new();
    out.extend(b"RIFF");
    out.extend_from_slice(&((44 + data.len()) as u32).to_le_bytes());
    out.extend(b"WAVE");
    out.extend(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2u32).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend(data);
    out
}

/// A continuous sine; the first N frames voiced at `amp`, then silence so it ends quiet.
fn tone_then_silence(rate: u32, voiced: usize, freq: f32, amp: f32, silent: usize) -> Vec<f32> {
    let mut s = Vec::with_capacity(voiced + silent);
    for i in 0..voiced { s.push(amp * (2.0 * std::f32::consts::PI * freq * (i as f32 / rate as f32)).sin()); }
    s.extend(std::iter::repeat(0f32).take(silent));
    s
}

#[test]
fn media_cli_writes_and_reports_like_the_server() {
    let s = Server::start(&[Model::default()], &[], json!({}));
    let dir = std::env::temp_dir().join(format!("am2strata-media-cli-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).unwrap();

    // image --out writes a PNG (its signature) and prints where it went.
    let png = dir.join("shot.png");
    let (ok, out, err) = run(&s, &["image", "--out", png.to_str().unwrap(), "--", "a red dot on blue"]);
    assert!(ok && !out.is_empty(), "stdout was {out:?}, stderr {err:?}");
    assert_eq!(&std::fs::read(&png).unwrap()[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], "PNG signature");
    assert!(out.contains(png.to_str().unwrap()), "says the saved path: {out}");

    // speak --out writes a WAV (RIFF....WAVE) — the mock's valid short take.
    let wav = dir.join("speak.wav");
    let (ok, out, err) = run(&s, &["speak", "--out", wav.to_str().unwrap(), "--", "hello there and goodbye"]);
    assert!(ok && !out.is_empty(), "stdout was {out:?}, stderr {err:?}");
    assert_eq!(&std::fs::read(&wav).unwrap()[0..4], b"RIFF");
    assert_eq!(&std::fs::read(&wav).unwrap()[8..12], b"WAVE");
    assert!(out.contains(wav.to_str().unwrap()), "says the saved path: {out}");

    // song --out writes a WAV too (the mock makes it on request; ace-step is downloaded in the catalog).
    let song = dir.join("song.wav");
    let (ok, out, err) = run(&s, &["song", "--out", song.to_str().unwrap(), "--", "a gentle ballad in A minor"]);
    assert!(ok && !out.is_empty(), "stdout was {out:?}, stderr {err:?}");
    assert_eq!(&std::fs::read(&song).unwrap()[0..4], b"RIFF");
    assert!(out.contains(song.to_str().unwrap()), "says the saved path: {out}");

    // transcribe prints its text; whisper-base is downloaded in the mock catalog (whisper-1 maps to a model that is not).
    let input = dir.join("take.wav");
    std::fs::write(&input, wav_16(8_000, &[0.2f32; 400])).unwrap();
    let (ok, out, err) = run(&s, &["transcribe", "--model", "whisper-base", "--", input.to_str().unwrap()]);
    assert!(ok && !out.is_empty(), "stdout was {out:?}, stderr {err:?}");
    assert_eq!(out.trim(), "Hello world.", "the transcript on stdout: {out:?}");

    // check exits 0 for a good take (voiced ... quiet end, not clipped) and prints pass; the clipped one fails.
    let good = dir.join("good.wav");
    std::fs::write(&good, wav_16(16_000, &tone_then_silence(16_000, 8_000, 220.0, 0.3, 8_000))).unwrap();
    let (ok, out, err) = run(&s, &["check", "--", good.to_str().unwrap()]);
    assert!(ok && !out.is_empty(), "a good take passes -> stdout was {out:?}, stderr {err:?}");
    assert!(out.contains("pass") && !out.contains("FAIL"), "each finding printed as pass: {out}");

    let bad = dir.join("bad.wav");
    let clipped: Vec<f32> = (0..400).map(|i| if i % 2 == 0 { 1.3 } else { -1.3 }).collect();
    std::fs::write(&bad, wav_16(8_000, &clipped)).unwrap();
    let (ok, out, err) = run(&s, &["check", "--", bad.to_str().unwrap()]);
    assert!(!ok, "a clipped take exits non-zero: {err}");
    assert!(err.contains("not ok") || out.contains("FAIL"), "reports the failing finding: {out} {err}");

    // analyze reports the music facts of a take over HTTP: a sustained A3 (220 Hz) reads as its tonic, and it exits 0.
    let chant = dir.join("chant.wav");
    std::fs::write(&chant, wav_16(8_000, &tone_then_silence(8_000, 16_000, 220.0, 0.3, 8_000))).unwrap();
    let (ok, out, err) = run(&s, &["analyze", "--", chant.to_str().unwrap()]);
    assert!(ok && !out.is_empty(), "an analyze of a take prints music facts: stdout {out:?}, stderr {err:?}");
    assert!(out.contains("A3"), "the sustained A3 (220 Hz) is reported as the tonic: {out}");

    // an unknown job id is refused with the server's message (it names the id) and a non-zero exit.
    let (ok, _, err) = run(&s, &["job", "--", "mj-999"]);
    assert!(!ok && !err.is_empty(), "unknown job exits non-zero: {err}");
    assert!(err.contains("mj-999"), "the server's error says the id: {err}");

    // a file answer given without --out is refused before anything is sent.
    let (ok, out, err) = run(&s, &["image", "--", "a dot"]);
    assert!(!ok && !err.is_empty(), "missing --out is refused: {out} {err}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// The endpoint forwards any audio to the media engine's `analyze` op and returns its analysis without the `event` field, so a
/// client calling /v1/media/analyze gets exactly what the engine reports. `from_s`/`seconds` reach the engine (recorded on the
/// engine's own command line) and come back echoed, proving a window was applied; it reads an MP3 too, which WAV-only would not.
#[test]
fn analyze_endpoint_forwards_to_the_engine_and_carries_the_window() {
    let s = Server::start(&[Model::default()], &[], json!({}));

    // POST to the endpoint with a take (here a WAV base64 data URL; an MP3 works the same way it never did when WAV-only) and a
    // window, exactly like /v1/media/analyze's documented body {"audio", "from_s", "seconds"}.
    let chant = wav_16(8_000, &tone_then_silence(8_000, 16_000, 220.0, 0.3, 8_000));
    use base64::Engine as _;
    let data_url = format!("data:audio/wav;base64,{}", base64::engine::general_purpose::STANDARD.encode(&chant));
    let (code, j) = s.post("/v1/media/analyze", &json!({"audio": data_url, "from_s": 30, "seconds": 60}));

    assert_eq!(code, 200, "the analyze endpoint answers: {j}");
    assert!(j.get("event").is_none(), "the analysis comes back without its event field: {j}");
    // the window reached the engine and came back echoed ...
    assert_eq!(j["from_s"].as_f64(), Some(30.0), "from_s echoed: {j}");
    assert_eq!(j["seconds"].as_f64(), Some(60.0), "seconds echoed: {j}");
    // ... recorded on the engine's own command line, so it is not only the mock being asked to lie about it
    let rec = std::fs::read_to_string(s.dir.join("media-record.txt")).unwrap_or_default();
    assert!(rec.lines().any(|l| l.contains("\"op\":\"analyze\"") && l.contains("\"from_s\":30") && l.contains("\"seconds\":60")),
            "the analyze command carried from_s/seconds to the engine: {rec}");

    // no key is refused, as with /v1/media/check
    let no_key = reqwest::blocking::Client::new().post(format!("{}/v1/media/analyze", s.base)).json(&json!({"audio": "AAAA"})).send().unwrap();
    assert_eq!(no_key.status().as_u16(), 401, "a request without a key is unauthorized");
}
