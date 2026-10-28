//! A stand-in for am2strata-media in the tests: the same JSON-lines protocol, a 1x1 PNG per image, every command
//! recorded to $MOCK_MEDIA_RECORD. A prompt containing FAIL answers with an error.

use std::io::{BufRead, Write};

const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

fn emit(v: serde_json::Value) {
    let mut o = std::io::stdout().lock();
    let _ = writeln!(o, "{v}");
    let _ = o.flush();
}

fn main() {
    let record = std::env::var("MOCK_MEDIA_RECORD").ok();
    emit(serde_json::json!({"event": "ready", "cuda": false, "version": "mock", "video_formats": ["mp4", "apng", "gif"]}));
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        if let Some(r) = &record {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(r) {
                let _ = writeln!(f, "{line}");
            }
        }
        let cmd: serde_json::Value = serde_json::from_str(&line).unwrap_or_default();
        match cmd["op"].as_str().unwrap_or("") {
            "generate" => {
                let r = &cmd["request"];
                // a slow job, for the queue's tests
                if let Some(ms) = std::env::var("MOCK_MEDIA_DELAY_MS").ok().and_then(|v| v.parse::<u64>().ok()) {
                    std::thread::sleep(std::time::Duration::from_millis(ms));
                }
                if r["prompt"].as_str().unwrap_or("").contains("FAIL") {
                    emit(serde_json::json!({"event": "error", "message": "mock failure"}));
                    continue;
                }
                let steps = r["steps"].as_u64().unwrap_or(8);
                for s in 1..=steps {
                    emit(serde_json::json!({"event": "step", "step": s, "total": steps}));
                }
                if r["frames"].is_u64() {
                    emit(serde_json::json!({"event": "video", "format": r["output_format"], "width": r["width"], "height": r["height"],
                                            "frames": r["frames"], "fps": r["fps"], "data": "AAAAIGZ0eXBpc29t"}));
                } else {
                    emit(serde_json::json!({"event": "image", "index": 0, "width": r["width"], "height": r["height"], "png": PNG}));
                }
                emit(serde_json::json!({"event": "done", "ms": 5, "seed": r["seed"]}));
            }
            "transcribe" => {
                let n = cmd["audio"].as_str().unwrap_or("").len();
                let text = if cmd["translate"] == true { "Hello in English." } else { "Hello world." };
                emit(serde_json::json!({"event": "segment", "id": 0, "start": 0.0, "end": 1.5, "text": text}));
                emit(serde_json::json!({"event": "transcript", "text": text, "language": cmd["language"].as_str().unwrap_or("en"),
                    "duration": 1.5, "task": if cmd["translate"] == true { "translate" } else { "transcribe" }, "bytes_b64": n,
                    "device": cmd["device"], "segments": [{"id": 0, "start": 0.0, "end": 1.5, "text": text}]}));
            }
            "speak" => {
                emit(serde_json::json!({"event": "part", "done": 1, "total": 1}));
                // a valid 1-sample WAV
                emit(serde_json::json!({"event": "speech", "format": "wav", "sample_rate": 44100, "seconds": 0.5,
                    "data": "UklGRiYAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQIAAAAAAA=="}));
            }
            "unload" | "unload_audio" => emit(serde_json::json!({"event": "unloaded"})),
            "load" => emit(serde_json::json!({"event": "loaded", "model": cmd["model"]})),
            "pull" => emit(serde_json::json!({"event": "pulled", "model": cmd["model"]})),
            "list" => emit(serde_json::json!({"event": "models", "models": [
                {"name": "z-image-turbo:q8", "kind": "image", "family": "z-image", "size_gb": 13.8, "downloaded": true, "gated": false},
                {"name": "flux2-klein:q4", "kind": "image", "family": "flux2", "size_gb": 10.1, "downloaded": false, "gated": false},
                {"name": "whisper-base", "kind": "audio", "task": "stt", "family": "whisper", "size_gb": 0.29, "downloaded": true, "gated": false},
                {"name": "parler-tts-mini-v1", "kind": "audio", "task": "tts", "family": "parler-tts", "size_gb": 3.51, "downloaded": true, "gated": false},
                {"name": "whisper-large-v3-turbo", "kind": "audio", "task": "stt", "family": "whisper", "size_gb": 1.62, "downloaded": false, "gated": false},
                {"name": "wan22-ti2v-5b:q8", "kind": "video", "family": "wan", "size_gb": 16.9, "downloaded": true, "gated": false,
                 "defaults": {"steps": 30, "guidance": 5.0, "width": 1280, "height": 704, "frames": 121, "fps": 24}}]})),
            _ => emit(serde_json::json!({"event": "error", "message": "unknown op"})),
        }
    }
}
