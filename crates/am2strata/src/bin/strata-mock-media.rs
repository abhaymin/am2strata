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
    emit(serde_json::json!({"event": "ready", "cuda": false, "version": "mock"}));
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
                if r["prompt"].as_str().unwrap_or("").contains("FAIL") {
                    emit(serde_json::json!({"event": "error", "message": "mock failure"}));
                    continue;
                }
                let steps = r["steps"].as_u64().unwrap_or(8);
                for s in 1..=steps {
                    emit(serde_json::json!({"event": "step", "step": s, "total": steps}));
                }
                emit(serde_json::json!({"event": "image", "index": 0, "width": r["width"], "height": r["height"], "png": PNG}));
                emit(serde_json::json!({"event": "done", "ms": 5, "seed": r["seed"]}));
            }
            "unload" => emit(serde_json::json!({"event": "unloaded"})),
            "load" => emit(serde_json::json!({"event": "loaded", "model": cmd["model"]})),
            "pull" => emit(serde_json::json!({"event": "pulled", "model": cmd["model"]})),
            _ => emit(serde_json::json!({"event": "error", "message": "unknown op"})),
        }
    }
}
