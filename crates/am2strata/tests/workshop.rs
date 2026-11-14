//! The Workshop's server side end to end against the mock engine, with a stand-in for the `am2strata-workshop`
//! command (the real conversions are tested in strata-workshop against llama.cpp's own output): jobs queue and run one
//! at a time, report their output or their error, can be cancelled, and cannot touch files outside the Workshop's
//! folders. Only the server's own key may use it.

mod common;

use common::*;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

/// A stand-in for am2strata-workshop: prints the same JSON events. An input whose name holds "slow" takes 30 s (for
/// cancel), "fail" fails, anything else "converts" by writing the output file.
fn fake_workshop() -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("fake-workshop-{}.sh", std::process::id()));
    std::fs::write(&p, r#"#!/bin/sh
# --json <kind> <input> ... --out <output> ...
shift; kind="$1"; input="$2"; out=""
while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done
echo "{\"event\":\"start\",\"what\":\"$kind $input\"}"
case "$input" in
  *slow*) sleep 30 ;;
  *fail*) echo '{"event":"error","message":"this model is not a Llama"}'; exit 1 ;;
esac
mkdir -p "$(dirname "$out")" 2>/dev/null; echo data > "$out"
echo "{\"event\":\"log\",\"message\":\"12 tensors\"}"
echo "{\"event\":\"done\",\"output\":\"$out\",\"kind\":\"gguf\",\"bytes\":5,\"seconds\":0.1}"
"#).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    p
}

fn wait_state(s: &Server, id: u64, want: &str) -> Value {
    let t0 = Instant::now();
    loop {
        let (_, j) = s.get(&format!("/v1/workshop/jobs/{id}"), true);
        if j["state"] == want {
            return j;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "job {id} is {} not {want}: {j}", j["state"]);
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn jobs_run_one_at_a_time_report_cancel_and_stay_inside_their_folders() {
    // the server inherits this (the stand-in command), set before it starts
    unsafe { std::env::set_var("AM2STRATA_WORKSHOP_EXE", fake_workshop()) };
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    // an HF model folder inside the models folder
    let model = s.dir.join("hf-model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    std::fs::write(model.join("model.safetensors"), "x").unwrap();

    assert_eq!(s.get("/v1/workshop/jobs", false).0, 401, "needs the key");
    let (code, f) = s.get("/v1/workshop/files", true);
    assert_eq!(code, 200);
    assert!(f["hf_models"].as_array().unwrap().iter().any(|p| p.as_str().unwrap().ends_with("hf-model")), "{f}");

    // a convert job runs and reports its output, in the Workshop folder by default
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let out = done["result"]["output"].as_str().unwrap();
    assert!(out.contains("/workshop/") && out.ends_with("hf-model-f16.gguf"), "{out}");
    assert!(std::path::Path::new(out).exists());
    assert!(done["log"].as_array().unwrap().iter().any(|l| l == "12 tensors"));

    // a failing job says why
    let bad = s.dir.join("fail-model");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(bad.join("config.json"), "{}").unwrap();
    let (_, j) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": bad}));
    let failed = wait_state(&s, j["id"].as_u64().unwrap(), "failed");
    assert_eq!(failed["error"], "this model is not a Llama");

    // one at a time: a slow job runs, the next waits queued; cancelling the slow one lets the next run
    let slow = s.dir.join("slow-model");
    std::fs::create_dir_all(&slow).unwrap();
    std::fs::write(slow.join("config.json"), "{}").unwrap();
    let (_, a) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": slow}));
    let (_, b) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model, "outtype": "f32"}));
    let (a, b) = (a["id"].as_u64().unwrap(), b["id"].as_u64().unwrap());
    wait_state(&s, a, "running");
    assert_eq!(s.get(&format!("/v1/workshop/jobs/{b}"), true).1["state"], "queued");
    let (code, c) = s.delete(&format!("/v1/workshop/jobs/{a}"));
    assert_eq!((code, c["state"].as_str()), (200, Some("cancelled")));
    wait_state(&s, b, "done");
    assert_eq!(s.delete(&format!("/v1/workshop/jobs/{b}")).0, 409, "a finished job cannot be cancelled");

    // nothing outside the Workshop's folders, in or out
    let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": "/etc"}));
    assert_eq!(code, 400);
    assert!(e["error"]["message"].as_str().unwrap().contains("outside the Workshop"), "{e}");
    let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model, "output": "/tmp/x.gguf"}));
    assert_eq!(code, 400);
    let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model.join("../../..")}));
    assert_eq!(code, 400, "no way out with ..");
    // a symlink inside a Workshop folder that points outside it is outside
    #[cfg(unix)]
    {
        let outside = std::env::temp_dir().join(format!("am2-ws-outside-{}", std::process::id()));
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("config.json"), "{}").unwrap();
        std::os::unix::fs::symlink(&outside, s.dir.join("sneaky")).unwrap();
        let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": s.dir.join("sneaky")}));
        assert_eq!(code, 400, "{e}");
        let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model, "output": s.dir.join("sneaky/out.gguf")}));
        assert_eq!(code, 400, "an output through the link is outside too");
        let _ = std::fs::remove_dir_all(&outside);
    }
    // a refused output makes no folders, and a relative path is refused
    let nowhere = std::env::temp_dir().join(format!("am2-ws-nowhere-{}", std::process::id()));
    let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model, "output": nowhere.join("a/b/out.gguf")}));
    assert_eq!(code, 400);
    assert!(!nowhere.exists(), "no folder made outside the Workshop's folders");
    assert_eq!(s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": "hf-model"})).0, 400, "relative path");
    // and requests that make no sense are refused before anything runs
    assert_eq!(s.post("/v1/workshop/jobs", &json!({"kind": "quantize", "input": out, "type": "q3_x"})).0, 400);
    assert_eq!(s.post("/v1/workshop/jobs", &json!({"kind": "merge", "base": model, "adapter": model})).0, 400, "not an adapter");
    assert_eq!(s.post("/v1/workshop/jobs", &json!({"kind": "train"})).0, 400);
    let (_, list) = s.get("/v1/workshop/jobs", true);
    assert_eq!(list["jobs"].as_array().unwrap().len(), 4);
}

/// A small but complete llama GGUF: the metadata and BPE tokenizer an engine needs (tokens, types, merges, bos/eos).
fn small_llama_gguf(path: &std::path::Path) {
    use strata_pack::gguf_write::{Kv, TensorSpec, Writer};
    let tokens: Vec<String> = ["<s>", "</s>", "t", "o", "to", "a"].iter().map(|s| s.to_string()).collect();
    let kv: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("llama".into())), ("general.name", Kv::Str("Small Llama".into())),
        ("llama.context_length", Kv::U32(2048)), ("llama.embedding_length", Kv::U32(4)), ("llama.block_count", Kv::U32(1)),
        ("llama.feed_forward_length", Kv::U32(8)), ("llama.attention.head_count", Kv::U32(1)),
        ("llama.attention.head_count_kv", Kv::U32(1)), ("llama.rope.dimension_count", Kv::U32(4)),
        ("llama.attention.layer_norm_rms_epsilon", Kv::F32(1e-5)), ("llama.vocab_size", Kv::U32(6)),
        ("tokenizer.ggml.model", Kv::Str("gpt2".into())), ("tokenizer.ggml.pre", Kv::Str("gpt-2".into())),
        ("tokenizer.ggml.tokens", Kv::Strings(tokens)), ("tokenizer.ggml.token_type", Kv::Ints(vec![3, 3, 1, 1, 1, 1])),
        ("tokenizer.ggml.merges", Kv::Strings(vec!["t o".into()])),
        ("tokenizer.ggml.bos_token_id", Kv::U32(0)), ("tokenizer.ggml.eos_token_id", Kv::U32(1)),
    ];
    let raw: Vec<u8> = (0..24u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut w = Writer::create_typed(path, &kv, vec![TensorSpec { name: "token_embd.weight".into(), shape: vec![6, 4], ggml_type: 0, nbytes: 96 }]).unwrap();
    w.tensor_data(&raw).unwrap();
    w.end_tensor().unwrap();
    w.finish().unwrap();
}

#[test]
fn a_gguf_made_strata_ready_appears_in_the_model_list_and_is_not_overwritten() {
    // a stand-in for the Rust engine (a llama model runs on it), so the model is listed on any machine, CI included
    let home = std::env::temp_dir().join(format!("am2-ws-home-{}", std::process::id()));
    let engine = home.join("crates/strata-hybrid/target/release/am2strata-hybrid");
    std::fs::create_dir_all(engine.parent().unwrap()).unwrap();
    std::fs::write(&engine, "#!/bin/sh\n").unwrap();
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_HOME": home}}));
    // a GGUF llama.cpp itself would not load (a BPE vocabulary without merges) is refused, saying why
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../strata-workshop/tests/fixtures/tiny-llama.f32.gguf");
    let nomerges = s.dir.join("workshop/tiny-llama.f32.gguf");
    std::fs::create_dir_all(nomerges.parent().unwrap()).unwrap();
    std::fs::copy(&src, &nomerges).unwrap();
    let (_, j) = s.post("/v1/workshop/jobs", &json!({"kind": "strata", "input": nomerges}));
    let failed = wait_state(&s, j["id"].as_u64().unwrap(), "failed");
    assert!(failed["error"].as_str().unwrap().contains("merges"), "{failed}");

    let gguf = s.dir.join("workshop/small-llama.gguf");
    small_llama_gguf(&gguf);
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "strata", "input": gguf, "name": "my-tiny", "display_name": "My Tiny Llama"}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    assert_eq!(done["result"]["architecture"], "llama");
    let cfg_path = s.dir.join("strata-my-tiny.json");
    let cfg: Value = serde_json::from_slice(&std::fs::read(&cfg_path).unwrap()).unwrap();
    assert_eq!((cfg["model_name"].as_str(), cfg["display_name"].as_str()), (Some("my-tiny"), Some("My Tiny Llama")));
    assert!(std::path::Path::new(done["result"]["tokenizer"].as_str().unwrap()).join("tokenizer.json").exists(), "tokenizer folder");
    let (_, l) = s.get("/api/models", true);
    assert!(l["models"].as_array().unwrap().iter().any(|m| m["id"] == "my-tiny"), "listed: {l}");
    // the same name again is refused before anything runs; with overwrite it is allowed
    let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "strata", "input": gguf, "name": "my-tiny"}));
    assert_eq!(code, 400);
    assert!(e["error"]["message"].as_str().unwrap().contains("already installed"), "{e}");
    let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "strata", "input": gguf, "name": "my-tiny", "overwrite": true}));
    assert_eq!(code, 202);
}

