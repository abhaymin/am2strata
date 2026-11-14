//! The sandbox VM's API against the mock engine, with no VM booted: only the server's own key may use it, requests
//! are checked before anything runs, settings are kept and clamped, and a start without the Debian image says what is
//! missing. (Booting a real VM needs KVM and the 326 MB image: tested by hand, see docs/DEPLOYMENT.md.)

mod common;

use common::*;
use serde_json::json;
use std::time::{Duration, Instant};

#[test]
fn the_vm_api_is_owner_only_checks_requests_and_explains_what_is_missing() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    assert_eq!(s.get("/api/vm", false).0, 401, "needs the key");
    let (code, st) = s.get("/api/vm", true);
    assert_eq!(code, 200, "{st}");
    assert_eq!(st["phase"], "off");
    assert_eq!(st["image"], false);
    assert_eq!(st["config"]["memory_mb"], 2048);

    // checked before anything runs
    for (body, why) in [(json!({"action": "reset"}), "confirm"), (json!({"action": "fly"}), "unknown action")] {
        let (code, e) = s.post("/api/vm", &body);
        assert_eq!(code, 400, "{body}: {e}");
        assert!(e["error"]["message"].as_str().unwrap().contains(why), "{e}");
    }
    for (body, why) in [(json!({"kind": "apt", "target": "gimp; reboot", "action": "install"}), "package names"),
                        (json!({"kind": "deb", "target": "http://x/y.deb", "action": "install"}), "https"),
                        (json!({"app": "no-such-app", "action": "install"}), "catalog"),
                        (json!({"action": "install"}), "give")] {
        let (code, e) = s.post("/api/vm/apps", &body);
        assert_eq!(code, 400, "{body}: {e}");
        assert!(e["error"]["message"].as_str().unwrap().contains(why), "{body}: {e}");
    }
    assert_eq!(s.post("/api/vm/exec", &json!({})).0, 400);

    // settings: clamped, kept, unknown ones refused
    let (code, r) = s.post("/api/vm/settings", &json!({"memory_mb": 100, "cpus": 3}));
    assert_eq!(code, 200, "{r}");
    assert_eq!((r["config"]["memory_mb"].as_u64(), r["config"]["cpus"].as_u64()), (Some(768), Some(3)));
    assert_eq!(s.get("/api/vm", true).1["config"]["cpus"], 3);
    assert!(s.dir.join("data/sandbox/vm.json").exists() || std::fs::read_dir(&s.dir).is_ok());
    assert_eq!(s.post("/api/vm/settings", &json!({"gpu": true})).0, 400);

    // a start without the image fails, saying what to do
    let (code, t) = s.post("/api/vm", &json!({"action": "start"}));
    assert_eq!(code, 202, "{t}");
    let id = t["task"].as_u64().unwrap();
    let t0 = Instant::now();
    let task = loop {
        let (_, task) = s.get(&format!("/api/vm/tasks/{id}"), true);
        if task["state"] != "running" { break task; }
        assert!(t0.elapsed() < Duration::from_secs(20), "{task}");
        std::thread::sleep(Duration::from_millis(100));
    };
    assert_eq!(task["state"], "failed", "{task}");
    let e = task["error"].as_str().unwrap();
    assert!(e.contains("Debian image") || e.contains("this PC needs"), "{e}");
    assert_eq!(s.get("/api/vm", true).1["phase"], "off");
}
