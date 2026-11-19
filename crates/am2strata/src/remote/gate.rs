//! The content gate: before a request's text leaves this machine for a provider that does not take NSFW work, a small
//! local classifier (`am2strata-textgate`, DistilBERT on the CPU, kept warm) rates it. The gate fails closed: a text
//! it cannot rate, cannot read (pictures, a language the classifier was not trained on, a text too long) or is not
//! sure of stays here. Nothing of the text is logged.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};

/// What x-strata-note says when the gate kept a request here.
pub const KEPT: &str = "kept local (content)";

#[derive(Debug, PartialEq)]
pub enum Verdict {
    Release,
    /// why it stays (for the log: never the text)
    Hold(&'static str),
}

/// The text of a request body that a provider would read: every string in it, less the things that are not prose
/// (data: URLs and long base64). Returns it with whether the body carries pictures or other files.
pub fn outgoing(body: &Value) -> (String, bool) {
    fn walk(v: &Value, out: &mut String, files: &mut bool) {
        match v {
            Value::String(s) => {
                let blob = s.starts_with("data:") || (s.len() > 2048 && !s.contains(' ') && !s.contains('\n'));
                if blob {
                    *files = true;
                } else if !s.trim().is_empty() {
                    out.push_str(s);
                    out.push('\n');
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out, files)),
            Value::Object(o) => {
                for (k, x) in o {
                    // the request's own settings are not content
                    if matches!(k.as_str(), "model" | "role" | "type" | "id" | "tool_call_id" | "call_id" | "finish_reason" | "media_type" | "mime_type" | "mimeType") {
                        continue;
                    }
                    if matches!(k.as_str(), "image_url" | "inlineData" | "inline_data" | "input_audio" | "file") {
                        *files = true;
                    }
                    walk(x, out, files);
                }
            }
            _ => {}
        }
    }
    let (mut out, mut files) = (String::new(), false);
    walk(body, &mut out, &mut files);
    (out, files)
}

/// Whether the classifier (English) can read this text: at most a tenth of its letters outside the Latin script.
pub fn readable(text: &str) -> bool {
    let (mut letters, mut other) = (0usize, 0usize);
    for c in text.chars().filter(|c| c.is_alphabetic()) {
        letters += 1;
        if !(c.is_ascii() || ('\u{00C0}'..='\u{024F}').contains(&c)) {
            other += 1;
        }
    }
    other * 10 <= letters
}

/// The decision from what is known. `rated`: the classifier's answer, None when it gave none.
pub fn decide(files: bool, readable: bool, rated: Option<&Value>, release_under: f64) -> Verdict {
    if files {
        return Verdict::Hold("pictures or files, which the gate cannot read");
    }
    if !readable {
        return Verdict::Hold("a language the classifier does not read");
    }
    let Some(r) = rated else { return Verdict::Hold("the classifier did not answer") };
    if r["truncated"] != json!(false) {
        return Verdict::Hold("too long to read whole");
    }
    match r["scores"]["nsfw"].as_f64() {
        Some(p) if p.is_finite() && p < release_under => Verdict::Release,
        Some(_) => Verdict::Hold("rated NSFW or not surely safe"),
        None => Verdict::Hold("the classifier gave no score"),
    }
}

struct Proc {
    _child: Child,
    stdin: ChildStdin,
    lines: tokio::io::Lines<BufReader<ChildStdout>>,
}

pub struct Gate {
    exe: Option<PathBuf>,
    model: PathBuf,
    proc: tokio::sync::Mutex<Option<Proc>>,
}

/// $AM2STRATA_TEXTGATE_EXE, else crates/strata-media's release build beside this binary's checkout (as the media engine).
fn find_exe() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AM2STRATA_TEXTGATE_EXE").map(PathBuf::from) {
        return p.is_file().then_some(p);
    }
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().skip(1).take(5).map(|d| d.join("crates/strata-media/target/release/am2strata-textgate")).find(|p| p.is_file())
}

impl Gate {
    pub fn new(data_dir: &Path) -> Gate {
        Gate { exe: find_exe(), model: data_dir.join("models/classifiers/distilbert-nsfw-text-classifier"), proc: tokio::sync::Mutex::new(None) }
    }

    /// Whether the classifier and its model are on this machine (for /rtenai).
    #[allow(dead_code)]
    pub fn installed(&self, model: Option<&str>) -> bool {
        self.exe.is_some() && model.map(PathBuf::from).unwrap_or_else(|| self.model.clone()).join("model.safetensors").is_file()
    }

    async fn start(&self, model: &Path) -> Option<Proc> {
        let mut child = tokio::process::Command::new(self.exe.as_ref()?).arg(model)
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null())
            .kill_on_drop(true).spawn().ok()?;
        let stdin = child.stdin.take()?;
        let mut lines = BufReader::new(child.stdout.take()?).lines();
        let first = tokio::time::timeout(Duration::from_secs(60), lines.next_line()).await.ok()?.ok()??;
        (serde_json::from_str::<Value>(&first).ok()?["ready"] == json!(true)).then_some(Proc { _child: child, stdin, lines })
    }

    /// The classifier's answer for `text`; None when it could not be asked (the process is then started anew next time).
    async fn rate(&self, text: &str, model: Option<&str>) -> Option<Value> {
        let mut g = self.proc.lock().await;
        if g.is_none() {
            let dir = model.map(PathBuf::from).unwrap_or_else(|| self.model.clone());
            *g = self.start(&dir).await;
        }
        let p = g.as_mut()?;
        let line = format!("{}\n", json!({"text": text}));
        let asked = async {
            p.stdin.write_all(line.as_bytes()).await.ok()?;
            p.stdin.flush().await.ok()?;
            p.lines.next_line().await.ok()?
        };
        let got = tokio::time::timeout(Duration::from_secs(60), asked).await.ok().flatten()
            .and_then(|l| serde_json::from_str::<Value>(&l).ok()).filter(|v| v.get("error").is_none());
        if got.is_none() {
            *g = None;
        }
        got
    }

    /// May this body go to a provider that does not take NSFW work?
    pub async fn check(&self, body: &Value, model: Option<&str>, release_under: f64) -> Verdict {
        let (text, files) = outgoing(body);
        let can_read = readable(&text);
        // nothing to ask the classifier when the answer is already no
        let rated = if files || !can_read { None } else { self.rate(&text, model).await };
        decide(files, can_read, rated.as_ref(), release_under)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gate_reads_every_text_a_provider_would_and_sees_pictures() {
        let (t, f) = outgoing(&json!({"model": "minimax/x", "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "What is this?"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "a web page's text"}],
            "tools": [{"type": "function", "function": {"name": "search", "description": "Searches the web"}}]}));
        assert!(!f && t.contains("Be brief.") && t.contains("What is this?") && t.contains("a web page's text") && t.contains("Searches the web"));
        assert!(!t.contains("minimax/x") && !t.contains("c1"), "the request's own settings are not content");
        assert!(outgoing(&json!({"messages": [{"content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]})).1);
        assert!(outgoing(&json!({"messages": [{"content": [{"type": "image", "source": {"data": "A".repeat(4000)}}]}]})).1);
        assert!(outgoing(&json!({"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": "QUJD"}}]}]})).1);
    }

    #[test]
    fn only_text_the_classifier_reads_is_rated() {
        assert!(readable("Fix this loop, s'il vous plaît: für i in 0..n") && readable("") && readable("12345 + 6 = ?"));
        assert!(!readable("ॐ त्र्यम्बकं यजामहे") && !readable("Translate: これは何ですか") && !readable("привет, как дела"));
    }

    #[test]
    fn the_gate_releases_only_what_is_surely_safe() {
        let r = |p: f64, cut: bool| json!({"scores": {"safe": 1.0 - p, "nsfw": p}, "windows": 1, "truncated": cut});
        assert_eq!(decide(false, true, Some(&r(0.01, false)), 0.2), Verdict::Release);
        // every other case stays here: never a false release
        assert!(matches!(decide(false, true, Some(&r(0.2, false)), 0.2), Verdict::Hold(_)), "the limit itself holds");
        assert!(matches!(decide(false, true, Some(&r(0.99, false)), 0.2), Verdict::Hold(_)));
        assert!(matches!(decide(false, true, Some(&r(0.01, true)), 0.2), Verdict::Hold(_)), "not read whole");
        assert!(matches!(decide(true, true, Some(&r(0.01, false)), 0.2), Verdict::Hold(_)), "pictures");
        assert!(matches!(decide(false, false, Some(&r(0.01, false)), 0.2), Verdict::Hold(_)), "another language");
        assert!(matches!(decide(false, true, None, 0.2), Verdict::Hold(_)), "no classifier");
        assert!(matches!(decide(false, true, Some(&json!({"scores": {}})), 0.2), Verdict::Hold(_)), "no score");
        assert!(matches!(decide(false, true, Some(&json!({"scores": {"nsfw": "0.0"}, "truncated": false})), 0.2), Verdict::Hold(_)));
        assert!(matches!(decide(false, true, Some(&json!({"scores": {"nsfw": 0.01}})), 0.2), Verdict::Hold(_)), "no word on truncation");
    }

    #[tokio::test]
    async fn without_the_classifier_everything_stays_here() {
        let g = Gate { exe: None, model: PathBuf::from("/nonexistent"), proc: tokio::sync::Mutex::new(None) };
        assert!(matches!(g.check(&json!({"messages": [{"role": "user", "content": "hello"}]}), None, 0.2).await, Verdict::Hold(_)));
        assert!(!g.installed(None));
    }
}
