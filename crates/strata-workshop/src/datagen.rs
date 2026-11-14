//! Training data made by am2Strata's own model — the Workshop's data recipes (like Unsloth Studio's).
//!
//! Recipes:
//! - **qa**: from documents (text, Markdown, code, PDF): chunked, and for each chunk the model writes question–answer
//!   pairs grounded only in that chunk; each record keeps where it came from.
//! - **instruct**: from a few seed examples (a JSONL of chat records), the model writes new, varied ones on the same
//!   theme (self-instruct).
//! - **chat**: from documents, short multi-turn conversations grounded in each chunk.
//!
//! Then quality: dropped are records the filters refuse (too short or long, refusals, empty or echoed answers),
//! near-duplicates (MinHash), and — with a judge pass — answers the model scores below the cut-off as not supported by
//! their source. Every count is reported. The output is `dataset.jsonl` (`{"messages":[...],"meta":{...}}` per line)
//! and `dataset.card.md` (sources, settings, model, counts, and that the user is responsible for the sources' rights).
//!
//! The model is reached through the server's own chat API (`AM2STRATA_URL`, `AM2STRATA_API_KEY`), so requests queue
//! with everyone else's and show in the Monitor.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::dataset::{self, Record, Rules};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recipe {
    Qa,
    Instruct,
    Chat,
}

impl Recipe {
    pub fn from_s(s: &str) -> Result<Recipe> {
        Ok(match s {
            "qa" => Recipe::Qa,
            "instruct" => Recipe::Instruct,
            "chat" => Recipe::Chat,
            other => bail!("recipe {other:?}: qa, instruct or chat"),
        })
    }
    pub fn as_str(self) -> &'static str {
        match self { Recipe::Qa => "qa", Recipe::Instruct => "instruct", Recipe::Chat => "chat" }
    }
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub recipe: Recipe,
    /// documents (files or folders) for qa/chat
    pub sources: Vec<PathBuf>,
    /// a JSONL of example records for instruct
    pub seeds: Option<PathBuf>,
    /// how many records to aim for
    pub count: usize,
    pub per_chunk: usize,
    pub chunk_chars: usize,
    pub overlap: usize,
    pub temperature: f64,
    /// a second pass scores each answer 1–5 for how well its source supports it; below this it is dropped (0 = off)
    pub min_score: u8,
    pub topic: Option<String>,
    pub language: Option<String>,
    pub out: PathBuf,
}

// ------------------------------------------------------------------------------------------------------- sources

/// The text files a source path stands for (a folder: its text, Markdown, code and PDF files, two levels deep).
pub fn source_files(paths: &[PathBuf]) -> Vec<PathBuf> {
    const EXT: [&str; 22] = ["txt", "md", "markdown", "rst", "pdf", "html", "htm", "json", "csv", "rs", "py", "js", "ts",
                             "go", "java", "c", "h", "cpp", "hpp", "sh", "toml", "yaml"];
    fn walk(p: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if p.is_file() {
            if p.extension().and_then(|e| e.to_str()).is_some_and(|e| EXT.contains(&e.to_lowercase().as_str())) {
                out.push(p.to_path_buf());
            }
        } else if p.is_dir() && depth > 0 {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(p).into_iter().flatten().flatten().map(|e| e.path())
                .filter(|q| !q.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.'))).collect();
            entries.sort();
            for e in entries {
                walk(&e, depth - 1, out);
            }
        }
    }
    let mut out = vec![];
    for p in paths {
        walk(p, 3, &mut out);
    }
    out
}

/// A document's text: PDFs through `pdftotext`, HTML with its tags taken out, anything else as UTF-8.
pub fn read_text(path: &Path) -> Result<String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if ext == "pdf" {
        let o = std::process::Command::new("pdftotext").args(["-layout", "-enc", "UTF-8"]).arg(path).arg("-").output()
            .map_err(|e| anyhow!("{}: pdftotext ({e}); install poppler-utils to read PDFs", path.display()))?;
        if !o.status.success() {
            bail!("{}: pdftotext failed: {}", path.display(), String::from_utf8_lossy(&o.stderr).trim());
        }
        return Ok(String::from_utf8_lossy(&o.stdout).into_owned());
    }
    let raw = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    Ok(if ext == "html" || ext == "htm" { strip_tags(&text) } else { text })
}

fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let lower = html.to_lowercase();
    let mut skip_until: Option<&str> = None;
    for (i, c) in html.char_indices() {
        if let Some(end) = skip_until {
            if lower[i..].starts_with(end) { skip_until = None; } else { continue; }
        }
        match c {
            '<' => {
                if lower[i..].starts_with("<script") { skip_until = Some("</script"); }
                else if lower[i..].starts_with("<style") { skip_until = Some("</style"); }
                in_tag = true;
            }
            '>' if in_tag => { in_tag = false; out.push(' '); }
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ------------------------------------------------------------------------------------------------------- prompts

const SYSTEM: &str = "You write training data for a language model. You answer with JSON only: no prose before or \
after it, no Markdown fences.";

/// The request for one chunk (qa/chat) or one batch (instruct).
pub fn prompt(s: &Settings, chunk: Option<&str>, seeds: &[Record]) -> String {
    let lang = s.language.as_deref().map(|l| format!(" Write in {l}.")).unwrap_or_default();
    let topic = s.topic.as_deref().map(|t| format!(" Focus on: {t}.")).unwrap_or_default();
    match s.recipe {
        Recipe::Qa => format!(
            "From the text below, write {n} question-and-answer pairs a user might really ask. Each answer must be \
             fully supported by the text alone (no outside facts), complete, and in your own words; vary the kind of \
             question (facts, how, why, comparisons).{topic}{lang}\n\nAnswer as a JSON array: \
             [{{\"question\": \"...\", \"answer\": \"...\"}}]\n\nTEXT:\n{text}",
            n = s.per_chunk, text = chunk.unwrap_or("")),
        Recipe::Chat => format!(
            "From the text below, write {n} short conversations (2 or 3 user turns each) between a user and a helpful \
             assistant, where every assistant reply is supported by the text alone.{topic}{lang}\n\nAnswer as a JSON \
             array of conversations: [[{{\"role\": \"user\", \"content\": \"...\"}}, {{\"role\": \"assistant\", \
             \"content\": \"...\"}}, ...]]\n\nTEXT:\n{text}",
            n = s.per_chunk, text = chunk.unwrap_or("")),
        Recipe::Instruct => {
            let examples: Vec<Value> = seeds.iter().take(5).map(|r| json!(r.messages.iter()
                .map(|(role, content)| json!({"role": role, "content": content})).collect::<Vec<_>>())).collect();
            format!(
                "Here are example conversations:\n{ex}\n\nWrite {n} NEW examples of the same kind and quality: different \
                 tasks, different wording, a range of difficulty; never copy an example.{topic}{lang}\n\nAnswer as a \
                 JSON array of conversations in the same format.",
                ex = serde_json::to_string_pretty(&examples).unwrap_or_default(), n = s.per_chunk)
        }
    }
}

/// The judge's request: how well `source` supports `answer` to `question`, 1–5.
pub fn judge_prompt(question: &str, answer: &str, source: Option<&str>) -> String {
    match source {
        Some(src) => format!(
            "Score how well the SOURCE supports the ANSWER to the QUESTION, from 1 (not at all, or wrong) to 5 (fully \
             supported and correct). Answer as JSON: {{\"score\": N, \"why\": \"...\"}}\n\nSOURCE:\n{src}\n\nQUESTION:\n\
             {question}\n\nANSWER:\n{answer}"),
        None => format!(
            "Score this ANSWER to the QUESTION from 1 (wrong, unhelpful) to 5 (correct, complete, helpful). Answer as \
             JSON: {{\"score\": N, \"why\": \"...\"}}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}"),
    }
}

/// The first JSON value in a model's reply (after any thinking or Markdown fence), or None.
pub fn extract_json(reply: &str) -> Option<Value> {
    let r = match reply.rfind("</think>") { Some(i) => &reply[i + 8..], None => reply };
    let start = r.find(['[', '{'])?;
    let bytes = r.as_bytes();
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            match b { b'\\' if !esc => esc = true, b'"' if !esc => in_str = false, _ => esc = false }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&r[start..=i]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// The records in a reply, for the recipe: qa pairs become a user + assistant turn; conversations keep their turns.
pub fn records_from(recipe: Recipe, v: &Value) -> Vec<Record> {
    let items = match v { Value::Array(a) => a.clone(), other => vec![other.clone()] };
    let mut out = vec![];
    for it in items {
        let messages: Vec<(String, String)> = match recipe {
            Recipe::Qa => match (it["question"].as_str(), it["answer"].as_str()) {
                (Some(q), Some(a)) => vec![("user".into(), q.trim().into()), ("assistant".into(), a.trim().into())],
                _ => continue,
            },
            Recipe::Chat | Recipe::Instruct => {
                let turns = it.as_array().or_else(|| it["messages"].as_array()).cloned().unwrap_or_default();
                turns.iter().filter_map(|t| Some((t["role"].as_str()?.to_string(), t["content"].as_str()?.trim().to_string())))
                    .filter(|(r, _)| matches!(r.as_str(), "system" | "user" | "assistant")).collect()
            }
        };
        // a usable record: at least one user turn followed by an assistant turn
        let ok = messages.iter().position(|(r, _)| r == "user")
            .is_some_and(|u| messages.iter().skip(u + 1).any(|(r, _)| r == "assistant"));
        if ok {
            out.push(Record { messages });
        }
    }
    out
}

// --------------------------------------------------------------------------------------------------------- model

/// The server's own chat API (the model that is loaded), as `am2strata-workshop datagen` reaches it.
pub struct Model {
    url: String,
    key: Option<String>,
    client: reqwest::blocking::Client,
    pub name: String,
}

impl Model {
    pub fn from_env() -> Result<Model> {
        let url = std::env::var("AM2STRATA_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".into()).trim_end_matches('/').to_string();
        let key = std::env::var("AM2STRATA_API_KEY").ok().filter(|k| !k.is_empty());
        let client = reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(600)).build()?;
        let mut m = Model { url, key, client, name: String::new() };
        let models = m.get("/v1/models")?;
        m.name = models["data"].as_array().and_then(|a| a.iter().find(|x| x["loaded"] == true).or(a.first()))
            .and_then(|x| x["id"].as_str()).ok_or_else(|| anyhow!("the server lists no model"))?.to_string();
        Ok(m)
    }

    fn get(&self, path: &str) -> Result<Value> {
        let mut rb = self.client.get(format!("{}{path}", self.url));
        if let Some(k) = &self.key { rb = rb.bearer_auth(k); }
        let r = rb.send().with_context(|| format!("{} does not answer", self.url))?;
        anyhow::ensure!(r.status().is_success(), "{}{path}: HTTP {}", self.url, r.status());
        Ok(r.json()?)
    }

    /// One answer (its content; thinking is not part of it).
    pub fn ask(&self, system: &str, user: &str, temperature: f64, max_tokens: u32) -> Result<String> {
        let body = json!({"model": self.name, "temperature": temperature, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]});
        let mut rb = self.client.post(format!("{}/v1/chat/completions", self.url)).json(&body);
        if let Some(k) = &self.key { rb = rb.bearer_auth(k); }
        let r = rb.send().with_context(|| format!("{} does not answer", self.url))?;
        let ok = r.status().is_success();
        let v: Value = r.json().unwrap_or_default();
        if !ok {
            bail!("the model's answer failed: {}", v["error"]["message"].as_str().unwrap_or("unknown error"));
        }
        Ok(v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
    }
}

// ----------------------------------------------------------------------------------------------------------- run

/// What a run did, for its card and its last event.
#[derive(Debug, Default, Clone)]
pub struct Report {
    pub sources: Vec<String>,
    pub chunks: usize,
    pub asked: usize,
    pub unparsable: usize,
    pub generated: usize,
    pub filtered: std::collections::BTreeMap<String, usize>,
    pub duplicates: usize,
    pub judged_out: usize,
    pub kept: usize,
}

/// Generate the dataset. `progress(done, total, message)` is called as it goes; `should_stop()` ends it early (the
/// records so far are still written).
pub fn run(s: &Settings, model: &Model, mut progress: impl FnMut(usize, usize, &str), should_stop: impl Fn() -> bool) -> Result<Report> {
    let mut rep = Report::default();
    // the work: chunks of the documents, or batches from the seeds
    let mut units: Vec<(Option<String>, Option<String>)> = vec![]; // (chunk text, its source "file:start-end")
    let mut seeds = vec![];
    match s.recipe {
        Recipe::Qa | Recipe::Chat => {
            let files = source_files(&s.sources);
            if files.is_empty() {
                bail!("no documents found in the sources (text, Markdown, code, HTML or PDF files)");
            }
            for f in &files {
                let text = read_text(f)?;
                rep.sources.push(f.display().to_string());
                for c in dataset::chunk(&text, s.chunk_chars, s.overlap) {
                    if c.text.trim().chars().count() >= 200 {
                        units.push((Some(c.text), Some(format!("{}:{}-{}", f.display(), c.start, c.end))));
                    }
                }
            }
            rep.chunks = units.len();
            // only as many chunks as the count needs (spread over the documents)
            let need = s.count.div_ceil(s.per_chunk.max(1));
            if units.len() > need {
                let step = units.len() as f64 / need as f64;
                units = (0..need).map(|i| units[(i as f64 * step) as usize].clone()).collect();
            }
        }
        Recipe::Instruct => {
            let path = s.seeds.as_ref().ok_or_else(|| anyhow!("instruct needs seed examples (a JSONL of chat records)"))?;
            let (recs, _bad) = dataset::read_jsonl(path)?;
            anyhow::ensure!(!recs.is_empty(), "{}: no usable seed records", path.display());
            rep.sources.push(path.display().to_string());
            seeds = recs;
            units = vec![(None, None); s.count.div_ceil(s.per_chunk.max(1))];
        }
    }
    let total = units.len();
    let mut records: Vec<(Record, Option<String>, Option<String>)> = vec![]; // (record, source ref, chunk)
    for (i, (chunk, src)) in units.iter().enumerate() {
        if should_stop() {
            progress(i, total, "stopped: writing what was made so far");
            break;
        }
        rep.asked += 1;
        let reply = model.ask(SYSTEM, &prompt(s, chunk.as_deref(), &seeds), s.temperature, 4096)?;
        let made = extract_json(&reply).map(|v| records_from(s.recipe, &v)).unwrap_or_default();
        if made.is_empty() {
            rep.unparsable += 1;
        }
        rep.generated += made.len();
        for r in made {
            records.push((r, src.clone(), chunk.clone()));
        }
        progress(i + 1, total, &format!("{} records so far", records.len()));
        if records.len() >= s.count * 2 {
            break; // enough to filter down from
        }
    }

    // quality: filters, then near-duplicates, then the judge
    let rules = Rules::default();
    records.retain(|(r, _, _)| match dataset::filter(r, &rules) {
        Some(why) => { *rep.filtered.entry(why.to_string()).or_default() += 1; false }
        None => true,
    });
    let texts: Vec<String> = records.iter().map(|(r, _, _)| r.messages.iter().map(|(_, c)| c.as_str()).collect::<Vec<_>>().join("\n")).collect();
    let drop = dataset::near_duplicates(&texts, 0.8);
    rep.duplicates = drop.len();
    let mut keep: Vec<bool> = vec![true; records.len()];
    for d in drop { keep[d] = false; }
    let mut kept: Vec<(Record, Option<String>, Option<u8>)> = vec![];
    for (i, (r, src, chunk)) in records.into_iter().enumerate() {
        if !keep[i] { continue; }
        let mut score = None;
        if s.min_score > 0 {
            let q = r.messages.iter().find(|(role, _)| role == "user").map(|(_, c)| c.as_str()).unwrap_or("");
            let a = r.messages.iter().rev().find(|(role, _)| role == "assistant").map(|(_, c)| c.as_str()).unwrap_or("");
            let reply = model.ask(SYSTEM, &judge_prompt(q, a, chunk.as_deref()), 0.0, 512)?;
            let sc = extract_json(&reply).and_then(|v| v["score"].as_u64()).unwrap_or(0) as u8;
            if sc < s.min_score {
                rep.judged_out += 1;
                continue;
            }
            score = Some(sc);
        }
        kept.push((r, src, score));
        if kept.len() >= s.count { break; }
    }
    rep.kept = kept.len();

    // write the dataset and its card
    if let Some(d) = s.out.parent() { std::fs::create_dir_all(d)?; }
    let mut lines = String::new();
    for (r, src, score) in &kept {
        let mut meta = json!({"recipe": s.recipe.as_str(), "model": model.name});
        if let Some(src) = src { meta["source"] = json!(src); }
        if let Some(sc) = score { meta["score"] = json!(sc); }
        lines.push_str(&json!({"messages": r.messages.iter().map(|(role, c)| json!({"role": role, "content": c})).collect::<Vec<_>>(),
                               "meta": meta}).to_string());
        lines.push('\n');
    }
    std::fs::write(&s.out, lines).with_context(|| format!("writing {}", s.out.display()))?;
    std::fs::write(card_path(&s.out), card(s, &model.name, &rep))?;
    Ok(rep)
}

pub fn card_path(out: &Path) -> PathBuf {
    let stem = out.file_stem().and_then(|x| x.to_str()).unwrap_or("dataset");
    out.with_file_name(format!("{stem}.card.md"))
}

/// The dataset card: what it was made from, how, and what was dropped.
pub fn card(s: &Settings, model: &str, r: &Report) -> String {
    let filtered: Vec<String> = r.filtered.iter().map(|(k, v)| format!("{v} {k}")).collect();
    format!(
"# Dataset card: {name}

Made by am2Strata's Workshop with the **{recipe}** recipe, using the model **{model}**.

| | |
|---|---|
| Records kept | {kept} |
| Records written by the model | {gen} |
| Requests to the model | {asked} ({unparsable} with no usable JSON) |
| Dropped by the filters | {filtered} |
| Near-duplicates dropped | {dups} |
| Dropped by the judge (score below {min}) | {judged} |
| Chunks of the documents | {chunks} (chunk size {cs} characters, overlap {ov}) |
| Temperature | {temp} |

## Sources

{sources}

## Notes

- Every record's `meta.source` names the document and the character range it was made from (qa and chat recipes).
- **You are responsible for having the right to use the source documents** for training, and for checking the data
  before you train on it (Workshop › Review).
- Generated by a language model: it can be wrong, even after the filters and the judge.
",
        name = s.out.file_name().and_then(|n| n.to_str()).unwrap_or("dataset"), recipe = s.recipe.as_str(), model = model,
        kept = r.kept, gen = r.generated, asked = r.asked, unparsable = r.unparsable,
        filtered = if filtered.is_empty() { "none".into() } else { filtered.join(", ") }, dups = r.duplicates,
        min = if s.min_score > 0 { s.min_score.to_string() } else { "off".into() }, judged = r.judged_out,
        chunks = r.chunks, cs = s.chunk_chars, ov = s.overlap, temp = s.temperature,
        sources = r.sources.iter().map(|x| format!("- `{x}`")).collect::<Vec<_>>().join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_found_after_thinking_and_fences() {
        let v = extract_json("<think>let me see [not this]</think>\n```json\n[{\"question\": \"a]\", \"answer\": \"b\"}]\n```").unwrap();
        assert_eq!(v[0]["question"], "a]", "a bracket inside a string does not end the array");
        assert!(extract_json("no json here").is_none());
        assert_eq!(extract_json("{\"score\": 4, \"why\": \"ok\"} trailing").unwrap()["score"], 4);
    }

    #[test]
    fn records_from_each_recipe() {
        let qa = records_from(Recipe::Qa, &json!([{"question": " Q? ", "answer": "A."}, {"question": "no answer"}]));
        assert_eq!(qa.len(), 1);
        assert_eq!(qa[0].messages, vec![("user".into(), "Q?".into()), ("assistant".into(), "A.".into())]);
        let chat = records_from(Recipe::Chat, &json!([[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                                                       [{"role": "assistant", "content": "only me"}]]));
        assert_eq!(chat.len(), 1, "a conversation needs a user turn then an assistant turn");
        let inst = records_from(Recipe::Instruct, &json!([{"messages": [{"role": "user", "content": "do"}, {"role": "assistant", "content": "done"}]}]));
        assert_eq!(inst.len(), 1);
    }

    #[test]
    fn sources_are_found_and_read() {
        let d = std::env::temp_dir().join(format!("am2-datagen-{}", std::process::id()));
        std::fs::create_dir_all(d.join("sub")).unwrap();
        std::fs::write(d.join("a.md"), "# Title\nSome text.").unwrap();
        std::fs::write(d.join("sub/b.html"), "<html><style>x{}</style><p>Hello <b>there</b></p><script>var a</script></html>").unwrap();
        std::fs::write(d.join("skip.bin"), [0u8, 1, 2]).unwrap();
        std::fs::write(d.join(".hidden.md"), "no").unwrap();
        let files = source_files(&[d.clone()]);
        assert_eq!(files.len(), 2, "{files:?}");
        assert_eq!(read_text(&d.join("sub/b.html")).unwrap(), "Hello there");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn prompts_carry_the_settings() {
        let s = Settings { recipe: Recipe::Qa, sources: vec![], seeds: None, count: 10, per_chunk: 3, chunk_chars: 2000,
                           overlap: 200, temperature: 0.7, min_score: 3, topic: Some("billing".into()),
                           language: Some("German".into()), out: PathBuf::from("/tmp/x.jsonl") };
        let p = prompt(&s, Some("THE TEXT"), &[]);
        assert!(p.contains("3 question-and-answer pairs") && p.contains("billing") && p.contains("German") && p.contains("THE TEXT"));
        assert!(judge_prompt("q", "a", Some("src")).contains("SOURCE:\nsrc"));
        let c = card(&s, "tiel", &Report { kept: 7, generated: 9, ..Default::default() });
        assert!(c.contains("Records kept | 7") && c.contains("responsible for having the right"));
    }
}
