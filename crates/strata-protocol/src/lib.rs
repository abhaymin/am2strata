//! The PolyStrata engine line protocol (see docs/PROTOCOL.md).
//!
//! The server writes one command per line to the engine's stdin and reads one event per line from its stdout.
//! Both `strata` (Qwen3.8) and `strata-poly` speak it. This crate only formats commands and parses events; it does
//! no I/O.

use std::fmt::Write as _;

/// Sampling for one request. `None` fields are left out, so the engine's default applies (greedy when no
/// temperature is sent).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sampling {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub penalty_last_n: Option<i64>,
    pub seed: Option<i64>,
}

/// Python `repr(float)` for the values sent: integral values keep ".0" (the Python server writes `{x!r}`).
pub fn py_float(f: f64) -> String {
    if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

impl Sampling {
    /// The ` key=value` suffix of a GEN line, with the Python server's rules (serve/server.py sampling_keys).
    pub fn keys(&self) -> String {
        let mut k = String::new();
        if let Some(t) = self.temperature.filter(|t| *t > 0.0) {
            let _ = write!(k, " temperature={}", py_float(t));
        }
        if let Some(p) = self.top_p.filter(|p| *p < 1.0) {
            let _ = write!(k, " top_p={}", py_float(p));
        }
        if let Some(tk) = self.top_k.filter(|t| *t >= 0) {
            // the engine's sampled path keeps at most 64 candidates: 0 ("off") and wider lists get all 64
            let _ = write!(k, " top_k={}", if (1..=64).contains(&tk) { tk } else { 64 });
        }
        if let Some(m) = self.min_p.filter(|m| *m > 0.0 && *m <= 1.0) {
            let _ = write!(k, " min_p={}", py_float(m));
        }
        let rp = self.repetition_penalty.filter(|v| *v != 1.0);
        let pf = self.frequency_penalty.filter(|v| *v != 0.0);
        let pp = self.presence_penalty.filter(|v| *v != 0.0);
        if let Some(v) = rp {
            let _ = write!(k, " penalty_repeat={}", py_float(v));
        }
        if let Some(v) = pf {
            let _ = write!(k, " penalty_freq={}", py_float(v));
        }
        if let Some(v) = pp {
            let _ = write!(k, " penalty_present={}", py_float(v));
        }
        if rp.is_some() || pf.is_some() || pp.is_some() {
            // a penalty without a window counts over nothing: the engine's default is the last 64 tokens
            let _ = write!(k, " penalty_last_n={}", self.penalty_last_n.filter(|n| *n > 0).unwrap_or(64));
        }
        if let Some(s) = self.seed.filter(|s| *s > 0) {
            let _ = write!(k, " seed={s}");
        }
        k
    }
}

/// Pictures of a request: the combined embeddings file the image encoder wrote (`GENI`), the token that marks
/// their rows (strata-poly finds them by it), and each picture's key (its bytes' hash, PolyStrata #2228).
#[derive(Debug, Clone, Default)]
pub struct Images {
    pub embeddings: String,
    pub pad: u32,
    pub keys: Vec<String>,
}

/// A generation request: `GEN <max_new>[ key=value ...] <id,id,...>`, or `GENI <max_new>[ keys] <file> <ids>`.
#[derive(Debug, Clone)]
pub struct Gen<'a> {
    pub max_new: u32,
    pub sampling: &'a Sampling,
    /// Where the prompt's last turn begins (`turn_at=POS`): strata-poly keeps a conversation checkpoint there.
    pub turn_at: Option<usize>,
    pub ids: &'a [u32],
    pub images: Option<&'a Images>,
}

impl Gen<'_> {
    pub fn line(&self) -> String {
        let verb = if self.images.is_some() { "GENI" } else { "GEN" };
        let mut s = format!("{verb} {}{}", self.max_new, self.sampling.keys());
        if let Some(img) = self.images {
            let _ = write!(s, " image_pad={}", img.pad);
            let keys: Vec<&String> =
                img.keys.iter().filter(|k| !k.is_empty() && k.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())).collect();
            if !keys.is_empty() && keys.len() == img.keys.len() {
                let _ = write!(s, " img_keys={}", keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(","));
            }
        }
        if let Some(at) = self.turn_at {
            let _ = write!(s, " turn_at={at}");
        }
        if let Some(img) = self.images {
            let _ = write!(s, " {}", img.embeddings);
        }
        s.push(' ');
        for (i, id) in self.ids.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(s, "{id}");
        }
        s.push('\n');
        s
    }
}

/// `turn_at` for a prompt: the position of the last token that starts a turn (`<|im_start|>`), as the Python
/// server's `turn_key` finds it.
pub fn turn_at(ids: &[u32], turn_ids: &[u32]) -> Option<usize> {
    (1..ids.len()).rev().find(|&i| turn_ids.contains(&ids[i]))
}

pub const STOP: &str = "STOP\n";
pub const QUIT: &str = "QUIT\n";

/// The figures of a `DONE` line (serve/server.py _parse_done).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct Done {
    pub generated: u64,
    pub prompt_tokens: u64,
    pub prompt_ms: f64,
    pub decode_ms: f64,
    /// `eos`, `length`, `stop`, `nonfinite`, ...
    pub finish: String,
    pub drafts_accepted: Option<u64>,
    pub drafts_offered: Option<u64>,
    pub reused: Option<u64>,
    pub hits: Option<u64>,
    pub lookups: Option<u64>,
    pub prompt_read: Option<u64>,
}

/// One line from the engine.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// `T <id>`: a generated token.
    Token(u32),
    /// `PP <done> <total> [ms] [tok/s]`: prompt-reading progress, one per chunk.
    Progress { done: u64, total: u64, ms: Option<f64>, tok_s: Option<f64> },
    /// `RESUME <n>`: tokens of the prompt reused from the cache.
    Resume(u64),
    /// `DONE ...`: the request finished.
    Done(Done),
    /// `ERR <message>`: the request failed.
    Err(String),
    /// `INFO key=value ...`: engine facts (batch_slots, versions); sent at start-up and on VRAM changes.
    Info(String),
    /// `READY`: start-up finished; the engine takes commands.
    Ready(String),
    /// Anything else (the engine's own log lines go to stderr; this should be rare).
    Other(String),
}

fn num<T: std::str::FromStr>(f: &[&str], i: usize) -> Option<T> {
    f.get(i).and_then(|s| s.parse().ok())
}

impl Event {
    pub fn parse(line: &str) -> Event {
        let line = line.trim_end_matches(['\r', '\n']);
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("T") => match num(&f, 1) {
                Some(id) => Event::Token(id),
                None => Event::Other(line.into()),
            },
            Some("PP") => match (num(&f, 1), num(&f, 2)) {
                (Some(done), Some(total)) => Event::Progress { done, total, ms: num(&f, 3), tok_s: num(&f, 4) },
                _ => Event::Other(line.into()),
            },
            Some("RESUME") => num(&f, 1).map_or_else(|| Event::Other(line.into()), Event::Resume),
            Some("DONE") => Event::Done(Done {
                generated: num(&f, 1).unwrap_or(0),
                prompt_tokens: num(&f, 2).unwrap_or(0),
                prompt_ms: num(&f, 3).unwrap_or(0.0),
                decode_ms: num(&f, 4).unwrap_or(0.0),
                finish: f.get(5).map(|s| s.to_string()).unwrap_or_default(),
                drafts_accepted: num(&f, 6),
                drafts_offered: num(&f, 7),
                reused: num(&f, 8),
                hits: num(&f, 9),
                lookups: num(&f, 10),
                prompt_read: num(&f, 14),
            }),
            Some("ERR") => Event::Err(line.get(4..).unwrap_or("").trim().to_string()),
            Some("INFO") => Event::Info(line.get(5..).unwrap_or("").trim().to_string()),
            Some(s) if s.starts_with("READY") => Event::Ready(line.into()),
            _ => Event::Other(line.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_line_matches_python_format() {
        let s = Sampling { temperature: Some(0.7), top_p: Some(0.8), top_k: Some(20), presence_penalty: Some(1.5),
                           ..Default::default() };
        let g = Gen { max_new: 600, sampling: &s, turn_at: Some(3), ids: &[1, 2, 3, 4], images: None };
        assert_eq!(g.line(),
            "GEN 600 temperature=0.7 top_p=0.8 top_k=20 penalty_present=1.5 penalty_last_n=64 turn_at=3 1,2,3,4\n");
        let img = Images { embeddings: "/dev/shm/x.sve".into(), pad: 248056, keys: vec!["ab12".into(), "cd34".into()] };
        let g = Gen { max_new: 10, sampling: &Sampling::default(), turn_at: Some(2), ids: &[7, 8], images: Some(&img) };
        assert_eq!(g.line(), "GENI 10 image_pad=248056 img_keys=ab12,cd34 turn_at=2 /dev/shm/x.sve 7,8\n");
        let greedy = Sampling { temperature: Some(0.0), top_p: Some(1.0), top_k: Some(0), ..Default::default() };
        assert_eq!(greedy.keys(), " top_k=64");
        assert_eq!(Sampling { temperature: Some(1.0), ..Default::default() }.keys(), " temperature=1.0");
    }

    #[test]
    fn parses_events() {
        assert_eq!(Event::parse("T 248046\n"), Event::Token(248046));
        assert_eq!(Event::parse("PP 8192 23494 5210.5 1572.3"),
                   Event::Progress { done: 8192, total: 23494, ms: Some(5210.5), tok_s: Some(1572.3) });
        assert_eq!(Event::parse("RESUME 120"), Event::Resume(120));
        assert_eq!(Event::parse("ERR context full"), Event::Err("context full".into()));
        let Event::Done(d) = Event::parse("DONE 352 23494 14931.2 4625.0 eos 210 270 0") else { panic!() };
        assert_eq!((d.generated, d.prompt_tokens, d.finish.as_str(), d.drafts_accepted, d.reused),
                   (352, 23494, "eos", Some(210), Some(0)));
        assert!(matches!(Event::parse("READY strata-poly"), Event::Ready(_)));
    }

    #[test]
    fn turn_at_finds_last_turn_start() {
        assert_eq!(turn_at(&[9, 1, 5, 6, 1, 7], &[1]), Some(4));
        assert_eq!(turn_at(&[1, 5], &[1]), None);
    }
}
