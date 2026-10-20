//! Byte-level BPE tokenizer for the tokenizer directories PolyStrata's `tools/strata_tokenizer.py` writes
//! (`vocab.json`, `merges.txt`, `token_type.json`, `tokenizer.json`).
//!
//! This is a port of that Python tokenizer, not a reimplementation from first principles: the pre-tokenizer
//! patterns, the merge order (lowest rank first, leftmost on ties) and the special-token rules must match it
//! exactly, and the golden files recorded from it are the test.
//!
//! Special tokens: GGUF token type 4 (USER_DEFINED: `<think>`, `<tool_call>`) is matched always; type 3
//! (CONTROL: `<|im_start|>`, `<|im_end|>`) only with `parse_special`. Both are matched on the raw text before the
//! byte mapping, longest literal first.

use aho_corasick::{AhoCorasick, MatchKind};
use fancy_regex::Regex;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading {path}: {source}")]
    Io { path: String, source: std::io::Error },
    #[error("parsing {path}: {source}")]
    Json { path: String, source: serde_json::Error },
    #[error("tokenizer: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `qwen35` pre-tokenizer (llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35). Note the `\p{M}` classes: without them a combining
/// mark is swallowed into the following punctuation run.
const QWEN35_PATTERN: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// `glm4` / `glm5` / `lfm2` pre-tokenizer: no `\p{M}`, digits in runs of up to three, with ignore_merges.
const GLM4_PATTERN: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Words with more symbols than this go through the heap merge (same result, O(n log n)).
const HEAP_MIN: usize = 64;

#[derive(serde::Deserialize, Default)]
struct Meta {
    #[serde(default)]
    pre: Option<String>,
    #[serde(default)]
    special_ids: HashMap<String, i64>,
}

pub struct Tokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    token_types: Vec<i32>,
    pre: String,
    ignore_merges: bool,
    pattern: Regex,
    /// literal matchers: all specials (types 3 and 4), and the always-matched ones (type 4)
    special_all: Option<(AhoCorasick, Vec<u32>)>,
    special_always: Option<(AhoCorasick, Vec<u32>)>,
    special_ids: HashMap<String, i64>,
    byte_to_unicode: [char; 256],
    unicode_to_byte: HashMap<char, u8>,
}

/// The GPT-2 byte encoder: 256 bytes -> 256 printable characters. Bytes 33..126, 161..172 and 174..255 map to
/// themselves; the rest shift into 256+n in byte order.
fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (0x21..0x7F).chain(0xA1..0xAD).chain(0xAE..0x100).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        out[*b as usize] = char::from_u32(*c).expect("valid scalar");
    }
    out
}

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| Error::Io { path: path.display().to_string(), source: e })
}

fn parse<T: serde::de::DeserializeOwned>(path: &Path, text: &str) -> Result<T> {
    serde_json::from_str(text).map_err(|e| Error::Json { path: path.display().to_string(), source: e })
}

fn literal_matcher(lits: Vec<(String, u32)>) -> Result<Option<(AhoCorasick, Vec<u32>)>> {
    if lits.is_empty() {
        return Ok(None);
    }
    // Python sorts by length, longest first, and alternates; LeftmostLongest gives the same match.
    let (pats, ids): (Vec<String>, Vec<u32>) = lits.into_iter().unzip();
    let ac = AhoCorasick::builder()
        .match_kind(MatchKind::LeftmostLongest)
        .build(&pats)
        .map_err(|e| Error::Invalid(format!("special token matcher: {e}")))?;
    Ok(Some((ac, ids)))
}

impl Tokenizer {
    /// Load a tokenizer directory written by `strata_tokenizer.py extract`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let vocab_path = dir.join("vocab.json");
        let vocab: HashMap<String, u32> = parse(&vocab_path, &read(&vocab_path)?)?;
        let mut tokens = vec![String::new(); vocab.len()];
        for (t, i) in &vocab {
            let i = *i as usize;
            if i >= tokens.len() {
                return Err(Error::Invalid(format!("token id {i} outside a vocabulary of {}", tokens.len())));
            }
            tokens[i] = t.clone();
        }
        let merges: Vec<String> = if dir.join("merges.json").exists() {
            let p = dir.join("merges.json");
            parse(&p, &read(&p)?)?
        } else {
            read(&dir.join("merges.txt"))?.split('\n').map(str::to_owned).collect()
        };
        let types_path = dir.join("token_type.json");
        let token_types: Vec<i32> = parse(&types_path, &read(&types_path)?)?;
        let meta: Meta = if dir.join("tokenizer.json").exists() {
            let p = dir.join("tokenizer.json");
            parse(&p, &read(&p)?)?
        } else {
            Meta::default()
        };
        Self::new(tokens, merges, token_types, meta.pre.unwrap_or_else(|| "qwen35".into()), meta.special_ids)
    }

    pub fn new(
        tokens: Vec<String>,
        merges: Vec<String>,
        token_types: Vec<i32>,
        pre: String,
        special_ids: HashMap<String, i64>,
    ) -> Result<Self> {
        if matches!(pre.as_str(), "gemma4" | "joyai-llm" | "deepseek-v3") {
            return Err(Error::Invalid(format!("pre-tokenizer {pre:?} is not ported yet")));
        }
        let mut ids = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            if ids.insert(t.clone(), i as u32).is_some() {
                return Err(Error::Invalid(format!("vocabulary has a duplicate token {t:?}")));
            }
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            let mut parts = m.split(' ');
            let (Some(a), Some(b), None) = (parts.next(), parts.next(), parts.next()) else {
                return Err(Error::Invalid(format!("merge {i} is not a pair: {m:?}")));
            };
            if !ids.contains_key(a) || !ids.contains_key(b) {
                return Err(Error::Invalid(format!("merge {i} names a token outside the vocabulary: {m:?}")));
            }
            ranks.insert((a.to_owned(), b.to_owned()), i as u32);
        }
        let pattern_src = match pre.as_str() {
            "glm4" | "glm5" | "lfm2" => GLM4_PATTERN,
            _ => QWEN35_PATTERN,
        };
        let pattern = Regex::new(pattern_src).map_err(|e| Error::Invalid(format!("pre-tokenizer pattern: {e}")))?;
        let mut all = Vec::new();
        let mut always = Vec::new();
        for (i, ty) in token_types.iter().enumerate() {
            if *ty == 3 || *ty == 4 {
                all.push((tokens[i].clone(), i as u32));
                if *ty == 4 {
                    always.push((tokens[i].clone(), i as u32));
                }
            }
        }
        let byte_to_unicode = bytes_to_unicode();
        let unicode_to_byte = byte_to_unicode.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        Ok(Self {
            ignore_merges: matches!(pre.as_str(), "glm4" | "glm5" | "lfm2"),
            tokens,
            ids,
            ranks,
            token_types,
            pre,
            pattern,
            special_all: literal_matcher(all)?,
            special_always: literal_matcher(always)?,
            special_ids,
            byte_to_unicode,
            unicode_to_byte,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn pre(&self) -> &str {
        &self.pre
    }

    /// A GGUF special id such as `tokenizer.ggml.eos_token_id`.
    pub fn special_id(&self, key: &str) -> Option<u32> {
        self.special_ids.get(key).map(|v| *v as u32)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    pub fn token_type(&self, id: u32) -> i32 {
        self.token_types.get(id as usize).copied().unwrap_or(1)
    }

    /// Tokenize `text`. `parse_special` controls only the CONTROL literals (type 3); USER_DEFINED (type 4) are
    /// matched either way.
    pub fn encode(&self, text: &str, parse_special: bool) -> Result<Vec<u32>> {
        let matcher = if parse_special { &self.special_all } else { &self.special_always };
        let mut out = Vec::new();
        let Some((ac, lit_ids)) = matcher else {
            self.encode_plain(text, &mut out)?;
            return Ok(out);
        };
        let mut pos = 0;
        for m in ac.find_iter(text) {
            if m.start() > pos {
                self.encode_plain(&text[pos..m.start()], &mut out)?;
            }
            out.push(lit_ids[m.pattern().as_usize()]);
            pos = m.end();
        }
        if pos < text.len() {
            self.encode_plain(&text[pos..], &mut out)?;
        }
        Ok(out)
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for m in self.pattern.find_iter(text) {
            let piece = m.map_err(|e| Error::Invalid(format!("pre-tokenizer: {e}")))?.as_str();
            let mapped: String = piece.bytes().map(|b| self.byte_to_unicode[b as usize]).collect();
            if self.ignore_merges {
                if let Some(id) = self.ids.get(&mapped) {
                    out.push(*id);
                    continue;
                }
            }
            for sym in self.bpe(&mapped) {
                match self.ids.get(&sym) {
                    Some(id) => out.push(*id),
                    None => return Err(Error::Invalid(format!("BPE produced a token outside the vocabulary: {sym:?}"))),
                }
            }
        }
        Ok(())
    }

    fn rank(&self, a: &str, b: &str) -> Option<u32> {
        // (String, String) keys: build a temporary pair. Allocation here is acceptable for the reference port;
        // a production version would intern symbols to ids.
        self.ranks.get(&(a.to_owned(), b.to_owned())).copied()
    }

    /// Merge by lowest rank first, leftmost on ties (Python's strict `<` scan).
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if parts.len() > HEAP_MIN {
            return self.bpe_heap(parts);
        }
        while parts.len() > 1 {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len() - 1 {
                if let Some(r) = self.rank(&parts[i], &parts[i + 1]) {
                    if best.map_or(true, |(_, br)| r < br) {
                        best = Some((i, r));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let right = parts.remove(i + 1);
            parts[i].push_str(&right);
        }
        parts
    }

    /// The same merges in the same order with a heap keyed (rank, position) over a linked list of symbols.
    fn bpe_heap(&self, mut parts: Vec<String>) -> Vec<String> {
        let n = parts.len();
        let mut alive = vec![true; n];
        let mut nxt: Vec<isize> = (1..=n as isize).collect();
        nxt[n - 1] = -1;
        let mut prv: Vec<isize> = (-1..n as isize - 1).collect();
        let mut heap: BinaryHeap<Reverse<(u32, usize, String, String)>> = BinaryHeap::new();
        for i in 0..n - 1 {
            if let Some(r) = self.rank(&parts[i], &parts[i + 1]) {
                heap.push(Reverse((r, i, parts[i].clone(), parts[i + 1].clone())));
            }
        }
        while let Some(Reverse((_, i, left, right))) = heap.pop() {
            let j = nxt[i];
            if !alive[i] || parts[i] != left || j < 0 || !alive[j as usize] || parts[j as usize] != right {
                continue;
            }
            let j = j as usize;
            parts[i] = format!("{left}{right}");
            alive[j] = false;
            let k = nxt[j];
            nxt[i] = k;
            if k >= 0 {
                prv[k as usize] = i as isize;
                if let Some(r2) = self.rank(&parts[i], &parts[k as usize]) {
                    heap.push(Reverse((r2, i, parts[i].clone(), parts[k as usize].clone())));
                }
            }
            let p = prv[i];
            if p >= 0 {
                if let Some(r2) = self.rank(&parts[p as usize], &parts[i]) {
                    heap.push(Reverse((r2, p as usize, parts[p as usize].clone(), parts[i].clone())));
                }
            }
        }
        parts.into_iter().zip(alive).filter_map(|(s, a)| a.then_some(s)).collect()
    }

    /// The raw bytes of one token. CONTROL and USER_DEFINED tokens are their text.
    pub fn token_bytes(&self, id: u32) -> Result<Vec<u8>> {
        let t = self
            .tokens
            .get(id as usize)
            .ok_or_else(|| Error::Invalid(format!("token id {id} is outside the vocabulary ({})", self.tokens.len())))?;
        if matches!(self.token_type(id), 3 | 4) {
            return Ok(t.as_bytes().to_vec());
        }
        t.chars()
            .map(|c| {
                self.unicode_to_byte
                    .get(&c)
                    .copied()
                    .ok_or_else(|| Error::Invalid(format!("token {id} has a character outside the byte alphabet: {c:?}")))
            })
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut raw = Vec::new();
        for id in ids {
            raw.extend(self.token_bytes(*id)?);
        }
        Ok(String::from_utf8_lossy(&raw).into_owned())
    }
}

/// Incremental UTF-8 decoding of a token stream: holds a character split across tokens until it is complete.
pub struct Detokenizer {
    pending: Vec<u8>,
}

impl Default for Detokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Detokenizer {
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// Feed one token's bytes; returns the complete text it finishes.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..good]).expect("valid prefix"));
                    match e.error_len() {
                        // an invalid sequence: replace it, as errors="replace" does, and go on
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..good + bad);
                        }
                        // an incomplete character at the end: keep it for the next token
                        None => {
                            self.pending.drain(..good);
                            return out;
                        }
                    }
                }
            }
        }
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}
