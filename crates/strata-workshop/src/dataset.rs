//! Training-dataset tools: chunking text for retrieval, near-duplicate detection, chat-record filtering, and JSONL
//! loading. Pure functions — the only file I/O lives in [`read_jsonl`].

/// A piece of source text plus its character offsets into the original input.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    /// The chunked text. Always equals `text[start..end]` of the original input.
    pub text: String,
    /// Start character offset into the original input (inclusive).
    pub start: usize,
    /// End character offset into the original input (exclusive).
    pub end: usize,
}

/// Whether a word's last visible character is a sentence terminator.
fn ends_with_punct(word: &[char]) -> bool {
    word.iter().rev().any(|c| matches!(c, '.' | '!' | '?'))
}

/// Split `text` into retrieval chunks no longer than `max_chars`, preferring to break on paragraph and sentence
/// boundaries (never mid-word). The last `overlap` characters of each chunk are repeated at the start of the next.
pub fn chunk(text: &str, max_chars: usize, overlap: usize) -> Vec<Chunk> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n == 0 || max_chars == 0 {
        return Vec::new();
    }
    // an overlap as long as a chunk would never move forward: keep it under half a chunk
    let overlap = overlap.min(max_chars / 2);

    // Tokenize into words with the whitespace run that follows each word. A `Token` spans `[start, end)` in char space;
    // `paragraph` is true when the whitespace after the word breaks to a new line. Sentence-end detection keys off the last visible character of the word.
    struct Token {
        start: usize,
        end: usize,
        paragraph: bool,
        sentence_end: bool,
    }
    let mut tokens: Vec<Token> = Vec::new();
    let mut i = 0usize;
    while i < n {
        // Skip leading whitespace so the word starts on a real character.
        while i < n && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i;
        while i < n && !chars[i].is_whitespace() {
            i += 1;
        }
        let end = i;
        // Measure the trailing whitespace run.
        let ws_start = i;
        while i < n && chars[i].is_whitespace() {
            i += 1;
        }
        let word: &[char] = &chars[start..end];
        let sentence_end = ends_with_punct(word) || end == n;
        let paragraph = chars[ws_start..i].contains(&'\n');
        tokens.push(Token { start, end, paragraph, sentence_end });
    }

    let mut out: Vec<Chunk> = Vec::new();
    let mut prev_end: Option<usize> = None;
    // the first word of the previous chunk: each chunk starts strictly after it, so the loop always moves forward
    // whatever the overlap and the sentence lengths are
    let mut prev_first: Option<usize> = None;
    let mut idx = 0usize;
    while idx < tokens.len() {
        // Choose this chunk's start: overlap into the previous chunk, or zero for the first one.
        let start = match prev_end {
            Some(pe) => pe.saturating_sub(overlap).min(n),
            None => 0,
        };

        // If the start lands inside a word, back up to that word's real start so `text[start..]` is meaningful; the lost
        // characters are just overlap noise anyway.
        let mut first = tokens.iter().position(|t| t.start >= start).unwrap_or(tokens.len() - 1);
        if let Some(p) = prev_first {
            if first <= p {
                first = p + 1;
                if first >= tokens.len() {
                    break;
                }
            }
        }
        prev_first = Some(first);
        let chunk_start = tokens[first].start;

        // Greedily consume whole words until the next one would push the chunk over `max_chars`. Prefer cutting at a
        // sentence/paragraph boundary, but only among words that actually fit; when none fit within the limit we fall
        // back to a hard word-boundary cut so no chunk ever exceeds `max_chars` (the single-long-word case is handled
        // below as its own chunk).
        let mut len = 0usize;
        let mut j = first;
        let mut last_boundary: Option<usize> = None;
        while j < tokens.len() {
            let space = if j == first { 0 } else { 1 };
            let add = (tokens[j].end - tokens[j].start) + space;
            if len + add > max_chars {
                break;
            }
            len += add;
            if tokens[j].sentence_end || tokens[j].paragraph {
                last_boundary = Some(j);
            }
            j += 1;
        }

        // A single word longer than the limit is its own chunk and is never dropped.
        if len == 0 {
            out.push(Chunk { text: chars[tokens[first].start..tokens[first].end].iter().collect(), start: tokens[first].start, end: tokens[first].end });
            prev_end = Some(tokens[first].end);
            idx = first + 1;
            continue;
        }

        let cut = last_boundary.unwrap_or(j - 1);
        let end = tokens[cut].end.min(n);
        let chunk_text: String = chars[chunk_start..end].iter().collect();
        out.push(Chunk { text: chunk_text, start: chunk_start, end });
        // Next chunk overlaps into this one's tail.
        prev_end = Some(end.min(n));
        idx = cut + 1;
    }

    out
}

// ---------------------------------------------------------------------------
// near_duplicates: MinHash of word 5-gram shingles
// ---------------------------------------------------------------------------

const SHINGLE_SIZE: usize = 5;
const NUM_HASHES: usize = 128;
/// Largest Mersenne prime < 2^63, used as the modulus for the Kirsch–Mitzenmacher double hashing.
const HASH_MOD: u128 = (1u128 << 61) - 1;

/// Lowercase words of `text`, keeping only runs of alphanumeric characters.
fn tokenize_words(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|s: &str| s.to_string())
        .collect()
}

/// Consecutive `SHINGLE_SIZE`-word shingles joined by spaces.
fn word_shingles(words: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    if words.len() <= SHINGLE_SIZE {
        if !words.is_empty() {
            out.push(words.join(" "));
        }
        return out;
    }
    for w in words.windows(SHINGLE_SIZE) {
        out.push(w.join(" "));
    }
    out
}

/// Two independent 64-bit hashes of a string via SHA-256, reduced into `[0, HASH_MOD)`.
fn double_hash(shingle: &str) -> (u64, u64) {
    use sha2::{Digest, Sha256};
    let mut h1 = Sha256::new();
    h1.update(b"nh1\x00");
    h1.update(shingle.as_bytes());
    let d1 = h1.finalize();
    let mut h2 = Sha256::new();
    h2.update(b"nh2\x00");
    h2.update(shingle.as_bytes());
    let d2 = h2.finalize();
    (
        u64::from_le_bytes(d1[..8].try_into().unwrap()) % HASH_MOD as u64,
        u64::from_le_bytes(d2[..8].try_into().unwrap()) % HASH_MOD as u64,
    )
}

/// MinHash signature: for each of `NUM_HASHES` linear combinations of the two base hashes, the minimum over all
/// shingles. A smaller sentinel (`u64::MAX`) means "no shingle yet".
fn min_hash_signature(text: &str) -> [u64; NUM_HASHES] {
    let mut sig = [u64::MAX; NUM_HASHES];
    let shingles = word_shingles(&tokenize_words(text));
    if shingles.is_empty() {
        return sig;
    }
    for (k, slot) in sig.iter_mut().enumerate() {
        for s in &shingles {
            let (h1, h2) = double_hash(s);
            // g(x) = h1 + k*h2 (mod HASH_MOD), kept in u128 to avoid overflow.
            let v = ((h1 as u128) + (k as u128) * (h2 as u128)) % HASH_MOD;
            if v < *slot as u128 {
                *slot = v as u64;
            }
        }
    }
    sig
}

/// MinHash Jaccard estimate: the fraction of the 128 minimum hashes that are equal in the two signatures.
fn jaccard_estimate(a: &[u64; NUM_HASHES], b: &[u64; NUM_HASHES]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / NUM_HASHES as f64
}

/// Return the indices to drop — later texts that are near-duplicates (MinHash Jaccard ≥ `threshold`) of an earlier,
/// kept text. The first of each duplicate group is always kept.
pub fn near_duplicates(texts: &[String], threshold: f64) -> Vec<usize> {
    let sigs: Vec<[u64; NUM_HASHES]> = texts.iter().map(|s| min_hash_signature(s)).collect();
    let mut drop = vec![false; texts.len()];
    for (i, si) in sigs.iter().enumerate() {
        if drop[i] {
            continue; // already dropped; only keep the first of a cluster as the representative.
        }
        for (j, sj) in sigs.iter().enumerate().skip(i + 1) {
            if !drop[j] && jaccard_estimate(si, sj) >= threshold {
                drop[j] = true;
            }
        }
    }
    drop.into_iter().enumerate().filter(|(_, d)| *d).map(|(i, _)| i).collect()
}

// ---------------------------------------------------------------------------
// filter: drop a chat record for a reason, or keep it (None)
// ---------------------------------------------------------------------------

/// A single chat exchange — ordered `(role, content)` messages.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub messages: Vec<(String, String)>,
}

/// Length bounds used by [`filter`]. Refusal phrases are a fixed builtin list.
#[derive(Debug, Clone, Copy)]
pub struct Rules {
    /// Minimum accepted answer length in characters.
    pub min_chars: usize,
    /// Maximum accepted answer length in characters.
    pub max_chars: usize,
}

impl Default for Rules {
    fn default() -> Self {
        Rules { min_chars: 10, max_chars: 8192 }
    }
}

/// Refusal phrases (compared case-insensitively, as substrings of the answer).
const REFUSAL_PHRASES: &[&str] = &[
    "i can't", "i cannot", "as an ai", "as an artificial intelligence", "i am sorry i", "sorry, i",
    "i am unable to", "cannot fulfill", "i apologize but", "as a language model", "however, i",
];

fn answer_of(record: &Record) -> Option<&str> {
    record.messages.iter().rev().find(|(role, _)| role.eq_ignore_ascii_case("assistant")).map(|(_, c)| c.as_str())
}

fn question_of(record: &Record) -> Option<&str> {
    record.messages.iter().rev().find(|(role, _)| role.eq_ignore_ascii_case("user") || role.eq_ignore_ascii_case("human")).map(|(_, c)| c.as_str())
}

/// Return the reason a chat record should be dropped, or `None` to keep it.
pub fn filter(record: &Record, rules: &Rules) -> Option<&'static str> {
    let answer = match answer_of(record) {
        Some(a) => a,
        None => return None,
    };
    let trimmed = answer.trim();

    // An answer with no content.
    if trimmed.is_empty() {
        return Some("empty_answer");
    }
    // A canned refusal.
    let lower = answer.to_lowercase();
    if REFUSAL_PHRASES.iter().any(|p| lower.contains(p)) {
        return Some("refusal");
    }
    // Out of bounds length.
    if trimmed.chars().count() < rules.min_chars {
        return Some("too_short");
    }
    if trimmed.chars().count() > rules.max_chars {
        return Some("too_long");
    }
    // The answer just echoes the question back.
    if let Some(q) = question_of(record) {
        if q.trim() == trimmed {
            return Some("answer_equals_question");
        }
    }

    None
}

// ---------------------------------------------------------------------------
// read_jsonl — load chat records, reporting bad lines instead of panicking
// ---------------------------------------------------------------------------

/// Load a JSONL dataset into [`Record`]s. Each line is either `{"messages":[{role,content},...]}` or
/// `{"prompt","completion"}` (expanded into a user + assistant pair). Malformed lines are collected with their line
/// number and a human-readable reason rather than aborting the read.
pub fn read_jsonl(path: &std::path::Path) -> anyhow::Result<(Vec<Record>, Vec<(usize, String)>)> {
    let text = std::fs::read_to_string(path)?;
    let mut records = Vec::new();
    let mut bad: Vec<(usize, String)> = Vec::new();

    for (i, line) in text.lines().enumerate() {
        let lineno = i + 1;
        if line.trim().is_empty() {
            continue; // ignore blank lines, the usual JSONL convention.
        }
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                bad.push((lineno, format!("invalid JSON: {e}")));
                continue;
            }
        };

        match parse_line(&value) {
            Ok(Some(record)) => records.push(record),
            Ok(None) => {}
            Err(why) => bad.push((lineno, why.to_string())),
        }
    }

    Ok((records, bad))
}

/// Parse one decoded JSON object into a [`Record`], or return a reason string if the line's schema is unusable.
fn parse_line(value: &serde_json::Value) -> anyhow::Result<Option<Record>> {
    let obj = value.as_object().ok_or_else(|| anyhow::anyhow!("record is not a JSON object"))?;

    // `{"messages":[...]}` form.
    if let Some(msgs) = obj.get("messages").and_then(|m| m.as_array()) {
        let mut messages = Vec::with_capacity(msgs.len());
        for (mi, m) in msgs.iter().enumerate() {
            let role = m.get("role").and_then(|r| r.as_str()).ok_or_else(|| anyhow::anyhow!("message #{mi} missing string 'role'"))?;
            let content = m.get("content").and_then(|c| c.as_str()).ok_or_else(|| anyhow::anyhow!("message #{mi} missing string 'content'"))?;
            messages.push((role.to_string(), content.to_string()));
        }
        return Ok(Some(Record { messages }));
    }

    // `{"prompt","completion"}` form.
    if let (Some(prompt), Some(completion)) = (obj.get("prompt"), obj.get("completion")) {
        if let (Some(p), Some(c)) = (prompt.as_str(), completion.as_str()) {
            return Ok(Some(Record {
                messages: vec![
                    ("user".to_string(), p.to_string()),
                    ("assistant".to_string(), c.to_string()),
                ],
            }));
        }
        return Err(anyhow::anyhow!("record has prompt/completion but they are not strings"));
    }

    Err(anyhow::anyhow!("record is missing both 'messages' and 'prompt'/'completion'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn by_offset(text: &str, c: &Chunk) -> bool {
        text.chars().skip(c.start).take(c.end - c.start).collect::<String>() == c.text
    }

    #[test]
    fn offsets_map_back_to_input() {
        let text = "Hello world. This is a sentence. And another one.";
        let chunks = chunk(text, 20, 3);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(by_offset(text, c), "{:?} does not map back", c);
        }
    }

    #[test]
    fn no_chunk_exceeds_limit() {
        let text = "The quick brown fox jumps over the lazy dog. Again and again.";
        for c in chunk(text, 15, 2) {
            assert!(c.text.chars().count() <= 15, "chunk {} > 15", c.text);
        }
    }

    #[test]
    fn long_word_is_its_own_chunk_not_dropped() {
        let text = "ab cd SUPERLONGWORD ef gh";
        let chunks = chunk(text, 8, 0);
        assert!(chunks.iter().any(|c| c.text == "SUPERLONGWORD"), "long word dropped: {chunks:?}");
        // The long word must appear verbatim with offsets that map back.
        let w = chunks.iter().find(|c| c.text == "SUPERLONGWORD").unwrap();
        assert!(by_offset(text, w));
    }

    #[test]
    fn overlap_repeats_tail_of_previous() {
        let text = "Alpha beta gamma delta epsilon.";
        let chunks = chunk(text, 12, 4);
        if chunks.len() >= 2 {
            let (a, b) = (&chunks[0], &chunks[1]);
            let overlap_len = 4.min(a.text.chars().count());
            let a_tail: String = a.text.chars().skip(a.text.chars().count() - overlap_len).collect();
            let b_head: String = b.text.chars().take(overlap_len).collect();
            assert_eq!(a_tail, b_head, "overlap mismatch {a:?} {b:?}");
        }
    }

    #[test]
    fn empty_and_whitespace_only() {
        assert!(chunk("", 10, 2).is_empty());
        assert!(chunk("   \n  ", 10, 2).is_empty());
    }

    #[test]
    fn catches_identical_and_one_word_change_at_0_8() {
        // Identical pairs always match (Jaccard = 1.0).
        let a2 = "Quantum field theory unifies special relativity and mechanics".to_string();
        let b2 = "Quantum field theory unifies special relativity and mechanics".to_string();
        assert_eq!(near_duplicates(&[a2.clone(), b2.clone()], 0.8), vec![1]);

        // On long text a single changed word alters only a small fraction of the 5-gram shingles, so MinHash still
        // estimates Jaccard ≥ threshold and flags it as a near-duplicate.
        let long = "Neural networks learn hierarchical representations from large corpora by optimising a loss function \
                    through gradient descent while regularisation techniques help prevent overfitting on the held out set, so \
                    practitioners monitor validation metrics and adjust the learning rate whenever generalisation gaps appear \
                    between training and evaluation runs across many distributed devices in a modern datacenter cluster";
        let mut changed: String = long.to_string();
        // Replace one interior word ("regularisation") with a different but plausible word.
        if let Some(pos) = changed.find("regularisation") {
            changed.replace_range(pos..pos + "regularisation".len(), "normalisation");
        }
        assert_eq!(near_duplicates(&[long.to_string(), changed], 0.8), vec![1]);
    }

    #[test]
    fn does_not_catch_unrelated_texts() {
        let a = "The quick brown fox jumps over the lazy dog".to_string();
        let b = "Bananas orchestrate quantum gravity beneath silent oceans".to_string();
        let drop = near_duplicates(&[a, b], 0.8);
        assert!(drop.is_empty(), "unrelated should not be dropped: {drop:?}");
    }

    #[test]
    fn keeps_first_of_cluster() {
        let a = "apple banana cherry date fig grape".to_string();
        let b = "apple banana cherry date fig grape".to_string();
        let c = "apple banana cherry date fig grape".to_string();
        assert_eq!(near_duplicates(&[a, b, c], 0.9), vec![1, 2]);
    }

    fn rec(pairs: &[(&str, &str)]) -> Record {
        Record { messages: pairs.iter().map(|(r, c)| (r.to_string(), c.to_string())).collect() }
    }

    #[test]
    fn keeps_good_record() {
        let rules = Rules::default();
        let r = rec(&[("user", "What is 2+2?"), ("assistant", "It is four.")]);
        assert_eq!(filter(&r, &rules), None);
    }

    #[test]
    fn drops_too_short() {
        let rules = Rules { min_chars: 10, max_chars: 8192 };
        let r = rec(&[("user", "Explain gravity."), ("assistant", "Yes.")]);
        assert_eq!(filter(&r, &rules), Some("too_short"));
    }

    #[test]
    fn drops_too_long() {
        let rules = Rules { min_chars: 10, max_chars: 20 };
        let long = "a".repeat(50);
        let r = rec(&[("user", "q?"), ("assistant", long.as_str())]);
        assert_eq!(filter(&r, &rules), Some("too_long"));
    }

    #[test]
    fn drops_refusal() {
        let rules = Rules::default();
        for phrase in ["I can't help with that.", "As an AI, I don't know.", "I cannot do that."] {
            let r = rec(&[("user", "do X"), ("assistant", phrase)]);
            assert_eq!(filter(&r, &rules), Some("refusal"), "failed for {phrase:?}");
        }
    }

    #[test]
    fn drops_empty_answer() {
        let rules = Rules::default();
        let r = rec(&[("user", "hello?"), ("assistant", "   \n  ")]);
        assert_eq!(filter(&r, &rules), Some("empty_answer"));
    }

    #[test]
    fn drops_answer_equal_to_question() {
        let rules = Rules::default();
        let r = rec(&[("user", "What is the capital of France?"), ("assistant", "What is the capital of France?")]);
        assert_eq!(filter(&r, &rules), Some("answer_equals_question"));
    }

    #[test]
    fn reads_jsonl_mixed_good_and_bad() {
        use std::io::Write;
        let mut f = std::fs::File::create(std::env::temp_dir().join("strata-dataset-test.jsonl")).unwrap();
        writeln!(f, r#"{{"messages":[{{"role":"user","content":"hi"}},{{"role":"assistant","content":"hello there"}}]}}"#).unwrap();
        writeln!(f, r#"{{"prompt":"q1","completion":"a1"}}"#).unwrap();
        writeln!(f, "{{ not valid json").unwrap();
        writeln!(f, r#"{{"messages":[{{"role":"user"}}]}}"#).unwrap(); // missing content
        writeln!(f, r#"{{"unknown":"schema"}}"#).unwrap();
        f.flush().unwrap();
        let path = std::env::temp_dir().join("strata-dataset-test.jsonl");

        let (records, bad) = read_jsonl(&path).expect("should not panic");

        // Two good records: one messages-form, one prompt/completion-form.
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].messages, vec![("user".to_string(), "hi".to_string()), ("assistant".to_string(), "hello there".to_string())]);
        assert_eq!(records[1].messages, vec![("user".to_string(), "q1".to_string()), ("assistant".to_string(), "a1".to_string())]);

        // Three bad lines, reported with their line numbers and a reason.
        assert_eq!(bad.len(), 3);
        let nums: Vec<usize> = bad.iter().map(|(n, _)| *n).collect();
        assert!(nums.contains(&3) && nums.contains(&4) && nums.contains(&5), "line numbers: {nums:?}");
        assert!(bad.iter().all(|(_, why)| !why.is_empty()));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_overlap_as_long_as_a_chunk_still_finishes() {
        // Kilo's case: a short first sentence, then an overlap longer than that chunk
        let short = format!("Hi. {}", "one two three four five six seven eight nine ten ".repeat(5));
        for overlap in [5, 20, 100] {
            let cs = chunk(&short, 15, overlap);
            assert!(!cs.is_empty() && cs.len() < 200, "overlap {overlap}: {} chunks", cs.len());
            assert!(cs.windows(2).all(|w| w[1].start > w[0].start), "every chunk moves forward");
            assert_eq!(cs.last().unwrap().end, short.trim_end().chars().count(), "the text is covered to its end");
        }
        let text = "one two three four five six seven eight nine ten ".repeat(20);
        for overlap in [10, 15, 100] {
            let cs = chunk(&text, 15, overlap);
            assert!(!cs.is_empty() && cs.len() < 400, "overlap {overlap}: {} chunks", cs.len());
            assert!(cs.windows(2).all(|w| w[1].start > w[0].start), "every chunk moves forward");
        }
    }
}
