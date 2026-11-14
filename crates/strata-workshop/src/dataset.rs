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

    // Tokenize into words with the whitespace run that follows each word. A `Token` spans `[start, end)` in char space;
    // `ws_after` holds the length (in chars) of trailing whitespace, and `paragraph` is true when that whitespace breaks
    // to a new line. Sentence-end detection keys off the last visible character of the word.
    struct Token {
        start: usize,
        end: usize,
        ws_after: usize,
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
        let ws_after = i - ws_start;
        let word: &[char] = &chars[start..end];
        let sentence_end = ends_with_punct(word) || end == n;
        let paragraph = chars[ws_start..ws_start + ws_after]
            .iter()
            .any(|c| *c == '\n');
        tokens.push(Token { start, end, ws_after, paragraph, sentence_end });
    }

    let mut out: Vec<Chunk> = Vec::new();
    let mut prev_end: Option<usize> = None;
    let mut idx = 0usize;
    while idx < tokens.len() {
        // Choose this chunk's start: overlap into the previous chunk, or zero for the first one.
        let start = match prev_end {
            Some(pe) => pe.saturating_sub(overlap).min(n),
            None => 0,
        };

        // If the start lands inside a word, back up to that word's real start so `text[start..]` is meaningful; the lost
        // characters are just overlap noise anyway.
        let first = tokens.iter().position(|t| t.start >= start).unwrap_or(tokens.len() - 1);
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

/// MinHash Jaccard estimate: fraction of the 128 hash signs shared by the two signatures.
fn jaccard_estimate(a: &[u64; NUM_HASHES], b: &[u64; NUM_HASHES]) -> f64 {
    let mut matches = 0usize;
    for (x, y) in a.iter().zip(b) {
        if (x ^ y) & (u64::MAX >> 1) == 0 {
            matches += 1;
        }
    }
    matches as f64 / NUM_HASHES as f64
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
}

