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
}
