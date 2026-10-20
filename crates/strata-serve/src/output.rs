//! Splits a streamed answer into reasoning (`<think>...</think>`) and content without ever emitting part of a tag.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    /// Before any output: a reply may open with <think> itself.
    Start,
    Reasoning,
    Content,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Delta {
    pub reasoning: String,
    pub content: String,
}

impl Delta {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty()
    }
}

pub struct OutputParser {
    state: State,
    buf: String,
    /// content right after </think>: its leading blank lines are dropped
    content_started: bool,
}

/// Length of the longest suffix of `s` that is a proper prefix of `tag` (bytes; tags are ASCII).
fn partial_suffix(s: &str, tag: &str) -> usize {
    (1..tag.len().min(s.len() + 1)).rev().find(|&n| s.is_char_boundary(s.len() - n) && tag.starts_with(&s[s.len() - n..])).unwrap_or(0)
}

impl OutputParser {
    /// `in_reasoning`: the rendered prompt left a `<think>` open, so the reply starts inside the reasoning.
    pub fn new(in_reasoning: bool) -> Self {
        Self { state: if in_reasoning { State::Reasoning } else { State::Start }, buf: String::new(), content_started: false }
    }

    pub fn push(&mut self, text: &str) -> Delta {
        self.buf.push_str(text);
        let mut d = Delta::default();
        loop {
            match self.state {
                State::Start => {
                    let trimmed = self.buf.trim_start();
                    if trimmed.starts_with(OPEN) {
                        let at = self.buf.len() - trimmed.len() + OPEN.len();
                        self.buf.drain(..at);
                        self.state = State::Reasoning;
                        continue;
                    }
                    if !trimmed.is_empty() && OPEN.starts_with(trimmed) {
                        return d; // could still become <think>
                    }
                    if trimmed.is_empty() {
                        return d;
                    }
                    self.state = State::Content;
                }
                State::Reasoning => {
                    if let Some(i) = self.buf.find(CLOSE) {
                        d.reasoning.push_str(&self.buf[..i]);
                        self.buf.drain(..i + CLOSE.len());
                        self.state = State::Content;
                        continue;
                    }
                    let keep = partial_suffix(&self.buf, CLOSE);
                    let emit = self.buf.len() - keep;
                    d.reasoning.push_str(&self.buf[..emit]);
                    self.buf.drain(..emit);
                    return d;
                }
                State::Content => {
                    if !self.content_started {
                        let t = self.buf.trim_start_matches(['\n', '\r']);
                        if t.is_empty() {
                            self.buf.clear();
                            return d;
                        }
                        let cut = self.buf.len() - t.len();
                        self.buf.drain(..cut);
                        self.content_started = true;
                    }
                    d.content.push_str(&self.buf);
                    self.buf.clear();
                    return d;
                }
            }
        }
    }

    /// The text held back at the end (a partial tag that never completed).
    pub fn finish(&mut self) -> Delta {
        let rest = std::mem::take(&mut self.buf);
        match self.state {
            State::Reasoning => Delta { reasoning: rest, ..Default::default() },
            _ => Delta { content: rest.trim_start_matches(['\n', '\r']).to_string(), ..Default::default() },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(in_reasoning: bool, pieces: &[&str]) -> Delta {
        let mut p = OutputParser::new(in_reasoning);
        let mut all = Delta::default();
        for s in pieces {
            let d = p.push(s);
            all.reasoning += &d.reasoning;
            all.content += &d.content;
        }
        let d = p.finish();
        all.reasoning += &d.reasoning;
        all.content += &d.content;
        all
    }

    #[test]
    fn splits_reasoning_across_split_tags() {
        let d = run(true, &["Let me", " think.</th", "ink>\n\nThe answer", " is 4."]);
        assert_eq!(d.reasoning, "Let me think.");
        assert_eq!(d.content, "The answer is 4.");
    }

    #[test]
    fn reply_that_opens_its_own_think() {
        let d = run(false, &["<thi", "nk>hmm</think>\n\nok"]);
        assert_eq!((d.reasoning.as_str(), d.content.as_str()), ("hmm", "ok"));
    }

    #[test]
    fn plain_content() {
        let d = run(false, &["Hello", " world"]);
        assert_eq!((d.reasoning.as_str(), d.content.as_str()), ("", "Hello world"));
    }
}
