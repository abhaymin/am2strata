//! Port of gguf-py's `Metadata.get_model_id_components` (gguf/naming_convention.py). llama.cpp's converter names a model's parts from its folder/id: `general.basename`, `general.finetune`, `general.version` (and, derived only from the parameter count, `general.size_label`), so it can write them as real GGUF metadata keys instead of title-casing the whole folder into a single `general.name`. This module is that parsing: given a model id like `meta-llama/Llama-3.1-8B-Instruct` and the parameter count, it recovers the org, basename, finetune, version and size label via the same four hand-written regexes (gguf-py's `re.fullmatch` patterns with `re.IGNORECASE`) and the same bloom-7b1 / context-length normalisation. The fixtures in tests/fixtures/model_id_components.json pin all 60 reference cases exercised by [`model_id_components`].

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

/// A model's name parts, as pulled from its HF folder/id by gguf-py (`get_model_id_components`) — `None` where the id carries no such part (e.g. a bare org-less `tiny-llama-hf` yields a basename of `None`, meaning llama.cpp falls back to title-casing the whole folder for `general.name`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Components {
    /// The full model name (the part after `<org>/`, or the whole id when there is no org), verbatim — what gguf-py returns as its first element and what llama.cpp title-cases into `general.name` when it has no basename.
    pub name: Option<String>,
    /// The HuggingFace org component (`<org>/<name>`), or `None` when absent / a dotted `./foo` false match.
    pub org: Option<String>,
    /// The base model name (leading, non-version parts), joined by `-`; `None` when ambiguous.
    pub basename: Option<String>,
    /// Finetune / task tag (`chat`, `instruct`, `lora`, a context length, ...), joined by `-`.
    pub finetune: Option<String>,
    /// Version part (`v0.3`, `2507`, ...), joined by `-`; excludes any part also tagged as basename.
    pub version: Option<String>,
    /// Model size tag (`8B`, `8x7B`, ...), joined by `-` (order-preserving, deduplicated).
    pub size_label: Option<String>,
}

impl Components {
    /// The `' ' in model_id` early return: the id is a free-form sentence, so the whole thing is the name and nothing else parses out.
    fn name_only(model_id: &str) -> Self {
        Components {
            name: Some(model_id.to_string()),
            org: None,
            basename: None,
            finetune: None,
            version: None,
            size_label: None,
        }
    }
}

fn rx(src: &str) -> Regex {
    Regex::new(src).expect("naming regex compiles")
}

/// gguf-py's four `re.fullmatch` patterns (`re.IGNORECASE`) for annotating each dash-separated part. Each alternation is grouped so a whole-string match means one alternative spans the entire token, exactly like Python's fullmatch.
static RE_VERSION: LazyLock<Regex> = LazyLock::new(|| rx(r"(?i)(?:v|iter)?\d+(?:\.\d+)*"));
static RE_TYPE: LazyLock<Regex> = LazyLock::new(|| rx(r"(?i)(?:i?q\d(?:_\w)*|b?fp?(?:16|32))"));
static RE_SIZE: LazyLock<Regex> = LazyLock::new(|| rx(r"(?i)(?:([A]|\d+[x])?\d+(?:[._]\d+)?[KMBT][\d]?|small|mini|medium|large|x?xl)"));
static RE_FINETUNE: LazyLock<Regex> = LazyLock::new(|| rx(r"(?i)(?:chat|instruct|vision|lora)"));

/// Whole-string match, emulating Python's `re.fullmatch`: the match must start at 0 and end at the string's length.
fn full(rx: &Regex, s: &str) -> bool {
    match rx.find(s) {
        Some(m) => m.start() == 0 && m.end() == s.len(),
        None => false,
    }
}

/// String with its last character removed (`part[:-1]`).
fn chop(s: &str) -> String {
    let mut it = s.chars();
    it.next_back();
    it.collect()
}

/// gguf-py's `" KMBT".find(part[-1])`: the magnitude index of a size suffix letter (`K`=1, `M`=2, `B`=3, `T`=4).
fn suffix_index(c: char) -> i32 {
    match c { 'K' => 1, 'M' => 2, 'B' => 3, 'T' => 4, _ => -1 }
}

/// Split on a separator, drop empty parts, and rejoin with `joined` (used for gguf-py's `replace('_', '.')`).
fn split_join(sep: char, joined: char, s: &str) -> String {
    s.split(sep)
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join(&joined.to_string())
}

/// A single dash-separated part's size label, with gguf-py's bloom-7b1 and context-length normalisation applied. The `total_params` (`<0` for a LoRA adapter) decides whether the suffix is instead treated as a context length (lowered).
fn normalize_size(part: &mut String, name_types: &mut HashSet<&'static str>, total_params: i64) {
    *part = split_join('_', '.', part);
    let mut s = part.chars().collect::<Vec<char>>();

    // Handle weird bloom-7b1 notation ("7b1" -> "7.1b"): a trailing digit becomes `<head>.<last><second_last>`.
    if let Some(&last) = s.last() {
        if last.is_ascii_digit() {
            let second_last = s[s.len() - 2];
            let head: String = s[..s.len() - 2].iter().collect();
            // part[:-2] + "." + part[-1] + part[-2]
            let mut rebuilt = head;
            rebuilt.push('.');
            rebuilt.push(last);
            rebuilt.push(second_last);
            *part = rebuilt;
            s = part.chars().collect::<Vec<char>>();
        }
    }

    // Normalize the size suffixes: a lowercase `k/m/b/t` preceded by a digit is upper-cased (part[:-1] + part[-1].upper()).
    if s.len() > 1 && s[s.len() - 2].is_ascii_digit() {
        let lastc = s[s.len() - 1];
        if matches!(lastc, 'k' | 'm' | 'b' | 't') {
            *part = chop(part);
            part.push(lastc.to_ascii_uppercase());
        }
    }

    // Decide between size label and context length from the parameter count (gguf-py's try/except ValueError pass).
    if total_params != 0 {
        if let Ok(label_params) = chop(part).parse::<f64>() {
            let lastc = part.chars().last().expect("a non-empty suffix");
            let label = label_params * 1000f64.powi(suffix_index(lastc));
            let is_context_length = (total_params < 0 && label < (total_params.unsigned_abs() / 8) as f64)
                || (total_params > 0 && (label - total_params as f64).abs() > ((7 * total_params) / 8) as f64);
            if is_context_length {
                name_types.insert("finetune");
                // Lowercase the size when it's a context length ("30B" -> "30b"): keep part[:-1], lowercase part[-1].
                let lc = part.chars().last().expect("a suffix char").to_ascii_lowercase();
                *part = chop(part);
                part.push(lc);
            }
        }
    }

    if name_types.is_empty() {
        name_types.insert("size_label");
    }
}

fn joined<'a>(parts: &'a [String], types: &'a [HashSet<&'static str>], sel: &'static str) -> Option<String> {
    let v: Vec<&str> = parts.iter().zip(types).filter(|(_, t)| t.contains(sel)).map(|(p, _)| p.as_str()).collect();
    if v.is_empty() { None } else { Some(v.join("-")) }
}

fn joined_filter<'a>(parts: &'a [String], types: &'a [HashSet<&'static str>], sel: impl Fn(&HashSet<&'static str>) -> bool) -> Option<String> {
    let v: Vec<&str> = parts.iter().zip(types).filter(|(_, t)| sel(t)).map(|(p, _)| p.as_str()).collect();
    if v.is_empty() { None } else { Some(v.join("-")) }
}

/// Join size labels preserving part order and deduplicating (gguf-py's `dict.fromkeys`).
fn deduped_join(parts: &[String], types: &[HashSet<&'static str>]) -> Option<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (p, t) in parts.iter().zip(types) {
        if t.contains("size_label") && seen.insert(p.as_str()) {
            out.push(p.as_str());
        }
    }
    if out.is_empty() { None } else { Some(out.join("-")) }
}

/// Port of gguf-py's `get_model_id_components`: parse `model_id` and the total parameter count into a model's name parts.
pub fn model_id_components(model_id: &str, total_params: i64) -> Components {
    // A free-form sentence id has no org/size/version structure — the whole thing is just the name.
    if model_id.contains(' ') {
        return Components::name_only(model_id);
    }

    // '<org>/<name>' style ids split on the first slash; a missing org leaves the whole id as the name.
    let (org, full_name) = match model_id.find('/') {
        Some(slash) => (&model_id[..slash], &model_id[slash + 1..]),
        None => ("", model_id),
    };
    // Avoid an accidental './foo' / '../foo' org.
    let org = if org.is_empty() || org.starts_with('.') { None } else { Some(org.to_string()) };

    // Dash-split the name, dropping leading/trailing/consecutive-dash empty parts.
    let mut name_parts: Vec<String> = full_name.split('-').filter(|p| !p.is_empty()).map(|p| p.to_string()).collect();
    // Guard against a single empty token from an all-dash name.
    if name_parts.len() == 1 && name_parts[0].is_empty() {
        name_parts.pop();
    }
    let mut name_types: Vec<HashSet<&'static str>> = vec![HashSet::new(); name_parts.len()];

    // Annotate each part (version / quant type / size label / finetune tag), mirroring gguf-py's loop.
    for i in 0..name_parts.len() {
        let part = &name_parts[i];
        let t = &mut name_types[i];
        if full(&RE_VERSION, part) {
            t.insert("version");
        } else if full(&RE_TYPE, part) {
            name_parts[i] = part.to_uppercase();
            t.insert("type");
        } else if i > 0 && full(&RE_SIZE, part) {
            normalize_size(&mut name_parts[i], t, total_params);
        } else if i > 0 && full(&RE_FINETUNE, part) {
            if total_params < 0 && part.eq_ignore_ascii_case("lora") {
                // Redundant "lora" in a LoRA adapter's finetune is re-tagged as a quant type, not a finetune.
                t.insert("type");
            } else {
                t.insert("finetune");
            }
        }
    }

    // Drop word-based size labels when any number-based size label is present (gguf-py's "ignore word-based sizes" step).
    if name_types.iter().zip(&name_parts).any(|(t, n)| t.contains("size_label") && n.chars().any(|c| c.is_ascii_digit())) {
        for (t, n) in name_types.iter_mut().zip(&name_parts) {
            if t.contains("size_label") && n.chars().all(|c| c.is_alphabetic()) {
                t.remove("size_label");
            }
        }
    }

    // The basename is the run of leading alpha (or version) parts; later untagged parts become finetune.
    let mut at_start = true;
    for i in 0..name_parts.len() {
        let (part, t) = (&name_parts[i], &mut name_types[i]);
        if at_start && ((t.is_empty() && part.chars().next().is_some_and(|c| c.is_alphabetic())) || t.contains("version")) {
            t.insert("basename");
        } else {
            if at_start {
                at_start = false;
            }
            if t.is_empty() {
                t.insert("finetune");
            }
        }
    }

    // Strip the basename annotation off any trailing version (a version ending the name is not part of the basename).
    for i in (0..name_parts.len()).rev() {
        let t = &mut name_types[i];
        if t.contains("basename") && t.len() > 1 {
            t.remove("basename");
        } else {
            break;
        }
    }

    // Build the outputs (order-preserving), then drop the basename when nothing else parsed out — too ambiguous to name.
    let basename = joined(&name_parts, &name_types, "basename");
    let finetune = joined(&name_parts, &name_types, "finetune");
    let version = joined_filter(&name_parts, &name_types, |t| t.contains("version") && !t.contains("basename"));
    let size_label = deduped_join(&name_parts, &name_types);

    let ambiguous = size_label.is_none() && finetune.is_none() && version.is_none();
    Components {
        name: Some(full_name.to_string()),
        org,
        basename: if ambiguous { None } else { basename },
        finetune,
        version,
        size_label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn matches_the_60_reference_cases() {
        // The reviewer ran the Python port over 60 model ids and recorded each (model_id, total_params) plus its expected basename/finetune/version/size_label/org/name in this fixture.
        let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model_id_components.json");
        let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&fx).unwrap()).expect("fixture is JSON");
        let arr = doc.as_array().expect("fixture is a JSON array");
        assert_eq!(arr.len(), 60, "the fixture must hold all 60 reference cases");

        for (idx, entry) in arr.iter().enumerate() {
            let model_id = entry["model_id"].as_str().unwrap();
            let total_params: i64 = entry["total_params"].as_i64().unwrap();
            let got = model_id_components(model_id, total_params);
            let expect = Components {
                name: entry["name"].as_str().map(str::to_string),
                org: entry["org"].as_str().map(str::to_string),
                basename: entry["basename"].as_str().map(str::to_string),
                finetune: entry["finetune"].as_str().map(str::to_string),
                version: entry["version"].as_str().map(str::to_string),
                size_label: entry["size_label"].as_str().map(str::to_string),
            };
            assert_eq!(got, expect, "case #{idx}: {model_id:?} total_params={total_params}");
        }
    }

    #[test]
    fn a_space_makes_the_whole_id_the_name() {
        let c = model_id_components("my little model", 0);
        assert_eq!(c.name.as_deref(), Some("my little model"));
        assert_eq!(c.org, None);
        assert_eq!(c.basename, None);
    }

    #[test]
    fn dotted_org_is_ignored() {
        // "./foo/Bar-8B" -> the first slash yields org "." (dropped as a path artifact); only the name remains, with its
        // own basename/size still parsed out.
        let c = model_id_components("./foo/Bar-8B", 0);
        assert_eq!(c.org, None);
        assert_eq!(c.name.as_deref(), Some("foo/Bar-8B"));
        assert_eq!(c.basename.as_deref(), Some("foo/Bar"));
        assert_eq!(c.size_label.as_deref(), Some("8B"));
    }

    #[test]
    fn an_unsuffixed_token_is_not_a_size_label() {
        // "16_k" does not match the size pattern ([KMBT] is unreachable past the underscore), so it stays untagged and
        // falls to the finetune bucket; its name segment is left verbatim.
        let c = model_id_components("foo-16_k", 0);
        assert_eq!(c.name.as_deref(), Some("foo-16_k"));
        assert_eq!(c.finetune.as_deref(), Some("16_k"));
    }
}
