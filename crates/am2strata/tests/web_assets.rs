//! web assets: the client JS this box serves, checked here because it has no node to parse it (item 1). A duplicate top-level
//! declaration such as `const COMPUTE_KEYS` is a SyntaxError that stops the whole app from loading, so this fails if any name
//! written after `const`, `let` or `function` at the start of a line appears more than once.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// The top-level declarations in one JS source: every `const NAME`, `let NAME` and `function NAME` at the start of a line,
/// each as "keyword name" so the same thing written twice (like COMPUTE_KEYS) is caught.
fn top_level_names(js: &str) -> Vec<String> {
    // (?m) makes ^ match each line's start; group 1 is the keyword, group 2 the declared name — together they identify one declaration.
    let re = regex::Regex::new(r"(?m)^(const|let|function)\s+([A-Za-z_$][\w$]*)").expect("valid regex");
    re.captures_iter(js).map(|c| format!("{} {}", c.get(1).unwrap().as_str(), c.get(2).unwrap().as_str())).collect()
}

/// Read `web/app.js` from the repo root (the tests' CARGO_MANIFEST_DIR is the crate folder, two levels below it) and check its
/// top-level declarations for duplicates; a duplicate is what stopped the web app from loading in the first place.
fn assert_no_duplicate_declarations() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web/app.js");
    let js = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    let mut seen = BTreeSet::new();
    let mut dupes = Vec::new();
    for decl in top_level_names(&js) {
        if !seen.insert(decl.clone()) { dupes.push(decl); }
    }
    assert!(dupes.is_empty(), "top-level names declared twice: {}", dupes.join(", "));
}

/// No top-level `const`/`let`/`function` name is declared twice, so the client JS loads as one unit.
#[test]
fn web_app_has_no_duplicate_top_level_declaration() {
    assert_no_duplicate_declarations();
}

/// The specific regression guard: COMPUTE_KEYS used to be written at the top of two sections and that SyntaxError loaded nothing.
#[test]
fn compute_keys_is_declared_once() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web/app.js");
    let js = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    const pattern: &str = "const COMPUTE_KEYS";
    let count = js.lines().filter(|l| l.trim_start().starts_with(pattern)).count();
    assert_eq!(count, 1, "COMPUTE_KEYS must be declared once");
}
