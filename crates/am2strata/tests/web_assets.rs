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

/// Read one web asset from the repo root (the crate is two levels below `web/`), like the checks above.
fn web_asset(path: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../web/{path}"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// The Tools tab is a fifth kind in Media Generation, its three cards built like the Speech to text card.
#[test]
fn tools_tab_has_the_three_media_cards() {
    let html = web_asset("index.html");
    assert!(html.contains(r#"id="tools-forms""#), "the Tools tab has a #tools-forms container");
    for id in ["check-form", "dub-form", "prod-form"] {
        assert!(html.contains(format!(r#"id="{id}""#).as_str()), "the Tools tab has an #{id} form");
    }
    assert!(html.contains(r#"data-kind="tools""#), "a data-kind=\"tools\" button shows the Tools tab");
}

/// The Tools cards talk to the media endpoints: each posts its action, and productions follow /jobs part by part.
#[test]
fn tools_cards_post_the_media_actions() {
    let js = web_asset("app.js");
    for url in ["v1/media/check", "v1/media/dub", "v1/media/productions", "v1/media/jobs/"] {
        assert!(js.contains(url), "app.js calls {url}");
    }
}

/// Settings › Clients: editing a key reuses the New-key form's fields and saves with POST /api/keys/{id}.
#[test]
fn editing_a_key_reuses_the_new_form_and_saves_with_post() {
    // the edit opens the same fields as "New key", filled with this key's values.
    for id in ["key-name", "key-tools", "key-expires", "key-req-day", "key-tok-day", "key-models"] {
        let js = web_asset("api-keys.js");
        assert!(js.contains(format!("el(\"{id}\")").as_str()), "the edit fills #{id} like the New key form");
    }

    // ... and saves with POST to this key's id. The list also gains an Edit button (active keys only) whose click opens it.
    let js = web_asset("api-keys.js");
    assert!(js.contains("api/keys/${"), "the edit builds api/keys/{{id}}");
    assert!(js.contains("\"POST\""), "the edit posts to save the changed fields");
    assert!(js.contains("data-key-edit"), "each active key row shows an Edit button that opens the form");

    // index.html has those same fields in the Clients section (reused for editing, no separate form needed).
    let html = web_asset("index.html");
    for id in ["key-name", "key-tools", "key-expires", "key-req-day", "key-tok-day", "key-models"] {
        assert!(html.contains(format!("id=\"{id}\"").as_str()), "the reused edit form has #{id}");
    }
}
