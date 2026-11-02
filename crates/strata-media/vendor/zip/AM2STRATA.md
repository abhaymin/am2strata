# zip 9.2.2, labelled 8.99.0

The `zip` crate, version 9.2.2 from crates.io, sha256 `e408c6cc4699dd017d53dc080cd51c2cea2b5354af57a1479bd265ed42911487`, MIT.
Only `src/`, `LICENSE`, `README.md` and `Cargo.toml` are kept. Two things differ from the published crate, both in `Cargo.toml`:
`version`, and the removed example, test and bench entries.

Two API shims put zip 8's signatures back, because both forks call them: `ZipFile::name()` and `ZipArchive::file_names()` return
`&str`. zip 9's versions are kept as `decoded_name()` and `decoded_file_names()`. Neither touches header parsing or the fix.
Because of the shims and the missing `tests/data`, zip's own unit tests do not build here.

**Why:** candle-core-mold and mold-ai-inference, both pinned git forks, require `zip ^8.6`, and every 8.x is affected by
CVE-2026-115905 (GHSA-m665-v7pm-7grj, critical, fixed in 9.0.5), as well as CVE-2026-115906 to 115909 (fixed in 9.0.0).
`[patch.crates-io]` only accepts a version the requirement allows, hence the label.

**Remove it** when both forks require `zip` 9.0.5 or newer: delete this folder and its `[patch.crates-io]` line, then run
`cargo update -p zip`.
