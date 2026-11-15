//! Datasets: the records a browser job scrapes, surveys or catalogs, kept on this PC (`<data>/datasets/<name>.jsonl`, one
//! JSON object per line, plus `<name>.meta.json`). The model writes rows with `browser_extract {dataset}` and
//! `browser_save`; the Browser tab and `am2strata-run browser datasets` list, show, export (JSON, JSONL, CSV) and delete them.
//!
//! - **Names** are letters, digits, `-` and `_` (1–64), so a name can never reach outside the folder.
//! - **A key** (`"key": "url"`, or several fields joined) makes a dataset a catalog: a row whose key is already there
//!   replaces the old one instead of adding a copy, so a job run twice keeps one record per item.
//! - **Columns** are the union of every row's fields in first-seen order, for CSV and the table view.
//! - Each row gets `_saved` (UTC seconds) and, when the job gave one, `_source` (the page it came from).

use serde_json::{json, Map, Value};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The most rows one dataset holds (a runaway scraping loop stops there, with a message).
pub const MAX_ROWS: usize = 200_000;

pub fn dir(data_dir: &Path) -> PathBuf { data_dir.join("datasets") }

/// A safe dataset name, or why not.
pub fn check_name(name: &str) -> Result<&str, String> {
    let ok = !name.is_empty() && name.len() <= 64 && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if ok { Ok(name) } else { Err(format!("{name:?} is not a dataset name: use 1-64 letters, digits, - or _")) }
}

fn rows_path(data_dir: &Path, name: &str) -> PathBuf { dir(data_dir).join(format!("{name}.jsonl")) }
fn meta_path(data_dir: &Path, name: &str) -> PathBuf { dir(data_dir).join(format!("{name}.meta.json")) }

/// The key of a row for a catalog's de-duplication: the key fields' values joined, or None when a field is missing.
pub fn row_key(row: &Value, key: &[String]) -> Option<String> {
    if key.is_empty() { return None; }
    let mut parts = Vec::new();
    for k in key {
        let v = row.get(k)?;
        parts.push(match v { Value::String(s) => s.trim().to_string(), other => other.to_string() });
    }
    Some(parts.join("\u{1f}"))
}

/// Every row of a dataset (empty when it does not exist).
pub fn read(data_dir: &Path, name: &str) -> Result<Vec<Value>, String> {
    check_name(name)?;
    let text = match std::fs::read_to_string(rows_path(data_dir, name)) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("reading dataset {name}: {e}")),
    };
    Ok(text.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect())
}

/// The dataset's settings (its key and description), `{}` when none.
pub fn meta(data_dir: &Path, name: &str) -> Value {
    std::fs::read_to_string(meta_path(data_dir, name)).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_else(|| json!({}))
}

/// What a save did: rows added, rows that replaced an older one with the same key, rows skipped (not objects), and the total.
#[derive(Debug, PartialEq, Eq)]
pub struct Saved { pub added: usize, pub replaced: usize, pub skipped: usize, pub total: usize }

/// Add rows to a dataset (creating it), replacing rows with the same key when the dataset has one. `key` given here sets
/// the dataset's key (kept for later saves); `source` is the page they came from.
pub fn save(data_dir: &Path, name: &str, rows: &[Value], key: Option<&[String]>, source: Option<&str>, about: Option<&str>) -> Result<Saved, String> {
    check_name(name)?;
    std::fs::create_dir_all(dir(data_dir)).map_err(|e| format!("creating the datasets folder: {e}"))?;
    let mut m = meta(data_dir, name);
    if let Some(k) = key.filter(|k| !k.is_empty()) { m["key"] = json!(k); }
    if let Some(a) = about.filter(|a| !a.trim().is_empty()) { m["about"] = json!(a.trim()); }
    if m.get("created").is_none() { m["created"] = json!(now()); }
    m["updated"] = json!(now());
    let key: Vec<String> = m["key"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let mut all = read(data_dir, name)?;
    let mut index: std::collections::HashMap<String, usize> = all.iter().enumerate()
        .filter_map(|(i, r)| row_key(r, &key).map(|k| (k, i))).collect();
    let (mut added, mut replaced, mut skipped) = (0, 0, 0);
    for r in rows {
        let Some(obj) = r.as_object() else { skipped += 1; continue };
        let mut row: Map<String, Value> = obj.iter().filter(|(k, _)| !k.starts_with('_')).map(|(k, v)| (k.clone(), v.clone())).collect();
        if row.is_empty() { skipped += 1; continue; }
        row.insert("_saved".into(), json!(now()));
        if let Some(s) = source.filter(|s| !s.is_empty()) { row.insert("_source".into(), json!(s)); }
        let row = Value::Object(row);
        match row_key(&row, &key).and_then(|k| index.get(&k).copied().map(|i| (k, i))) {
            Some((_, i)) => { all[i] = row; replaced += 1; }
            None => {
                if all.len() >= MAX_ROWS {
                    return Err(format!("dataset {name} is full ({MAX_ROWS} rows); start another one"));
                }
                if let Some(k) = row_key(&row, &key) { index.insert(k, all.len()); }
                all.push(row);
                added += 1;
            }
        }
    }
    write_all(data_dir, name, &all)?;
    m["rows"] = json!(all.len());
    std::fs::write(meta_path(data_dir, name), serde_json::to_vec_pretty(&m).unwrap()).map_err(|e| e.to_string())?;
    Ok(Saved { added, replaced, skipped, total: all.len() })
}

fn write_all(data_dir: &Path, name: &str, rows: &[Value]) -> Result<(), String> {
    let path = rows_path(data_dir, name);
    let tmp = path.with_extension("jsonl.tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp).map_err(|e| format!("writing dataset {name}: {e}"))?);
    for r in rows {
        serde_json::to_writer(&mut f, r).map_err(|e| e.to_string())?;
        f.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    f.flush().map_err(|e| e.to_string())?;
    drop(f);
    std::fs::rename(&tmp, &path).map_err(|e| format!("writing dataset {name}: {e}"))
}

/// Every dataset: name, rows, columns, key, about and when it changed.
pub fn list(data_dir: &Path) -> Vec<Value> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir(data_dir)).into_iter().flatten().flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        let Some(name) = n.strip_suffix(".jsonl") else { continue };
        if check_name(name).is_err() { continue; }
        let rows = read(data_dir, name).unwrap_or_default();
        let m = meta(data_dir, name);
        out.push(json!({"name": name, "rows": rows.len(), "columns": columns(&rows), "key": m["key"], "about": m["about"],
                        "updated": m["updated"], "created": m["created"]}));
    }
    out.sort_by(|a, b| b["updated"].as_u64().cmp(&a["updated"].as_u64()));
    out
}

/// The union of the rows' fields in first-seen order (the `_` fields last).
pub fn columns(rows: &[Value]) -> Vec<String> {
    let mut cols: Vec<String> = Vec::new();
    for r in rows {
        if let Some(o) = r.as_object() {
            for k in o.keys() {
                if !cols.contains(k) { cols.push(k.clone()); }
            }
        }
    }
    let (mut plain, under): (Vec<String>, Vec<String>) = cols.into_iter().partition(|c| !c.starts_with('_'));
    plain.extend(under);
    plain
}

fn cell(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// The rows as CSV (RFC 4180: fields with a comma, quote or line break are quoted). A field that starts with `=`, `+`, `-`
/// or `@` gets a leading `'`, so a spreadsheet does not run a scraped page's text as a formula.
pub fn to_csv(rows: &[Value]) -> String {
    let cols = columns(rows);
    let esc = |s: &str| {
        let s = if s.starts_with(['=', '+', '-', '@']) && s.parse::<f64>().is_err() { format!("'{s}") } else { s.to_string() };
        if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s }
    };
    let mut out = cols.iter().map(|c| esc(c)).collect::<Vec<_>>().join(",");
    out.push_str("\r\n");
    for r in rows {
        out.push_str(&cols.iter().map(|c| esc(&cell(r.get(c)))).collect::<Vec<_>>().join(","));
        out.push_str("\r\n");
    }
    out
}

pub fn delete(data_dir: &Path, name: &str) -> Result<bool, String> {
    check_name(name)?;
    let had = rows_path(data_dir, name).exists();
    let _ = std::fs::remove_file(rows_path(data_dir, name));
    let _ = std::fs::remove_file(meta_path(data_dir, name));
    Ok(had)
}

fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("am2-ds-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn names_cannot_leave_the_folder() {
        assert!(check_name("books_2026-11").is_ok());
        for bad in ["", "../x", "a/b", "a.b", "x y", &"a".repeat(65)] {
            assert!(check_name(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_keyed_dataset_replaces_instead_of_copying() {
        let d = tmp();
        let k = vec!["url".to_string()];
        let s = save(&d, "books", &[json!({"url": "a", "t": 1}), json!({"url": "b", "t": 2})], Some(&k), Some("https://x"), None).unwrap();
        assert_eq!(s, Saved { added: 2, replaced: 0, skipped: 0, total: 2 });
        // the key is remembered: a later save without one still de-duplicates
        let s = save(&d, "books", &[json!({"url": "a", "t": 9}), json!("not a row"), json!({"url": "c"})], None, None, None).unwrap();
        assert_eq!(s, Saved { added: 1, replaced: 1, skipped: 1, total: 3 });
        let rows = read(&d, "books").unwrap();
        assert_eq!(rows[0]["t"], 9);
        assert_eq!(rows[0].get("_source"), None, "the replacing row came with no source");
        assert_eq!(rows[1]["_source"], "https://x");
        assert_eq!(list(&d)[0]["rows"], 3);
        assert!(delete(&d, "books").unwrap());
        assert!(read(&d, "books").unwrap().is_empty());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn csv_quotes_and_defuses_formulas() {
        let rows = vec![json!({"a": "x,y", "b": "say \"hi\""}), json!({"a": "=HYPERLINK(1)", "c": -3, "b": "-12.5"})];
        let csv = to_csv(&rows);
        let lines: Vec<&str> = csv.split("\r\n").collect();
        assert_eq!(lines[0], "a,b,c");
        assert_eq!(lines[1], "\"x,y\",\"say \"\"hi\"\"\",");
        assert_eq!(lines[2], "'=HYPERLINK(1),-12.5,-3", "numbers keep their sign; a formula does not run");
    }

    #[test]
    fn row_keys_join_fields_and_need_all_of_them() {
        let k = vec!["a".to_string(), "b".to_string()];
        assert_eq!(row_key(&json!({"a": " x ", "b": 2}), &k).as_deref(), Some("x\u{1f}2"));
        assert_eq!(row_key(&json!({"a": "x"}), &k), None);
        assert_eq!(row_key(&json!({"a": "x"}), &[]), None);
    }
}
