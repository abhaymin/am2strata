//! Browser use against a real headless Chrome and local test pages (forms, tables with pages, a map, errors, links): every
//! tool the model and the API use, checked end to end. The browser is chrome-headless-shell from `AM2_TEST_CHROME`, else
//! the installer's `<data>/browser`, else a Chrome on PATH; with none here the tests say so and pass (CI installs one:
//! .github/workflows/ci.yml).

// This whole file is test-only: browser.rs includes it under #[cfg(test)], so the local pages below are no route of the server.
use super::*;
use axum::response::Html;

const FORM: &str = r#"<!doctype html><link rel=icon href="data:,"><title>Survey</title><form id=f action="/sent">
<label for=n>Your name</label><input id=n name=name>
<label>Email <input type=email name=email placeholder="you@example.com"></label>
<label for=pw>Password</label><input id=pw type=password name=pw>
<fieldset><legend>How did you hear about us?</legend>
 <label><input type=radio name=heard value=friend> A friend</label><label><input type=radio name=heard value=web> The web</label></fieldset>
<label><input type=checkbox name=news> Send me news</label>
<label for=c>Country</label><select id=c name=country><option value="">Pick</option><option value=de>Germany</option><option value=in>India</option></select>
<label for=t>Comments</label><textarea id=t name=comments></textarea>
<button type=submit>Submit</button></form>
<script>window.changes=0;document.querySelectorAll('input,select,textarea').forEach(e=>e.addEventListener('change',()=>window.changes++))</script>"#;

const MERGED: &str = r#"<!doctype html><link rel=icon href="data:,"><title>Cities</title><table><tr><th>x</th></tr></table><table>
<thead><tr><th rowspan=2>City</th><th rowspan=2>Country</th><th colspan=2>Population[a]</th></tr><tr><th>2018</th><th>2025</th></tr></thead>
<tbody><tr><td><a href="/tokyo">Tokyo</a></td><td rowspan=2>Japan</td><td>37</td><td>36</td></tr><tr><td>Osaka</td><td>19</td><td>18</td></tr></tbody></table>"#;

// footnote marks two ways Wikipedia keeps them: a real <sup class=reference> element and plain text left on the cell.
const FOOTNOTES: &str = r##"<!doctype html><link rel=icon href="data:,"><title>Largest cities</title><table>
<thead><tr><th>City</th><th>Population<sup class="reference">[2]</sup></th></tr></thead>
<tbody><tr><td>Tokyo<sup class="reference"><a href="#c1">[a]</a></sup></td><td>37,468,000<sup class="reference">[3]</sup></td></tr>
<tr><td>Delhi[12]</td><td>C[1]</td></tr><tr><td>Osaka [capital]</td><td>arr[idx]</td></tr></tbody></table>"##;

const MAP: &str = r#"<!doctype html><link rel=icon href="data:,"><title>Map</title><body style="margin:0"><div id=m style="width:1280px;height:800px;background:#9cf"></div>
<script>window.log=[];const m=document.getElementById('m');let down=null;
m.addEventListener('mousedown',e=>{down=[e.clientX,e.clientY]});
m.addEventListener('mouseup',e=>{const dx=down?e.clientX-down[0]:0,dy=down?e.clientY-down[1]:0;if(dx||dy)log.push('drag '+dx+','+dy);down=null});
m.addEventListener('wheel',e=>{log.push('wheel '+Math.sign(e.deltaY))});
m.addEventListener('dblclick',e=>log.push('dbl'));
m.addEventListener('contextmenu',e=>{e.preventDefault();log.push('right')});
document.addEventListener('keydown',e=>log.push('key '+e.key+(e.ctrlKey?' ctrl':'')));</script>"#;

const ERRORS: &str = r#"<!doctype html><link rel=icon href="data:,"><title>Broken</title><h1>Oops page</h1><img src="/missing.png"><a href="/gone">dead</a><a href="/table">ok</a>
<script>console.error('boom from the page'); setTimeout(()=>{ throw new Error('late failure') }, 10)</script>"#;

fn table(page: u32) -> String {
    let rows: String = (1..=2).map(|i| { let n = (page - 1) * 2 + i; format!("<tr><td><a href=\"/item/{n}\">Book {n}</a></td><td>{}.00</td></tr>", n * 10) }).collect();
    let items: String = (1..=2).map(|i| { let n = (page - 1) * 2 + i; format!("<div class=card><h3>Card {n}</h3><span class=price>{n}</span><a href=\"/c/{n}\">more</a></div>") }).collect();
    let next = if page < 2 { "<a class=next href=\"/table?p=2\">Next</a>" } else { "" };
    format!("<!doctype html><link rel=icon href=data:,><title>Books p{page}</title><table class=layout><tr><td>menu</td></tr></table><table><thead><tr><th>Title</th><th>Price</th></tr></thead><tbody>{rows}</tbody></table>{items}{next}")
}

async fn serve() -> String {
    use axum::routing::get;
    let app = axum::Router::new()
        .route("/form", get(|| async { Html(FORM) }))
        .route("/sent", get(|| async { Html("<title>Thanks</title>sent") }))
        .route("/remember", get(|| async { (
            [("set-cookie", "visitor=seen; Path=/; Max-Age=86400")],
            Html("<title>Remember</title><script>localStorage.setItem('mark','set')</script>set")) }))
        .route("/look", get(|h: axum::http::HeaderMap| async move { Html(format!(
            "<title>Look</title><p id=c>{}</p>", h.get("cookie").and_then(|v| v.to_str().ok()).unwrap_or("none"))) }))
        .route("/merged", get(|| async { Html(MERGED) }))
        .route("/footnotes", get(|| async { Html(FOOTNOTES) }))
        .route("/map", get(|| async { Html(MAP) }))
        .route("/errors", get(|| async { Html(ERRORS) }))
        .route("/table", get(|q: axum::extract::Query<HashMap<String, String>>| async move {
            Html(table(q.get("p").and_then(|p| p.parse().ok()).unwrap_or(1))) }));
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    format!("http://{addr}")
}

fn chrome() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AM2_TEST_CHROME").map(PathBuf::from).filter(|p| p.is_file()) { return Some(p); }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    for d in [home.join("Development/Strata-data"), home.join(".local/share/am2strata")] {
        if let Some(p) = find_chrome(&d) { return Some(p); }
    }
    None
}

fn browsers(data: &Path) -> Option<Browsers> {
    let exe = chrome().or_else(|| { eprintln!("no chrome-headless-shell here: the live browser tests are skipped"); None })?;
    let mut rest = std::collections::BTreeMap::new();
    rest.insert("browser".into(), json!({"exe": exe, "allow_script": true}));
    rest.insert("allow_private_urls".into(), json!(true));
    Some(Browsers::new(Settings::from_config(&rest, data)))
}

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!("am2-live-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The page's answer to `js`: tried again for up to 10 s while the page is still loading or its script not yet run
/// (a slow runner, the macOS Intel one, read a page before its script had made `window.log`).
async fn eval(b: &Browsers, conv: &str, js: &str) -> Value {
    let mut last = String::new();
    for _ in 0..50 {
        match b.current(conv).await.unwrap().evaluate(js).await {
            Ok(v) => return v.into_value::<Value>().unwrap(),
            Err(e) => last = e.to_string(),
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    panic!("the page did not answer {js:?} in 10 s: {last}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_browser_tools_work_on_real_pages() {
    let _ = tracing_subscriber::fmt().with_env_filter("am2strata=debug").with_test_writer().try_init();
    let data = tmp();
    let Some(b) = browsers(&data) else { return };
    let base = serve().await;
    let c = "t1";

    // ---- forms: elements by number, fill by label, radio by question, risk on a password, nothing sent
    b.call(c, "browser_open", &json!({"url": format!("{base}/form")})).await.unwrap();
    let els = b.call(c, "browser_elements", &json!({"kind": "fields"})).await.unwrap().text;
    assert!(els.contains("textbox \"Your name\"") && els.contains("select \"Country\"") && els.contains("options: Pick | Germany | India"), "{els}");
    assert!(els.contains("radio \"A friend (group heard)\""), "{els}");
    let fill = json!({"fields": [{"field": "Your name", "value": "Ada"}, {"field": "you@example.com", "value": "ada@example.com"},
        {"field": "How did you hear about us?", "value": "The web"}, {"field": "Send me news", "value": true},
        {"field": "Country", "value": "India"}, {"field": "Comments", "value": "Fine"}, {"field": "Shoe size", "value": "9"}]});
    assert!(b.risk(c, "browser_fill", &fill).await.is_none(), "no password field: no question");
    let r = b.call(c, "browser_fill", &fill).await.unwrap().text;
    assert!(r.contains("✗ Shoe size") && r.contains("Nothing was sent"), "{r}");
    let v = eval(&b, c, "JSON.stringify([...new FormData(document.getElementById('f'))])").await;
    let got: Value = serde_json::from_str(v.as_str().unwrap()).unwrap();
    assert_eq!(got, json!([["name","Ada"],["email","ada@example.com"],["pw",""],["heard","web"],["news","on"],["country","in"],["comments","Fine"]]));
    assert!(eval(&b, c, "window.changes").await.as_u64().unwrap() >= 5, "the page saw change events");
    assert!(b.risk(c, "browser_fill", &json!({"fields": [{"field": "Password", "value": "x"}]})).await.unwrap().what.contains("password"));
    // a ref from the list clicks that control; the submit button is a risk
    let btns = b.call(c, "browser_elements", &json!({"kind": "buttons", "contains": "submit"})).await.unwrap().text;
    let r#ref = btns.lines().find(|l| l.contains("\"Submit\"")).and_then(|l| l.split(']').next()).map(|s| s.trim_start_matches('[').to_string()).unwrap();
    // a select by its CSS (this used to fail always: the script's answer was not parsed)
    b.call(c, "browser_select", &json!({"selector": "#c", "value": "de"})).await.unwrap();
    assert_eq!(eval(&b, c, "document.getElementById('c').value").await, json!("de"));
    assert!(b.call(c, "browser_select", &json!({"selector": "#c", "value": "fr"})).await.unwrap_err().contains("available: , de, in"));
    // a publishing click by text asks too
    assert!(b.risk(c, "browser_click", &json!({"text": "Submit"})).await.is_some(), "the submit button by its text is a risk");
    assert!(b.risk(c, "browser_click", &json!({"ref": r#ref})).await.unwrap().what.starts_with("submit a form"));
    b.call(c, "browser_click", &json!({"ref": r#ref})).await.unwrap();
    assert!(b.current(c).await.unwrap().url().await.unwrap().unwrap().contains("/sent"), "the ref clicked Submit");

    // ---- scraping: a table, items with fields, two pages through "Next", into a keyed dataset (again: replaced, not copied)
    b.call(c, "browser_open", &json!({"url": format!("{base}/table")})).await.unwrap();
    let t = b.call(c, "browser_extract", &json!({"kind": "table"})).await.unwrap().text;
    assert!(t.contains("table 1 of 2 (the biggest"), "{t}");
    assert!(t.starts_with("2 row(s)") && t.contains("\"Title\": \"Book 1\"") && t.contains("\"Title_link\": \"") && t.contains("/item/1"), "{t}");
    let args = json!({"kind": "items", "selector": ".card", "fields": {"name": "h3", "price": ".price", "url": "a@href"}, "next": "Next", "pages": 5,
                      "dataset": "cards", "key": ["url"]});
    let t = b.call(c, "browser_extract", &args).await.unwrap().text;
    assert!(t.starts_with("4 row(s) from 2 page(s)") && t.contains("no next page after page 2") && t.contains("4 added"), "{t}");
    b.call(c, "browser_open", &json!({"url": format!("{base}/table")})).await.unwrap();
    let t = b.call(c, "browser_extract", &args).await.unwrap().text;
    assert!(t.contains("0 added, 4 replaced; it holds 4 rows"), "{t}");
    let rows = crate::datasets::read(&data, "cards").unwrap();
    assert_eq!(rows[3]["name"], "Card 4");
    assert!(rows[3]["url"].as_str().unwrap().ends_with("/c/4") && rows[3]["source"].as_str().unwrap().contains("p=2"));
    let meta = b.call(c, "browser_extract", &json!({"kind": "meta"})).await.unwrap().text;
    assert!(meta.contains("\"title\": \"Books p2\""), "{meta}");
    let s = b.call(c, "browser_save", &json!({"dataset": "notes", "rows": [{"a": 1}, "x"]})).await.unwrap().text;
    assert!(s.contains("1 added") && s.contains("1 skipped"), "{s}");

    // merged header cells name their columns, and a row-spanning cell fills the rows under it
    b.call(c, "browser_open", &json!({"url": format!("{base}/merged")})).await.unwrap();
    let t = b.call(c, "browser_extract", &json!({"kind": "table"})).await.unwrap().text;
    assert!(t.contains("\"Population 2025\": \"18\"") && t.contains("\"City_link\": \"") && t.matches("\"Country\": \"Japan\"").count() == 2, "{t}");

    // footnotes are dropped from cells and headers: a <sup class=reference> element and plain-text marks, but not C[1]
    b.call(c, "browser_open", &json!({"url": format!("{base}/footnotes")})).await.unwrap();
    let t = b.call(c, "browser_extract", &json!({"kind": "table"})).await.unwrap().text;
    assert!(t.contains("table 0 of 1"), "{t}");
    assert!(t.contains("\"City\": \"Tokyo\"") && t.contains("\"Population\": \"37,468,000\""), "{t}");
    assert!(t.contains("\"City\": \"Delhi\"") && t.contains("\"Population\": \"C[1]\""), "{t}");
    // a bracketed word that is not a footnote shape stays
    assert!(t.contains("\"City\": \"Osaka [capital]\"") && t.contains("\"Population\": \"arr[idx]\""), "{t}");
    assert!(!t.contains("Population[2]"), "the header mark fell off: {t}");

    // a data: page opens as itself (not "https://data:…")
    let d = b.call(c, "browser_open", &json!({"url": "data:text/html,<title>Inline</title><p>made here</p>"})).await.unwrap().text;
    assert!(d.contains("made here"), "{d}");

    // ---- a map: drag pans, the wheel zooms both ways, double and right clicks, keys with modifiers
    b.call(c, "browser_open", &json!({"url": format!("{base}/map")})).await.unwrap();
    b.call(c, "browser_mouse", &json!({"action": "drag", "x": 600, "y": 400, "to_x": 500, "to_y": 450})).await.unwrap();
    b.call(c, "browser_mouse", &json!({"action": "wheel", "x": 600, "y": 400, "dy": -200, "times": 2})).await.unwrap();
    b.call(c, "browser_mouse", &json!({"action": "wheel", "x": 600, "y": 400})).await.unwrap();
    b.call(c, "browser_mouse", &json!({"action": "double", "x": 100, "y": 100})).await.unwrap();
    b.call(c, "browser_mouse", &json!({"action": "right", "x": 100, "y": 100})).await.unwrap();
    b.call(c, "browser_key", &json!({"keys": "ArrowLeft", "times": 2})).await.unwrap();
    b.call(c, "browser_key", &json!({"keys": "Ctrl+a"})).await.unwrap();
    b.call(c, "browser_key", &json!({"keys": "+"})).await.unwrap();
    // the page's own record of what it got (printed in full when it differs: a macOS runner once did, without saying how)
    let log = eval(&b, c, "window.log.join('|')").await;
    let want = "drag -100,50|wheel -1|wheel -1|wheel 1|dbl|right|key ArrowLeft|key ArrowLeft|key a ctrl|key +";
    assert_eq!(log.as_str(), Some(want), "the map page got {log} on {}", std::env::consts::OS);

    // ---- site checks: status, console and script errors, failed requests, broken links; console listing
    b.call(c, "browser_open", &json!({"url": format!("{base}/errors")})).await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    let r = b.call(c, "browser_check", &json!({"checks": [{"status": 200}, {"text": "oops"}, {"no_text": "welcome"}, {"title": "broken"},
        {"selector": "img", "count": 1}, {"console_errors": 0}, {"failed_requests": 0}, {"broken_links": 10}, {"load_ms": 30000}, {"frobnicate": 1}]})).await.unwrap().text;
    assert!(r.starts_with("FAIL") && r.contains("6 passed, 4 failed"), "{r}");
    assert!(r.contains("✓ status 200") && r.contains("✗ console errors ≤ 0: 2 error(s)") && r.contains("✗ failed requests ≤ 0: 1 failed"), "{r}");
    assert!(r.contains("✗ broken links: 1 of 2 broken: 404") && r.contains("/gone"), "{r}");
    let con = b.call(c, "browser_console", &json!({})).await.unwrap().text;
    assert!(!con.contains("/sent") && !con.contains("/gone 404"), "only this page's log: {con}");
    assert!(con.contains("HTTP 200") && con.contains("late failure") && con.contains("[error] boom from the page") && con.contains("404: ") && con.contains("missing.png"), "{con}");
    let sh = b.call(c, "browser_screenshot", &json!({"full": true})).await.unwrap();
    assert!(sh.screenshot.is_some_and(|s| s.len() > 1000));
    let v = b.call(c, "browser_script", &json!({"expression": "document.title + ' ' + (1 + 1)"})).await.unwrap().text;
    assert!(v.contains("\"Broken 2\""), "{v}");
    b.call(c, "browser_back", &json!({})).await.unwrap();
    let f = b.call(c, "browser_forward", &json!({})).await.unwrap().text;
    assert!(f.contains("/errors"), "{f}");

    // ---- a tab whose DevTools session dies (driven from elsewhere): the next step reconnects by itself
    let page = b.current(c).await.unwrap();
    let sid = page.session_id().clone();
    {
        let run = b.run.lock().await;
        let _ = run.as_ref().unwrap().browser.execute(chromiumoxide::cdp::browser_protocol::target::DetachFromTargetParams::builder().session_id(sid).build()).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    let back = b.call(c, "browser_open", &json!({"url": format!("{base}/table")})).await;
    assert!(back.as_ref().is_ok_and(|s| s.text.contains("Books p1")), "the step after a lost session works: {:?}", back.map(|s| s.text));

    // ---- the whole DevTools connection dies (the browser went away under us): the next step starts a fresh one
    {
        let mut run = b.run.lock().await;
        if let Some(child) = run.as_mut().unwrap().child.as_mut() { let _ = child.kill(); }
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    let back = tokio::time::timeout(Duration::from_secs(150), b.call(c, "browser_open", &json!({"url": format!("{base}/table")}))).await
        .expect("the step after a lost connection hangs");
    assert!(back.as_ref().is_ok_and(|s| s.text.contains("Books p1")), "the step after a lost connection works: {:?}", back.map(|s| s.text));

    // ---- a site test with no model, and its JUnit report
    let suite = json!({"name": "shop", "base": base, "steps": [
        {"open": "/table"}, {"expect": [{"status": 200}, {"text": "Book 1"}, {"console_errors": 0}]},
        {"extract": {"kind": "table"}, "rows": {"min": 2}}, {"click": "Next"}, {"expect": [{"url": "p=2"}, {"text": "Book 3"}]},
        {"open": "/form"}, {"fill": [{"field": "Your name", "value": "Bo"}]}, {"expect": [{"selector": "#n"}]},
        {"expect": [{"text": "not here"}]}, {"open": "/never-reached"}]});
    let rep = crate::browser_test::run(&b, &suite).await;
    assert_eq!((rep["passed"].as_u64(), rep["failed"].as_u64(), rep["skipped"].as_u64(), rep["ok"].as_bool()), (Some(8), Some(1), Some(1), Some(false)), "{rep:#}");
    assert!(rep["steps"][8]["image"].as_str().is_some_and(|i| i.starts_with("data:image/jpeg")), "a failed step keeps a screenshot");
    assert!(crate::browser_test::junit(&rep).contains("failures=\"1\" skipped=\"1\""));
    assert!(b.current(&format!("test:x")).await.is_none());

    b.close().await;
    let _ = std::fs::remove_dir_all(data);
}

/// Each browser-use session is thrown away: what one stores (a cookie, local storage) another never sees, and it is gone
/// once the session ends, when the same conversation starts again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_browser_session_is_thrown_away_when_it_ends() {
    let data = tmp();
    let Some(b) = browsers(&data) else { return };
    let base = serve().await;
    b.call("one", "browser_open", &json!({"url": format!("{base}/remember")})).await.unwrap();
    b.call("one", "browser_open", &json!({"url": format!("{base}/look")})).await.unwrap();
    assert_eq!(eval(&b, "one", "document.getElementById('c').textContent").await, json!("visitor=seen"), "the session keeps its own cookie");
    assert_eq!(eval(&b, "one", "localStorage.getItem('mark')").await, json!("set"));
    // another session at the same time: nothing of the first
    b.call("two", "browser_open", &json!({"url": format!("{base}/look")})).await.unwrap();
    assert_eq!(eval(&b, "two", "document.getElementById('c').textContent").await, json!("none"), "another session sees no cookie");
    assert_eq!(eval(&b, "two", "String(localStorage.getItem('mark'))").await, json!("null"));
    // the first session ends: its context goes; the same conversation starts clean
    b.close_session("one").await;
    b.call("one", "browser_open", &json!({"url": format!("{base}/look")})).await.unwrap();
    assert_eq!(eval(&b, "one", "document.getElementById('c').textContent").await, json!("none"), "nothing kept after the session ended");
    assert_eq!(eval(&b, "one", "String(localStorage.getItem('mark'))").await, json!("null"));
    b.close().await;
    let _ = std::fs::remove_dir_all(&data);
}
