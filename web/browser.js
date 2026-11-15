// The Browser tab: drive a browser by hand (address bar, click, scroll and type on the live picture), run its tools
// (read, elements, extract, checks, fill), see the datasets jobs collected, write and run site tests, answer the steps
// MCP clients wait on, and how to connect an agent (the server's browser_api.rs). Its own scope: only browserTab is shared.
(() => {
  const el = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const store = {get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch (e) { return d; } },
                 set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* storage off */ } }};
  let on = false, timer = null, busy = false, status = null, imgSize = [1280, 800];

  async function api(path, opts = {}) {
    const r = await fetch(path, {...opts, headers: headers(Boolean(opts.body))});
    const ct = r.headers.get("content-type") || "";
    const j = ct.includes("json") ? await r.json().catch(() => ({})) : {text: await r.text()};
    if (!r.ok) throw new Error((j.error && j.error.message) || `HTTP ${r.status}`);
    return j;
  }
  const target = () => el("bv-target").value || "";
  const out = (t) => { el("bv-out").textContent = t || ""; };

  async function step(tool, args = {}) {
    if (busy) return null;
    busy = true;
    document.body.classList.add("bv-busy");
    out(`${tool} …`);
    try {
      const r = await api("/api/browser/step", {method: "POST", body: JSON.stringify({target: target(), tool, args})});
      out((r.ok ? "" : "✗ ") + r.text + `\n(${r.ms} ms)`);
      if (r.image) show(r.image);
      await refreshView(true);
      return r;
    } catch (e) { out("✗ " + e.message); return null; }
    finally { busy = false; document.body.classList.remove("bv-busy"); }
  }

  function show(src) {
    const img = el("bv-img");
    img.src = src;
    img.hidden = false;
    el("bv-empty").hidden = true;
  }

  async function refreshView(force) {
    if (!on || (busy && !force)) return;
    try {
      const v = await api(`/api/browser/view?target=${encodeURIComponent(target())}`);
      if (!v.open) return;
      if (v.image) show(v.image);
      el("bv-here").textContent = `${v.title || ""}  ${v.url || ""}`.trim();
      if (document.activeElement !== el("bv-url") && v.url) el("bv-url").value = v.url;
    } catch (e) { /* the browser is starting, or gone */ }
  }

  async function refreshStatus() {
    try {
      status = await api("/api/browser");
    } catch (e) { out("✗ " + e.message); return; }
    const sel = el("bv-target"), keep = sel.value || store.get("am2-browser-target", "");
    sel.innerHTML = status.targets.map((t) => `<option value="${t.id}" ${t.available ? "" : "disabled"} title="${esc(t.why || t.about)}">${esc(t.label)}${t.available ? "" : " (off)"}</option>`).join("");
    const pick = status.targets.find((t) => t.id === keep && t.available) || status.targets.find((t) => t.default && t.available) || status.targets.find((t) => t.available);
    if (pick) sel.value = pick.id;
    const w = el("bv-wait");
    w.hidden = !status.waiting.length;
    w.innerHTML = status.waiting.map((x) => `<div><b>Waiting for you:</b> ${esc(x.what)} <span class="muted">(${esc(x.from)}, ${x.waiting_s} s)</span>
      <button class="st-btn st-btn--primary" data-ok="${esc(x.id)}">Allow</button> <button class="st-btn st-btn--secondary" data-no="${esc(x.id)}">Deny</button></div>`).join("");
    renderDatasets(status.datasets);
    renderTests(status.tests);
  }

  async function answer(id, ok) {
    try { await api("/api/tools/approve", {method: "POST", body: JSON.stringify({id, ok})}); } catch (e) { out("✗ " + e.message); }
    refreshStatus();
  }

  // ---- datasets
  function renderDatasets(list) {
    el("bd-list").innerHTML = list.length ? list.map((d) => `<div class="bv-ds"><b>${esc(d.name)}</b><span class="muted small">${d.rows} rows${d.key ? " · key " + esc(d.key.join("+")) : ""}</span>
      <button class="st-btn st-btn--secondary" data-view="${esc(d.name)}">View</button>
      <a class="st-btn st-btn--secondary" data-dl="${esc(d.name)}" data-f="csv" href="#">CSV</a><a class="st-btn st-btn--secondary" data-dl="${esc(d.name)}" data-f="json" href="#">JSON</a>
      <button class="st-btn st-btn--secondary" data-delds="${esc(d.name)}" title="Delete it">✕</button></div>`).join("")
      : `<p class="muted small">None yet. Extract with "Save to dataset", or ask a chat to collect records (browser_extract / browser_save).</p>`;
  }
  async function viewDataset(name) {
    try {
      const d = await api(`/api/datasets/${encodeURIComponent(name)}?limit=200`);
      const cols = d.columns.slice(0, 12);
      el("bd-view").innerHTML = `<p class="small"><b>${esc(name)}</b>: ${d.total} rows${d.total > 200 ? " (the first 200)" : ""}</p><div class="bv-table"><table><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
        <tbody>${d.rows.map((r) => `<tr>${cols.map((c) => { const v = r[c]; const t = v == null ? "" : typeof v === "string" ? v : JSON.stringify(v); return `<td title="${esc(t)}">${esc(t)}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`;
    } catch (e) { out("✗ " + e.message); }
  }
  async function download(name, f) {
    const r = await fetch(`/api/datasets/${encodeURIComponent(name)}?format=${f}`, {headers: headers(false)});
    if (!r.ok) { out("✗ HTTP " + r.status); return; }
    const blob = f === "json" ? new Blob([JSON.stringify((await r.json()).rows, null, 2)], {type: "application/json"}) : await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name}.${f}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }

  // ---- tests
  const SAMPLE = {base: "https://example.com", steps: [{open: "/"}, {expect: [{status: 200}, {title: "Example"}, {console_errors: 0}, {broken_links: 20}]},
    {extract: {kind: "links"}, rows: {min: 1}}]};
  function renderTests(list) {
    const sel = el("bt-pick"), keep = sel.value;
    sel.innerHTML = `<option value="">New test…</option>` + list.map((t) => `<option value="${esc(t.name)}">${esc(t.name)} (${t.steps} steps)</option>`).join("");
    sel.value = list.some((t) => t.name === keep) ? keep : "";
    if (!el("bt-suite").value) el("bt-suite").value = JSON.stringify(SAMPLE, null, 2);
  }
  function suite() {
    try { return JSON.parse(el("bt-suite").value); } catch (e) { out("✗ The test is not valid JSON: " + e.message); return null; }
  }
  async function runTest() {
    const s = suite();
    if (!s) return;
    el("bt-report").innerHTML = `<p class="muted small">Running…</p>`;
    try {
      const r = await api(`/api/browser/test?target=${encodeURIComponent(target())}`, {method: "POST", body: JSON.stringify({suite: s})});
      el("bt-report").innerHTML = `<p><b>${r.ok ? "PASS" : "FAIL"}</b>: ${r.passed} passed, ${r.failed} failed, ${r.skipped} not run, ${(r.ms / 1000).toFixed(1)} s</p>` +
        r.steps.map((x) => `<div class="bv-step ${x.ok ? "" : "fail"}"><b>${x.n}. ${esc(JSON.stringify(x.step))}</b> ${x.ms} ms
          ${x.error ? `<div>✗ ${esc(x.error)}</div>` : ""}${x.checks ? `<div>${esc(x.text)}</div>` : ""}${x.image && !x.ok ? `<img src="${x.image}" alt="The page when it failed">` : ""}</div>`).join("");
    } catch (e) { el("bt-report").textContent = "✗ " + e.message; }
  }

  // ---- the picture: click, wheel, keys
  function at(ev) {
    const img = el("bv-img"), r = img.getBoundingClientRect();
    const iw = img.naturalWidth || imgSize[0], ih = img.naturalHeight || imgSize[1];
    const s = Math.min(r.width / iw, r.height / ih), ox = (r.width - iw * s) / 2, oy = (r.height - ih * s) / 2;
    return {x: Math.round((ev.clientX - r.left - ox) / s), y: Math.round((ev.clientY - r.top - oy) / s)};
  }
  let wheelAcc = 0, wheelTimer = null, wheelAt = null;

  function wire() {
    el("bv-go").addEventListener("submit", (e) => { e.preventDefault(); const u = el("bv-url").value.trim(); if (u) step("browser_open", {url: u}); });
    el("bv-back").onclick = () => step("browser_back");
    el("bv-fwd").onclick = () => step("browser_forward");
    el("bv-target").onchange = () => { store.set("am2-browser-target", target()); refreshView(true); };
    el("bv-live").onchange = () => schedule();
    el("bv-img").addEventListener("click", (e) => { el("bv-screen").focus(); const p = at(e); step("browser_mouse", {action: "click", ...p}); });
    el("bv-img").addEventListener("dblclick", (e) => { const p = at(e); step("browser_mouse", {action: "double", ...p}); });
    el("bv-img").addEventListener("contextmenu", (e) => { e.preventDefault(); const p = at(e); step("browser_mouse", {action: "right", ...p}); });
    let drag = null;
    el("bv-img").addEventListener("mousedown", (e) => { if (e.button === 0) drag = at(e); });
    el("bv-img").addEventListener("mouseup", (e) => {
      if (!drag) return; const p = at(e), d = drag; drag = null;
      if (Math.abs(p.x - d.x) + Math.abs(p.y - d.y) > 12) { e.stopImmediatePropagation(); step("browser_mouse", {action: "drag", x: d.x, y: d.y, to_x: p.x, to_y: p.y}); }
    }, true);
    el("bv-img").addEventListener("dragstart", (e) => e.preventDefault());
    el("bv-screen").addEventListener("wheel", (e) => {
      e.preventDefault(); wheelAcc += e.deltaY; wheelAt = at(e);
      clearTimeout(wheelTimer);
      wheelTimer = setTimeout(() => { const dy = Math.max(-2000, Math.min(2000, Math.round(wheelAcc))); wheelAcc = 0; if (dy) step("browser_mouse", {action: "wheel", ...wheelAt, dy}); }, 250);
    }, {passive: false});
    let typed = "", typeTimer = null;
    el("bv-screen").addEventListener("keydown", (e) => {
      if (e.target !== el("bv-screen")) return;
      const special = e.key.length > 1 || e.ctrlKey || e.metaKey || e.altKey;
      e.preventDefault();
      if (!special) { typed += e.key; clearTimeout(typeTimer); typeTimer = setTimeout(() => { const t = typed; typed = ""; step("browser_type", {text: t, append: true}); }, 400); return; }
      if (["Shift", "Control", "Alt", "Meta"].includes(e.key)) return;
      const keys = [e.ctrlKey && "Ctrl", e.altKey && "Alt", e.metaKey && "Meta", e.shiftKey && e.key.length > 1 && "Shift", e.key === " " ? "Space" : e.key].filter(Boolean).join("+");
      step("browser_key", {keys});
    });
    for (const b of document.querySelectorAll("[data-step]")) b.onclick = () => step(b.dataset.step, b.dataset.args ? JSON.parse(b.dataset.args) : {});
    el("bx-run").onclick = () => {
      const args = {kind: el("bx-kind").value};
      for (const [k, id] of [["selector", "bx-sel"], ["next", "bx-next"], ["dataset", "bx-ds"]]) { const v = el(id).value.trim(); if (v) args[k] = v; }
      const pages = parseInt(el("bx-pages").value, 10); if (pages > 1) args.pages = pages;
      const key = el("bx-key").value.trim(); if (key) args.key = key.split(/[,+ ]+/).filter(Boolean);
      const f = el("bx-fields").value.trim();
      if (f) { try { args.fields = JSON.parse(f); } catch (e) { out("✗ Fields are not valid JSON"); return; } }
      step("browser_extract", args).then(() => refreshStatus());
    };
    el("bc-run").onclick = () => { try { step("browser_check", {checks: JSON.parse(el("bc-checks").value)}); } catch (e) { out("✗ The checks are not valid JSON"); } };
    el("bf-run").onclick = () => { try { step("browser_fill", {fields: JSON.parse(el("bf-fields").value)}); } catch (e) { out("✗ The fields are not valid JSON"); } };
    for (const t of document.querySelectorAll("[data-bv]")) t.onclick = () => {
      for (const x of document.querySelectorAll("[data-bv]")) x.setAttribute("aria-selected", String(x === t));
      for (const p of document.querySelectorAll(".bv-pane")) p.hidden = p.dataset.pane !== t.dataset.bv;
    };
    el("bv-wait").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; answer(b.dataset.ok || b.dataset.no, Boolean(b.dataset.ok)); });
    el("bd-refresh").onclick = refreshStatus;
    el("bd-list").addEventListener("click", async (e) => {
      const b = e.target.closest("[data-view],[data-dl],[data-delds]"); if (!b) return; e.preventDefault();
      if (b.dataset.view) viewDataset(b.dataset.view);
      if (b.dataset.dl) download(b.dataset.dl, b.dataset.f);
      if (b.dataset.delds && confirm(`Delete the dataset ${b.dataset.delds}? Its records cannot be brought back.`)) {
        try { await api(`/api/datasets/${encodeURIComponent(b.dataset.delds)}`, {method: "DELETE"}); el("bd-view").innerHTML = ""; } catch (x) { out("✗ " + x.message); }
        refreshStatus();
      }
    });
    el("bt-pick").onchange = async () => {
      const n = el("bt-pick").value; el("bt-name").value = n;
      if (!n) { el("bt-suite").value = JSON.stringify(SAMPLE, null, 2); return; }
      try { const s = await api(`/api/browser/tests/${encodeURIComponent(n)}`); delete s.name; el("bt-suite").value = JSON.stringify(s, null, 2); } catch (e) { out("✗ " + e.message); }
    };
    el("bt-save").onclick = async () => {
      const n = el("bt-name").value.trim(), s = suite(); if (!s) return;
      if (!n) { out("✗ Give the test a name"); return; }
      try { await api(`/api/browser/tests/${encodeURIComponent(n)}`, {method: "PUT", body: JSON.stringify(s)}); out(`Saved test ${n}.`); await refreshStatus(); el("bt-pick").value = n; } catch (e) { out("✗ " + e.message); }
    };
    el("bt-del").onclick = async () => {
      const n = el("bt-pick").value; if (!n || !confirm(`Delete the test ${n}?`)) return;
      try { await api(`/api/browser/tests/${encodeURIComponent(n)}`, {method: "DELETE"}); } catch (e) { out("✗ " + e.message); }
      refreshStatus();
    };
    el("bt-run").onclick = runTest;
    el("bv-chat-target").value = store.get("am2-browser-chat-target", "");
    el("bv-chat-target").onchange = () => store.set("am2-browser-chat-target", el("bv-chat-target").value);
    const base = location.origin;
    el("bv-mcp").textContent = `MCP address (Streamable HTTP):\n  ${base}/v1/mcp/browser            (?target=server|mine|sandbox)\n\nClaude Code:\n  claude mcp add --transport http am2strata-browser ${base}/v1/mcp/browser \\\n    --header "Authorization: Bearer <your am2Strata key>"\n\nOther clients (opencode, Cursor, VS Code…): an HTTP MCP server at that address\nwith the same Authorization header.\n\nCLI:\n  am2strata-run browser open https://example.com\n  am2strata-run browser extract --kind table --dataset prices\n  am2strata-run browser test site-test.json --junit report.xml`;
  }

  function schedule() {
    clearInterval(timer);
    timer = null;
    if (!on) return;
    timer = setInterval(() => { if (el("bv-live").checked) refreshView(false); }, 2000);
  }
  let statusTimer = null;

  window.browserTab = (visible) => {
    on = visible;
    clearInterval(statusTimer);
    if (!visible) { schedule(); return; }
    refreshStatus().then(() => refreshView(true));
    statusTimer = setInterval(refreshStatus, 4000);
    schedule();
  };
  const start = () => { wire(); if (!el("view-browser").hidden) window.browserTab(true); };   // opened straight on #browser
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
