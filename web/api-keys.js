// Settings › Clients: API keys made while the server runs, and everyone's usage (the server's api_keys.rs).
// Its own file and its own scope: only loadApiKeys is shared with app.js (which calls it when the section opens).
(() => {
  const el = (id) => document.getElementById(id);
  let keys = [];
  let toolSets = [];

  async function api(path, opts = {}) {
    const r = await fetch(path, {...opts, headers: headers(Boolean(opts.body))});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error((j.error && j.error.message) || `HTTP ${r.status}`);
    return j;
  }

  const day = (ts) => (ts ? new Date(ts * 1000).toLocaleDateString(undefined, {year: "numeric", month: "short", day: "numeric"}) : "–");
  function ago(ts) {
    if (!ts) return "never used";
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 90) return "used just now";
    if (s < 5400) return `used ${Math.round(s / 60)} min ago`;
    if (s < 129600) return `used ${Math.round(s / 3600)} h ago`;
    return `used ${day(ts)}`;
  }
  const use = (c) => `${fmt(c.requests)} request${c.requests === 1 ? "" : "s"}, ${kfmt(c.total_tokens)} tokens`;

  function keyRow(k) {
    const off = k.status !== "active";
    const lim = [];
    if (k.limits.requests_per_day != null) lim.push(`${fmt(k.limits.requests_per_day)} requests a day`);
    if (k.limits.tokens_per_day != null) lim.push(`${kfmt(k.limits.tokens_per_day)} tokens a day`);
    const facts = [`<code>${esc(k.prefix)}…</code>`, `made ${day(k.created_at)}`, ago(k.last_used)];
    if (k.status === "revoked") facts.push(`revoked ${day(k.revoked_at)}`);
    else if (k.expires_at) facts.push(`${k.status === "expired" ? "expired" : "expires"} ${day(k.expires_at)}`);
    const chips = (k.tools || []).map((t) => `<span class="chip">${esc(t)}</span>`).join("") || '<span class="muted small">no tools</span>';
    const models = (k.usage.models || []).map((m) => `<span class="chip">${esc(m)}</span>`).join("");
    return `<div class="mcp-server key-row${off ? " key-row--off" : ""}"><div class="mcp-server__head"><strong>${esc(k.name)}</strong>` +
      `<span class="st-badge${off ? " st-badge--error" : ""}">${esc(k.status)}</span>` +
      `<span class="mcp-server__btns"><button type="button" class="st-btn st-btn--secondary" data-key-usage="${esc(k.id)}">Usage</button>` +
      `${k.status === "revoked" ? "" : `<button type="button" class="st-btn st-btn--danger" data-key-revoke="${esc(k.id)}">Revoke</button>`}</span></div>` +
      `<div class="muted small key-facts">${facts.join(" · ")}</div>` +
      `<div class="small key-facts">Today: ${use(k.usage.today)} · 30 days: ${use(k.usage.days_30)}${lim.length ? ` · limit: ${lim.join(", ")}` : ""}` +
      `${k.models ? ` · only ${k.models.map(esc).join(", ")}` : ""}</div>` +
      `<div class="mcp-server__tools">${chips}</div>` +
      `${models ? `<div class="mcp-server__tools"><span class="muted small">models used</span>${models}</div>` : ""}</div>`;
  }

  function renderTools() {
    if (el("key-tools").childElementCount) return;
    el("key-tools").innerHTML = [...toolSets, "mcp:*"].map((t) =>
      `<label class="key-tool"><input type="checkbox" value="${esc(t)}"> ${esc(t === "mcp:*" ? "MCP servers" : t)}</label>`).join("");
  }

  async function loadApiKeys() {
    if (!el("keys-list")) return;
    try {
      const j = await api("api/keys");
      keys = j.keys || [];
      toolSets = j.tool_sets || [];
      renderTools();
      el("keys-open").hidden = j.main_key_set !== false;
      el("keys-list").innerHTML = keys.map(keyRow).join("") || '<p class="muted">No keys yet: make the first one below.</p>';
      const sel = el("usage-key"), was = sel.value;
      sel.innerHTML = '<option value="">Everyone</option><option value="owner">The main key</option>' +
        keys.map((k) => `<option value="${esc(k.id)}">${esc(k.name)}${k.status === "active" ? "" : ` (${esc(k.status)})`}</option>`).join("");
      sel.value = [...sel.options].some((o) => o.value === was) ? was : "";
    } catch (e) { el("keys-list").innerHTML = `<div class="msg-error">${esc(e.message)}</div>`; }
    loadUsage();
  }

  const COLS = [["requests", "Requests"], ["prompt_tokens", "Prompt"], ["completion_tokens", "Completion"], ["cached_tokens", "Cached"],
                ["media_jobs", "Media"], ["errors", "Errors"]];
  function table(first, rows, name) {
    if (!rows.length) return "";
    return `<div class="usage-scroll"><table class="st-table usage-table"><thead><tr><th>${first}</th>` +
      `${COLS.map(([, t]) => `<th class="num">${t}</th>`).join("")}</tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${name(r)}</td>${COLS.map(([k]) => `<td class="num">${fmt(r[k])}</td>`).join("")}</tr>`).join("") +
      "</tbody></table></div>";
  }

  async function loadUsage() {
    const who = el("usage-key").value, days = el("usage-days").value;
    try {
      const j = await api(`api/usage?days=${encodeURIComponent(days)}${who ? `&key=${encodeURIComponent(who)}` : ""}`);
      const t = j.totals;
      el("usage-sum").textContent = `${use(t)} (${kfmt(t.prompt_tokens)} prompt, ${kfmt(t.cached_tokens)} of it cached, ` +
        `${kfmt(t.completion_tokens)} completion), ${fmt(t.media_jobs)} media job${t.media_jobs === 1 ? "" : "s"}, ${fmt(t.errors)} error${t.errors === 1 ? "" : "s"}`;
      // one bar a day: tokens, or requests while nothing has used tokens
      const by = t.total_tokens ? "total_tokens" : "requests";
      const peak = Math.max(1, ...j.by_day.map((d) => d[by]));
      el("usage-bars").innerHTML = j.by_day.map((d) =>
        `<div class="usage-bar" title="${esc(d.day)}: ${esc(use(d))}"><i style="height:${d[by] ? Math.max(3, Math.round(d[by] * 100 / peak)) : 0}%"></i></div>`).join("");
      el("usage-bars").setAttribute("aria-label", `${by === "requests" ? "Requests" : "Tokens"} a day, ${j.from} to ${j.to}`);
      el("usage-axis").innerHTML = `<span>${esc(j.from)}</span><span>${by === "requests" ? "requests" : "tokens"} a day, most ${kfmt(peak)}</span><span>${esc(j.to)}</span>`;
      el("usage-models").innerHTML = table("Model", j.by_model, (r) => esc(r.model)) || '<p class="muted small">Nothing used in these days.</p>';
      el("usage-keys").innerHTML = who ? "" : table("Key", j.by_key, (r) => `${esc(r.name)}${r.kind === "client" ? ' <span class="muted small">profile</span>' : ""}`);
    } catch (e) { el("usage-sum").textContent = ""; el("usage-models").innerHTML = `<div class="msg-error">${esc(e.message)}</div>`; }
  }

  if (el("key-form")) {
    el("key-form").onsubmit = async (e) => {
      e.preventDefault();
      const num = (id) => { const v = parseInt(el(id).value, 10); return v > 0 ? v : null; };
      const body = {name: el("key-name").value.trim(),
                    tools: [...el("key-tools").querySelectorAll("input:checked")].map((c) => c.value),
                    limits: {requests_per_day: num("key-req-day"), tokens_per_day: num("key-tok-day")},
                    models: el("key-models").value.split(",").map((m) => m.trim()).filter(Boolean)};
      if (el("key-expires").value) body.expires_in = el("key-expires").value;
      try {
        const j = await api("api/keys", {method: "POST", body: JSON.stringify(body)});
        el("key-once-name").textContent = j.name;
        el("key-once-value").textContent = j.key;
        el("key-once").hidden = false;
        el("key-once").scrollIntoView({block: "center"});
        el("key-form").reset();
        loadApiKeys();
      } catch (err) { toast("error", "The key was not made", err.message, 6000); }
    };
    el("key-once-copy").onclick = () => copyText(el("key-once-value").textContent);
    el("key-once-done").onclick = () => { el("key-once-value").textContent = ""; el("key-once").hidden = true; };
    el("keys-list").onclick = async (e) => {
      const revoke = e.target.closest("[data-key-revoke]"), show = e.target.closest("[data-key-usage]");
      if (show) {
        el("usage-key").value = show.dataset.keyUsage;
        await loadUsage();
        el("usage-card").scrollIntoView({block: "start"});
      }
      if (!revoke) return;
      const k = keys.find((x) => x.id === revoke.dataset.keyRevoke);
      if (!k || !confirm(`Revoke the key "${k.name}"?\n\nWhatever uses it stops working at once. This cannot be undone; its usage stays.`)) return;
      try {
        await api(`api/keys/${encodeURIComponent(k.id)}`, {method: "DELETE"});
        toast("success", "Key revoked", `${k.name} no longer works.`);
        loadApiKeys();
      } catch (err) { toast("error", "The key was not revoked", err.message, 6000); }
    };
    el("usage-key").onchange = loadUsage;
    el("usage-days").onchange = loadUsage;
  }

  window.loadApiKeys = loadApiKeys;
  if (typeof section !== "undefined" && section === "clients") loadApiKeys();
})();
