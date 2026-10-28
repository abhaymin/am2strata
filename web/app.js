// web/app.js - the am2Strata web app (derived from PolyStrata's, MIT): Chat, Monitor, Settings, API calls. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it

const store = {
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); } catch (e) { /* private mode: in memory only */ } },
};

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", "Copied to clipboard", "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// three choices: system (follows the computer or phone, live), light, dark; and an accent colour (Settings › Appearance)
const darkQuery = matchMedia("(prefers-color-scheme: dark)");
let themeMode = store.get("theme-mode", "system");
function applyTheme() {
  const dark = themeMode === "dark" || (themeMode === "system" && darkQuery.matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  for (const g of ["theme-seg", "theme-cards"]) for (const b of $(g).children) b.setAttribute("aria-checked", String(b.dataset.v === themeMode));
  $("tc-system-hint").textContent = `Follows your computer or phone (now ${darkQuery.matches ? "dark" : "light"})`;
}
function setThemeMode(mode) { themeMode = mode; store.set("theme-mode", mode); applyTheme(); }
for (const g of ["theme-seg", "theme-cards"]) for (const b of $(g).children) b.onclick = () => setThemeMode(b.dataset.v);
darkQuery.addEventListener("change", applyTheme);
applyTheme();
function applyAccent() {
  const a = store.get("accent", null);
  if (a) document.documentElement.style.setProperty("--st-accent", a); else document.documentElement.style.removeProperty("--st-accent");
  for (const b of $("accent-swatches").children) b.setAttribute("aria-checked", String(b.dataset.v === (a || "#3DD68C")));
}
for (const b of $("accent-swatches").children) b.onclick = () => { store.set("accent", b.dataset.v === "#3DD68C" ? null : b.dataset.v); applyAccent(); };
applyAccent();

let tab = "chat";
// the media models again when their tabs open (after start-up: `health` is set further down)
function refreshMedia() {
  try { if (health.image_generation) loadMediaModels(); } catch (e) { /* still starting */ }
}
function showTab(name) {
  tab = ["chat", "images", "monitor", "settings", "calls"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "images", "monitor", "settings", "calls"]) $(`view-${v}`).hidden = v !== tab;
  if (tab === "images") { $("img-prompt").focus(); refreshMedia(); }
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "settings") { showSection(section); refreshMedia(); }
  if (tab === "calls") refreshCalls();
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
function headers(json = false) {
  const h = {};
  const key = store.get("apikey", "");
  if (key) h.Authorization = "Bearer " + key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = store.get("apikey", "");
$("api-key").onchange = () => { store.set("apikey", $("api-key").value.trim()); toast("success", "API key saved", "Kept in this browser only."); };

let health = {model: "strata", images: false, max_context: 0};
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    $("attach-btn").title = health.images ? "Attach a text file, a picture, a video or a PDF (or drop it here)"
                                          : "Attach a text file or a PDF (or drop it here)";
    $("chat-empty-sub").textContent = `${health.model} runs on this PC. Nothing leaves it.`;
    $("tab-btn-images").hidden = !health.image_generation;
    if (health.image_generation && !media.models.length) loadMediaModels();
    else if (!health.image_generation) pickerNote("Media generation is not built on this server");
    if (location.hash === "#images" && health.image_generation) showTab("images");
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// The installed models (GET api/models).  The list is shown when there is more than one to choose from; choosing
// one switches the server to it, which stops the model in memory first and starts a new chat.
async function loadModels() {
  const sel = $("model-select");
  let models = [];
  try {
    const r = await fetch("api/models", {headers: headers()});
    if (r.ok) models = (await r.json()).models || [];
  } catch (e) { return; }
  sel.innerHTML = models.map((m) =>
    `<option value="${esc(m.id)}"${m.current ? " selected" : ""}${m.status === "unavailable" ? " disabled" : ""}>` +
    `${esc(m.display_name && m.display_name !== m.id ? `${m.display_name} (${m.id})` : m.id)}` +
    `${m.status === "unavailable" ? " - unavailable" : ""}</option>`).join("");
  sel.hidden = models.length < 2;
  sel.dataset.current = (models.find((m) => m.current) || {}).id || "";
  installed = models;
  renderModels();
}
let installed = [], updatesOf = {};
// Updates: models downloaded here remember their Hugging Face files; a newer upload of them is offered as an update
$("check-updates").onclick = async () => {
  $("check-updates").disabled = true;
  let n = 0, known = 0;
  for (const m of installed) {
    $("updates-msg").textContent = `Checking ${m.id}\u2026`;
    try {
      const r = await fetch(`api/models/updates?id=${encodeURIComponent(m.id)}`, {headers: headers()});
      if (r.ok) { const u = await r.json(); updatesOf[m.id] = u; known += u.known; n += u.update ? 1 : 0; }
    } catch (e) { /* the next one */ }
  }
  $("updates-msg").textContent = `${n} update(s) available · ${installed.length - known} model(s) not downloaded here, so their source is not known`;
  $("check-updates").disabled = false;
  renderModels();
};
function renderModels() {
  const box = $("models-list");
  if (!installed.length) { box.innerHTML = `<p class="muted">No models are installed yet: add one from Hugging Face below.</p>`; return; }
  box.innerHTML = installed.map((m) => {
    const facts = [m.context ? `${ctxfmt(m.context)} context` : null, m.multimodal ? "pictures, video, PDF" : "text"].filter(Boolean).join(" · ");
    const u = updatesOf[m.id];
    const upd = !u ? "" : u.update ? `<span class="st-badge st-badge--reading" title="${esc(u.changed.join(", "))}">Update available (${(u.bytes / 1e9).toFixed(1)} GB)</span>` +
        `<button class="st-btn st-btn--secondary" data-update="${esc(m.id)}"${m.current ? " disabled title=\"Unload it first\"" : ""}>Update</button>`
      : u.known ? `<span class="st-badge">Up to date</span>` : "";
    return `<div class="model-row"><div class="model-row__name"><b>${esc(m.display_name || m.id)}</b><span>${esc(m.id)} · ${esc(facts)}</span></div>` + upd +
      `<span class="st-badge ${m.current ? "st-badge--generating" : ""}">${m.current ? "Loaded" : "Available"}</span>` +
      (m.current ? `<button class="st-btn st-btn--secondary" data-unload>Unload</button>`
                 : `<button class="st-btn st-btn--primary" data-load="${esc(m.id)}">Load</button>`) + `</div>`;
  }).join("");
}
// Media models (GET api/media/models): the media engine's catalog, what is downloaded, the image model in use.
let media = {models: [], image_model: "", pull: {state: "idle"}}, mmKind = "image", mmTimer = null;
function pickerNote(text) {
  for (const sel of document.querySelectorAll(".media-model")) {
    if (!sel.options.length || sel.options[0].value === "") sel.innerHTML = `<option value="">${esc(text)}</option>`;
  }
}
async function loadMediaModels() {
  pickerNote("Loading models\u2026 (the media engine starts on first use)");
  try {
    const r = await fetch("api/media/models", {headers: headers()});
    if (!r.ok) {
      const why = r.status === 401 ? "An API key is needed (About > Settings)." : `The server answered ${r.status}.`;
      $("mm-status").textContent = why; pickerNote(`Models not loaded: ${why}`); return;
    }
    media = await r.json();
  } catch (e) { pickerNote("Models not loaded: the server is not reachable"); return; }
  renderMediaModels();
  fillModelPickers();
  if (media.audio_device) $("audio-device").value = media.audio_device;
  const sttReady = (media.models || []).some((m) => m.kind === "audio" && m.task === "stt" && m.downloaded);
  $("mic-btn").hidden = !sttReady;
  clearTimeout(mmTimer);
  if (media.pull && media.pull.state === "pulling") mmTimer = setTimeout(loadMediaModels, 3000);
}
function renderMediaModels() {
  const q = $("mm-search").value.trim().toLowerCase(), only = $("mm-downloaded").checked;
  const rows = (media.models || []).filter((m) => (mmKind === "all" || m.kind === mmKind) && (!only || m.downloaded) &&
    (!q || `${m.name} ${m.display_name || ""} ${m.family || ""} ${m.description || ""}`.toLowerCase().includes(q)));
  rows.sort((a, b) => (b.downloaded - a.downloaded) || a.name.localeCompare(b.name));
  const p = media.pull || {};
  const n = (media.models || []).filter((m) => m.downloaded).length;
  $("mm-status").textContent = !media.engine ? (media.note || "The media engine is not built.")
    : p.state === "pulling" ? `Downloading ${p.model}\u2026 (generation waits until it finishes)`
    : p.state === "error" ? `Download of ${p.model} failed: ${p.error}`
    : `${n} downloaded · ${(media.models || []).length} in the catalog · images use ${media.image_model} · videos use ${media.video_model} · ` +
      `speech to text ${media.stt_model} · text to speech ${media.tts_model}`;
  $("mm-list").innerHTML = rows.length ? rows.map((m) => {
    const inUse = (m.kind === "image" && m.name === media.image_model) || (m.kind === "video" && m.name === media.video_model) ||
      (m.kind === "audio" && (m.task === "tts" ? m.name === media.tts_model : m.name === media.stt_model));
    const facts = [m.family, m.size_gb ? `${m.size_gb} GB` : null, m.gated ? "licence to accept on Hugging Face" : null, m.description]
      .filter(Boolean).join(" · ");
    const badge = m.planned ? `<span class="st-badge">Planned (${esc(m.planned)})</span>`
      : inUse ? `<span class="st-badge st-badge--generating">In use</span>`
      : m.downloaded ? `<span class="st-badge">Downloaded</span>` : `<span class="st-badge">Not downloaded</span>`;
    const pulling = p.state === "pulling";
    const btn = m.planned ? ""
      : !m.downloaded ? `<button class="st-btn st-btn--secondary" data-mpull="${esc(m.name)}"${pulling || !media.engine ? " disabled" : ""}>${pulling && p.model === m.name ? "Downloading\u2026" : "Download"}</button>`
      : (m.kind === "image" || m.kind === "video") && !inUse ? `<button class="st-btn st-btn--primary" data-muse="${esc(m.name)}">Use for ${m.kind}s</button>`
      : m.kind === "audio" && !inUse ? `<button class="st-btn st-btn--primary" data-muse="${esc(m.name)}">Use for ${m.task === "tts" ? "speech" : "transcripts"}</button>` : "";
    return `<div class="model-row"><div class="model-row__name"><b>${esc(m.display_name || m.name)}</b><span>${esc(m.name)} · ${esc(facts)}</span></div>${badge}${btn}</div>`;
  }).join("") : `<p class="muted" style="padding:12px 14px">Nothing matches.</p>`;
}
$("mm-kind").addEventListener("click", (e) => {
  const b = e.target.closest("[data-kind]"); if (!b) return;
  mmKind = b.dataset.kind;
  for (const x of $("mm-kind").children) x.setAttribute("aria-selected", String(x === b));
  renderMediaModels();
});
$("mm-search").oninput = renderMediaModels;
$("mm-downloaded").onchange = renderMediaModels;
$("mm-list").addEventListener("click", async (e) => {
  const pull = e.target.closest("[data-mpull]"), use = e.target.closest("[data-muse]");
  const post = async (path, model) => {
    const r = await fetch(path, {method: "POST", headers: headers(true), body: JSON.stringify({model})});
    if (!r.ok) throw new Error(((await r.json()).error || {}).message || `HTTP ${r.status}`);
  };
  try {
    if (pull) {
      const m = media.models.find((x) => x.name === pull.dataset.mpull);
      if (!confirm(`Download ${m.name} (${m.size_gb} GB)?${m.gated ? "\n\nIts licence must be accepted on Hugging Face first, with your token saved." : ""}`)) return;
      await post("api/media/models/pull", m.name);
      toast("info", "Downloading", `${m.name}: resumable, in the background.`);
    }
    if (use) {
      await post("api/media/models/default", use.dataset.muse);
      toast("success", "Model changed", `New jobs use ${use.dataset.muse}.`);
    }
  } catch (err) { toast("error", pull ? "Not downloaded" : "Not changed", String(err.message || err), 8000); }
  loadMediaModels();
});
// Media Generation: Image and Video, each with its model picker (the downloaded models of that kind); Audio is coming
// the Video form's sizes: the chosen model's own (its catalog default), turned, and square for Wan 2.2
function fillVideoSizes() {
  const m = (media.models || []).find((x) => x.name === $("vid-model").value);
  const d = (m && m.defaults) || {};
  const sizes = [];
  if (d.width && d.height) { sizes.push(`${d.width}x${d.height}`, `${d.height}x${d.width}`); }
  if (m && /wan22/.test(m.name)) sizes.push("640x640");
  const keep = $("vid-size").value;
  $("vid-size").innerHTML = `<option value="">model default${d.width ? ` (${d.width}x${d.height})` : ""}</option>` +
    [...new Set(sizes)].map((z) => `<option${z === keep ? " selected" : ""}>${z}</option>`).join("");
}
function fillModelPickers() {
  for (const sel of document.querySelectorAll(".media-model")) {
    const kind = sel.dataset.kind, task = sel.dataset.task;
    const current = kind === "video" ? media.video_model : kind === "audio" ? (task === "tts" ? media.tts_model : media.stt_model) : media.image_model;
    const have = (media.models || []).filter((m) => m.kind === kind && m.downloaded && (!task || m.task === task));
    const opts = have.map((m) => `<option value="${esc(m.name)}"${m.name === current ? " selected" : ""}>${esc(m.display_name || m.name)} (${esc(m.name)}, ${m.size_gb} GB)</option>`);
    if (!have.some((m) => m.name === current)) opts.unshift(`<option value="${esc(current)}" selected>${esc(current)} - not downloaded</option>`);
    opts.push(`<option value="__more">Download more in Settings &gt; Models&hellip;</option>`);
    sel.innerHTML = opts.join("");
  }
  fillVideoSizes();
}
for (const sel of document.querySelectorAll(".media-model")) {
  sel.onchange = async () => {
    if (sel.value === "__more") { fillModelPickers(); showTab("settings"); return; }
    try {
      const r = await fetch("api/media/models/default", {method: "POST", headers: headers(true), body: JSON.stringify({model: sel.value})});
      if (!r.ok) throw new Error(((await r.json()).error || {}).message || `HTTP ${r.status}`);
      toast("success", "Model changed", `New ${sel.dataset.task ? (sel.dataset.task === "tts" ? "speech" : "transcripts") : sel.dataset.kind + "s"} use ${sel.value}.`);
    } catch (err) { toast("error", "Not changed", String(err.message || err), 8000); }
    loadMediaModels();
  };
}
$("media-kinds").addEventListener("click", (e) => {
  const b = e.target.closest("[data-kind]"); if (!b) return;
  for (const x of $("media-kinds").children) x.setAttribute("aria-selected", String(x === b));
  const k = b.dataset.kind;
  $("img-form").hidden = k !== "image"; $("vid-form").hidden = k !== "video"; $("audio-forms").hidden = k !== "audio";
});
$("vid-model").addEventListener("change", fillVideoSizes);
$("vid-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const prompt = $("vid-prompt").value.trim();
  if (!prompt) return;
  const body = {prompt, model: $("vid-model").value, format: "mp4"};
  if ($("vid-size").value) body.size = $("vid-size").value;
  if ($("vid-frames").value) body.frames = Number($("vid-frames").value);
  if ($("vid-fps").value) body.fps = Number($("vid-fps").value);
  if ($("vid-seed").value !== "") body.seed = Number($("vid-seed").value);
  $("vid-go").disabled = true;
  const t0 = performance.now();
  $("vid-hint").textContent = "Generating\u2026 (several minutes; keep this tab open)";
  // the steps as Monitor sees them
  const prog = setInterval(async () => {
    try {
      const l = (await (await fetch("metrics", {headers: headers()})).json()).live || {};
      if (l.state === "media" && l.steps) $("vid-hint").textContent = `Generating\u2026 step ${l.step} of ${l.steps} · ${fmt(l.elapsed_s, 0)} s`;
      else if (l.queued) $("vid-hint").textContent = `Waiting for the GPU (${l.queued} ahead)\u2026`;
    } catch (e) { /* the next tick */ }
  }, 2000);
  try {
    const r = await fetch("v1/videos/generations", {method: "POST", headers: headers(true), body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    const d = j.data[0], fig = document.createElement("figure");
    const mime = {mp4: "video/mp4", webp: "image/webp", gif: "image/gif", apng: "image/apng"}[d.format] || "video/mp4";
    const url = URL.createObjectURL(new Blob([Uint8Array.from(atob(d.b64), (c) => c.charCodeAt(0))], {type: mime}));
    let el;
    if (d.format === "mp4") { el = document.createElement("video"); el.controls = true; el.loop = true; el.autoplay = true; el.muted = true; }
    else el = document.createElement("img");
    el.src = url;
    const cap = document.createElement("figcaption");
    cap.innerHTML = `${esc(j.size)}, ${j.frames} frames at ${j.fps} fps, seed ${d.seed}, ${((performance.now() - t0) / 1000).toFixed(0)} s: ${esc(prompt)} · <a href="${url}" download="am2strata-${d.seed}.${d.format}">save</a>`;
    fig.append(el, cap);
    $("img-results").prepend(fig);
    $("vid-hint").textContent = j.strata && j.strata.frames_note ? `Done (${j.strata.frames_note}).` : "Done.";
  } catch (e) {
    $("vid-hint").textContent = "";
    toast("error", "No video", String(e.message || e), 10000);
  } finally {
    clearInterval(prog);
    $("vid-go").disabled = false;
  }
};

$("models-list").addEventListener("click", async (e) => {
  const load = e.target.closest("[data-load]"), unload = e.target.closest("[data-unload]"), upd = e.target.closest("[data-update]");
  if (upd) {
    const u = updatesOf[upd.dataset.update];
    if (!confirm(`Download the new version of ${upd.dataset.update}?\n\n${u.changed.join("\n")}\n${(u.bytes / 1e9).toFixed(1)} GB from ${u.repo}. The old files stay until the new ones are checked; the pack is built again on the next load.`)) return;
    try {
      const r = await fetch("api/models/update", {method: "POST", headers: headers(true), body: JSON.stringify({id: upd.dataset.update})});
      const j = await r.json();
      if (!r.ok) throw new Error((j.error || {}).message || `HTTP ${r.status}`);
      toast("info", "Updating", j.state === "current" ? j.message : "Downloading in the background; progress shows under Add a model.");
      hubStatus();
    } catch (err) { toast("error", "Not updated", String(err.message || err), 8000); }
    return;
  }
  if (load) { $("model-select").value = load.dataset.load; $("model-select").onchange(); }
  if (unload) {
    if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
    unload.disabled = true;
    try {
      const r = await fetch("api/models/unload", {method: "POST", headers: headers(true), body: "{}"});
      if (!r.ok) throw new Error(((await r.json()).error || {}).message || `HTTP ${r.status}`);
      toast("success", "Model unloaded", "Its GPU memory is free; the next request loads it again.");
    } catch (err) { toast("error", "Not unloaded", String(err.message || err), 6000); }
    loadModels(); loadHealth();
  }
});
$("model-select").onchange = async () => {
  const sel = $("model-select"), name = sel.value, before = sel.dataset.current;
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); sel.value = before; return; }
  sel.disabled = true;
  toast("info", `Switching to ${name}`, "The model in memory is stopped first, then this one is loaded.", 8000);
  let message = "";
  try {
    const r = await fetch("api/models/load", {method: "POST", headers: headers(true), body: JSON.stringify({model: name})});
    if (!r.ok) message = ((await r.json()).error || {}).message || `the server answered ${r.status}`;
  } catch (e) { message = String(e); }
  sel.disabled = false;
  await loadHealth();
  await loadModels();
  if (message) { toast("error", "The model was not switched", message, 10000); return; }
  newChat();
  toast("success", `${name} is loaded`, "A new chat was started.");
};

// ------------------------------------------------------------------ Monitor
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge", unit: "t/s", series: "tok_s"},
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">-</span><span class="st-metric__sub" id="ms-speed">Decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">-</span><span class="st-metric__sub" id="ms-prefill">Prefill</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");

function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
async function poll() {
  try {
    const r = await fetch(reqShowAll ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      setPill("error", "API key needed");
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it under About > Settings.", 6000); }
    } else if (r.ok) {
      lastMetrics = await r.json();
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", "Server not reachable");
  }
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  setTimeout(poll, 1000);
}

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || [])[0];
  // the header pill
  if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? `Reading prompt · ${pct}%` : "Reading prompt");
  } else if (live.state === "generating") {
    setPill("generating", `Generating · ${fmt(live.tok_s, 1)} tok/s`);
  } else if (live.state === "media") {
    setPill("generating", `Making ${live.kind === "video" ? "a video" : "an image"}${live.steps ? ` · step ${live.step}/${live.steps}` : ""}`);
  } else {
    setPill("idle", "Idle");
  }
  if (live.queued > 0) setPill("queued", `${live.queued} queued`);
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept);
  if (tab === "monitor") renderConvCache(m.conversation_cache);
  if (tab === "settings") renderAbout(eng, hw, st);
}

// #596: the conversation cache - the prompt's state the engine keeps between requests (always), and the whole
// conversations it parks in RAM when "--conversation-cache-mib N" is in the run config's args (opt-in)
function since(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${fmt(s / 60)} min ago` : `${fmt(s / 3600, 1)} h ago`;
}
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;                                  // an older server
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} GB`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? `${fmt(c.requests_reused)} of ${fmt(c.requests)} requests reused part of their prompt` : "";
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}% of all prompt tokens)` : "";
  const event = c.last_event ? `${c.last_event === "parked" ? "Parked" : "Restored"} ${fmt(c.last_tokens)} tokens, ${since(c.last_at)}` : null;
  facts($("cc-facts"), [
    ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : null],
    ["Reused since start", c.requests ? `${fmt(c.reused_tokens)} tokens${share}` : null],
    ["Parked / restored", c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}${c.evictions ? ` · ${fmt(c.evictions)} evicted` : ""}` : null],
    ["Last switch", c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full."
    : "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).";
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  return `Since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed}`;
}
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept) {
  // model state
  const on = live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on || b.dataset.s === live.state);
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`;
    } else {
      detail = `${fmt(live.prompt_tokens)} tokens`;
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (live.state === "media") {
    label = `Making ${live.kind === "video" ? "a video" : "an image"}`;
    delete prog.dataset.tone;
    pct = live.steps ? (100 * live.step) / live.steps : 0;
    detail = `${live.model}${live.steps ? ` · step ${live.step} of ${live.steps}` : ""} · ${fmt(live.elapsed_s, 0)} s`;
  } else if (last && last.kind) {
    delete prog.dataset.tone;
    detail = `last: ${last.kind}, ${last.output || ""} in ${fmt(last.duration_s, 0)} s`;
  } else if (last) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(last.output_tokens)} tokens${last.decode_tok_s ? ` at ${fmt(last.decode_tok_s, 1)} tok/s` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // the eight cards
  const speed = live.state === "generating" ? live.tok_s : last ? last.decode_tok_s : null;
  setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s",
            live.state === "generating" ? "Decode now" : last ? "Decode last request" : "Decode");
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
                : last && last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null;
  setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s",
            live.state === "reading" ? "Prefill now" : live.state === "generating" ? "Prefill this request" : last ? "Prefill last request" : "Prefill");
  spark("sp-speed", h.tok_s);
  spark("sp-prefill", h.prefill_tok_s_mean);
  // a model split across several cards (issue #112): the cards show their total / mean / hottest, and each card's own
  const per = (f) => (hw.gpus || []).map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const multi = (hw.gpus || []).length > 1;
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || (st.gpu_note ? "not available" : ""));
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`))
                  : eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : (st.gpu_note ? "not available on Windows AMD yet" : ""));
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            multi ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit` : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            hw.gpu_pcie_rx_mb == null ? "" : `to GPU ${fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s` +
            (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? ` · idle Gen${hw.gpu_pcie_gen}` : ""));
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : "needs psutil (setup installs it)");
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);

  // context fill: the running request, else the last one
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";         // a near-zero arc would draw just its round cap
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // recent requests
  const body = $("req-body");
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="9" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      if (r.kind) {
        // a media job: kind, model and what it made across the token columns
        const tm = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
        return `<tr><td>${esc(tm)}</td><td><span class="st-badge ${cls}">${esc(text)}</span> <span class="st-badge st-badge--reading">${esc(r.kind)}</span></td>
          <td colspan="6" class="muted">${esc(r.model || "")} · ${esc(r.output || "")}${r.steps ? ` · ${r.steps} steps` : ""}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
      }
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
      // #588: the VRAM share; the PCIe share (--pcie-frac) beside it when there is one
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
        (r.pcie_share ? ` <span class="muted" title="routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed">+${(r.pcie_share * 100).toFixed(1)}% PCIe</span>` : "");
      // Reused tokens are not prefetched again. Match the Monitor's existing Prefill metric.
      const fresh = r.prompt_tokens == null ? null : Math.max(0, r.prompt_tokens - (r.reused || 0));
      const prefillRate = fresh > 0 && r.prompt_ms > 0 ? fresh / (r.prompt_ms / 1000) : null;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(prefillRate)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return `${mode === "project" ? "Projection" : "Additive"} control vector on layers ${a}–${b}` +
         `${single ? ` (layer ${single.replace("single", "")}'s direction)` : ""}. Per chat in Sampling. Its package ` +
         "describes the vector as a refusal-direction projection; measure the speed yourself";
}
function renderAbout(eng, hw, st) {
  const kv = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit"}[eng.kv] || eng.kv;
  facts($("facts-engine"), [
    ["Model", eng.model],
    ["Engine", eng.version ? `v${eng.version}` : "built from source"],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", kv ? `${kv}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM` : ", all in VRAM"}` : null],
    ["Experts in VRAM", eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    ["Speculation", eng.spec ? `MTP drafts up to ${Math.max(0, (eng.mtp_max || eng.spec) - 1)} tokens${eng.lookup ? ", prompt lookup on" : ""}` : null],
    ["Images", eng.images ? "on" : "off"],
    ["Experimental speed projection", projectionText(eng.cvec)],
  ]);
  facts($("facts-hw"), [
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : (st.gpu_note || "not readable (NVML)")],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  const base = location.origin;
  facts($("facts-api"), [
    ["OpenAI base URL", `${base}/v1`, true],
    ["Anthropic base URL", base, true],
    ["Responses API (Codex)", `${base}/v1/responses`, true],
    ["Model name", eng.id || eng.model, true],
  ]);
  $("curl-example").textContent = `curl ${base}/v1/chat/completions \\\n  -H "Authorization: Bearer $AM2STRATA_KEY" -H "Content-Type: application/json" \\\n` +
    `  -d '{"model": "${eng.id || eng.model}", "messages": [{"role": "user", "content": "Hello"}]}'`;
  facts($("facts-server"), [
    ["Address", base],
    ["Server", "am2Strata (Rust)"],
    ["Model", eng.model ? `${eng.model}${eng.max_context ? ` · ${ctxfmt(eng.max_context)} context` : ""}` : "none loaded (the next request loads the default)"],
    ["API monitor", callsOn ? "on: the API calls tab keeps the last 100 calls in memory" : "off"],
    ["Video", health.video ? "frames cut on the server with ffmpeg; the web app decodes videos itself" : "the web app decodes videos itself; API clients need ffmpeg on the server"],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request,
// "strata_mcp": true, which only this page sends); the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    mcpInfo = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  renderMcp();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((mcpInfo.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
const MCP_STATE = {ready: ["st-badge--generating", "Connected"], starting: ["st-badge--reading", "Starting"],
                   failed: ["st-badge--error", "Failed"], stopped: ["st-badge--queued", "Stopped"], idle: ["", "Waiting"]};
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  $("mcp-empty").hidden = !!servers.length;
  $("mcp-row").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? `${fmt(mcpInfo.tools)} tools · ${ready.length} of ${servers.length} servers connected` : "";
  $("mcp-row-sub").textContent = mcpInfo.tools ? `${fmt(mcpInfo.tools)} tools from ${ready.map((s) => s.name).join(", ")}; the model calls them when it decides to`
                                               : "no server is connected yet (see the Monitor)";
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, text] = MCP_STATE[s.status] || ["", s.status];
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${fmt(s.tools.length)} tools${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ Model settings (GET / POST /config, #564)
// A few documented keys of the run config (strata-<model>.json), for every client, from the next start on.  The
// server lists them, checks every value and keeps every other key of the file as it is.
let cfgKeys = [];
// the run config's keys, in groups
const CFG_GROUPS = [["sampling", "Sampling defaults"], ["thinking", "Thinking"], ["context", "Context and long chats"],
                    ["memory", "Loading and memory"], ["server", "Server"], ["other", "Other"]];
function cfgGroup(key) {
  if (key.startsWith("sampling.")) return "sampling";
  if (/reasoning|thinking|effort/.test(key)) return "thinking";
  if (/max_tokens|drop_oldest|ctx|context/.test(key)) return "context";
  if (/unload|lazy|vram|kv|expert|silence/.test(key)) return "memory";
  if (/aliases|monitor|switch|browser|host|port|key/.test(key)) return "server";
  return "other";
}
async function loadConfig() {
  let r;
  try { r = await fetch("config", {headers: headers()}); } catch (e) { return; }
  if (!r.ok) { $("cfg-card").hidden = true; return; }       // no run config, an older server, or no key yet
  const c = await r.json();
  cfgKeys = c.keys || [];
  $("cfg-file").textContent = c.file || "";
  const rows = cfgKeys.map((k, i) => {
    const id = `cfg-${i}`, v = k.value;
    let input;
    if (k.kind === "bool" || k.kind === "enum") {
      const opts = k.kind === "bool" ? [["true", "on"], ["false", "off"]] : k.choices.map((x) => [x, x]);
      const cur = v == null ? "" : String(v);
      input = `<select class="st-input" id="${id}"><option value=""${cur === "" ? " selected" : ""}>default</option>` +
        opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
    } else {
      const text = v == null ? "" : Array.isArray(v) ? v.join(", ") : String(v);
      input = `<input class="st-input" id="${id}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
        `value="${esc(text)}" placeholder="default" autocomplete="off">`;
    }
    return {group: cfgGroup(k.key), html: `<div class="cfg-row"><label for="${id}"><span>${esc(k.help)}</span><code>${esc(k.key)}</code></label>${input}</div>`};
  }).reduce((acc, x) => { (acc[x.group] = acc[x.group] || []).push(x.html); return acc; }, {});
  $("cfg-form").innerHTML = CFG_GROUPS.filter(([g]) => rows[g]).map(([g, title]) =>
    `<fieldset class="cfg-group"><legend>${title}</legend>${rows[g].join("")}</fieldset>`).join("");
  $("cfg-card").hidden = false;
}
function configValue(k, el) {
  const s = el.value.trim();
  if (s === "") return null;
  if (k.kind === "bool") return s === "true";
  if (k.kind === "number") return Number(s);
  return s;                                         // enum, or names (the server splits them at commas)
}
$("cfg-reload").addEventListener("click", async () => {
  await saveConfig();
  const cur = $("model-select").dataset.current;
  if (cur && !$("cfg-msg").dataset.failed) { $("model-select").value = cur; reloadingModel = true; $("model-select").onchange(); }
});
let reloadingModel = false;
$("cfg-save").addEventListener("click", () => saveConfig());
async function saveConfig() {
  delete $("cfg-msg").dataset.failed;
  const set = {};
  cfgKeys.forEach((k, i) => {
    const v = configValue(k, $(`cfg-${i}`));
    const old = Array.isArray(k.value) ? k.value.join(", ") : k.value;
    if (JSON.stringify(v) !== JSON.stringify(old ?? null)) set[k.key] = v;
  });
  if (!Object.keys(set).length) { $("cfg-msg").textContent = "Nothing changed."; return; }
  try {
    const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set})});
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    $("cfg-msg").textContent = b.changed.length
      ? `Saved (${b.changed.join(", ")}); the earlier file is ${b.file}.bak. Start the model again to use it.` : "Nothing changed.";
    loadConfig();
  } catch (e) {
    $("cfg-msg").textContent = "";
    $("cfg-msg").dataset.failed = "1";
    toast("error", "Not saved", String(e.message || e), 6000);
  }
}

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
// what the canvas can do with a code block: preview HTML/SVG in a sandboxed frame, run Python on the server
function canvasKind(lang, code) {
  const l = (lang || "").toLowerCase();
  if (["html", "htm", "svg", "xml+svg"].includes(l)) return "html";
  if (!l && /^\s*(<!doctype html|<html|<svg)/i.test(code)) return "html";
  if (["python", "py", "python3"].includes(l)) return "python";
  if (l === "mermaid" || (!l && /^\s*(flowchart|graph|sequenceDiagram|stateDiagram|classDiagram|erDiagram|gantt|mindmap)\b/.test(code))) return "mermaid";
  return null;
}
function codeBlock(lang, code) {
  const kind = canvasKind(lang, code);
  const btn = kind === "html" || kind === "mermaid" ? `<button type="button" class="code-run" data-canvas="${kind}">${icon("eye")}Preview</button>`
            : kind === "python" ? `<button type="button" class="code-run" data-canvas="python" aria-label="Run this Python code" title="Runs on this PC in a sandbox: no network, home folders hidden">${icon("play")}Run</button>` : "";
  return `<div class="st-code"><div class="st-code__head"><span>${esc(lang || "code")}</span><span class="st-code__btns">${btn}` +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="Copy code">${icon("copy")}</button></span></div>` +
    `<pre><code>${esc(code)}</code></pre></div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ------------------------------------------------------------------ Chat
// The sampling shown until the user changes it is the model card's thinking set. Untouched, the page sends NO sampling
// fields: the server picks the card's set for the request's thinking state (thinking 1.0 / 0.95 / 20, not thinking
// 0.7 / 0.8 / 20 with a presence penalty). Before 0.1.45 the page sent 0.6 itself; a stored 0.6 with the other two
// values untouched counts as "not set" too.
const DEFAULTS = {thinking: "high", temperature: 1.0, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, mcp: true, flows: true};
const OLD_DEFAULT_TEMPERATURE = 0.6;
function samplingUntouched(s) {
  const t = +s.temperature;
  return (t === DEFAULTS.temperature || t === OLD_DEFAULT_TEMPERATURE) && +s.top_p === DEFAULTS.top_p && +s.top_k === DEFAULTS.top_k;
}
let settings = {...DEFAULTS, ...store.get("sampling", {})};
if (samplingUntouched(settings)) settings.temperature = DEFAULTS.temperature;
// chats: [{id, title, time, model, messages}], the newest first; the one on screen is `current`
let chats = store.get("chats", null);
if (!Array.isArray(chats)) {                          // the one chat of before (the "chat" key) becomes the first
  const old = store.get("chat", []);
  chats = old.length ? [{id: Date.now().toString(36), title: "", time: Date.now(), model: "", messages: old}] : [];
}
let current = store.get("current-chat", chats[0] ? chats[0].id : null);
if (!chats.some((c) => c.id === current)) current = chats[0] ? chats[0].id : null;
let messages = current ? chats.find((c) => c.id === current).messages : [];
let attachments = [];                 // {kind: "image"|"file"|"video"|"pdf", name, url | text | frames}
let busy = null;                      // {controller, msg, cutoff?}

const slim = (list) => list.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name})),
                                           files: (m.files || []).map((f) => ({name: f.name})),
                                           media: (m.media || []).map((x) => ({kind: x.kind, name: x.name}))}));
function chatTitle(list) {
  const first = list.find((m) => m.role === "user");
  if (!first) return "New chat";
  const t = (first.text || "").replace(/\s+/g, " ").trim();
  return t ? (t.length > 48 ? t.slice(0, 46) + "…" : t) : ((first.media || [])[0] || (first.images || [])[0] || (first.files || [])[0] || {name: "Attachment"}).name;
}
function saveChat() {
  if (messages.length) {
    let c = chats.find((x) => x.id === current);
    if (!c) { c = {id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), model: health.model}; chats.unshift(c); current = c.id; }
    c.messages = messages; c.title = chatTitle(messages); c.time = Date.now(); c.model = c.model || health.model;
  }
  store.set("chats", chats.slice(0, 200).map((c) => ({...c, messages: slim(c.messages)})));
  store.set("current-chat", current);
  renderChatList();
}
function newChat() {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return false; }
  messages = []; current = null; attachments = []; renderAttachments();
  store.set("current-chat", null);
  renderChat(); renderChatList(); openChats(false);
  return true;
}
function openChat(id) {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  const c = chats.find((x) => x.id === id);
  if (!c) return;
  current = id; messages = c.messages; store.set("current-chat", id);
  renderChat(); renderChatList(); openChats(false);
}
function deleteChat(id) {
  const i = chats.findIndex((x) => x.id === id);
  if (i < 0) return;
  if (busy && id === current) { toast("warn", "Still writing", "Stop the answer first."); return; }
  const [gone] = chats.splice(i, 1);
  if (id === current) { messages = []; current = null; renderChat(); }
  store.set("chats", chats.map((c) => ({...c, messages: slim(c.messages)})));
  renderChatList();
  toast("info", "Chat deleted", gone.title || "", 6000, {label: "Undo", run: () => { chats.splice(i, 0, gone); saveChat(); }});
}
function renderChatList() {
  const q = ($("chat-search").value || "").toLowerCase();
  const list = chats.filter((c) => !q || (c.title || "").toLowerCase().includes(q) ||
                                   c.messages.some((m) => (m.text || "").toLowerCase().includes(q)));
  const day = (t) => { const d = new Date(t), now = new Date(); const diff = Math.floor((new Date(now.toDateString()) - new Date(d.toDateString())) / 864e5);
                       return diff <= 0 ? "Today" : diff === 1 ? "Yesterday" : diff < 7 ? "This week" : "Older"; };
  let group = null, html = "";
  for (const c of list) {
    const g = day(c.time);
    if (g !== group) { html += `<div class="chat-group">${g}</div>`; group = g; }
    const n = c.messages.filter((m) => m.role === "user").length;
    html += `<div class="chat-item" aria-current="${c.id === current}"><button type="button" data-chat="${esc(c.id)}"><b>${esc(c.title || "New chat")}</b>` +
      `<span>${esc(c.model || "")}${c.model ? " · " : ""}${n} message${n === 1 ? "" : "s"}</span></button>` +
      `<button type="button" class="st-btn st-btn--icon" data-del="${esc(c.id)}" aria-label="Delete this chat">${icon("trash")}</button></div>`;
  }
  $("chat-list").innerHTML = html || `<p class="muted small">${q ? "No chat matches." : "No chats yet."}</p>`;
}
$("chat-list").addEventListener("click", (e) => {
  const d = e.target.closest("[data-del]"); if (d) { deleteChat(d.dataset.del); return; }
  const c = e.target.closest("[data-chat]"); if (c) openChat(c.dataset.chat);
});
$("chat-search").addEventListener("input", renderChatList);
function openChats(open) { $("chats").dataset.open = String(open); $("menu-btn").setAttribute("aria-expanded", String(open)); }
$("menu-btn").onclick = () => { if (tab !== "chat") showTab("chat"); openChats($("chats").dataset.open !== "true"); };
for (const b of $("starters").children) b.onclick = () => { $("input").value = b.dataset.q; autosize(); $("input").focus();
  if (/picture|PDF|video|file/.test(b.dataset.q)) $("file").click(); };
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.media && m.media.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const x of m.media) {
        if (x.kind === "video" && x.frames) {        // the frames the model reads, in order
          const box = document.createElement("div");
          box.className = "msg-media";
          box.innerHTML = `<span><b></b> · ${x.duration.toFixed(1)} s · ${x.frames.length} frames sent in order</span><div class="frame-strip"></div>`;
          box.querySelector("b").textContent = x.name;
          for (const u of x.frames.filter((_, k) => k % 2 === 0)) { const im = document.createElement("img"); im.src = u; im.alt = ""; box.lastChild.appendChild(im); }
          wrap.appendChild(box);
          continue;
        }
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon(x.kind === "video" ? "image" : "attach", "st-icon st-icon--sm");
        c.append(x.kind === "pdf" ? `${x.name} · read as text and page pictures` : x.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || "image"; wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || "image"); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = `You · ${timeStr(m.time)}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy the answer" title="Copy">${icon("copy")}</button>` +
      `<button class="st-btn st-btn--icon" data-msg-speak aria-label="Read aloud" title="Read aloud" hidden>${icon("volume")}</button>` +
      `<button class="st-btn st-btn--icon" data-msg-regen aria-label="Regenerate" title="Regenerate">↻</button>` +
      `<button class="st-btn st-btn--icon" data-msg-prev aria-label="Previous variant" title="Previous variant">‹</button>` +
      `<span class="swipe-count" data-swipe-count></span>` +
      `<button class="st-btn st-btn--icon" data-msg-next aria-label="Next variant" title="Next variant">›</button></div>`;
    updateAssistant(el, m, false);
  }
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", "Writing"], running: ["st-badge--generating", "Running"], done: ["", "Done"],
                    error: ["st-badge--error", "Error"], skipped: ["st-badge--queued", "Not run"]};
function toolHtml(t, k) {
  const [cls, label] = TOOL_STATE[t.state] || ["", t.state];
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool-call__label">Arguments</div><pre class="tool-call__pre">${esc(args || "(being written)")}</pre>`;
    if (t.result != null) {
      body += `<div class="tool-call__label">${t.ok ? "Result" : "Error"}${t.chars ? ` · ${fmt(t.chars)} characters` : ""}` +
              `${t.truncated ? ", cut for the model" : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// the answer's text with the tool blocks where the model called them
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return markdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += markdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k);
  });
  return html + markdown(m.text.slice(pos));
}
// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms,
                      state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
  }
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? "Thinking…" :
      m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts";
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = answerHtml(m);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? "Stopped" : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
  const sp = el.querySelector("[data-msg-speak]");
  if (sp) sp.hidden = streaming || !m.text || !(media.models || []).some((x) => x.kind === "audio" && x.task === "tts" && x.downloaded);
  // variants (PolyStrata PR 74): regenerate off while anything streams; the arrows when there is more than one
  const n = Array.isArray(m.swipes) ? m.swipes.length : 1, idx = m.swipeIndex || 0;
  const rg = el.querySelector("[data-msg-regen]"), pv = el.querySelector("[data-msg-prev]");
  const nx = el.querySelector("[data-msg-next]"), cnt = el.querySelector("[data-swipe-count]");
  if (rg) rg.disabled = streaming || !!busy;
  for (const b of [pv, nx, cnt]) if (b) b.hidden = streaming || n < 2;
  if (cnt) cnt.textContent = n > 1 ? `${idx + 1}/${n}` : "";
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  scrollDown(true);
}
function nearBottom() { const s = $("chat-scroll"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }
function scrollDown(force) { const s = $("chat-scroll"); if (force || nearBottom()) s.scrollTop = s.scrollHeight; }

$("chat").addEventListener("click", (e) => {
  const cv = e.target.closest("[data-canvas]");
  if (cv) { openCanvas(cv.dataset.canvas, cv.closest(".st-code").querySelector("pre").textContent); return; }
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  const rg = e.target.closest("[data-msg-regen]");
  if (rg) { regenerateMessage(+rg.closest(".st-msg").dataset.i); return; }
  const sk = e.target.closest("[data-msg-speak]");
  if (sk) { readAloud(+sk.closest(".st-msg").dataset.i, sk); return; }
  const pv = e.target.closest("[data-msg-prev]");
  if (pv) { swipeGo(+pv.closest(".st-msg").dataset.i, -1); return; }
  const nx = e.target.closest("[data-msg-next]");
  if (nx) { swipeGo(+nx.closest(".st-msg").dataset.i, +1); return; }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

function apiMessages() {
  const out = [];
  // regenerating an earlier answer: the history up to it (the later messages stay in the chat, not in this request)
  const end = busy && busy.cutoff != null ? busy.cutoff + 1 : messages.length;
  for (let k = 0; k < end; k++) {
    const m = messages[k];
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const media = (m.media || []).filter((x) => x.frames || x.url);
      const text = userText(m);
      const parts = [{type: "text", text}, ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))];
      for (const x of media) {
        if (x.kind === "video") {          // the browser's frames, in order: how MiniCPM-V and the others read a video
          parts.push({type: "text", text: `[A video, "${x.name}", ${x.duration.toFixed(1)} s, as ${x.frames.length} frames in order:]`});
          parts.push(...x.frames.map((u) => ({type: "image_url", image_url: {url: u}})));
        } else {                           // the server reads the PDF: its text, and its pages as pictures
          parts.push({type: "file", file: {filename: x.name, file_data: x.url}});
        }
      }
      out.push({role: "user", content: parts.length > 1 ? parts : text});
    } else if (!(busy && busy.msg === m)) {            // the answer being asked for now is not history yet
      out.push(...assistantMessages(m));
    }
  }
  return out;
}
// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  // #1392: a turn with no answer text (only reasoning, a stop before the first content token, or an error) still goes
  // back as an assistant turn, so the history keeps alternating and a reasoning model sees that it already answered.
  // The server leaves such a turn out of the prompt itself (serve/frontend.py, #843); reasoning_content rides along.
  if (!ran.length) {
    const msg = {role: "assistant", content: m.text || ""};
    if (!m.text && m.reasoning) msg.reasoning_content = m.reasoning;
    return [msg];
  }
  const out = [];
  let pos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) out.push({role: "tool", tool_call_id: t.id, content: t.result});
    pos = at;
  }
  const rest = m.text.slice(pos).trim();
  if (rest) out.push({role: "assistant", content: rest});
  return out;
}

function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on;
  $("composer-hint").textContent = on ? "" : "Shift+Enter: new line";
}

async function send() {
  const text = $("input").value.trim();
  if ((!text && !attachments.length) || busy) return;
  messages.push({role: "user", text, images: attachments.filter((a) => a.kind === "image"),
                 files: attachments.filter((a) => a.kind === "file"),
                 media: attachments.filter((a) => a.kind === "video" || a.kind === "pdf"), time: Date.now()});
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m};
  setBusy(true);

  try {
    await streamInto(m, el, controller.signal);
  } catch (e) {
    m.error = e.message || String(e);
    toast("error", "The request failed", m.error, 6000);
  }
  busy = null;
  setBusy(false);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
  if (!m.error && !m.stopped) afterAnswer(m);
}

// ------------------------------------------------------------------ variants (PolyStrata PR 74: swipes)
// Each answer keeps its variants: m.text / reasoning / tools / meta hold the current one, m.swipes[m.swipeIndex]
// mirrors it. Regenerating drops the variants after the current one and adds the new one at the end.
function swipeSnapshot(m) {
  return {text: m.text || "", reasoning: m.reasoning || "", tools: m.tools || null, meta: m.meta || null};
}
function swipeInit(m) {
  if (!Array.isArray(m.swipes) || !m.swipes.length) { m.swipes = [swipeSnapshot(m)]; m.swipeIndex = 0; }
}
function swipeSave(m) { swipeInit(m); m.swipes[m.swipeIndex] = swipeSnapshot(m); }
function swipeLoad(m) {
  swipeInit(m);
  const v = m.swipes[m.swipeIndex] || {};
  Object.assign(m, {text: v.text || "", reasoning: v.reasoning || "", tools: v.tools || null, meta: v.meta || null,
                    thinkSecs: null, stopped: false, error: null, limit: null});
}
function swipeGo(i, delta) {
  if (busy) return;
  const m = messages[i];
  if (!m || m.role !== "assistant") return;
  swipeInit(m);
  const n = m.swipes.length;
  if (n < 2) return;
  swipeSave(m);
  m.swipeIndex = (m.swipeIndex + delta + n) % n;
  swipeLoad(m);
  const el = $("chat").querySelector(`.st-msg[data-i="${i}"]`);
  if (el) updateAssistant(el, m, false);
  saveChat();
}
// A new answer at messages[i], kept beside the old one; on failure the old one comes back.
async function regenerateMessage(i) {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  const m = messages[i];
  if (!m || m.role !== "assistant") return;
  const el = $("chat").querySelector(`.st-msg[data-i="${i}"]`);
  if (!el) return;
  swipeSave(m);
  m.swipes.length = m.swipeIndex + 1;
  const keep = swipeSnapshot(m);
  Object.assign(m, {text: "", reasoning: "", tools: null, meta: null, thinkSecs: null, stopped: false, error: null, limit: null});
  const controller = new AbortController();
  busy = {controller, msg: m, cutoff: i};
  if (i < messages.length - 1) {
    toast("info", "Regenerating an earlier answer",
          `${messages.length - i - 1} later message(s) stay in the chat; this request sees the chat up to here.`, 7000);
  }
  setBusy(true);
  updateAssistant(el, m, true);
  try {
    await streamInto(m, el, controller.signal);
    m.swipes.push(swipeSnapshot(m));
    m.swipeIndex = m.swipes.length - 1;
  } catch (e) {
    toast("error", "Regeneration failed", e.message || String(e), 6000);
    Object.assign(m, keep, {error: null});
    m.swipes[m.swipeIndex] = keep;
  }
  busy = null;
  setBusy(false);
  updateAssistant(el, m, false);
  saveChat();
}

// The request and its stream into answer `m` (shared by send and regenerate). A Stop returns normally with m.stopped
// set (the partial answer is kept); any other failure is thrown.
async function streamInto(m, el, signal) {
  const body = {model: health.model, messages: withHints(apiMessages()), stream: true,
                reasoning_effort: settings.thinking};
  if (!(settings.temperature > 0)) {
    body.temperature = 0;
  } else if (!samplingUntouched(settings)) {
    Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  }
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;   // this server may run MCP tools for it
  if (store.get("memory-web", true)) body.strata_memory = true;              // Settings > Memory

  let firstAt = null, thinkStart = null, usage = null, frame = 0;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body), signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = "This server needs an API key: add it under About > Settings.";
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        if (j.error) throw new Error(j.error.message || "the engine reported an error");
        if (j.usage) usage = j.usage;
        if (j.strata_mcp) onTool(m, j.strata_mcp);
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (frame) cancelAnimationFrame(frame);
    if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
    if (e.name !== "AbortError") throw e;
    m.stopped = true;
  }
  if (frame) cancelAnimationFrame(frame);
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    m.meta = `${fmt(n)} tokens${secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : ""}${m.stopped ? " · stopped" : ""}` +
             (projectionLoaded() ? (settings.esp ? " · projection on" : " · projection off") : "");
  } else if (m.stopped) {
    m.meta = "Stopped";
  }
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) m.meta = `${m.meta ? `${m.meta} · ` : ""}${ran} tool call${ran > 1 ? "s" : ""}`;
  if (m.limit) m.meta = `${m.meta || ""} · stopped at the limit of ${m.limit} tool rounds (mcp.max_rounds)`;
  updateAssistant(el, m, false);
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);

$("new-btn").onclick = () => { if (newChat()) $("input").focus(); };
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", "Nothing to save yet"); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `am2strata-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
const VIDEO_FRAMES = 16;              // frames a video is sent as (evenly spread over it)
// A video's frames, decoded by the browser itself (no ffmpeg needed on the server), as JPEG data URLs
async function videoFrames(file, n) {
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.muted = true; v.preload = "auto"; v.playsInline = true; v.src = url;
  try {
    await new Promise((ok, bad) => { v.onloadeddata = ok; v.onerror = () => bad(new Error("this browser cannot play this video format")); });
    let dur = v.duration;
    if (!isFinite(dur)) {                // a recorder's WebM says no length: seeking past the end makes the browser find it
      v.currentTime = 1e101;
      await new Promise((ok) => { v.ontimeupdate = () => { v.ontimeupdate = null; ok(); }; setTimeout(ok, 3000); });
      dur = v.duration;
      v.currentTime = 0;                 // wait for this seek too, or the first frame is the video's end
      await new Promise((ok) => { v.onseeked = ok; setTimeout(ok, 3000); });
    }
    if (!isFinite(dur) || dur <= 0) throw new Error("the video's length could not be read");
    const scale = Math.min(1, 768 / Math.max(v.videoWidth, v.videoHeight));
    const c = document.createElement("canvas");
    c.width = Math.max(2, Math.round(v.videoWidth * scale)); c.height = Math.max(2, Math.round(v.videoHeight * scale));
    const g = c.getContext("2d"), frames = [];
    for (let i = 0; i < n; i++) {
      const at = Math.min(dur - 0.01, dur * (i + 0.5) / n);
      // the handler first, then the seek; a seek to where it already is fires no event, hence the timeout
      await new Promise((ok) => { const t = setTimeout(ok, 2000); v.onseeked = () => { clearTimeout(t); ok(); }; v.currentTime = at; });
      g.drawImage(v, 0, 0, c.width, c.height);
      frames.push(c.toDataURL("image/jpeg", 0.8));
    }
    return {frames, duration: dur};
  } finally { URL.revokeObjectURL(url); }
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("video/") || /\.(mp4|m4v|mov|webm|mkv|avi|ogv)$/i.test(f.name)) {
      if (!health.images) { toast("warn", "Video needs a picture model", "This model reads text only: load one that reads pictures (MiniCPM-V)."); continue; }
      const chip = {kind: "video", name: f.name, frames: null, duration: 0};
      toast("info", "Reading the video", `${f.name}: taking ${VIDEO_FRAMES} frames…`, 2500);
      videoFrames(f, VIDEO_FRAMES).then((r) => { Object.assign(chip, r); attachments.push(chip); renderAttachments(); })
        .catch((e) => toast("error", "Video not read", `${f.name}: ${e.message}. Try an MP4 (H.264) or WebM file.`, 6000));
      continue;
    }
    if (f.type === "application/pdf" || /\.pdf$/i.test(f.name)) {
      if (f.size > 50e6) { toast("warn", "PDF too large", `${f.name} is over 50 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "pdf", name: f.name, url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", "Pictures are off", "This model was set up for text only."); continue; }
      if (f.size > 20e6) { toast("warn", "Picture too large", `${f.name} is over 20 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || "pasted image", url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", "Not a supported file", `${f.name}: attach text files (code, notes, logs, data), PDFs${health.images ? ", pictures or videos" : ""}.`); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", "File too large", `${f.name} is over 512 KB.`); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = a.kind === "image" ? `<img src="${a.url}" alt="">` : a.kind === "video" ? `<img src="${a.frames[0]}" alt="">`
                                                      : icon("attach", "st-icon st-icon--sm");
    c.append(a.kind === "video" ? `${a.name} · ${a.frames.length} frames · ${a.duration.toFixed(0)} s` : a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", "Remove");
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat-scroll", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  if (!health.images) return;
  const files = [...(e.clipboardData || {}).files || []].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ------------------------------------------------------------------ the sampling drawer
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) { loadDrawer(); loadShared(); loadMcp(); }
}
function loadDrawer(s = settings) {
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-esp").setAttribute("aria-checked", String(s.esp !== false));
  $("esp-row").hidden = !projectionLoaded();
  $("s-mcp").setAttribute("aria-checked", String(s.mcp !== false));
  $("s-flows").setAttribute("aria-checked", String(s.flows !== false));
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  outputs();
}
// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false;
async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) sharedOn = !!(await r.json()).shared;
  } catch (e) { /* an older server: the switch just stays off */ }
  $("s-share").setAttribute("aria-checked", String(sharedOn));
}
function sharedDefaults(s) {
  const d = {reasoning_effort: s.thinking};
  if (!(+s.temperature > 0)) d.temperature = 0;
  else if (!samplingUntouched(s)) Object.assign(d, {temperature: +s.temperature, top_p: +s.top_p, top_k: +s.top_k});
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  if (projectionLoaded()) d.experimental_speed_projection = s.esp !== false;
  return d;
}
async function saveShared(on, s) {
  const r = await fetch("settings", {method: "POST", headers: headers(true),
                                      body: JSON.stringify({defaults: on ? sharedDefaults(s) : null})});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  sharedOn = !!(await r.json()).shared;
}
// the engine was started with the experimental-speed-projection control vector (INFO cvec=...)
function projectionLoaded() {
  const c = lastMetrics && lastMetrics.engine ? lastMetrics.engine.cvec : 0;
  return !!c && c !== "0";
}
function outputs() {
  const t = +$("s-temp").value;
  $("o-temp").textContent = t === 0 ? "0 · greedy" : t.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  $("o-topk").textContent = $("s-topk").value;
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  $("o-thinking").textContent = sel ? {none: "answers right away", low: "short", medium: "medium", high: "thorough (default)"}[sel.dataset.v] : "";
  for (const id of ["s-topp", "s-topk"]) $(id).disabled = t === 0;
}
for (const b of $("s-thinking").children) b.onclick = () => { for (const x of $("s-thinking").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const id of ["s-temp", "s-topp", "s-topk"]) $(id).oninput = outputs;
$("s-show").onclick = () => $("s-show").setAttribute("aria-checked", String($("s-show").getAttribute("aria-checked") !== "true"));
$("s-esp").onclick = () => $("s-esp").setAttribute("aria-checked", String($("s-esp").getAttribute("aria-checked") !== "true"));
$("s-mcp").onclick = () => $("s-mcp").setAttribute("aria-checked", String($("s-mcp").getAttribute("aria-checked") !== "true"));
$("s-flows").onclick = () => $("s-flows").setAttribute("aria-checked", String($("s-flows").getAttribute("aria-checked") !== "true"));
$("s-share").onclick = () => $("s-share").setAttribute("aria-checked", String($("s-share").getAttribute("aria-checked") !== "true"));
$("s-reset").onclick = () => loadDrawer(DEFAULTS);
$("s-apply").onclick = async () => {
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  settings = {thinking: sel ? sel.dataset.v : "high", temperature: +$("s-temp").value, top_p: +$("s-topp").value,
              top_k: +$("s-topk").value, max: $("s-max").value.trim(), seed: $("s-seed").value.trim(),
              show: $("s-show").getAttribute("aria-checked") === "true",
              esp: $("s-esp").getAttribute("aria-checked") === "true",
              mcp: $("s-mcp").getAttribute("aria-checked") === "true",
              flows: $("s-flows").getAttribute("aria-checked") === "true"};
  store.set("sampling", settings);
  renderThinking(); loadDefaults();
  const share = $("s-share").getAttribute("aria-checked") === "true";
  openDrawer(false);
  if (share || sharedOn) {
    try {
      await saveShared(share, settings);
      toast("success", "Sampling saved", share ? "Other apps (omp, API clients) use these settings from their next request."
                                               : "Other apps use their own settings again.");
    } catch (e) {
      toast("error", "Saved here, but not for other apps", e.message, 6000);
    }
    return;
  }
  toast("success", "Sampling saved", settings.temperature === 0 ? "Greedy: the same question gives the same answer." : "");
};
function renderThinking() {
  for (const g of ["c-thinking", "d-thinking"]) for (const b of $(g).children) b.setAttribute("aria-checked", String(b.dataset.v === settings.thinking));
}
for (const b of $("c-thinking").children) b.onclick = () => { settings.thinking = b.dataset.v; store.set("sampling", settings); renderThinking();
  if (sharedOn) saveShared(true, settings).catch(() => {}); };
renderThinking();
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });

// ------------------------------------------------------------------ Settings: sections
let section = store.get("settings-section", "models");
function showSection(name) {
  section = ["models", "config", "defaults", "server", "tools", "memory", "appearance", "connect"].includes(name) ? name : "models";
  store.set("settings-section", section);
  for (const b of $("set-nav").children) b.setAttribute("aria-current", String(b.dataset.sec === section));
  for (const p of document.querySelectorAll(".set-panel")) p.hidden = p.id !== `sec-${section}`;
  if (section === "models") { loadModels(); hubStatus(); }
  if (section === "config") loadConfig();
  if (section === "defaults") { loadShared().then(loadDefaults); }
  if (section === "tools") loadMcp();
  if (section === "memory") loadMemory();
  if (lastMetrics) render(lastMetrics);
}
for (const b of $("set-nav").children) b.onclick = () => showSection(b.dataset.sec);

// Settings › Chat defaults: the same settings as the drawer (this browser), optionally shared with other apps
function loadDefaults() {
  renderThinking();
  $("d-max").value = settings.max || ""; $("d-seed").value = settings.seed || "";
  $("d-share").setAttribute("aria-checked", String(sharedOn));
}
for (const b of $("d-thinking").children) b.onclick = () => { for (const x of $("d-thinking").children) x.setAttribute("aria-checked", String(x === b)); };
$("d-share").onclick = () => $("d-share").setAttribute("aria-checked", String($("d-share").getAttribute("aria-checked") !== "true"));
$("d-save").onclick = async () => {
  const sel = [...$("d-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  settings = {...settings, thinking: sel ? sel.dataset.v : settings.thinking, max: $("d-max").value.trim(), seed: $("d-seed").value.trim()};
  store.set("sampling", settings);
  renderThinking();
  const share = $("d-share").getAttribute("aria-checked") === "true";
  if (share || sharedOn) {
    try { await saveShared(share, settings); toast("success", "Defaults saved", share ? "Other apps use them from their next request." : "Other apps use their own settings again."); }
    catch (e) { toast("error", "Saved here, but not for other apps", e.message, 6000); }
    return;
  }
  toast("success", "Defaults saved", "For this browser.");
};

// Settings › Models: add one from Hugging Face (POST /api/hub/quants, POST + GET /api/hub/pull)
async function hubCall(path, body) {
  const r = await fetch(path, body ? {method: "POST", headers: headers(true), body: JSON.stringify(body)} : {headers: headers(), cache: "no-store"});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((j.error && j.error.message) || `HTTP ${r.status}`);
  return j;
}
const bytes = (n) => n >= 1073741824 ? `${gb(n)} GB` : `${fmt(n / 1048576)} MB`;
$("hf-repo").oninput = () => { $("hf-quant").disabled = true; $("hf-quant").innerHTML = "<option>List the repository first</option>"; $("hf-pull").disabled = true; };
$("hf-form").onsubmit = async (e) => {
  e.preventDefault();
  const repo = $("hf-repo").value.trim();
  if (!repo) { $("hf-repo").focus(); return; }
  $("hf-list").disabled = true; $("hf-msg").textContent = "Asking Hugging Face…";
  try {
    const j = await hubCall("api/hub/quants", {repo, token: $("hf-token").value.trim() || undefined});
    const pick = j.quants.find((q) => /Q4_K_M/i.test(q.quant)) || j.quants[0];
    $("hf-quant").innerHTML = j.quants.map((q) => `<option value="${esc(q.quant)}"${q === pick ? " selected" : ""}>${esc(q.quant)} · ${bytes(q.size)}${q.files > 1 ? ` · ${q.files} files` : ""}</option>`).join("");
    $("hf-quant").disabled = false; $("hf-pull").disabled = false;
    $("hf-msg").textContent = `${j.repo}: ${j.quants.length} quantization${j.quants.length === 1 ? "" : "s"}`;
  } catch (err) { $("hf-msg").textContent = ""; toast("error", "Not listed", String(err.message || err), 8000); }
  $("hf-list").disabled = false;
};
$("hf-pull").onclick = async () => {
  $("hf-pull").disabled = true;
  try {
    await hubCall("api/hub/pull", {repo: $("hf-repo").value.trim(), quant: $("hf-quant").value, token: $("hf-token").value.trim() || undefined});
    hubStatus();
  } catch (err) { $("hf-pull").disabled = false; toast("error", "Not started", String(err.message || err), 8000); }
};
let hubTimer = null, hubWas = null;
async function hubStatus() {
  clearTimeout(hubTimer);
  let j;
  try { j = await hubCall("api/hub/pull"); } catch (e) { return; }
  const running = j.state === "running";
  $("hf-progress").hidden = !running;
  if (running) {
    const pct = j.total ? (100 * j.done / j.total) : 0;
    $("hf-file").textContent = j.file || `${j.repo} ${j.quant}`;
    $("hf-rate").textContent = j.total ? `${bytes(j.done)} of ${bytes(j.total)} · ${bytes(j.speed)}/s` : "starting…";
    $("hf-bar").style.width = `${pct.toFixed(1)}%`;
    $("hf-msg").textContent = `Downloading ${j.repo} ${j.quant}`;
    $("hf-pull").disabled = true;
    hubTimer = setTimeout(hubStatus, 1000);
  } else if (hubWas === "running") {
    $("hf-pull").disabled = false;
    $("hf-msg").textContent = j.message || "";
    if (j.state === "done") { toast("success", "Download finished", j.message || "", 10000); loadModels(); }
    else toast("error", "Download failed", j.message || "", 10000);
  }
  hubWas = j.state;
}

// ------------------------------------------------------------------ API calls (the API monitor's /api/requests)
let callsOn = true, callSel = null, callDetail = null, callPane = store.get("call-pane", "output"), calls = [], callsTimer = null;
const secs = (n) => typeof n === "number" ? `${n.toFixed(2)} s` : "–";
const callActive = (r) => !["completed", "error", "disconnected"].includes(r.state);
const callTone = (st) => st === "completed" ? "st-badge--generating" : st === "error" ? "st-badge--error" : st === "disconnected" ? "" : "st-badge--queued";
async function refreshCalls() {
  clearTimeout(callsTimer);
  if (tab !== "calls") return;
  try {
    const r = await fetch("api/requests", {headers: headers(), cache: "no-store"});
    if (r.status === 404) { callsOn = false; $("calls-off").hidden = false; $("calls-list").innerHTML = ""; return; }
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    callsOn = true; $("calls-off").hidden = true;
    calls = (await r.json()).requests || [];
    renderCalls();
    if (callSel) await showCall(callSel, true);
  } catch (e) { /* the server restarts: try again */ }
  callsTimer = setTimeout(refreshCalls, 2000);
}
let callsStamp = "";
function renderCalls() {
  const q = $("calls-search").value.toLowerCase(), f = $("calls-filter").value;
  const list = calls.filter((r) => `${r.id} ${r.path} ${r.model}`.toLowerCase().includes(q) && (f === "all" || (f === "active" ? callActive(r) : r.state === f)));
  const stamp = JSON.stringify([list, callSel]);
  if (stamp === callsStamp) return;
  callsStamp = stamp;
  $("calls-list").innerHTML = list.length ? list.map((r) => {
    const sp = r.timings && r.timings.predicted_per_second;
    return `<button type="button" class="call-item" data-call="${esc(r.id)}" aria-current="${r.id === callSel}">` +
      `<span class="call-item__top"><span>${new Date(r.started_at * 1000).toLocaleTimeString()}</span><span class="st-badge ${callTone(r.state)}">${esc(r.state)}</span></span>` +
      `<span class="call-item__path">POST ${esc(r.path)}</span>` +
      `<span class="call-item__meta">${esc(r.model || "")} · ${secs(r.wallclock_s)}${sp ? ` · ${sp.toFixed(1)} tok/s` : ""}` +
      `${r.stream ? " · stream" : ""}${r.http_status && r.http_status !== 200 ? ` · HTTP ${r.http_status}` : ""}</span></button>`;
  }).join("") : `<p class="muted small">${calls.length ? "No call matches." : "No API calls yet: they appear here as clients send them."}</p>`;
}
$("calls-list").addEventListener("click", (e) => { const b = e.target.closest("[data-call]"); if (b) showCall(b.dataset.call); });
$("calls-search").oninput = renderCalls; $("calls-filter").onchange = renderCalls;
async function showCall(id, quiet = false) {
  callSel = id; if (!quiet) renderCalls();
  let v;
  try { v = await (await fetch(`api/requests?id=${encodeURIComponent(id)}`, {headers: headers(), cache: "no-store"})).json(); } catch (e) { return; }
  if (callSel !== id || !v || v.error && !v.id) return;
  callDetail = v;
  $("calls-empty").hidden = true; $("call-body").hidden = false;
  $("call-title").textContent = `${v.id} · ${v.path}`;
  $("call-state").textContent = v.state; $("call-state").className = `st-badge ${callTone(v.state)}`;
  const u = v.usage || {};
  const cell = (k, val) => `<div><span class="muted small">${k}</span><b>${esc(val)}</b></div>`;
  $("call-facts").innerHTML = [
    cell("Started", new Date(v.started_at * 1000).toLocaleTimeString()),
    cell("Wall time", secs(v.wallclock_s ?? (Date.now() / 1000 - v.started_at))),
    cell("Load", secs(v.load_s)), cell("Queue", secs(v.queue_s)), cell("First token", secs(v.first_token_s)),
    cell("Prompt", u.prompt_tokens ?? u.input_tokens ?? "–"), cell("Output", u.completion_tokens ?? u.output_tokens ?? "–"),
    cell("Speed", v.timings && v.timings.predicted_per_second ? `${v.timings.predicted_per_second.toFixed(1)} tok/s` : "–"),
    cell("Format", v.response_format || "text"),
  ].join("") + (v.error ? `<div class="call-error">${esc(v.error.message || "")}</div>` : "");
  callContent();
}
function callContent() {
  for (const b of $("call-tabs").children) b.setAttribute("aria-selected", String(b.dataset.p === callPane));
  if (!callDetail) return;
  let t = callDetail[callPane] || (callPane === "response" && callDetail.stream ? "A streamed response: see Output." : "Nothing.");
  try { t = JSON.stringify(JSON.parse(t), null, 2); } catch (e) { /* plain text */ }
  if (callDetail[`${callPane}_truncated`]) t += "\n\n[the monitor kept the first 256K characters; the client got all of it]";
  if ($("call-content").textContent !== t) $("call-content").textContent = t;
}
for (const b of $("call-tabs").children) b.onclick = () => { callPane = b.dataset.p; store.set("call-pane", callPane); callContent(); };

// ------------------------------------------------------------------ the canvas: what an answer makes, beside the chat
// Its controls follow what it shows (data-mode):
//   html    a page from a code block: Preview, Code, and Console when the page has issues (they can go back to the
//           model). Only a refresh button: no address bar.
//   flow    a Mermaid diagram (a ```mermaid block, or a flow the model drew because it was asked how something works):
//           Flow, Source, Sketch; exports to Markdown, PNG, PDF, and GIF when the arrows are animated.
//   python  a run on this PC (POST /api/run, bubblewrap sandbox, only on a click): Output, Code.
//   web     another site (a link in an answer): back, forward, refresh, an address bar, open in a new tab.
// Pages run in sandboxed frames without this page's origin: they cannot reach the server's API or this browser's
// storage. Our own pages go in inline (srcdoc) into a fresh frame each time; they also get an address on the server
// (/preview/<id>, CSP sandbox) for opening in their own tab.
const SANDBOX_PAGE = "allow-scripts allow-forms allow-modals allow-popups allow-downloads";
const CONSOLE_SHIM = `<script>(function(){var P=function(t,a){try{parent.postMessage({__am2:1,t:t,a:Array.prototype.map.call(a,function(x){` +
  `try{return typeof x==="string"?x:x instanceof Error?x.stack||String(x):JSON.stringify(x)}catch(e){return String(x)}}).join(" ")},"*")}catch(e){}};` +
  `["log","info","warn","error","debug"].forEach(function(k){var o=console[k];console[k]=function(){P(k,arguments);return o.apply(console,arguments)}});` +
  `addEventListener("error",function(e){P("error",[(e.message||"error")+(e.lineno?" (line "+e.lineno+(e.colno?":"+e.colno:"")+")":"")])});` +
  `addEventListener("unhandledrejection",function(e){P("error",["Unhandled promise rejection: "+(e.reason&&e.reason.message||e.reason)])});})();</script>`;
function instrument(html) {
  const m = html.match(/<head[^>]*>/i) || html.match(/<html[^>]*>/i) || html.match(/<!doctype[^>]*>/i);
  return m ? html.slice(0, m.index + m[0].length) + CONSOLE_SHIM + html.slice(m.index + m[0].length) : CONSOLE_SHIM + html;
}
// signs that a model stopped before finishing the page
function cutOff(html) {
  const why = [];
  const open = (html.match(/<script\b/gi) || []).length, close = (html.match(/<\/script>/gi) || []).length;
  if (open > close) why.push("a <script> is never closed, so none of its code runs");
  if (/<html\b/i.test(html) && !/<\/html>/i.test(html)) why.push("there is no </html>");
  if ((html.match(/<svg\b/gi) || []).length > (html.match(/<\/svg>/gi) || []).length) why.push("an <svg> is never closed");
  return why;
}

let cv = null;                                       // {mode, code, result, html, title}
let tabNow = "page";
const MODE_TAB = {html: "page", python: "output", flow: "flow", web: "page"};
function setMode(mode) {
  const c = $("canvas");
  c.dataset.mode = mode;
  for (const el of c.querySelectorAll('[class*="m-"]')) el.hidden = !el.classList.contains(`m-${mode}`);
  $("tab-console").hidden = !(mode === "html" && issues.length);
  $("canvas-tabs").hidden = mode === "web";
  $("canvas").hidden = false;
}
function canvasTab(v) {
  tabNow = v;
  for (const b of $("canvas-tabs").querySelectorAll("[data-v]")) b.setAttribute("aria-selected", String(b.dataset.v === v));
  for (const p of document.querySelectorAll(".canvas__pane")) p.classList.toggle("on", p.dataset.pane === v);
  $("canvas-ask").hidden = !(v === "output" && cv && cv.result);
  if (v === "flow") flowShow();
  if (v === "sketch") { wbRender(); note("Sketch: drag to draw; Erase removes what you click; Attach to chat sends it as a picture. Kept in this browser."); }
  if (v === "source") note("Edit the Mermaid source; the Flow redraws.");
}
for (const b of $("canvas-tabs").querySelectorAll("[data-v]")) b.onclick = () => canvasTab(b.dataset.v);
const note = (t) => { $("canvas-note").textContent = t; };

// ---- the page frame (html and web)
function newFrame(sandbox) {
  const old = $("canvas-frame"), f = document.createElement("iframe");
  f.className = "canvas__frame"; f.id = "canvas-frame"; f.title = "Page";
  f.setAttribute("sandbox", sandbox); f.setAttribute("referrerpolicy", "no-referrer");
  old.replaceWith(f);
  return f;
}
async function previewUrl(html) {
  try {
    const r = await fetch("api/preview", {method: "POST", headers: headers(true), body: JSON.stringify({html})});
    if (r.ok) return (await r.json()).url;
  } catch (e) { /* an older server */ }
  return null;
}
function showHtml() {
  clearConsole();
  for (const w of cv.hints || []) issue("hint", w);
  newFrame(SANDBOX_PAGE).srcdoc = cv.html;
}
$("cb-reload").onclick = () => { if (!cv) return; if (cv.mode === "html") showHtml(); else if (cv.mode === "web") loadWeb(); };

// ---- the console: everything the page logs; the tab shows only when there are issues
let issues = [], conLines = [];
function clearConsole() { issues = []; conLines = []; $("canvas-console").innerHTML = ""; $("tab-console").hidden = true; }
function conLine(t, text) {
  conLines.push({t, text});
  const d = document.createElement("div"); d.className = "con-line"; d.dataset.t = t; d.textContent = text;
  $("canvas-console").appendChild(d);
  if (conLines.length > 500) { conLines.shift(); $("canvas-console").firstChild.remove(); }
}
function issue(t, text) {
  conLine(t, text);
  issues.push({t, text});
  if (cv && cv.mode === "html") {
    $("tab-console").hidden = false;
    $("console-count").textContent = issues.length;
    note(`The page has ${issues.length} issue${issues.length === 1 ? "" : "s"}: see Console.`);
  }
}
window.addEventListener("message", (e) => {
  const f = $("canvas-frame");
  if (e.data && e.data.__am2 && f && e.source === f.contentWindow) {
    if (e.data.t === "error" || e.data.t === "warn") issue(e.data.t, e.data.a); else conLine(e.data.t, e.data.a);
  }
  if (e.data && e.data.__am2dg && e.source === $("dg-frame").contentWindow) flowMessage(e.data);
});
$("console-send").onclick = () => {
  if (!issues.length) return;
  const list = issues.slice(0, 30).map((x) => `- ${x.t === "hint" ? "" : x.t + ": "}${x.text.slice(0, 500)}`).join("\n");
  $("input").value = `The page you wrote has ${issues.length === 1 ? "an issue" : "issues"} in the browser:\n${list}\n\n` +
    "Please fix ${it} and send the whole corrected file in one html block.".replace("${it}", issues.length === 1 ? "it" : "them");
  autosize(); $("input").focus();
  toast("info", "Issues are in the message box", "Send it to ask the model for a fix.");
};

// ---- opening something
async function openCanvas(kind, code) {
  if (kind === "mermaid") { openFlow(code); return; }
  cv = {mode: kind, code, result: null};
  $("canvas-code").textContent = code;
  if (kind === "python") {
    $("canvas-title").textContent = "Python";
    setMode("python"); canvasTab("output"); runCanvas(); return;
  }
  cv.title = (code.match(/<title>([^<]{1,80})<\/title>/i) || [, "Page preview"])[1];
  cv.hints = cutOff(code).map((w) => `This code looks cut off: ${w}. The model stopped before finishing the page.`);
  cv.html = instrument(code);
  $("canvas-title").textContent = cv.title;
  setMode("html");
  showHtml();
  canvasTab("page");
  if (!issues.length) note("Sandboxed: scripts run, but the page cannot reach am2Strata or this browser's data.");
  cv.url = await previewUrl(cv.html);                // for opening it in its own tab later
}

// ---- python
async function runCanvas() {
  if (!cv || cv.mode !== "python") return;
  const out = $("canvas-output");
  canvasTab("output");
  $("canvas-run").disabled = true;
  out.innerHTML = `<div class="run-status"><span class="st-badge st-badge--reading">Running</span><span class="muted">on this PC, in a sandbox</span></div>`;
  note("");
  try {
    const r = await fetch("api/run", {method: "POST", headers: headers(true), body: JSON.stringify({language: "python", code: cv.code})});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error((j.error && j.error.message) || `HTTP ${r.status}`);
    cv.result = j;
    renderRun(j);
  } catch (e) {
    out.innerHTML = `<div class="run-status"><span class="st-badge st-badge--error">Not run</span></div><pre class="run-stream run-stream--err"></pre>`;
    out.querySelector("pre").textContent = String(e.message || e);
  }
  $("canvas-run").disabled = false;
  canvasTab("output");
}
const b64text = (d) => new TextDecoder().decode(Uint8Array.from(atob(d), (c) => c.charCodeAt(0)));
function renderRun(j) {
  const out = $("canvas-output");
  const ok = j.exit_code === 0 && !j.timed_out;
  out.innerHTML = `<div class="run-status"><span class="st-badge ${ok ? "st-badge--generating" : "st-badge--error"}">${j.timed_out ? "Time limit" : ok ? "Done" : `Exit code ${j.exit_code}`}</span>` +
    `<span class="muted">${j.duration_s.toFixed(2)} s${j.files.length ? ` · ${j.files.length} file${j.files.length === 1 ? "" : "s"}` : ""}${j.truncated ? " · output cut at 200 KB" : ""}</span></div>`;
  const stream = (label, text, err) => {
    if (!text) return;
    const l = document.createElement("span"); l.className = "run-label"; l.textContent = label;
    const p = document.createElement("pre"); p.className = "run-stream" + (err ? " run-stream--err" : ""); p.textContent = text;
    out.append(l, p);
  };
  stream("Output", j.stdout, false);
  stream("Errors", j.stderr, true);
  if (!j.stdout && !j.stderr && !j.files.length) stream("Output", "(nothing printed)", false);
  for (const f of j.files) {
    const box = document.createElement("div"); box.className = "run-file";
    const row = document.createElement("div"); row.className = "run-file__row";
    const name = document.createElement("b"); name.className = "mono"; name.textContent = f.name;
    const size = document.createElement("span"); size.className = "muted"; size.textContent = f.size >= 1048576 ? `${fmt(f.size / 1048576, 1)} MB` : `${fmt(f.size / 1024, 1)} KB`;
    row.append(name, size);
    if (f.data) {
      const a = document.createElement("a"); a.className = "st-btn st-btn--secondary"; a.style.height = "30px"; a.textContent = "Download";
      a.download = f.name.split("/").pop(); a.href = `data:${f.mime};base64,${f.data}`; row.append(a);
    } else if (f.note) { const n = document.createElement("span"); n.className = "muted"; n.textContent = f.note; row.append(n); }
    box.append(row);
    if (f.data && f.mime.startsWith("image/")) { const im = document.createElement("img"); im.src = `data:${f.mime};base64,${f.data}`; im.alt = f.name; box.append(im); }
    if (f.data && f.mime === "text/html") {
      const b = document.createElement("button"); b.className = "st-btn st-btn--secondary"; b.style.height = "30px"; b.textContent = "Preview";
      b.onclick = () => openCanvas("html", b64text(f.data)); row.append(b);
    }
    if (f.data && /^(text\/(csv|plain)|application\/json)$/.test(f.mime) && f.size < 65536) {
      const p = document.createElement("pre"); p.className = "run-stream"; p.textContent = b64text(f.data); box.append(p);
    }
    out.append(box);
  }
  note("Ran in a sandbox: no network, home folders hidden, system read-only.");
}
$("canvas-run").onclick = runCanvas;
$("canvas-ask").onclick = () => {
  if (!cv || !cv.result) return;
  const j = cv.result, cut = (t) => (t.length > 6000 ? t.slice(0, 6000) + "\n[...]" : t);
  let t = `I ran the code. Exit code ${j.timed_out ? "(time limit)" : j.exit_code}.`;
  if (j.stdout) t += `\n\nOutput:\n\`\`\`\n${cut(j.stdout)}\n\`\`\``;
  if (j.stderr) t += `\n\nErrors:\n\`\`\`\n${cut(j.stderr)}\n\`\`\``;
  if (j.files.length) t += `\n\nFiles written: ${j.files.map((f) => f.name).join(", ")}`;
  $("input").value = t + "\n\n"; autosize(); $("input").focus();
};

// ---- another site (a link in an answer): the only mode with an address bar
const webNav = {list: [], i: -1};
function loadWeb() {
  const u = webNav.list[webNav.i];
  if (!u) return;
  $("cb-url").value = u;
  // another origin keeps its own origin (its cookies and storage work); this server's never does
  const foreign = new URL(u).origin !== location.origin;
  newFrame(SANDBOX_PAGE + (foreign ? " allow-same-origin" : "")).src = u;
  $("cb-back").disabled = webNav.i <= 0; $("cb-fwd").disabled = webNav.i >= webNav.list.length - 1;
  $("web-hint").hidden = false;
  note("Another site, inside the canvas. Ctrl-click a link to open it in a browser tab instead.");
}
function openWeb(url) {
  cv = {mode: "web", code: ""};
  webNav.list = webNav.list.slice(0, webNav.i + 1); webNav.list.push(url); webNav.i = webNav.list.length - 1;
  setMode("web"); canvasTab("page"); loadWeb();
}
$("cb-back").onclick = () => { if (webNav.i > 0) { webNav.i--; loadWeb(); } };
$("cb-fwd").onclick = () => { if (webNav.i < webNav.list.length - 1) { webNav.i++; loadWeb(); } };
$("cb-form").onsubmit = (ev) => {
  ev.preventDefault();
  let u = $("cb-url").value.trim();
  if (!u) return;
  if (!/^https?:\/\//i.test(u)) u = /\s/.test(u) || !/\./.test(u) ? `https://duckduckgo.com/html/?q=${encodeURIComponent(u)}`
                                                                 : (/^(localhost|127\.|\[::1\])/.test(u) ? "http://" : "https://") + u;
  openWeb(u);
};
const openTab = () => { const u = cv && cv.mode === "web" ? webNav.list[webNav.i] : cv && cv.url; if (u) window.open(u, "_blank", "noopener,noreferrer"); };
$("cb-open").onclick = openTab;
$("web-hint-open").onclick = openTab;
// links in answers open in the canvas; Ctrl/Cmd/Shift-click or the middle button keep the browser's own tab
$("chat").addEventListener("click", (e) => {
  const a = e.target.closest(".st-msg a[href^='http']");
  if (!a || e.ctrlKey || e.metaKey || e.shiftKey || e.button !== 0) return;
  e.preventDefault(); e.stopPropagation();
  openWeb(a.href);
}, true);

$("canvas-close").onclick = () => { $("canvas").hidden = true; newFrame(SANDBOX_PAGE); };
$("canvas-wide").onclick = () => { const c = $("canvas"); c.dataset.wide = String(c.dataset.wide !== "true"); $("canvas-wide").title = c.dataset.wide === "true" ? "Back beside the chat" : "Full width"; };
function save(blob, name) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ---- resizing: drag the left edge; the width is kept in this browser
(function () {
  const c = $("canvas"), grip = $("canvas-grip");
  const w = store.get("canvas-width", null);
  if (w) c.style.setProperty("--canvas-w", `${w}px`);
  grip.addEventListener("pointerdown", (e) => {
    e.preventDefault(); try { grip.setPointerCapture(e.pointerId); } catch (err) { /* not a live pointer */ }
    grip.classList.add("dragging"); document.body.classList.add("resizing");
    const right = c.getBoundingClientRect().right;
    const side = $("chats").offsetParent ? $("chats").getBoundingClientRect().width : 0;   // the chat keeps at least 360 px
    const max = Math.max(320, c.parentElement.getBoundingClientRect().width - side - 360);
    const move = (ev) => c.style.setProperty("--canvas-w", `${Math.round(Math.max(320, Math.min(right - ev.clientX, max)))}px`);
    const up = () => {
      grip.removeEventListener("pointermove", move); grip.removeEventListener("pointerup", up);
      grip.classList.remove("dragging"); document.body.classList.remove("resizing");
      store.set("canvas-width", parseInt(c.style.getPropertyValue("--canvas-w"), 10) || null);
    };
    grip.addEventListener("pointermove", move); grip.addEventListener("pointerup", up);
  });
  grip.ondblclick = () => { c.style.removeProperty("--canvas-w"); store.set("canvas-width", null); };
})();

// ---- flows: Mermaid, drawn in a sandboxed frame with the library inline (the frame loads nothing)
const DIAGRAMS = {
  flow: "flowchart TD\n  A([Request arrives]) --> B{Authorized?}\n  B -- no --> X[401 error]\n  B -- yes --> C{Model loaded?}\n  C -- no --> L[Load the engine] --> D\n  C -- yes --> D[Build the prompt]\n  D --> E[Generate tokens]\n  E --> F([Stream the answer])",
  data: "%% animate\nflowchart LR\n  U[Browser] -->|chat request| S[am2Strata]\n  S -->|token ids| E[(Engine)]\n  E -->|KV cache| G[GPU VRAM]\n  E -->|experts| R[System RAM]\n  E -->|tokens| S\n  S -->|SSE stream| U",
  seq: "sequenceDiagram\n  participant B as Browser\n  participant S as am2Strata\n  participant E as Engine\n  B->>S: POST /v1/chat/completions\n  S->>E: GEN prompt ids\n  loop each token\n    E-->>S: TOK id\n    S-->>B: data: delta\n  end\n  E-->>S: DONE timings\n  S-->>B: [DONE]",
  state: "stateDiagram-v2\n  [*] --> Idle\n  Idle --> Reading: request\n  Reading --> Generating: prompt read\n  Generating --> Idle: done\n  Generating --> Idle: stopped\n  Reading --> Error: too long\n  Error --> Idle",
  class: "classDiagram\n  class Manager {\n    +models()\n    +load(id)\n    +unload()\n  }\n  class Engine {\n    +generate(ids)\n    +stop()\n  }\n  class Server\n  Server --> Manager\n  Manager --> Engine : drives",
  machine: "%% animate\nflowchart LR\n  subgraph Input\n    A[Camera] --> B[Frame grabber]\n  end\n  subgraph Processing\n    B --> C[Resize] --> D[Vision encoder]\n    D --> E[Language model]\n  end\n  subgraph Output\n    E --> F[Answer]\n    E --> G[(Log)]\n  end",
};
const animated = (src) => /^\s*%%\s*animate/m.test(src);
let mermaidJs = null;
async function mermaidSource() {
  if (!mermaidJs) mermaidJs = (await (await fetch("web/vendor/mermaid.min.js")).text()).replace(/<\/script/gi, "<\\/script").replace(/<!--/g, "<\\!--");
  return mermaidJs;
}
function flowColors() {
  const dark = document.documentElement.dataset.theme === "dark";
  return {dark, bg: dark ? "#151B23" : "#FFFFFF", ink: dark ? "#E6EDF3" : "#1A2230"};
}
// labels as SVG text (no foreignObject), so the picture can be drawn onto a canvas for PNG, PDF and GIF
function flowDoc(lib) {
  const {dark, bg, ink} = flowColors();
  const css = `html,body{margin:0;height:100%;background:${bg};color:${ink};font-family:system-ui,sans-serif}` +
    `#v{width:100%;height:100%;overflow:hidden;cursor:grab;display:flex;align-items:center;justify-content:center}#v svg{max-width:none!important;flex:none}` +
    `#e{position:absolute;left:12px;right:12px;bottom:12px;padding:10px;border-radius:8px;background:#FDE3E1;color:#A8281E;font:12px ui-monospace,monospace;white-space:pre-wrap;display:none}` +
    `.anim .flowchart-link,.anim .edgePath path,.anim .messageLine0,.anim .messageLine1,.anim .transition{stroke-dasharray:7 5;animation:am2dash .9s linear infinite}@keyframes am2dash{to{stroke-dashoffset:-24}}` +
    `#z{position:absolute;right:10px;top:10px;font:11px system-ui;opacity:.55}`;
  const js = `mermaid.initialize({startOnLoad:false,theme:${JSON.stringify(dark ? "dark" : "default")},securityLevel:"strict",fontFamily:"system-ui,sans-serif",htmlLabels:false,flowchart:{htmlLabels:false},class:{htmlLabels:false}});` +
    `var v=document.getElementById("v"),er=document.getElementById("e"),s=1,x=0,y=0,n=0;` +
    `function t(){var g=v.querySelector("svg");if(g)g.style.transform="translate("+x+"px,"+y+"px) scale("+s+")"}` +
    `function fit(){var g=v.querySelector("svg");if(!g||!g.viewBox||!g.viewBox.baseVal)return;var b=g.viewBox.baseVal;if(!b.width)return;g.setAttribute("width",b.width);g.setAttribute("height",b.height);` +
    `s=Math.min((v.clientWidth-32)/b.width,(v.clientHeight-32)/b.height,1.5);x=0;y=0;t()}` +
    `v.addEventListener("wheel",function(e){e.preventDefault();s=Math.min(6,Math.max(.2,s*(e.deltaY<0?1.1:.9)));t()},{passive:false});` +
    `var d=null;v.onpointerdown=function(e){d=[e.clientX-x,e.clientY-y];v.setPointerCapture(e.pointerId)};v.onpointermove=function(e){if(d){x=e.clientX-d[0];y=e.clientY-d[1];t()}};v.onpointerup=function(){d=null};` +
    `v.ondblclick=fit;addEventListener("resize",fit);` +
    `async function r(src){try{var o=await mermaid.render("d"+(++n),src);v.innerHTML=o.svg;v.className=/^\\s*%%\\s*animate/m.test(src)?"anim":"";er.style.display="none";fit();` +
    `parent.postMessage({__am2dg:1,ok:true},"*")}catch(e){er.textContent=String(e.message||e);er.style.display="block";parent.postMessage({__am2dg:1,ok:false,error:String(e.message||e)},"*")}}` +
    `addEventListener("message",function(e){if(!e.data)return;if(typeof e.data.src==="string")r(e.data.src);` +
    `if(e.data.exportSvg){var g=v.querySelector("svg");if(!g){parent.postMessage({__am2dg:1,svg:null},"*");return}var c=g.cloneNode(true);c.removeAttribute("style");` +
    `c.setAttribute("xmlns","http://www.w3.org/2000/svg");var b=g.viewBox.baseVal;parent.postMessage({__am2dg:1,svg:c.outerHTML,w:b.width,h:b.height},"*")}});` +
    `parent.postMessage({__am2dg:1,ready:true},"*");`;
  return `<!doctype html><html><head><meta charset="utf-8"><title>Flow</title><style>${css}</style><script>${lib}<\/script></head>` +
    `<body><div id="v"></div><div id="e"></div><div id="z">scroll: zoom · drag: move · double-click: fit</div><script>${js}<\/script></body></html>`;
}
let dgReady = false, dgTheme = null, dgTimer = null, svgWait = null;
async function flowFrame() {
  const theme = document.documentElement.dataset.theme;
  if (dgTheme === theme) return;
  dgTheme = theme; dgReady = false;
  try { $("dg-frame").srcdoc = flowDoc(await mermaidSource()); }
  catch (e) { dgTheme = null; note("The diagram library did not load: " + (e.message || e)); }
}
function flowRender() { if (dgReady) $("dg-frame").contentWindow.postMessage({src: $("dg-src").value}, "*"); }
function flowMessage(d) {
  if (d.ready) { dgReady = true; flowRender(); return; }
  if ("svg" in d) { if (svgWait) { svgWait(d); svgWait = null; } return; }
  if (tabNow === "flow") note(d.ok ? `Scroll to zoom, drag to move.${animated($("dg-src").value) ? " The arrows are animated: export a GIF to keep that." : ""}`
                                   : "The diagram has an error (shown in red); fix it under Source.");
}
function flowShow() { flowFrame(); flowRender(); $("export-gif").hidden = !animated($("dg-src").value); }
function openFlow(src) {
  cv = {mode: "flow", code: src};
  $("dg-src").value = src; store.set("board-diagram", src);
  const first = (src.match(/^\s*(?:%%[^\n]*\n\s*)*(\w+)/) || [, "Flow"])[1];
  $("canvas-title").textContent = {flowchart: "Flow", graph: "Flow", sequenceDiagram: "Sequence", stateDiagram: "States", "stateDiagram-v2": "States",
                                   classDiagram: "Structure", erDiagram: "Data model", gantt: "Timeline", mindmap: "Mind map"}[first] || "Flow";
  setMode("flow"); canvasTab("flow");
}
$("dg-src").addEventListener("input", () => { if (cv && cv.mode === "flow") cv.code = $("dg-src").value; store.set("board-diagram", $("dg-src").value); clearTimeout(dgTimer); dgTimer = setTimeout(flowRender, 350); });
$("dg-template").onchange = () => { const t = DIAGRAMS[$("dg-template").value]; if (t) { $("dg-src").value = t; store.set("board-diagram", t); flowRender(); } $("dg-template").value = ""; };
$("dg-ask").onclick = () => { $("input").value = `Explain this diagram step by step:\n\n\`\`\`mermaid\n${$("dg-src").value}\n\`\`\`\n\n`; autosize(); $("input").focus(); };
$("dg-src").value = store.get("board-diagram", DIAGRAMS.flow);

// ---- flow exports: Markdown, PNG, PDF (static), GIF (animated arrows)
function flowSvg() {
  return new Promise((ok, fail) => {
    if (!dgReady) { fail(new Error("the flow is not drawn yet")); return; }
    svgWait = (d) => (d.svg ? ok(d) : fail(new Error("nothing is drawn")));
    $("dg-frame").contentWindow.postMessage({exportSvg: true}, "*");
    setTimeout(() => { if (svgWait) { svgWait = null; fail(new Error("the flow did not answer")); } }, 4000);
  });
}
async function svgToCanvas(svg, w, h, extraCss = "", scale = 2) {
  const {bg} = flowColors(), pad = 24;
  const inner = svg.replace(/<svg\b([^>]*)>/, (m0, a) => `<svg${a.replace(/\s(width|height)="[^"]*"/g, "")} width="${w}" height="${h}" x="${pad}" y="${pad}">` +
                                                       (extraCss ? `<style>${extraCss}</style>` : ""));
  const W = w + 2 * pad, H = h + 2 * pad;
  const doc = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}"><rect width="100%" height="100%" fill="${bg}"/>${inner}</svg>`;
  const img = new Image();
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(doc);
  await img.decode();
  const c = document.createElement("canvas"); c.width = Math.round(W * scale); c.height = Math.round(H * scale);
  const g = c.getContext("2d"); g.scale(scale, scale); g.drawImage(img, 0, 0);
  return c;
}
function pdfFromJpeg(jpeg, pxW, pxH, scale) {
  const enc = new TextEncoder(), parts = [], offsets = [];
  let len = 0;
  const put = (x) => { const b = typeof x === "string" ? enc.encode(x) : x; parts.push(b); len += b.length; };
  const w = (pxW / scale) * 0.75, h = (pxH / scale) * 0.75;           // CSS px -> points
  const content = `q ${w.toFixed(2)} 0 0 ${h.toFixed(2)} 0 0 cm /Im0 Do Q`;
  put("%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");
  const obj = (n, body) => { offsets[n] = len; put(`${n} 0 obj\n`); for (const b of [].concat(body)) put(b); put("\nendobj\n"); };
  obj(1, "<< /Type /Catalog /Pages 2 0 R >>");
  obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
  obj(3, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${w.toFixed(2)} ${h.toFixed(2)}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>`);
  obj(4, [`<< /Type /XObject /Subtype /Image /Width ${pxW} /Height ${pxH} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpeg.length} >>\nstream\n`, jpeg, "\nendstream"]);
  obj(5, `<< /Length ${content.length} >>\nstream\n${content}\nendstream`);
  const xref = len;
  put(`xref\n0 6\n0000000000 65535 f \n${[1, 2, 3, 4, 5].map((n) => String(offsets[n]).padStart(10, "0") + " 00000 n \n").join("")}`);
  put(`trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
  return new Blob(parts, {type: "application/pdf"});
}
// a small GIF89a encoder: a fixed 256-colour palette (a 6×6×6 cube and 40 greys suits flat diagrams), LZW, looping
function gifEncode(frames, w, h, delayCs) {
  const pal = new Uint8Array(768);
  let k = 0;
  for (let r = 0; r < 6; r++) for (let g = 0; g < 6; g++) for (let b = 0; b < 6; b++) { pal[k * 3] = r * 51; pal[k * 3 + 1] = g * 51; pal[k * 3 + 2] = b * 51; k++; }
  for (let i = 0; i < 40; i++) { const v = Math.round(i * 255 / 39); pal[k * 3] = pal[k * 3 + 1] = pal[k * 3 + 2] = v; k++; }
  const index = (r, g, b) => {
    const cr = Math.round(r / 51), cg = Math.round(g / 51), cb = Math.round(b / 51);
    const dc = (r - cr * 51) ** 2 + (g - cg * 51) ** 2 + (b - cb * 51) ** 2;
    const gi = Math.round(((r + g + b) / 3) * 39 / 255), gv = Math.round(gi * 255 / 39);
    return (r - gv) ** 2 + (g - gv) ** 2 + (b - gv) ** 2 < dc ? 216 + gi : cr * 36 + cg * 6 + cb;
  };
  const out = [];
  const b8 = (v) => out.push(v & 255), b16 = (v) => { b8(v); b8(v >> 8); }, str = (s) => { for (const ch of s) b8(ch.charCodeAt(0)); };
  str("GIF89a"); b16(w); b16(h); b8(0xF7); b8(0); b8(0);
  for (const v of pal) b8(v);
  b8(0x21); b8(0xFF); b8(11); str("NETSCAPE2.0"); b8(3); b8(1); b16(0); b8(0);
  for (const rgba of frames) {
    const px = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < px.length; i++, j += 4) px[i] = index(rgba[j], rgba[j + 1], rgba[j + 2]);
    b8(0x21); b8(0xF9); b8(4); b8(0); b16(delayCs); b8(0); b8(0);
    b8(0x2C); b16(0); b16(0); b16(w); b16(h); b8(0);
    const data = lzw(px, 8);
    b8(8);
    for (let i = 0; i < data.length; i += 255) { const n = Math.min(255, data.length - i); b8(n); for (let q = 0; q < n; q++) out.push(data[i + q]); }
    b8(0);
  }
  b8(0x3B);
  return new Blob([new Uint8Array(out)], {type: "image/gif"});
}
function lzw(px, min) {
  const clear = 1 << min, eoi = clear + 1, bytes = [];
  let size = min + 1, maxcode = (1 << size) - 1, next = eoi + 1, cur = 0, nbits = 0, reset = false;
  const dict = new Map();
  const emit = (code) => {
    cur |= code << nbits; nbits += size;
    while (nbits >= 8) { bytes.push(cur & 255); cur >>>= 8; nbits -= 8; }
    if (reset) { size = min + 1; maxcode = (1 << size) - 1; reset = false; }
    else if (next > maxcode) { size++; maxcode = size === 12 ? 4096 : (1 << size) - 1; }
  };
  emit(clear);
  let prefix = px[0];
  for (let i = 1; i < px.length; i++) {
    const c = px[i], key = prefix * 256 + c, hit = dict.get(key);
    if (hit !== undefined) { prefix = hit; continue; }
    emit(prefix);
    if (next < 4096) dict.set(key, next++);
    else { dict.clear(); next = eoi + 1; reset = true; emit(clear); }
    prefix = c;
  }
  emit(prefix); emit(eoi);
  if (nbits > 0) bytes.push(cur & 255);
  return bytes;
}
const flowName = () => ($("canvas-title").textContent || "flow").toLowerCase().replace(/[^a-z0-9]+/g, "-");
async function exportFlow(kind) {
  const src = $("dg-src").value;
  if (kind === "md") {
    const title = chats.find((c) => c.id === current);
    save(new Blob([`# ${$("canvas-title").textContent}${title && title.title ? ` — ${title.title}` : ""}\n\n\`\`\`mermaid\n${src}\n\`\`\`\n`], {type: "text/markdown"}), `${flowName()}.md`);
    return;
  }
  try {
    const d = await flowSvg();
    if (kind === "png") { const c = await svgToCanvas(d.svg, d.w, d.h); c.toBlob((b) => save(b, `${flowName()}.png`), "image/png"); }
    if (kind === "pdf") {
      const c = await svgToCanvas(d.svg, d.w, d.h);
      const jpeg = Uint8Array.from(atob(c.toDataURL("image/jpeg", 0.92).split(",")[1]), (ch) => ch.charCodeAt(0));
      save(pdfFromJpeg(jpeg, c.width, c.height, 2), `${flowName()}.pdf`);
    }
    if (kind === "gif") {
      note("Making the GIF…");
      const N = 12, frames = [];
      let W = 0, H = 0;
      const sel = ".flowchart-link,.edgePath path,.messageLine0,.messageLine1,.transition";
      for (let f = 0; f < N; f++) {   // the arrows' dashes moved one step per frame: 12 px a loop, as on screen
        const c = await svgToCanvas(d.svg, d.w, d.h, `${sel}{stroke-dasharray:7 5!important;stroke-dashoffset:${-f}px!important;animation:none!important}`, 1);
        W = c.width; H = c.height;
        frames.push(c.getContext("2d").getImageData(0, 0, W, H).data);
      }
      save(gifEncode(frames, W, H, 7), `${flowName()}.gif`);
      note("GIF saved: the animated arrows, looping.");
    }
  } catch (e) { toast("error", "Not exported", String(e.message || e), 6000); }
}
for (const b of $("flow-export").querySelectorAll("[data-export]")) b.onclick = () => exportFlow(b.dataset.export);

// ---- flows from answers: the model is asked to draw one when a question is about how something flows or works,
// and a finished answer that has one opens it here
const FLOW_HINT = "When the user asks how something works or flows, or asks to explain a process, architecture, algorithm, pipeline, " +
  "protocol or sequence of steps, include one diagram in a ```mermaid code block (flowchart, sequenceDiagram or stateDiagram-v2) with " +
  "short labels; the user's app draws it next to your answer. When it shows data or steps moving, make its first line `%% animate`. " +
  "Explain in text as usual; for other questions, no diagram.";
function withHints(msgs) {
  return settings.flows === false ? msgs : [{role: "system", content: FLOW_HINT}, ...msgs];
}
function afterAnswer(m) {
  const blocks = [...(m.text || "").matchAll(/```mermaid[^\n]*\n([\s\S]*?)```/g)];
  if (!blocks.length) return;
  if (!$("canvas").hidden && cv && cv.mode !== "flow") return;     // do not replace a page the user is looking at
  openFlow(blocks[blocks.length - 1][1].trim());
}

// ---- the sketch: shapes in an SVG, kept in this browser
let wb = store.get("board-shapes", []), wbUndo = [], wbTool = "rect", wbColor = "#3DD68C";
const WB_COLORS = ["#3DD68C", "#6CB6FF", "#F2B84B", "#FF7B72", "#C297FF", "currentColor"];
$("wb-colors").innerHTML = WB_COLORS.map((c) => `<button type="button" role="radio" data-v="${c}" style="background:${c === "currentColor" ? "var(--st-ink)" : c}" aria-label="${c === "currentColor" ? "Ink" : c}"></button>`).join("");
function wbSel() {
  for (const b of $("wb-tools").children) b.setAttribute("aria-checked", String(b.dataset.v === wbTool));
  for (const b of $("wb-colors").children) b.setAttribute("aria-checked", String(b.dataset.v === wbColor));
  $("wb").classList.toggle("erasing", wbTool === "erase");
}
for (const b of $("wb-tools").children) b.onclick = () => { wbTool = b.dataset.v; wbSel(); };
for (const b of $("wb-colors").children) b.onclick = () => { wbColor = b.dataset.v; wbSel(); };
wbSel();
const SVGNS = "http://www.w3.org/2000/svg";
function arrowHead(x1, y1, x2, y2) {
  const a = Math.atan2(y2 - y1, x2 - x1), l = 12;
  return `${x2},${y2} ${x2 - l * Math.cos(a - 0.45)},${y2 - l * Math.sin(a - 0.45)} ${x2 - l * Math.cos(a + 0.45)},${y2 - l * Math.sin(a + 0.45)}`;
}
function shapeSvg(sh, i) {
  const c = sh.c, k = `data-k="${i}"`, st = `stroke="${c}" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"`;
  const fill = c === "currentColor" ? "transparent" : c + "22";
  if (sh.k === "pen") return `<polyline ${k} ${st} points="${sh.p.map((q) => q.join(",")).join(" ")}"/>`;
  if (sh.k === "rect") return `<rect ${k} x="${sh.x}" y="${sh.y}" width="${sh.w}" height="${sh.h}" rx="10" stroke="${c}" stroke-width="2.2" fill="${fill}"/>`;
  if (sh.k === "ellipse") return `<ellipse ${k} cx="${sh.x + sh.w / 2}" cy="${sh.y + sh.h / 2}" rx="${sh.w / 2}" ry="${sh.h / 2}" stroke="${c}" stroke-width="2.2" fill="${fill}"/>`;
  if (sh.k === "diamond") { const cx = sh.x + sh.w / 2, cy = sh.y + sh.h / 2;
    return `<polygon ${k} points="${cx},${sh.y} ${sh.x + sh.w},${cy} ${cx},${sh.y + sh.h} ${sh.x},${cy}" stroke="${c}" stroke-width="2.2" fill="${fill}" stroke-linejoin="round"/>`; }
  if (sh.k === "arrow") return `<g ${k}><line ${st} x1="${sh.x1}" y1="${sh.y1}" x2="${sh.x2}" y2="${sh.y2}"/><polygon points="${arrowHead(sh.x1, sh.y1, sh.x2, sh.y2)}" fill="${c}"/></g>`;
  if (sh.k === "text") return `<text ${k} x="${sh.x}" y="${sh.y}" fill="${c}" style="font:500 16px Outfit,system-ui,sans-serif">${esc(sh.t)}</text>`;
  return "";
}
function wbRender(extra) { $("wb").innerHTML = wb.map(shapeSvg).join("") + (extra ? shapeSvg(extra, -1) : ""); }
function wbSave() { store.set("board-shapes", wb); }
function wbPush(sh) { wbUndo.push(wb.slice()); wb.push(sh); wbSave(); wbRender(); }
function wbPoint(e) { const r = $("wb").getBoundingClientRect(); return [Math.round(e.clientX - r.left), Math.round(e.clientY - r.top)]; }
$("wb").addEventListener("pointerdown", (e) => {
  const [x, y] = wbPoint(e);
  if (wbTool === "erase") {
    const t = e.target.closest("[data-k]");
    if (t && +t.dataset.k >= 0) { wbUndo.push(wb.slice()); wb.splice(+t.dataset.k, 1); wbSave(); wbRender(); }
    return;
  }
  if (wbTool === "text") {
    const inp = document.createElement("input"); inp.className = "wb-input"; inp.style.left = `${x}px`; inp.style.top = `${y - 14}px`;
    $("board-draw").appendChild(inp); setTimeout(() => inp.focus(), 0);
    const done = () => { const t = inp.value.trim(); inp.remove(); if (t) wbPush({k: "text", x, y: y + 4, t, c: wbColor}); };
    inp.onkeydown = (ev) => { if (ev.key === "Enter") done(); if (ev.key === "Escape") { inp.value = ""; done(); } };
    inp.onblur = done;
    return;
  }
  try { $("wb").setPointerCapture(e.pointerId); } catch (err) { /* not a live pointer */ }
  const cur = wbTool === "pen" ? {k: "pen", p: [[x, y]], c: wbColor} : wbTool === "arrow" ? {k: "arrow", x1: x, y1: y, x2: x, y2: y, c: wbColor}
            : {k: wbTool, x, y, w: 0, h: 0, c: wbColor, ox: x, oy: y};
  const move = (ev) => {
    const [mx, my] = wbPoint(ev);
    if (cur.k === "pen") cur.p.push([mx, my]);
    else if (cur.k === "arrow") { cur.x2 = mx; cur.y2 = my; }
    else { cur.x = Math.min(cur.ox, mx); cur.y = Math.min(cur.oy, my); cur.w = Math.abs(mx - cur.ox); cur.h = Math.abs(my - cur.oy); }
    wbRender(cur);
  };
  const up = () => {
    $("wb").removeEventListener("pointermove", move); $("wb").removeEventListener("pointerup", up);
    delete cur.ox; delete cur.oy;
    const tiny = cur.k === "pen" ? cur.p.length < 2 : cur.k === "arrow" ? Math.hypot(cur.x2 - cur.x1, cur.y2 - cur.y1) < 6 : cur.w < 6 || cur.h < 6;
    if (!tiny) wbPush(cur); else wbRender();
  };
  $("wb").addEventListener("pointermove", move); $("wb").addEventListener("pointerup", up);
});
function wbUndoOne() { if (wbUndo.length) { wb = wbUndo.pop(); wbSave(); wbRender(); } }
$("wb-undo").onclick = wbUndoOne;
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "z" && !$("canvas").hidden && tabNow === "sketch" && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); wbUndoOne(); }
});
$("wb-clear").onclick = () => { if (!wb.length) return; wbUndo.push(wb.slice()); wb = []; wbSave(); wbRender(); toast("info", "Sketch cleared", "", 5000, {label: "Undo", run: wbUndoOne}); };
async function sketchPng() {
  const r = $("wb").getBoundingClientRect();
  if (!wb.length) { toast("info", "The sketch is empty"); return null; }
  const css = getComputedStyle(document.documentElement);
  const body = wb.map(shapeSvg).join("").replaceAll("currentColor", css.getPropertyValue("--st-ink").trim());
  const xml = `<svg xmlns="${SVGNS}" width="${r.width}" height="${r.height}"><rect width="100%" height="100%" fill="${css.getPropertyValue("--st-surface").trim()}"/>${body}</svg>`;
  const img = new Image(); img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(xml);
  await img.decode();
  const c = document.createElement("canvas"); c.width = r.width * 2; c.height = r.height * 2;
  const g = c.getContext("2d"); g.scale(2, 2); g.drawImage(img, 0, 0);
  return c.toDataURL("image/png");
}
$("wb-send").onclick = async () => {
  const png = await sketchPng(); if (!png) return;
  attachments.push({kind: "image", name: "sketch.png", url: png}); renderAttachments();
  toast("success", "Sketch attached", "It goes with your next message (a model that reads pictures can see it).");
  $("input").focus();
};

// ------------------------------------------------------------------ start
setBusy(false);
renderChat();
renderChatList();
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
loadModels();
loadHealth().then(loadMcp).then(() => { if (startQuestion) { $("input").value = startQuestion; send(); } });
showTab(location.hash.slice(1) || "chat");
poll();

// ------------------------------------------------------------------ images (POST /v1/images/generations)
$("img-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const prompt = $("img-prompt").value.trim();
  if (!prompt) return;
  const body = {prompt, size: $("img-size").value, response_format: "b64_json", model: $("img-model").value};
  if ($("img-seed").value !== "") body.seed = Number($("img-seed").value);
  $("img-go").disabled = true;
  const t0 = performance.now();
  $("img-hint").textContent = "Generating\u2026 (a minute or more on a small GPU)";
  try {
    const r = await fetch("v1/images/generations", {method: "POST", headers: headers(true), body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    for (const d of j.data) {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + d.b64_json;
      img.alt = prompt;
      const cap = document.createElement("figcaption");
      cap.textContent = `${j.size}, seed ${d.seed}, ${((performance.now() - t0) / 1000).toFixed(0)} s: ${prompt}`;
      fig.append(img, cap);
      $("img-results").prepend(fig);
    }
    $("img-hint").textContent = "Done. Right-click an image to save it.";
  } catch (e) {
    $("img-hint").textContent = "";
    toast("error", "No image", String(e.message || e), 8000);
  } finally {
    $("img-go").disabled = false;
  }
};

// ------------------------------------------------------------------ Settings > Memory (GET/POST/DELETE /api/memory)
async function loadMemory() {
  $("mem-web").setAttribute("aria-checked", String(store.get("memory-web", true)));
  let j;
  try {
    const r = await fetch("api/memory", {headers: headers()});
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    j = await r.json();
  } catch (e) { $("mem-list").innerHTML = `<p class="muted">Not loaded: ${esc(String(e.message || e))}</p>`; return; }
  for (const [id, k] of [["mem-tools", "tools"], ["mem-inject", "inject"], ["mem-extend", "extend_context"]]) $(id).setAttribute("aria-checked", String(!!j.settings[k]));
  const facts = (j.facts || []).slice().sort((a, b) => b.created - a.created);
  $("mem-sum").textContent = `${facts.length} fact(s) · ${j.conversations_archived} chat(s) with an archive`;
  const who = {all: "everyone", web: "this web app", api: "API clients"};
  $("mem-list").innerHTML = facts.length ? facts.map((f) => `<div class="model-row"><div class="model-row__name"><b>${esc(f.text)}</b>` +
    `<span>${esc(who[f.scope] || f.scope)} · written by ${esc(f.source)} · ${new Date(f.created * 1000).toLocaleString()}</span></div>` +
    `<button class="st-btn st-btn--secondary" data-forget="${esc(f.id)}">Forget</button></div>`).join("")
    : `<p class="muted" style="padding:12px 14px">No facts yet. Tell the model something worth keeping, or add one above.</p>`;
}
$("mem-web").onclick = () => {
  const on = $("mem-web").getAttribute("aria-checked") !== "true";
  store.set("memory-web", on);
  $("mem-web").setAttribute("aria-checked", String(on));
};
for (const [id, k] of [["mem-tools", "tools"], ["mem-inject", "inject"], ["mem-extend", "extend_context"]]) {
  $(id).onclick = async () => {
    const on = $(id).getAttribute("aria-checked") !== "true";
    try {
      const r = await fetch("api/memory/settings", {method: "POST", headers: headers(true), body: JSON.stringify({[k]: on})});
      if (!r.ok) throw new Error(((await r.json()).error || {}).message || `HTTP ${r.status}`);
    } catch (e) { toast("error", "Not changed", String(e.message || e), 6000); }
    loadMemory();
  };
}
$("mem-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const text = $("mem-text").value.trim();
  if (!text) return;
  try {
    const r = await fetch("api/memory", {method: "POST", headers: headers(true), body: JSON.stringify({text, scope: $("mem-scope").value})});
    if (!r.ok) throw new Error(((await r.json()).error || {}).message || `HTTP ${r.status}`);
    $("mem-text").value = "";
  } catch (e) { toast("error", "Not added", String(e.message || e), 6000); }
  loadMemory();
};
$("mem-list").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-forget]"); if (!b) return;
  try {
    const r = await fetch(`api/memory/${encodeURIComponent(b.dataset.forget)}`, {method: "DELETE", headers: headers(true)});
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
  } catch (err) { toast("error", "Not forgotten", String(err.message || err), 6000); }
  loadMemory();
});

// ------------------------------------------------------------------ speech (POST /v1/audio/transcriptions, /v1/audio/speech)
// The microphone is recorded as raw samples (Web Audio) and sent as a 16 kHz WAV: what the server's decoder reads
// (browsers record Opus otherwise). getUserMedia needs a secure page: https, or http://localhost.
const recorder = {stream: null, ctx: null, node: null, chunks: [], rate: 48000, on: false};
async function recStart() {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    throw new Error("the microphone needs this page on https or on http://localhost (the browser allows it nowhere else)");
  }
  recorder.stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
  recorder.ctx = new AudioContext();
  recorder.rate = recorder.ctx.sampleRate;
  const src = recorder.ctx.createMediaStreamSource(recorder.stream);
  recorder.node = recorder.ctx.createScriptProcessor(4096, 1, 1);
  recorder.chunks = [];
  recorder.node.onaudioprocess = (e) => recorder.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(recorder.node);
  recorder.node.connect(recorder.ctx.destination);
  recorder.on = true;
}
async function recStop() {
  recorder.on = false;
  recorder.node && recorder.node.disconnect();
  recorder.stream && recorder.stream.getTracks().forEach((t) => t.stop());
  await (recorder.ctx && recorder.ctx.close());
  const n = recorder.chunks.reduce((a, c) => a + c.length, 0);
  if (!n) throw new Error("nothing was recorded");
  const all = new Float32Array(n);
  let at = 0;
  for (const c of recorder.chunks) { all.set(c, at); at += c.length; }
  // to 16 kHz with the browser's own resampler
  const off = new OfflineAudioContext(1, Math.ceil(n * 16000 / recorder.rate), 16000);
  const buf = off.createBuffer(1, n, recorder.rate);
  buf.copyToChannel(all, 0);
  const s = off.createBufferSource();
  s.buffer = buf; s.connect(off.destination); s.start();
  const out = (await off.startRendering()).getChannelData(0);
  return wavBlob(out, 16000);
}
function wavBlob(x, rate) {
  const b = new DataView(new ArrayBuffer(44 + x.length * 2));
  const str = (o, t) => { for (let i = 0; i < t.length; i++) b.setUint8(o + i, t.charCodeAt(i)); };
  str(0, "RIFF"); b.setUint32(4, 36 + x.length * 2, true); str(8, "WAVEfmt "); b.setUint32(16, 16, true);
  b.setUint16(20, 1, true); b.setUint16(22, 1, true); b.setUint32(24, rate, true); b.setUint32(28, rate * 2, true);
  b.setUint16(32, 2, true); b.setUint16(34, 16, true); str(36, "data"); b.setUint32(40, x.length * 2, true);
  for (let i = 0; i < x.length; i++) b.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 32767, true);
  return new Blob([b], {type: "audio/wav"});
}
async function transcribe(blob, {model, language, translate} = {}) {
  const f = new FormData();
  f.append("file", blob, blob.name || "speech.wav");
  f.append("response_format", "verbose_json");
  if (model) f.append("model", model);
  if (language) f.append("language", language);
  const r = await fetch(translate ? "v1/audio/translations" : "v1/audio/transcriptions", {method: "POST", headers: headers(), body: f});
  const j = await r.json();
  if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
  return j;
}
async function speakText(text, {model, voice} = {}) {
  const r = await fetch("v1/audio/speech", {method: "POST", headers: headers(true), body: JSON.stringify({input: text, model, voice})});
  if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error((j.error && j.error.message) || r.statusText); }
  return {blob: await r.blob(), seconds: +(r.headers.get("x-strata-seconds") || 0)};
}
let sttBlob = null;
$("stt-rec").onclick = async () => {
  const b = $("stt-rec");
  try {
    if (!recorder.on) {
      await recStart();
      b.querySelector("span").textContent = "Stop"; b.classList.add("rec-on");
      $("stt-what").textContent = "recording…";
    } else {
      sttBlob = await recStop();
      b.querySelector("span").textContent = "Record"; b.classList.remove("rec-on");
      $("stt-what").textContent = `recorded ${(sttBlob.size / 32000).toFixed(1)} s`;
      $("stt-go").disabled = false;
    }
  } catch (e) { b.querySelector("span").textContent = "Record"; b.classList.remove("rec-on"); toast("error", "No recording", String(e.message || e), 8000); }
};
$("stt-file").onchange = () => {
  const f = $("stt-file").files[0];
  if (!f) return;
  sttBlob = f;
  $("stt-what").textContent = `${f.name} (${bytes(f.size)})`;
  $("stt-go").disabled = false;
};
$("stt-form").onsubmit = async (ev) => {
  ev.preventDefault();
  if (!sttBlob) return;
  $("stt-go").disabled = true;
  const t0 = performance.now();
  $("stt-hint").textContent = "Transcribing…";
  try {
    const j = await transcribe(sttBlob, {model: $("stt-model").value, language: $("stt-lang").value, translate: $("stt-task").value === "translate"});
    $("stt-out").hidden = false;
    $("stt-text").value = j.text || "";
    $("stt-meta").textContent = `${j.duration} s of audio${j.language ? `, ${j.language}` : ""} · ${((performance.now() - t0) / 1000).toFixed(1)} s on the ${(j.strata || {}).device || "?"}`;
    $("stt-hint").textContent = "";
  } catch (e) { $("stt-hint").textContent = ""; toast("error", "No transcript", String(e.message || e), 8000); }
  $("stt-go").disabled = false;
};
$("stt-copy").onclick = () => copyText($("stt-text").value, $("stt-copy"));
$("stt-chat").onclick = () => { $("input").value = $("stt-text").value; showTab("chat"); autosize(); $("input").focus(); };
$("tts-voice").onchange = () => { $("tts-desc").hidden = $("tts-voice").value !== "custom"; };
$("tts-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const text = $("tts-text").value.trim();
  if (!text) return;
  const voice = $("tts-voice").value === "custom" ? $("tts-desc").value.trim() || "laura" : $("tts-voice").value;
  $("tts-go").disabled = true;
  const t0 = performance.now();
  $("tts-hint").textContent = "Speaking… (a sentence takes some seconds on the CPU)";
  try {
    const {blob, seconds} = await speakText(text, {model: $("tts-model").value, voice});
    const url = URL.createObjectURL(blob);
    const fig = document.createElement("figure");
    const a = document.createElement("audio");
    a.controls = true; a.src = url; a.autoplay = true;
    const cap = document.createElement("figcaption");
    cap.innerHTML = `${fmt(seconds, 1)} s of speech in ${((performance.now() - t0) / 1000).toFixed(0)} s: ${esc(text.slice(0, 120))} · <a href="${url}" download="am2strata-speech.wav">save</a>`;
    fig.append(a, cap);
    $("img-results").prepend(fig);
    $("img-results").hidden = false;
    $("tts-hint").textContent = "";
  } catch (e) { $("tts-hint").textContent = ""; toast("error", "No speech", String(e.message || e), 8000); }
  $("tts-go").disabled = false;
};
$("audio-device").onchange = async () => {
  try {
    const r = await fetch("api/media/audio/device", {method: "POST", headers: headers(true), body: JSON.stringify({device: $("audio-device").value})});
    if (!r.ok) throw new Error(((await r.json()).error || {}).message || `HTTP ${r.status}`);
    toast("success", "Saved", `Speech runs on: ${$("audio-device").selectedOptions[0].textContent}.`);
  } catch (e) { toast("error", "Not changed", String(e.message || e), 6000); }
};
// the chat: dictate a message; read an answer aloud
$("mic-btn").onclick = async () => {
  const b = $("mic-btn");
  try {
    if (!recorder.on) {
      await recStart();
      b.classList.add("rec-on"); b.title = "Stop and transcribe";
    } else {
      b.classList.remove("rec-on"); b.title = "Speak your message (speech to text)";
      const blob = await recStop();
      b.disabled = true;
      $("composer-hint").textContent = "Transcribing…";
      const j = await transcribe(blob, {model: media.stt_model});
      const cur = $("input").value;
      $("input").value = cur ? `${cur.trimEnd()} ${j.text}` : j.text;
      autosize(); $("input").focus();
      $("composer-hint").textContent = "";
    }
  } catch (e) { b.classList.remove("rec-on"); $("composer-hint").textContent = ""; toast("error", "No dictation", String(e.message || e), 8000); }
  b.disabled = false;
};
let speaking = null;     // {audio, button}
function speakable(md) {
  return md.replace(/```[\s\S]*?```/g, " (code) ").replace(/`([^`]*)`/g, "$1").replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[*_#>|~]+/g, " ").replace(/\s+/g, " ").trim();
}
async function readAloud(i, btn) {
  if (speaking) { speaking.audio.pause(); speaking.button.classList.remove("rec-on"); const was = speaking.button; speaking = null; if (was === btn) return; }
  const m = messages[i];
  const text = speakable(m.text || "").slice(0, 4000);
  if (!text) return;
  btn.classList.add("rec-on"); btn.disabled = true;
  try {
    const {blob} = await speakText(text, {model: media.tts_model});
    const audio = new Audio(URL.createObjectURL(blob));
    speaking = {audio, button: btn};
    audio.onended = () => { btn.classList.remove("rec-on"); speaking = null; };
    audio.play();
  } catch (e) { btn.classList.remove("rec-on"); toast("error", "Not read aloud", String(e.message || e), 8000); }
  btn.disabled = false;
}
