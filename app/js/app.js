/* Hieroglyph Dictionary — app shell.
 * Screen 1: draw → auto-match → 5 canonical candidates.
 * Screen 2: tap a candidate → sign details (from glyphs.json) → back.
 * All inference is local (onnxruntime-web, wasm). */
import { rgbaToGray, preprocess } from "./preprocess.mjs";
import { GlyphMatcher } from "./matcher.mjs";

const $ = (id) => document.getElementById(id);
const ort = globalThis.ort;

let cfg, glyphs, matcher;
let strokes = [];
let current = null;
let matchTimer = null;
let matchBusy = false;
let matchQueued = false;

/* ------------------------------------------------ boot */
async function fetchProgress(url, onPart) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +r.headers.get("Content-Length") || 0;
  if (!r.body || !total) { const b = await r.arrayBuffer(); onPart(1); return b; }
  const reader = r.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onPart(got / total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

async function boot() {
  const status = $("status"), text = $("status-text"), fill = $("progress-fill");
  status.classList.remove("error");
  $("btn-retry").hidden = true;
  try {
    if (typeof ort === "undefined") {
      throw new Error("This browser does not support the on-device recognizer " +
                      "(WebAssembly required).");
    }
    // Must be an absolute URL: onnxruntime-web dynamically imports the wasm
    // glue module from *within* ort.min.js's own scope, so a relative prefix
    // like "./vendor/ort/" gets resolved a second time against that script's
    // own directory (.../vendor/ort/vendor/ort/...). An absolute URL sidesteps
    // that, and (unlike a root-absolute "/vendor/ort/") still works when the
    // app is hosted under a subpath (e.g. GitHub Pages project sites).
    ort.env.wasm.wasmPaths = new URL("./vendor/ort/", document.baseURI).href;
    ort.env.wasm.numThreads = 1;

    text.textContent = "Loading sign data…";
    cfg = await (await fetch("./data/config.json")).json();
    const [meta, glyphsJson] = await Promise.all([
      (await fetch("./data/index_meta.json")).json(),
      (await fetch("./data/glyphs.json")).json(),
    ]);
    glyphs = glyphsJson;

    text.textContent = "Loading recognizer model…";
    const parts = { model: 0, index: 0 };
    const upd = () => { fill.style.width =
      `${Math.round(100 * (0.9 * parts.model + 0.1 * parts.index))}%`; };
    const [modelBuffer, indexBuffer] = await Promise.all([
      fetchProgress(`./data/${cfg.model}`, (p) => { parts.model = p; upd(); }),
      fetchProgress("./data/index.bin", (p) => { parts.index = p; upd(); }),
    ]);

    text.textContent = "Starting…";
    matcher = await GlyphMatcher.create({ ort, modelBuffer, indexBuffer, meta });
    await runMatchTensor(new Float32Array(cfg.size * cfg.size).fill(-1));  // warm up
    status.hidden = true;
    if (location.search.includes("selftest")) await selftest();
  } catch (e) {
    console.error(e);
    status.classList.add("error");
    text.textContent = e.message || "Failed to load. Check your connection.";
    $("btn-retry").hidden = false;
  }
}
$("btn-retry").onclick = boot;

/* ------------------------------------------------ canvas */
const pad = $("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });
const PEN = 9;                              // on the 512px backing store
ctx.lineCap = ctx.lineJoin = "round";

function redraw() {
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, pad.width, pad.height);
  ctx.strokeStyle = "#1c1a17";
  ctx.lineWidth = PEN;
  for (const s of strokes) {
    ctx.beginPath();
    s.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (s.length === 1) ctx.lineTo(s[0][0] + 0.1, s[0][1]);
    ctx.stroke();
  }
  $("pad-hint").style.opacity = strokes.length ? 0 : 1;
}
function pos(e) {
  const r = pad.getBoundingClientRect();
  return [(e.clientX - r.left) * pad.width / r.width,
          (e.clientY - r.top) * pad.height / r.height];
}
pad.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  current = [pos(e)];
  strokes.push(current);
  pad.setPointerCapture(e.pointerId);
  redraw();
});
pad.addEventListener("pointermove", (e) => {
  if (current) { current.push(pos(e)); redraw(); }
});
const endStroke = () => {
  if (!current) return;
  current = null;
  scheduleMatch();
};
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);

$("btn-clear").onclick = () => {
  strokes = []; redraw();
  $("results").hidden = true;
};
$("btn-undo").onclick = () => {
  strokes.pop(); redraw();
  strokes.length ? scheduleMatch(120) : ($("results").hidden = true);
};
redraw();

/* ------------------------------------------------ matching */
function scheduleMatch(delay = 350) {
  clearTimeout(matchTimer);
  matchTimer = setTimeout(runMatch, delay);
}

/* The encoder sees stroke width RELATIVE to the drawing's extent (the crop is
 * letterboxed to cfg.size), so a fixed on-screen pen still varies 3x with how
 * large the user draws. For matching, re-render the vector strokes at ~2% of
 * the glyph extent (~3 px after 160 px letterboxing — the best operating point
 * in review P3). The visible pen stays cosmetic. */
const matchCanvas = document.createElement("canvas");
matchCanvas.width = matchCanvas.height = 512;
const mctx = matchCanvas.getContext("2d", { willReadFrequently: true });
mctx.lineCap = mctx.lineJoin = "round";

function renderNormalized() {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const s of strokes)
    for (const [x, y] of s) {
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  const extent = Math.max(x1 - x0, y1 - y0);
  const lw = Math.min(24, Math.max(3, 0.021 * extent));
  mctx.fillStyle = "#fff";
  mctx.fillRect(0, 0, matchCanvas.width, matchCanvas.height);
  mctx.strokeStyle = "#000";
  mctx.lineWidth = lw;
  for (const s of strokes) {
    mctx.beginPath();
    s.forEach(([x, y], i) => (i ? mctx.lineTo(x, y) : mctx.moveTo(x, y)));
    if (s.length === 1) mctx.lineTo(s[0][0] + 0.1, s[0][1]);
    mctx.stroke();
  }
  return mctx.getImageData(0, 0, matchCanvas.width, matchCanvas.height);
}

async function runMatchTensor(t) {
  return matcher.match(t, cfg.size, cfg.top_k || 5);
}

async function runMatch() {
  if (!matcher || !strokes.length) return;
  if (matchBusy) { matchQueued = true; return; }
  matchBusy = true;
  try {
    const im = renderNormalized();
    const gray = rgbaToGray(im.data, im.width, im.height);
    const hits = await runMatchTensor(preprocess(gray, cfg));
    renderResults(hits);
  } catch (e) {
    console.error("match failed:", e);
  } finally {
    matchBusy = false;
    if (matchQueued) { matchQueued = false; scheduleMatch(50); }
  }
}

function renderResults(hits) {
  const cards = $("cards");
  cards.innerHTML = "";
  const low = !hits.length || hits[0].score < (cfg.score_threshold ?? 0.6) ||
              (hits[0].margin ?? 1) < (cfg.margin_threshold ?? 0.03);
  $("lowconf").hidden = !low;
  hits.forEach((h, i) => {
    const g = glyphs[h.label] || {};
    const el = document.createElement("button");
    el.className = "card" + (i === 0 ? " best" : "");
    el.innerHTML =
      `<img src="./glyphs/${encodeURIComponent(h.label)}.png" alt="" loading="lazy"
            onerror="this.style.visibility='hidden'">
       <span class="code">${h.label}</span>
       <span class="sim">${h.score.toFixed(2)}</span>` +
      (i === 0 ? `<span class="tag">best match</span>` : "");
    el.title = g.desc || h.label;
    el.onclick = () => showDetail(h.label);
    cards.appendChild(el);
  });
  $("results").hidden = false;
}

/* ------------------------------------------------ detail screen */
function showDetail(label) {
  const g = glyphs[label] || {};
  $("detail-code").textContent = label;
  $("detail-img").src = `./glyphs/${encodeURIComponent(label)}.png`;
  $("detail-name").textContent = g.desc || "(no description)";
  $("detail-details").textContent = g.details || "";
  $("detail-gardiner").textContent = label + (g.priority ? " · common sign" : "");
  const cp = g.char ? g.char.codePointAt(0) : null;
  $("detail-unicode").textContent = cp
    ? `${g.char}  U+${cp.toString(16).toUpperCase()}` : "—";
  $("screen-draw").hidden = true;
  $("screen-detail").hidden = false;
  scrollTo(0, 0);
}
$("btn-back").onclick = () => {           // drawing + results are preserved
  $("screen-detail").hidden = true;
  $("screen-draw").hidden = false;
};
$("btn-next").onclick = () => {           // fresh canvas for the next sign
  clearTimeout(matchTimer);
  matchQueued = false;
  strokes = [];
  current = null;
  redraw();
  $("results").hidden = true;
  $("screen-detail").hidden = true;
  $("screen-draw").hidden = false;
  scrollTo(0, 0);
};

/* ------------------------------------------------ about */
$("about-link").onclick = (e) => { e.preventDefault(); $("about").showModal(); };
$("about-close").onclick = () => $("about").close();

/* ------------------------------------------------ selftest (?selftest) */
async function selftest() {
  const fixtures = await (await fetch("./data/selftest.json")).json();
  const out = [];
  for (const f of fixtures) {
    const img = new Image();
    img.src = "data:image/png;base64," + f.png_b64;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = img.width; c.height = img.height;
    const cc = c.getContext("2d");
    cc.fillStyle = "#fff"; cc.fillRect(0, 0, c.width, c.height);
    cc.drawImage(img, 0, 0);
    const im = cc.getImageData(0, 0, c.width, c.height);
    const hits = await runMatchTensor(
      preprocess(rgbaToGray(im.data, im.width, im.height), cfg));
    const exp = f.expected_top5.map((x) => x[0]);
    out.push({ name: f.name, pass: hits[0].label === exp[0],
               got: hits.map((h) => `${h.label}:${h.score.toFixed(3)}`),
               expected: f.expected_top5.map((x) => `${x[0]}:${x[1]}`) });
  }
  console.table(out);
  alert("selftest: " + (out.every((o) => o.pass) ? "PASS" : "MISMATCH — see console")
        + ` (${out.filter((o) => o.pass).length}/${out.length})`);
}

/* ------------------------------------------------ go */
if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js");
boot();
