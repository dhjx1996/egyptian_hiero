/*
 * Local drawing collection — future training data.
 *
 * The moment the user taps a candidate sign, we snapshot the drawing together
 * with the label they picked and the recognizer's ranking. Later actions are
 * folded back in as quality-control flags (see app.js): pressing "back" marks
 * the pick as doubted; "Next sign" marks it confirmed; picking a different
 * candidate for the same drawing links the two so a wrong→right correction is
 * recoverable.
 *
 * Everything stays in the browser (IndexedDB) — nothing is uploaded. Export
 * and clear are driven from the About dialog. Intended for a small trusted
 * group: the JSON export is transferred manually and each user clears their
 * own store to reclaim space.
 */

const DB = "glyph-collect";
const STORE = "samples";
const VER = 2;                             // v2 adds the `ts` index for trimming
const MAX = 5000;                          // hard cap; oldest trimmed on save

let _db = null;
function db() {
  if (_db) return Promise.resolve(_db);
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, VER);
    r.onupgradeneeded = () => {
      const d = r.result;
      const s = d.objectStoreNames.contains(STORE)
        ? r.transaction.objectStore(STORE)          // upgrading v1 → v2
        : d.createObjectStore(STORE, { keyPath: "id" });
      if (!s.indexNames.contains("ts")) s.createIndex("ts", "ts");
    };
    r.onsuccess = () => res((_db = r.result));
    r.onerror = () => rej(r.error);
  });
}

function req(r) {
  return new Promise((res, rej) => {
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

async function withStore(mode, fn) {
  const d = await db();
  const s = d.transaction(STORE, mode).objectStore(STORE);
  return fn(s);
}

/** Persist one snapshot, then trim to the newest MAX records (oldest by `ts`
 *  dropped). Failures are swallowed (collection must never break the lookup
 *  flow) but logged. */
export function saveSample(rec) {
  return db()
    .then(
      (d) =>
        new Promise((res, rej) => {
          const t = d.transaction(STORE, "readwrite");
          const s = t.objectStore(STORE);
          s.put(rec);
          const cnt = s.count();
          cnt.onsuccess = () => {
            let over = cnt.result - MAX;
            if (over <= 0) return;
            const cur = s.index("ts").openCursor();   // ascending = oldest first
            cur.onsuccess = () => {
              const c = cur.result;
              if (c && over > 0) { c.delete(); over--; c.continue(); }
            };
          };
          t.oncomplete = () => res();
          t.onerror = () => rej(t.error);
          t.onabort = () => rej(t.error);
        })
    )
    .catch((e) => console.warn("collect: save failed", e));
}

/** Merge a partial update into an existing sample, get+put in one transaction
 *  so it can't race a concurrent auto-commit. */
export function patchSample(id, patch) {
  return db()
    .then(
      (d) =>
        new Promise((res, rej) => {
          const t = d.transaction(STORE, "readwrite");
          const s = t.objectStore(STORE);
          const g = s.get(id);
          g.onsuccess = () => {
            const rec = g.result;
            if (rec) s.put(Object.assign(rec, patch));
          };
          t.oncomplete = () => res();
          t.onerror = () => rej(t.error);
          t.onabort = () => rej(t.error);
        })
    )
    .catch((e) => console.warn("collect: patch failed", e));
}

export async function countSamples() {
  try {
    return await withStore("readonly", (s) => req(s.count()));
  } catch {
    return 0;
  }
}

async function allSamples() {
  try {
    return await withStore("readonly", (s) => req(s.getAll()));
  } catch {
    return [];
  }
}

/** All stored samples, newest first — for the history list. */
export async function listSamples() {
  return (await allSamples()).sort((a, b) => b.ts - a.ts);
}

export async function clearSamples() {
  try {
    await withStore("readwrite", (s) => req(s.clear()));
  } catch (e) {
    console.warn("collect: clear failed", e);
  }
}

/** Download every stored drawing as one JSON file (embedded PNG data URLs +
 *  vector strokes + labels + flags). Returns the row count. */
export async function exportSamples() {
  const rows = await allSamples();
  const blob = new Blob([JSON.stringify(rows)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `glyph-drawings-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
  return rows.length;
}
