/* Cache-first service worker: the app shell is precached; the model, index and
 * glyph thumbnails are cached on first use, so the app is fully offline after
 * one visit. Bump VERSION on any deploy to invalidate. */
const VERSION = "v1";
const CORE = [
  ".", "index.html", "css/style.css", "js/app.js", "js/preprocess.mjs",
  "js/matcher.mjs", "manifest.webmanifest", "icons/icon-192.png",
  "icons/icon-512.png", "vendor/ort/ort.min.js",
  "data/config.json", "data/index_meta.json", "data/glyphs.json",
  "data/index.bin",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(CORE))
    .then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then((hit) =>
      hit ||
      fetch(e.request).then((res) => {
        if (res.ok && new URL(e.request.url).origin === location.origin) {
          const copy = res.clone();
          caches.open(VERSION).then((c) => c.put(e.request, copy));
        }
        return res;
      })
    )
  );
});
