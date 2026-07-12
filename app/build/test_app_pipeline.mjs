/*
 * End-to-end verification of the app's inference path OUTSIDE the browser:
 * decodes the selftest PNG fixtures, runs the REAL app code
 * (js/preprocess.mjs + js/matcher.mjs) with onnxruntime-web, and compares the
 * top-5 against the torch fp32 reference recorded by export_app_assets.py.
 *
 *   cd app/build && npm install && node test_app_pipeline.mjs
 *
 * Pass criteria: top-1 label identical on every fixture (the int8/fp32 and
 * OpenCV-vs-JS-resampling differences must not change the answer).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import ortPkg from "onnxruntime-web";
const ort = ortPkg.default ?? ortPkg;
import { PNG } from "pngjs";
import { rgbaToGray, preprocess } from "../js/preprocess.mjs";
import { GlyphMatcher } from "../js/matcher.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, "..", "data");

const cfg = JSON.parse(readFileSync(join(DATA, "config.json")));
const meta = JSON.parse(readFileSync(join(DATA, "index_meta.json")));
const fixtures = JSON.parse(readFileSync(join(DATA, "selftest.json")));

// optional override: node test_app_pipeline.mjs --model model.onnx
const mi = process.argv.indexOf("--model");
if (mi > -1) cfg.model = process.argv[mi + 1];
console.log(`testing model: ${cfg.model}`);

ort.env.wasm.numThreads = 1;

const matcher = await GlyphMatcher.create({
  ort,
  modelBuffer: new Uint8Array(readFileSync(join(DATA, cfg.model))),
  indexBuffer: new Uint8Array(readFileSync(join(DATA, "index.bin"))),
  meta,
});

let failures = 0;
for (const f of fixtures) {
  const png = PNG.sync.read(Buffer.from(f.png_b64, "base64"));
  const gray = rgbaToGray(png.data, png.width, png.height);
  const t = preprocess(gray, cfg);
  const hits = await matcher.match(t, cfg.size, 5);
  const exp = f.expected_top5;
  const ok = hits[0].label === exp[0][0];
  const overlap = hits.filter((h) => exp.some((e) => e[0] === h.label)).length;
  const dScore = Math.abs(hits[0].score - exp[0][1]).toFixed(3);
  console.log(`${ok ? "PASS" : "FAIL"} ${f.name} (truth ${f.truth})`);
  console.log(`  js  : ${hits.map((h) => `${h.label}:${h.score.toFixed(3)}`).join("  ")}`);
  console.log(`  ref : ${exp.map((e) => `${e[0]}:${e[1]}`).join("  ")}`);
  console.log(`  top5 overlap ${overlap}/5, top1 score delta ${dScore}`);
  if (!ok) failures++;
}
console.log(failures ? `\n${failures} fixture(s) FAILED` : "\nall fixtures passed");
process.exit(failures ? 1 : 0);
