/*
 * Query preprocessing — a faithful port of hieromatch/data.py
 * (load_gray → crop_ink → letterbox → to_tensor). Pure JS on plain arrays so
 * the exact same code runs in the browser and in the node build test.
 *
 * A "gray image" here is {data: Uint8Array|Uint8ClampedArray, width, height}.
 */

/** RGBA (canvas ImageData / decoded PNG) → grayscale, alpha composited over white.
 *  Same luminance weights as PIL Image.convert("L"). */
export function rgbaToGray(rgba, width, height) {
  const out = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    const a = rgba[p + 3] / 255;
    const r = rgba[p] * a + 255 * (1 - a);
    const g = rgba[p + 1] * a + 255 * (1 - a);
    const b = rgba[p + 2] * a + 255 * (1 - a);
    out[i] = (r * 299 + g * 587 + b * 114) / 1000;
  }
  return { data: out, width, height };
}

/** Bounding box of ink (pixels < thresh), padded; null if the image is blank.
 *  Mirrors data.crop_ink(g, thresh=200, pad=6). */
export function cropInk(img, thresh = 200, pad = 6) {
  const { data, width, height } = img;
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      if (data[row + x] < thresh) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null;
  y0 = Math.max(0, y0 - pad); y1 = Math.min(height, y1 + 1 + pad);
  x0 = Math.max(0, x0 - pad); x1 = Math.min(width, x1 + 1 + pad);
  const w = x1 - x0, h = y1 - y0;
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    out.set(data.subarray((y0 + y) * width + x0, (y0 + y) * width + x1), y * w);
  return { data: out, width: w, height: h };
}

/** Separable resize: area-average when shrinking (cv2.INTER_AREA), linear when
 *  enlarging (what INTER_AREA degenerates to). */
export function resize(img, dstW, dstH) {
  const pass = (data, sw, sh, dw) => {        // resample rows: (sw x sh) -> (dw x sh)
    const out = new Float32Array(dw * sh);
    const scale = sw / dw;
    for (let x = 0; x < dw; x++) {
      if (scale >= 1) {
        const lo = x * scale, hi = lo + scale;
        const iLo = Math.floor(lo), iHi = Math.min(sw, Math.ceil(hi));
        for (let y = 0; y < sh; y++) {
          let acc = 0, wsum = 0;
          for (let i = iLo; i < iHi; i++) {
            const w = Math.min(hi, i + 1) - Math.max(lo, i);
            acc += data[y * sw + i] * w; wsum += w;
          }
          out[y * dw + x] = acc / wsum;
        }
      } else {
        const cx = (x + 0.5) * scale - 0.5;
        const i0 = Math.max(0, Math.floor(cx)), i1 = Math.min(sw - 1, i0 + 1);
        const t = Math.min(1, Math.max(0, cx - i0));
        for (let y = 0; y < sh; y++)
          out[y * dw + x] = data[y * sw + i0] * (1 - t) + data[y * sw + i1] * t;
      }
    }
    return out;
  };
  const transpose = (data, w, h) => {
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) out[x * h + y] = data[y * w + x];
    return out;
  };
  let d = pass(Float32Array.from(img.data), img.width, img.height, dstW); // rows
  d = transpose(d, dstW, img.height);
  d = pass(d, img.height, dstW, dstH);                                    // cols
  d = transpose(d, dstH, dstW);
  const out = new Uint8Array(dstW * dstH);
  for (let i = 0; i < out.length; i++) out[i] = Math.max(0, Math.min(255, Math.round(d[i])));
  return { data: out, width: dstW, height: dstH };
}

/** Fit the ink crop into a size×size white square, centered.
 *  Mirrors data.letterbox(g, size) with margin 0.08 and no jitter. */
export function letterbox(img, size, { thresh = 200, pad = 6, margin = 0.08 } = {}) {
  const c = cropInk(img, thresh, pad);
  const canvas = new Uint8Array(size * size).fill(255);
  if (c === null) return { data: canvas, width: size, height: size };
  const s = (size * (1 - margin)) / Math.max(c.width, c.height);
  const nw = Math.max(1, Math.round(c.width * s));
  const nh = Math.max(1, Math.round(c.height * s));
  const r = resize(c, nw, nh);
  const y0 = (size - nh) >> 1, x0 = (size - nw) >> 1;
  for (let y = 0; y < nh; y++)
    canvas.set(r.data.subarray(y * nw, (y + 1) * nw), (y0 + y) * size + x0);
  return { data: canvas, width: size, height: size };
}

/** gray → model input: ink → +1, background → -1 (data.to_tensor). */
export function toTensor(img) {
  const out = new Float32Array(img.data.length);
  for (let i = 0; i < out.length; i++) out[i] = (255 - img.data[i]) / 127.5 - 1;
  return out;
}

/** Full pipeline: gray image → Float32Array [1,1,size,size] input. */
export function preprocess(gray, cfg) {
  const lb = letterbox(gray, cfg.size, cfg);
  return toTensor(lb);
}
