/*
 * On-device glyph matcher: ONNX encoder (image → L2-normalized embedding) +
 * cosine nearest-prototype over the exported index, with the same per-label
 * max aggregation as pipelines/matching/match.py::Matcher.
 *
 * Environment-agnostic: the caller injects the onnxruntime module (`ort`) and
 * the fetched buffers, so this runs identically in the browser and in node.
 */
export class GlyphMatcher {
  constructor({ ort, session, index, meta }) {
    this.ort = ort;
    this.session = session;
    this.index = index;                    // Float32Array, count*dim, L2-normed rows
    this.dim = meta.dim;
    this.count = meta.count;
    this.labels = meta.labels;
    this.kinds = meta.kinds;
  }

  static async create({ ort, modelBuffer, indexBuffer, meta, sessionOptions = {} }) {
    const session = await ort.InferenceSession.create(
      modelBuffer instanceof Uint8Array ? modelBuffer : new Uint8Array(modelBuffer),
      { executionProviders: ["wasm"], ...sessionOptions });
    const index = new Float32Array(
      indexBuffer instanceof ArrayBuffer ? indexBuffer
        : indexBuffer.buffer.slice(indexBuffer.byteOffset,
                                   indexBuffer.byteOffset + indexBuffer.byteLength));
    if (index.length !== meta.count * meta.dim)
      throw new Error(`index size mismatch: ${index.length} != ${meta.count}x${meta.dim}`);
    return new GlyphMatcher({ ort, session, index, meta });
  }

  /** tensor: Float32Array of size*size (one query). Returns top-k
   *  [{label, score, kind, margin}] sorted by score desc. */
  async match(tensor, size, top = 5) {
    const input = new this.ort.Tensor("float32", tensor, [1, 1, size, size]);
    const out = await this.session.run({ image: input });
    const q = out.embedding.data;          // (dim,), already L2-normalized

    const best = new Map();                // label -> {score, kind}
    const { index, dim } = this;
    for (let p = 0; p < this.count; p++) {
      let s = 0;
      const off = p * dim;
      for (let d = 0; d < dim; d++) s += index[off + d] * q[d];
      const lab = this.labels[p];
      const cur = best.get(lab);
      if (cur === undefined || s > cur.score)
        best.set(lab, { score: s, kind: this.kinds[p] });
    }
    const ranked = [...best.entries()]
      .sort((a, b) => b[1].score - a[1].score)
      .slice(0, top)
      .map(([label, v]) => ({ label, score: v.score, kind: v.kind }));
    if (ranked.length > 1) ranked[0].margin = ranked[0].score - ranked[1].score;
    return ranked;
  }
}
