// ce-tabnet — WebGPU pipeline-stage ENGINE (ES module; usable from join.html AND a Web Worker).
// =============================================================================================
// This is the compute core for ONE pipeline stage (a contiguous range of Llama-family transformer
// layers). It takes a layer-range shard's weights, builds reusable WebGPU compute pipelines for the
// transformer block (RMSNorm, quantized matmul w/ Q4/Q8 dequant, RoPE, GQA attention with a LOCAL
// KV cache, SwiGLU MLP, residual), runs the forward pass on incoming activations, and emits either
// the next stage's activation (`hidden`) or, on the last stage, a sampled `token_id`.
//
// It also ships a REAL CPU/WASM-fallback path (plain JS typed-array kernels, identical math) that
// runs when `navigator.gpu` is absent — so an old phone can still serve a small stage.
//
// -----------------------------------------------------------------------------------------------
// PUBLIC ENGINE API  (matches inference.js / module-contract.md §D Impl 5, same handshake):
//
//   await initStage({ arch, dims, layers:[lo,hi], is_first, is_last, weights, opts? }) -> StageCtx
//   await forward(ctx, { seq_id, token_pos, hidden, token_ids, temperature?, top_p? }) ->
//        { hidden: Float16Array }   when !is_last   (send to next stage)
//        { token_id: number }       when  is_last   (sampled token, send back)
//   freeStage(ctx)
//
// Plus a Web-Worker harness (bottom of file): if loaded as a Worker, it speaks a tiny JSON/binary
// postMessage protocol {op:"init"|"forward"|"free"} so the tab can keep the UI thread free. The same
// module is import-able directly (synchronous, same-thread) by join.html — both entry points share
// the SAME engine code below.
//
// -----------------------------------------------------------------------------------------------
// SHARED `weights` OBJECT (verbatim with shard-loader.js, Implementer 6):
//   {
//     arch: "llama",
//     dims: { hidden_dim, n_heads, n_kv_heads, vocab, rope_theta },
//     embed?:   ArrayBuffer,                                   // present iff is_first
//     lm_head?: ArrayBuffer, norm?: ArrayBuffer,               // present iff is_last (final RMSNorm + head)
//     layers: [{ idx, buf:ArrayBuffer, quant:"q4_0"|"q8_0"|"f16"|"f32" }]  // exactly [lo,hi), in order
//   }
//
// Each per-layer `buf` is a packed concatenation of that layer's tensors, in this fixed order
// (this is the tabnet model-pack layout; see models/README.md). Offsets are computed from `dims`:
//   attn_norm   [hidden]                (RMSNorm weight, always f32/f16, NOT quantized)
//   wq          [hidden, n_heads*head_dim]      (quantized)
//   wk          [hidden, n_kv*head_dim]         (quantized)
//   wv          [hidden, n_kv*head_dim]         (quantized)
//   wo          [n_heads*head_dim, hidden]      (quantized)
//   ffn_norm    [hidden]                (RMSNorm weight)
//   w_gate      [hidden, ffn_dim]               (quantized)
//   w_up        [hidden, ffn_dim]               (quantized)
//   w_down      [ffn_dim, hidden]               (quantized)
// where head_dim = hidden/n_heads and ffn_dim = round to the pack's value (default ~ 8/3*hidden,
// rounded; carried in the manifest as dims.ffn_dim when present, else derived).
//
// ACTIVATION framing: the `hidden` Float16Array this engine produces/consumes is exactly what
// protocol.encodeActivation/decodeActivation move on the wire (Implementer 1). We never frame here;
// tabnet-node.js does the wire framing. We only deal in Float16Array (fp16) hidden states.
//
// =============================================================================================

// Float16Array shim: not all JS engines expose it. We carry fp16 as the bit pattern in a Uint16Array
// on the wire (tabnet-node frames it); f16ArrToF32 reads it. Alias the constructor so
// `new Float16Array(buffer)` works everywhere by treating it as Uint16Array of fp16 bits. Installed at
// the TOP so every function below can safely reference Float16Array.
if (typeof globalThis.Float16Array === "undefined") {
  globalThis.Float16Array = class Float16ArrayPolyfill extends Uint16Array {};
}

// ---------------------------------------------------------------------------------------------
// fp16 <-> fp32 helpers (the activation wire dtype is fp16; WebGPU storage compute is fp32 here for
// portability — `shader-f16` is an optional feature we use when present, else we dequant to f32).
// ---------------------------------------------------------------------------------------------
function f16ToF32(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
function f32ToF16(val) {
  if (Number.isNaN(val)) return 0x7e00;
  const sign = val < 0 || Object.is(val, -0) ? 0x8000 : 0;
  val = Math.abs(val);
  if (val === Infinity) return sign | 0x7c00;
  if (val === 0) return sign;
  let e = Math.floor(Math.log2(val));
  e = Math.max(-25, Math.min(16, e));
  let m;
  if (e < -14) { m = Math.round(val / Math.pow(2, -24)); return sign | (m & 0x3ff); } // subnormal
  m = Math.round((val / Math.pow(2, e) - 1) * 1024);
  if (m === 1024) { e += 1; m = 0; }
  if (e > 15) return sign | 0x7c00;
  return sign | ((e + 15) << 10) | (m & 0x3ff);
}
function f16ArrToF32(u16) { const o = new Float32Array(u16.length); for (let i = 0; i < u16.length; i++) o[i] = f16ToF32(u16[i]); return o; }
function f32ArrToF16(f32) { const o = new Uint16Array(f32.length); for (let i = 0; i < f32.length; i++) o[i] = f32ToF16(f32[i]); return o; }

// ---------------------------------------------------------------------------------------------
// Quantized-weight dequant on CPU (used by the WASM/CPU path AND to upload f32 weights to WebGPU
// buffers when the device lacks an int4 path). q4_0 / q8_0 use the llama.cpp block layout:
//   q4_0 block = 32 weights: [f16 scale][16 bytes of 4-bit nibbles]; w = (nib - 8) * scale
//   q8_0 block = 32 weights: [f16 scale][32 int8];                   w = q * scale
// f16 / f32 weights are stored raw.
// We dequant lazily, per matmul, to keep memory down on the CPU path; on the GPU path we dequant
// once at init into f32 storage buffers (a clear, correct baseline — see "Throughput" note below).
// ---------------------------------------------------------------------------------------------
const QK = 32;
function dequantToF32(buf, off, count, quant) {
  const out = new Float32Array(count);
  if (quant === "f32") { out.set(new Float32Array(buf, off, count)); return out; }
  if (quant === "f16") { const u = new Uint16Array(buf, off, count); for (let i = 0; i < count; i++) out[i] = f16ToF32(u[i]); return out; }
  const dv = new DataView(buf);
  const nblocks = Math.ceil(count / QK);
  let o = off, w = 0;
  if (quant === "q8_0") {
    for (let b = 0; b < nblocks; b++) {
      const scale = f16ToF32(dv.getUint16(o, true)); o += 2;
      for (let j = 0; j < QK && w < count; j++, w++) out[w] = dv.getInt8(o + j) * scale;
      o += QK;
    }
  } else if (quant === "q4_0") {
    for (let b = 0; b < nblocks; b++) {
      const scale = f16ToF32(dv.getUint16(o, true)); o += 2;
      for (let j = 0; j < QK / 2 && w < count; j++) {
        const byte = dv.getUint8(o + j);
        out[w++] = ((byte & 0x0f) - 8) * scale;
        if (w < count) out[w++] = ((byte >> 4) - 8) * scale;
      }
      o += QK / 2;
    }
  } else {
    throw new Error(`unsupported quant ${quant}`);
  }
  return out;
}
function quantBytesFor(count, quant) {
  if (quant === "f32") return count * 4;
  if (quant === "f16") return count * 2;
  const nblocks = Math.ceil(count / QK);
  if (quant === "q8_0") return nblocks * (2 + QK);
  if (quant === "q4_0") return nblocks * (2 + QK / 2);
  throw new Error(`unsupported quant ${quant}`);
}

// ---------------------------------------------------------------------------------------------
// Layout: slice a packed per-layer buffer into named tensor views (offsets in BYTES, dequant lazily).
// ---------------------------------------------------------------------------------------------
function layerLayout(dims) {
  const H = dims.hidden_dim, nH = dims.n_heads, nKV = dims.n_kv_heads;
  const headDim = Math.floor(H / nH);
  const qDim = nH * headDim, kvDim = nKV * headDim;
  const ffn = dims.ffn_dim || roundFfn(H);
  return { H, nH, nKV, headDim, qDim, kvDim, ffn,
    tensors: [
      ["attn_norm", H,          "norm"],
      ["wq",        H * qDim,    "q"],
      ["wk",        H * kvDim,   "q"],
      ["wv",        H * kvDim,   "q"],
      ["wo",        qDim * H,    "q"],
      ["ffn_norm",  H,          "norm"],
      ["w_gate",    H * ffn,     "q"],
      ["w_up",      H * ffn,     "q"],
      ["w_down",    ffn * H,     "q"],
    ] };
}
function roundFfn(H) { // Llama SwiGLU: 8/3*H rounded up to a multiple of 256
  const x = Math.floor((8 * H) / 3);
  return Math.ceil(x / 256) * 256;
}

// Build CPU-side dequantized weight views for one layer (used by both paths; GPU path uploads these).
function dequantLayer(buf, quant, dims) {
  const L = layerLayout(dims);
  const w = {};
  let off = 0;
  for (const [name, count, kind] of L.tensors) {
    // RMSNorm weights are never block-quantized in the pack; they're stored f16. Matmul weights use `quant`.
    const effective = kind === "norm" ? "f16" : quant;
    w[name] = dequantToF32(buf, off, count, effective);
    off += quantBytesFor(count, effective);
  }
  w._layout = L;
  return w;
}

// =============================================================================================
// WGSL kernels (string sources). f32 storage compute — portable across all WebGPU adapters.
// All matmuls are tiled "row-of-output per invocation" GEMV (decode is seq=1; prefill loops rows on
// CPU side, calling GEMV per row — simple + correct; see Throughput note for the batched-GEMM upgrade).
// =============================================================================================
const WGSL = {
  // y = rmsnorm(x) * weight   (x:[H], weight:[H] -> y:[H])
  rmsnorm: `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> P: vec4<u32>;        // P.x = H
var<workgroup> ss: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let H = P.x; let t = lid.x;
  var acc = 0.0;
  var i = t; loop { if (i >= H) { break; } acc += x[i]*x[i]; i += 256u; }
  ss[t] = acc; workgroupBarrier();
  var s = 128u; loop { if (s == 0u) { break; } if (t < s) { ss[t] += ss[t+s]; } workgroupBarrier(); s = s >> 1u; }
  let inv = inverseSqrt(ss[0] / f32(H) + 1e-5);
  var j = t; loop { if (j >= H) { break; } y[j] = x[j]*inv*w[j]; j += 256u; }
}`,

  // y = W * x   where W is [rows, cols] row-major (f32), x:[cols] -> y:[rows]. One row per workgroup.
  matvec: `
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> P: vec4<u32>;        // P.x = rows, P.y = cols
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let cols = P.y; let row = wid.x; let t = lid.x; let base = row * cols;
  var acc = 0.0;
  var i = t; loop { if (i >= cols) { break; } acc += W[base+i]*x[i]; i += 256u; }
  red[t] = acc; workgroupBarrier();
  var s = 128u; loop { if (s == 0u) { break; } if (t < s) { red[t] += red[t+s]; } workgroupBarrier(); s = s >> 1u; }
  if (t == 0u) { y[row] = red[0]; }
}`,

  // RoPE in place on a [n_heads*head_dim] vector (NeoX/Llama interleaved-half layout).
  rope: `
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<uniform> P: vec4<u32>;        // P.x=n_vec_heads, P.y=head_dim, P.z=pos
@group(0) @binding(2) var<uniform> F: vec4<f32>;        // F.x = rope_theta
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nH = P.x; let hd = P.y; let pos = f32(P.z); let half = hd/2u;
  let idx = gid.x; let total = nH*half;
  if (idx >= total) { return; }
  let h = idx / half; let i = idx % half;
  let base = h*hd;
  let freq = pow(F.x, -2.0*f32(i)/f32(hd));
  let ang = pos*freq; let c = cos(ang); let s = sin(ang);
  let a = q[base+i]; let b = q[base+half+i];
  q[base+i] = a*c - b*s;
  q[base+half+i] = a*s + b*c;
}`,

  // GQA attention for ONE query step: scores over cached keys, softmax, weighted sum of values.
  // K,V caches are [seq, n_kv*head_dim]. Each output head maps to kv head (h / (nH/nKV)).
  // One workgroup per query head; sequential over context (decode is cheap; correctness-first).
  attn: `
@group(0) @binding(0) var<storage, read> q: array<f32>;          // [nH*hd]
@group(0) @binding(1) var<storage, read> kc: array<f32>;         // [seq*nKV*hd]
@group(0) @binding(2) var<storage, read> vc: array<f32>;         // [seq*nKV*hd]
@group(0) @binding(3) var<storage, read_write> o: array<f32>;    // [nH*hd]
@group(0) @binding(4) var<uniform> P: vec4<u32>;                 // x=nH y=nKV z=hd w=seqlen
var<workgroup> sc: array<f32, 4096>;                             // scores (caps context @4096)
@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
  let nH=P.x; let nKV=P.y; let hd=P.z; let S=P.w;
  let h = wid.x; let g = nH/nKV; let kvh = h/g;
  let qbase = h*hd; let scale = 1.0/sqrt(f32(hd));
  var mx = -1e30;
  for (var s=0u; s<S; s++) {
    let kb = s*nKV*hd + kvh*hd;
    var dot=0.0; for (var d=0u; d<hd; d++) { dot += q[qbase+d]*kc[kb+d]; }
    dot *= scale; sc[s]=dot; if (dot>mx){mx=dot;}
  }
  var den=0.0; for (var s=0u; s<S; s++){ let e=exp(sc[s]-mx); sc[s]=e; den+=e; }
  for (var d=0u; d<hd; d++){
    var acc=0.0;
    for (var s=0u; s<S; s++){ let vb=s*nKV*hd+kvh*hd; acc += sc[s]*vc[vb+d]; }
    o[qbase+d]=acc/den;
  }
}`,

  // SwiGLU: out = (silu(gate) * up); silu(x)=x*sigmoid(x). gate,up:[ffn] -> out:[ffn]
  swiglu: `
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> P: vec4<u32>;     // P.x = ffn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n=P.x; let i=gid.x; if (i>=n){return;}
  let g=gate[i]; let s=g/(1.0+exp(-g)); out[i]=s*up[i];
}`,

  // elementwise add: a += b
  add: `
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> P: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n=P.x; let i=gid.x; if (i<n){ a[i]=a[i]+b[i]; }
}`,
};

// =============================================================================================
// WebGPU backend
// =============================================================================================
async function initWebGPU() {
  if (!(typeof navigator !== "undefined" && navigator.gpu)) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  return { adapter, device, limits: adapter.limits };
}

function mkPipeline(device, src) {
  const module = device.createShaderModule({ code: src });
  return device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
}

// ---------------------------------------------------------------------------------------------
// GPU-BUFFER REUSE FOR THROUGHPUT (the part that matters for "high throughput"):
//
//  1. WEIGHTS uploaded ONCE at init into persistent storage buffers (GpuStage.weights[]). They are
//     the multi-hundred-MB cost; they never re-upload per token. This is the dominant win — a decode
//     step touches only KB of activations, not GB of weights.
//  2. SCRATCH buffers (h, xn, q, k, v, attn, gate, up, act, ...) are allocated ONCE (scratchBuf keys)
//     and REUSED every forward() call. No per-token allocation => no GC pauses, stable VRAM.
//  3. KV-CACHE buffers are persistent per (seq, layer), grown in place (appendKvGpu copies one row).
//     They never leave the GPU and never cross the network (architecture.md §2.3).
//  4. UNIFORMS come from a fixed ring pool (_uniPool) instead of one allocation per dispatch.
//  5. Pipelines (compiled WGSL) are created ONCE in the constructor and reused for every layer/step.
//
// CORRECTNESS-FIRST vs PEAK PERF (honest): this engine dequantizes weights to f32 at init and uses
// GEMV (one output row per workgroup, seq=1). That is correct and portable. The documented upgrades
// for peak throughput, none of which change the engine's public API:
//   - keep weights in their quantized blocks on-GPU and dequant INSIDE the matmul kernel (4x less VRAM
//     => 4x more layers per tab => fewer tabs for the same model);
//   - tiled GEMM (process a micro-batch of M tokens at once) so prefill and multi-request decode share
//     one matmul — this is the GPU side of the micro-batching the pipeline does across tabs;
//   - enable the `shader-f16` feature to compute in fp16 (2x throughput on supporting GPUs);
//   - batch all per-layer dispatches into ONE command encoder (fewer submits) — already partly done.
// These are localized to this file; the wire protocol and stage interface are unaffected.
// ---------------------------------------------------------------------------------------------
class GpuStage {
  constructor(gpu, dims) {
    this.device = gpu.device;
    this.dims = dims;
    this.L = layerLayout(dims);
    this.pipelines = {};
    for (const k of Object.keys(WGSL)) this.pipelines[k] = mkPipeline(this.device, WGSL[k]);
    this.weights = [];   // per-layer GPU buffers (uploaded once at init — the big win)
    this.scratch = {};   // reusable scratch buffers keyed by name (reused every step, never reallocated)
    // Uniform ring: vec4 uniforms are tiny but written EVERY dispatch. We round-robin a fixed pool of
    // preallocated UNIFORM buffers instead of allocating one per dispatch (no per-token GC churn).
    this._uniPool = [];
    for (let i = 0; i < 64; i++) this._uniPool.push(this.buf(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
    this._uniIdx = 0;
  }

  buf(bytes, usage) {
    return this.device.createBuffer({ size: Math.max(16, (bytes + 3) & ~3), usage });
  }
  // Reusable scratch: same key -> same GPU buffer across forward() calls (no per-token allocation).
  scratchBuf(key, floats) {
    const bytes = floats * 4;
    let b = this.scratch[key];
    if (!b || b._floats < floats) {
      if (b) b.destroy?.();
      b = this.buf(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      b._floats = floats; this.scratch[key] = b;
    }
    return b;
  }
  storage(f32) {
    const b = this.buf(f32.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.device.queue.writeBuffer(b, 0, f32);
    return b;
  }
  uni(vals) { // vec4 uniform (u32 or f32 packed by caller); drawn from the reusable ring pool.
    const b = this._uniPool[this._uniIdx];
    this._uniIdx = (this._uniIdx + 1) % this._uniPool.length;
    this.device.queue.writeBuffer(b, 0, vals.buffer ? vals : new Uint32Array(vals));
    return b;
  }

  // Upload one layer's dequantized weights into persistent GPU storage buffers (called ONCE per layer).
  uploadLayer(w) {
    const g = {};
    for (const name of ["attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w_gate", "w_up", "w_down"]) {
      g[name] = this.storage(w[name]);
    }
    this.weights.push(g);
  }
  uploadExtra(name, f32) { this[name] = this.storage(f32); }

  dispatch(pipe, binds, groups) {
    const layout = this.pipelines[pipe].getBindGroupLayout(0);
    const entries = binds.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    const bg = this.device.createBindGroup({ layout, entries });
    const enc = this.device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pipelines[pipe]); p.setBindGroup(0, bg);
    p.dispatchWorkgroups(...groups); p.end();
    this.device.queue.submit([enc.finish()]);
  }

  async read(buf, floats) {
    const rb = this.buf(floats * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, rb, 0, floats * 4);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0, floats * 4));
    rb.unmap(); rb.destroy();
    return out;
  }

  destroy() {
    for (const lyr of this.weights) for (const k in lyr) lyr[k].destroy?.();
    for (const k in this.scratch) this.scratch[k]?.destroy?.();
    this._uniPool.forEach((b) => b.destroy?.());
    ["embed", "norm", "lm_head"].forEach((k) => this[k]?.destroy?.());
  }
}

// =============================================================================================
// initStage
// =============================================================================================
export async function initStage(spec) {
  const { dims, layers, is_first, is_last, weights } = spec;
  const gpu = await initWebGPU();
  const backend = gpu ? "webgpu" : "wasm";

  const ctx = {
    arch: spec.arch || "llama",
    dims, layers, is_first, is_last,
    backend,
    headDim: Math.floor(dims.hidden_dim / dims.n_heads),
    ffn: dims.ffn_dim || roundFfn(dims.hidden_dim),
    kv: new Map(),              // seq_id -> { K:[layer][f32 flat seq*kvDim], V:[...], len }
    nLayers: layers[1] - layers[0],
    opts: spec.opts || {},
  };

  // Dequantize all layer weights on CPU (correctness-first; see Throughput note for staying-quantized).
  ctx.cpuLayers = weights.layers.map((l) => dequantLayer(l.buf, l.quant, dims));

  if (is_first) {
    ctx.embed = weights.embed
      ? dequantToF32(weights.embed, 0, dims.vocab * dims.hidden_dim, embedQuant(weights))
      : null;
  }
  if (is_last) {
    ctx.finalNorm = weights.norm ? dequantToF32(weights.norm, 0, dims.hidden_dim, "f16") : null;
    ctx.lmHead = weights.lm_head
      ? dequantToF32(weights.lm_head, 0, dims.vocab * dims.hidden_dim, lmHeadQuant(weights))
      : null;
  }

  if (backend === "webgpu") {
    ctx.g = new GpuStage(gpu, dims);
    for (const w of ctx.cpuLayers) ctx.g.uploadLayer(w);
    if (is_first && ctx.embed) ctx.g.uploadExtra("embed", ctx.embed);   // embed kept f32 on GPU
    if (is_last) {
      if (ctx.finalNorm) ctx.g.uploadExtra("norm", ctx.finalNorm);
      if (ctx.lmHead) ctx.g.uploadExtra("lm_head", ctx.lmHead);
    }
    // Once uploaded we can drop the big CPU copies of the matmul weights (keep norms small / on GPU).
    ctx.cpuLayers = null;
  }

  ctx.ready = true;
  return ctx;
}
// The pack stores embed/lm_head with their own quant; default to model quant. Carried as weights.*Quant
// if the loader set it, else assume the layer quant ("q4_0"). Conservative + overridable.
function embedQuant(w) { return w.embedQuant || w.quant || "q4_0"; }
function lmHeadQuant(w) { return w.lmHeadQuant || w.quant || "q4_0"; }

// =============================================================================================
// forward — run this stage's layers for one decode step (or prefill of token_ids on stage 0).
// =============================================================================================
export async function forward(ctx, req) {
  // The last stage needs an async logits readback, so it goes through forwardSafe (one code path).
  if (ctx.is_last) return forwardSafe(ctx, req);

  // ---- assemble the input hidden states for this step ----
  // stage 0 prefill: token_ids -> embed each, run all positions sequentially (fills KV).
  // stage 0 decode:  single token id arrives as token_ids:[id] at the given token_pos.
  // mid stage:       `hidden` (fp16 bits as Uint16/Float16Array) arrives from the previous stage.
  if (ctx.is_first) {
    const ids = req.token_ids && req.token_ids.length ? req.token_ids : [req.token_id];
    let last;
    for (let p = 0; p < ids.length; p++) {
      const pos = (req.token_pos || 0) + p;
      last = await runLayers(ctx, req.seq_id, pos, embedLookup(ctx, ids[p]));
    }
    return { hidden: f32ArrToF16(last) };
  }
  const h = await runLayers(ctx, req.seq_id, req.token_pos, f16ArrToF32(req.hidden));
  return { hidden: f32ArrToF16(h) };
}

// ----- the transformer block stack (dispatches to GPU or CPU kernels) -----
async function runLayers(ctx, seqId, pos, h) {
  return ctx.backend === "webgpu" ? runLayersGpu(ctx, seqId, pos, h) : runLayersCpu(ctx, seqId, pos, h);
}

function ensureKv(ctx, seqId) {
  let kv = ctx.kv.get(seqId);
  if (!kv) {
    kv = { K: Array.from({ length: ctx.nLayers }, () => []), V: Array.from({ length: ctx.nLayers }, () => []), len: 0 };
    ctx.kv.set(seqId, kv);
  }
  return kv;
}

// =============================================================================================
// CPU / WASM-fallback kernels (real math; the slow-but-correct reference path).
// =============================================================================================
function embedLookup(ctx, id) {
  const H = ctx.dims.hidden_dim;
  const src = ctx.embed;
  const out = new Float32Array(H);
  out.set(src.subarray(id * H, id * H + H));
  return out;
}
function rmsnormCpu(x, w, H) {
  let s = 0; for (let i = 0; i < H; i++) s += x[i] * x[i];
  const inv = 1 / Math.sqrt(s / H + 1e-5);
  const o = new Float32Array(H);
  for (let i = 0; i < H; i++) o[i] = x[i] * inv * w[i];
  return o;
}
function matvecCpu(W, x, rows, cols) {
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) { let a = 0; const b = r * cols; for (let c = 0; c < cols; c++) a += W[b + c] * x[c]; y[r] = a; }
  return y;
}
function ropeCpu(vec, nHeads, headDim, pos, theta) {
  const half = headDim >> 1;
  for (let h = 0; h < nHeads; h++) {
    const base = h * headDim;
    for (let i = 0; i < half; i++) {
      const freq = Math.pow(theta, (-2 * i) / headDim);
      const ang = pos * freq, c = Math.cos(ang), s = Math.sin(ang);
      const a = vec[base + i], b = vec[base + half + i];
      vec[base + i] = a * c - b * s;
      vec[base + half + i] = a * s + b * c;
    }
  }
}
function runLayersCpu(ctx, seqId, pos, h) {
  const d = ctx.dims, H = d.hidden_dim, nH = d.n_heads, nKV = d.n_kv_heads, hd = ctx.headDim;
  const qDim = nH * hd, kvDim = nKV * hd, ffn = ctx.ffn, g = nH / nKV;
  const kv = ensureKv(ctx, seqId);
  for (let li = 0; li < ctx.nLayers; li++) {
    const w = ctx.cpuLayers[li];
    // --- attention ---
    const xn = rmsnormCpu(h, w.attn_norm, H);
    const q = matvecCpu(w.wq, xn, qDim, H);
    const k = matvecCpu(w.wk, xn, kvDim, H);
    const v = matvecCpu(w.wv, xn, kvDim, H);
    ropeCpu(q, nH, hd, pos, d.rope_theta);
    ropeCpu(k, nKV, hd, pos, d.rope_theta);
    kv.K[li].push(k); kv.V[li].push(v);
    const S = kv.K[li].length;
    const attnOut = new Float32Array(qDim);
    const scale = 1 / Math.sqrt(hd);
    for (let head = 0; head < nH; head++) {
      const kvh = Math.floor(head / g), qb = head * hd;
      const scores = new Float32Array(S); let mx = -Infinity;
      for (let s = 0; s < S; s++) { const kk = kv.K[li][s]; const kb = kvh * hd; let dot = 0; for (let dd = 0; dd < hd; dd++) dot += q[qb + dd] * kk[kb + dd]; dot *= scale; scores[s] = dot; if (dot > mx) mx = dot; }
      let den = 0; for (let s = 0; s < S; s++) { scores[s] = Math.exp(scores[s] - mx); den += scores[s]; }
      for (let dd = 0; dd < hd; dd++) { let acc = 0; for (let s = 0; s < S; s++) acc += scores[s] * kv.V[li][s][kvh * hd + dd]; attnOut[qb + dd] = acc / den; }
    }
    const proj = matvecCpu(w.wo, attnOut, H, qDim);
    for (let i = 0; i < H; i++) h[i] += proj[i];        // residual
    // --- MLP (SwiGLU) ---
    const xn2 = rmsnormCpu(h, w.ffn_norm, H);
    const gate = matvecCpu(w.w_gate, xn2, ffn, H);
    const up = matvecCpu(w.w_up, xn2, ffn, H);
    const act = new Float32Array(ffn);
    for (let i = 0; i < ffn; i++) { const gv = gate[i]; act[i] = (gv / (1 + Math.exp(-gv))) * up[i]; }
    const down = matvecCpu(w.w_down, act, H, ffn);
    for (let i = 0; i < H; i++) h[i] += down[i];        // residual
  }
  kv.len = (kv.K[0] ? kv.K[0].length : 0);
  return h;
}
function runLastCpu(ctx, h) {
  const H = ctx.dims.hidden_dim;
  const hn = rmsnormCpu(h, ctx.finalNorm, H);
  return matvecCpu(ctx.lmHead, hn, ctx.dims.vocab, H);
}

// =============================================================================================
// WebGPU kernels (mirror of the CPU path; uses persistent weight buffers + reusable scratch).
// =============================================================================================
async function runLayersGpu(ctx, seqId, pos, hF32) {
  const g = ctx.g, d = ctx.dims, H = d.hidden_dim, nH = d.n_heads, nKV = d.n_kv_heads, hd = ctx.headDim;
  const qDim = nH * hd, kvDim = nKV * hd, ffn = ctx.ffn;
  const kv = ensureKvGpu(ctx, seqId);
  const WG = (n) => Math.ceil(n / 256);

  let hBuf = g.scratchBuf("h", H); g.device.queue.writeBuffer(hBuf, 0, hF32);
  const xn = g.scratchBuf("xn", H);
  const q = g.scratchBuf("q", qDim), k = g.scratchBuf("k", kvDim), v = g.scratchBuf("v", kvDim);
  const attn = g.scratchBuf("attn", qDim), proj = g.scratchBuf("proj", H);
  const xn2 = g.scratchBuf("xn2", H);
  const gate = g.scratchBuf("gate", ffn), up = g.scratchBuf("up", ffn), act = g.scratchBuf("act", ffn);
  const down = g.scratchBuf("down", H);

  for (let li = 0; li < ctx.nLayers; li++) {
    const w = g.weights[li];
    const uH = g.uni([H, 0, 0, 0]);
    // attn norm
    g.dispatch("rmsnorm", [hBuf, w.attn_norm, xn, uH], [1]);
    // qkv
    g.dispatch("matvec", [w.wq, xn, q, g.uni([qDim, H, 0, 0])], [qDim]);
    g.dispatch("matvec", [w.wk, xn, k, g.uni([kvDim, H, 0, 0])], [kvDim]);
    g.dispatch("matvec", [w.wv, xn, v, g.uni([kvDim, H, 0, 0])], [kvDim]);
    // rope (q over nH heads, k over nKV heads). rope shader = workgroup_size(64); total invocations = heads*half.
    const WG64 = (n) => Math.max(1, Math.ceil(n / 64));
    g.dispatch("rope", [q, g.uni([nH, hd, pos, 0]), g.uni(new Float32Array([d.rope_theta, 0, 0, 0]))], [WG64(nH * (hd >> 1))]);
    g.dispatch("rope", [k, g.uni([nKV, hd, pos, 0]), g.uni(new Float32Array([d.rope_theta, 0, 0, 0]))], [WG64(nKV * (hd >> 1))]);
    // append to KV cache (read back q/k/v? no — keep K,V on GPU; we grow per-layer cache buffers)
    appendKvGpu(ctx, kv, li, k, v, kvDim);
    const S = kv.len[li];
    // attention: one workgroup per query head
    g.dispatch("attn", [q, kv.K[li], kv.V[li], attn, g.uni([nH, nKV, hd, S])], [nH]);
    // o-proj + residual
    g.dispatch("matvec", [w.wo, attn, proj, g.uni([H, qDim, 0, 0])], [H]);
    g.dispatch("add", [hBuf, proj, uH], [WG(H)]);
    // MLP
    g.dispatch("rmsnorm", [hBuf, w.ffn_norm, xn2, uH], [1]);
    g.dispatch("matvec", [w.w_gate, xn2, gate, g.uni([ffn, H, 0, 0])], [ffn]);
    g.dispatch("matvec", [w.w_up, xn2, up, g.uni([ffn, H, 0, 0])], [ffn]);
    g.dispatch("swiglu", [gate, up, act, g.uni([ffn, 0, 0, 0])], [WG(ffn)]);
    g.dispatch("matvec", [w.w_down, act, down, g.uni([H, ffn, 0, 0])], [H]);
    g.dispatch("add", [hBuf, down, uH], [WG(H)]);
  }
  return g.read(hBuf, H);   // -> f32 [H], handed to the next stage as fp16, or to forwardSafe for last
}

// GPU KV cache: persistent, growing per-layer K/V buffers (never leave the tab).
function ensureKvGpu(ctx, seqId) {
  let kv = ctx.kv.get(seqId);
  if (!kv) { kv = { K: new Array(ctx.nLayers).fill(null), V: new Array(ctx.nLayers).fill(null), len: new Array(ctx.nLayers).fill(0), cap: new Array(ctx.nLayers).fill(0) }; ctx.kv.set(seqId, kv); }
  return kv;
}
function appendKvGpu(ctx, kv, li, kBuf, vBuf, kvDim) {
  const g = ctx.g;
  const maxCtx = ctx.opts.max_ctx || 4096;
  if (!kv.K[li]) {
    kv.K[li] = g.buf(maxCtx * kvDim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    kv.V[li] = g.buf(maxCtx * kvDim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  }
  const off = kv.len[li] * kvDim * 4;
  const enc = g.device.createCommandEncoder();
  enc.copyBufferToBuffer(kBuf, 0, kv.K[li], off, kvDim * 4);
  enc.copyBufferToBuffer(vBuf, 0, kv.V[li], off, kvDim * 4);
  g.device.queue.submit([enc.finish()]);
  kv.len[li] += 1;
}

// =============================================================================================
// sampling (temperature / nucleus top-p) — runs on CPU over the logits readback.
// =============================================================================================
export function sample(logits, temperature = 0.7, top_p = 0.95) {
  const n = logits.length;
  if (temperature <= 0) { let bi = 0, bv = -Infinity; for (let i = 0; i < n; i++) if (logits[i] > bv) { bv = logits[i]; bi = i; } return bi; }
  let mx = -Infinity; for (let i = 0; i < n; i++) if (logits[i] > mx) mx = logits[i];
  const probs = new Float32Array(n); let sum = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp((logits[i] - mx) / temperature); probs[i] = e; sum += e; }
  for (let i = 0; i < n; i++) probs[i] /= sum;
  // nucleus: sort indices by prob desc, keep until cumulative >= top_p
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
  let cum = 0, cut = n;
  for (let r = 0; r < n; r++) { cum += probs[idx[r]]; if (cum >= top_p) { cut = r + 1; break; } }
  let r = Math.random() * cum, acc = 0;
  for (let j = 0; j < cut; j++) { acc += probs[idx[j]]; if (r <= acc) return idx[j]; }
  return idx[0];
}

// =============================================================================================
// freeStage
// =============================================================================================
export function freeStage(ctx) {
  if (!ctx) return;
  if (ctx.backend === "webgpu" && ctx.g) {
    for (const kv of ctx.kv.values()) { (kv.K || []).forEach((b) => b?.destroy?.()); (kv.V || []).forEach((b) => b?.destroy?.()); }
    ctx.g.destroy();
  }
  ctx.kv.clear(); ctx.ready = false;
}

// =============================================================================================
// forwardSafe — the last stage's path: run layers, then ASYNC-read logits and sample.
// (Used by forward() for is_last, and exported as forwardLast for explicit callers.)
// =============================================================================================
export async function forwardSafe(ctx, req) {
  if (!ctx.is_last) return forward(ctx, req);
  // run this stage's layers, then async-read logits
  const d = ctx.dims, H = d.hidden_dim;
  let h;
  if (ctx.is_first) {
    const ids = req.token_ids && req.token_ids.length ? req.token_ids : [req.token_id];
    let last; for (let p = 0; p < ids.length; p++) last = await runLayers(ctx, req.seq_id, (req.token_pos || 0) + p, embedLookup(ctx, ids[p]));
    h = last;
  } else {
    h = await runLayers(ctx, req.seq_id, req.token_pos, f16ArrToF32(req.hidden));
  }
  let logits;
  if (ctx.backend === "webgpu") {
    const g = ctx.g, V = d.vocab;
    const hBuf = g.storage(h);
    const hn = g.scratchBuf("hn", H);
    g.dispatch("rmsnorm", [hBuf, g.norm, hn, g.uni([H, 0, 0, 0])], [1]);
    const lg = g.scratchBuf("logits", V);
    g.dispatch("matvec", [g.lm_head, hn, lg, g.uni([V, H, 0, 0])], [V]);
    logits = await g.read(lg, V);
    hBuf.destroy();
  } else {
    logits = runLastCpu(ctx, h);
  }
  return { token_id: sample(logits, req.temperature ?? 0.7, req.top_p ?? 0.95) };
}

// =============================================================================================
// Web Worker harness. If this module is the Worker's entry, handle postMessage ops so heavy compute
// stays off the UI thread. The tab can EITHER import {initStage,forward,freeStage} directly OR spin
// `new Worker('inference-worker.js',{type:'module'})` and talk the protocol below.
//
//   main -> worker : {op:"init", spec}              spec = initStage arg (weights buffers transferred)
//   worker -> main : {op:"ready", info:{backend,nLayers}}
//   main -> worker : {op:"forward", req}            req = forward arg (hidden ArrayBuffer transferred)
//   worker -> main : {op:"result", out}             out = {hidden:ArrayBuffer}|{token_id}
//   main -> worker : {op:"free"}                    -> {op:"freed"}
// =============================================================================================
const inWorker = typeof self !== "undefined" && typeof window === "undefined" && typeof self.postMessage === "function";
if (inWorker) {
  let CTX = null;
  self.onmessage = async (e) => {
    const m = e.data || {};
    try {
      if (m.op === "init") {
        CTX = await initStage(m.spec);
        self.postMessage({ op: "ready", info: { backend: CTX.backend, nLayers: CTX.nLayers } });
      } else if (m.op === "forward") {
        const req = m.req;
        if (req.hidden && req.hidden instanceof ArrayBuffer) req.hidden = new Float16Array(req.hidden);
        const out = await forwardSafe(CTX, req);
        if (out.hidden) self.postMessage({ op: "result", out: { hidden: out.hidden.buffer } }, [out.hidden.buffer]);
        else self.postMessage({ op: "result", out });
      } else if (m.op === "free") {
        freeStage(CTX); CTX = null; self.postMessage({ op: "freed" });
      }
    } catch (err) {
      self.postMessage({ op: "error", message: String(err && err.message || err) });
    }
  };
}

export { forwardSafe as forwardLast };
