// ce-tabnet shard loader — content-addressed weight fetch for ONE pipeline stage.
// Owner: Implementer 6. See docs/module-contract.md §D (Impl 6), models/README.md, architecture.md §7.
//
// A "stage" holds a contiguous layer range [lo, hi). This module fetches exactly those layers'
// content-addressed objects (plus embed/lm_head/norm when the stage is first/last), verifies each
// object's SHA-256 (the CID), streams with progress, and caches every object by CID in the Cache
// Storage API (with an IndexedDB index for LRU eviction) so a returning tab — or a freshly recruited
// replica during healing — is effectively instant and never re-downloads bytes it already holds.
//
// Transports supported, chosen per manifest object, all uniform behind fetchOne():
//   - HTTP(S) range fetch:  GET <baseUrl><file>  with `Range: bytes=a-b`  (R2 / S3 / CDN / python http.server)
//   - whole-object fetch:   GET <baseUrl><cid>.bin                         (one object per file, no range)
//   - CE blob endpoint:     GET <ceBlobBase><cid>  or  <ceBlobBase>?cid=<cid>  (CE blob gateway)
//
// SHARED INTERFACE (agree verbatim with inference.js, Implementer 5):
//   load(...) resolves to a `weights` object:
//     {
//       arch, dims:{hidden_dim,n_heads,n_kv_heads,vocab,rope_theta},
//       embed?:   ArrayBuffer,                       // present iff needEmbed (stage 0)
//       lm_head?: ArrayBuffer, norm?: ArrayBuffer,   // present iff needLmHead (last stage)
//       layers: [{ idx, buf:ArrayBuffer, quant, parts?:[{role,buf}] }]  // exactly [lo,hi), in order
//     }
//   inference.initStage consumes this object and uploads buffers to WebGPU.
//
// Pure data dependency on model-config.js (the model registry). No protocol/DO imports.

import { MODELS } from "./model-config.js";

// ---------------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------------
//
// load(spec) — fetch one stage's weights.
//   spec = {
//     model_id,                 // key into MODELS (supplies defaults for baseUrl/manifestRef)
//     baseUrl?,                 // override the HTTP(S) range/CDN base (defaults to MODELS[model_id].baseUrl)
//     manifestRef?,             // override manifest filename (default MODELS[model_id].manifestRef)
//     manifest?,                // pre-fetched manifest object (skips the manifest GET; used by the DO/UI)
//     ceBlobBase?,              // if set, fetch objects from a CE blob gateway by CID instead of baseUrl
//     layers: [lo, hi],         // the contiguous layer range this stage owns (hi exclusive)
//     needEmbed?:  bool,        // stage 0 wants token-embedding weights
//     needLmHead?: bool,        // last stage wants final norm + lm_head
//     onProgress?: (p) => {},   // p = { phase, loaded, total, objects, objectsDone, cid? }
//     signal?,                  // optional AbortSignal to cancel the whole load (healing / tab close)
//     concurrency?,             // max parallel object fetches (default 4 — friendly to phones)
//     verify?: bool,            // SHA-256 verify each object against its CID (default true)
//   }
//   -> Promise<weights>  (the SHARED INTERFACE shape above)
//
// Also exported: prefetch(spec) (warm the cache without holding the weights object),
//   evictModel(model_id), cacheStats(), and the lower-level fetchManifest().
// ---------------------------------------------------------------------------------------------------

const DEFAULT_CONCURRENCY = 4;

export async function load(spec) {
  const {
    model_id,
    layers,
    needEmbed = false,
    needLmHead = false,
    onProgress = noop,
    signal,
    concurrency = DEFAULT_CONCURRENCY,
    verify = true,
  } = spec;

  const m = MODELS[model_id] || {};
  const baseUrl = normalizeBase(spec.baseUrl ?? m.baseUrl ?? "");
  const ceBlobBase = spec.ceBlobBase ? normalizeBase(spec.ceBlobBase) : null;
  const manifestRef = spec.manifestRef ?? m.manifestRef ?? "manifest.json";

  if (!Array.isArray(layers) || layers.length !== 2) {
    throw new Error("shard-loader.load: spec.layers must be [lo, hi)");
  }
  const [lo, hi] = layers;
  if (!(hi > lo)) throw new Error(`shard-loader.load: empty layer range [${lo},${hi})`);

  onProgress({ phase: "manifest", loaded: 0, total: 0, objects: 0, objectsDone: 0 });
  const manifest = spec.manifest ?? (await fetchManifest(baseUrl + manifestRef, signal));
  validateManifestRange(manifest, lo, hi);

  // Build the object plan: which content-addressed objects this stage must fetch, with sizes for progress.
  const plan = [];
  for (let i = lo; i < hi; i++) {
    const L = manifest.layers[i];
    if (!L) throw new Error(`manifest missing layer ${i}`);
    plan.push({ kind: "layer", idx: i, obj: L });
  }
  if (needEmbed) {
    if (!manifest.embed) throw new Error("needEmbed but manifest has no embed object");
    plan.push({ kind: "embed", obj: manifest.embed });
  }
  if (needLmHead) {
    if (!manifest.lm_head || !manifest.norm) throw new Error("needLmHead but manifest lacks lm_head/norm");
    plan.push({ kind: "lm_head", obj: manifest.lm_head });
    plan.push({ kind: "norm", obj: manifest.norm });
  }

  // Total bytes for a single monotonic progress bar across the whole stage load.
  const total = plan.reduce((sum, p) => sum + objectBytes(p.obj), 0);
  let loaded = 0;
  let objectsDone = 0;
  const objects = plan.length;
  const tickProgress = (deltaBytes, cid) => {
    loaded += deltaBytes;
    onProgress({ phase: "fetch", loaded, total, objects, objectsDone, cid });
  };

  const ctx = { baseUrl, ceBlobBase, manifest, signal, verify, tickProgress };

  // Fetch all objects with a bounded-concurrency pool (keeps phone memory + sockets sane).
  const results = new Array(plan.length);
  await pool(plan, concurrency, async (item, slot) => {
    throwIfAborted(signal);
    const data = await fetchObjectFull(ctx, item.obj);
    results[slot] = { item, data };
    objectsDone += 1;
    onProgress({ phase: "fetch", loaded, total, objects, objectsDone, cid: cidOf(item.obj) });
  });

  // Assemble the SHARED weights object in stage order.
  const out = {
    arch: manifest.arch,
    dims: dimsOf(manifest),
    layers: new Array(hi - lo),
  };
  for (const { item, data } of results) {
    if (item.kind === "layer") {
      out.layers[item.idx - lo] = shapeLayer(item, data);
    } else if (item.kind === "embed") {
      out.embed = singleBuf(data);
    } else if (item.kind === "lm_head") {
      out.lm_head = singleBuf(data);
    } else if (item.kind === "norm") {
      out.norm = singleBuf(data);
    }
  }
  onProgress({ phase: "done", loaded: total, total, objects, objectsDone, cid: null });
  return out;
}

// Warm the cache for a stage without holding the weights object (e.g. proactive replica prep).
export async function prefetch(spec) {
  await load({ ...spec, onProgress: spec.onProgress ?? noop, verify: spec.verify ?? true });
  return true;
}

export async function fetchManifest(url, signal) {
  const r = await fetch(url, { signal, cache: "no-cache" });
  if (!r.ok) throw new Error(`manifest fetch ${r.status} @ ${url}`);
  return r.json();
}

// ---------------------------------------------------------------------------------------------------
// Object fetch (range + CE blob + cache + verify), returns ArrayBuffer or {parts:[{role,buf}]}
// ---------------------------------------------------------------------------------------------------

// An object is one of:
//   { cid, bytes, quant?, shape?, file?, url?, byte_range?[a,b] }                      // single object
//   { parts: [{ role, cid, bytes, file?, url?, byte_range? }], quant? }                // sub-layer parts
// fetchObjectFull resolves to ArrayBuffer (single) or { parts:[{role,buf}] } (multi-part).
async function fetchObjectFull(ctx, obj) {
  if (Array.isArray(obj.parts) && obj.parts.length) {
    const parts = new Array(obj.parts.length);
    // Parts of one object are small; fetch them sequentially to bound memory on tiny devices.
    for (let i = 0; i < obj.parts.length; i++) {
      parts[i] = { role: obj.parts[i].role, buf: await fetchOne(ctx, obj.parts[i]) };
    }
    return { parts };
  }
  return fetchOne(ctx, obj);
}

async function fetchOne(ctx, obj) {
  const cid = cidOf(obj);
  if (!cid) throw new Error("object has no cid (content addressing is required)");

  // 1) Cache hit — return immediately, still counting bytes toward progress.
  const cached = await cacheGet(cid);
  if (cached) {
    ctx.tickProgress(cached.byteLength, cid);
    touchIndex(cid); // LRU recency bump (fire-and-forget)
    return cached;
  }

  // 2) Network fetch (range or whole), streamed with progress.
  const { url, headers } = resolveRequest(ctx, obj);
  const expected = objectBytes(obj);
  const buf = await fetchStreaming(url, headers, ctx.signal, (delta) => ctx.tickProgress(delta, cid), expected);

  // 3) Integrity: the CID IS the SHA-256 of the object content (verify-on-arrival).
  if (ctx.verify) {
    const ok = await verifyCid(buf, cid);
    if (!ok) throw new Error(`integrity check failed for ${cid} (got ${await sha256Hex(buf)})`);
  }

  // 4) Cache for next time (returning tab / replica fan-out is then instant).
  await cachePut(cid, buf, ctx.manifest.model_id || "");
  return buf;
}

// Decide URL + headers for an object across the three transports.
function resolveRequest(ctx, obj) {
  const cid = cidOf(obj);
  // CE blob gateway path: address purely by CID.
  if (ctx.ceBlobBase) {
    const url = ctx.ceBlobBase.includes("?")
      ? `${ctx.ceBlobBase}${encodeURIComponent(cid)}`   // e.g. "...?cid="  → "...?cid=<cid>"
      : `${ctx.ceBlobBase}${cid}`;                       // e.g. "gateway/blob/" → "gateway/blob/<cid>"
    return { url, headers: {} }; // CE blobs are whole objects addressed by CID; no range needed
  }
  // HTTP(S) path. Explicit url > file in baseUrl > "<cid>.bin" in baseUrl.
  const url = obj.url ?? `${ctx.baseUrl}${obj.file ?? `${cid}.bin`}`;
  const headers = {};
  if (Array.isArray(obj.byte_range)) {
    // Object lives inside a packed file (e.g. several layers in one .bin) — slice it server-side.
    headers.Range = `bytes=${obj.byte_range[0]}-${obj.byte_range[1]}`;
  }
  return { url, headers };
}

// Stream a response body, reporting incremental progress; returns the full ArrayBuffer.
async function fetchStreaming(url, headers, signal, onDelta, expectedBytes) {
  const r = await fetch(url, { headers, signal });
  // 200 for whole-object, 206 for a satisfied Range request.
  if (!r.ok && r.status !== 206) throw new Error(`object fetch ${r.status} @ ${url}`);

  // If the server ignored a Range request (returned 200 for the whole packed file), slice locally.
  const wantedRange = headers.Range;
  if (wantedRange && r.status === 200) {
    const whole = await drain(r, signal, onDelta);
    const [a, b] = wantedRange.replace("bytes=", "").split("-").map(Number);
    return whole.slice(a, b + 1);
  }

  // Prefer streaming so progress is smooth and memory peaks are bounded.
  if (r.body && typeof r.body.getReader === "function") {
    return drain(r, signal, onDelta, expectedBytes);
  }
  // Fallback: no streaming body available (some test shims) — single read.
  const buf = await r.arrayBuffer();
  onDelta(buf.byteLength);
  return buf;
}

async function drain(response, signal, onDelta, expectedBytes) {
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    throwIfAborted(signal);
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onDelta(value.byteLength);
  }
  // Concatenate into one contiguous ArrayBuffer.
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  void expectedBytes; // advisory only; actual size is authoritative
  return out.buffer;
}

// ---------------------------------------------------------------------------------------------------
// Integrity — CID === lowercase hex SHA-256 of the object bytes
// ---------------------------------------------------------------------------------------------------

async function verifyCid(buf, cid) {
  // Accept a few CID spellings: raw 64-hex, or "sha256:<hex>", or "sha256-<hex>".
  const hex = normalizeCidToHex(cid);
  if (!hex) return true; // non-hash CID scheme (e.g. CE multibase) — skip strict hex check, trust transport
  const got = await sha256Hex(buf);
  return timingSafeEq(got, hex);
}

function normalizeCidToHex(cid) {
  const s = String(cid).toLowerCase();
  const m = s.match(/^(?:sha-?256[:\-])?([0-9a-f]{64})$/);
  return m ? m[1] : null;
}

async function sha256Hex(buf) {
  const ab = buf instanceof ArrayBuffer ? buf : buf.buffer;
  const digest = await crypto.subtle.digest("SHA-256", ab);
  const bytes = new Uint8Array(digest);
  let hex = "";
  for (let i = 0; i < bytes.length; i++) hex += bytes[i].toString(16).padStart(2, "0");
  return hex;
}

function timingSafeEq(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

// ---------------------------------------------------------------------------------------------------
// Cache: Cache Storage API for the bytes + IndexedDB for an LRU index. Returning tab = instant.
// ---------------------------------------------------------------------------------------------------

const CACHE_NAME = "ce-tabnet-shards-v1";
const IDB_NAME = "ce-tabnet-shard-index";
const IDB_STORE = "objects";
// Soft cap on total cached bytes; LRU-evict beyond this. 8 GB default (Cache Storage is large on desktop).
const CACHE_BUDGET_BYTES = 8 * 1024 * 1024 * 1024;

const hasCacheStorage = () => typeof caches !== "undefined";
const hasIDB = () => typeof indexedDB !== "undefined";

function cacheKey(cid) { return `https://ce-tabnet.shard/${encodeURIComponent(cid)}`; }

async function cacheGet(cid) {
  if (!hasCacheStorage()) return memGet(cid);
  try {
    const c = await caches.open(CACHE_NAME);
    const res = await c.match(cacheKey(cid));
    if (!res) return null;
    return await res.arrayBuffer();
  } catch {
    return memGet(cid);
  }
}

async function cachePut(cid, buf, model_id) {
  if (!hasCacheStorage()) { memPut(cid, buf); return; }
  try {
    const c = await caches.open(CACHE_NAME);
    await c.put(cacheKey(cid), new Response(buf, {
      headers: { "Content-Type": "application/octet-stream", "Content-Length": String(buf.byteLength) },
    }));
    await putIndex(cid, buf.byteLength, model_id);
    await maybeEvict();
  } catch {
    memPut(cid, buf);
  }
}

// In-memory fallback (no Cache Storage / private mode). Bounded by simple insertion-order LRU.
const mem = new Map();
let memBytes = 0;
const MEM_BUDGET = 512 * 1024 * 1024;
function memGet(cid) { const b = mem.get(cid); if (b) { mem.delete(cid); mem.set(cid, b); } return b || null; }
function memPut(cid, buf) {
  if (mem.has(cid)) { memBytes -= mem.get(cid).byteLength; mem.delete(cid); }
  mem.set(cid, buf); memBytes += buf.byteLength;
  while (memBytes > MEM_BUDGET && mem.size > 1) {
    const k = mem.keys().next().value; memBytes -= mem.get(k).byteLength; mem.delete(k);
  }
}

// ---- IndexedDB LRU index (cid -> {bytes, model_id, lastUsed}) ----
function idb() {
  return new Promise((resolve) => {
    if (!hasIDB()) return resolve(null);
    let req;
    try { req = indexedDB.open(IDB_NAME, 1); } catch { return resolve(null); }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        const os = db.createObjectStore(IDB_STORE, { keyPath: "cid" });
        os.createIndex("lastUsed", "lastUsed");
        os.createIndex("model_id", "model_id");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null); // degrade gracefully
  });
}

function tx(db, mode, fn) {
  return new Promise((resolve, reject) => {
    if (!db) return resolve(undefined);
    const t = db.transaction(IDB_STORE, mode);
    const store = t.objectStore(IDB_STORE);
    let result;
    Promise.resolve(fn(store, t)).then((r) => { result = r; }).catch(reject);
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

async function putIndex(cid, bytes, model_id) {
  const db = await idb();
  await tx(db, "readwrite", (store) => store.put({ cid, bytes, model_id, lastUsed: Date.now() }));
}

async function touchIndex(cid) {
  const db = await idb();
  await tx(db, "readwrite", (store) => {
    const g = store.get(cid);
    g.onsuccess = () => { const rec = g.result; if (rec) { rec.lastUsed = Date.now(); store.put(rec); } };
  });
}

async function maybeEvict() {
  const db = await idb();
  if (!db) return;
  const rows = await tx(db, "readonly", (store) => new Promise((res) => {
    const out = []; const cur = store.index("lastUsed").openCursor();
    cur.onsuccess = () => { const c = cur.result; if (c) { out.push(c.value); c.continue(); } else res(out); };
  }));
  if (!rows) return;
  let total = rows.reduce((s, r) => s + (r.bytes || 0), 0);
  if (total <= CACHE_BUDGET_BYTES) return;
  // Evict least-recently-used first (rows are ascending by lastUsed).
  const cache = hasCacheStorage() ? await caches.open(CACHE_NAME) : null;
  for (const r of rows) {
    if (total <= CACHE_BUDGET_BYTES) break;
    if (cache) await cache.delete(cacheKey(r.cid));
    await tx(db, "readwrite", (store) => store.delete(r.cid));
    total -= r.bytes || 0;
  }
}

export async function cacheStats() {
  const db = await idb();
  const rows = (await tx(db, "readonly", (store) => new Promise((res) => {
    const out = []; const cur = store.openCursor();
    cur.onsuccess = () => { const c = cur.result; if (c) { out.push(c.value); c.continue(); } else res(out); };
  }))) || [];
  return {
    objects: rows.length,
    bytes: rows.reduce((s, r) => s + (r.bytes || 0), 0),
    budget: CACHE_BUDGET_BYTES,
    backend: hasCacheStorage() ? "cache-storage" : "memory",
  };
}

export async function evictModel(model_id) {
  const db = await idb();
  if (!db) return 0;
  const rows = await tx(db, "readonly", (store) => new Promise((res) => {
    const out = []; const cur = store.index("model_id").openCursor(IDBKeyRange.only(model_id));
    cur.onsuccess = () => { const c = cur.result; if (c) { out.push(c.value); c.continue(); } else res(out); };
  }));
  const cache = hasCacheStorage() ? await caches.open(CACHE_NAME) : null;
  let n = 0;
  for (const r of rows || []) {
    if (cache) await cache.delete(cacheKey(r.cid));
    await tx(db, "readwrite", (store) => store.delete(r.cid));
    n++;
  }
  return n;
}

// ---------------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------------

function noop() {}
function normalizeBase(b) { return b && !b.endsWith("/") && !b.includes("?") ? b + "/" : b; }
function cidOf(obj) { return obj.cid ?? obj.id ?? null; }
function objectBytes(obj) {
  if (Array.isArray(obj.parts)) return obj.parts.reduce((s, p) => s + (p.bytes || 0), 0);
  if (typeof obj.bytes === "number") return obj.bytes;
  if (Array.isArray(obj.byte_range)) return obj.byte_range[1] - obj.byte_range[0] + 1;
  return 0;
}
function dimsOf(m) {
  return {
    hidden_dim: m.hidden_dim, n_heads: m.n_heads, n_kv_heads: m.n_kv_heads,
    vocab: m.vocab, rope_theta: m.rope_theta,
  };
}
function singleBuf(data) { return data instanceof ArrayBuffer ? data : data.buf ?? data; }
function shapeLayer(item, data) {
  if (data && data.parts) return { idx: item.idx, parts: data.parts, quant: item.obj.quant };
  return { idx: item.idx, buf: data, quant: item.obj.quant };
}

function validateManifestRange(manifest, lo, hi) {
  if (!manifest || !Array.isArray(manifest.layers)) throw new Error("manifest missing layers[]");
  if (lo < 0 || hi > manifest.layers.length) {
    throw new Error(`stage range [${lo},${hi}) out of bounds for ${manifest.layers.length}-layer model`);
  }
}

function throwIfAborted(signal) {
  if (signal && signal.aborted) {
    const e = new Error("aborted");
    e.name = "AbortError";
    throw e;
  }
}

// Bounded-concurrency worker pool over an array; fn(item, originalIndex).
async function pool(items, concurrency, fn) {
  const n = items.length;
  let next = 0;
  const workers = new Array(Math.min(Math.max(1, concurrency), Math.max(1, n))).fill(0).map(async () => {
    for (;;) {
      const i = next++;
      if (i >= n) return;
      await fn(items[i], i);
    }
  });
  await Promise.all(workers);
}
