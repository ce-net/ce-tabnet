// ce-tabnet tab runtime — the stage tab's lifecycle: join, load, compute, heal.
// Owner: Implementer 4. See docs/module-contract.md §D (Impl 4) and docs/architecture.md §2,§6.
//
// Reuses the CE browser-node identity + capability detection from web/site/node.html
// (gpuName, cpuBench, vramMb, detect). A tab IS a CE browser node.

import * as P from "./protocol.js";
import { load as loadShard } from "./shard-loader.js";
// The real pipeline-stage engine lives in inference-worker.js (WebGPU + a real CPU/WASM
// fallback). inference.js is only an interface reference (its bodies throw on purpose).
// We import the engine same-thread here; importing the module also installs the
// Float16Array polyfill used below for activation framing.
import { initStage, forward, freeStage } from "./inference-worker.js";

// ---- CE identity (same scheme as node.html) ----
export function nodeId() {
  let id = localStorage.getItem("ce_node_id");
  if (!id) { const b = new Uint8Array(32); crypto.getRandomValues(b);
    id = [...b].map((x) => x.toString(16).padStart(2, "0")).join(""); localStorage.setItem("ce_node_id", id); }
  return id;
}

// ---- capability detection (ported verbatim from node.html) ----
export function gpuName() { try { const c = document.createElement("canvas"); const gl = c.getContext("webgl") || c.getContext("experimental-webgl");
  if (!gl) return ""; const d = gl.getExtension("WEBGL_debug_renderer_info");
  let s = d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  return (s || "").replace(/\s*\(.*?\)\s*/g, " ").replace(/(ANGLE|OpenGL|Metal|Direct3D|Vulkan).*/i, "").trim(); } catch { return ""; } }
export function cpuBench() { const t0 = performance.now(); let x = 1, iters = 0;
  while (performance.now() - t0 < 25) { for (let i = 0; i < 50000; i++) x += Math.sqrt(i + x * 1e-9); iters += 50000; }
  const ms = performance.now() - t0; return x === Infinity ? 0 : Math.round(iters / ms / 1000 * 10) / 10; }
export async function vramMb() { try { if (!navigator.gpu) return 0; const a = await navigator.gpu.requestAdapter(); if (!a) return 0;
  const l = a.limits || {}; const max = Math.max(l.maxBufferSize || 0, l.maxStorageBufferBindingSize || 0); return max ? Math.round(max / 1048576) : 0; } catch { return 0; } }
export async function detect() {
  const cores = navigator.hardwareConcurrency || 0;
  const ram = navigator.deviceMemory || 0;
  let storage = 0; try { const e = await navigator.storage?.estimate?.(); if (e?.quota) storage = e.quota / 1e9; } catch {}
  return { cores, ram_gb: ram, storage_gb: Math.round(storage * 10) / 10, gpu: gpuName(),
           webgpu: !!navigator.gpu, platform: navigator.platform || "", vram_mb: await vramMb(), cpu_mark: cpuBench() };
}

// ---- the runtime ----
// new TabnetNode({ run, wsBase, onLog, onState }).start()
export class TabnetNode {
  constructor({ run, wsBase, onLog = () => {}, onState = () => {}, onMetrics = () => {} }) {
    this.run = run; this.wsBase = wsBase; this.onLog = onLog; this.onState = onState; this.onMetrics = onMetrics;
    this.id = nodeId(); this.ws = null; this.hb = null; this.ctx = null; this.stage = null;
    // honest live-throughput accounting: every activation/token that actually passes through THIS stage.
    this.tokensThrough = 0;        // total tokens this stage has computed since assignment
    this._win = [];                // timestamps (ms) of recent tokens, for a rolling tok/s rate
    this.S = null; this.R = 1;     // pipeline shape, learned from welcome/route-update if provided
  }

  // Rolling tokens/sec over a 5 s window — what's genuinely "flowing through you" right now.
  _tick() {
    const now = performance.now();
    this.tokensThrough++;
    this._win.push(now);
    while (this._win.length && now - this._win[0] > 5000) this._win.shift();
    const span = this._win.length > 1 ? (now - this._win[0]) / 1000 : 0;
    const rate = span > 0 ? (this._win.length - 1) / span : 0;
    this.onMetrics({ tokensThrough: this.tokensThrough, tokPerSec: Math.round(rate * 10) / 10, stage: this.stage });
  }

  async start() {
    this.caps = await detect();
    this.onState({ phase: "connecting", caps: this.caps });
    this._connect();
  }

  _connect() {
    this.ws = new WebSocket(`${this.wsBase}/run/${this.run}/ws?role=stage`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = () => {
      this.ws.send(JSON.stringify(P.join(this.run, this.id, this.caps)));
      clearInterval(this.hb);
      this.hb = setInterval(() => this._send(P.hb(this.run, this.id)), 10000);
      this.onLog("joined run " + this.run);
    };
    this.ws.onmessage = (ev) => this._onMessage(ev);
    this.ws.onclose = () => { clearInterval(this.hb); this.onState({ phase: "disconnected" }); setTimeout(() => this._connect(), 2000); };
    this.ws.onerror = () => { try { this.ws.close(); } catch {} };
  }

  async _onMessage(ev) {
    // binary frame = activation hot path; text = JSON control/relay
    if (ev.data instanceof ArrayBuffer) { const { meta, payload } = P.decodeActivation(ev.data); return this._onActivation(meta, payload); }
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    switch (m.t) {
      case P.T.WELCOME:       if (typeof m.S === "number") this.S = m.S; if (typeof m.R === "number") this.R = m.R; return;
      case P.T.ASSIGN_STAGE:  if (typeof m.S === "number") this.S = m.S; return this._onAssign(m);
      case P.T.ROUTE_UPDATE:  this.prevNode = m.prev_node; this.nextNode = m.next_node; return;
      case P.T.PROMPT_BEGIN:  return this._onPromptBegin(m);          // stage 0
      case P.T.ACTIVATION:    return this._onActivation(m, null);     // JSON fallback path
      case P.T.TOKEN:         return this._onTokenForStage0(m);       // stage 0: next decode step
      case P.T.RECRUIT:       return this._onRecruit(m);              // healing
      case P.T.EVICT:         return this._onEvict(m);
      default: return;
    }
  }

  // ---- runtime bodies ----
  async _onAssign(m) {
    this.stage = m.stage; this.isFirst = m.is_first; this.isLast = m.is_last;
    this.prevNode = m.prev_node; this.nextNode = m.next_node;
    this._modelId = m.model_id;
    // Per-sequence decode bookkeeping for stage 0 (autoregressive driver) and the last stage.
    this.seqState = this.seqState || new Map(); // seq_id -> { pos, max_tokens, temperature, top_p }
    this.onState({ phase: "loading", stage: m.stage, layers: m.layers });
    const weights = await loadShard({ model_id: m.model_id, manifestRef: m.manifest_ref,
      layers: m.layers, needEmbed: m.is_first, needLmHead: m.is_last });
    this.ctx = await initStage({ arch: weights.arch, dims: weights.dims, layers: m.layers,
      is_first: m.is_first, is_last: m.is_last, weights });
    this._send(P.ready(this.run, this.id, m.stage, m.layers, weightBytes(weights)));
    this.onState({ phase: "ready", stage: m.stage });
    this.onLog(`stage ${m.stage} ready (layers ${m.layers[0]}..${m.layers[1]})`);
  }
  async _onActivation(meta, payload) {
    // run our layers, then forward hidden to the next stage OR emit a token if this is the last stage
    const out = await forward(this.ctx, { seq_id: meta.seq_id, token_pos: meta.token_pos,
      hidden: payload ? new Float16Array(payload) : meta.data });
    this._tick();
    if (this.isLast) {
      // The last stage sampled the next token. The coordinator fans it to operators AND
      // back to stage 0 to drive the next decode step (autoregression closes the loop).
      this._send(P.token(this.run, this.id, meta.seq_id, meta.token_pos, out.token_id, "", false));
    } else {
      this._sendActivation(meta.seq_id, meta.token_pos, out.hidden);
    }
  }
  async _onPromptBegin(m) {
    // Stage 0 prefills the prompt token_ids (positions 0..L-1), then the produced hidden
    // state for the LAST prompt position hops down the pipeline. We remember the next
    // decode position and generation limits so inbound sampled tokens continue the sequence.
    const ids = (m.token_ids && m.token_ids.length) ? m.token_ids : [1]; // BOS fallback if not pre-tokenized
    this.seqState = this.seqState || new Map();
    this.seqState.set(m.seq_id, {
      pos: ids.length,                       // next position to decode after the prompt
      max_tokens: m.max_tokens ?? 256,
      generated: 0,
      temperature: m.temperature ?? 0.7,
      top_p: m.top_p ?? 0.95,
    });
    const out = await forward(this.ctx, { seq_id: m.seq_id, token_pos: 0, token_ids: ids });
    this._tick();
    this._sendActivation(m.seq_id, ids.length - 1, out.hidden);
  }
  // Stage 0 receives the just-sampled token (relayed by the coordinator) and runs the next
  // decode step, embedding that single token at the next position. This is the autoregressive
  // driver; it stops at max_tokens (the operator/coordinator also enforce the limit + done).
  async _onTokenForStage0(m) {
    if (!this.isFirst || !this.ctx) return;
    const st = this.seqState && this.seqState.get(m.seq_id);
    if (!st) return;
    st.generated += 1;
    if (m.done || st.generated >= st.max_tokens) return; // sequence finished; do not drive further
    const pos = st.pos;
    st.pos += 1;
    const out = await forward(this.ctx, { seq_id: m.seq_id, token_pos: pos, token_ids: [m.token_id] });
    this._tick();
    this._sendActivation(m.seq_id, pos, out.hidden);
  }
  async _onRecruit(m) { return this._onAssign({ ...m, model_id: this._modelId, is_first: false, is_last: false }); }
  _onEvict(m) { if (this.ctx) { freeStage(this.ctx); this.ctx = null; } this.onLog("evicted: " + m.reason); }

  _sendActivation(seq_id, token_pos, hidden) {
    const meta = { run: this.run, node: this.id, seq_id, token_pos,
      from_stage: this.stage, to_stage: this.stage + 1, dtype: "f16", shape: [hidden.length] };
    this._send(P.encodeActivation(meta, hidden));   // binary hot path
  }
  _send(obj) { try { this.ws.send(obj instanceof ArrayBuffer ? obj : JSON.stringify(obj)); } catch {} }
}

// Sum the bytes a stage actually loaded (handles single-buffer and multi-part layer objects).
function weightBytes(weights) {
  let n = 0;
  for (const l of weights.layers || []) {
    if (l.buf) n += l.buf.byteLength;
    else if (l.parts) for (const p of l.parts) n += (p.buf?.byteLength || 0);
  }
  if (weights.embed) n += weights.embed.byteLength;
  if (weights.lm_head) n += weights.lm_head.byteLength;
  if (weights.norm) n += weights.norm.byteLength;
  return n;
}
