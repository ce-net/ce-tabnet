// ce-tabnet wire protocol — SINGLE SOURCE OF TRUTH for message shapes.
// Owner: Implementer 1. Imported read-only by every other module.
// Pure functions, no I/O, no imports. See docs/module-contract.md §B.
//
// Every message is a JSON object with a `t` (type) field. Constructors below build
// well-formed messages; `validate(msg)` checks required fields per type. The activation
// hot-path additionally supports a binary frame (encodeActivation/decodeActivation).

export const T = Object.freeze({
  // tab -> DO
  JOIN: "join", READY: "ready", HB: "hb", ACTIVATION: "activation",
  TOKEN: "token", SIGNAL: "signal", METRICS: "metrics", LEAVE: "leave",
  // operator -> DO
  CREATE_RUN: "create-run", PROMPT: "prompt", CANCEL: "cancel", SUBSCRIBE_TOKENS: "subscribe-tokens",
  // DO -> tab
  WELCOME: "welcome", ASSIGN_STAGE: "assign-stage", ROUTE_UPDATE: "route-update",
  PROMPT_BEGIN: "prompt-begin", RECRUIT: "recruit", EVICT: "evict",
  // DO -> operator
  RUN_STATE: "run-state", SEQ_STATUS: "seq-status", ERROR: "error",
  // mesh request/reply ack (coordinator -> caller). The DO replied implicitly over the WS; on the
  // mesh every request gets an explicit reply, so a generic ACK carries "accepted" + optional data.
  ACK: "ack",
});

// ---- tab -> DO ----
export const join = (run, node, caps) => ({ t: T.JOIN, run, node, role: "stage", caps });
export const ready = (run, node, stage, loaded_layers, weight_bytes) =>
  ({ t: T.READY, run, node, stage, loaded_layers, weight_bytes });
export const hb = (run, node) => ({ t: T.HB, run, node });
export const activation = (run, node, seq_id, token_pos, from_stage, to_stage, dtype, shape, data) =>
  ({ t: T.ACTIVATION, run, node, seq_id, token_pos, from_stage, to_stage, dtype, shape, data });
export const token = (run, node, seq_id, token_pos, token_id, text, done) =>
  ({ t: T.TOKEN, run, node, seq_id, token_pos, token_id, text, done });
export const signal = (run, node, to_node, kind, payload) =>
  ({ t: T.SIGNAL, run, node, to_node, kind, payload });
export const metrics = (run, node, stage, tokens, compute_ms, vram_used_mb) =>
  ({ t: T.METRICS, run, node, stage, tokens, compute_ms, vram_used_mb });
export const leave = (run, node) => ({ t: T.LEAVE, run, node });

// ---- operator -> DO ----
export const createRun = (run, model_id, opts = {}) => ({
  t: T.CREATE_RUN, run, model_id,
  stages: opts.stages ?? null, replicas: opts.replicas ?? 1,
  fastpath: !!opts.fastpath, microbatch: opts.microbatch ?? 8, cap: opts.cap ?? null,
});
export const prompt = (run, seq_id, body, opts = {}) => ({
  t: T.PROMPT, run, seq_id,
  prompt: typeof body === "string" ? body : undefined,
  token_ids: Array.isArray(body) ? body : undefined,
  max_tokens: opts.max_tokens ?? 256, temperature: opts.temperature ?? 0.7, top_p: opts.top_p ?? 0.95,
});
export const cancel = (run, seq_id) => ({ t: T.CANCEL, run, seq_id });
export const subscribeTokens = (run, seq_id = null) => ({ t: T.SUBSCRIBE_TOKENS, run, seq_id });

// ---- DO -> tab ----
export const welcome = (run, node, you_are, plan_version) =>
  ({ t: T.WELCOME, run, node, you_are, plan_version });
export const assignStage = (run, node, s) => ({
  t: T.ASSIGN_STAGE, run, node, stage: s.stage, layers: s.layers,
  is_first: s.is_first, is_last: s.is_last, prev_node: s.prev_node ?? null, next_node: s.next_node ?? null,
  model_id: s.model_id, manifest_ref: s.manifest_ref, replica_of: s.replica_of ?? null,
});
export const routeUpdate = (run, stage, prev_node, next_node) =>
  ({ t: T.ROUTE_UPDATE, run, stage, prev_node, next_node });
export const promptBegin = (run, seq_id, token_ids, max_tokens, temperature, top_p) =>
  ({ t: T.PROMPT_BEGIN, run, seq_id, token_ids, max_tokens, temperature, top_p });
export const recruit = (run, stage, layers, manifest_ref) =>
  ({ t: T.RECRUIT, run, stage, layers, manifest_ref });
export const evict = (run, node, reason) => ({ t: T.EVICT, run, node, reason });

// ---- DO -> operator ----
export const runState = (run, model_id, S, R, stages, ready_, tabs_needed, tabs_present) =>
  ({ t: T.RUN_STATE, run, model_id, S, R, stages, ready: ready_, tabs_needed, tabs_present });
export const seqStatus = (run, seq_id, status, detail = null) =>
  ({ t: T.SEQ_STATUS, run, seq_id, status, detail });
export const error = (run, code, message) => ({ t: T.ERROR, run, code, message });

// ---- mesh request/reply ack ----
// Generic "request accepted" reply for the mesh request/reply path (join/ready/hb/prompt/...).
// `data` carries any per-request extras (e.g. the assigned seq_id). The DO needed no such message
// because a WS frame is fire-and-forget; mesh request() always resolves with a reply, so this is it.
export const ack = (run, data = {}) => ({ t: T.ACK, run, ok: true, ...data });

// ---- validation ----
const REQUIRED = {
  [T.JOIN]: ["run", "node", "caps"],
  [T.READY]: ["run", "node", "stage", "loaded_layers"],
  [T.ACTIVATION]: ["run", "seq_id", "token_pos", "from_stage", "to_stage"],
  [T.TOKEN]: ["run", "seq_id", "token_pos"],
  [T.CREATE_RUN]: ["run", "model_id"],
  [T.PROMPT]: ["run", "seq_id"],
  [T.ASSIGN_STAGE]: ["run", "node", "stage", "layers", "model_id", "manifest_ref"],
};
export function validate(msg) {
  if (!msg || typeof msg !== "object" || typeof msg.t !== "string") return false;
  const req = REQUIRED[msg.t];
  if (!req) return true; // unknown/forward-compat types pass through
  return req.every((k) => msg[k] !== undefined && msg[k] !== null);
}

// ---- activation binary framing (hot path) ----
// Frame = 4-byte little-endian header length, then UTF-8 JSON header, then raw payload bytes.
// Header carries everything except `data`; payload is the activation tensor as the declared dtype.
export function encodeActivation(meta, payload /* ArrayBuffer | TypedArray */) {
  const header = JSON.stringify({ ...meta, data: undefined });
  const hbytes = new TextEncoder().encode(header);
  const pbuf = payload instanceof ArrayBuffer ? payload : payload.buffer;
  const out = new Uint8Array(4 + hbytes.length + pbuf.byteLength);
  new DataView(out.buffer).setUint32(0, hbytes.length, true);
  out.set(hbytes, 4);
  out.set(new Uint8Array(pbuf), 4 + hbytes.length);
  return out.buffer;
}
export function decodeActivation(frame /* ArrayBuffer */) {
  const dv = new DataView(frame);
  const hlen = dv.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(frame, 4, hlen)));
  const payload = frame.slice(4 + hlen);
  return { meta: header, payload };
}
