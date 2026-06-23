# ce-tabnet — Module Contract

This is the build spec for **6 parallel implementers**. It defines (1) the wire protocol, (2) the Durable
Object API, and (3) a per-file ownership list with explicit interfaces, so the six can build with **zero
collisions**. Read `architecture.md` first.

The single source of truth for message shapes is **`web/protocol.js`** — every other module imports its
constructors/validators rather than hand-rolling JSON. Owner of `protocol.js` ships it first (it has no
dependencies); the other five import it.

---

## A. Roles in a Run

- **operator** — defines a Run, sends prompts, receives token streams (`orchestrate.html`).
- **stage tab** — a browser tab that holds one pipeline stage and computes its layers (`join.html`).
- **Run DO** — the Cloudflare Durable Object that owns topology, assignment, relay, healing.

All three speak the **same JSON message envelope** over WebSocket to the DO.

---

## B. Wire protocol (JSON over WebSocket)

Envelope: every message is a JSON object with a `t` (type) field. Unknown `t` is ignored (forward-compat).
IDs: `run` = run id (string), `node` = tab node id (hex, from `localStorage.ce_node_id`, same as `node.html`),
`seq_id` = per-request sequence id, `token_pos` = position within that sequence.

### B.1 Tab → DO

| `t` | Fields | Meaning |
|---|---|---|
| `join` | `run`, `node`, `role:"stage"`, `caps:{cores,ram_gb,storage_gb,gpu,webgpu,vram_mb,cpu_mark,platform}` | Tab joins a Run as a candidate stage. `caps` shape is **identical to `node.html` detect()**. |
| `ready` | `run`, `node`, `stage`, `loaded_layers:[lo,hi)`, `weight_bytes` | Tab finished loading its assigned stage's weights into WebGPU and can compute. |
| `hb` | `run`, `node` | Heartbeat (every 10 s, same cadence as `node.html`). |
| `activation` | `run`, `node`, `seq_id`, `token_pos`, `from_stage`, `to_stage`, `dtype`, `shape:[...]`, `data` (base64 or binary frame) | Hidden-state `h` produced by `from_stage`, destined for `to_stage`. The DO relays it (Path A) to the holder of `to_stage`. For Path B this is sent peer-to-peer and NOT to the DO. |
| `token` | `run`, `node`, `seq_id`, `token_pos`, `token_id`, `text`, `done:bool` | Last stage's sampled token; DO fans it to the operator's stream AND to stage 0 for the next decode step. |
| `signal` | `run`, `node`, `to_node`, `kind:"offer"|"answer"|"ice"`, `payload` | WebRTC signaling relayed by the DO to `to_node` (Path B only). |
| `metrics` | `run`, `node`, `stage`, `tokens`, `compute_ms`, `vram_used_mb` | Periodic per-stage telemetry for the console. |
| `leave` | `run`, `node` | Graceful departure (lets the DO reroute proactively). |

### B.2 Operator → DO

| `t` | Fields | Meaning |
|---|---|---|
| `create-run` | `run`, `model_id`, `stages` (requested S, optional), `replicas` (R, default 1), `fastpath:bool`, `microbatch:int`, `cap?` | Create/define a Run for a model from `model-config.js`. DO computes the stage plan once enough tabs join. |
| `prompt` | `run`, `seq_id`, `prompt` (string) or `token_ids:[...]`, `max_tokens`, `temperature`, `top_p` | Submit a generation request. |
| `cancel` | `run`, `seq_id` | Cancel an in-flight generation. |
| `subscribe-tokens` | `run`, `seq_id?` | Operator stream wants token + status events (all seqs if `seq_id` omitted). |

### B.3 DO → Tab

| `t` | Fields | Meaning |
|---|---|---|
| `welcome` | `run`, `node`, `you_are:"stage"|"spare"|"replica"`, `plan_version` | Ack join; tells the tab its provisional role. |
| `assign-stage` | `run`, `node`, `stage`, `layers:[lo,hi)`, `is_first:bool`, `is_last:bool`, `prev_node`, `next_node`, `model_id`, `manifest_ref`, `replica_of?` | Authoritative assignment. Tab fetches `layers` from the model pack and loads them. `prev_node`/`next_node` are the activation neighbors (or null for first/last). |
| `route-update` | `run`, `stage`, `prev_node`, `next_node` | Topology changed (a neighbor was rerouted); update where you send/expect activations. |
| `activation` | (same fields as B.1 `activation`) | Relayed inbound activation for the stage this tab holds (Path A). |
| `token` | (same as B.1 `token`) | For stage 0: the previous token to begin the next decode step. |
| `prompt-begin` | `run`, `seq_id`, `token_ids`, `max_tokens`, `temperature`, `top_p` | Sent to **stage 0** to start a new sequence (prefill). |
| `signal` | `run`, `from_node`, `kind`, `payload` | WebRTC signaling from a peer (Path B). |
| `recruit` | `run`, `stage`, `layers:[lo,hi)`, `manifest_ref` | Ask a spare tab to take over an orphaned stage (healing). |
| `evict` | `run`, `node`, `reason` | DO tells a tab to drop its stage (e.g. demotion after a duplicate-stage merge). |

### B.4 DO → Operator

| `t` | Fields | Meaning |
|---|---|---|
| `run-state` | `run`, `model_id`, `S`, `R`, `stages:[{stage,layers,holders:[node],status}]`, `ready:bool`, `tabs_needed`, `tabs_present` | Full topology snapshot for the console; pushed on every change. |
| `token` | `run`, `seq_id`, `token_pos`, `token_id`, `text`, `done` | Streamed generated token for display. |
| `seq-status` | `run`, `seq_id`, `status:"queued"|"prefill"|"decoding"|"done"|"error"|"cancelled"`, `detail?` | Generation lifecycle. |
| `error` | `run`, `code`, `message` | Any error (no capacity, bad model, etc.). |

### B.5 Activation framing note

Activations are the only large/hot messages. Implementers MUST support **binary WebSocket frames** for
`activation` (a small JSON header line + a raw `ArrayBuffer` payload, or an agreed binary layout) to avoid
base64 overhead on the hot path. `protocol.js` provides `encodeActivation(meta, float16Array)` /
`decodeActivation(frame)` so the framing detail is owned in one place. JSON base64 is the correctness
fallback for the demo; binary is required before large Runs.

---

## C. Durable Object API (the Run DO)

The Worker (`worker.js`) routes:
- `GET /run/:id/ws?role=stage|operator` → upgrade to WebSocket, forward to DO instance `idFromName(id)`.
- `POST /run/:id` (JSON `create-run`) → idempotent create.
- `GET /run/:id/state` → JSON `run-state` snapshot (for polling clients/health).

The Run DO (`run-do.js`) class surface (methods the implementer must provide):

```
class Run {                       // one instance per run id
  fetch(request)                  // handles ws upgrade + POST create/state
  // --- WebSocket lifecycle ---
  onTabConnect(ws, node)          // register socket
  onMessage(ws, msg)              // dispatch by msg.t (the B.1/B.2 tables)
  onClose(ws)                     // mark node gone -> heal()

  // --- core logic ---
  createRun({model_id, stages, replicas, fastpath, microbatch, cap})
  planStages()                    // memory-weighted PlacementPlanner: assign [lo,hi) per tab from caps
  assign(node)                    // give a joining tab a stage or spare/replica role -> assign-stage
  relayActivation(msg)            // Path A: forward activation to holder(next stage)
  relaySignal(msg)                // Path B: forward WebRTC offer/answer/ice
  routeToken(msg)                 // fan sampled token to operator stream + stage 0
  startSequence(seq)              // prompt -> prompt-begin to stage 0; track seq state
  heal(deadNode)                  // promote replica OR mark stage recruiting + recruit() a spare
  heartbeatSweep()                // alarm(): evict nodes stale > 35s, trigger heal
  snapshot()                      // build run-state for operators
}
```

DO storage (persisted, so a Run is resumable): `model_id`, `S`, `R`, `fastpath`, `microbatch`, the stage plan
(`stage -> {layers, holders}`), and `plan_version`. Use a DO **alarm** (every ~5 s) for `heartbeatSweep()`.

**Placement algorithm (`planStages`) — exact rule for the implementer:**
1. Collect live candidate tabs with their `caps.vram_mb` (fallback `ram_gb*1024*0.5` if no WebGPU).
2. Target stage count `S` = operator's `stages` if given and feasible; else
   `ceil(model_weight_bytes / median(tab_budget))`.
3. Distribute the model's `n_layers` across `S` stages **proportional to each holder's budget** (largest tab
   gets the most layers), so every stage's weight bytes ≤ its holder's budget. Record exact `[lo,hi)`.
4. Assign embed to stage 0's holder, lm_head+final norm to stage S-1's holder; pick the two beefiest tabs for
   those if possible.
5. For `R>1`, duplicate each stage onto the next-beefiest free tabs as replicas.
6. Emit `assign-stage` to each holder; emit `run-state` to operators. A Run is `ready` when every stage has
   ≥1 holder reporting `ready`.

---

## D. File ownership — 6 implementers, no collisions

Each implementer owns a disjoint set of files and a clear interface boundary. The only shared file is
`protocol.js` (owned by Implementer 1, consumed read-only by others). All web files are **vanilla JS modules,
no build step**; coordinator files are Cloudflare Worker/DO JS.

### Implementer 1 — Protocol + Model config (foundation, ship first)
**Owns:** `web/protocol.js`, `web/model-config.js`
- `protocol.js`: message constructors (`join()`, `ready()`, `activation()`, `token()`, `assignStage()`, …),
  a `validate(msg)` per type, and `encodeActivation`/`decodeActivation` binary framing. Pure functions, no
  I/O. **No imports.**
- `model-config.js`: the model registry. Exports `MODELS = { "<model_id>": {n_layers, hidden_dim, n_heads,
  n_kv_heads, vocab, rope_theta, arch:"llama", quant, baseUrl, manifestRef, approx_weight_bytes} }` and a
  helper `defaultStagePlan(model_id, tabBudgets[]) -> [{stage, layers:[lo,hi)}]` used by both the DO planner
  (imported server-side) and the console (for "tabs needed" display). Ships with one **small real model**
  entry for the demo (e.g. `tinyllama-1.1b-q4`).
**Interface guarantee:** message field names match §B exactly; do not rename without updating this doc.

### Implementer 2 — Run Durable Object (coordination core)
**Owns:** `coordinator/run-do.js`
- Implements the `Run` class in §C: assignment, `planStages` (calls `model-config.defaultStagePlan`),
  Path-A activation relay, token fan-out, WebRTC signaling relay, heartbeat alarm, healing.
- Imports `protocol.js` (validators) and `model-config.js` (model dims + plan) — both copied/symlinked into the
  Worker bundle (wrangler bundles them).
**Interface guarantee:** honors every DO→Tab / DO→Operator message in §B; persists the plan to DO storage.

### Implementer 3 — Worker entry + wrangler config + deploy docs
**Owns:** `coordinator/worker.js`, `coordinator/wrangler.toml`, deploy section of `README.md`
- `worker.js`: route `/run/:id/ws`, `/run/:id` (POST), `/run/:id/state` to the DO; CORS; nothing stateful.
- `wrangler.toml`: DO binding (`[[durable_objects.bindings]] name="RUN" class_name="Run"`), migrations, routes,
  account placeholders. Deploy-ready; **do not deploy** (needs user creds). Document the one command.
**Interface guarantee:** DO binding name `RUN`, class `Run`; `idFromName(runId)` addressing.

### Implementer 4 — Tab runtime + join page (lifecycle + healing client)
**Owns:** `web/tabnet-node.js`, `web/join.html`
- `join.html`: reuse `node.html`'s look + the capability detection functions (`gpuName`, `cpuBench`,
  `vramMb`, `detect`) — copy them in (no build). UI: status dot, assigned stage, layers loaded, tokens served.
- `tabnet-node.js`: open WS to `/run/:id/ws?role=stage`, send `join` with caps, handle `assign-stage` (call
  `shard-loader.load(layers)` then `inference.initStage(...)`, send `ready`), handle inbound `activation`
  (call `inference.forward(...)` → send `activation` to next or `token` if last), handle `prompt-begin` (stage
  0 prefill), heartbeats, `recruit`/`route-update` healing, optional WebRTC (Path B) via `signal`.
**Interface guarantee:** depends on `inference.js` (Impl 5) and `shard-loader.js` (Impl 6) via the interfaces
below; depends on `protocol.js`.

### Implementer 5 — WebGPU inference (the compute)
**Owns:** `web/inference.js`
- Exports:
  - `async initStage({arch, dims, layers:[lo,hi), is_first, is_last, weights /*from shard-loader*/}) -> StageCtx`
  - `async forward(ctx, {seq_id, token_pos, hidden /*Float16Array|null for stage0*/, token_ids /*for stage0 prefill*/}) -> {hidden /*to next stage*/ | token_id /*if is_last*/}`
  - `freeStage(ctx)`
- WGSL kernels: RMSNorm, quantized matmul (Q4/int8 dequant), RoPE, attention with local KV cache, SwiGLU MLP,
  sampling (temp/top-p) on the last stage; `embed_tokens` on stage 0.
- **WASM/CPU fallback** path when `!navigator.gpu` (same outputs, slower).
**Interface guarantee:** `forward` consumes/produces the activation tensor `protocol.js` frames; KV cache is
internal to `StageCtx` and never leaves the tab.

### Implementer 6 — Shard loader + model pack format + orchestrator UI
**Owns:** `web/shard-loader.js`, `web/orchestrate.html`, `models/README.md`
- `shard-loader.js`: `async load({baseUrl, manifestRef, layers:[lo,hi), needEmbed, needLmHead}) -> weights`
  — fetch the manifest, HTTP **range-fetch** each required object, verify hash (CID), return buffers ready for
  `inference.initStage`. LRU cache in `caches`/IndexedDB so re-fetch on reload is cheap.
- `orchestrate.html`: operator console — pick a model (`model-config.MODELS`), show **tabs-needed math**,
  `create-run`, render the live `run-state` pipeline (stages, holders, status, healing animation), a prompt
  box that sends `prompt` and renders streamed `token`s. Reuse `node.html` styling (copy the CSS block).
- `models/README.md`: how to produce a tabnet model pack (quantize per-layer, content-address, write
  `manifest.json`, host with range support) — the recipe `scaling-to-300b.md` §5 references.
**Interface guarantee:** the `weights` object shape it returns is exactly what `inference.initStage` expects;
agree that shape in a 10-line comment block at the top of both `shard-loader.js` and `inference.js`.

### Collision-avoidance summary

| File | Owner | Imports (read-only) |
|---|---|---|
| `web/protocol.js` | Impl 1 | — |
| `web/model-config.js` | Impl 1 | — |
| `coordinator/run-do.js` | Impl 2 | protocol, model-config |
| `coordinator/worker.js` | Impl 3 | — (routes to DO) |
| `coordinator/wrangler.toml` | Impl 3 | — |
| `web/tabnet-node.js` | Impl 4 | protocol, inference, shard-loader |
| `web/join.html` | Impl 4 | tabnet-node |
| `web/inference.js` | Impl 5 | protocol (framing only) |
| `web/shard-loader.js` | Impl 6 | model-config |
| `web/orchestrate.html` | Impl 6 | protocol, model-config |
| `models/README.md` | Impl 6 | — |

The two cross-team interfaces that need a handshake (write them as shared comment blocks before coding):
1. **weights object** (`shard-loader.load` ↔ `inference.initStage`) — Impl 5 ↔ Impl 6.
2. **activation tensor framing** (`protocol.encodeActivation/decodeActivation`) — Impl 1 defines, Impl 4/5 use.

---

## E. Milestones (build order)

1. **M0 — protocol + config** (Impl 1): `protocol.js`, `model-config.js` with the small demo model.
2. **M1 — coordinator skeleton** (Impl 2 + 3): DO accepts joins, does fixed-equal stage assignment, relays
   activations and tokens over WS; Worker routes; `wrangler dev` runs locally.
3. **M2 — single-stage end-to-end** (Impl 4 + 5 + 6): one tab loads the whole small model (S=1), prefill +
   decode + sampling, tokens stream to the console. Correctness baseline vs. reference.
4. **M3 — real pipeline** (all): S=3 across 3 tabs; activations hop stage→stage via the DO; healing on tab
   close (recruit + range-refetch).
5. **M4 — throughput** (Impl 2 + 4): micro-batching window, concurrent prompts, metrics in the console.
6. **M5 — scale + fast path** (Impl 2 + 4 + 6): memory-weighted placement, WebRTC Path B, int8 activations,
   bigger model packs (7B → 70B → 300B-class) per `scaling-to-300b.md`.
