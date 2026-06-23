# ce-tabnet — Architecture

**One large LLM, sharded by layer across many ordinary browser tabs, coordinated so it is dead simple:
open a URL, lend a tab.**

This document is the authoritative design. It is technically honest: it states the memory and bandwidth
arithmetic up front, picks the only distributed strategy that works over a wide-area browser network
(pipeline parallelism), and shows exactly how the framework scales from a 3-tab demo to a 300B model on
~100–300 tabs.

---

## 0. Why pipeline parallelism, and nothing else

There are three ways to split a transformer across machines:

| Strategy | What crosses the network per token | Verdict for tabs |
|---|---|---|
| **Tensor parallel** (split each matmul) | An **all-reduce of full activations every layer** — dozens of synchronous round-trips per token | **Dead on arrival.** Browser-to-browser RTT is 10–150 ms; a 96-layer model would pay that latency dozens of times per token. The network becomes slower than a single CPU. This is the explicit warning in `PLAN/09-hospital-inference.md`. |
| **Data parallel** (full model per node) | Nothing (each node is independent) | Impossible here — no single tab holds the whole model. |
| **Pipeline parallel** (contiguous layer ranges per node) | **One hidden-state activation vector per token per stage boundary (~KB)** | **The answer.** Bandwidth is tiny, hops are sequential not synchronous, and micro-batching hides the per-hop latency behind throughput. |

ce-tabnet is **pipeline-parallel only**, exactly as the CE infer design mandates. Each tab owns a
contiguous block of transformer layers. The only thing that travels between tabs is the **boundary hidden
state** — a `[hidden_dim]` vector per token (a few KB at fp16/int8).

---

## 1. The big picture (one Run)

```
                              ┌─────────────────────────────────────────────┐
   operator (orchestrate.html)│        Cloudflare Durable Object "Run"        │
   defines a Run, sends prompt│  - live topology: stage -> tab(s)             │
   ───────────────────────────▶  - stage assignment on join                  │
                              │  - WebRTC signaling (fast path)               │
                              │  - WebSocket activation relay (fallback path) │
                              │  - health / heartbeat / reroute               │
                              └───────▲───────────────▲───────────────▲───────┘
                                      │ join/assign   │ signal/relay   │
                  ┌───────────────────┘               │               └───────────────────┐
                  │                                    │                                   │
          ┌───────┴───────┐                   ┌────────┴───────┐                  ┌────────┴───────┐
          │  TAB  (stage 0)│  activation ───▶ │  TAB (stage 1) │  activation ──▶  │ TAB (stage S-1)│
          │  embed + L0..k │  (WebRTC or WS)  │  L(k+1)..m     │                  │ L..N + lm_head │
          │  WebGPU weights│ ◀── token ───────┤  WebGPU weights│ ◀── token ────── │ samples token  │
          └────────────────┘   (back-channel) └────────────────┘                  └───────┬────────┘
                  ▲                                                                        │
                  └──────────────────────── token streamed back to operator ──────────────┘
```

A **Run** is one model instance: a fixed pipeline plan of `S` stages. Tabs join the Run, get assigned a
stage, load that stage's weights, and start computing. The operator (or any client with the Run id) sends
prompts and receives streamed tokens.

---

## 2. Pipeline parallelism across tabs (the core mechanism)

### 2.1 Stages = contiguous layer ranges

A model with `N` transformer layers is split into `S` stages. Stage `s` owns layers `[lo_s, hi_s)`.
Stage 0 additionally owns the **token embedding** + the first layers. The last stage owns the final layers
plus the **final norm + LM head** (the logits projection). The split is **memory-weighted**: a beefy laptop
gets more layers than a phone (see §5 capability-aware placement). This is the EXO/Petals scheme.

Example: a 32-layer model across 4 tabs of equal size → stages own layers `[0,8) [8,16) [16,24) [24,32)`.
Stage 0 also holds `embed_tokens`; stage 3 also holds `norm` + `lm_head`.

### 2.2 The forward pass for one token

1. **Stage 0** receives the prompt token ids. It embeds them and runs layers `[lo_0, hi_0)` on WebGPU,
   producing a hidden-state tensor `h` of shape `[seq, hidden_dim]` (for decode, `seq=1`: just `[hidden_dim]`).
2. Stage 0 sends `h` to **stage 1** over the activation channel (WebRTC datachannel, or DO WebSocket relay).
3. Each middle stage receives `h`, runs its layers, sends the new `h` onward.
4. **Stage S-1** runs its layers, applies the final norm + LM head → `logits`, **samples** the next token,
   and sends the **sampled token id** back to the operator (and to stage 0 for the next decode step).

Only `h` (one `[hidden_dim]` vector per token in decode) ever crosses the network between stages. That is
the whole point: **~KB/token of cross-network traffic**, regardless of model size.

### 2.3 KV cache lives on the stage that owns the layers

Each stage keeps the **KV cache for its own layers, locally in its own WebGPU buffers**. The KV cache never
crosses the network — it is purely local state of the layers a tab owns. This is essential: it means the
per-token network cost stays at one activation vector even as the context grows. A stage's memory budget
must therefore include both its weights and its KV cache (budgeted in §4).

### 2.4 Micro-batching = throughput (the "high throughput" claim)

A naive pipeline computes one token at a time: while stage 1 works, stages 0, 2, 3 sit idle. Utilization is
`1/S`. That is unacceptable.

ce-tabnet keeps **multiple tokens/requests in flight**, classic GPipe-style micro-batching:

```
time ─▶
stage0:  t0  t1  t2  t3  t4  ...
stage1:      t0  t1  t2  t3  ...
stage2:          t0  t1  t2  ...
stage3:              t0  t1  ...      ← every stage busy once the pipe is full
```

Sources of in-flight work:
- **Multiple concurrent requests** (different users / different prompts) — independent, trivially pipelined.
- **Decode-stage interleaving within one request:** as soon as stage 0 hands token *t* to stage 1, it can
  start token *t+1* of the **same** sequence — but only after it knows token *t* (autoregression). So
  within a *single* sequence the pipeline cannot be fully filled by decode alone. Throughput for a single
  stream is therefore bounded; **aggregate throughput across many concurrent streams is where tabnet shines**
  (and is exactly the load profile of "many people sharing the demo"). This is stated honestly: a single
  user sees latency `≈ S × per-stage-time + (S-1) × hop-latency`; the *system* sees high tokens/sec across
  users.
- **Prefill is naturally batched:** the prompt's `seq` tokens all flow through as one tensor, so prefill is
  one big micro-batch per stage and pipelines perfectly.

The pipeline depth in flight is capped by `MICROBATCH_WINDOW` (per Run, default 8) to bound memory.

### 2.5 Back-channel for the sampled token

Stage S-1 samples the token. Two things need it: the **operator** (to display) and **stage 0** (to start the
next decode step). The token id is tiny (one integer), so it is always sent via the DO WebSocket (the slow
but universal path); the latency-sensitive bulk (`h`) uses the fast path. The DO fans the token out to the
operator's stream and to stage 0.

---

## 3. The activation channel: WebRTC vs WebSocket relay (and which we pick)

Two ways to move `h` from stage `s` to stage `s+1`:

### Path A — DO WebSocket relay (always available, simplest)
Every tab holds **one** WebSocket to the Run DO. Stage `s` sends `{type:"activation", run, seq_id, token_pos, to_stage:s+1, payload}` to the DO; the DO forwards it to the tab currently owning stage `s+1`. The DO is a
star hub.
- **Pros:** zero NAT problems (tabs only ever talk to Cloudflare), trivial to implement, works on every
  device/network including locked-down phones, and the DO already knows the topology so routing is a map
  lookup. Reroute on tab death is instant (just point the next hop at the replica).
- **Cons:** every activation makes **two** WAN trips (tab→CF→tab) and passes through the DO (which is
  single-threaded per Run). Activation payloads are KB-sized, so DO CPU is not the bottleneck, but each hop
  pays Cloudflare round-trip latency.

### Path B — WebRTC datachannel, DO does signaling only
The DO matches adjacent stages and brokers an SDP/ICE handshake; thereafter stage `s` sends `h` **directly**
to stage `s+1` over a peer-to-peer datachannel (one WAN trip, possibly LAN-direct between two laptops at
home).
- **Pros:** lowest latency, no per-activation load on Cloudflare, can be LAN-direct.
- **Cons:** NAT traversal (needs STUN, sometimes TURN), more moving parts, datachannel setup time, and a
  small fraction of restrictive networks fail to connect and must fall back anyway.

### Decision

**Ship Path A (WebSocket relay) as the default and the only path required for the demo and for correctness.**
It is the simplest thing that works everywhere, and for KB-sized activations the dominant cost is RTT, which
both paths pay at least once. **Path B (WebRTC) is an opt-in fast-path optimization** the DO can negotiate
*between adjacent stages only*, transparently falling back to Path A on any failure. The wire protocol
(see `module-contract.md`) is identical from the application's view; only the transport differs. Implement A
first, B behind a `fastpath: true` Run flag.

> Note for the home-laptops scenario in the tagline: when several of your old laptops are on the *same LAN*,
> WebRTC can connect them directly over the LAN, so adjacent-stage hops never leave the house. That is a real
> win there, but it is an optimization on top of a correct WebSocket baseline.

---

## 4. The math: tabs needed vs model size

This is the honest core. Three quantities decide everything.

### 4.1 Weight memory per tab

```
total_weight_bytes ≈ params × bits_per_param / 8
per_tab_weight_budget = usable_webgpu_mem − kv_cache − activations − runtime_overhead
tabs_needed (for weights) ≈ ceil(total_weight_bytes / per_tab_weight_budget)
```

**Per-tab budget, realistically:** a browser tab with WebGPU can address roughly **0.5–2 GB** of GPU buffers
for weights. Limits that bite:
- `maxStorageBufferBindingSize` and `maxBufferSize` (often 128 MiB–2 GiB depending on device/browser) cap a
  *single* buffer; weights must be **chunked into many buffers** to exceed a single binding limit.
- Total device VRAM shared with the OS/compositor.
- Mobile devices are at the **0.5 GB** end; gaming laptops reach the **2 GB** end.

We budget **1.0 GB of weights per tab** as the planning default (`PER_TAB_WEIGHT_GB`), leaving headroom for
KV cache + activations + the WASM/JS heap.

### 4.2 KV cache per tab

```
kv_bytes_per_layer ≈ 2 (K and V) × seq_len × n_kv_heads × head_dim × bytes_per_elt
kv_per_tab ≈ kv_bytes_per_layer × layers_on_this_tab
```
For a 4096-context, 4096-hidden, GQA model at fp16, KV is on the order of tens of MB per layer-block — small
relative to weights, but it grows with context and must be counted. tabnet caps context per Run and counts KV
in each stage's budget.

### 4.3 Activation bandwidth per token

```
activation_bytes_per_hop ≈ hidden_dim × bytes_per_elt        (decode, seq=1)
                         ≈ 4096 × 2  ≈ 8 KB   at fp16
per_token_network_bytes ≈ activation_bytes_per_hop × (S − 1)  hops
```
At 8 KB/hop and, say, 100 stages, a single token moves ~800 KB across the whole pipeline — **and crucially
each individual hop is only 8 KB.** Even at 30 tokens/sec aggregate, no hop needs more than a few hundred
KB/s. **Bandwidth is never the bottleneck; latency-per-hop is**, which micro-batching hides for aggregate
throughput.

We further allow **activation quantization to int8** (4 KB/hop) for the cross-network payload, decompressed
on arrival, when the fast path is bandwidth-constrained.

### 4.4 Worked example — 300B at 4-bit

```
params            = 300 × 10^9
bits_per_param    = 4         (4-bit quant, e.g. Q4)
total_weight_bytes= 300e9 × 4 / 8 = 150 × 10^9 bytes = 150 GB

PER_TAB_WEIGHT_GB = 1.0 GB    (conservative browser budget)
tabs_needed       = ceil(150 GB / 1.0 GB) = 150 tabs

with mixed devices (phones at 0.5 GB, laptops at 2 GB), the count lands in the 100–300 range,
which is why we say "~100–300 tabs" for 300B@4-bit.
```

Other points on the curve (at `PER_TAB_WEIGHT_GB = 1.0`):

| Model | Params | 4-bit weights | Tabs @1GB | Tabs @2GB |
|---|---|---|---|---|
| TinyLlama-class demo | 1.1B | ~0.6 GB | 1 | 1 |
| 7B | 7B | ~3.5 GB | 4 | 2 |
| 13B | 13B | ~6.5 GB | 7 | 4 |
| 70B | 70B | ~35 GB | 35 | 18 |
| **300B** | **300B** | **~150 GB** | **150** | **75** |

**This table is the contract.** The framework must let an operator declare any of these `S` values and have
the system assign that many stages. The demo runs at the top row (1–4 tabs, a small real model); the 300B row
is the same code path with more tabs.

### 4.5 Latency budget (honest)

Single-stream decode latency ≈ `S × (per_stage_compute + hop_latency)`. With 150 stages and even 5 ms/hop
that is ~0.75 s of pure hop latency per token for one stream — slow for a *single* user. This is the
fundamental pipeline-depth tax and we do not hide it. The product answer is **throughput, not single-stream
latency**: 150 stages running 8 micro-batches deep serve many concurrent users at high aggregate tokens/sec.
For low single-stream latency you want *fewer, fatter* stages (fewer tabs, each with more VRAM), which is the
natural knob: tabnet always uses the **fewest stages that fit**, given the available per-tab budgets.

---

## 5. Capability-aware placement (reuse node.html detection)

Every joining tab runs the **same capability detection as `web/site/node.html`**: `cores`, `ram_gb`,
`storage_gb`, WebGPU adapter limits (`vram_mb` from `maxBufferSize`/`maxStorageBufferBindingSize`),
`cpu_mark` micro-benchmark, GPU renderer string. It reports these in its `join` message.

The DO's `PlacementPlanner` uses them:
- **Memory-weighted stage sizing:** assign more layers to tabs with larger `vram_mb`. A 2 GB laptop may hold
  16 layers while a 0.5 GB phone holds 4. Stages are therefore *not* equal-sized; the plan records each
  stage's exact `[lo, hi)` so weight chunks are fetched accordingly.
- **Order by latency where known:** if CE latency-graph data is available for these nodes, place
  network-adjacent (low-RTT) tabs as pipeline neighbors to minimize hop latency. Absent that, order is
  assignment order.
- **Reject under-budget devices** from large stages; a phone that cannot hold even the smallest stage is
  offered a **replica** role (redundancy) or politely declined.

This is the in-browser analogue of `ce-infer-core::probe()` self-tiering and the `swarm select_hosts()`
ranking pattern. Implementers MUST reuse the exact detection functions from `node.html`
(`gpuName`, `cpuBench`, `vramMb`, `detect`) — copied into `shard-loader.js`/`tabnet-node.js` so the pages
stay build-free.

---

## 6. Redundancy and healing (Petals-style)

Tabs are ephemeral — people close them. The Run survives:

- **Replication factor `R` per stage.** A stage may be held by `R ≥ 1` tabs. The DO routes each activation to
  the *primary* holder and keeps warm replicas. Default `R=1` for the demo, `R=2+` for serious Runs.
- **Heartbeats** (reusing the `node.html` 10 s `hb` cadence). A tab missing `STALE` (35 s, same as the hub)
  is declared dead.
- **Reroute on death:**
  1. If a **replica** of the dead stage exists, the DO promotes it (just repoints the previous/next hop) —
     near-instant, in-flight micro-batches for that stage are replayed from the last stage boundary.
  2. If **no replica**, the stage is **orphaned**: the DO marks it `recruiting`, and the **next tab to join**
     (or an existing spare) is assigned that exact `[lo, hi)` range and **re-fetches just that layer range**
     from the content-addressed weight store (the layers are CE blobs; any tab can fetch any range). Once the
     new tab reports `ready`, routing resumes.
  3. In-flight requests that crossed the dead stage are **retried from the pipeline head** (the operator
     holds the prompt; partial decodes are restartable because only committed tokens are durable). This is
     the Petals failure model: lose a stage, re-route or re-recruit, replay.
- **Idempotent activations:** every activation carries `(seq_id, token_pos, stage)` so a replayed or
  duplicated activation after a reroute is detected and not double-counted.

Because weights are content-addressed and chunked per layer range, **healing a stage is just an HTTP range
fetch** — no central coordinator ships GB of weights. That is the property that makes the system scale and
self-repair.

---

## 7. Weight distribution: content-addressed chunks over HTTP range

A tabnet **model pack** (see `model-config.js` and `models/README.md`) is the model's weights laid out so any
**layer range** can be fetched independently:

- Weights are quantized (e.g. Q4) and serialized **per layer** (or per small layer group), plus the
  embedding and LM head as their own objects.
- Each object is **content-addressed** (a CID / SHA-256). The manifest maps `layer_index -> {cid, byte_range,
  shape, quant}` and records `hidden_dim`, `n_layers`, `n_heads`, `vocab`, etc.
- A stage owning `[lo, hi)` fetches exactly those layer objects. Fetching uses **HTTP range requests** against
  a CDN/R2/blob endpoint, so a stage downloads only its slice and the browser/CDN cache makes re-fetches and
  replica fan-out cheap.
- **In CE terms:** the model pack is a CE blob set; CIDs *are* integrity (verify-on-arrival); peers/CDN that
  already hold a chunk serve it BitTorrent-style. The HTTP-range endpoint can be R2 (for the public demo) or a
  CE blob gateway (for the meshy version) — `shard-loader.js` only needs `{baseUrl, manifest}`.

Loading sequence per tab: resolve assigned `[lo, hi)` → for each layer object, `fetch(baseUrl+cid, {headers:
{Range}})` → verify hash → upload to WebGPU buffers (chunked under `maxBufferSize`) → report `ready`.

---

## 8. Inference compute in the tab (WebGPU + WASM fallback)

`inference.js` implements **one transformer stage** on WebGPU:
- Standard decoder block: RMSNorm → QKV proj → RoPE → attention (with local KV cache) → output proj →
  RMSNorm → SwiGLU MLP → residual. (Llama-family layout; `model-config.js` declares the exact variant.)
- **Quantized matmul** WGSL kernels (dequantize-on-the-fly for Q4/int8 weights) — the heavy lifting.
- **KV cache** in persistent WebGPU buffers, appended each decode step.
- Stage 0 also runs `embed_tokens`; stage S-1 also runs final norm + `lm_head` + **sampling** (temperature /
  top-p, done on GPU or a tiny CPU readback of logits).

**Fallback:** if `navigator.gpu` is absent (Path: older devices, locked-down browsers), `inference.js` falls
back to a **WASM/CPU** kernel path (the same math, much slower). The fallback keeps a phone *able to
participate* as a small stage even without WebGPU, honoring "any old laptop." The capability report flags
`webgpu:false` so the planner gives such tabs the smallest stages or replica duty.

This is real compute, not a stub: the module contract specifies the exact kernel set each implementer owns.

---

## 9. Coordination: the Cloudflare Durable Object "Run"

A **Durable Object instance == one Run** (one model instance). Chosen because:
- Durable Objects give a **single-threaded, consistent, in-memory authority** per Run — perfect for holding
  live topology and doing serialized stage assignment without races.
- They hold **many WebSockets** (one per tab + operator streams) and can fan messages out — exactly the
  hub/relay shape, but serverless and globally addressable, **no local cargo, scales to many tabs.**
- They survive brief disconnects and can persist the pipeline plan to DO storage so a Run is resumable.

The DO is the **only** stateful coordinator. The Worker (`worker.js`) is a thin router: it maps a Run id to
its DO and upgrades WebSockets. The DO owns: the topology map (stage → holders), heartbeats, stage
assignment on join, activation relay (Path A) or WebRTC signaling (Path B), token fan-out, and reroute logic.

> We **do not rewrite the Rust ce-hub.** The Rust hub remains the local/dev rendezvous for generic WASM
> tasks; ce-tabnet's coordination is the JS/Cloudflare layer the task brief asks us to extend. The wire
> vocabulary (`hello`/`hb`/caps) is deliberately the same dialect so a tab is recognizably a CE browser node.

Full DO API and message schemas: see [`module-contract.md`](module-contract.md).

---

## 10. How it ties to CE

| tabnet concept | CE primitive | Mapping |
|---|---|---|
| A tab | CE **browser node** (`node.html`) | Same identity (`ce_node_id` in localStorage), same capability detection, same heartbeat dialect. |
| Stage capacity advertisement | CE atlas self-tags | A tab advertises `["tabnet","stage:<run>","webgpu","vram:<mb>"]`; the planner reads them like `/atlas`. |
| Stage = billable work | CE economy (JobBid/Settle, payment channels) | Holding a stage and serving tokens is metered; the operator pays per token via a payment channel; tabs earn credits — the same model as running WASM tasks for the hub. |
| Placement / ranking | CE benchmark + latency graph (`ce-bench`, `ce-sched`) | `cpu_mark`/`vram_mb` + latency-graph RTT feed the `PlacementPlanner`, same signals `swarm select_hosts()` uses. |
| Weight chunks | CE content-addressed **blobs** + `ce-pin` | Model pack = blob set; CIDs are integrity; replicas fetch ranges peer-to-peer. |
| Auth (who may run/operate a Run) | `ce-cap` capability chains | A Run can require a signed cap to operate (`tabnet:operate`) or to join as a paid stage (`tabnet:stage`); abilities are opaque strings, attenuating chains rooted at the operator/org key. (Optional for the open demo; on by default for paid Runs.) |
| Activation transfer | CE mesh stream by node id (future) | Today: WebRTC/WS via the DO. The same `(seq_id, token_pos, stage)` envelope maps onto a libp2p stream once browser nodes are first-class mesh peers — no protocol change. |

The strategy choice (pipeline-only) and the failure model (Petals re-route/re-recruit) are taken directly
from `PLAN/09-hospital-inference.md` §2.2 (v2 sharding scaffold) — ce-tabnet is the **browser-native,
public-internet realization** of that scaffold.

---

## 11. Security and abuse notes (honest)

- **Untrusted tabs:** a malicious stage could return wrong activations (poisoning the output). Mitigations:
  optional **redundant stages with cross-checking** (compare two replicas' activations; mismatch → eject),
  and **capability-gated paid Runs** so stages have skin in the game (slashing via the CE economy). The open
  demo runs trust-on-faith; paid Runs use `R≥2` + reputation.
- **Weight confidentiality:** weights are public for open models; for private models, restrict the blob
  endpoint with capability-gated signed URLs (out of scope for the demo).
- **DO as SPOF per Run:** each Run depends on its DO; that is acceptable (the DO is highly available on
  Cloudflare) and Runs are independent, so blast radius is one model instance.

---

## 12. What ships now vs. what scales

- **Now (demo, this repo):** join/orchestrate pages, the Run DO (assignment + WS relay + heartbeat +
  reroute), `shard-loader` (range fetch + WebGPU upload), `inference.js` (WebGPU stage + WASM fallback) for a
  **small real model** at 1–4 stages, micro-batching across concurrent prompts. End-to-end token streaming.
- **Scales unchanged to 300B:** more tabs → more stages; the planner, the relay, the healing, and the
  range-fetch are all `S`-agnostic. Path B (WebRTC) and activation int8 are the throughput/latency
  optimizations to enable for large Runs. The only *new artifact* for 300B is the **300B model pack**
  (sharded/quantized weights as blobs) — see `scaling-to-300b.md`.
