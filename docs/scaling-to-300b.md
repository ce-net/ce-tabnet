# Scaling ce-tabnet to 300B — the honest recipe

This document makes the tagline truthfully achievable:

> *"I ran a 300B-parameter LLM across my old laptops at home in browser tabs with high throughput using
> ce-net."*

It states exactly what model, how many tabs of what size, the throughput you can expect, and — critically —
**how a reader reproduces a smaller version today** with the same code path.

No hand-waving: the 300B number falls straight out of arithmetic, and we show the arithmetic.

---

## 1. The arithmetic (repeat of the load-bearing math)

```
weights_bytes  = params × bits_per_param / 8
tabs_for_weights = ceil(weights_bytes / PER_TAB_WEIGHT_BUDGET)
```

A browser tab can realistically dedicate **0.5–2 GB** of WebGPU buffer space to weights (mobile ≈ 0.5 GB,
gaming laptop ≈ 2 GB). We plan at **1 GB/tab** and leave the rest for KV cache, activations, and the JS/WASM
heap.

| Model (4-bit) | Weights | Tabs @0.5 GB (phones) | Tabs @1 GB (typical) | Tabs @2 GB (good laptops) |
|---|---|---|---|---|
| 7B | ~3.5 GB | 7 | 4 | 2 |
| 70B | ~35 GB | 70 | 35 | 18 |
| **300B** | **~150 GB** | **300** | **150** | **75** |

So **300B @ 4-bit needs ~100–300 tabs** depending on device mix. With a fleet of *old laptops* (call it
1.5 GB usable each), you need roughly `150 / 1.5 ≈ 100` tabs. That is "my old laptops at home" only if home
has ~100 browser tabs' worth of GPU — i.e. a LAN party, a lab, or pooling with friends over the internet.
**We say this plainly:** 300B is a *cluster-of-tabs* claim, not a two-laptop claim. The framework makes the
cluster trivial to assemble (open a URL), which is the actual innovation.

---

## 2. Which open model for the 300B target

Pick a real, open, **Mixture-of-Experts or dense** model in the right class and ship a **tabnet model pack**
for it. Concrete candidates (choose by license + availability at build time):

- **Dense ~300B:** Falcon-180B is the largest widely-available *dense* open model (~180B); a true 300B dense
  open model may require Llama-3.1-405B-class weights (405B → ~200 GB @4-bit → ~130–200 tabs). The 405B model
  is the most credible "300B-plus" open dense target.
- **MoE (better fit for pipelines):** Mixtral-8x22B (~141B total) or DeepSeek-V3 / Qwen-class MoE
  (hundreds of B total, ~37B active). MoE is attractive because **only the active experts compute per token**,
  but pipeline-parallel tabnet still must *store* all experts across tabs (storage = total params), while
  *compute* per token is the active subset. So MoE lowers per-token FLOPs (helps throughput) but **not** the
  tab count (storage-bound). For tabnet, MoE expert-parallel placement is a natural extension: assign whole
  experts to tabs (each tab holds a few experts of a layer) — a documented v2 layout.

**Recommended 300B-class target:** **Llama-3.1-405B, Q4** (the honest "300B-plus" dense model) packed as
per-layer blobs. Reason: dense → clean contiguous layer-range sharding (the simplest correct tabnet layout),
fully open weights, well-understood architecture that `inference.js` already targets (Llama family).

---

## 3. Stage plan for 405B-class @ Q4 across ~130 tabs

Assume 405B → ~126 transformer layers, `hidden_dim ≈ 16384`, ~200 GB weights @ Q4.

```
PER_TAB_WEIGHT_BUDGET = 1.5 GB   (old-laptop class)
tabs_for_weights      = ceil(200 / 1.5) ≈ 134 tabs
layers_per_tab        = ceil(126 / 134) ≈ 1 layer per tab   (some tabs hold 1, big ones hold 2-3)
```

- The **PlacementPlanner** sizes stages by each tab's reported `vram_mb`: a 3 GB laptop holds 2 layers, a
  0.8 GB phone holds part of a layer (layers can be split further into attention/MLP sub-objects in the model
  pack for very small devices — the manifest supports sub-layer objects).
- Stage 0 also holds `embed_tokens` (vocab × hidden — a few hundred MB at Q4; give it to a beefier tab).
- The last stage also holds `norm` + `lm_head` (another few hundred MB; also a beefy tab).
- **Replication `R=2`** for resilience at this scale → ~268 tab-slots, i.e. you want **~200–300 real tabs**
  to run 405B comfortably with healing. This is the "~100–300 tabs" range, stated honestly.

### Activation bandwidth at this scale
`hidden_dim=16384` × 2 bytes (fp16) = **32 KB per hop per token**, or **16 KB at int8**. Across 134 stages a
single token moves ~4 MB end-to-end, but **each hop is only 32 KB** and hops are sequential. At, say, 20
tokens/sec aggregate that is ~640 KB/s per hop — trivial for any home connection. **Bandwidth is never the
limit; per-hop RTT × stage-count is the single-stream latency cost** (see below).

---

## 4. Expected throughput and latency (honest numbers)

Let:
- `c` = per-stage compute time per token (depends on layers/tab and GPU; assume ~5–20 ms on an old laptop GPU
  for 1–2 layers at Q4).
- `r` = per-hop network latency (WebSocket-relay path ≈ 30–80 ms WAN; WebRTC LAN-direct ≈ 1–10 ms).

**Single-stream decode latency** ≈ `S × (c + r)`. With `S=134`, `c=10 ms`, `r=40 ms` → ~6.7 s/token. **Slow
for one user.** That is the pipeline-depth tax and we do not hide it. Two ways it gets usable:
1. **Fewer, fatter stages.** If you have 30 laptops with 6 GB each, `S≈34`, latency ≈ `34×50 ms ≈ 1.7 s/token`
   — still slow, but the system was never meant for low single-stream latency at 300B.
2. **WebRTC LAN-direct** (`r≈5 ms`) and overlapping compute/transfer drops the hop term sharply.

**Aggregate throughput** (the real product metric) ≈ `1 / (max-stage-time)` tokens/sec **per concurrent
stream**, multiplied by the number of streams the pipeline depth keeps in flight. With micro-batch window 8
and 8 concurrent users, the *system* delivers `~8 × (1 / c)` tokens/sec across users once the pipe is full —
e.g. `8 × (1/10 ms) = 800 tokens/sec` aggregate in the optimistic case, realistically tens to low hundreds of
tokens/sec aggregate on old hardware. **That is the "high throughput" claim: high aggregate tokens/sec across
many concurrent prompts, not low latency for one prompt.** Stated this way it is true.

---

## 5. Reproduce a SMALL version today (same code path)

You do not need 300B to prove the architecture. The demo uses the **identical** join page, DO, relay,
loader, and inference path — just a small model and few stages.

### Step A — pick a tiny real model
**TinyLlama-1.1B** or **Qwen2.5-0.5B**, quantized to Q4 (~0.3–0.6 GB). Single-tab-capable, so you can verify
correctness against an un-sharded reference, then *force* it across multiple stages to exercise the pipeline.

### Step B — build the tabnet model pack
See `models/README.md`. In short:
1. Quantize the model to Q4 per-layer.
2. Serialize each layer (and embed + lm_head) as its own content-addressed object.
3. Write `manifest.json`: `{n_layers, hidden_dim, n_heads, n_kv_heads, vocab, rope_theta, layers:[{idx, cid,
   bytes, quant}], embed:{cid,...}, lm_head:{cid,...}}`.
4. Host the objects on any HTTP server that supports **range requests** (R2, S3, even `python -m http.server`
   supports ranges) and put the base URL in `model-config.js`.

### Step C — run it
```bash
npm install                 # wrangler only
npm run dev:coordinator     # Run DO at http://127.0.0.1:8787
npm run serve:web           # pages at http://127.0.0.1:8973
```
1. Open `orchestrate.html`, choose the small model, set **stages = 1** first → verify a correct completion in
   one tab (sanity vs. reference).
2. Set **stages = 3**, create the Run, open the join URL in **3 tabs** (or 3 devices on your LAN). Watch them
   turn green as each loads its layer range into WebGPU.
3. Send a prompt; tokens stream back. You have now run a real model **pipeline-parallel across browser tabs**.
4. **Kill a tab** mid-generation to watch healing: the DO marks the stage `recruiting`, you open one more tab,
   it re-fetches that layer range, and generation resumes (Petals-style).

### Step D — scale the demo up honestly
- Move to **7B Q4** (≈4 tabs @1 GB) on a few laptops — a genuinely useful model, fully sharded.
- Then **13B** (≈7 tabs), then **70B** (≈35 tabs) if you can muster the devices. Each step is *only* a bigger
  model pack and more tabs; **no code changes.**
- The 405B/300B row is the same: produce the model pack, gather ~100–300 tabs (a class, a meetup, friends over
  the internet, a rack of old laptops), point them at the join URL.

---

## 6. The truthful version of the claim

What you can say honestly after this ships:

- ✅ "ce-tabnet shards any transformer by layer across browser tabs; I ran a real model **pipeline-parallel
  across N tabs** with token streaming and live healing."
- ✅ "The architecture scales to 300B: 300B @ 4-bit is ~150 GB of weights → ~100–300 tabs at ~1 GB/tab, and
  the system assigns that many stages with the same code I used for the 3-tab demo."
- ✅ "Throughput is high in aggregate (many concurrent prompts keep every stage busy via micro-batching)."
- ⚠️ Only claim the literal "I ran 300B" sentence once you have **actually assembled ~100–300 capable tabs and
  a 300B model pack and observed tokens** — the framework makes that reproducible, but do not assert the run
  before it happens. This repo ships the capability and the recipe, not a fabricated 300B run.
