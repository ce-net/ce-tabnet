# @ce-net/tabnet

**Run a large LLM sharded across many ordinary browser tabs — laptops and phones — coordinated so it is dead simple: open a URL, lend a tab.**

> The target this architecture makes *truthfully achievable*:
> *"I ran a 300B-parameter LLM across my old laptops at home in browser tabs with high throughput using ce-net — let's go bigger."*

ce-tabnet shards one transformer model **by layer** across N browser tabs. Each tab loads exactly one
**pipeline stage** (a contiguous range of transformer layers) into WebGPU, the tabs form a chain, and
**hidden-state activations (~KB per token) flow stage to stage**. The first tab embeds the prompt; the last
tab samples a token and streams it back. Many tokens are kept **in flight at once** (micro-batching) so
every tab stays busy — that is where throughput comes from. This is the pipeline-parallel strategy blessed
by [`PLAN/09-hospital-inference.md`](../PLAN/09-hospital-inference.md): **pipeline-parallel only, never
tensor-parallel over the network** (per-layer all-reduce barriers make a network slower than one box).

---

## Architecture

```
 operator (web/orchestrator.html)            Cloudflare Worker + Durable Object
        │  define run, prompt                  one DO instance == one run
        │  watch topology, stream tokens       (coordinator/src/index.js: RunCoordinator)
        ▼                                                 │ assigns stages, relays
   ┌──────────────┐    activation (~KB/tok)    ┌──────────┴──────────┐   activations,
   │ tab 0  stage0│ ───────────────────────▶   │      RunCoordinator   │   fans tokens,
   │ +embed       │                            │  topology · healing   │   heals dropout
   │ layers[0..k) │ ◀── token (feedback) ───   └──────────┬──────────┘
   └──────────────┘                                       │
   ┌──────────────┐                            ┌──────────┴──────────┐
   │ tab 1  stage1│  ...  hop ...  ┌───────────│ tab N-1  stageN-1    │
   │ layers[k..m) │ ─────────────▶ │ +lm_head  │ samples next token   │
   └──────────────┘                └───────────┴─────────────────────┘
```

- **Tabs are CE browser nodes.** Each detects its caps (cores, RAM, WebGPU, VRAM budget) exactly like
  `web/site/node.html`, gets a local node id, and joins a run over a WebSocket.
- **Coordinator is a Cloudflare Worker + Durable Object** (`coordinator/src/index.js`, class
  `RunCoordinator`). One DO per run id. It assigns memory-weighted contiguous layer ranges to tabs,
  relays activation frames stage→stage (binary frames forwarded verbatim), fans sampled tokens to the
  operator and back to stage 0 for autoregression, and heals on dropout (promote a replica or recruit a
  spare and tell it which `[lo,hi)` slice to re-fetch). It uses the **WebSocket Hibernation API** so it can
  hold hundreds of tab sockets cheaply.
- **Weights are content-addressed.** `web/shard-loader.js` fetches a stage's layer objects by CID (HTTP
  range / whole-object / CE blob gateway), verifies each against its SHA-256, and caches them in the Cache
  Storage API so a returning or recruited tab is instant.
- **The engine is real.** `web/inference-worker.js` implements the transformer block on WebGPU (RMSNorm,
  quantized matvec with q4_0/q8_0 dequant, RoPE, GQA attention with a local KV cache, SwiGLU, residuals,
  sampling) **and** a correct CPU/WASM fallback for devices without WebGPU.

```
ce-tabnet/
├── README.md                    # this file
├── package.json                 # @ce-net/tabnet — type:module, zero runtime deps
├── .gitignore
├── dev/
│   └── serve.js                 # zero-dep static server + `--selftest` mock pipeline
├── web/                         # vanilla JS + WebGPU, NO build step — just open the files
│   ├── config.js                # THE coordinator-URL resolver (shared by both pages)
│   ├── join.html                # "lend a tab": detect caps, join a run, become a stage
│   ├── orchestrator.html        # operator console: define a run, watch topology, send prompts
│   ├── tabnet-node.js           # tab runtime: connection, stage lifecycle, autoregression, healing
│   ├── inference-worker.js      # the REAL WebGPU stage engine (+ CPU/WASM fallback); also a Web Worker
│   ├── inference.js             # interface reference (the engine lives in inference-worker.js)
│   ├── shard-loader.js          # content-addressed weight fetch (HTTP range), verify, cache
│   ├── protocol.js              # shared wire-message constructors/validators (single source of truth)
│   └── model-config.js          # model registry + stage-plan math (imported by the DO and the UI)
├── coordinator/                 # Cloudflare Worker + Durable Object (the only stateful piece)
│   ├── wrangler.jsonc           # deploy config (fill account_id; do NOT deploy without creds)
│   ├── src/index.js             # Worker router + RunCoordinator DurableObject
│   └── README.md                # coordinator API + deploy
├── models/
│   ├── registry.json            # canonical model dimensions + honest tab math (7B…405B…300B)
│   ├── plan.js                  # caps-aware memory-weighted planner (richer sibling of model-config)
│   └── README.md                # how to produce a tabnet model pack
├── tools/
│   └── shard-model.js           # model packer: content-address layers + write manifest.json
└── docs/
    ├── architecture.md          # the full design
    ├── scaling-to-300b.md       # the honest 300B recipe + the math + how to reproduce small today
    └── module-contract.md       # wire protocol + DO API contract
```

---

## Try it in 30 seconds (local)

You need **Node 18+**. No build step, no cargo, no Cloudflare account for local dev.

```bash
cd ce-tabnet

# 1. Prove the whole message flow works WITHOUT a browser or GPU (mock 2-stage pipeline):
node dev/serve.js --selftest          # exits 0, prints the full join→prompt→token trace

# 2. Serve the pages:
node dev/serve.js                     # http://127.0.0.1:8973

# 3. In another terminal, start the coordinator (Cloudflare Worker + DO, emulated by miniflare):
npx wrangler dev --config coordinator/wrangler.jsonc   # ws://127.0.0.1:8787

# 4. Open the operator console, create a run, lend tabs:
#    http://127.0.0.1:8973/orchestrator.html
#    - pick "tinyllama-1.1b-q4", set Stages = 2 or 3, click "Create run"
#    - copy the join link, open it in that many browser tabs (or phones on the LAN)
#    - each tab loads its stage and turns green; type a prompt → tokens stream back
```

The pages auto-target `ws://127.0.0.1:8787` on localhost (see `web/config.js`). Override the coordinator
anywhere with `?hub=wss://your-worker.example.workers.dev`.

> Running real weights end-to-end in the browser requires a produced **model pack** (per
> [`models/README.md`](models/README.md)) hosted with HTTP range support. The selftest and the topology/UI
> work today with no pack; the WebGPU compute path runs as soon as a pack URL is reachable.

---

## Deploy the coordinator (one command — needs YOUR Cloudflare creds; not run for you)

1. Fill `account_id` and pick an egress (workers.dev subdomain or a `ce-net.com/tabnet/*` route) in
   [`coordinator/wrangler.jsonc`](coordinator/wrangler.jsonc).
2. `npx wrangler login`
3. **One command:**
   ```bash
   npx wrangler deploy --config coordinator/wrangler.jsonc
   ```

Validate the bundle without deploying:
```bash
npx wrangler deploy --dry-run --config coordinator/wrangler.jsonc
```

See [`coordinator/README.md`](coordinator/README.md) for the full HTTP/WebSocket API.

---

## The scaling math (honest)

```
weights_bytes    = params × bits_per_param / 8         # Q4 ≈ 4.5 bits/param incl. block scale
tabs_for_weights = ceil(weights_bytes / per_tab_budget)
```

A browser tab realistically dedicates **0.5–2 GB** of WebGPU buffer space to weights (phone ≈ 0.5 GB,
gaming laptop ≈ 2 GB). Planning at **1 GB/tab**:

| Model (4-bit) | Weights | @0.5 GB (phones) | @1 GB (typical) | @2 GB (good laptops) |
|---|---|---|---|---|
| 7B   | ~3.5 GB | 7   | 4   | 2   |
| 70B  | ~35 GB  | 70  | 35  | 18  |
| **300B** | **~150 GB** | **300** | **150** | **75** |
| 405B | ~200 GB | 400 | 200 | 100 |

**300B @ 4-bit needs ~100–300 tabs** depending on device mix. This is a *cluster-of-tabs* claim, not a
two-laptop claim — and the framework's actual innovation is making that cluster trivial to assemble: every
participant just opens a URL. Only the boundary activation (`hidden_dim × 2` bytes, fp16) crosses the
network per hop — a few KB per token — so the network is never the bottleneck; per-hop latency and
aggregate tab GPU are. Full recipe: [`docs/scaling-to-300b.md`](docs/scaling-to-300b.md).

---

## What is real vs needs-more-tabs / needs-deploy

**Real and verified today (no GPU, no deploy):**
- The wire protocol (`web/protocol.js`) and its binary activation framing — round-trip tested.
- The stage-planning math (`web/model-config.js`, `models/plan.js`) — contiguous layer coverage, honest tab
  counts including the 300B/405B arithmetic.
- The full coordination message flow — `node dev/serve.js --selftest` drives a mock 2-stage pipeline:
  join → assign → ready → prompt → prefill → activation hops → sampled token → fan-out → autoregressive
  continuation → coordinator-enforced `max_tokens` → done.
- The coordinator bundles and validates: `npx wrangler deploy --dry-run` succeeds.
- The static dev server serves all pages with correct MIME types.

**Real code, exercised in a browser (no deploy needed beyond a hosted model pack):**
- WebGPU transformer-stage compute + CPU/WASM fallback (`web/inference-worker.js`).
- Content-addressed shard loading, SHA-256 verification, Cache-Storage caching (`web/shard-loader.js`).
- The operator console topology graph, live tok/s, per-stage latency, healing visualization.

**Needs deployment:** a public coordinator (one `wrangler deploy`) so tabs on different networks can join.

**Needs more tabs / a produced model pack:** running 70B/300B/405B for real. The code path is identical to
the small demo — only the stage count and the model pack change. **No 300B run has been performed**; the
repo ships the architecture that scales to it and a demonstrable small-scale pipeline.
