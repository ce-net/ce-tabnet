# ce-tabnet — launch post copy

This file holds the launch copy for X and Reddit, plus a **pre-post checklist** so the tagline is literally
true before you publish. The whole point is to be loud *and* honest: describe the capability and the math,
and only claim a run you have actually performed.

The aspirational line:

> **"I ran a 300B-parameter LLM across my old laptops at home in browser tabs with high throughput using
> ce-net — let's go bigger."**

Read the **Honesty rules** at the bottom first. The number you put in the headline must match a run you
actually did. Templates below have `{...}` slots for the real numbers from your run.

---

## SHORT — X / Twitter

**Option A (what you actually ran — use this first):**

> Sharded a real LLM across {N} browser tabs on my old laptops at home. No install — each tab opens a URL
> and becomes one *pipeline stage* (a slice of the model's layers). Only ~KB of hidden state crosses the
> network per token. {tok_s} tok/s. The architecture scales to 300B (~150GB @ 4-bit = ~100–300 tabs).
> Open-source, runs on @Cloudflare Workers. Let's go bigger 👇
> {link}

**Option B (once you have ~100–300 capable tabs + a 300B/405B pack and have actually run it):**

> I ran a 300B-parameter LLM across browser tabs on machines at home + friends', with high throughput,
> using ce-net. Each tab = one pipeline stage (a range of transformer layers); only hidden-state activations
> (~KB/token) cross the wire. Open a URL, lend a tab. Let's go bigger.
> {link}

**One-liner variant:**

> Pipeline-parallel LLM inference across ordinary browser tabs. Open a URL → your tab becomes a stage of a
> model too big for any one device. 300B is ~150GB @ 4-bit ≈ ~150 tabs — and the framework makes assembling
> that cluster a link-share. ce-tabnet, MIT, on Cloudflare. {link}

---

## LONG — Reddit (r/LocalLLaMA, r/MachineLearning, r/selfhosted)

**Title:**
`Pipeline-parallel LLM inference across ordinary browser tabs — open a URL, lend a tab. The path to 300B across a cluster of tabs (and the honest math).`

**Body:**

I built **ce-tabnet**: it shards one transformer model **by layer** across many browser tabs. Each tab —
on a laptop or a phone — loads exactly one **pipeline stage** (a contiguous range of transformer layers)
into WebGPU. The tabs form a chain. A token's hidden state (a few KB) enters stage 0, is computed, and hops
tab → tab to the last stage, which samples the next token and streams it back. **Only those tiny activations
cross the network — never the weights.** No install: a participant opens a URL and their tab is computing.

**Why pipeline-parallel and not tensor-parallel?** Tensor parallelism needs an all-reduce *every layer* —
over a network that barrier makes a "cluster" slower than a single box. Pipeline parallelism crosses the
network only at layer boundaries (~`hidden_dim × 2` bytes per token), so the network is never the
bottleneck. This is the strategy real distributed-inference systems (Petals, EXO) converge on.

**The honest math — this is the load-bearing part:**

A browser tab realistically holds **0.5–2 GB** of WebGPU weights (phone ≈ 0.5 GB, gaming laptop ≈ 2 GB).
So:
- 7B @ 4-bit ≈ 3.5 GB → ~4 tabs
- 70B @ 4-bit ≈ 35 GB → ~35 tabs
- **300B @ 4-bit ≈ 150 GB → ~100–300 tabs** (≈150 at 1 GB/tab)
- 405B @ 4-bit ≈ 200 GB → ~100–200 tabs

There's no way around that arithmetic and I don't pretend there is. **300B is a cluster-of-tabs claim, not
a two-laptop claim.** The actual innovation isn't beating the memory wall — it's that assembling the cluster
is a *link share*: you, your old laptops, your friends, a lab, a LAN party. Throughput comes from
**micro-batching** — many tokens in flight across the pipeline at once, so every tab stays busy even though
a single token must traverse every stage (that's the latency tax of pipelines).

**How it's built:**
- **Tabs are CE browser nodes** (capability detection + node identity, same pattern as ce-net's in-browser
  node). Vanilla JS + WebGPU, **no build step** — open the HTML.
- **Coordinator = a Cloudflare Worker + Durable Object**, one DO per run. It does memory-weighted stage
  assignment (beefiest tabs get the most layers + the embed/lm-head ends), relays activation frames, fans
  tokens, and **heals** on dropout (promote a replica, or recruit a spare and tell it exactly which layer
  range to re-fetch). Uses the WebSocket Hibernation API so it holds hundreds of sockets cheaply.
- **Weights are content-addressed** and fetched by HTTP range, SHA-256-verified on arrival, cached so a
  returning/recruited tab is instant.
- **Real engine:** WebGPU kernels (RMSNorm, q4/q8 dequant matvec, RoPE, GQA attention with a local KV
  cache, SwiGLU, sampling) with a correct CPU/WASM fallback for GPU-less devices.

**What I'm actually showing in this post:** {describe exactly what you ran — e.g. "TinyLlama-1.1B forced
across 3 tabs on two laptops + a phone, {tok_s} tok/s end-to-end, with a tab killed mid-generation to show
the pipeline heal."} The same code path runs the small demo and a 300B-class model — only the stage count
and the model pack change.

**Try it yourself in 30 seconds:** {see section below}

Repo (MIT): {link}. Honest scaling recipe + the full math: `docs/scaling-to-300b.md`. Feedback welcome —
especially from anyone who wants to pool tabs and actually push the stage count up. Let's go bigger.

---

## "Try it yourself in 30 seconds"

**Easiest (once the coordinator is deployed):** open the operator console link, click **Create run**, then
open the **join link** in a few tabs (or send it to friends). Each tab becomes a stage; type a prompt and
watch tokens stream.

> 1. Open **{operator-console-link}**
> 2. Pick a model, click **Create run**, copy the **join link**
> 3. Open the join link in 2–3 tabs (or phones) — each turns green as it loads its stage
> 4. Type a prompt → tokens stream back through the pipeline

**Fully local (Node 18+, no account):**

```bash
git clone {repo} && cd ce-tabnet
node dev/serve.js --selftest      # prove the pipeline message-flow end to end (no GPU needed)
node dev/serve.js                 # serve the pages at http://127.0.0.1:8973
npx wrangler dev --config coordinator/wrangler.jsonc   # coordinator at ws://127.0.0.1:8787
# open http://127.0.0.1:8973/orchestrator.html → create a run → lend tabs
```

---

## PRE-POST CHECKLIST — do these before publishing so the headline is TRUE

The repo as shipped is verified at the **coordination + math + small-pipeline** level. To make a *headline
claim about running a model*, complete the relevant rows and put the **real numbers** in the templates.

- [ ] **Deploy the coordinator.** `npx wrangler deploy --config coordinator/wrangler.jsonc` (needs your
      Cloudflare creds). Put the operator-console URL in the "Try it" section.
- [ ] **Produce a model pack** for whatever you'll demo (start with `tinyllama-1.1b-q4`), host it with HTTP
      range support (R2/S3/CDN), and point `web/model-config.js` `baseUrl` at it. See `models/README.md`.
- [ ] **Verify correctness first:** run the model **unsharded** (1 stage) and confirm coherent output, then
      force it across 2–3 stages and confirm the *same* output. Pipeline parallelism must not change results.
- [ ] **Run the real small demo** across ≥2 physical devices (e.g. two laptops, or a laptop + a phone).
      Record: model, **# tabs/stages**, **tok/s end-to-end**, **time-to-first-token**. Use these in Option A.
- [ ] **Capture proof:** screen-record the operator console (topology filling, tokens streaming, the
      per-stage latency table). Optionally kill a tab mid-run to show healing. Attach the clip/GIF.
- [ ] **Only if you literally ran 300B/405B:** confirm you had ~100–300 capable tabs and a produced 300B/405B
      pack, recorded the throughput, and have a clip. *Then* you may use Option B / the literal-300B headline.
      If you did **not** do this, use **Option A** and the "architecture scales to 300B" framing.
- [ ] **Sanity-check every number** in the post against your recording. No rounding up beyond your data.

---

## HONESTY RULES (non-negotiable)

- **Never claim a run you didn't do.** If you demoed a 1.1B across 3 tabs, the headline is about *that*, plus
  "the architecture scales to 300B (here's the math)." The 300B number is fine to *discuss* (it's just
  arithmetic); it is **not** fine to state "I ran 300B" unless you did.
- **300B is a cluster-of-tabs claim.** "My old laptops at home" only literally reaches 300B if "home" has
  ~100+ tabs' worth of GPU (a lab/LAN party/pooling with friends). Say so — it's a stronger story than a lie.
- **"High throughput" = micro-batched aggregate**, with a single-stream pipeline-depth latency tax. State the
  real measured tok/s; don't imply a single token is fast.
- **The weights never cross the network; only ~KB of activations per token do.** This is true and worth
  repeating — it's why the approach works.
- Link `docs/scaling-to-300b.md` so anyone can check the arithmetic themselves.
