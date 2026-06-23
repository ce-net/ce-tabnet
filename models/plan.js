// ce-tabnet stage planner.
//
// Given a model (layer count + per-layer/embed/lm_head weight bytes) and a set of
// joining tabs (their reported device caps), produce a memory-weighted pipeline plan:
//   stage -> contiguous layer range [lo, hi), the holder tab, plus embed/lm_head
//   placement and R replicas per stage.
//
// This is the richer, caps-aware sibling of web/model-config.js::defaultStagePlan.
// model-config.defaultStagePlan takes raw byte budgets and returns ranges only; this
// module takes real device caps, derives honest per-tab WEIGHT budgets from them,
// assigns actual holder tabs + replicas + the embed/lm_head extras, and reports
// feasibility (and sub-layer split hints when a layer is too big for any tab).
//
// Pure JS, no I/O, no imports. Usable in the browser (orchestrate.html), in the DO
// planner (planStages), and in tooling/tests under Node (import as an ES module).
//
// The registry shape consumed here is models/registry.json (or the equivalent entry
// from web/model-config.js MODELS). A "model" object needs at minimum:
//   { n_layers, approx_weight_bytes }
// and SHOULD provide for accurate placement:
//   { per_layer_bytes, embed_bytes, lm_head_bytes, hidden_dim, n_kv_heads, head_dim }
//
// A "cap" object is exactly the node.html detect() shape:
//   { cores, ram_gb, storage_gb, gpu, webgpu, platform, vram_mb, cpu_mark }
// plus an identifying `node` (hex node id) so the plan can name holders.

// -----------------------------------------------------------------------------
// Budget defaults (honest; see docs/architecture.md §4, docs/scaling-to-300b.md §1)
// -----------------------------------------------------------------------------

const GiB = 1024 * 1024 * 1024;
const MiB = 1024 * 1024;

export const BUDGET = Object.freeze({
  // Conservative cross-device weight budget when caps give us nothing usable.
  FALLBACK_WEIGHT_BYTES: 1.0 * GiB,
  // Fraction of usable GPU memory we dare spend on WEIGHTS (rest is KV cache,
  // activations, the WGSL runtime, and the JS/WASM heap).
  WEIGHT_FRACTION_OF_VRAM: 0.6,
  // node.html's vram_mb is the max single-buffer binding limit (maxBufferSize /
  // maxStorageBufferBindingSize) — a LOWER BOUND on real VRAM, not the total. Real
  // devices hold several such buffers, so multiply to estimate total addressable
  // weight memory. Kept conservative.
  VRAM_BINDING_TO_TOTAL: 4,
  // When WebGPU is absent we run the WASM/CPU fallback out of system RAM. Spend a
  // small slice of reported RAM on weights (CPU path is slow; keep stages tiny).
  CPU_RAM_FRACTION_FOR_WEIGHTS: 0.25,
  // Floor / ceiling so a single garbage reading can't blow up the plan.
  MIN_WEIGHT_BYTES: 64 * MiB,
  MAX_WEIGHT_BYTES: 6.0 * GiB,
});

// Estimate how many bytes of WEIGHTS one tab can hold, from its reported caps.
// Honest and defensive: many browsers under-report, so we floor/ceil and fall back.
export function tabWeightBudget(caps, opts = {}) {
  const fallback = opts.fallback ?? BUDGET.FALLBACK_WEIGHT_BYTES;
  if (!caps || typeof caps !== "object") return fallback;

  let bytes = 0;
  if (caps.webgpu && caps.vram_mb && caps.vram_mb > 0) {
    const totalVram = caps.vram_mb * MiB * BUDGET.VRAM_BINDING_TO_TOTAL;
    bytes = totalVram * BUDGET.WEIGHT_FRACTION_OF_VRAM;
  } else if (caps.ram_gb && caps.ram_gb > 0) {
    // No usable WebGPU number => CPU/WASM fallback budget from system RAM.
    bytes = caps.ram_gb * GiB * BUDGET.CPU_RAM_FRACTION_FOR_WEIGHTS;
  } else {
    bytes = fallback;
  }
  // Clamp.
  bytes = Math.max(BUDGET.MIN_WEIGHT_BYTES, Math.min(BUDGET.MAX_WEIGHT_BYTES, bytes));
  return Math.floor(bytes);
}

// Per-layer weight cost for a model. Prefer the measured per_layer_bytes from the
// registry; otherwise approximate from total weights minus embed/lm_head.
export function perLayerBytes(model) {
  if (model.per_layer_bytes && model.per_layer_bytes > 0) return model.per_layer_bytes;
  const extras = (model.embed_bytes || 0) + (model.lm_head_bytes || 0);
  const layerTotal = Math.max(1, (model.approx_weight_bytes || 0) - extras);
  return Math.ceil(layerTotal / Math.max(1, model.n_layers));
}

// KV-cache bytes one layer consumes at a given context length (fp16). Counted so a
// stage's budget includes cache, not just weights. (2 = K and V, 2 = bytes/fp16.)
export function kvBytesPerLayer(model, ctx) {
  const kvHeads = model.n_kv_heads || model.n_heads || 1;
  const headDim = model.head_dim || (model.hidden_dim && model.n_heads ? model.hidden_dim / model.n_heads : 64);
  return 2 * ctx * kvHeads * headDim * 2;
}

// -----------------------------------------------------------------------------
// Tab math (display): how many tabs a model needs at a given uniform budget.
// -----------------------------------------------------------------------------

export function tabsNeeded(model, perTabBytes = BUDGET.FALLBACK_WEIGHT_BYTES) {
  if (!model) throw new Error("tabsNeeded: model required");
  return Math.max(1, Math.ceil((model.approx_weight_bytes || 0) / perTabBytes));
}

// The honest tab-math table for a model across common per-tab budgets. Drives the
// "tabs needed" panel in orchestrate.html. Returns plain numbers, no claims.
export function tabMath(model) {
  const budgets = { "0.5GB": 0.5 * GiB, "1GB": 1.0 * GiB, "1.5GB": 1.5 * GiB, "2GB": 2.0 * GiB };
  const out = { weights_bytes: model.approx_weight_bytes || 0, weights_gb: round1((model.approx_weight_bytes || 0) / GiB) };
  out.tabs = {};
  for (const [k, b] of Object.entries(budgets)) out.tabs[k] = tabsNeeded(model, b);
  out.per_layer_gb = round2(perLayerBytes(model) / GiB);
  // A layer that exceeds the smallest budget needs sub-layer (attn/mlp) split objects.
  out.layer_needs_split_below_gb = round2(perLayerBytes(model) / GiB);
  return out;
}

// -----------------------------------------------------------------------------
// The planner.
// -----------------------------------------------------------------------------
//
// planStages(model, tabs, opts) -> Plan
//
//   model : a registry model entry (see top-of-file contract).
//   tabs  : array of { node, caps } — the live candidate tabs (join order = array order).
//   opts  : {
//             stages,        // operator-requested S (optional). If feasible, honored;
//                            // else the planner picks the fewest stages that fit.
//             replicas,      // R per stage (default 1).
//             context,       // max context length for KV budgeting (default model/registry).
//             model_id,      // string id to stamp on assignments (defaults model.model_id).
//             manifest_ref,  // manifest path (defaults model.manifestRef || "manifest.json").
//             weightFraction // override BUDGET.WEIGHT_FRACTION_OF_VRAM for the run.
//           }
//
// Returns:
//   {
//     model_id, n_layers, S, R, context,
//     feasible,            // bool: every layer is placed and every stage has a holder
//     reason,              // string when !feasible (or notes when feasible)
//     stages: [
//       {
//         stage, layers:[lo,hi),          // contiguous layer range this stage owns
//         is_first, is_last,
//         holder: node|null,              // primary holder tab (null => recruiting)
//         replicas:[node,...],            // R-1 replica holders (may be short if scarce)
//         prev_node, next_node,           // activation neighbors (primary holders)
//         has_embed, has_lm_head,         // extra weights pinned to this stage
//         weight_bytes,                   // est. bytes this stage must load (layers + extras)
//         kv_bytes,                       // est. KV cache at `context`
//         budget_bytes,                   // holder's derived weight budget
//         fits,                           // weight_bytes <= budget_bytes
//         model_id, manifest_ref,
//       }, ...
//     ],
//     unassigned_layers:[lo,hi)|null,     // layers no tab could hold (=> need more/bigger tabs)
//     spares:[node,...],                  // tabs beyond what the plan needs (healing pool)
//     splitHints:[{stage, layers, reason, suggest:"parts:attn+mlp"}], // sub-layer split advice
//     tabsNeeded,                         // honest count at the fallback budget (display)
//     plan_version,                       // monotonic stamp (caller may override)
//   }
//
// Algorithm (matches docs/module-contract.md §C placement rule):
//   1. Derive each tab's weight budget from caps (tabWeightBudget).
//   2. Sort tabs largest-budget-first (beefiest tabs anchor embed/lm_head + biggest ranges).
//   3. Walk layers low->high, greedily packing as many layers as fit each successive tab's
//      budget (after reserving embed on stage 0 and lm_head on the last stage).
//   4. If `opts.stages` is given and feasible (>= the greedy minimum and <= #tabs), rebalance
//      layers into exactly S stages, still proportional to budget.
//   5. Wire prev/next neighbors; attach R-1 replicas per stage from the leftover beefiest tabs.
//   6. Report feasibility, spares, and split hints for layers too big for any single tab.

export function planStages(model, tabs, opts = {}) {
  if (!model) throw new Error("planStages: model required");
  const N = model.n_layers;
  const R = Math.max(1, opts.replicas ?? 1);
  const ctx = opts.context ?? model.max_context ?? 4096;
  const model_id = opts.model_id ?? model.model_id ?? "unknown";
  const manifest_ref = opts.manifest_ref ?? model.manifestRef ?? "manifest.json";
  const wf = opts.weightFraction ?? BUDGET.WEIGHT_FRACTION_OF_VRAM;

  const layerBytes = perLayerBytes(model);
  const embedBytes = model.embed_bytes || 0;
  const lmHeadBytes = model.lm_head_bytes || 0;
  const kvPerLayer = kvBytesPerLayer(model, ctx);

  // 1. Derive budgets, keep node ids.
  const cand = (tabs || []).map((t) => ({
    node: t.node,
    caps: t.caps || {},
    // Reserve a slice for KV cache on top of weights when budgeting layer capacity.
    budget: tabWeightBudget(t.caps, { fallback: BUDGET.FALLBACK_WEIGHT_BYTES }) *
            (wf / BUDGET.WEIGHT_FRACTION_OF_VRAM),
  }));

  // 2. Beefiest first.
  const sorted = [...cand].sort((a, b) => b.budget - a.budget);

  // How many whole layers fit in a budget after optionally reserving an extra (embed/lm_head).
  const layersFit = (budget, reserve, kvSlack) => {
    const usable = budget - reserve;
    // each layer costs weights + its KV cache at full context
    const perLayerCost = layerBytes + (kvSlack ? kvPerLayer : 0);
    return Math.floor(usable / Math.max(1, perLayerCost));
  };

  // ---- 3. Greedy minimum plan over available tabs ----
  // First stage reserves embed; we don't know the last stage until we finish, so we
  // do a first greedy pass to find the count, then pin lm_head onto the final stage.
  const greedy = [];
  let lo = 0, ti = 0;
  let splitHints = [];
  while (lo < N && ti < sorted.length) {
    const tab = sorted[ti];
    const reserve = (greedy.length === 0) ? embedBytes : 0; // embed pinned to stage 0
    let canHold = layersFit(tab.budget, reserve, true);
    if (canHold < 1) {
      // This tab can't hold even one layer of this model. If NO tab can hold a single
      // layer, the layer itself is too big => emit a split hint and stop (infeasible
      // without sub-layer objects).
      const anyCanHold = sorted.some((s) => layersFit(s.budget, 0, true) >= 1);
      if (!anyCanHold) {
        splitHints.push({
          stage: greedy.length,
          layers: [lo, lo + 1],
          reason: `one layer (~${round2(layerBytes / GiB)} GB + KV) exceeds every tab's weight budget`,
          suggest: "parts:attn+mlp (split each layer into sub-objects in the manifest; see models/README.md)",
        });
        break;
      }
      // Otherwise this particular tab is just too small for a layer; skip it (it becomes a spare/replica).
      ti += 1;
      continue;
    }
    const hi = Math.min(N, lo + canHold);
    greedy.push({ holder: tab, layers: [lo, hi] });
    lo = hi;
    ti += 1;
  }
  const placedAll = lo >= N;
  const minStages = greedy.length;

  // ---- 4. Decide final S ----
  let S = minStages;
  const reqS = opts.stages;
  const liveTabs = sorted.length;
  if (reqS && Number.isInteger(reqS) && reqS >= 1) {
    // Honor the request only if feasible: at least the greedy minimum (so layers fit)
    // and no more than the number of distinct holder tabs we have.
    if (reqS >= minStages && reqS <= Math.max(minStages, liveTabs) && reqS <= N) {
      S = reqS;
    }
    // else: keep S = minStages (the fewest that fit). reason recorded below.
  }

  // ---- Rebuild stage layer ranges for exactly S stages, budget-proportional ----
  // Pick the S beefiest tabs as holders; distribute N layers ∝ their budgets, but never
  // more layers than a holder can actually fit (so we don't overcommit a small tab).
  const holders = sorted.slice(0, Math.min(S, sorted.length));
  let stages = distributeLayers(N, holders, {
    layerBytes, kvPerLayer, embedBytes, lmHeadBytes,
  });

  // If we couldn't place every layer (too few/small tabs), the tail is unassigned.
  let coveredHi = stages.length ? stages[stages.length - 1].layers[1] : 0;
  let unassigned = coveredHi < N ? [coveredHi, N] : null;

  // ---- 5. Pin embed (stage 0) and lm_head (last stage), wire neighbors ----
  const lastIdx = stages.length - 1;
  stages = stages.map((s, i) => ({
    stage: i,
    layers: s.layers,
    is_first: i === 0,
    is_last: i === lastIdx,
    holder: s.holder ? s.holder.node : null,
    replicas: [],
    prev_node: null,
    next_node: null,
    has_embed: i === 0,
    has_lm_head: i === lastIdx,
    budget_bytes: s.holder ? Math.round(s.holder.budget) : 0,
    model_id,
    manifest_ref,
  }));

  // Per-stage byte accounting + fit check (now that embed/lm_head are pinned).
  for (const s of stages) {
    const nLayers = s.layers[1] - s.layers[0];
    let wb = nLayers * layerBytes;
    if (s.has_embed) wb += embedBytes;
    if (s.has_lm_head) wb += lmHeadBytes;
    s.weight_bytes = wb;
    s.kv_bytes = nLayers * kvPerLayer;
    s.fits = s.budget_bytes === 0 ? false : (wb + s.kv_bytes) <= s.budget_bytes;
    if (!s.fits && s.holder) {
      splitHints.push({
        stage: s.stage,
        layers: s.layers,
        reason: `stage weights (${round2(wb / GiB)} GB) + KV (${round2(s.kv_bytes / GiB)} GB) exceed holder budget (${round2(s.budget_bytes / GiB)} GB)`,
        suggest: nLayers <= 1
          ? "parts:attn+mlp (split the single oversized layer)"
          : "more stages (raise S) or fewer layers on this holder",
      });
    }
  }

  // Wire activation neighbors (primary holders).
  for (let i = 0; i < stages.length; i++) {
    stages[i].prev_node = i > 0 ? stages[i - 1].holder : null;
    stages[i].next_node = i < stages.length - 1 ? stages[i + 1].holder : null;
  }

  // ---- 6. Replicas from leftover beefiest tabs, then spares ----
  const usedNodes = new Set(stages.map((s) => s.holder).filter(Boolean));
  const leftovers = sorted.filter((t) => !usedNodes.has(t.node));
  // Assign replicas round-robin to the stages whose layer range each leftover can hold,
  // preferring stages with the fewest replicas (even redundancy), beefiest tabs first.
  for (const tab of leftovers) {
    if (stages.every((s) => s.replicas.length >= R - 1)) break;
    // candidate stages this tab can hold and still needs replicas
    const fitStages = stages
      .filter((s) => s.replicas.length < R - 1 && stageFitsTab(s, tab.budget, layerBytes, kvPerLayer, embedBytes, lmHeadBytes))
      .sort((a, b) => a.replicas.length - b.replicas.length);
    if (fitStages.length === 0) continue; // can't replicate any needed stage => spare
    fitStages[0].replicas.push(tab.node);
    usedNodes.add(tab.node);
  }
  const spares = sorted.filter((t) => !usedNodes.has(t.node)).map((t) => t.node);

  // ---- feasibility + reason ----
  const everyStageHolder = stages.length > 0 && stages.every((s) => s.holder);
  const feasible = everyStageHolder && !unassigned && stages.every((s) => s.fits);
  let reason;
  if (!feasible) {
    const parts = [];
    if (cand.length === 0) parts.push("no tabs available");
    else if (stages.length === 0 && splitHints.length) parts.push("no tab can hold even one layer (see splitHints; need sub-layer split objects or bigger tabs)");
    else if (stages.length === 0) parts.push("no tab can hold a stage");
    if (unassigned) parts.push(`layers ${unassigned[0]}-${unassigned[1]} unplaced (need more/bigger tabs)`);
    if (everyStageHolder && stages.some((s) => !s.fits)) parts.push("some stage exceeds its holder's budget (see splitHints)");
    reason = parts.join("; ") || "infeasible";
  } else {
    const noteParts = [];
    if (reqS && S !== reqS) noteParts.push(`requested S=${reqS} not feasible; using S=${S} (fewest stages that fit)`);
    if (stages.some((s) => s.replicas.length < R - 1)) noteParts.push(`replication under target R=${R} on some stages (not enough spare tabs)`);
    reason = noteParts.join("; ") || "ok";
  }

  return {
    model_id,
    n_layers: N,
    S: stages.length,
    R,
    context: ctx,
    feasible,
    reason,
    stages,
    unassigned_layers: unassigned,
    spares,
    splitHints,
    tabsNeeded: tabsNeeded(model),
    plan_version: opts.plan_version ?? 1,
  };
}

// Distribute N layers across `holders` (already sorted beefiest-first) proportional to
// each holder's budget, capped so no holder gets more layers than it can hold. Reserves
// embed on the first holder and lm_head on the last so their budgets account for extras.
function distributeLayers(N, holders, { layerBytes, kvPerLayer, embedBytes, lmHeadBytes }) {
  const H = holders.length;
  if (H === 0) return [];
  const perLayerCost = layerBytes + kvPerLayer;

  // Capacity in layers per holder, after reserving extras on the ends.
  const cap = holders.map((h, i) => {
    let reserve = 0;
    if (i === 0) reserve += embedBytes;
    if (i === H - 1) reserve += lmHeadBytes;
    return Math.max(0, Math.floor((h.budget - reserve) / perLayerCost));
  });

  // Proportional target by budget, then clamp to capacity, then fix the total to N.
  const totalBudget = holders.reduce((s, h) => s + h.budget, 0) || 1;
  let target = holders.map((h) => (h.budget / totalBudget) * N);
  // Round to ints with largest-remainder so they sum to N.
  let want = largestRemainder(target, N);
  // Clamp to capacity.
  want = want.map((w, i) => Math.min(w, cap[i]));

  // Redistribute any shortfall (clamped layers) to holders with spare capacity, beefiest first.
  let assigned = want.reduce((s, w) => s + w, 0);
  let deficit = N - assigned;
  let guard = 0;
  while (deficit > 0 && guard++ < N * 2) {
    let moved = false;
    for (let i = 0; i < H && deficit > 0; i++) {
      if (want[i] < cap[i]) { want[i] += 1; deficit -= 1; moved = true; }
    }
    if (!moved) break; // no spare capacity anywhere => tail will be unassigned
  }

  // Build contiguous ranges, skipping holders that ended up with 0 layers.
  const stages = [];
  let lo = 0;
  for (let i = 0; i < H && lo < N; i++) {
    const cnt = want[i];
    if (cnt <= 0) continue;
    const hi = Math.min(N, lo + cnt);
    stages.push({ holder: holders[i], layers: [lo, hi] });
    lo = hi;
  }
  return stages;
}

function stageFitsTab(stage, budget, layerBytes, kvPerLayer, embedBytes, lmHeadBytes) {
  const nLayers = stage.layers[1] - stage.layers[0];
  let wb = nLayers * (layerBytes + kvPerLayer);
  if (stage.has_embed) wb += embedBytes;
  if (stage.has_lm_head) wb += lmHeadBytes;
  return budget >= wb;
}

// Largest-remainder rounding so a vector of floats rounds to ints summing to `total`.
function largestRemainder(floats, total) {
  const floor = floats.map((f) => Math.floor(f));
  let used = floor.reduce((s, x) => s + x, 0);
  let rem = total - used;
  const order = floats
    .map((f, i) => ({ i, frac: f - Math.floor(f) }))
    .sort((a, b) => b.frac - a.frac);
  const out = [...floor];
  for (let k = 0; k < order.length && rem > 0; k++) { out[order[k].i] += 1; rem -= 1; }
  return out;
}

const round1 = (x) => Math.round(x * 10) / 10;
const round2 = (x) => Math.round(x * 100) / 100;
