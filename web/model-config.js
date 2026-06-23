// ce-tabnet model registry + stage planning math.
// Owner: Implementer 1. Imported by the DO (server-side) and the orchestrator UI.
// No I/O — pure data + planning functions. See docs/module-contract.md §D (Impl 1).

// Planning default: how many bytes of weights we budget per tab for WebGPU. ~1 GB is a
// conservative cross-device figure (phones ~0.5 GB, good laptops ~2 GB). See docs/architecture.md §4.
export const PER_TAB_WEIGHT_BYTES = 1.0 * 1024 * 1024 * 1024;

export const MODELS = {
  // --- the bundled small REAL model for the demo (produce its pack per models/README.md) ---
  "tinyllama-1.1b-q4": {
    arch: "llama",
    n_layers: 22,
    hidden_dim: 2048,
    n_heads: 32,
    n_kv_heads: 4,
    vocab: 32000,
    rope_theta: 10000.0,
    quant: "q4_0",
    approx_weight_bytes: 620_000_000,
    baseUrl: "https://r2.ce-net.com/tabnet/tinyllama-1.1b-q4/", // host with HTTP range support
    manifestRef: "manifest.json",
  },

  // --- larger targets (add packs as you build them; same code path) ---
  "llama-7b-q4": {
    arch: "llama", n_layers: 32, hidden_dim: 4096, n_heads: 32, n_kv_heads: 32,
    vocab: 32000, rope_theta: 10000.0, quant: "q4_0", approx_weight_bytes: 3_500_000_000,
    baseUrl: "https://r2.ce-net.com/tabnet/llama-7b-q4/", manifestRef: "manifest.json",
  },

  // --- the 300B-class target (see docs/scaling-to-300b.md). Pack must be produced before use. ---
  "llama-405b-q4": {
    arch: "llama", n_layers: 126, hidden_dim: 16384, n_heads: 128, n_kv_heads: 8,
    vocab: 128256, rope_theta: 500000.0, quant: "q4_0", approx_weight_bytes: 200_000_000_000,
    baseUrl: "https://r2.ce-net.com/tabnet/llama-405b-q4/", manifestRef: "manifest.json",
  },
};

// How many tabs (stages) a model needs given a per-tab budget. Honest arithmetic.
export function tabsNeeded(model_id, perTabBytes = PER_TAB_WEIGHT_BYTES) {
  const m = MODELS[model_id];
  if (!m) throw new Error(`unknown model ${model_id}`);
  return Math.max(1, Math.ceil(m.approx_weight_bytes / perTabBytes));
}

// Memory-weighted contiguous layer-range plan.
// tabBudgets: array of per-tab weight budgets in bytes (largest tab gets the most layers).
// Returns [{ stage, layers:[lo,hi) }] covering all n_layers. Used by the DO planner and the UI.
export function defaultStagePlan(model_id, tabBudgets) {
  const m = MODELS[model_id];
  if (!m) throw new Error(`unknown model ${model_id}`);
  const N = m.n_layers;
  // approximate per-layer weight cost (embed + lm_head are extra; the DO assigns those to beefy stages)
  const perLayer = m.approx_weight_bytes / N;

  // If no budgets supplied, fall back to the byte-budget count with equal stages.
  if (!tabBudgets || tabBudgets.length === 0) {
    const S = tabsNeeded(model_id);
    return equalStages(N, S);
  }

  // Greedy: walk tabs largest-first, pack layers until the tab's budget is exhausted.
  const sorted = [...tabBudgets].sort((a, b) => b - a);
  const plan = [];
  let lo = 0, stage = 0, ti = 0;
  while (lo < N) {
    const budget = sorted[Math.min(ti, sorted.length - 1)];
    const canHold = Math.max(1, Math.floor(budget / perLayer));
    const hi = Math.min(N, lo + canHold);
    plan.push({ stage, layers: [lo, hi] });
    lo = hi; stage += 1; ti += 1;
  }
  return plan;
}

function equalStages(N, S) {
  const plan = [];
  const per = Math.ceil(N / S);
  for (let stage = 0, lo = 0; lo < N; stage++, lo += per) {
    plan.push({ stage, layers: [lo, Math.min(N, lo + per)] });
  }
  return plan;
}
