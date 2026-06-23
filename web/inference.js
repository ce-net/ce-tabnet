// ce-tabnet stage inference — INTERFACE REFERENCE ONLY (the shared contract shape).
// Owner: Implementer 5. See docs/module-contract.md §D (Impl 5) and docs/architecture.md §8.
//
// NOTE: This file documents the stage-engine interface. The REAL, runnable engine
// (WebGPU kernels + a real CPU/WASM fallback) is implemented in `inference-worker.js`,
// which exports the same { initStage, forward, freeStage } and is what tabnet-node.js
// imports. The bodies below intentionally throw so nothing accidentally runs the stub.
//
// SHARED INTERFACE (agree verbatim with shard-loader.js, Implementer 6 — see its header for `weights`):
//   initStage({ arch, dims, layers:[lo,hi], is_first, is_last, weights }) -> StageCtx
//   forward(ctx, { seq_id, token_pos, hidden, token_ids }) ->
//        { hidden: Float16Array }   when !is_last   (send to next stage)
//        { token_id: number }       when  is_last   (sampled token, send back)
//   freeStage(ctx)
//
// KV cache lives INSIDE StageCtx (per-stage, never crosses the network) — see architecture.md §2.3.
// Activation tensors are framed by protocol.encodeActivation/decodeActivation (Implementer 1).

export async function initStage(spec) {
  const gpu = navigator.gpu ? await initWebGPU() : null;
  const ctx = {
    ...spec,
    backend: gpu ? "webgpu" : "wasm",
    device: gpu?.device ?? null,
    kv: new Map(),          // seq_id -> per-layer KV buffers (WebGPU buffers or typed arrays)
    pipelines: null,        // compiled WGSL pipelines (rmsnorm, q4_matmul, rope, attn, swiglu, sample)
  };
  // TODO (Impl 5):
  //   - upload spec.weights.layers[*].buf into WebGPU storage buffers (chunk under maxBufferSize)
  //   - if is_first: upload embed; if is_last: upload norm + lm_head
  //   - compile WGSL kernels into ctx.pipelines
  //   - build the WASM/CPU fallback path when backend === "wasm"
  ctx.ready = true;
  return ctx;
}

export async function forward(ctx, req) {
  // TODO (Impl 5): the real decoder-stack compute. Sketch:
  //   let h;
  //   if (ctx.is_first) h = embed(ctx, req.token_ids ?? [lastTokenOf(req)]);
  //   else              h = req.hidden;                         // from previous stage
  //   for each layer in ctx.layers:
  //     h = rmsnorm(h); qkv; rope(token_pos); attn(with ctx.kv[seq_id]); o_proj;
  //     h = h + residual; rmsnorm; swiglu_mlp; h = h + residual;
  //   if (ctx.is_last) { h = norm(h); logits = lm_head(h);
  //                      return { token_id: sample(logits, req.temperature, req.top_p) }; }
  //   return { hidden: toFloat16(h) };
  throw new Error("inference.forward not yet implemented (Implementer 5)");
}

export function freeStage(ctx) { /* TODO: destroy WebGPU buffers, clear KV. */ ctx.ready = false; }

async function initWebGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  return { adapter, device };
}
