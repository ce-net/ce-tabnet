# Building a ce-tabnet model pack

A **model pack** lets any browser tab fetch **just one layer range** of the model. It is the content-addressed,
range-fetchable layout that makes pipeline sharding (and healing) possible. Owner: Implementer 6.

## Layout

```
<model_id>/
  manifest.json            # the index (dims + per-layer object map)
  embed.<cid>.bin          # token embedding weights (Q4) — fetched by stage 0
  layer_000.<cid>.bin      # transformer layer 0 weights (Q4)
  layer_001.<cid>.bin
  ...
  layer_NNN.<cid>.bin
  norm.<cid>.bin           # final norm — fetched by the last stage
  lm_head.<cid>.bin        # logits projection — fetched by the last stage
```

Each `.bin` is content-addressed: `<cid>` is the SHA-256 (or CE blob CID) of the file. The loader verifies the
hash on arrival — content addressing IS integrity (same rule as CE blobs in `PLAN/09-hospital-inference.md`).

## manifest.json

```json
{
  "model_id": "tinyllama-1.1b-q4",
  "arch": "llama",
  "n_layers": 22,
  "hidden_dim": 2048,
  "n_heads": 32,
  "n_kv_heads": 4,
  "vocab": 32000,
  "rope_theta": 10000.0,
  "quant": "q4_0",
  "approx_weight_bytes": 620000000,
  "embed":   { "cid": "...", "bytes": 131072000, "shape": [32000, 2048] },
  "layers": [
    { "idx": 0, "cid": "...", "bytes": 22000000, "quant": "q4_0" },
    { "idx": 1, "cid": "...", "bytes": 22000000, "quant": "q4_0" }
  ],
  "norm":    { "cid": "...", "bytes": 8192 },
  "lm_head": { "cid": "...", "bytes": 131072000, "shape": [32000, 2048] }
}
```

For very small devices, a layer may be split into sub-objects (`attn`, `mlp`) so a phone can hold a fraction of
a layer — add `"parts": [{role:"attn",cid,bytes},{role:"mlp",cid,bytes}]` instead of a single `cid`.

## Producing the pack (recipe)

1. **Quantize** the model to a browser-friendly 4-bit scheme (Q4_0 / Q4_K style). Reuse a known quantizer
   (llama.cpp `quantize`, or a Python export) — do NOT write a quantizer.
2. **Split per layer:** export each transformer layer's tensors into its own file in the WebGPU-ready buffer
   layout `inference.js` expects (agree the exact tensor order in the shared comment block with Implementer 5).
3. **Content-address:** SHA-256 each file, name it `<base>.<cid>.bin`.
4. **Write `manifest.json`** as above.
5. **Host with HTTP range support:** R2, S3, a CE blob gateway, or even `python3 -m http.server` (it supports
   `Range`). Put the base URL + `manifestRef` into `web/model-config.js`.

## CE-native variant

Publish each object as a CE blob (`ce put`/`put_object`) and host a CE blob gateway; the loader's `baseUrl`
then points at the gateway and `cid` is the CE object CID. Replicas fetch ranges peer-to-peer (ce-pin), exactly
like the hospital-infer weight distribution. The public demo can just use R2.
