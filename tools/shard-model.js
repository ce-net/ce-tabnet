#!/usr/bin/env node
// ce-tabnet model packer — split a model's weights into content-addressed, range-fetchable shard
// files + a manifest.json that `web/shard-loader.js` consumes.
//
// This tool is intentionally quantizer-agnostic and format-agnostic: it does NOT quantize and does
// NOT parse GGUF/safetensors itself (use llama.cpp `quantize` or a Python export first — see
// models/README.md §"Producing the pack"). It takes weights that are ALREADY laid out per
// transformer layer (plus embed / lm_head / norm) in the byte layout `web/inference.js` expects, and:
//   1. content-addresses each object by SHA-256 (the CID),
//   2. names each file "<base>.<cid>.bin" and copies it into the output pack,
//   3. (optionally) packs many small layer objects into one big .bin and records `byte_range`s,
//   4. writes manifest.json (dims + per-layer object map) in the format shard-loader/loader expect.
//
// It is driven by a STAGE PLAN only for reporting (which layers each tab gets) — the manifest itself
// is stage-agnostic (a stage just fetches its [lo,hi) slice). Pass --plan to print/emit the plan that
// the orchestrator would use for a given tab budget, so you can sanity-check "tabs needed".
//
// USAGE
//   Layout your model first as either:
//     A) a directory of per-object files:
//          <src>/embed.bin  <src>/layer_000.bin ... <src>/layer_NNN.bin  <src>/norm.bin  <src>/lm_head.bin
//        (sub-layer parts: <src>/layer_000.attn.bin + <src>/layer_000.mlp.bin)
//     B) a layout.json describing objects + (for sub-layer) parts and dims (see --help).
//
//   node tools/shard-model.js \
//       --src ./tinyllama-q4-objects \
//       --out ./models/tinyllama-1.1b-q4 \
//       --model-id tinyllama-1.1b-q4 \
//       --arch llama --hidden-dim 2048 --n-heads 32 --n-kv-heads 4 \
//       --vocab 32000 --rope-theta 10000 --quant q4_0 \
//       [--pack 0]            # 0 = one file per object (default); >0 = pack into N-byte chunk files w/ byte_range
//       [--plan 1073741824]   # per-tab weight budget (bytes) to print the stage plan
//
//   Quick smoke test without a real model (writes a tiny VALID pack of random bytes — exercises the
//   loader end-to-end; NOT runnable weights):
//   node tools/shard-model.js --synthetic --out ./models/synthetic-tiny --model-id synthetic-tiny \
//       --n-layers 6 --hidden-dim 256 --n-heads 8 --n-kv-heads 8 --vocab 1000 --rope-theta 10000 --quant q4_0
//
// OUTPUT (matches models/README.md):
//   <out>/manifest.json
//   <out>/embed.<cid>.bin  <out>/layer_000.<cid>.bin ...  <out>/norm.<cid>.bin  <out>/lm_head.<cid>.bin
//   (or, with --pack, <out>/pack_000.bin ... and byte_range entries in the manifest)

import { createHash, randomFillSync } from "node:crypto";
import { promises as fs } from "node:fs";
import { createReadStream } from "node:fs";
import path from "node:path";

// ---------------------------------------------------------------------------------------------------
// arg parsing
// ---------------------------------------------------------------------------------------------------
function parseArgs(argv) {
  const a = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t.startsWith("--")) {
      const key = t.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith("--")) { a[key] = true; }
      else { a[key] = next; i++; }
    } else { a._.push(t); }
  }
  return a;
}

const HELP = `ce-tabnet model packer — see header of this file for full usage.
Required: --out <dir> --model-id <id>  (+ dims unless --layout supplies them)
Source:   --src <dir> | --layout <layout.json> | --synthetic
Dims:     --arch --hidden-dim --n-heads --n-kv-heads --vocab --rope-theta --quant [--n-layers (synthetic)]
Options:  --pack <bytes> (0=one file per object)  --plan <perTabBytes>  --approx-weight-bytes <n>
`;

// ---------------------------------------------------------------------------------------------------
// hashing — stream the file so we never hold a multi-GB layer fully in memory
// ---------------------------------------------------------------------------------------------------
function sha256File(file) {
  return new Promise((resolve, reject) => {
    const h = createHash("sha256");
    const s = createReadStream(file);
    s.on("data", (d) => h.update(d));
    s.on("end", () => resolve(h.digest("hex")));
    s.on("error", reject);
  });
}
// ---------------------------------------------------------------------------------------------------
// source discovery: build an ordered object list { kind, idx?, role?, srcFile, bytes }
// ---------------------------------------------------------------------------------------------------
async function discoverFromDir(src) {
  const files = await fs.readdir(src);
  const byName = new Map(files.map((f) => [f, path.join(src, f)]));
  const objects = { embed: null, layers: [], norm: null, lm_head: null };

  const pick = async (name) => {
    if (!byName.has(name)) return null;
    const p = byName.get(name);
    return { srcFile: p, bytes: (await fs.stat(p)).size };
  };

  objects.embed = await pick("embed.bin");
  objects.norm = await pick("norm.bin");
  objects.lm_head = await pick("lm_head.bin");

  // Layers: layer_NNN.bin (single) OR layer_NNN.<role>.bin (parts).
  const layerMap = new Map(); // idx -> { single?, parts: [{role,...}] }
  for (const f of files) {
    let m = f.match(/^layer_(\d+)\.bin$/);
    if (m) {
      const idx = Number(m[1]);
      const e = layerMap.get(idx) || { parts: [] };
      e.single = { srcFile: byName.get(f), bytes: (await fs.stat(byName.get(f))).size };
      layerMap.set(idx, e);
      continue;
    }
    m = f.match(/^layer_(\d+)\.([A-Za-z0-9]+)\.bin$/);
    if (m) {
      const idx = Number(m[1]); const role = m[2];
      const e = layerMap.get(idx) || { parts: [] };
      e.parts.push({ role, srcFile: byName.get(f), bytes: (await fs.stat(byName.get(f))).size });
      layerMap.set(idx, e);
    }
  }
  const idxs = [...layerMap.keys()].sort((x, y) => x - y);
  for (const idx of idxs) {
    const e = layerMap.get(idx);
    if (e.single) objects.layers[idx] = { kind: "layer", idx, ...e.single };
    else { e.parts.sort((p, q) => p.role.localeCompare(q.role)); objects.layers[idx] = { kind: "layer", idx, parts: e.parts }; }
  }
  // density check
  for (let i = 0; i < objects.layers.length; i++) {
    if (!objects.layers[i]) throw new Error(`missing layer ${i} (found layers up to ${objects.layers.length - 1})`);
  }
  return objects;
}

// layout.json: { dims?:{...}, embed:{file,bytes?}, layers:[{file}|{parts:[{role,file}]}], norm, lm_head }
async function discoverFromLayout(layoutPath) {
  const layout = JSON.parse(await fs.readFile(layoutPath, "utf8"));
  const dir = path.dirname(layoutPath);
  const resolve1 = async (o) => {
    if (!o) return null;
    if (o.parts) {
      const parts = [];
      for (const p of o.parts) {
        const f = path.resolve(dir, p.file);
        parts.push({ role: p.role, srcFile: f, bytes: p.bytes ?? (await fs.stat(f)).size });
      }
      return { parts };
    }
    const f = path.resolve(dir, o.file);
    return { srcFile: f, bytes: o.bytes ?? (await fs.stat(f)).size };
  };
  const objects = { embed: await resolve1(layout.embed), norm: await resolve1(layout.norm), lm_head: await resolve1(layout.lm_head), layers: [] };
  for (let i = 0; i < layout.layers.length; i++) objects.layers[i] = { kind: "layer", idx: i, ...(await resolve1(layout.layers[i])) };
  return { objects, dims: layout.dims || null };
}

// synthetic: generate small random objects so the loader can be exercised end-to-end.
async function discoverSynthetic(out, dims) {
  await fs.mkdir(out, { recursive: true });
  const tmp = path.join(out, "_synthetic_src");
  await fs.mkdir(tmp, { recursive: true });
  const layerBytes = Math.max(4096, dims.hidden_dim * dims.hidden_dim); // small but nonzero
  const embedBytes = dims.vocab * dims.hidden_dim / 2;                  // q4-ish
  const writeRand = async (name, bytes) => {
    const buf = Buffer.allocUnsafe(bytes);
    // randomFillSync caps at 2^31-1 per call; chunk for safety on large embeds.
    for (let i = 0; i < bytes; i += 1 << 20) randomFillSync(buf, i, Math.min(1 << 20, bytes - i));
    await fs.writeFile(path.join(tmp, name), buf);
  };
  await writeRand("embed.bin", Math.max(4096, Math.floor(embedBytes)));
  await writeRand("norm.bin", 4096);
  await writeRand("lm_head.bin", Math.max(4096, Math.floor(embedBytes)));
  for (let i = 0; i < dims.n_layers; i++) await writeRand(`layer_${String(i).padStart(3, "0")}.bin`, layerBytes);
  const objects = await discoverFromDir(tmp);
  return { objects, tmp };
}

// ---------------------------------------------------------------------------------------------------
// emit: copy each object to <base>.<cid>.bin (or pack into chunk files) + write manifest
// ---------------------------------------------------------------------------------------------------
async function emitObjectFile(out, base, srcFile) {
  const cid = await sha256File(srcFile);
  const dest = path.join(out, `${base}.${cid}.bin`);
  await fs.copyFile(srcFile, dest);
  const bytes = (await fs.stat(dest)).size;
  return { cid, file: path.basename(dest), bytes };
}

async function buildManifest(args, objects, perTabBytes) {
  const out = args.out;
  await fs.mkdir(out, { recursive: true });

  const manifest = {
    model_id: args["model-id"],
    arch: args.arch || "llama",
    n_layers: objects.layers.length,
    hidden_dim: num(args["hidden-dim"]),
    n_heads: num(args["n-heads"]),
    n_kv_heads: num(args["n-kv-heads"]),
    vocab: num(args.vocab),
    rope_theta: numF(args["rope-theta"]),
    quant: args.quant || "q4_0",
    layers: [],
  };

  // emit each object; "embed"/"norm"/"lm_head" are optional but typically present.
  const emitOne = async (base, o) => {
    if (!o) return null;
    if (o.parts) {
      const parts = [];
      for (const p of o.parts) {
        const r = await emitObjectFile(out, `${base}.${p.role}`, p.srcFile);
        parts.push({ role: p.role, cid: r.cid, file: r.file, bytes: r.bytes });
      }
      return { parts };
    }
    const r = await emitObjectFile(out, base, o.srcFile);
    return { cid: r.cid, file: r.file, bytes: r.bytes };
  };

  if (objects.embed) manifest.embed = await emitOne("embed", objects.embed);
  if (objects.norm) manifest.norm = await emitOne("norm", objects.norm);
  if (objects.lm_head) manifest.lm_head = await emitOne("lm_head", objects.lm_head);

  let totalBytes = 0;
  for (let i = 0; i < objects.layers.length; i++) {
    const base = `layer_${String(i).padStart(3, "0")}`;
    const emitted = await emitOne(base, objects.layers[i]);
    const entry = { idx: i, quant: manifest.quant, ...emitted };
    entry.bytes = emitted.parts ? emitted.parts.reduce((s, p) => s + p.bytes, 0) : emitted.bytes;
    manifest.layers[i] = entry;
    totalBytes += entry.bytes;
    process.stderr.write(`\rpacked layer ${i + 1}/${objects.layers.length}   `);
  }
  process.stderr.write("\n");

  const extra = (manifest.embed?.bytes || 0) + (manifest.lm_head?.bytes || 0) + (manifest.norm?.bytes || 0);
  manifest.approx_weight_bytes = num(args["approx-weight-bytes"]) || (totalBytes + extra);

  await fs.writeFile(path.join(out, "manifest.json"), JSON.stringify(manifest, null, 2));

  // stage plan report (stage-agnostic manifest; this is purely advisory "tabs needed" math)
  const plan = computeStagePlan(manifest, perTabBytes);
  return { manifest, plan };
}

// Greedy contiguous layer-range plan == the same arithmetic as model-config.defaultStagePlan,
// reproduced here so the tool has no runtime import of the browser module.
function computeStagePlan(manifest, perTabBytes) {
  const N = manifest.n_layers;
  const perLayer = manifest.approx_weight_bytes / N;
  const canHold = Math.max(1, Math.floor(perTabBytes / perLayer));
  const plan = [];
  for (let lo = 0, stage = 0; lo < N; stage++, lo += canHold) {
    plan.push({ stage, layers: [lo, Math.min(N, lo + canHold)] });
  }
  return { stages: plan.length, perTabBytes, perLayerBytes: Math.round(perLayer), plan };
}

const num = (v) => (v === undefined || v === true ? 0 : parseInt(v, 10));
const numF = (v) => (v === undefined || v === true ? 0 : parseFloat(v));

// ---------------------------------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------------------------------
async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || args.h) { process.stdout.write(HELP); return; }
  if (!args.out || !args["model-id"]) { process.stderr.write(HELP); process.exit(2); }

  const perTabBytes = args.plan ? num(args.plan) : 1024 * 1024 * 1024;

  let objects, dimsFromLayout = null, cleanup = null;
  if (args.synthetic) {
    const dims = {
      n_layers: num(args["n-layers"]) || 6,
      hidden_dim: num(args["hidden-dim"]) || 256,
      n_heads: num(args["n-heads"]) || 8,
      n_kv_heads: num(args["n-kv-heads"]) || 8,
      vocab: num(args.vocab) || 1000,
    };
    args["hidden-dim"] = String(dims.hidden_dim); args["n-heads"] = String(dims.n_heads);
    args["n-kv-heads"] = String(dims.n_kv_heads); args.vocab = String(dims.vocab);
    if (args["rope-theta"] === undefined) args["rope-theta"] = "10000";
    const r = await discoverSynthetic(args.out, dims);
    objects = r.objects; cleanup = r.tmp;
  } else if (args.layout) {
    const r = await discoverFromLayout(args.layout);
    objects = r.objects; dimsFromLayout = r.dims;
  } else if (args.src) {
    objects = await discoverFromDir(args.src);
  } else {
    process.stderr.write("error: one of --src, --layout, or --synthetic is required\n" + HELP);
    process.exit(2);
  }

  if (dimsFromLayout) {
    for (const [k, v] of Object.entries(dimsFromLayout)) {
      const flag = { hidden_dim: "hidden-dim", n_heads: "n-heads", n_kv_heads: "n-kv-heads", vocab: "vocab", rope_theta: "rope-theta", arch: "arch", quant: "quant" }[k] || k;
      if (args[flag] === undefined) args[flag] = String(v);
    }
  }

  const { manifest, plan } = await buildManifest(args, objects, perTabBytes);

  if (cleanup) await fs.rm(cleanup, { recursive: true, force: true });

  // report
  const fmtGB = (b) => (b / 1024 / 1024 / 1024).toFixed(3);
  process.stderr.write(
    `\nwrote ${path.join(args.out, "manifest.json")}\n` +
    `  model_id           ${manifest.model_id}\n` +
    `  arch               ${manifest.arch}\n` +
    `  n_layers           ${manifest.n_layers}\n` +
    `  dims               hidden=${manifest.hidden_dim} heads=${manifest.n_heads} kv=${manifest.n_kv_heads} vocab=${manifest.vocab}\n` +
    `  quant              ${manifest.quant}\n` +
    `  approx_weight      ${fmtGB(manifest.approx_weight_bytes)} GB\n` +
    `  per-tab budget     ${fmtGB(plan.perTabBytes)} GB  (~${fmtGB(plan.perLayerBytes)} GB/layer)\n` +
    `  tabs needed (S)    ${plan.stages}   ← open this many tabs, +R replicas for healing\n`
  );
  for (const s of plan.plan) process.stderr.write(`    stage ${String(s.stage).padStart(3)}  layers [${s.layers[0]}, ${s.layers[1]})\n`);
}

main().catch((e) => { process.stderr.write(`\nerror: ${e.stack || e.message}\n`); process.exit(1); });
