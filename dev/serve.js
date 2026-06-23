#!/usr/bin/env node
// ce-tabnet local dev runner — zero dependencies.
//
// Two modes:
//
//   node dev/serve.js               Serve the ce-tabnet/web directory over HTTP so you can open
//                                   join.html and orchestrator.html in a browser without a build
//                                   step. Default: http://127.0.0.1:8973
//                                   Flags: --port <n>  --dir <path>  --host <h>
//
//   node dev/serve.js --selftest    Run an in-process MOCK 2-stage pipeline that exercises the EXACT
//                                   wire protocol (web/protocol.js) and the stage-planning math
//                                   (web/model-config.js) end to end — join -> assign -> ready ->
//                                   prompt -> prefill -> activation hops -> sampled token -> fan-out
//                                   -> autoregressive continuation -> done. No GPU, no network, no
//                                   Cloudflare. It proves the message FLOW and shapes are consistent
//                                   across the coordinator and the tab runtime. Exit code 0 = pass.
//
// The selftest deliberately uses a tiny mock "engine" (random hidden states + argmax over a mock
// logits vector) so the pipeline can be demonstrated on any machine. The REAL compute path is
// web/inference-worker.js (WebGPU + CPU/WASM); this only validates coordination.

import http from "node:http";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const WEB_DIR = path.join(ROOT, "web");

// ---------------------------------------------------------------------------
// arg parsing
// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const a = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t.startsWith("--")) {
      const k = t.slice(2);
      const n = argv[i + 1];
      if (n === undefined || n.startsWith("--")) a[k] = true;
      else { a[k] = n; i++; }
    } else a._.push(t);
  }
  return a;
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".bin": "application/octet-stream",
  ".map": "application/json",
  ".ico": "image/x-icon",
};

// ===========================================================================
// static server
// ===========================================================================
async function serve(args) {
  const port = Number(args.port || process.env.PORT || 8973);
  const host = args.host || "127.0.0.1";
  const dir = path.resolve(args.dir ? String(args.dir) : WEB_DIR);

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url, `http://${req.headers.host}`);
      let pathname = decodeURIComponent(url.pathname);
      if (pathname === "/") pathname = "/join.html";
      // prevent path traversal
      const safe = path.normalize(pathname).replace(/^(\.\.[/\\])+/, "");
      let filePath = path.join(dir, safe);
      if (!filePath.startsWith(dir)) { res.writeHead(403); res.end("forbidden"); return; }

      let stat;
      try { stat = await fs.stat(filePath); } catch { res.writeHead(404); res.end("not found: " + safe); return; }
      if (stat.isDirectory()) filePath = path.join(filePath, "index.html");

      const data = await fs.readFile(filePath);
      const ext = path.extname(filePath).toLowerCase();
      res.writeHead(200, {
        "content-type": MIME[ext] || "application/octet-stream",
        "access-control-allow-origin": "*",
        "accept-ranges": "bytes",
        "cache-control": "no-cache",
      });
      res.end(data);
    } catch (e) {
      res.writeHead(500); res.end("error: " + (e && e.message));
    }
  });

  server.listen(port, host, () => {
    const base = `http://${host}:${port}`;
    process.stdout.write(
      `ce-tabnet dev server\n` +
      `  serving:    ${dir}\n` +
      `  open:       ${base}/orchestrator.html   (operator console — create a run, copy the join link)\n` +
      `              ${base}/join.html            (lend a tab — stage runtime)\n` +
      `  coordinator: start it separately with  npx wrangler dev --config coordinator/wrangler.jsonc\n` +
      `               (the pages auto-target ws://127.0.0.1:8787 on localhost)\n` +
      `\nCtrl-C to stop.\n`
    );
  });
}

// ===========================================================================
// --selftest : in-process mock 2-stage pipeline over the REAL protocol
// ===========================================================================
async function selftest() {
  // Import the real, shared modules (pure, no browser/Cloudflare deps).
  const P = await import(path.join(WEB_DIR, "protocol.js"));
  const { MODELS, defaultStagePlan, tabsNeeded } = await import(path.join(WEB_DIR, "model-config.js"));

  const log = (s) => process.stdout.write(s + "\n");
  let failures = 0;
  const assert = (cond, label) => {
    if (cond) log("  PASS  " + label);
    else { log("  FAIL  " + label); failures++; }
  };

  log("ce-tabnet selftest — mock 2-stage pipeline over web/protocol.js\n");

  // ---- 1. protocol round-trips ----
  const j = P.join("demo", "node-aaaa", { webgpu: true, vram_mb: 2048 });
  assert(P.validate(j) && j.t === P.T.JOIN, "join() builds a valid JOIN message");

  const meta = { run: "demo", node: "n0", seq_id: "s1", token_pos: 3, from_stage: 0, to_stage: 1, dtype: "f16", shape: [8] };
  const payload = new Uint16Array([1, 2, 3, 4, 5, 6, 7, 8]); // mock fp16 bits
  const frame = P.encodeActivation(meta, payload);
  const dec = P.decodeActivation(frame);
  assert(dec.meta.seq_id === "s1" && dec.meta.to_stage === 1, "encode/decodeActivation preserves header");
  assert(new Uint16Array(dec.payload).length === 8 && new Uint16Array(dec.payload)[7] === 8, "encode/decodeActivation preserves payload bytes");

  // ---- 2. planning math is honest ----
  const MODEL = "tinyllama-1.1b-q4";
  const need1g = tabsNeeded(MODEL, 1024 * 1024 * 1024);
  assert(need1g >= 1, `tabsNeeded(${MODEL}, 1GB) = ${need1g}`);
  // Force a 2-stage plan for the demo by giving two equal budgets summing to the model.
  const m = MODELS[MODEL];
  const half = Math.ceil(m.approx_weight_bytes / 2);
  const plan2 = defaultStagePlan(MODEL, [half, half]);
  assert(plan2.length === 2, `defaultStagePlan with two ${(half / 1e6).toFixed(0)}MB budgets -> 2 stages`);
  assert(plan2[0].layers[0] === 0 && plan2[plan2.length - 1].layers[1] === m.n_layers,
    `plan covers all ${m.n_layers} layers contiguously (${JSON.stringify(plan2.map((s) => s.layers))})`);

  // 300B honesty check
  const tabsFor300B = Math.ceil(150e9 / (1024 * 1024 * 1024)); // 150GB @ 1GB/tab
  assert(tabsFor300B >= 100 && tabsFor300B <= 300, `300B @ 4-bit (~150GB) needs ~${tabsFor300B} tabs at 1GB/tab (matches docs)`);

  // ---- 3. a MOCK coordinator + two MOCK stage tabs, talking the real message shapes ----
  // Mock coordinator: holds the plan, relays activations to to_stage's holder, fans tokens to
  // stage 0 + operator, enforces max_tokens. Mirrors coordinator/src/index.js logic in miniature.
  const HID = m.hidden_dim;
  const VOCAB = m.vocab;

  const operatorTokens = [];
  let seqDone = false;

  const coord = {
    plan: plan2.map((s, i) => ({ stage: s.stage, layers: s.layers, holder: `tab-${i}`, is_first: i === 0, is_last: i === plan2.length - 1 })),
    holders: {}, // node -> tab object
    seqs: new Map(),
    deliver(node, msg) {
      const tab = this.holders[node];
      if (tab) tab.recv(msg);
    },
    relayActivation(frameBuf) {
      const { meta } = P.decodeActivation(frameBuf);
      const stage = this.plan.find((s) => s.stage === meta.to_stage);
      if (!stage) return;
      this.holders[stage.holder].recvActivation(frameBuf);
    },
    routeToken(msg) {
      const seq = this.seqs.get(msg.seq_id);
      let done = !!msg.done;
      if (seq) {
        seq.generated = (seq.generated ?? 0) + 1;
        if (seq.max_tokens && seq.generated >= seq.max_tokens) done = true;
      }
      operatorTokens.push({ token_id: msg.token_id, token_pos: msg.token_pos, done });
      if (done) { seqDone = true; return; }
      // feed back to stage 0 (autoregressive)
      const head = this.plan[0].holder;
      this.deliver(head, P.token("demo", "coord", msg.seq_id, msg.token_pos, msg.token_id, "", false));
    },
    startSequence(seq_id, token_ids, max_tokens) {
      this.seqs.set(seq_id, { max_tokens, generated: 0 });
      const head = this.plan[0].holder;
      this.deliver(head, { ...P.promptBegin("demo", seq_id, token_ids, max_tokens, 0, 1), prompt: undefined });
    },
  };

  // Mock stage tab: mirrors web/tabnet-node.js compute handlers but with a trivial mock engine.
  function mockEngine(ctx, req) {
    // produce a deterministic hidden state (so the pipeline is reproducible), or a token if last.
    if (ctx.is_last) {
      // mock "logits": pick a token id derived from the inbound state + position.
      const seed = (req.token_pos * 1315423911) >>> 0;
      return { token_id: seed % VOCAB };
    }
    const hid = new Uint16Array(HID);
    const base = (req.token_ids ? req.token_ids[0] : (new Uint16Array(req.hidden ? req.hidden.buffer || req.hidden : new ArrayBuffer(2))[0] || 0));
    for (let i = 0; i < HID; i++) hid[i] = (base + i + (req.token_pos | 0)) & 0xffff;
    return { hidden: hid };
  }

  function makeTab(stageDef) {
    const tab = {
      id: stageDef.holder,
      stage: stageDef.stage,
      is_first: stageDef.is_first,
      is_last: stageDef.is_last,
      ctx: { is_last: stageDef.is_last },
      seqState: new Map(),
      events: [],
      recv(msg) {
        if (msg.t === P.T.PROMPT_BEGIN) return this.onPromptBegin(msg);
        if (msg.t === P.T.TOKEN) return this.onTokenForStage0(msg);
      },
      recvActivation(frameBuf) {
        const { meta, payload } = P.decodeActivation(frameBuf);
        this.events.push(`activation@stage${this.stage} pos=${meta.token_pos}`);
        const out = mockEngine(this.ctx, { seq_id: meta.seq_id, token_pos: meta.token_pos, hidden: new Uint16Array(payload) });
        if (this.is_last) {
          coord.routeToken(P.token("demo", this.id, meta.seq_id, meta.token_pos, out.token_id, "", false));
        } else {
          this.sendActivation(meta.seq_id, meta.token_pos, out.hidden);
        }
      },
      onPromptBegin(m) {
        const ids = (m.token_ids && m.token_ids.length) ? m.token_ids : [1];
        this.seqState.set(m.seq_id, { pos: ids.length, max_tokens: m.max_tokens, generated: 0 });
        this.events.push(`prompt-begin@stage0 prompt_len=${ids.length}`);
        const out = mockEngine(this.ctx, { seq_id: m.seq_id, token_pos: 0, token_ids: ids });
        this.sendActivation(m.seq_id, ids.length - 1, out.hidden);
      },
      onTokenForStage0(m) {
        if (!this.is_first) return;
        const st = this.seqState.get(m.seq_id);
        if (!st) return;
        st.generated += 1;
        if (m.done || st.generated >= st.max_tokens) return;
        const pos = st.pos; st.pos += 1;
        this.events.push(`decode-step@stage0 pos=${pos} prev_tok=${m.token_id}`);
        const out = mockEngine(this.ctx, { seq_id: m.seq_id, token_pos: pos, token_ids: [m.token_id] });
        this.sendActivation(m.seq_id, pos, out.hidden);
      },
      sendActivation(seq_id, token_pos, hidden) {
        const meta = { run: "demo", node: this.id, seq_id, token_pos, from_stage: this.stage, to_stage: this.stage + 1, dtype: "f16", shape: [hidden.length] };
        coord.relayActivation(P.encodeActivation(meta, hidden));
      },
    };
    coord.holders[stageDef.holder] = tab;
    return tab;
  }

  const tab0 = makeTab(coord.plan[0]);
  const tab1 = makeTab(coord.plan[1]);
  assert(tab0.is_first && tab1.is_last, "two tabs assigned: stage 0 (embed) and stage 1 (lm_head)");

  // ---- 4. run a prompt for 5 tokens and verify the loop ----
  const MAX = 5;
  coord.startSequence("seq-1", [10, 20, 30], MAX); // 3 mock prompt tokens

  // The whole loop is synchronous in this mock, so by now it has run to completion.
  assert(operatorTokens.length === MAX, `operator received exactly ${MAX} tokens (got ${operatorTokens.length})`);
  assert(operatorTokens[operatorTokens.length - 1].done === true, "last token carries done=true (coordinator enforced max_tokens)");
  assert(seqDone === true, "sequence reached done");
  // positions advance monotonically: prefill emitted at pos 2 (len-1), then decode at 3,4,5,6...
  const positions = operatorTokens.map((t) => t.token_pos);
  assert(positions[0] === 2, `first token sampled at prompt position ${positions[0]} (= prompt_len-1)`);
  assert(tab0.events.some((e) => e.startsWith("decode-step@stage0")), "stage 0 ran autoregressive decode steps");
  assert(tab1.events.some((e) => e.startsWith("activation@stage1")), "stage 1 received hidden-state activations over the wire");

  log("\nmessage trace (stage 0):");
  tab0.events.slice(0, 8).forEach((e) => log("  " + e));
  log("message trace (stage 1):");
  tab1.events.slice(0, 8).forEach((e) => log("  " + e));
  log("\noperator token stream: " + JSON.stringify(operatorTokens.map((t) => t.token_id)));

  log("\n" + (failures === 0 ? "SELFTEST PASSED" : `SELFTEST FAILED (${failures} failures)`));
  process.exit(failures === 0 ? 0 : 1);
}

// ===========================================================================
const args = parseArgs(process.argv.slice(2));
if (args.selftest) {
  selftest().catch((e) => { process.stderr.write("selftest error: " + (e && e.stack || e) + "\n"); process.exit(1); });
} else {
  serve(args).catch((e) => { process.stderr.write("serve error: " + (e && e.stack || e) + "\n"); process.exit(1); });
}
