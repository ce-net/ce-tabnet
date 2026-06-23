// ce-tabnet coordinator — Cloudflare Worker + Durable Object "RunCoordinator".
//
// One DO instance == one Run (one model instance, addressed by idFromName(runId)).
// It owns the live pipeline topology (stage -> tab(s)), assigns stages to tabs on join
// by device capability, relays activation/token/heartbeat messages along the pipeline,
// heals on tab dropout, and exposes a small HTTP API to start a run and stream tokens out.
//
// Modern Workers/DO patterns:
//   - `export default { fetch }` Worker entry (thin router).
//   - `RunCoordinator extends DurableObject`.
//   - WebSocket **Hibernation API**: state.acceptWebSocket(ws) + webSocketMessage/Close/Error
//     handlers, so the DO can evict from memory between messages and survive without burning
//     wall-clock — essential when hundreds of tabs hold idle-ish sockets.
//   - DO storage persists the run config + plan so a Run is resumable across hibernation.
//   - state.getTags()/serializeAttachment carry per-socket identity across hibernation.
//
// Shared, read-only imports (bundled by wrangler from ../../web/):
//   protocol.js     — wire message constructors + validate() (single source of truth, §B)
//   model-config.js — model registry + memory-weighted defaultStagePlan (§C planner)
//
// See docs/architecture.md §9 and docs/module-contract.md §B/§C.

import { DurableObject } from "cloudflare:workers";
import * as P from "../../web/protocol.js";
import {
  MODELS,
  defaultStagePlan,
  tabsNeeded,
  PER_TAB_WEIGHT_BYTES,
} from "../../web/model-config.js";

// Same staleness window + sweep cadence as ce-hub (web/ce-hub/src/main.rs) and node.html (10s hb).
const STALE_MS = 35_000;
const SWEEP_MS = 5_000;

// ---------------------------------------------------------------------------
// Worker entry — stateless router. Maps a run id to its DO and forwards.
//   GET  /run/:id/ws?role=stage|operator  -> WebSocket upgrade (hibernatable)
//   POST /run/:id                          -> create-run (JSON body)
//   POST /run/:id/prompt                   -> submit a prompt over HTTP (returns seq_id)
//   GET  /run/:id/tokens?seq_id=...        -> SSE stream of generated tokens (HTTP egress)
//   GET  /run/:id/state                    -> JSON run-state snapshot
// ---------------------------------------------------------------------------
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "content-type",
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });

    if (url.pathname === "/" || url.pathname === "/health") {
      return json({ ok: true, service: "ce-tabnet-coordinator", models: Object.keys(MODELS) });
    }

    // /run/:id  with optional trailing /ws | /state | /prompt | /tokens
    const m = url.pathname.match(/^\/run\/([^/]+)(\/ws|\/state|\/prompt|\/tokens)?$/);
    if (!m) return json({ error: "not_found" }, 404);

    const runId = m[1];
    const id = env.RUN.idFromName(runId);
    const stub = env.RUN.get(id);
    // The DO needs to know its own run id for outgoing messages; pass it via header
    // (idFromName is one-way, the DO can't recover the name otherwise).
    const fwd = new Request(request, request);
    fwd.headers.set("x-ce-run", runId);
    return stub.fetch(fwd);
  },
};

function json(obj, status = 200, extra = {}) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", ...CORS, ...extra },
  });
}

// ---------------------------------------------------------------------------
// RunCoordinator Durable Object — one per run id.
// ---------------------------------------------------------------------------
export class RunCoordinator extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.ctx = ctx;
    this.env = env;

    // ---- live, in-memory topology (rebuilt lazily from storage after hibernation) ----
    // node -> { caps, stage, role, ready, lastSeen, weight_bytes }
    this.tabs = new Map();
    // operator HTTP/SSE token sinks: seq_id -> Set<WritableStreamDefaultWriter>
    this.tokenSinks = new Map();
    this.config = null; // { run, model_id, S, R, fastpath, microbatch, plan_version }
    this.plan = []; // [{ stage, layers:[lo,hi], holders:[node], status }]
    this.seqs = new Map(); // seq_id -> { status, token_pos, max_tokens, ... }
    this.hydrated = false;

    // Re-hydrate persisted config/plan/tab-meta synchronously inside the constructor's
    // blockConcurrencyWhile so the first request sees a consistent world after hibernation.
    this.ctx.blockConcurrencyWhile(async () => {
      this.config = (await this.ctx.storage.get("config")) ?? null;
      this.plan = (await this.ctx.storage.get("plan")) ?? [];
      const tabMeta = (await this.ctx.storage.get("tabMeta")) ?? {};
      for (const [node, meta] of Object.entries(tabMeta)) this.tabs.set(node, meta);
      this.hydrated = true;
      // Reattach hibernated sockets to their node identity.
      for (const ws of this.ctx.getWebSockets()) {
        const att = safeAttachment(ws);
        if (att?.node && this.tabs.has(att.node)) this.tabs.get(att.node).ws = ws;
      }
    });
  }

  // ===========================================================================
  // HTTP / WS entry
  // ===========================================================================
  async fetch(request) {
    const url = new URL(request.url);
    const runId = request.headers.get("x-ce-run") || url.pathname.split("/")[2] || "run";
    if (!this.config?.run) {
      // Stash the run id so outgoing messages can carry it (idFromName is one-way).
      this._runId = runId;
    }
    this._runId = runId;

    // WebSocket upgrade (hibernatable)
    if (request.headers.get("Upgrade") === "websocket") {
      const role = url.searchParams.get("role") === "operator" ? "operator" : "stage";
      const pair = new WebSocketPair();
      const [client, server] = [pair[0], pair[1]];
      // Hibernation API: accept with a tag so we can find sockets by role/run later.
      this.ctx.acceptWebSocket(server, [role, `run:${runId}`]);
      server.serializeAttachment({ role, run: runId, node: null });
      await this._ensureAlarm();
      return new Response(null, { status: 101, webSocket: client });
    }

    // POST /run/:id            -> create-run
    if (request.method === "POST" && url.pathname.endsWith(`/run/${runId}`)) {
      const body = await request.json().catch(() => ({}));
      this.createRun({ ...body, run: runId });
      return json({ ok: true, run: runId, state: this.snapshot() });
    }

    // POST /run/:id/prompt     -> submit prompt over HTTP, returns seq_id
    if (request.method === "POST" && url.pathname.endsWith("/prompt")) {
      const body = await request.json().catch(() => ({}));
      const seq_id = body.seq_id || `seq-${crypto.randomUUID()}`;
      const ok = this.startSequence({ ...body, run: runId, seq_id });
      if (!ok) return json({ error: "run_not_ready", detail: "no full pipeline" }, 409);
      return json({ ok: true, seq_id });
    }

    // GET /run/:id/tokens?seq_id=...  -> SSE token egress (HTTP, no WS needed)
    if (request.method === "GET" && url.pathname.endsWith("/tokens")) {
      const seq_id = url.searchParams.get("seq_id") || "*";
      return this._sseTokens(runId, seq_id);
    }

    // GET /run/:id/state       -> JSON snapshot
    if (url.pathname.endsWith("/state")) return json(this.snapshot());

    return json({ ok: true, run: runId });
  }

  // ===========================================================================
  // WebSocket Hibernation handlers (replace addEventListener)
  // ===========================================================================
  async webSocketMessage(ws, raw) {
    // Activations may arrive as binary frames (hot path); everything else is JSON text.
    if (raw instanceof ArrayBuffer) return this._onBinaryActivation(ws, raw);

    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    if (!P.validate(msg)) return;

    const att = safeAttachment(ws);
    const role = att?.role ?? "stage";

    switch (msg.t) {
      case P.T.JOIN:
        return this._onJoin(ws, msg);
      case P.T.READY:
        return this._onReady(msg);
      case P.T.HB:
        return this._touch(msg.node);
      case P.T.ACTIVATION:
        return this.relayActivation(msg);
      case P.T.TOKEN:
        return this.routeToken(msg);
      case P.T.SIGNAL:
        return this.relaySignal(msg);
      case P.T.METRICS:
        return this._onMetrics(msg);
      case P.T.LEAVE:
        return this._dropNode(msg.node);
      case P.T.CREATE_RUN:
        return this.createRun(msg);
      case P.T.PROMPT:
        return this.startSequence(msg);
      case P.T.CANCEL:
        return this._cancelSeq(msg.seq_id);
      case P.T.SUBSCRIBE_TOKENS:
        // operator over WS: register it and push the current snapshot.
        this._markOperator(ws);
        return this._send(ws, this.snapshot());
      default:
        return;
    }
  }

  async webSocketClose(ws) {
    const att = safeAttachment(ws);
    if (att?.role === "operator") {
      // nothing to heal for operators
      return;
    }
    if (att?.node) this._dropNode(att.node);
  }

  async webSocketError(ws) {
    const att = safeAttachment(ws);
    if (att?.node) this._dropNode(att.node);
  }

  // ===========================================================================
  // core logic
  // ===========================================================================

  // create-run: define the model instance + an initial (capability-blind) plan.
  // Idempotent — re-issuing with the same model is a no-op except plan_version bump
  // only when device set warrants a replan.
  createRun(body) {
    const m = MODELS[body.model_id];
    if (!m) {
      this._broadcastError("unknown_model", `no such model: ${body.model_id}`);
      return;
    }
    const run = body.run || this._runId;
    this.config = {
      run,
      model_id: body.model_id,
      S: body.stages ?? null, // resolved by planStages() from live caps
      R: body.replicas ?? 1,
      fastpath: !!body.fastpath,
      microbatch: body.microbatch ?? 8,
      cap: body.cap ?? null,
      plan_version: (this.config?.plan_version ?? 0) + 1,
    };
    this.ctx.storage.put("config", this.config);
    // (Re)compute the plan against whatever tabs are currently present.
    this.planStages();
  }

  // Memory-weighted PlacementPlanner: split the model's n_layers across the live tabs,
  // largest-budget tabs get the most layers (docs/module-contract.md §C step 1-6).
  planStages() {
    if (!this.config) return;
    const m = MODELS[this.config.model_id];
    const live = this._liveTabs();

    // Budget per tab: prefer reported WebGPU vram; else half of RAM for non-WebGPU CPU tabs.
    const budgetOf = (t) => {
      const v = t.caps?.vram_mb ? t.caps.vram_mb * 1024 * 1024 : 0;
      if (v > 0) return v;
      const ram = (t.caps?.ram_gb ?? 1) * 1024 * 1024 * 1024;
      return Math.max(256 * 1024 * 1024, ram * 0.5);
    };

    // Sort tabs beefiest-first; the planner assigns contiguous layer ranges greedily.
    const ranked = [...live].sort((a, b) => budgetOf(b[1]) - budgetOf(a[1]));
    const budgets = ranked.map(([, t]) => budgetOf(t));

    // Decide S. Operator-requested S wins if feasible; else derive from budgets.
    let basePlan;
    if (budgets.length === 0) {
      // No tabs yet: show the theoretical plan (equal stages) so the console can render math.
      const S = this.config.S ?? tabsNeeded(this.config.model_id);
      basePlan = defaultStagePlan(this.config.model_id, []).slice(0, S);
      // defaultStagePlan([]) already returns S-equal stages; trust it.
      basePlan = defaultStagePlan(this.config.model_id, []);
    } else {
      basePlan = defaultStagePlan(this.config.model_id, budgets);
    }

    const S = basePlan.length;
    this.config.S = S;

    // Merge with any existing plan to preserve holders/status across replans.
    const prevByStage = new Map(this.plan.map((s) => [s.stage, s]));
    this.plan = basePlan.map((s) => {
      const prev = prevByStage.get(s.stage);
      return {
        stage: s.stage,
        layers: s.layers,
        holders: prev?.holders ?? [],
        status: prev?.holders?.length ? prev.status : "recruiting",
      };
    });

    // Assign the ranked tabs to stages in order (beefiest tab -> stage 0, which also
    // carries the embedding; the last assigned tab carries lm_head + final norm).
    const assignedNodes = new Set(
      this.plan.flatMap((s) => s.holders),
    );
    let si = 0;
    for (const [node, t] of ranked) {
      if (assignedNodes.has(node)) continue;
      // find the first stage that still needs a primary holder
      while (si < this.plan.length && this.plan[si].holders.length >= this.config.R) si++;
      if (si >= this.plan.length) {
        // surplus tab -> spare (available for healing / replica duty)
        t.stage = null;
        t.role = "spare";
        this._sendTo(node, P.welcome(this.config.run, node, "spare", this.config.plan_version));
        continue;
      }
      const stage = this.plan[si];
      const isReplica = stage.holders.length >= 1;
      stage.holders.push(node);
      t.stage = stage.stage;
      t.role = isReplica ? "replica" : "stage";
      if (!isReplica && stage.status === "recruiting") stage.status = "loading";
      this._emitAssign(node, stage, isReplica);
    }

    this._persistPlan();
    this._recomputeNeighbors();
    this._broadcastState();
  }

  // Give a single joining tab a stage/replica/spare role (incremental assign, no full replan).
  assign(node) {
    const t = this.tabs.get(node);
    if (!t || !this.config) {
      if (t) this._sendTo(node, P.welcome(this._runId, node, "spare", this.config?.plan_version ?? 0));
      return;
    }
    // Find the first stage lacking a primary holder, else lacking a replica (< R), else spare.
    let target = this.plan.find((s) => s.holders.length === 0);
    if (!target) target = this.plan.find((s) => s.holders.length < this.config.R);
    if (!target) {
      t.stage = null;
      t.role = "spare";
      this._sendTo(node, P.welcome(this.config.run, node, "spare", this.config.plan_version));
      this._broadcastState();
      return;
    }
    const isReplica = target.holders.length >= 1;
    target.holders.push(node);
    t.stage = target.stage;
    t.role = isReplica ? "replica" : "stage";
    if (!isReplica && target.status === "recruiting") target.status = "loading";
    this._emitAssign(node, target, isReplica);
    this._persistPlan();
    this._recomputeNeighbors();
    this._broadcastState();
  }

  // Path A: forward an activation (hidden state) to the primary holder of the next stage.
  relayActivation(msg) {
    const stage = this.plan.find((s) => s.stage === msg.to_stage);
    if (!stage || stage.holders.length === 0) {
      // Orphaned next stage: park nothing — healing will re-route; sender will retry.
      this._broadcastError("stage_orphaned", `to_stage ${msg.to_stage} has no holder`);
      return;
    }
    const target = stage.holders[0]; // primary
    // Forward verbatim; the DO is a pass-through relay for activations.
    this._sendTo(target, msg);
  }

  // Path B: relay WebRTC offer/answer/ice to a specific peer node.
  relaySignal(msg) {
    if (!msg.to_node) return;
    this._sendTo(
      msg.to_node,
      P.signal(this.config?.run ?? this._runId, msg.node, msg.to_node, msg.kind, msg.payload),
    );
  }

  // Last stage sampled a token: fan it to (a) operator token sinks, (b) stage 0 for next decode.
  routeToken(msg) {
    const seq = this.seqs.get(msg.seq_id);
    let done = !!msg.done;
    if (seq) {
      seq.token_pos = msg.token_pos;
      seq.generated = (seq.generated ?? 0) + 1;
      // Authoritative stop: the coordinator owns max_tokens so generation halts even if a
      // tab misbehaves. Mark this token as the final one when we reach the cap.
      if (seq.max_tokens && seq.generated >= seq.max_tokens) done = true;
      if (done) seq.status = "done";
      else if (seq.status === "prefill") seq.status = "decoding";
    }
    // (a) operator egress (WS operators + SSE sinks)
    const out = P.token(
      this.config?.run ?? this._runId,
      msg.seq_id,
      msg.token_pos,
      msg.token_id,
      msg.text,
      done,
    );
    this._fanToOperators(out);
    this._fanToSinks(msg.seq_id, out);

    // (b) feed stage 0 to drive the next decode step (unless done/cancelled).
    if (!done && (!seq || seq.status !== "cancelled")) {
      const head = this.plan[0]?.holders?.[0];
      if (head) this._sendTo(head, out); // stage 0 treats inbound `token` as "previous token"
    } else if (done) {
      this._fanToOperators(P.seqStatus(this.config?.run ?? this._runId, msg.seq_id, "done"));
      this._fanToSinks(msg.seq_id, P.seqStatus(this.config?.run ?? this._runId, msg.seq_id, "done"));
      this._closeSinks(msg.seq_id);
    }
  }

  // prompt -> prompt-begin to stage 0; track sequence state. Returns false if not ready.
  startSequence(msg) {
    const head = this.plan[0]?.holders?.[0];
    const ready = this._isReady();
    if (!head || !ready) return false;

    const seq_id = msg.seq_id || `seq-${crypto.randomUUID()}`;
    this.seqs.set(seq_id, {
      status: "queued",
      token_pos: 0,
      max_tokens: msg.max_tokens ?? 256,
      prompt: msg.prompt ?? null,
      token_ids: msg.token_ids ?? null,
    });

    // Stage 0 tokenizes the prompt itself (it owns the embedding/tokenizer side);
    // we pass token_ids if the operator pre-tokenized, else the raw prompt string.
    const begin = {
      ...P.promptBegin(
        this.config.run,
        seq_id,
        msg.token_ids ?? [],
        msg.max_tokens ?? 256,
        msg.temperature ?? 0.7,
        msg.top_p ?? 0.95,
      ),
      prompt: msg.prompt ?? undefined, // extra field: raw string for tab-side tokenization
    };
    this.seqs.get(seq_id).status = "prefill";
    this._sendTo(head, begin);

    const status = P.seqStatus(this.config.run, seq_id, "prefill");
    this._fanToOperators(status);
    this._fanToSinks(seq_id, status);
    return true;
  }

  // Reroute on a dead/departed node: promote a replica or recruit a spare + re-fetch range.
  heal(deadNode) {
    if (!this.config) return;
    let changed = false;
    for (const stage of this.plan) {
      const idx = stage.holders.indexOf(deadNode);
      if (idx === -1) continue;
      stage.holders.splice(idx, 1);
      changed = true;

      if (stage.holders.length > 0) {
        // A replica remains -> it becomes primary. Just repoint neighbors. (Petals promote.)
        stage.status = "ready";
        this._broadcastError(
          "stage_failover",
          `stage ${stage.stage}: promoted replica ${stage.holders[0]} after ${deadNode} died`,
        );
      } else {
        // Orphaned: mark recruiting and try to draft a live spare immediately.
        stage.status = "recruiting";
        const spare = this._firstSpare();
        if (spare) {
          stage.holders.push(spare);
          const t = this.tabs.get(spare);
          if (t) {
            t.stage = stage.stage;
            t.role = "stage";
          }
          stage.status = "loading";
          // recruit() tells the spare its exact [lo,hi) so it range-fetches just that slice.
          this._sendTo(
            spare,
            P.recruit(this.config.run, stage.stage, stage.layers, MODELS[this.config.model_id].manifestRef),
          );
          this._emitAssign(spare, stage, false);
        }
        // In-flight sequences that crossed this stage are retried from the head
        // (operator holds the prompt; only committed tokens are durable — see arch §6).
        for (const [seq_id, seq] of this.seqs) {
          if (seq.status === "prefill" || seq.status === "decoding") {
            seq.status = "queued";
            this._fanToOperators(
              P.seqStatus(this.config.run, seq_id, "queued", `retry: stage ${stage.stage} lost`),
            );
          }
        }
      }
    }
    if (changed) {
      this._persistPlan();
      this._recomputeNeighbors();
      this._broadcastState();
    }
  }

  // alarm(): evict stale nodes (> STALE_MS since last hb), heal, then re-arm.
  async alarm() {
    const now = Date.now();
    const dead = [];
    for (const [node, t] of this.tabs) if (now - (t.lastSeen ?? 0) > STALE_MS) dead.push(node);
    for (const node of dead) this._dropNode(node);
    await this._ensureAlarm(true);
  }

  // Full topology snapshot for operators / GET /state.
  snapshot() {
    const present = this._liveTabs().length;
    const needed = this.config ? tabsNeeded(this.config.model_id) : 0;
    return P.runState(
      this.config?.run ?? this._runId,
      this.config?.model_id ?? null,
      this.config?.S ?? 0,
      this.config?.R ?? 1,
      this.plan,
      this._isReady(),
      needed,
      present,
    );
  }

  // ===========================================================================
  // lifecycle helpers
  // ===========================================================================
  _onJoin(ws, msg) {
    const node = msg.node;
    const existing = this.tabs.get(node) ?? {};
    this.tabs.set(node, {
      ...existing,
      ws,
      role: existing.role ?? "spare",
      caps: msg.caps,
      stage: existing.stage ?? null,
      ready: false,
      lastSeen: Date.now(),
    });
    // Bind this socket to the node id so hibernation can re-attach + close can find it.
    const att = safeAttachment(ws) ?? {};
    ws.serializeAttachment({ ...att, node, role: "stage", run: this.config?.run ?? this._runId });
    this._persistTabMeta();

    this._send(ws, P.welcome(this.config?.run ?? this._runId, node, "spare", this.config?.plan_version ?? 0));
    this.assign(node); // incremental placement
  }

  _onReady(msg) {
    const t = this.tabs.get(msg.node);
    if (!t) return;
    t.ready = true;
    t.lastSeen = Date.now();
    t.weight_bytes = msg.weight_bytes ?? t.weight_bytes;
    // Mark the stage ready if its primary holder is this node and it now reports ready.
    const stage = this.plan.find((s) => s.holders[0] === msg.node);
    if (stage) stage.status = "ready";
    this._persistTabMeta();
    this._broadcastState();
  }

  _onMetrics(msg) {
    // Surface per-stage telemetry to operators (console renders tokens/compute_ms/vram).
    this._fanToOperators(msg);
  }

  _touch(node) {
    const t = this.tabs.get(node);
    if (t) t.lastSeen = Date.now();
  }

  _dropNode(node) {
    if (!this.tabs.has(node)) return;
    this.tabs.delete(node);
    this._persistTabMeta();
    this.heal(node);
  }

  _markOperator(ws) {
    const att = safeAttachment(ws) ?? {};
    ws.serializeAttachment({ ...att, role: "operator" });
  }

  async _ensureAlarm(force = false) {
    const cur = await this.ctx.storage.getAlarm();
    if (force || cur == null) this.ctx.storage.setAlarm(Date.now() + SWEEP_MS);
  }

  // ===========================================================================
  // routing / neighbor wiring
  // ===========================================================================

  // Recompute each stage's prev/next primary holder and push route-update where it changed.
  _recomputeNeighbors() {
    for (let i = 0; i < this.plan.length; i++) {
      const stage = this.plan[i];
      const prev = i > 0 ? this.plan[i - 1].holders[0] ?? null : null;
      const next = i < this.plan.length - 1 ? this.plan[i + 1].holders[0] ?? null : null;
      for (const holder of stage.holders) {
        this._sendTo(holder, P.routeUpdate(this.config.run, stage.stage, prev, next));
      }
    }
  }

  _emitAssign(node, stage, isReplica) {
    const i = this.plan.findIndex((s) => s.stage === stage.stage);
    const prev = i > 0 ? this.plan[i - 1].holders[0] ?? null : null;
    const next = i < this.plan.length - 1 ? this.plan[i + 1].holders[0] ?? null : null;
    const m = MODELS[this.config.model_id];
    this._sendTo(
      node,
      P.assignStage(this.config.run, node, {
        stage: stage.stage,
        layers: stage.layers,
        is_first: i === 0,
        is_last: i === this.plan.length - 1,
        prev_node: prev,
        next_node: next,
        model_id: this.config.model_id,
        manifest_ref: m.manifestRef,
        replica_of: isReplica ? stage.holders[0] : null,
      }),
    );
  }

  // ===========================================================================
  // egress: send to a node, fan to operators, SSE sinks
  // ===========================================================================
  _sendTo(node, obj) {
    const t = this.tabs.get(node);
    if (!t) return;
    let ws = t.ws;
    if (!ws) {
      // After hibernation the live ws ref may be missing; recover it by tag/attachment.
      ws = this._findSocketForNode(node);
      if (ws) t.ws = ws;
    }
    if (ws) this._send(ws, obj);
  }

  _findSocketForNode(node) {
    for (const ws of this.ctx.getWebSockets()) {
      const att = safeAttachment(ws);
      if (att?.node === node) return ws;
    }
    return null;
  }

  _fanToOperators(obj) {
    for (const ws of this.ctx.getWebSockets("operator")) this._send(ws, obj);
  }

  _send(ws, obj) {
    try {
      ws.send(typeof obj === "string" ? obj : JSON.stringify(obj));
    } catch {
      /* socket closing/closed */
    }
  }

  // ---- SSE token egress (HTTP clients that don't want a WS) ----
  _sseTokens(runId, seq_id) {
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const enc = new TextEncoder();
    const write = (event, data) =>
      writer.write(enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)).catch(() => {});

    if (!this.tokenSinks.has(seq_id)) this.tokenSinks.set(seq_id, new Set());
    const sink = { writer, write };
    this.tokenSinks.get(seq_id).add(sink);

    // greet + current snapshot
    write("hello", { run: runId, seq_id });
    write("state", this.snapshot());

    return new Response(readable, {
      headers: {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
        ...CORS,
      },
    });
  }

  _fanToSinks(seq_id, obj) {
    const event = obj.t === P.T.TOKEN ? "token" : obj.t === P.T.SEQ_STATUS ? "status" : "msg";
    for (const key of [seq_id, "*"]) {
      const set = this.tokenSinks.get(key);
      if (!set) continue;
      for (const s of set) s.write(event, obj);
    }
  }

  _closeSinks(seq_id) {
    const set = this.tokenSinks.get(seq_id);
    if (!set) return;
    for (const s of set) s.writer.close().catch(() => {});
    this.tokenSinks.delete(seq_id);
  }

  // ===========================================================================
  // small helpers
  // ===========================================================================
  _liveTabs() {
    const now = Date.now();
    return [...this.tabs.entries()].filter(([, t]) => now - (t.lastSeen ?? 0) <= STALE_MS);
  }

  _firstSpare() {
    for (const [node, t] of this._liveTabs()) if (t.role === "spare" || t.stage == null) return node;
    return null;
  }

  _isReady() {
    return (
      this.plan.length > 0 &&
      this.plan.every((s) => s.holders.length > 0 && s.status === "ready")
    );
  }

  _cancelSeq(seq_id) {
    const s = this.seqs.get(seq_id);
    if (s) s.status = "cancelled";
    this._fanToOperators(P.seqStatus(this.config?.run ?? this._runId, seq_id, "cancelled"));
    this._fanToSinks(seq_id, P.seqStatus(this.config?.run ?? this._runId, seq_id, "cancelled"));
    this._closeSinks(seq_id);
    // tell stage 0 to stop driving this sequence
    const head = this.plan[0]?.holders?.[0];
    if (head) this._sendTo(head, P.cancel(this.config?.run ?? this._runId, seq_id));
  }

  _broadcastState() {
    this._fanToOperators(this.snapshot());
  }

  _broadcastError(code, message) {
    this._fanToOperators(P.error(this.config?.run ?? this._runId, code, message));
  }

  _persistPlan() {
    this.ctx.storage.put("plan", this.plan);
    if (this.config) this.ctx.storage.put("config", this.config);
  }

  _persistTabMeta() {
    // Persist a ws-free copy so it survives hibernation (the live ws can't be serialized).
    const out = {};
    for (const [node, t] of this.tabs) {
      out[node] = {
        role: t.role,
        caps: t.caps,
        stage: t.stage,
        ready: t.ready,
        lastSeen: t.lastSeen,
        weight_bytes: t.weight_bytes ?? 0,
      };
    }
    this.ctx.storage.put("tabMeta", out);
  }

  // Binary activation frame on the hot path: decode header, re-frame to the next holder.
  _onBinaryActivation(ws, frame) {
    let meta;
    try {
      ({ meta } = P.decodeActivation(frame));
    } catch {
      return;
    }
    if (typeof meta?.to_stage !== "number") return;
    const stage = this.plan.find((s) => s.stage === meta.to_stage);
    if (!stage || stage.holders.length === 0) {
      this._broadcastError("stage_orphaned", `to_stage ${meta.to_stage} has no holder (binary)`);
      return;
    }
    const target = this.tabs.get(stage.holders[0]);
    let tws = target?.ws ?? this._findSocketForNode(stage.holders[0]);
    if (tws) {
      try {
        tws.send(frame); // forward the raw binary frame verbatim — zero re-encode
      } catch {
        /* closing */
      }
    }
  }
}

// Read a socket's serialized attachment safely (null after a cold start with no attachment).
function safeAttachment(ws) {
  try {
    return ws.deserializeAttachment?.() ?? null;
  } catch {
    return null;
  }
}
