// ce-tabnet mesh coordinator — the Run topology authority, ON THE CE MESH.
//
// This is the former Cloudflare Durable Object `RunCoordinator` (coordinator/src/index.js)
// re-homed onto the CE mesh via the @ce-net/sdk serve()/locate() framework. There is NO central
// Cloudflare Worker or Durable Object anymore: ONE tab in a run hosts this coordinator as a real
// mesh service, named `tabnet/<run>`, that peers discover with the DHT (locate) and call over
// libp2p request/reply (call) — relay/NAT traversal included, no stored ip:port, no *.workers.dev.
//
// What moved, and to which @ce-net/sdk API:
//   - room membership + stage assignment + placement + healing  -> stays here, in `RunCoordinator`
//     below (ported ~verbatim from the DO), but now driven by mesh requests instead of WS frames.
//   - "who hosts which stage/shard" + neighbor wiring            -> request/reply (serve + call):
//        a peer sends JOIN/READY/HB/PROMPT/CANCEL/SUBSCRIBE/LEAVE as a request on `tabnet/<run>`;
//        the coordinator replies with welcome/assign-stage/route-update/etc. as the reply payload
//        (plus directed `mesh.send` for assignments that target a specific node).
//   - run-state + token egress to operators                     -> publish/subscribe:
//        coordinator publishes run-state on `tabnet/<run>/state` and tokens on `tabnet/<run>/tokens`.
//   - activation hops (hidden state, hot path)                  -> direct peer-to-peer mesh.send
//        between the assigned `prev_node`/`next_node` CE NodeIds — NOT relayed through here.
//        (The old DO was a star relay for activations; the mesh routes them directly, one hop.)
//   - the autoregressive token feedback to stage 0              -> directed `mesh.send` to stage 0.
//
// The wire MESSAGE SHAPES are unchanged: this module imports web/protocol.js (single source of
// truth) exactly as the DO did, and bincode/JSON is replaced by JSON bytes over the mesh. The
// model registry + placement math (web/model-config.js) are imported unchanged too.
//
// Pure-ish: the only I/O is through the injected `ce` (a @ce-net/sdk CeClient) and the @ce-net/sdk
// serve/register helpers, passed in by mesh-transport.js so this file has no direct SDK import path
// baked in (keeps it testable headless — see dev/serve.js --selftest).

import * as P from "./protocol.js";
import { MODELS, defaultStagePlan, tabsNeeded } from "./model-config.js";

// Same staleness window + sweep cadence as the DO (and ce-hub / node.html 10s hb).
const STALE_MS = 35_000;
const SWEEP_MS = 5_000;

// Topic names for one run. `svc` is the request/reply service peers locate(); `state`/`tokens`
// are pub/sub topics; `act` is the per-node directed activation topic (peer-to-peer, hot path).
export function runTopics(run) {
  return {
    svc: `tabnet/${run}`,
    state: `tabnet/${run}/state`,
    tokens: `tabnet/${run}/tokens`,
    act: `tabnet/${run}/act`,
  };
}

const enc = new TextEncoder();
const dec = new TextDecoder();
export const encodeMsg = (obj) => enc.encode(JSON.stringify(obj));
export const decodeMsg = (bytes) => {
  try { return JSON.parse(dec.decode(bytes)); } catch { return null; }
};

// ===========================================================================
// RunCoordinator — one per run, hosted by one tab as a mesh service.
//
// Ported from coordinator/src/index.js. The differences from the DO version:
//   * No WebSockets / hibernation / DO storage — state is in-memory in the hosting tab. If the
//     hosting tab dies, the run's coordinator is gone; a surviving operator/stage can re-host it
//     (mesh-transport handles failover by re-electing a coordinator). This is the same SPOF the DO
//     had per-run (architecture §11), now mesh-native instead of Cloudflare-native.
//   * `_sendTo(node, msg)` is a directed mesh.send to that node's CE NodeId.
//   * operator egress is publish() to the state/tokens topics, not a WS fan-out.
//   * node ids ARE CE NodeIds (hex), so neighbor wiring routes real mesh peers.
// ===========================================================================
export class RunCoordinator {
  // deps: { ce, run, selfId, publishState, publishToken, sendTo, onLog }
  //   ce          — @ce-net/sdk CeClient (for completeness; not used directly here)
  //   run         — run id
  //   selfId      — this coordinator tab's CE NodeId (so it can host a stage too)
  //   publishState(obj)        — publish a run-state/error/status to the state topic (operators)
  //   publishToken(obj)        — publish a token/seq-status to the tokens topic (operators)
  //   sendTo(node, obj)        — directed mesh.send of a JSON control message to a node id
  //   onLog(msg)               — optional diagnostics
  constructor(deps) {
    this.ce = deps.ce;
    this.run = deps.run;
    this.selfId = deps.selfId ?? null;
    this._publishState = deps.publishState ?? (() => {});
    this._publishToken = deps.publishToken ?? (() => {});
    this._sendTo = deps.sendTo ?? (() => {});
    this._log = deps.onLog ?? (() => {});

    // node -> { caps, stage, role, ready, lastSeen, weight_bytes }
    this.tabs = new Map();
    this.config = null; // { run, model_id, S, R, fastpath, microbatch, plan_version }
    this.plan = []; // [{ stage, layers:[lo,hi], holders:[node], status }]
    this.seqs = new Map(); // seq_id -> { status, token_pos, max_tokens, ... }

    this._sweep = null;
  }

  // Start the stale-node sweep (replaces the DO alarm()).
  startSweep() {
    if (this._sweep) return;
    this._sweep = setInterval(() => this._tickSweep(), SWEEP_MS);
  }
  stopSweep() {
    if (this._sweep) { clearInterval(this._sweep); this._sweep = null; }
  }
  _tickSweep() {
    const now = Date.now();
    const dead = [];
    for (const [node, t] of this.tabs) if (now - (t.lastSeen ?? 0) > STALE_MS) dead.push(node);
    for (const node of dead) this._dropNode(node);
  }

  // ---------------------------------------------------------------------------
  // Request dispatch — the mesh request/reply entry point.
  // A peer sends one validated protocol message as a request on `tabnet/<run>`; we mutate state
  // and return a reply message (always something, so the caller's request() never times out).
  // `from` is the authenticated sender CE NodeId (the node verified it).
  // ---------------------------------------------------------------------------
  handleRequest(from, msg) {
    if (!msg || typeof msg.t !== "string" || !P.validate(msg)) {
      return P.error(this.run, "bad_message", "unrecognized or invalid message");
    }
    switch (msg.t) {
      case P.T.JOIN:             return this._onJoin(from, msg);
      case P.T.READY:            this._onReady(msg); return P.ack(this.run);
      case P.T.HB:               this._touch(msg.node); return P.ack(this.run);
      case P.T.LEAVE:            this._dropNode(msg.node); return P.ack(this.run);
      case P.T.METRICS:          this._publishToken(msg); return P.ack(this.run); // operators watch tokens topic
      case P.T.CREATE_RUN:       this.createRun(msg); return this.snapshot();
      case P.T.PROMPT: {
        const ok = this.startSequence(msg);
        return ok ? P.ack(this.run, { seq_id: msg.seq_id })
                  : P.error(this.run, "run_not_ready", "no full pipeline");
      }
      case P.T.CANCEL:           this._cancelSeq(msg.seq_id); return P.ack(this.run);
      case P.T.SUBSCRIBE_TOKENS: return this.snapshot(); // reply with current state; future updates ride pub/sub
      default:                   return P.ack(this.run);  // forward-compat: accept unknown types
    }
  }

  // ---------------------------------------------------------------------------
  // create-run / placement / assignment — ported verbatim from the DO.
  // ---------------------------------------------------------------------------
  createRun(body) {
    const m = MODELS[body.model_id];
    if (!m) { this._broadcastError("unknown_model", `no such model: ${body.model_id}`); return; }
    const run = body.run || this.run;
    this.config = {
      run,
      model_id: body.model_id,
      S: body.stages ?? null,
      R: body.replicas ?? 1,
      fastpath: !!body.fastpath,
      microbatch: body.microbatch ?? 8,
      cap: body.cap ?? null,
      plan_version: (this.config?.plan_version ?? 0) + 1,
    };
    this.planStages();
  }

  planStages() {
    if (!this.config) return;
    const m = MODELS[this.config.model_id];
    const live = this._liveTabs();

    const budgetOf = (t) => {
      const v = t.caps?.vram_mb ? t.caps.vram_mb * 1024 * 1024 : 0;
      if (v > 0) return v;
      const ram = (t.caps?.ram_gb ?? 1) * 1024 * 1024 * 1024;
      return Math.max(256 * 1024 * 1024, ram * 0.5);
    };

    const ranked = [...live].sort((a, b) => budgetOf(b[1]) - budgetOf(a[1]));
    const budgets = ranked.map(([, t]) => budgetOf(t));

    let basePlan;
    if (budgets.length === 0) {
      basePlan = defaultStagePlan(this.config.model_id, []);
    } else {
      basePlan = defaultStagePlan(this.config.model_id, budgets);
    }

    const S = basePlan.length;
    this.config.S = S;

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

    const assignedNodes = new Set(this.plan.flatMap((s) => s.holders));
    let si = 0;
    for (const [node, t] of ranked) {
      if (assignedNodes.has(node)) continue;
      while (si < this.plan.length && this.plan[si].holders.length >= this.config.R) si++;
      if (si >= this.plan.length) {
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

    this._recomputeNeighbors();
    this._broadcastState();
  }

  // Give a single joining tab a stage/replica/spare role (incremental, no full replan).
  assign(node) {
    const t = this.tabs.get(node);
    if (!t || !this.config) {
      if (t) this._sendTo(node, P.welcome(this.run, node, "spare", this.config?.plan_version ?? 0));
      return;
    }
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
    this._recomputeNeighbors();
    this._broadcastState();
  }

  // ---------------------------------------------------------------------------
  // prompt lifecycle — ported from the DO (minus the WS plumbing).
  // ---------------------------------------------------------------------------
  startSequence(msg) {
    const head = this.plan[0]?.holders?.[0];
    const ready = this._isReady();
    if (!head || !ready) return false;

    const seq_id = msg.seq_id || `seq-${randomId()}`;
    this.seqs.set(seq_id, {
      status: "queued",
      token_pos: 0,
      max_tokens: msg.max_tokens ?? 256,
      prompt: msg.prompt ?? null,
      token_ids: msg.token_ids ?? null,
    });

    const begin = {
      ...P.promptBegin(
        this.config.run, seq_id,
        msg.token_ids ?? [],
        msg.max_tokens ?? 256,
        msg.temperature ?? 0.7,
        msg.top_p ?? 0.95,
      ),
      prompt: msg.prompt ?? undefined,
    };
    this.seqs.get(seq_id).status = "prefill";
    this._sendTo(head, begin);

    const status = P.seqStatus(this.config.run, seq_id, "prefill");
    this._publishToken(status);
    return true;
  }

  // Last stage sampled a token: publish to operators + feed back to stage 0 to drive next decode.
  // Invoked by mesh-transport when a `token` message arrives on the tokens-ingress path.
  routeToken(msg) {
    const seq = this.seqs.get(msg.seq_id);
    let done = !!msg.done;
    if (seq) {
      seq.token_pos = msg.token_pos;
      seq.generated = (seq.generated ?? 0) + 1;
      if (seq.max_tokens && seq.generated >= seq.max_tokens) done = true;
      if (done) seq.status = "done";
      else if (seq.status === "prefill") seq.status = "decoding";
    }
    // Re-emit with the correct 7-arg shape (run, node, seq_id, token_pos, token_id, text, done).
    const out = P.token(
      this.config?.run ?? this.run, msg.node ?? this.selfId,
      msg.seq_id, msg.token_pos, msg.token_id, msg.text ?? "", done,
    );
    this._publishToken(out);

    if (!done && (!seq || seq.status !== "cancelled")) {
      const head = this.plan[0]?.holders?.[0];
      if (head) this._sendTo(head, out); // stage 0 treats inbound `token` as "previous token"
    } else if (done) {
      this._publishToken(P.seqStatus(this.config?.run ?? this.run, msg.seq_id, "done"));
    }
  }

  // ---------------------------------------------------------------------------
  // healing — ported verbatim from the DO.
  // ---------------------------------------------------------------------------
  heal(deadNode) {
    if (!this.config) return;
    let changed = false;
    for (const stage of this.plan) {
      const idx = stage.holders.indexOf(deadNode);
      if (idx === -1) continue;
      stage.holders.splice(idx, 1);
      changed = true;

      if (stage.holders.length > 0) {
        stage.status = "ready";
        this._broadcastError(
          "stage_failover",
          `stage ${stage.stage}: promoted replica ${stage.holders[0]} after ${deadNode} died`,
        );
      } else {
        stage.status = "recruiting";
        const spare = this._firstSpare();
        if (spare) {
          stage.holders.push(spare);
          const t = this.tabs.get(spare);
          if (t) { t.stage = stage.stage; t.role = "stage"; }
          stage.status = "loading";
          this._sendTo(
            spare,
            P.recruit(this.config.run, stage.stage, stage.layers, MODELS[this.config.model_id].manifestRef),
          );
          this._emitAssign(spare, stage, false);
        }
        for (const [seq_id, seq] of this.seqs) {
          if (seq.status === "prefill" || seq.status === "decoding") {
            seq.status = "queued";
            this._publishToken(
              P.seqStatus(this.config.run, seq_id, "queued", `retry: stage ${stage.stage} lost`),
            );
          }
        }
      }
    }
    if (changed) {
      this._recomputeNeighbors();
      this._broadcastState();
    }
  }

  // Full topology snapshot for operators / state requests.
  snapshot() {
    const present = this._liveTabs().length;
    const needed = this.config ? tabsNeeded(this.config.model_id) : 0;
    return P.runState(
      this.config?.run ?? this.run,
      this.config?.model_id ?? null,
      this.config?.S ?? 0,
      this.config?.R ?? 1,
      this.plan,
      this._isReady(),
      needed,
      present,
    );
  }

  // ---------------------------------------------------------------------------
  // lifecycle helpers
  // ---------------------------------------------------------------------------
  _onJoin(from, msg) {
    const node = msg.node || from;
    const existing = this.tabs.get(node) ?? {};
    this.tabs.set(node, {
      ...existing,
      role: existing.role ?? "spare",
      caps: msg.caps,
      stage: existing.stage ?? null,
      ready: false,
      lastSeen: Date.now(),
    });
    this.assign(node); // incremental placement (sends assign-stage/welcome via _sendTo)
    // Reply with the provisional welcome so the joining tab gets an immediate ack even before its
    // directed assign-stage arrives.
    return P.welcome(this.config?.run ?? this.run, node, this.tabs.get(node).role, this.config?.plan_version ?? 0);
  }

  _onReady(msg) {
    const t = this.tabs.get(msg.node);
    if (!t) return;
    t.ready = true;
    t.lastSeen = Date.now();
    t.weight_bytes = msg.weight_bytes ?? t.weight_bytes;
    const stage = this.plan.find((s) => s.holders[0] === msg.node);
    if (stage) stage.status = "ready";
    this._broadcastState();
  }

  _touch(node) { const t = this.tabs.get(node); if (t) t.lastSeen = Date.now(); }

  _dropNode(node) {
    if (!this.tabs.has(node)) return;
    this.tabs.delete(node);
    this.heal(node);
  }

  // ---------------------------------------------------------------------------
  // routing / neighbor wiring — directed mesh.send to each holder.
  // ---------------------------------------------------------------------------
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
    this._sendTo(node, P.assignStage(this.config.run, node, {
      stage: stage.stage,
      layers: stage.layers,
      is_first: i === 0,
      is_last: i === this.plan.length - 1,
      prev_node: prev,
      next_node: next,
      model_id: this.config.model_id,
      manifest_ref: m.manifestRef,
      replica_of: isReplica ? stage.holders[0] : null,
    }));
  }

  // ---------------------------------------------------------------------------
  // small helpers
  // ---------------------------------------------------------------------------
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
    this._publishToken(P.seqStatus(this.config?.run ?? this.run, seq_id, "cancelled"));
    const head = this.plan[0]?.holders?.[0];
    if (head) this._sendTo(head, P.cancel(this.config?.run ?? this.run, seq_id));
  }
  _broadcastState() { this._publishState(this.snapshot()); }
  _broadcastError(code, message) { this._publishState(P.error(this.config?.run ?? this.run, code, message)); }
}

// A small random id for sequences when the caller didn't supply one (crypto.randomUUID where present).
function randomId() {
  try { return crypto.randomUUID(); } catch { return Math.random().toString(36).slice(2); }
}
