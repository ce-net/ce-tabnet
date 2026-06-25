// ce-tabnet mesh transport — the per-tab bridge between the tabnet runtime and the CE mesh.
//
// This replaces the Cloudflare WebSocket the tabs used to open to the Durable Object. Every tab is
// already a CE browser node (web/site/node.html injects `window.__ceNode`); @ce-net/sdk's
// connectNode() hands us a CeClient bound to that in-process node. From there:
//
//   * COORDINATOR HOSTING (operator tab): hostCoordinator() spins up a RunCoordinator (the former
//     DO logic, see mesh-coordinator.js) and serves it as a mesh service named `tabnet/<run>`,
//     re-advertised on the DHT via @ce-net/sdk register() so peers can locate() it. It also drains
//     the node's inbound message stream for (a) directed activation token-feedback / metrics and
//     (b) the per-node directed control channel.
//
//   * COORDINATOR CLIENT (stage tab + operator): callCoordinator() locate()s the live coordinator
//     for a run and request()s it over libp2p (failing over to the next-best instance). join/ready/
//     hb/prompt/cancel/subscribe all go through here — exactly the messages that used to be WS
//     frames to the DO.
//
//   * ACTIVATIONS (hot path): sendActivation()/onActivation() move the binary hidden-state frame
//     DIRECTLY between adjacent stages' CE NodeIds with mesh.send on `tabnet/<run>/act` — no central
//     relay (the DO was a star; the mesh is point-to-point, one hop).
//
//   * PUB/SUB: the coordinator publishes run-state on `tabnet/<run>/state` and tokens on
//     `tabnet/<run>/tokens`; operators subscribe() and receive them off the message stream.
//
// @ce-net/sdk APIs used: connectNode, serve, register, locate, call, and the CeClient.mesh
// surface (send / subscribe / publish / streamMessages / reply). No Cloudflare, no HTTP-to-a-worker.
//
// IMPORT NOTE: the SDK is loaded from an import map / bundler alias `@ce-net/sdk`. The browser pages
// resolve it via <script type="importmap"> (see config.js sdkSpecifier()); in headless tests the
// caller injects a stub. We import lazily through loadSdk() so this module also parses where the
// SDK isn't present (the pure RunCoordinator logic is tested via dev/serve.js --selftest without it).

import * as P from "./protocol.js";
import {
  RunCoordinator,
  runTopics,
  encodeMsg,
  decodeMsg,
} from "./mesh-coordinator.js";

// Lazily import the SDK so this file is importable without it (selftest, type-check of consumers).
let _sdk = null;
async function loadSdk(spec) {
  if (_sdk) return _sdk;
  _sdk = await import(spec || "@ce-net/sdk");
  return _sdk;
}

// How often the coordinator re-advertises itself on the DHT (provider records expire).
const ADVERTISE_MS = 30_000;

// ---------------------------------------------------------------------------
// MeshTransport — one per tab. Wraps a CeClient and the run's topics.
// ---------------------------------------------------------------------------
export class MeshTransport {
  // opts: { run, sdkSpec?, onLog? }
  constructor({ run, sdkSpec, onLog = () => {} }) {
    this.run = run;
    this.sdkSpec = sdkSpec;
    this.onLog = onLog;
    this.topics = runTopics(run);
    this.ce = null;
    this.selfId = null;
    this._ctrl = new AbortController();
    // directed-control handlers (svc topic), keyed by message type (set by the runtime via on()).
    this._onControl = new Map();
    // pub/sub handlers (state/tokens topics), keyed by message type (operator UI updates).
    this._onPubSub = new Map();
    this._onActivation = null; // (meta, payloadArrayBuffer) => void
    this._coordinator = null; // RunCoordinator if we host
    this._reading = false;
    this._coordId = null; // cached NodeId of the located coordinator (for directed token/metrics)
  }

  async connect() {
    const sdk = await loadSdk(this.sdkSpec);
    this.ce = sdk.connectNode(); // in-browser bridge (window.__ceNode) or same-origin /ce proxy
    // The tab's CE NodeId — what neighbor wiring routes to. Prefer the node's reported id.
    this.selfId = await resolveSelfId(this.ce);
    this.onLog(`mesh node ${shortId(this.selfId)} connected`);
    return this.selfId;
  }

  // ---- register typed handlers the runtime cares about ----
  // Directed control messages addressed to THIS node on the svc topic (assign/route/prompt-begin/
  // token-feedback/recruit/...). Used by stage tabs and by the coordinator host.
  onControl(type, fn) { this._onControl.set(type, fn); return this; }
  // Pub/sub messages on the state/tokens topics (run-state, token, seq-status, error, metrics).
  // Used by the operator UI. Kept separate from onControl so the same message type (e.g. `token`)
  // can mean "feedback to the coordinator" (directed) vs "display this" (broadcast).
  onPubSub(type, fn) { this._onPubSub.set(type, fn); return this; }
  onActivation(fn) { this._onActivation = fn; return this; }

  // ===========================================================================
  // STAGE / OPERATOR CLIENT — talk to the coordinator over the mesh.
  // ===========================================================================

  // Locate the live `tabnet/<run>` coordinator and request `msg`, returning its reply (parsed).
  // Mirrors the DO's "send a WS frame and (sometimes) get a reply" — but every request resolves.
  // Caches the coordinator NodeId from the located instance so directed token/metrics sends don't
  // re-locate per token.
  async callCoordinator(msg, opts = {}) {
    // If THIS tab hosts the coordinator (operator), answer in-process — no self-addressed mesh
    // round-trip, and works even before our own DHT advertisement has propagated.
    if (this._coordinator) {
      return this._coordinator.handleRequest(this.selfId, msg);
    }
    const sdk = await loadSdk(this.sdkSpec);
    const insts = await sdk.locate(this.ce, this.topics.svc, { want: 3, maxStaleSecs: 60 });
    if (insts.length === 0) throw new Error(`no coordinator for run ${this.run}`);
    let lastErr;
    for (const inst of insts) {
      try {
        const reply = await this.ce.mesh.request(
          inst.nodeId, this.topics.svc, encodeMsg(msg), opts.timeoutMs ?? 10_000,
        );
        this._coordId = inst.nodeId; // remember who answered
        return decodeMsg(reply);
      } catch (e) { lastErr = e; }
    }
    throw lastErr ?? new Error("coordinator request failed");
  }

  // Fire a directed control message to the coordinator (token feedback from the last stage,
  // periodic metrics). Uses the cached coordinator NodeId; locates once if unknown.
  async sendToCoordinator(obj) {
    if (!this._coordId) {
      const sdk = await loadSdk(this.sdkSpec);
      const insts = await sdk.locate(this.ce, this.topics.svc, { want: 1, maxStaleSecs: 60 });
      if (insts.length === 0) return;
      this._coordId = insts[0].nodeId;
    }
    await this.sendControl(this._coordId, obj);
  }

  // ===========================================================================
  // ACTIVATIONS — direct peer-to-peer hidden-state hop (no coordinator in the path).
  // ===========================================================================

  // Send a binary activation frame straight to the next stage's CE NodeId.
  async sendActivation(toNode, frame /* ArrayBuffer */) {
    if (!toNode) return;
    await this.ce.mesh.send(toNode, this.topics.act, new Uint8Array(frame));
  }

  // Send a directed control message (welcome/assign/route-update/prompt-begin/token/recruit/...)
  // to a specific node — used by the coordinator's _sendTo.
  async sendControl(toNode, obj) {
    if (!toNode) return;
    await this.ce.mesh.send(toNode, this.topics.svc, encodeMsg(obj));
  }

  // ===========================================================================
  // INBOUND STREAM — drain the node's message stream and dispatch.
  //   * activation frames on `act`            -> onActivation()
  //   * directed control on `svc` (replyToken==null) -> onControl[type]
  //   * pub/sub on `state`/`tokens`           -> onControl[type] (operators)
  // Requests (replyToken != null) on `svc` are handled by the coordinator's serve() loop, NOT here.
  // ===========================================================================
  async startReading(extraTopics = []) {
    if (this._reading) return;
    this._reading = true;
    const signal = this._ctrl.signal;

    for (const t of [this.topics.act, this.topics.svc, ...extraTopics]) {
      try { await this.ce.mesh.subscribe(t); } catch (e) { this.onLog(`subscribe ${t} failed: ${e}`); }
    }

    // Drain in the background; reconnect on stream end.
    (async () => {
      while (!signal.aborted) {
        try {
          for await (const m of this.ce.mesh.streamMessages({ signal })) {
            if (signal.aborted) break;
            this._dispatch(m);
          }
        } catch (e) {
          if (signal.aborted) break;
          this.onLog(`message stream error; reconnecting: ${e}`);
          await sleep(500, signal);
        }
        if (signal.aborted) break;
      }
    })();
  }

  _dispatch(m) {
    // Activation hot path: binary frame on the act topic.
    if (m.topic === this.topics.act) {
      if (!this._onActivation) return;
      let payload; try { payload = m.payload(); } catch { return; }
      // payload is the encoded activation frame (header + tensor) — decode the meta here.
      let meta; try { ({ meta } = P.decodeActivation(toArrayBuffer(payload))); } catch { return; }
      this._onActivation(meta, toArrayBuffer(payload));
      return;
    }

    // Requests with a replyToken on the svc topic are consumed by the coordinator's serve() loop.
    if (m.replyToken !== null && m.topic === this.topics.svc) return;

    let payload; try { payload = m.payload(); } catch { return; }
    const obj = decodeMsg(payload);
    if (!obj || typeof obj.t !== "string") return;

    // pub/sub topics -> operator UI handlers; svc directed control -> control handlers.
    if (m.topic === this.topics.state || m.topic === this.topics.tokens) {
      const fn = this._onPubSub.get(obj.t);
      if (fn) fn(obj, m.from);
      return;
    }
    const fn = this._onControl.get(obj.t);
    if (fn) fn(obj, m.from);
  }

  // ===========================================================================
  // COORDINATOR HOSTING — be the `tabnet/<run>` service (the former DO).
  // ===========================================================================
  async hostCoordinator(createRunMsg) {
    const sdk = await loadSdk(this.sdkSpec);

    const coord = new RunCoordinator({
      ce: this.ce,
      run: this.run,
      selfId: this.selfId,
      publishState: (obj) => { this.ce.mesh.publish(this.topics.state, encodeMsg(obj)).catch(() => {}); },
      publishToken: (obj) => { this.ce.mesh.publish(this.topics.tokens, encodeMsg(obj)).catch(() => {}); },
      sendTo: (node, obj) => { this.sendControl(node, obj).catch(() => {}); },
      onLog: this.onLog,
    });
    this._coordinator = coord;
    coord.startSweep();

    // Apply the initial create-run (the operator already chose model/stages/replicas).
    if (createRunMsg) coord.createRun({ ...createRunMsg, run: this.run });

    // The coordinator must also see token feedback from the LAST stage and metrics. Those arrive as
    // directed control messages to the coordinator's node id on the svc topic; route them in.
    this.onControl(P.T.TOKEN, (msg) => coord.routeToken(msg));
    this.onControl(P.T.METRICS, (msg) => coord._publishToken(msg));

    // serve() the request/reply loop on the svc topic. Each request is one protocol message; reply
    // with the coordinator's response. register() keeps us discoverable via the DHT.
    const handler = (req) => {
      const msg = decodeMsg(req.payload);
      const reply = coord.handleRequest(req.from, msg);
      return encodeMsg(reply ?? P.ack(this.run));
    };
    sdk.serve(this.ce, [this.topics.svc], handler, {
      signal: this._ctrl.signal,
      onWarn: (s, d) => this.onLog(`coordinator serve: ${s} ${d ?? ""}`),
    }).catch((e) => this.onLog(`coordinator serve loop ended: ${e}`));

    sdk.register(this.ce, this.topics.svc, ADVERTISE_MS, {
      signal: this._ctrl.signal,
      onWarn: (s, d) => this.onLog(`coordinator register: ${s} ${d ?? ""}`),
    }).catch((e) => this.onLog(`coordinator register ended: ${e}`));

    this.onLog(`hosting coordinator for run ${this.run} as ${this.topics.svc}`);
    return coord;
  }

  // Whether a live coordinator already exists for this run (so we don't double-host).
  async coordinatorExists() {
    const sdk = await loadSdk(this.sdkSpec);
    try {
      const insts = await sdk.locate(this.ce, this.topics.svc, { want: 1, maxStaleSecs: 60 });
      return insts.length > 0;
    } catch { return false; }
  }

  stop() {
    this._ctrl.abort();
    if (this._coordinator) this._coordinator.stopSweep();
  }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
async function resolveSelfId(ce) {
  // The in-browser bridge exposes the node id directly; fall back to /status.
  const bridge = globalThis.__ceNode;
  if (bridge && bridge.nodeId) return bridge.nodeId;
  try { const s = await ce.status(); return s.nodeId; } catch { return null; }
}

function shortId(id) { return id ? String(id).slice(0, 12) + "…" : "?"; }

function toArrayBuffer(u8) {
  if (u8 instanceof ArrayBuffer) return u8;
  return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
}

function sleep(ms, signal) {
  return new Promise((resolve) => {
    if (signal?.aborted) return resolve();
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); resolve(); }, { once: true });
  });
}
