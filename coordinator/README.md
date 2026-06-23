# ce-tabnet coordinator

A Cloudflare **Worker + Durable Object** that coordinates one sharded-LLM **Run** across many
browser tabs. One Durable Object instance == one Run (one model instance), addressed by
`idFromName(runId)`.

It is the only stateful component: it owns the live pipeline **topology** (stage → tab(s)),
**assigns** stages to tabs on join by device capability, **relays** activation / token /
heartbeat messages along the pipeline, **heals** on tab dropout (replica promote or spare
re-recruit + layer-range re-fetch), and exposes a small **HTTP API** to start a run and stream
tokens out.

- Entry: `src/index.js` — Worker `fetch` router + `RunCoordinator extends DurableObject`.
- Uses the **WebSocket Hibernation API** (`ctx.acceptWebSocket` + `webSocketMessage/Close/Error`)
  so the DO evicts from memory between messages — required to hold hundreds of tab sockets cheaply.
- Imports the shared wire protocol (`../../web/protocol.js`) and model registry
  (`../../web/model-config.js`); wrangler bundles them automatically.

See `../docs/architecture.md` §9 and `../docs/module-contract.md` §B/§C for the full design.

---

## Deploy (DO NOT run without the user's Cloudflare creds + confirmation)

1. Fill `account_id` and pick an egress (workers.dev or a `ce-net.com/tabnet/*` route) in
   `wrangler.jsonc`.
2. `npx wrangler login`
3. **One command:**

   ```sh
   npx wrangler deploy --config coordinator/wrangler.jsonc
   ```

## Local dev (no creds needed)

```sh
npx wrangler dev --config coordinator/wrangler.jsonc
# DO + WebSocket hibernation are emulated by miniflare locally.
```

---

## HTTP / WebSocket API

Base URL = the deployed Worker (`https://ce-tabnet.<acct>.workers.dev` or `ce-net.com/tabnet`).

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/run/:id/ws?role=stage` | Tab joins as a pipeline stage (WebSocket, hibernatable). |
| `GET`  | `/run/:id/ws?role=operator` | Operator console (WebSocket): receives `run-state`, `token`, `seq-status`, `error`. |
| `POST` | `/run/:id` | **create-run**. JSON body: `{ model_id, stages?, replicas?, fastpath?, microbatch? }`. Idempotent. |
| `POST` | `/run/:id/prompt` | Submit a prompt over HTTP. Body: `{ prompt | token_ids, max_tokens?, temperature?, top_p?, seq_id? }`. Returns `{ seq_id }` (409 if pipeline not ready). |
| `GET`  | `/run/:id/tokens?seq_id=...` | **SSE** token egress (use `seq_id=*` for all). Events: `token`, `status`, `state`. |
| `GET`  | `/run/:id/state` | JSON `run-state` snapshot (topology, readiness, tabs needed/present). |
| `GET`  | `/health` | Service liveness + known models. |

Tabs and operators may do **everything over the WebSocket** instead (the `subscribe-tokens`,
`prompt`, `create-run` message types exist on the wire). The HTTP `POST /prompt` + SSE `/tokens`
endpoints exist so a plain `curl`/`fetch` client can drive a run and read tokens with no WS.

### Quick end-to-end with curl (once tabs have joined and reported `ready`)

```sh
BASE=https://ce-tabnet.<acct>.workers.dev
RUN=demo

# 1. define the run
curl -X POST $BASE/run/$RUN -d '{"model_id":"tinyllama-1.1b-q4","replicas":1}'

# 2. open the run as a stage in N browser tabs:  $BASE/run/$RUN/ws?role=stage  (web/join.html does this)

# 3. start streaming tokens (leave this running)
curl -N "$BASE/run/$RUN/tokens?seq_id=s1" &

# 4. submit a prompt
curl -X POST $BASE/run/$RUN/prompt -d '{"seq_id":"s1","prompt":"Hello, world","max_tokens":64}'
```

---

## DO API + message handling (the contract)

`RunCoordinator` honors the wire protocol in `../web/protocol.js` (single source of truth).

**Inbound, tab → DO** (`webSocketMessage`):
`join` (register + capability-assign), `ready` (mark stage ready), `hb` (touch lastSeen),
`activation` (Path-A relay to next stage's primary holder — JSON or **binary frame**, forwarded
verbatim), `token` (fan to operators + feed back to stage 0), `signal` (Path-B WebRTC relay to
`to_node`), `metrics` (surface to operators), `leave`.

**Inbound, operator → DO**: `create-run`, `prompt` (→ `prompt-begin` to stage 0), `cancel`,
`subscribe-tokens`.

**Outbound, DO → tab**: `welcome`, `assign-stage` (with `[lo,hi)`, neighbors, manifest ref),
`route-update` (neighbor changed), `activation`/`token` (relayed), `prompt-begin`, `recruit`
(heal an orphaned stage), `evict`.

**Outbound, DO → operator**: `run-state` (full topology, pushed on every change), `token`,
`seq-status`, `error`.

**Placement** (`planStages`/`assign`): memory-weighted — each tab's budget = reported
`caps.vram_mb` (WebGPU) or `ram_gb*0.5` (CPU fallback); beefiest tabs get the most layers and the
embedding/lm_head ends. Surplus tabs become `spare`; with `replicas > 1` extra holders become
warm `replica`s.

**Healing** (`heal` + `alarm` every ~5 s, stale > 35 s): a dead stage with a replica is failed
over by promotion; an orphaned stage is marked `recruiting` and the first live spare is drafted,
told its exact `[lo,hi)` via `recruit` so it range-fetches only that slice; in-flight sequences
that crossed the lost stage are requeued (the operator still holds the prompt).

**Persistence / hibernation**: `config`, `plan`, and ws-free `tabMeta` are written to DO storage;
each socket's `{role, run, node}` is stored via `serializeAttachment`, so after hibernation the DO
re-attaches sockets to nodes and resumes routing without a reconnect.
