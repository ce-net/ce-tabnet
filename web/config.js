// ce-tabnet — front-end configuration: the ONE place the coordinator URL is decided.
//
// Both join.html (stage tabs) and orchestrator.html (operator console) import
// `coordinatorBase()` from here so they always agree on where the Cloudflare
// Worker + Durable Object lives. The runtime opens:  <base>/run/<run>/ws?role=...
//
// Resolution order (first match wins):
//   1. ?hub=...  query param            — explicit override, e.g. a workers.dev URL.
//   2. window.CE_TABNET_HUB             — set inline before this module loads (deploys).
//   3. localhost / 127.0.0.1 / file://  — local `dev/serve.js` + `wrangler dev`  => ws://127.0.0.1:8787
//   4. same origin + "/tabnet"          — production route under ce-net.com (see wrangler.jsonc routes).
//
// The coordinator base is a WebSocket origin (ws:// or wss://). The HTTP egress
// endpoints (/run/:id, /run/:id/state, /run/:id/prompt, /run/:id/tokens) share the
// same host, so httpBase() just swaps the scheme.

// Default port for `wrangler dev` (Cloudflare's local Worker emulator).
export const LOCAL_DEV_PORT = 8787;

function isLocal() {
  if (typeof location === "undefined") return true;
  return (
    location.protocol === "file:" ||
    ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)
  );
}

// The WebSocket base URL of the coordinator (no trailing slash, no path beyond /tabnet).
export function coordinatorBase(search = (typeof location !== "undefined" ? location.search : "")) {
  const params = new URLSearchParams(search || "");
  const fromQuery = params.get("hub");
  if (fromQuery) return stripTrailingSlash(fromQuery);

  if (typeof window !== "undefined" && window.CE_TABNET_HUB) {
    return stripTrailingSlash(window.CE_TABNET_HUB);
  }

  if (isLocal()) return `ws://127.0.0.1:${LOCAL_DEV_PORT}`;

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/tabnet`;
}

// HTTP(S) base for the same coordinator (token SSE egress, /state, /prompt over HTTP).
export function httpBase(wsBase = coordinatorBase()) {
  return wsBase.replace(/^ws/, "http");
}

function stripTrailingSlash(u) {
  return u.endsWith("/") ? u.slice(0, -1) : u;
}
