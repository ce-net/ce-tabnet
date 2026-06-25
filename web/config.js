// ce-tabnet — front-end configuration.
//
// Coordination moved OFF Cloudflare onto the CE mesh. There is no longer a coordinator URL: tabs
// find each other through the CE DHT (a coordinator tab serve()s `tabnet/<run>`; peers locate() it).
// So this module's job shrank to ONE thing: tell the pages where to import the @ce-net/sdk module
// from, so both join.html (stage tabs) and orchestrator.html (operator console) agree.
//
// Resolution order for the SDK module specifier (first match wins):
//   1. ?sdk=...  query param         — explicit override (e.g. a local build URL).
//   2. window.CE_TABNET_SDK          — set inline before this module loads (deploys).
//   3. "@ce-net/sdk"                 — resolved by an <script type="importmap"> the page provides
//                                      (points the bare specifier at the built ESM bundle / CDN).
//
// The page is expected to either ship an import map for "@ce-net/sdk" or set window.CE_TABNET_SDK /
// ?sdk= to a direct ESM URL. mesh-transport.js imports the SDK lazily through this specifier.

// The @ce-net/sdk module specifier the pages import the mesh framework from.
export function sdkSpecifier(search = (typeof location !== "undefined" ? location.search : "")) {
  const params = new URLSearchParams(search || "");
  const fromQuery = params.get("sdk");
  if (fromQuery) return fromQuery;
  if (typeof window !== "undefined" && window.CE_TABNET_SDK) return window.CE_TABNET_SDK;
  return "@ce-net/sdk";
}
