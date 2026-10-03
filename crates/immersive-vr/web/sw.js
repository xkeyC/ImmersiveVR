// Service worker: makes the page an installable app and keeps its shell for
// when the PC is not reachable yet (the stream itself always needs the PC).
// Network first: the client changes with the server binary, so a cached copy
// is only a fallback. Browsers refuse to register it on a certificate they do
// not trust (the self-signed one); the app still installs from the manifest.

const CACHE = "immersive-vr-shell";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== location.origin || url.pathname === "/ws") return;
  event.respondWith((async () => {
    try {
      const response = await fetch(request);
      if (response.ok) {
        const cache = await caches.open(CACHE);
        cache.put(request, response.clone());
      }
      return response;
    } catch (error) {
      const cached = await caches.match(request);
      if (cached) return cached;
      throw error;
    }
  })());
});
