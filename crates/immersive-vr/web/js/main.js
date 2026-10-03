// ImmersiveVR client: wires settings, stream, rendering and the two ways in
// (an XR session or the desktop preview).

import { gl, video } from "./gl.js";
import { startAudio } from "./audio.js";
import { markPanelDirty, panel, panelHooks, runAction } from "./panel.js";
import { locateButton, startPreview, stopPreview } from "./preview.js";
import { markScreenDirty } from "./screen.js";
import { loadSettings, onSettingsChange, SERVER_KEYS, settings } from "./settings.js";
import { connect, decoderState, onStreamInfo, onSummary, scheduleSend, stream } from "./stream.js";
import { $, app } from "./ui.js";
import { endXR, inSession, recenterXR, setupXR, updateEnterLabel } from "./xr.js";

const GEOMETRY = ["size", "curvature", "distance", "height"];

onSettingsChange((changes) => {
  if (Object.keys(changes).some((key) => SERVER_KEYS.includes(key))) scheduleSend();
  if (Object.keys(changes).some((key) => GEOMETRY.includes(key))) markScreenDirty();
  markPanelDirty();
});
onStreamInfo(() => { markScreenDirty(); markPanelDirty(); });
onSummary(() => {
  // getError reports (and clears) errors since the last call: once a second is enough.
  const error = gl.getError();
  if (error) app.glError = error;
  markPanelDirty();
});

Object.assign(panelHooks, {
  recenter: () => (inSession() ? recenterXR() : null),
  exit: () => (inSession() ? endXR() : stopPreview()),
});

$("preview").onclick = () => { startAudio(); startPreview(); };
$("exit-preview").onclick = stopPreview;

// Diagnostics and hooks for automated checks (scripts/page_check.mjs).
window.ivrState = () => ({
  info: stream.info && {
    generation: stream.info.generation, codec: stream.info.codec,
    codec_string: stream.info.codec_string, width: stream.info.layout.left.width,
  },
  decoder: decoderState(), haveTexture: video.ready, uploads: video.uploads,
  previewing: app.previewing, panel: { visible: panel.visible, hover: panel.hover, buttons: panel.buttons.length },
  settings: { ...settings }, status: $("status").textContent, problem: stream.problem,
});
window.ivrAction = runAction;
window.ivrLocate = locateButton;

// Installable app (PWA); refused on an untrusted certificate, which is fine.
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch((error) => console.info("[ivr] no service worker:", error.message));
}

await loadSettings();
connect();
await setupXR();
updateEnterLabel();
if (new URLSearchParams(location.search).has("preview")) startPreview();
