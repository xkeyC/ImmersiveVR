// DOM helpers and the page-level state shared by the modules.

export const $ = (id) => document.getElementById(id);

/** Where the page is: start screen, desktop preview or an XR session. */
export const app = {
  previewing: false,
  inXR: false,
  /** Whether the device offers `immersive-ar` (passthrough). */
  arSupported: false,
  /** The running session's mode and environment blend mode (diagnostics). */
  xrMode: "",
  blendMode: "",
  /** The last WebGL error seen (diagnostics); 0 = none. */
  glError: 0,
  /** Whether the screen is a compositor layer, and the XR frame rate (diagnostics). */
  layers: false,
  layerInfo: "",
  xrFps: 0,
};

export function setStatus(text) {
  $("status").textContent = text;
  console.info("[ivr]", text);
}

export function showStart(message) {
  $("start").style.display = "flex";
  if (message) setStatus(message);
}
