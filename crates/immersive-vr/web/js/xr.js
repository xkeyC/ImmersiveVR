// WebXR sessions: immersive-vr, or immersive-ar for passthrough (Quest shows
// the room behind a transparent clear color). The screen is a compositor
// layer where WebXR Layers are available (see layers.js), otherwise drawn
// into the eye buffers every display frame.

import { gl, setVideoMipmaps, uploadFrame, video } from "./gl.js";
import { layersAvailable, ScreenLayers } from "./layers.js";
import { multiply, yawOf } from "./math.js";
import { drawPanel, markPanelDirty, panel, placePanel, select, updateHover } from "./panel.js";
import { anchor, drawScreen } from "./screen.js";
import { setSettings, settings } from "./settings.js";
import { startAudio } from "./audio.js";
import { stream } from "./stream.js";
import { $, app, showStart, t } from "./ui.js";

let session = null;
let space = null;
let layers = null;
let frames = 0;
let framesSince = performance.now();
let recenterRequested = false;

export const inSession = () => session !== null;

/** Checks what the browser supports and enables the start button. */
export async function setupXR() {
  if (!navigator.xr) {
    $("enter").textContent = window.isSecureContext
      ? t("此浏览器不支持 WebXR", "This browser has no WebXR")
      : t("WebXR 不可用（请用 https 地址打开）", "WebXR is unavailable (open the https address)");
    return;
  }
  const [vr, ar] = await Promise.all([
    navigator.xr.isSessionSupported("immersive-vr").catch(() => false),
    navigator.xr.isSessionSupported("immersive-ar").catch(() => false),
  ]);
  app.arSupported = ar;
  markPanelDirty();
  if (!vr && !ar) { $("enter").textContent = t("此设备不支持沉浸式 WebXR", "This device has no immersive WebXR"); return; }
  $("enter").disabled = false;
  // Passthrough on or off before entering, beside the button (also in the panel).
  $("passthrough").hidden = !ar;
  $("passthrough").onclick = () => {
    setSettings({ passthrough: !settings.passthrough });
    updateEnterLabel();
  };
  updateEnterLabel();
  $("enter").onclick = () => {
    startAudio();
    startXR(settings.passthrough && ar ? "immersive-ar" : "immersive-vr");
  };
}

export function updateEnterLabel() {
  if (!$("enter").disabled) $("enter").textContent = t("进入 VR", "Enter VR");
  const toggle = $("passthrough");
  toggle.textContent = settings.passthrough ? t("透视：开", "Passthrough: on") : t("透视：关", "Passthrough: off");
  toggle.classList.toggle("on", settings.passthrough);
}

async function startXR(mode) {
  try {
    const next = await navigator.xr.requestSession(mode, { optionalFeatures: ["local-floor", "layers"] });
    session = next;
    app.inXR = true;
    app.xrMode = mode;
    app.blendMode = next.environmentBlendMode ?? "?";
    space = await next.requestReferenceSpace("local");
    layers = null;
    if (layersAvailable()) {
      try { layers = new ScreenLayers(next, space); }
      catch (error) { console.warn("[ivr] WebXR layers unavailable, drawing into the eye buffers", error); }
    }
    app.layers = layers !== null;
    // A refresh rate that is a multiple of the stream's (60 fps: 120 Hz)
    // shows every frame for the same number of refreshes; 72 or 90 Hz cannot.
    const fps = stream.info?.fps ?? 60;
    const rates = next.supportedFrameRates ? [...next.supportedFrameRates] : [];
    const even = rates.filter((rate) => Math.round(rate) % fps === 0).sort((a, b) => a - b)[0];
    if (even && next.updateTargetFrameRate) {
      next.updateTargetFrameRate(even).catch((error) => console.warn("[ivr] frame rate", error));
    }
    app.frameRates = rates.join("/");
    if (layers) {
      next.updateRenderState({ depthNear: 0.05, depthFar: 100 });
      // The eye pictures are drawn at the picture's own size: no minification.
      setVideoMipmaps(false);
    } else {
      // Same layer for both modes (alpha kept: passthrough needs it, VR ignores it).
      const layer = new XRWebGLLayer(next, gl, { framebufferScaleFactor: 1.2, alpha: true, antialias: false });
      next.updateRenderState({ baseLayer: layer, depthNear: 0.05, depthFar: 100 });
    }
    anchor.set = false;
    panel.visible = true;
    panel.placed = false;
    markPanelDirty();
    next.addEventListener("select", (event) => {
      const pose = event.frame.getPose(event.inputSource.targetRaySpace, space);
      const viewer = event.frame.getViewerPose(space);
      if (pose && viewer) select(...rayFromPose(pose.transform), poseInfo(viewer.transform));
    });
    next.addEventListener("squeeze", (event) => {
      const viewer = event.frame.getViewerPose(space);
      panel.visible = !panel.visible;
      if (panel.visible && viewer) { const v = poseInfo(viewer.transform); placePanel(v.position, v.yaw); }
      markPanelDirty();
    });
    next.addEventListener("end", () => {
      session = null;
      layers = null;
      app.inXR = false;
      app.xrMode = "";
      setVideoMipmaps(true);
      markPanelDirty();
      showStart();
    });
    $("start").style.display = "none";
    next.requestAnimationFrame(frame);
  } catch (error) {
    showStart(t(`无法进入 XR：${error.message}（可在开始界面重新点击「进入 VR」）`,
      `Cannot enter XR: ${error.message} (press "Enter VR" on the start screen to try again)`));
  }
}

export function endXR() {
  session?.end();
}

export function recenterXR() {
  recenterRequested = true;
}

function rayFromPose(transform) {
  const m = transform.matrix;
  return [[m[12], m[13], m[14]], [-m[8], -m[9], -m[10]]];
}

function poseInfo(transform) {
  const p = transform.position;
  return { position: [p.x, p.y, p.z], yaw: yawOf(transform.matrix) };
}

function frame(time, xrFrame) {
  const current = xrFrame.session;
  current.requestAnimationFrame(frame);
  frames++;
  if (time - framesSince >= 1000 || performance.now() - framesSince >= 1000) {
    app.xrFps = frames * 1000 / (performance.now() - framesSince);
    frames = 0;
    framesSince = performance.now();
  }
  const viewerPose = xrFrame.getViewerPose(space);
  if (!viewerPose) return;
  const viewer = poseInfo(viewerPose.transform);
  if (!anchor.set || recenterRequested) {
    Object.assign(anchor, viewer, { set: true });
    if (recenterRequested || !panel.placed) placePanel(viewer.position, viewer.yaw);
    recenterRequested = false;
  }
  uploadFrame();
  const pointers = [];
  for (const source of current.inputSources) {
    const pose = source.targetRaySpace && xrFrame.getPose(source.targetRaySpace, space);
    if (pose) {
      const [origin, direction] = rayFromPose(pose.transform);
      pointers.push({ origin, direction, length: 2.5 });
    }
  }
  updateHover(pointers);
  const viewProjection = (view) => multiply(view.projectionMatrix, view.transform.inverse.matrix);
  if (layers) {
    layers.drawScreen(xrFrame, video.uploads);
    app.layerInfo = layers.info;
    layers.drawOverlay(viewerPose, (view) => drawPanel(viewProjection(view), pointers));
    return;
  }
  const layer = current.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  const passthrough = current.environmentBlendMode && current.environmentBlendMode !== "opaque";
  // Passthrough: black over the room, as opaque as the panel's background setting says.
  if (passthrough) gl.clearColor(0, 0, 0, 1 - settings.background);
  else gl.clearColor(0.02, 0.02, 0.025, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  for (const view of viewerPose.views) {
    const viewport = layer.getViewport(view);
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    drawScreen(viewProjection(view), view.eye);
    drawPanel(viewProjection(view), pointers);
  }
}
