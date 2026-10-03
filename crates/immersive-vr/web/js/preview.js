// Desktop preview: the XR scene side by side (left eye | right eye) from a
// fixed head; the mouse is the pointer.

import { canvas, gl, uploadFrame } from "./gl.js";
import { multiply, perspective, transform, translation } from "./math.js";
import { buttonCenter, drawPanel, markPanelDirty, panel, placePanel, select, updateHover } from "./panel.js";
import { anchor, drawScreen } from "./screen.js";
import { $, app, showStart } from "./ui.js";

const IPD = 0.064;
const FOV = 70 * Math.PI / 180;
let mouseRay = null;

export function startPreview() {
  app.previewing = true;
  $("start").style.display = "none";
  for (const id of ["view", "exit-preview", "hint"]) $(id).style.display = "block";
  Object.assign(anchor, { position: [0, 0, 0], yaw: 0, set: true });
  placePanel([0, 0, 0], 0);
  panel.visible = true;
  markPanelDirty();
  resize();
  requestAnimationFrame(frame);
}

export function stopPreview() {
  app.previewing = false;
  for (const id of ["view", "exit-preview", "hint"]) $(id).style.display = "none";
  showStart();
}

function resize() {
  if (!app.previewing) return;
  canvas.width = Math.round(canvas.clientWidth * devicePixelRatio);
  canvas.height = Math.round(canvas.clientHeight * devicePixelRatio);
}

function eyes() {
  const half = canvas.width / 2;
  const projection = perspective(FOV, half / canvas.height, 0.05, 100);
  return ["left", "right"].map((eye, i) => {
    const x = (i === 0 ? -1 : 1) * IPD / 2;
    return { eye, viewport: [i * half, 0, half, canvas.height], viewProjection: multiply(projection, translation(-x, 0, 0)) };
  });
}

function frame() {
  if (!app.previewing) return;
  requestAnimationFrame(frame);
  uploadFrame();
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.02, 0.02, 0.025, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  updateHover(mouseRay ? [{ ...mouseRay, length: 2.5 }] : []);
  for (const eye of eyes()) {
    gl.viewport(...eye.viewport);
    drawScreen(eye.viewProjection, eye.eye);
    drawPanel(eye.viewProjection, []);
  }
}

function rayFromMouse(event) {
  const bounds = canvas.getBoundingClientRect();
  const px = (event.clientX - bounds.left) * devicePixelRatio, py = (event.clientY - bounds.top) * devicePixelRatio;
  const half = canvas.width / 2;
  const right = px >= half;
  const nx = ((px - (right ? half : 0)) / half) * 2 - 1, ny = 1 - (py / canvas.height) * 2;
  const t = Math.tan(FOV / 2), aspect = half / canvas.height;
  const d = [nx * t * aspect, ny * t, -1];
  const length = Math.hypot(...d);
  return { origin: [(right ? 1 : -1) * IPD / 2, 0, 0], direction: d.map((v) => v / length) };
}

/** Client coordinates of a panel button's center in the left eye (for automated checks). */
export function locateButton(action) {
  const center = app.previewing && buttonCenter(action);
  if (!center) return null;
  const clip = transform(eyes()[0].viewProjection, [...center, 1]);
  const bounds = canvas.getBoundingClientRect();
  return {
    x: bounds.left + (clip[0] / clip[3] + 1) / 2 * bounds.width / 2,
    y: bounds.top + (1 - clip[1] / clip[3]) / 2 * bounds.height,
  };
}

window.addEventListener("resize", resize);
canvas.addEventListener("mousemove", (event) => { mouseRay = rayFromMouse(event); });
canvas.addEventListener("mouseleave", () => { mouseRay = null; });
canvas.addEventListener("click", (event) => {
  const ray = rayFromMouse(event);
  select(ray.origin, ray.direction, { position: [0, 0, 0], yaw: 0 });
});
window.addEventListener("keydown", (event) => {
  if (!app.previewing) return;
  if (event.key === "Escape") stopPreview();
  if (event.key === "m" || event.key === "M") { panel.visible = !panel.visible; markPanelDirty(); }
});
