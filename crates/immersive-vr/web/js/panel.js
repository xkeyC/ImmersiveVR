// The control panel: a 2D canvas on a quad in front of the viewer, pointed at
// with a controller or hand ray (or the mouse in the preview) and clicked with
// trigger / pinch ("select").

import { gl, mesh, program, texture, TEXTURED_VS, video } from "./gl.js";
import { clamp, multiply, rotateY, rotationY, scaling, snap, translation } from "./math.js";
import { setSettings, settings } from "./settings.js";
import { CODEC_LABELS, stream } from "./stream.js";
import { audio } from "./audio.js";
import { app } from "./ui.js";

const PANEL_PX = [1280, 1040];
const PANEL_M = [0.8, 0.65];

const STEPPERS = [
  { key: "divergence", label: "立体强度", step: 0.1, min: 0, max: 10, format: (v) => `${v.toFixed(1)} %` },
  { key: "convergence", label: "会聚（出屏 ↔ 入屏）", step: 0.05, min: 0, max: 1, format: (v) => v.toFixed(2) },
  { key: "distance", label: "屏幕距离", step: 0.25, min: 0.75, max: 8, format: (v) => `${v.toFixed(2)} m` },
  { key: "size", label: "屏幕宽度", step: 0.2, min: 0.6, max: 8, format: (v) => `${v.toFixed(1)} m` },
  { key: "curvature", label: "曲率", step: 0.1, min: 0, max: 1, format: (v) => (v < 0.01 ? "平面" : v.toFixed(1)) },
  { key: "height", label: "高度", step: 0.1, min: -1.5, max: 1.5, format: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)} m` },
];

export const panel = { visible: true, placed: false, position: [0, -0.28, -0.85], yaw: 0, hover: null, buttons: [] };

/** What the panel's session buttons do; set by the XR and preview modules. */
export const panelHooks = { recenter() {}, exit() {} };

const canvas2d = document.createElement("canvas");
[canvas2d.width, canvas2d.height] = PANEL_PX;
const ctx = canvas2d.getContext("2d");
const panelTexture = texture(gl.LINEAR);
const quad = mesh(new Float32Array([-1, -1, 0, 0, 1, 1, -1, 0, 1, 1, -1, 1, 0, 0, 0, 1, 1, 0, 1, 0]), 5);
const rays = mesh(new Float32Array(12), 3);
let dirty = true;

const panelShader = program(TEXTURED_VS, `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D image;
out vec4 color;
void main() { color = texture(image, uv); }`, ["position", "texcoord"], ["mvp", "image"]);

const lineShader = program(`#version 300 es
in vec3 position;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(position, 1.0); }`, `#version 300 es
precision mediump float;
uniform vec4 tint;
out vec4 color;
void main() { color = tint; }`, ["position"], ["mvp", "tint"]);

/** Redraws the panel before its next frame. */
export function markPanelDirty() {
  dirty = true;
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function redraw() {
  dirty = false;
  const [W, H] = PANEL_PX;
  const buttons = [];
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(18, 20, 26, 0.92)";
  roundRect(0, 0, W, H, 36); ctx.fill();
  ctx.textBaseline = "middle";
  const text = (s, x, y, size = 34, color = "#e6e8ec", align = "left") => {
    ctx.font = `${size}px system-ui, sans-serif`; ctx.fillStyle = color; ctx.textAlign = align; ctx.fillText(s, x, y);
  };
  const button = (label, x, y, w, h, action, active = false) => {
    ctx.fillStyle = active ? "#4c8dff" : panel.hover === action ? "#3a4252" : "#262b35";
    roundRect(x, y, w, h, 14); ctx.fill();
    text(label, x + w / 2, y + h / 2, 32, "#fff", "center");
    buttons.push({ x, y, w, h, action });
  };
  // A track to click anywhere on: the value is where along it.
  const slider = (key, x, y, w, h) => {
    const value = settings[key];
    ctx.fillStyle = panel.hover === `slide:${key}` ? "#3a4252" : "#262b35";
    roundRect(x, y, w, h, h / 2); ctx.fill();
    ctx.fillStyle = "#4c8dff";
    roundRect(x, y, Math.max(h, w * value), h, h / 2); ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(x + Math.max(h / 2, w * value - h / 2), y + h / 2, h * 0.6, 0, Math.PI * 2); ctx.fill();
    // Clickable a little past both ends, so the ends (0 and 100 %) are easy to hit.
    buttons.push({ x: x - 40, y: y - 14, w: w + 80, h: h + 28, action: `slide:${key}`, track: { x, w } });
  };

  text("ImmersiveVR", 48, 56, 44);
  text(stream.problem || stream.summary || "等待画面…", 48, 100, 26, stream.problem ? "#ffb35c" : "#8a8f99");
  // Diagnostics: where a black screen comes from (no frames, no decode, no upload, GL error).
  const xr = app.inXR ? ` · ${app.layers ? `合成层 ${app.layerInfo}` : "眼缓冲"} · XR ${app.xrFps.toFixed(0)} fps` : "";
  const sound = audio.bufferedMs === null ? "" : ` · 声音缓冲 ${audio.bufferedMs.toFixed(0)} ms${audio.underruns ? `（断 ${audio.underruns}）` : ""}`;
  text(`${app.xrMode || "预览"}${app.blendMode && app.inXR ? ` · ${app.blendMode}` : ""}${xr} · 收 ${stream.received.toFixed(0)} / ` +
    `解码 ${stream.decoded.toFixed(0)} fps · 上传 ${video.uploads}${video.size ? ` (${video.size})` : ""} · ` +
    `GL ${app.glError ? `0x${app.glError.toString(16)}` : "正常"}${sound}`, 48, 132, 22, "#6f7682");

  let y = 150;
  const row = 92;
  text("传输分辨率", 48, y + 34);
  const resolutions = stream.info ? stream.info.resolutions : [1080, 1440, 2160];
  resolutions.forEach((h, i) => button(`${h}p`, 380 + i * 190, y, 170, 68, `resolution:${h}`, settings.resolution === h));
  y += row;
  text("编码", 48, y + 34);
  ["hevc", "av1"].forEach((c, i) => button(CODEC_LABELS[c], 380 + i * 190, y, 170, 68, `codec:${c}`, settings.codec === c));
  y += row;
  for (const s of STEPPERS) {
    text(s.label, 48, y + 34);
    button("−", 380, y, 110, 68, `step:${s.key}:-1`);
    text(s.format(settings[s.key]), 640, y + 34, 34, "#fff", "center");
    button("+", 790, y, 110, 68, `step:${s.key}:1`);
    y += row - 10;
  }
  // Passthrough (chosen on the start screen): how much of the room shows.
  if (app.xrMode === "immersive-ar") {
    text("背景透明度", 48, y + 34);
    slider("background", 380, y + 14, 520, 40);
    text(`${Math.round(settings.background * 100)} %`, 1010, y + 34, 34, "#fff", "center");
    y += row - 10;
  }
  y += 14;
  button("重新居中", 48, y, 220, 72, "recenter");
  button("隐藏面板", 288, y, 220, 72, "hide");
  button(app.inXR ? "退出 VR" : "退出预览", 528, y, 220, 72, "exit");
  button(settings.mute_pc ? "电脑静音：开" : "电脑静音：关", 768, y, 260, 72, "mutepc", settings.mute_pc);
  text("握持键开关面板 · 面板隐藏时按扳机/捏合唤出", 48, H - 36, 24, "#8a8f99");
  panel.buttons = buttons;

  gl.bindTexture(gl.TEXTURE_2D, panelTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas2d);
}

export function runAction(action) {
  const [kind, a, b] = action.split(":");
  if (kind === "resolution") setSettings({ resolution: Number(a) });
  else if (kind === "codec") setSettings({ codec: a });
  else if (kind === "mutepc") setSettings({ mute_pc: !settings.mute_pc });
  else if (kind === "step") {
    const s = STEPPERS.find((x) => x.key === a);
    setSettings({ [a]: clamp(snap(settings[a] + Number(b) * s.step, s.step / 100), s.min, s.max) });
  } else if (kind === "slide") setSettings({ [a]: clamp(Number(b), 0, 1) });
  else if (kind === "recenter") panelHooks.recenter();
  else if (kind === "hide") panel.visible = false;
  else if (kind === "exit") panelHooks.exit();
  dirty = true;
}

const panelMatrix = () => multiply(multiply(translation(...panel.position), rotationY(panel.yaw)),
  scaling(PANEL_M[0] / 2, PANEL_M[1] / 2, 1));

/** Where a ray (world space) hits the panel: canvas pixel coordinates and distance. */
function hitPanel(origin, direction) {
  if (!panel.visible) return null;
  const o = rotateY(origin.map((v, i) => v - panel.position[i]), -panel.yaw);
  const d = rotateY(direction, -panel.yaw);
  if (Math.abs(d[2]) < 1e-6) return null;
  const t = -o[2] / d[2];
  if (!(t > 0)) return null;
  const u = (o[0] + d[0] * t) / PANEL_M[0] + 0.5, v = 0.5 - (o[1] + d[1] * t) / PANEL_M[1];
  if (u < 0 || u > 1 || v < 0 || v > 1) return null;
  return { px: u * PANEL_PX[0], py: v * PANEL_PX[1], t };
}

/** The action under a hit: a button's, a slider's with where along it ("slide:key:0.42"), or "panel". */
function buttonAt(hit) {
  if (!hit) return null;
  const b = panel.buttons.find((b) => hit.px >= b.x && hit.px <= b.x + b.w && hit.py >= b.y && hit.py <= b.y + b.h);
  if (!b) return "panel";
  if (!b.track) return b.action;
  // Where along the track, in 5 % steps (so the left end is exactly 0).
  const along = Math.min(1, Math.max(0, (hit.px - b.track.x) / b.track.w));
  return `${b.action}:${(Math.round(along * 20) / 20).toFixed(2)}`;
}

/** Places the panel in front of a viewer (position, yaw), a little below eye level. */
export function placePanel(position, yaw) {
  const forward = rotateY([0, 0, -0.85], yaw);
  panel.position = [position[0] + forward[0], position[1] - 0.28, position[2] + forward[2]];
  panel.yaw = yaw;
  panel.placed = true;
}

/** A click along a ray: presses the button it hits, or brings back a hidden panel. */
export function select(origin, direction, viewer) {
  const action = buttonAt(hitPanel(origin, direction));
  if (!panel.visible) {
    panel.visible = true;
    placePanel(viewer.position, viewer.yaw);
  } else if (action && action !== "panel") {
    runAction(action);
  }
  dirty = true;
}

/** Hover state for this frame's pointer rays; sets each ray's drawn length. */
export function updateHover(pointers) {
  let hover = null;
  for (const ray of pointers) {
    const hit = hitPanel(ray.origin, ray.direction);
    ray.length = hit ? hit.t : 2.5;
    // A slider highlights as a whole, wherever along it.
    hover = hover ?? buttonAt(hit)?.replace(/^(slide:[^:]+):.*$/, "$1");
  }
  if (hover !== panel.hover) { panel.hover = hover; dirty = true; }
}

export function drawPanel(viewProjection, pointers) {
  if (dirty) redraw();
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  if (panel.visible) {
    const { p, u } = panelShader;
    gl.useProgram(p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, panelTexture);
    gl.uniform1i(u.image, 0);
    gl.uniformMatrix4fv(u.mvp, false, multiply(viewProjection, panelMatrix()));
    gl.bindVertexArray(quad.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  if (pointers.length) {
    const data = new Float32Array(pointers.flatMap(({ origin, direction, length }) =>
      [...origin, ...origin.map((v, i) => v + direction[i] * length)]));
    const { p, u } = lineShader;
    gl.useProgram(p);
    gl.uniformMatrix4fv(u.mvp, false, viewProjection);
    gl.uniform4fv(u.tint, [0.55, 0.75, 1.0, 0.8]);
    gl.bindVertexArray(rays.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, rays.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.LINES, 0, data.length / 3);
  }
  gl.disable(gl.BLEND);
}

/** The world-space center of a panel button (for automated checks). */
export function buttonCenter(action) {
  const b = panel.buttons.find((x) => x.action === action);
  if (!b) return null;
  const local = [((b.x + b.w / 2) / PANEL_PX[0] - 0.5) * PANEL_M[0], (0.5 - (b.y + b.h / 2) / PANEL_PX[1]) * PANEL_M[1], 0];
  const r = rotateY(local, panel.yaw);
  return [r[0] + panel.position[0], r[1] + panel.position[1], r[2] + panel.position[2]];
}
