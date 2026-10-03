// The virtual screen: anchored at the viewer's pose when a session starts (or
// on recenter), centered `distance` ahead and `height` above eye level, bent
// around the viewer by `curvature` (cylinder radius = distance / curvature).
// The PC renders both eyes (iw3 mlbw_l2) into one frame, left eye on top and
// right eye below; each eye here just shows its half.

import { gl, mesh, program, TEXTURED_VS, video } from "./gl.js";
import { IDENTITY, multiply, rotationY, translation } from "./math.js";
import { settings } from "./settings.js";
import { stream } from "./stream.js";

const shader = program(TEXTURED_VS, `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D frame;
uniform vec4 region;            // xy offset, zw size, in texture uv
out vec4 color;
void main() {
  color = vec4(texture(frame, region.xy + uv * region.zw).rgb, 1.0);
}`, ["position", "texcoord"], ["mvp", "frame", "region"]);

const COLUMNS = 96;
const strip = mesh(new Float32Array((COLUMNS + 1) * 2 * 5), 5);
// A full-viewport quad, top row of the picture at the top.
const fullscreen = mesh(new Float32Array([-1, -1, 0, 0, 1, 1, -1, 0, 1, 1, -1, 1, 0, 0, 0, 1, 1, 0, 1, 0]), 5);
let dirty = true;
/** Changes whenever the screen's shape or size settings change. */
export let geometryVersion = 0;

/** Where the screen hangs: a viewer position and yaw. */
export const anchor = { position: [0, 0, 0], yaw: 0, set: false };

/** The geometry follows the settings and the picture's aspect ratio. */
export function markScreenDirty() {
  dirty = true;
  geometryVersion++;
}

/** Below this curvature the screen is drawn flat (a very large cylinder for layers). */
export const FLAT = 0.01;

/**
 * The screen as a cylinder in anchor coordinates: radius, the arc it spans
 * (radians), width / height, and the cylinder's center (the screen's center
 * is `radius` in front of it, `distance` in front of the anchor).
 */
export function screenGeometry() {
  const color = stream.info?.layout.left;
  const aspectRatio = color ? color.width / color.height : 16 / 9;
  const { size, distance, height } = settings;
  const curvature = Math.max(settings.curvature, FLAT * 2);
  const radius = distance / curvature;
  return { radius, centralAngle: size / radius, aspectRatio, center: [0, height, radius - distance] };
}

function rebuild() {
  dirty = false;
  const color = stream.info?.layout.left;
  const aspect = color ? color.height / color.width : 9 / 16;
  const { size, distance, curvature, height } = settings;
  const half = size * aspect / 2;
  const data = [];
  for (let i = 0; i <= COLUMNS; i++) {
    const u = i / COLUMNS;
    let x, z;
    if (curvature < FLAT) {
      x = (u - 0.5) * size;
      z = -distance;
    } else {
      const radius = distance / curvature;
      const angle = (u - 0.5) * size / radius;
      x = radius * Math.sin(angle);
      z = radius - distance - radius * Math.cos(angle);
    }
    data.push(x, height + half, z, u, 0, x, height - half, z, u, 1);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, strip.buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.DYNAMIC_DRAW);
  strip.count = data.length / 5;
}

/** A region of the frame as texture uv. */
function region(r) {
  const { width, height } = stream.info.layout;
  return [r.x / width, r.y / height, r.width / width, r.height / height];
}

/** The half of the frame `eye` sees. */
function eyeRect(eye) {
  return stream.info.layout[eye === "right" ? "right" : "left"];
}

function bind(eye, mvp) {
  const { p, u } = shader;
  gl.useProgram(p);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, video.texture);
  gl.uniform1i(u.frame, 0);
  gl.uniformMatrix4fv(u.mvp, false, mvp);
  gl.uniform4fv(u.region, region(eyeRect(eye)));
}

/** Draws the screen as `eye` ("left" or "right") sees it, in 3D. */
export function drawScreen(viewProjection, eye) {
  if (!stream.info || !video.ready) return;
  if (dirty) rebuild();
  const model = multiply(translation(...anchor.position), rotationY(anchor.yaw));
  bind(eye, multiply(viewProjection, model));
  gl.bindVertexArray(strip.vao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, strip.count);
}

/** Fills the viewport with `eye`'s picture: that eye's image of a stereo compositor layer. */
export function drawEyePicture(eye) {
  if (!stream.info || !video.ready) return false;
  gl.bindVertexArray(fullscreen.vao);
  bind(eye, IDENTITY);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  return true;
}
