// The WebGL2 context, shader and mesh helpers, and the video texture.

import { noteShown, takeFrame } from "./stream.js";

export const canvas = document.getElementById("view");
export const gl = canvas.getContext("webgl2", { xrCompatible: true, antialias: true, alpha: true });

/** Compiles and links a program; attributes get locations 0, 1, ... in order. */
export function program(vertex, fragment, attributes, uniforms) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  };
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vertex));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fragment));
  attributes.forEach((name, index) => gl.bindAttribLocation(p, index, name));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (const name of uniforms) u[name] = gl.getUniformLocation(p, name);
  return { p, u };
}

/** A vertex array of `stride` floats per vertex: position (3) and, if stride > 3, texcoord (2). */
export function mesh(data, stride) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride * 4, 0);
  if (stride > 3) {
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride * 4, 12);
  }
  return { vao, buffer, count: data.length / stride };
}

export function texture(minFilter) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

export const TEXTURED_VS = `#version 300 es
in vec3 position;
in vec2 texcoord;
uniform mat4 mvp;
out vec2 uv;
void main() { uv = texcoord; gl_Position = mvp * vec4(position, 1.0); }`;

// The video texture: mipmapped and anisotropic so text stays steady when the
// screen is small or seen at an angle.
export const video = {
  texture: texture(gl.LINEAR_MIPMAP_LINEAR),
  /** Mipmaps help when the screen is drawn small; not needed when drawn 1:1. */
  mipmaps: true,
  ready: false,
  /** Frames uploaded so far, and the last upload's size (diagnostics). */
  uploads: 0,
  size: "",
};
const anisotropic = gl.getExtension("EXT_texture_filter_anisotropic");
if (anisotropic) {
  gl.texParameterf(gl.TEXTURE_2D, anisotropic.TEXTURE_MAX_ANISOTROPY_EXT,
    Math.min(8, gl.getParameter(anisotropic.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
}

export function setVideoMipmaps(on) {
  video.mipmaps = on;
  gl.bindTexture(gl.TEXTURE_2D, video.texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, on ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
  if (on && video.ready) gl.generateMipmap(gl.TEXTURE_2D);
}

/** Uploads the newest decoded frame, if there is one; true if it did. */
export function uploadFrame() {
  const next = takeFrame();
  if (!next) return false;
  gl.bindTexture(gl.TEXTURE_2D, video.texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, next.frame);
  if (video.mipmaps) gl.generateMipmap(gl.TEXTURE_2D);
  video.size = `${next.frame.displayWidth}×${next.frame.displayHeight}`;
  video.uploads++;
  noteShown(next.frame.timestamp);
  next.frame.close();
  video.ready = true;
  return true;
}
