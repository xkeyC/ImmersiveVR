// WebXR Layers: the screen as a compositor cylinder layer.
//
// The frame from the PC already holds both eyes (left on top, right below);
// each half is copied into that eye's image of a stereo layer once per new
// video frame, at its own resolution, and the headset's compositor shows it every display
// frame, reprojected and filtered by itself. Compared with drawing the
// screen into the eye buffers each display frame this costs far less GPU and
// keeps text sharper. The panel and pointer rays go into a transparent
// projection layer composited on top. With passthrough, an all-around black
// equirect layer at the bottom dims the room (the panel's background setting).

import { gl } from "./gl.js";
import { rotateY } from "./math.js";
import { anchor, drawEyePicture, geometryVersion, screenGeometry } from "./screen.js";
import { settings } from "./settings.js";
import { stream } from "./stream.js";

/**
 * Makes `sub`'s image the framebuffer's color target: its slice of a texture
 * array (imageIndex), or the texture itself. If that is incomplete, tries the
 * other way. Returns the framebuffer status.
 */
function attach(sub, array) {
  const target = (layered) => {
    // Detach first: a failed attachment would leave the previous one in place.
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
    if (layered) gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, sub.colorTexture, 0, sub.imageIndex ?? 0);
    else gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sub.colorTexture, 0);
    return gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  };
  const layered = array || (sub.imageIndex ?? 0) > 0;
  let status = target(layered);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.getError();  // the failed attachment's error, if any
    status = target(!layered);
  }
  return status;
}

/** How the eye images are stored, for the diagnostics line, e.g. "texture-array 2560×1440 L0@0,0 R1@0,0 同纹理". */
function describe(textureType, subs) {
  const { width, height } = subs[0].sub.viewport;
  const eyes = subs.map(({ eye, sub, status }) => `${eye[0].toUpperCase()}${sub.imageIndex ?? "-"}@${sub.viewport.x},${sub.viewport.y}` +
    (status === gl.FRAMEBUFFER_COMPLETE ? "" : ` FB 0x${status.toString(16)}`));
  const shared = subs[0].sub.colorTexture === subs[1].sub.colorTexture ? "同纹理" : "分纹理";
  return `${textureType} ${width}×${height} ${eyes.join(" ")} ${shared}`;
}

/** Whether this browser can make the layers this module needs. */
export function layersAvailable() {
  return typeof XRWebGLBinding !== "undefined" &&
    typeof XRWebGLBinding.prototype.createCylinderLayer === "function" &&
    typeof XRWebGLBinding.prototype.createProjectionLayer === "function";
}

export class ScreenLayers {
  /** Throws if the session does not support layers. */
  constructor(session, space) {
    this.session = session;
    this.space = space;
    this.binding = new XRWebGLBinding(session, gl);
    this.projection = this.binding.createProjectionLayer({
      textureType: "texture",
      colorFormat: gl.RGBA8,
      depthFormat: 0,
      scaleFactor: 1.0,
    });
    // Composite with alpha: transparent wherever no panel or ray is drawn.
    if ("blendTextureSourceAlpha" in this.projection) this.projection.blendTextureSourceAlpha = true;
    this.framebuffer = gl.createFramebuffer();
    this.dim = null;
    this.drawnBackground = null;
    if (session.environmentBlendMode && session.environmentBlendMode !== "opaque" &&
        typeof this.binding.createEquirectLayer === "function") {
      try {
        this.dim = this.binding.createEquirectLayer({
          space, viewPixelWidth: 16, viewPixelHeight: 8, layout: "mono",
          textureType: "texture", colorFormat: gl.RGBA8, isStatic: false,
        });
        Object.assign(this.dim, {
          radius: 0, centralHorizontalAngle: Math.PI * 2, upperVerticalAngle: Math.PI / 2, lowerVerticalAngle: -Math.PI / 2,
        });
        if ("blendTextureSourceAlpha" in this.dim) this.dim.blendTextureSourceAlpha = true;
      } catch (error) {
        console.warn("[ivr] no equirect layer: the room cannot be dimmed", error);
        this.dim = null;
      }
    }
    this.cylinder = null;
    this.size = "";
    this.placement = "";
    this.drawnUploads = -1;
    /** How the compositor stores the eye images (diagnostics). */
    this.info = "";
    session.updateRenderState({ layers: this.stack() });
  }

  /** The layers bottom to top: the dimming, the screen, the panel. */
  stack() {
    return [this.dim, this.cylinder, this.projection].filter(Boolean);
  }

  /** Fills the dimming layer with black as opaque as the background setting leaves it. */
  drawDim(xrFrame) {
    if (!this.dim) return;
    if (settings.background === this.drawnBackground && !this.dim.needsRedraw) return;
    const sub = this.binding.getSubImage(this.dim, xrFrame);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sub.colorTexture, 0);
    const v = sub.viewport;
    gl.viewport(v.x, v.y, v.width, v.height);
    gl.clearColor(0, 0, 0, 1 - settings.background);
    gl.clear(gl.COLOR_BUFFER_BIT);
    this.drawnBackground = settings.background;
  }

  /** A cylinder layer matching the stream's picture size; recreated when it changes. */
  ensureCylinder() {
    const eye = stream.info?.layout.left;
    if (!eye) return false;
    const size = `${eye.width}x${eye.height}`;
    if (this.cylinder && this.size === size) return true;
    this.cylinder?.destroy?.();
    // One image per eye. (Quest's browser showed a packed stereo-top-bottom
    // layer's whole texture to the right eye only, and a stereo layer of
    // plain textures only to the left eye: its eye images are the slices of
    // a texture array.) Plain textures where arrays are not offered.
    for (const textureType of ["texture-array", "texture"]) {
      try {
        this.cylinder = this.binding.createCylinderLayer({
          space: this.space,
          viewPixelWidth: eye.width,
          viewPixelHeight: eye.height,
          layout: "stereo",
          textureType,
          colorFormat: gl.RGBA8,
          isStatic: false,
        });
        this.textureType = textureType;
        break;
      } catch (error) {
        if (textureType === "texture") throw error;
        console.warn("[ivr] no texture-array cylinder layer", error);
      }
    }
    // Supersampling and sharpening for text where the compositor offers it.
    try { if ("quality" in this.cylinder) this.cylinder.quality = "text-optimized"; } catch { /* not offered */ }
    this.size = size;
    this.info = "";
    this.placement = "";
    this.drawnUploads = -1;
    this.session.updateRenderState({ layers: this.stack() });
    return true;
  }

  /** Moves and bends the cylinder to the current screen settings and anchor. */
  place() {
    const key = `${geometryVersion}|${anchor.position}|${anchor.yaw}`;
    if (key === this.placement) return;
    this.placement = key;
    const { radius, centralAngle, aspectRatio, center } = screenGeometry();
    const c = rotateY(center, anchor.yaw);
    const half = anchor.yaw / 2;
    this.cylinder.radius = radius;
    this.cylinder.centralAngle = centralAngle;
    this.cylinder.aspectRatio = aspectRatio;
    this.cylinder.transform = new XRRigidTransform(
      { x: anchor.position[0] + c[0], y: anchor.position[1] + c[1], z: anchor.position[2] + c[2] },
      { x: 0, y: Math.sin(half), z: 0, w: Math.cos(half) },
    );
  }

  /** Copies the newest frame into the layer if there is something new to show. */
  drawScreen(xrFrame, uploads) {
    this.drawDim(xrFrame);
    if (!this.ensureCylinder()) return;
    this.place();
    if (uploads === this.drawnUploads && !this.cylinder.needsRedraw) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    const subs = [];
    for (const eye of ["left", "right"]) {
      // The eye images may be slices of one texture array, separate
      // textures, or parts of one texture (the viewport says which part).
      const sub = this.binding.getSubImage(this.cylinder, xrFrame, eye);
      const status = attach(sub, this.textureType === "texture-array");
      const v = sub.viewport;
      gl.viewport(v.x, v.y, v.width, v.height);
      if (!drawEyePicture(eye)) return;
      subs.push({ eye, sub, status });
    }
    if (!this.info) this.info = describe(this.textureType, subs);
    this.drawnUploads = uploads;
  }

  /** Clears each view of the overlay and lets `draw(view)` paint into it. */
  drawOverlay(viewerPose, draw) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.enable(gl.SCISSOR_TEST);
    gl.clearColor(0, 0, 0, 0);
    for (const view of viewerPose.views) {
      const sub = this.binding.getViewSubImage(this.projection, view);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sub.colorTexture, 0);
      const v = sub.viewport;
      gl.viewport(v.x, v.y, v.width, v.height);
      // Views may share one texture: clear only this view's part.
      gl.scissor(v.x, v.y, v.width, v.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      draw(view);
    }
    gl.disable(gl.SCISSOR_TEST);
  }
}
