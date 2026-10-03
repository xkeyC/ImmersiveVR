# Monocular depth -> real-time stereo pair: survey for ImmersiveVR

> Research notes from the start of the project, kept for reference. The current design (iw3 mlbw_l2 on the PC, a CUDA warp kernel, NVENC) is described in the README.

Date: 2026-10-04. Scope: Windows desktop (1920x1080) plus Depth-Anything-V2-Small relative inverse depth (518x294, 0..1, larger = nearer, temporally smoothed). The stereo pair is synthesized on Quest 3 (WebGL2) or on the PC (RTX).

Shader at the time of the survey (since replaced by the near-to-far step search, see `crates/immersive-vr/web/index.html`):
`displayed(x) = source(x) + side*strength*(depth - convergence)`. It is solved with 4 fixed-point iterations `xs = x - side*k*(D(xs) - c)`. Defaults are k=0.012 (uv per eye), c=0.35, screen distance 2.5 m. It renders to an `XRWebGLLayer` with `framebufferScaleFactor: 1.5`.

---

## 0. TL;DR

1. **The fixed-point solver is the weakest part of the current pipeline.** The iteration `xs <- x - s*k*(D(xs)-c)` is a contraction only when `|k * dD/dx| < 1` in uv units. With a 518-texel depth map, a depth step of Δd across one texel has slope Δd*518 per uv. At k=0.012 the iteration stops converging once Δd > ~0.16, which covers nearly every object silhouette. At those edges it oscillates between the foreground and background solutions. The result is ghost or double edges, flicker, and no defined occlusion order. Every serious implementation (SuperDepth3D, Fubax VR.fx, DepthFlow, depthy) uses a **monotone linear search (steep-parallax / POM ray march) followed by refinement**. That search always returns the *nearest* intersecting surface, so foreground correctly occludes background.
2. **The best open reference is nagadomi/nunif `iw3` (MIT, weights also MIT).** Its default `row_flow_v3` and its `mlbw_*` methods are tiny networks. Their input is **only depth + divergence + convergence**, with no RGB. They output a per-pixel horizontal backward-warp offset at depth resolution: 1 layer for row_flow, 2 or 4 layers plus softmax weights for mlbw. The full-resolution work is just bilinear upsampling of that field plus `grid_sample`. That makes them a natural "PC computes the warp field, headset applies it" design. `iw3.desktop` exists and does real-time desktop streaming: PC-side warping, MJPEG half-SBS to the Quest browser, default `Any_V2_S`, divergence 1, convergence 1.
3. **What to implement first** (Quest, WebGL2):
   - (a) A cheap preprocessing pass at depth resolution: foreground "edge dilation" (iw3 Edge Fix / SuperDepth3D min filter), an optional background-depth channel, and border falloff.
   - (b) A per-eye ray-march warp with N ≈ 8-24 linear steps sized to the disparity range *in depth texels*, plus 2-3 bisection or secant steps, plus hole detection with background-side fill.
   - (c) Render it **once per decoded video frame** into two offscreen textures. Display them through a WebXR **quad layer** (stereo-left-right), not per display frame at 1.5x framebuffer scale.
4. **Later:** a PC-side iw3-style learned warp field (mlbw_l2/row_flow_v3, MIT) sent as a low-res offset/weight texture, or full PC forward warp plus lightweight inpaint (iw3 `forward_inpaint` / `mlbw_l2_inpaint`). Diffusion approaches (StereoCrafter, StereoDiffusion, Mono2Stereo) are far from real time.

---

## 1. Reference projects

### 1.1 nagadomi/nunif - iw3 (the most relevant)
- Repo: https://github.com/nagadomi/nunif (MIT; about 3.5k stars; last push 2026-09-18, so actively maintained). iw3 docs: https://github.com/nagadomi/nunif/blob/master/iw3/README.md
- Weights license: the author confirmed that row_flow_v3, mlbw and depth_aa weights are under the repo's MIT license (issue #718: https://github.com/nagadomi/nunif/issues/718). The iw3 README notes that DA-V2-Small is Apache-2.0, while B/L are CC-BY-NC and are not enabled by default.

**Stereo generation methods** (README table, https://github.com/nagadomi/nunif/blob/master/iw3/README.md, section "Stereo Generation Method"; code in `iw3/backward_warp.py`, `iw3/forward_warp.py`, `iw3/models/*.py`):

| method | what it does (from code) | notes |
|---|---|---|
| `grid_sample` / `backward` | Naive one-tap backward warp: `grid = mesh + (depth*shift - shift*conv)`, `F.grid_sample` bilinear, border padding (`apply_divergence_grid_sample`, backward_warp.py) | README: "Lots of ghost artifacts". This is what Desktop2Stereo does, and roughly what our shader does. |
| `forward` / `forward_fill` | "Depth order bilinear forward warping" (`depth_order_bilinear_forward_warp`, forward_warp.py). Each source pixel is splatted to floor/ceil target columns with bilinear weights. Writes are ordered by `argsort(depth)` with deterministic `index_copy_`, so nearer pixels overwrite farther ones (a z-buffer by sort). `fix_layered_holes` marks pixels where the warped source index is non-monotonic as holes. `shift_fill` then fills holes by repeatedly copying the horizontal neighbour from one direction ("super simple inpainting"). | Non-ML. Self-benchmark in the file: 512² 230 FPS, but **1080p only 22 FPS** on an RTX 3070 Ti (`_bench`). `forward_fill` output is also the training target for row_flow_v3. |
| `row_flow_v3` (**default**, CLI `--method row_flow` resolves to v3) | `RowFlowV3` (models/row_flow_v3.py): pixel_unshuffle (1x8 horizontal) -> 2 window-attention blocks (C=64) -> conv that outputs **one channel: a horizontal backward offset**. It is applied by `grid_sample` at full resolution after bilinear upsampling of the offset field (`backward_warp()` interpolates the grid when sizes differ). | Input features = `[depth, divergence_feat, convergence_feat]` only (`make_input_tensor(None, ...)`); **no RGB input**. Trained for 0 ≤ divergence ≤ 5 on synthetic forward_fill data. Self-benchmark: 480 FPS at 512² (3070 Ti). For divergence > the trained range it uses multiple "warp steps" (`calc_auto_warp_steps`). |
| `row_flow_v3_sym` | Same, but left and right offsets are exactly ±delta | "2x faster", experimental |
| `row_flow_v2` | Previous model, trained to imitate `apply_stereo_divergence_polylines` from stable-diffusion-webui-depthmap-script | Issue #60: "This model is just for processing speed" (https://github.com/nagadomi/nunif/issues/60) |
| `mlbw_l2` / `mlbw_l4` (+ `s` small variants) | "Multi-layer backward warping" (models/mlbw.py): outputs `num_layers` horizontal offsets plus `num_layers` softmax weights at depth resolution. Output = Σ_i w_i · grid_sample(rgb, grid + delta_i) (`apply_divergence_nn_delta_weight`). Separate weights per divergence level (d ≤ 4, ≤ 7, > 7). | PR #432 (https://github.com/nagadomi/nunif/pull/432): "effectively behaves like 3D warping"; it "does not improve the stretching in occluded regions. It improves the opposite issue: squashing caused by overlapping". The layers learned to separate only the two sides of a depth border. Self-benchmark at 512²: l2 305 FPS, l4 150, l2s 490, l4s 250. |
| `forward_inpaint`, `mlbw_l2_inpaint` | Forward warp (or MaskMLBW, which also predicts a hole mask). The hole is then redrawn by a light inpainting net (`LightInpaintV1`, about 40 FPS at 1080p per its benchmark comment); video variant inpaints 12 frames together | PR #484 (https://github.com/nagadomi/nunif/pull/484). Author recommends `mlbw_l2_inpaint`. |

**Parameterisation** (directly reusable):
- `--divergence` is the **total** L-R disparity range in **% of image width**. Code: per-eye `shift = divergence*0.01*0.5*W` px; the grid version uses `divergence*0.01` in [-1,1] grid units, which is the same thing (backward_warp.py, forward_warp.py). Issue #60: "divergence=1 shifts the pixel position up to 1% of the input image width". Default 2.0 for video; README says ">higher value, artifacts are more visible"; row_flow trained to 5, mlbw to 10.
- `--convergence` sets the zero-parallax point in normalized depth: `index_shift = depth*shift - shift*convergence`. README: "0 is good, but screen edge areas are hard to see"; "1 is the most friendly for curved display"; default 0.5. Auto convergence `sod_v1` (PR #600, https://github.com/nagadomi/nunif/pull/600) uses a U²-Net-P saliency model on 192² RGBD to put salient objects at convergence 0.5. It warns of extra temporal flicker.
- Our mapping in iw3 terms: `k = divergence/200` (uv per eye). Our k=0.012 corresponds to divergence 2.4, and c=0.35 corresponds to convergence 0.35.
- `--synthetic-view both|left|right`: README recommends `both`, which splits artifacts between the eyes.
- `--preserve-screen-border`: linearly ramps parallax to 0 over `divergence*0.75%` of the width at the left and right borders (`make_input_tensor`).
- `--foreground-scale` / `--mapper`: depth -> disparity remapping. `mapper.py` has softplus-based `mul_1..3`, inverse mappers, `shift_*` (which shift the relative depth in distance space, citing Depth-Anything issue #72), and `div_*` for metric depth. Default for relative depth is `none` (linear).
- `--edge-dilation` (GUI "Edge Fix"), default `[2,1]` (x,y); the desktop tool uses 2 (`dilation.py: dilate_edge`). Each iteration: `w = edge_weight(x)` (normalized local max-min range), `x2 = dilate(gaussian_blur(x))`, `x = x*(1-w) + x2*w`. So it is a soft max-filter applied only near edges, which **grows the foreground (high value) region**. README: "DepthAnything ... causes artifacts at foreground and background edges. This approach reduces artifacts by dilating foreground segments"; 4 is "most eye-friendly, but degrades depth accuracy". PR #484: "Edge Fix shifts the hole region away from the actual image boundary"; recommends 1 when inpainting.
- Flicker reduction `--ema-normalize`: EMA of the **min/max normalization scalars**, not per-pixel (`depth_scaler.py: EMAMinMaxScaler`). Default decay 0.75, optional look-ahead buffer. The desktop tool uses `enable_ema(decay, buffer_size=1)`.
- Depth AA (`--depth-aa`): a small window-attention net that anti-aliases depth edges (models/depth_aa.py).

**iw3.desktop** (https://github.com/nagadomi/nunif/blob/master/iw3/docs/desktop.md):
- Captures the desktop, runs depth and stereo **on the PC**, and serves an **MJPEG** (multipart/x-mixed-replace) Half-SBS stream over HTTP (`iw3/desktop/streaming_server.py`). The Quest browser shows it in full-screen "3D Side-by-Side" mode.
- Defaults in code (`iw3/desktop/utils.py`): `depth_model=Any_V2_S, divergence=1.0, convergence=1.0, ema_normalize=True`; stream-fps 30 (the doc says 15); stream height 1080; JPEG quality 90. Doc: "Due to `--batch-size 1` processing ... it may not be possible to achieve an FPS higher than 30".
- Doc warning: "depth estimation results for GUI windows and text may not be perfect. This tool is primarily intended for full-screen playback of images and videos."
- Also: `iw3/player` is a WebXR player for pre-converted SBS media (https://github.com/nagadomi/nunif/blob/master/iw3/player/README.md).

### 1.2 thygate/stable-diffusion-webui-depthmap-script - `stereoimage_generation.py` (MIT)
- https://github.com/thygate/stable-diffusion-webui-depthmap-script (MIT; last push 2024-08). File: `src/stereoimage_generation.py`.
- `apply_stereo_divergence_naive`: a per-row forward splat in "swipe order" so nearer pixels overwrite farther ones, then a fill (`naive` = nearest filled neighbour; `naive_interpolating` = linear interpolation across the gap).
- `apply_stereo_divergence_polylines` (`polylines_sharp`/`_soft`): each row is treated as a polyline whose vertices are displaced by `depth^exp * divergence_px`. The polyline is rasterized and, per output pixel, the segment with the largest "closeness" (nearest) supplies the colour. It is effectively **a per-row displaced mesh with z-test**: holes become stretched segments (rubber sheet), and "sharp" mode uses ±0.45 px pixel half-widths to reduce smear. This is the target that iw3 row_flow_v2 was trained to imitate.
- Parameters: `divergence` (% of width), `separation` (% shift), `stereo_balance` (split between eyes), `stereo_offset_exponent`.

### 1.3 lc700x/desktop2stereo (MIT)
- https://github.com/lc700x/desktop2stereo (MIT; 164 stars; pushed 2026-07). Real-time desktop to stereo for AMD/NVIDIA/Intel/Qualcomm/Apple (README). Supports DA-V2, Video Depth Anything, DA3, InfiniDepth, with ONNX/TensorRT/OpenVINO/CoreML/MIGraphX backends.
- **Warp = naive one-tap backward sampling** (`depth.py: make_sbs_core`): `shifts = -(depth - convergence)*depth_ratio * ipd_uv*W * 0.05`, then `grid_sample(xs ± shift)` with reflection padding. Defaults `ipd_uv=0.064, depth_ratio=2` give about 0.64% of width per eye at maximum.
- The Metal viewer shader (`metal_viewer.py`, `displaced_uv`) is the same idea on the GPU: a 3-tap horizontal depth blur (0.7/0.15/0.15), asymmetric shaping `d*(1+0.35*(1-d))`, `shift=(d-conv)*strength*eye`, and a **5% border falloff** `smoothstep(0,0.05,u)*smoothstep(1,0.95,u)`.
- Depth post-processing (`post_process_depth`): 2/98-percentile normalize, gamma 1.45, foreground scale (signed power around 0.5), Gaussian "anti_alias" blur. Temporal: `DepthStabilizer` per-pixel EMA with alpha 0.9.
- Takeaway: popular, but the warp itself is the "ghost artifacts" baseline. Useful mainly for capture/inference plumbing ideas.

### 1.4 BlueSkyDefender/Depth3D - ReShade SuperDepth3D (proprietary; study only)
- https://github.com/BlueSkyDefender/Depth3D (`Shaders/SuperDepth3D.fx`, v5.4.2; pushed 2026-10-01).
- **License: "Copyright (C) Depth3D - All Rights Reserved ... Unauthorized copying ... prohibited ... personal use"**, per the file header and https://blueskydefender.github.io/Depth3D/licensing.html. **Do not copy code.** The underlying algorithm is public (steep parallax mapping, McGuire & McGuire; POM, Tatarchuk). The header credits Philippe David's steep parallax code and Fubaxiusz's VR.fx.
- Uses the **game's depth buffer**, but the view synthesis is pure screen-space and maps directly to our case. `Parallax()` (around line 6521) works as follows:
  - **Divergence:** max per-eye divergence is 100 px at 2160p, scaled by `BUFFER_HEIGHT/2160` (so 50 px at 1080p) via `Min_Divergence`/`CalculateMaxDivergence`. "Depth Adjustment" slider defaults to 50%.
  - **Step count:** `Steps = |Divergence_px| * Perf`, with Perf 0.75/1.0/1.25 for the Performant/Normal/High levels. That is about one step per pixel of disparity range. A "distance-field skip" advances up to 2x (flat) or 1x (edges) layers per iteration, based on `(currentDepth - layerDepth)/layerStep`.
  - **March:** `while (CurrentDepthMapValue >= CurrentLayerDepth) { coord.x -= delta; depth = sample(coord); layer += layerStep; }`. This is classic steep parallax, starting from the ZPD-dependent layer `-Re_Scale_WN().x`.
  - **"De-artifacting" (on by default):** the march compares the depth at the coordinate with depth at a slightly offset coordinate. Where they differ (an edge), it uses `min(G, C)` (the nearer value in their convention), which effectively dilates the foreground during the march.
  - **Refinement:** if the depth jump between the last two samples is ≥ 0.12 ("hard mode"), it either does a "sharp gap seek" (up to `clamp(D,8,25)` extra steps past the occluder; moves only on a hit) or a **3-step binary search**. Otherwise it does the standard POM weighted interpolation `weight = after/(after-before)`.
  - **De-banding:** a position-stable hash jitters gap taps by < 1 px ("De-band the stretch").
  - **Hole mask:** `Hole_Px = |Δdepth| * D` (gap size in screen px), and the depth gradient under the final tap is measured in px. `Hole_Mask = max(smoothstep(1,4,Hole_Px), smoothstep(2,6,Edge_Px)*smoothstep(0.5,1,Hole_Px))` feeds later fill.
- **Depth pre-passes:**
  - `Disocclusion()`: a 3x3 min-dilate (`Min3x3`), then 8 horizontal taps at ±1.25-5% offsets scaled by divergence, tilted along the local silhouette slope, averaged. It returns `min(avg, depth)`, the "never push farther" rule.
  - `DepthSmoothPS`: an edge-aware smoothing that pulls each pixel toward the min of its L/R (and U/D) neighbours proportionally to the relative edge strength `|R-L|/center*25`.
  - In SD3D's convention smaller = nearer, so both passes **dilate the foreground**: the same idea as iw3 Edge Fix.
- Other relevant features: UI masking (alpha-channel UI mask, `Set_UI`), mouse cursor drawn in stereo after parallax, foveated quality, "ZPD boundary" handling (a pop-out at screen edges triggers a convergence change).

### 1.5 Fubaxiusz/fubax-shaders - VR.fx `Parallax()` (CC BY-NC-SA 4.0)
- https://github.com/Fubaxiusz/fubax-shaders/blob/596d06958e156d59ab6cd8717db5f442e95b2e6b/Shaders/VR.fx (the version SD3D cites).
- The cleanest compact reference of steep parallax plus a POM secant step for horizontal stereo:
  - `LayerDepth = 1/min(64, Steps)`; offset per step = `Offset*LayerDepth`; start at `x + Offset*Center`.
  - `while (layer < depth) { x -= delta; depth = D(x); layer += LayerDepth; }`
  - Then `weight = after/(after-before)` linear interpolation of the coordinate.
  - Then "gap masking" pushes the coordinate by `(before-after)*GapOffset*Offset*100` px.
- NC license, so don't copy. Re-implementing the textbook algorithm is fine.

### 1.6 BrokenSource/DepthFlow (AGPL-3.0) - GLSL depth ray-march for 2.5D "3D photo"
- https://github.com/BrokenSource/DepthFlow, `depthflow/resources/depthflow.glsl`.
- A full perspective camera ray against a heightfield, as a **two-stage search**:
  - Stage 1: forward "probe" steps of 1/50..1/120 until the ray is under the surface.
  - Stage 2: **backward** fine steps of 1/200..1/2000 until it is back outside.
- It also computes a "steep" heuristic (derivative × normal angle) for inpaint masking; `examples/vr.py` shows a stereoscopic camera mode. AGPL, so study only.

### 1.7 panrafal/depthy (MIT) - mobile-friendly WebGL depth parallax
- https://github.com/panrafal/depthy, `app/scripts/pixi/DepthPerspectiveFilter.glsl` (GLSL ES 1.0, `mediump`). The quality presets use **MAXSTEPS 4, 6, 16 or 40**.
- Marches from the near layer to the far layer and accumulates a "confidence" `step(dpos, depth)`. The coordinate is corrected by `(depth-dpos)/dstep*vstep` (a secant-like correction) and averaged until the confidence sum reaches `CONFIDENCE_MAX`.
- Optional "antialias" half-step back-tracking. It shows that 6-16 steps are usable on 2014-era mobile GPUs. MIT, so borrowing is OK.

### 1.8 VisionDepth/VisionDepth3D (now proprietary)
- https://github.com/VisionDepth/VisionDepth3D. The repo now has only docs plus a **proprietary EULA** (Free/Pro tiers). An earlier `LICENSE.txt` (2025-2026) was also "All rights reserved; non-commercial; no derivative works".
- Its "Hybrid3D" method document (https://github.com/VisionDepth/VisionDepth3D/blob/Main-Stable/VisionDepth3D_Method.md) is a good **checklist of production tricks**:
  - percentile+EMA depth normalization; subject tracking for convergence;
  - near/mid/far band weighting of disparity; "Subject Plane Lock";
  - shift smoothing in **flat regions only**, edge-stress maps, an EMA'd occlusion mask, and **temporal shift velocity limiting** before the shift EMA;
  - a continuous inverse warp (`grid_sample`) **blended with a depth-ordered forward warp only where the forward path's validity mask says it owns the pixel**;
  - **directional background-side** disocclusion repair with a foreground contour barrier;
  - post-warp halo cleanup; a **dynamic floating window** driven by the measured edge violation.
- Offline video converter (not real-time).

### 1.9 TencentARC/StereoCrafter (research-only license)
- https://github.com/TencentARC/StereoCrafter; paper https://arxiv.org/abs/2409.07447. License-Code.txt: "only for academic, research and education purposes ... refrain from using it for any commercial or production purposes".
- Pipeline: video depth, then `ForwardWarpStereo` (`depth_splatting_inference.py`). That is **softmax splatting**: weights `1.414^(disp - min)`, then `fw(im*w)/fw(w)`, plus an occlusion map `1 - fw(ones)`. `disp = (depth*2-1)*max_disp`, default `--max_disp 20` px. A fine-tuned SVD video-diffusion model then inpaints the occluded regions (`inpainting_inference.py`).
- Quality reference for offline work; far from real time.

### 1.10 Others worth knowing
- **StereoDiffusion** (training-free latent-diffusion stereo; arXiv https://arxiv.org/abs/2403.04965). Code: https://github.com/lez-s/StereoDiffusion (MIT). Not real time.
- **Mono2Stereo** (CVPR 2025 benchmark plus a dual-condition diffusion baseline and the SIoU metric; https://arxiv.org/abs/2503.22262). Code: https://github.com/song2yu/Mono2Stereo. Not real time.
- **Deep3D** (Xie, Girshick, Farhadi, ECCV 2016; https://arxiv.org/abs/1604.03650; https://github.com/piiswrong/deep3d). Predicts a per-pixel probability over discrete disparities. Output = Σ_d p_d · shift(I, d), a "selection layer". This is a soft multi-plane formulation that a shader can evaluate (N shifted fetches × weights), the ancestor of mlbw.
- **Layered Depth / 3D photo:** 3D Photography using Context-aware Layered Depth Inpainting (https://github.com/vt-vl-lab/3d-photo-inpainting, MIT; LDI plus learned colour/depth inpainting, offline). One Shot 3D Photography (https://facebookresearch.github.io/one_shot_3d_photography/; Facebook 3D Photos: LDI mesh + mobile inpainting). 3D Ken Burns (https://github.com/sniklaus/3d-ken-burns, CC BY-NC-SA).
- **MPI:** Stereo Magnification (https://arxiv.org/abs/1805.09817; https://github.com/google/stereo-magnification, Apache-2.0). Single-View MPI (https://arxiv.org/abs/2004.11364). Rendering an MPI is cheap (alpha-composite N planes), but predicting one per frame is heavy.
- **Bino** (https://bino3d.org/, GPLv3) is a stereo/VR video **player**; no monocular 2D->3D conversion.
- **3DGameBridge** (https://github.com/BramTeurlings/3DGameBridge, GPL-3.0) is an SR/Leia display weaving wrapper and ReShade addon. It consumes stereo from SuperDepth3D or geometry 3D; not a conversion method itself.
- **VITURE Immersive 3D:** marketing pages only (https://www.viture.com/en-US/blog/a-worlds-first-turn-2d-into-magical-3d-in-real-time). Three depth levels (Enhanced/Standard/Soft), Movie vs Game (lower-latency) modes, available on Mac/Windows after iOS. No technical disclosure was found; the DA-V2-S + shader DIBR guess remains unverified.

---

## 2. Algorithm families vs. our constraints

Notation: per-eye shift `s(x) = side*k*(D(x)-c)` (uv). Output pixel x shows source xs where `x = xs + s(xs)`.

### 2.1 One-tap backward ("grid_sample", Desktop2Stereo, iw3 `backward`)
`xs = x - s(x)`, using depth at the *destination*.
- Cost: 1 depth + 1 colour fetch.
- Wrong at every discontinuity: foreground texture is duplicated or ghosted into the background. iw3 says "Lots of ghost artifacts".

### 2.2 Fixed-point backward (our current shader)
Converges only where `|k·D'| < 1` (see TL;DR). It oscillates at edges, gives no nearest-surface guarantee, and with 4 iterations the edge result is effectively arbitrary.

### 2.3 Backward ray march / steep parallax + POM (SuperDepth3D, Fubax, DepthFlow, depthy)
Parametrize the candidate depth `t ∈ [1 → 0]` with `xs(t) = x - side*k*(t - c)` and march t from near to far. The **first** t where `D(xs(t)) ≥ t` is the nearest surface along the "ray". Refine it with a secant step (POM) or 2-3 bisections.
- Correct occlusion order. Disocclusions resolve onto the bilinear depth ramp, so they "stretch". Stretch can be detected (bracket depth jump, local gradient) and replaced by a background-side fill.
- Steps: the search spans `k*W` source pixels, which is **`k*518` depth texels** for our depth map. At k=0.012 that is only about 6 depth texels, so N=8 steps already samples finer than one texel. SuperDepth3D uses about 0.75-1.25 steps per screen pixel because its depth buffer is full resolution.
- Rule of thumb for us: `N = clamp(ceil(k*Wd/0.75), 6, 32)`, with Wd = 518. Use 2-3 bisection steps when the bracket's depth jump exceeds a threshold, otherwise one secant step.
- Fetch count ≈ N+3 tiny-texture taps + 1 colour tap: about 15 at k=0.012, about 31 at k=0.04.

### 2.4 Forward warp / splatting + hole fill (iw3 forward_fill, SD-webui naive, StereoCrafter softmax splat, VisionDepth forward pass)
- Correct ordering via sort, z-buffer or softmax weights. Holes are explicit, so they can be inpainted (directional background fill, push-pull blur, or learned).
- Needs scatter (compute plus atomics, or point rendering). WebGL2 has no compute shaders, but you can render **GL_POINTS or a grid mesh** with a depth test, which is the same thing. On the PC (CUDA/PyTorch) it is easy. iw3's PyTorch implementation is only 22 FPS at 1080p, so a custom CUDA or compute kernel would be needed.
- Bilinear splatting leaves cracks under magnification and needs the "layered holes" fix (iw3 `fix_layered_holes`).

### 2.5 Mesh / grid displacement
Render the depth map as a W_d×H_d grid (518×294 ≈ 152k vertices, ≈ 300k triangles). The vertex shader does `x' = x + side*k*(d-c)` with z = -d and the GPU depth test on; the fragment shader samples colour at the original uv.
- Nearly free on any GPU: z-buffer occlusion, one colour fetch per fragment, and sub-pixel exact for smooth regions.
- Disocclusions become stretched triangles (rubber sheet), the same visual as backward-warp stretch. Silhouettes follow the 518-wide depth grid, so they are blocky at about 3.7 screen px per cell. Edge-aware upsampled depth or a finer grid at about 960 wide helps.
- **Tearing** (drop triangles whose depth range exceeds a threshold) turns stretch into holes. Fill them by rendering a **background layer first**: a second mesh using foreground-eroded depth (min-filtered over the max disparity radius) and a blurred colour. This is the cheap 2-layer LDI used by 3D-photo systems (Facebook 3D Photos / One Shot 3D Photography).
- **True-3D option:** in VR you can instead unproject the grid to world space and render with the real per-eye cameras. You get head-motion parallax for free, but desktop UI would "swim" with head motion. For a virtual monitor, screen-space horizontal parallax (current approach) is safer; keep the true-3D mode as an optional "diorama" setting.

### 2.6 Layered / multi-plane
Deep3D-style "selection" (Σ_d p_d · shift(I,d)), iw3 mlbw (2-4 warp layers + softmax weights), MPI.
- The shader side is cheap: N layers × (1 offset fetch + 1 colour fetch).
- The weights need a learned predictor (PC side). mlbw is the practical, MIT-licensed instance, at 305-490 FPS for 512² on a 3070 Ti per its self-benchmark.

### 2.7 Learned / generative
- Real-time feasible: iw3 row_flow_v3 / mlbw_l2(s) (warp-field nets on depth only) plus `light_inpaint_v1` (about 40 FPS at 1080p on the author's GPU per the code comment).
- Not real time: StereoCrafter (SVD), StereoDiffusion, Mono2Stereo (diffusion), 3D-photo LDI inpainting.

**Summary table**

| family | occlusion order | holes | Quest WebGL2 cost | PC cost | quality |
|---|---|---|---|---|---|
| one-tap backward | wrong | ghost/duplicate | 2 taps | trivial | poor |
| fixed-point backward (current) | undefined at edges | flicker/ghost | 5 taps | - | poor-fair |
| ray-march backward + secant/bisect + bg fill | correct (nearest wins) | stretch, or bg-side fill | ~10-30 taps | trivial | good |
| grid mesh + z-test (+tear+bg layer) | correct | stretch / 2-layer fill | very low | trivial | good (edges at depth-grid res) |
| forward splat + fill | correct | explicit, fillable | needs points/mesh | moderate (custom kernel) | good-very good with inpaint |
| learned warp field (row_flow/mlbw) | learned | stretch (mlbw sharper overlaps) | 2-4 taps (given field) | small net on PC | very good |
| learned + inpaint (mlbw_l2_inpaint) | learned | inpainted | n/a (PC full SBS) | ~25-40 FPS 1080p class | best real-time-ish |
| diffusion (StereoCrafter etc.) | - | generated | - | far from real time | best offline |

---

## 3. Practical details

### 3.1 Depth -> disparity mapping
- Screen parallax for a point at distance Z, viewed on a screen at distance D with interocular distance e, is `p = e(1 - D/Z)`. That is **affine in 1/Z**. DA-V2 outputs **affine-invariant inverse depth** (relative disparity; the iw3 mapper cites the Depth-Anything issue #72 discussion: https://github.com/LiheYoung/Depth-Anything/issues/72).
  - So a linear map `shift = k*(d - c)` is geometrically consistent (up to unknown scale and shift), and you should **not** invert d.
  - Nonlinear mappers are perceptual tweaks: iw3 `mul_*`/`shift_*`/foreground-scale (https://github.com/nagadomi/nunif/blob/master/iw3/mapper.py), Desktop2Stereo gamma 1.45, and VisionDepth's near/mid/far bands.
- Normalization: per-frame min-max on a relative map makes the zero-parallax plane jump when content changes, for example when a window opens. Use percentiles (2/98) and **EMA the normalization scalars** (iw3 `EMAMinMaxScaler`, decay 0.75-0.99) rather than only a per-pixel EMA. Desktop2Stereo's per-pixel α=0.9 smears moving edges.
- Parameter equivalence: iw3 divergence = total % of width = `200*k`. Ours: k=0.012 means divergence 2.4. iw3 desktop default is divergence 1, convergence 1, so everything sits behind or at the screen and only 1% of width is used. Video default is 2 / 0.5.
- Pop-out vs. depth split: with c=0.35, d=1 gives `2k(1-c)` = 1.56% of width **in front**, and d=0 gives 0.84% behind.
  - Stereo production guidelines: Sky 3D asks for about 2% positive (behind) and about 1% negative (in front) of screen width (https://creativecow.net/?p=1035557). Avatar's measured range was -0.5% to +1.0% (https://videoprocessing.ai/stereo_quality/parallax-range-estimation-s3d.html).
  - So for a desktop, c ≈ 0.6-0.75 is a better default: c=0.6, k=0.012 gives 0.96% in front and 1.44% behind.
- VR-specific hard limit: positive parallax on the virtual screen must stay below the IPD in *physical* units, or the eyes must diverge. For screen width S (m), `p_max% = 6.3cm / S`. A 2.4 m-wide virtual screen allows at most about 2.6% of width behind. Angular disparity ≈ `p% * S / D` rad; e.g. 1% of a 2.4 m screen at 2.5 m is about 0.55°.
- Vergence-accommodation: Quest-class lenses focus at a fixed distance (Quest 2 about 1.3 m per https://echeng.com/articles/vr-optical-inserts). Comfort literature: Shibata et al., "The zone of comfort" (J. Vision 2011; https://pmc.ncbi.nlm.nih.gov/articles/PMC3150963/). Keep the virtual screen around 1.5-3 m and the content's disparity range modest. Let users scale it: VITURE ships Enhanced/Standard/Soft presets.
- Screen borders (window violation): objects in front of the screen cut by the frame edge are uncomfortable. Options: ramp parallax to 0 near the left/right border (iw3 `--preserve-screen-border`: ramp width `0.75*divergence`% of W; Desktop2Stereo: 5% smoothstep), or a floating window (VisionDepth).

### 3.2 Depth edges
- The core artifact: DA depth edges are soft and slightly misaligned with colour edges. After warping, either foreground colour leaks into the background (halo/ghost), or the stretch region eats into the silhouette.
- **Fix 1, dilate the foreground in depth** (iw3 Edge Fix, default 2 iterations at depth resolution; SuperDepth3D min-filter + "never push farther"). The silhouette's colour pixels then all move with the foreground, and the stretch or hole lands on background texture. Too much (≥4 iterations) visibly fattens objects.
- **Fix 2, edge-aware upsampling** of the 518×294 depth to about 960×540 or full resolution, guided by colour. Options: Joint Bilateral Upsampling (Kopf et al. 2007, https://johanneskopf.de/publications/jbu/), guided filter (He et al., https://people.csail.mit.edu/kaiming/eccv10/; fast guided filter https://arxiv.org/abs/1505.00996), or iw3's learned depth_aa. Do it **once per frame in its own pass** (or on the PC), never inside the ray-march loop. Caveat: text and UI edges are strong colour edges with no real depth edge, so use a conservative range sigma.
- **Fix 3, hole handling:** detect the stretch, then fill from the **background side** (iw3 `shift_fill`, VisionDepth directional repair). Hide streaks with a slight blur (iw3 `blur_blend`, box blur 7 over the hole mask) or jitter (SuperDepth3D de-band hash).
- **Depth travels through the video codec in our pipeline.** 8-bit, 4:2:0 and lossy coding create ringing and blocking at depth edges, which turn into edge wobble.
  - Keep the depth region strictly grey, aligned to 16 px macroblocks, and away from the colour region.
  - Consider a separate low-res depth channel or 10-bit.
  - A small 3×3 median, or the dilation itself, on the headset helps.

### 3.3 Temporal stability
- EMA on the normalization range (iw3), percentile normalization (Desktop2Stereo, VisionDepth), and scene-cut reset (iw3 uses TransNetV2 for video).
- Per-pixel temporal filtering: prefer motion- or edge-aware filtering (VisionDepth: velocity limiter on the shift field, then EMA, with smaller allowed deltas near edges). A plain per-pixel EMA ghosts on motion.
- Auto-convergence raises flicker (iw3 PR #600), so smooth it heavily (iw3 uses decay 0.98 for the convergence model).
- Video Depth Anything streaming models (iw3 `VDA_Stream_S`) give temporally consistent depth at the source.

### 3.4 UI / text
- iw3 desktop explicitly warns that GUI and text depth "may not be perfect".
- Text survives a **constant** shift across a glyph but not a **varying** one; sub-pixel gradients make letters wobble and shimmer. So:
  - (a) Strongly smooth or flatten the shift field in low-depth-gradient regions (VisionDepth "flat-region shift smoothing").
  - (b) Snap planar regions to a constant shift.
  - (c) Use convergence near the UI depth so UI sits at the zero-parallax plane (iw3 desktop defaults convergence=1).
  - (d) Desktop-specific idea: the PC knows window rectangles (DWM). It can send a per-window "flat" mask so normal app windows are planar, with depth only for video or game surfaces.
- Draw the mouse cursor after warping, at the depth under it or at the screen plane, rather than baking it into the frame. SuperDepth3D draws the cursor in stereo; iw3 bakes it by default (`--disable-draw-cursor`).

### 3.5 Quest 3 rendering budget
- Current: per display frame, per eye, a 5-tap shader over an `XRWebGLLayer` at 1.5× framebuffer scale. Fragment count scales with the screen's coverage of a large eye buffer.
- Better: run the warp **once per decoded video frame** (30-60 Hz) into two RGBA8 textures at source resolution, about 2 × 2.07 Mpx × ~15 taps. Present them with a **WebXR quad layer** (`XRWebGLBinding.createQuadLayer`, layout `stereo-left-right`). The compositor then resamples only once, which gives sharper text.
  - Spec: https://www.w3.org/TR/webxrlayers-1/
  - Meta: "only need to render when layer content updates"; "higher quality and at half the GPU usage" (https://developers.meta.com/horizon/documentation/web/webxr-layers/).
- Keep depth in its own small `R16F`/`RG16F` texture produced by a preprocessing pass. The march loop then hits a tiny, cache-resident texture; use `textureLod(..., 0.0)`.

---

## 4. Recommendation for ImmersiveVR

### Phase 1 (Quest, WebGL2, now)

**Pass A, depth prep.** Once per video frame, at 518×294 (or 2× upsampled), into an RG16F target:
1. Read the grey depth from the video region (optionally a 3×3 median against codec ringing).
2. Run a foreground edge-dilation: 1-2 horizontal iterations in iw3 style, `x = mix(x, max3(blur(x)), edgeWeight)`. Write the result as R = D.
3. Compute the background depth `Dbg` as a horizontal min-filter over ±ceil(k*Wd) texels (separable; 2 passes, or one wide pass with about 12 taps). Write it as G = Dbg (used for hole fill and hole classification).
4. Optional: a flat-region flatten mask (for text) folded into D.
5. Optional: border ramp. Multiply the *shift*, not D, by `smoothstep(0, b, u)*smoothstep(1, 1-b, u)` with b ≈ 0.03-0.05. Do this in pass B.

**Pass B, per-eye warp.** Once per video frame, at 1920×1080, into two RGBA8 textures. Then present both with one stereo-left-right quad layer (or as the current quad mesh if layers are not available).

```glsl
#version 300 es
precision highp float;
uniform sampler2D uColor;     // desktop RGB (video frame region)
uniform sampler2D uDepth;     // RG16F: r = D (fg-dilated), g = Dbg (fg-eroded); LINEAR, CLAMP
uniform vec4  uColorRect;     // packed-frame mapping, as today
uniform float k;              // per-eye max shift in uv (0.004..0.02); iw3 divergence = 200*k
uniform float c;              // convergence (0..1); 0.6..0.75 recommended for desktop
uniform float side;           // +1 left eye, -1 right eye
uniform float border;         // e.g. 0.04 -> parallax ramps to 0 at left/right edges
uniform int   N;              // linear steps = clamp(ceil(k*518.0/0.75), 6, 32)
in vec2 uv; out vec4 o;

float D(float x, float y){ return textureLod(uDepth, vec2(x,y), 0.0).r; }
float shiftScale(float x){ return smoothstep(0.0, border, x) * smoothstep(1.0, 1.0-border, x); }

void main(){
  float kk = k * shiftScale(uv.x);         // border falloff (approx: evaluated at output x)
  // Output x shows source xs with x = xs + side*kk*(D(xs)-c).
  // Candidate depth t in [1 -> 0]; xs(t) = x - side*kk*(t - c). First t with D(xs(t)) >= t = nearest surface.
  float tPrev = 1.0, fPrev = D(uv.x - side*kk*(1.0 - c), uv.y) - 1.0;   // f(t) = D(xs(t)) - t
  float tHit = 0.0, fHit = 0.0; bool hit = fPrev >= 0.0;
  if (hit) { tHit = 1.0; fHit = fPrev; }
  for (int i = 1; i <= 32; ++i) {
    if (i > N || hit) break;
    float t = 1.0 - float(i) / float(N);
    float f = D(uv.x - side*kk*(t - c), uv.y) - t;
    if (f >= 0.0) { hit = true; tHit = t; fHit = f; break; }
    tPrev = t; fPrev = f;
  }
  // refine root of f in [tHit, tPrev]: 3 bisection steps (robust at steep edges), then 1 secant step
  float a = tHit, b = tPrev;                 // f(a) >= 0, f(b) < 0
  for (int j = 0; j < 3; ++j) {
    float m = 0.5*(a+b);
    float fm = D(uv.x - side*kk*(m - c), uv.y) - m;
    if (fm >= 0.0) { a = m; fHit = fm; } else { b = m; fPrev = fm; }
  }
  float t = mix(a, b, clamp(fHit / max(fHit - fPrev, 1e-5), 0.0, 1.0));
  float xs = uv.x - side*kk*(t - c);

  // Hole (disocclusion) test: root sits on a steep depth ramp (not on a flat surface).
  float texel = 1.0 / 518.0;
  float g = abs(D(xs + texel, uv.y) - D(xs - texel, uv.y));       // depth jump across ~2 depth texels
  float holePx = g * kk * 1920.0;                                  // ramp size in screen px (SuperDepth3D-style)
  float hole = smoothstep(1.0, 4.0, holePx);

  vec3 col = texture(uColor, uColorRect.xy + vec2(clamp(xs,0.,1.), uv.y) * uColorRect.zw).rgb;
  if (hole > 0.0) {
    // Background-side fill: re-solve on the smooth fg-eroded depth (fixed point converges there),
    // then step a little further away from the foreground so we copy background, not edge pixels.
    float xb = uv.x;
    for (int j = 0; j < 3; ++j) xb = uv.x - side*kk*(textureLod(uDepth, vec2(xb, uv.y), 0.0).g - c);
    float dirToBg = sign(D(xs - texel, uv.y) - D(xs + texel, uv.y)); // toward lower depth = background
    xb = (dirToBg != 0.0) ? xs + dirToBg * 1.5 * texel : xb;          // alternative: use xb from Dbg solve
    vec3 bg = textureLod(uColor, uColorRect.xy + vec2(clamp(xb,0.,1.), uv.y) * uColorRect.zw, 1.0).rgb; // slight blur hides streaks
    col = mix(col, bg, hole);
  }
  o = vec4(col, 1.0);
}
```

Notes on the shader:
- Fetches: about N + 3 + 1 depth taps (tiny texture) plus 1-2 colour taps. k=0.012 gives N=9, about 15 taps per pixel. Run per *video* frame rather than per display frame, and drop the 1.5× eye-buffer scale.
- The two hole-fill variants above are alternatives; pick one after visual testing.
  - "Step off the edge toward lower depth" is the iw3 `shift_fill`-like replication.
  - "Solve on Dbg" tends to give a smoother stretched background.
  - Start with plain stretch (hole = 0) plus Edge Fix. That is what iw3's backward methods ship with by default; then add the fill.
- Tunables to expose: k, c, Edge-Fix iterations (0-3), border, hole on/off, N override.
- Validate visually with a test pattern: a vertical bar in front of text. At edges, check for no doubled bar edges, no flicker when the depth fluctuates slightly, and the background text intact next to the bar.

**Alternative Phase-1 path (if the ray march is too heavy):** a grid mesh with z-test.
- Vertex shader: `x' = x + side*k*(D(x)-c)`, `z = 1 - D`.
- Fragment shader: sample colour at the source uv.
- Draw it into the same offscreen eye textures. Same edge prep. Costs about 1 colour tap per pixel plus about 150k vertices per eye.
- Upgrade path: tear plus a background layer (Section 2.5).

### Phase 2 (PC side, RTX) - pick by quality need
1. **Learned warp field, headset applies it** (recommended next step).
   - Run iw3 `mlbw_l2s`/`mlbw_l2` (MIT code and weights) or `row_flow_v3` on the PC at depth resolution. Export to ONNX for our ORT/CUDA stack; the models are small window-attention CNNs.
   - Send per eye: 2 offsets + 2 weights (RGBA16F or 8-bit-quantized) at about 518×294. That is 2 textures, or one atlas, in the existing video packing or as a side channel.
   - The headset does `Σ_i w_i · texture(color, x + delta_i)`: 2 offset fetches + 2 colour fetches per pixel.
   - Divergence and convergence are network inputs (`make_input_tensor`), so slider changes round-trip to the PC. That is fine at desktop latencies. mlbw picks weights per divergence level (≤4, ≤7, >7).
   - Caveat: iw3 trains on its own depth normalization and mapper. Feed the same normalization: EMA min-max, mapper `none`, Edge Fix 1-2 on the depth.
2. **PC renders final SBS** (iw3.desktop model): forward warp + `light_inpaint` or `mlbw_l2_inpaint`.
   - Highest real-time quality, but it doubles encoded pixels (or halves horizontal resolution with half-SBS) and removes instant headset-side tuning.
   - A custom CUDA forward splat is needed for 60 fps at 1080p; iw3's PyTorch forward_fill is about 22 FPS at 1080p.
3. Not real time: StereoCrafter / StereoDiffusion / Mono2Stereo / LDI inpainting. Use them for offline content only.

### Licensing summary for borrowing
| source | license | can borrow code? |
|---|---|---|
| nunif / iw3 (code + weights) | MIT (weights confirmed MIT, issue #718) | yes, with attribution |
| stable-diffusion-webui-depthmap-script | MIT | yes |
| desktop2stereo | MIT | yes |
| depthy | MIT | yes |
| 3d-photo-inpainting | MIT | yes |
| stereo-magnification | Apache-2.0 | yes |
| Depth-Anything-V2-Small | Apache-2.0 (B/L are CC-BY-NC) | model: yes (S only) |
| DepthFlow | AGPL-3.0 | no (would force AGPL) |
| Fubax VR.fx | CC BY-NC-SA 4.0 | no for commercial; study only |
| SuperDepth3D | proprietary, all rights reserved | no; study only |
| VisionDepth3D | proprietary EULA | no (docs only) |
| StereoCrafter | research/education only | no |
| 3d-ken-burns | CC BY-NC-SA 4.0 | no |
| Bino, 3DGameBridge | GPL-3.0 | not relevant / copyleft |

The algorithms themselves (steep parallax / POM ray marching, forward splatting with depth ordering, JBU, guided filter) are textbook and free to re-implement.
