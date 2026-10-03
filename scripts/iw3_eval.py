"""Compares iw3 stereo generation with ImmersiveVR's shader warp on the same depth.

Runs inside a nunif checkout's environment (https://github.com/nagadomi/nunif, MIT):

    <nunif>/.venv/Scripts/python.exe scripts/iw3_eval.py --nunif <nunif> \
        --image photo.jpg --image desktop.png --out target/iw3

For each image: Depth-Anything-V2-Small (our ONNX export) -> iw3 edge
dilation -> min-max normalization, then left/right views from
  * iw3 row_flow_v3 (learned offset field, iw3's default),
  * iw3 mlbw_l2 (learned 2-layer offsets + blend weights),
  * iw3 backward (one depth sample per pixel; "lots of ghost artifacts"),
  * ImmersiveVR's shader step search (near-to-far + bisection), in torch.
Writes side-by-side PNGs (left | right) per method, edge crops, and timing
of the networks at the depth resolution.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image

DIVERGENCE = 2.0  # total left-right shift in % of width: ImmersiveVR strength 0.01 per eye
CONVERGENCE = 0.65
DEPTH_SIZE = (518, 294)


def depth_of(session: ort.InferenceSession, rgb: np.ndarray) -> np.ndarray:
    """Raw DA-V2-S relative inverse depth at DEPTH_SIZE, averaged with a flipped pass like iw3."""
    small = np.asarray(Image.fromarray(rgb).resize(DEPTH_SIZE, Image.BILINEAR))
    def run(img):
        bgra = np.concatenate([img[..., ::-1], np.full(img.shape[:2] + (1,), 255, np.uint8)], -1)[None]
        return session.run(["depth"], {"bgra": np.ascontiguousarray(bgra)})[0][0]
    return 0.5 * (run(small) + run(small[:, ::-1])[:, ::-1])


def step_search(color: torch.Tensor, depth: torch.Tensor, side: float, k: float, c: float) -> torch.Tensor:
    """The fragment shader in crates/immersive-vr/web/index.html, vectorized."""
    _, _, H, W = color.shape
    dh, dw = depth.shape[-2:]
    x = torch.linspace(0, 1, W, device=color.device).view(1, W).expand(H, W)
    y = torch.linspace(0, 1, H, device=color.device).view(H, 1).expand(H, W)
    fade = torch.clamp(x / 0.04, 0, 1) ** 2 * (3 - 2 * torch.clamp(x / 0.04, 0, 1))
    fade = fade * (lambda t: t * t * (3 - 2 * t))(torch.clamp((1 - x) / 0.04, 0, 1))
    kk = k * fade

    def depth_at(xs):
        grid = torch.stack([xs.clamp(0, 1) * 2 - 1, y * 2 - 1], -1)[None]
        return F.grid_sample(depth, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]

    def source(t):
        return x - side * kk * (t - c)

    steps = int(min(max(np.ceil(k * dw / 0.75), 6), 32))
    t = torch.zeros_like(x)
    found = depth_at(source(torch.ones_like(x))) >= 1.0
    t = torch.where(found, torch.ones_like(t), t)
    previous = torch.ones_like(x)
    for i in range(1, steps + 1):
        cand = torch.full_like(x, 1.0 - i / steps)
        hit_now = (~found) & (depth_at(source(cand)) >= cand)
        hit, miss = cand.clone(), previous.clone()
        for _ in range(4):
            mid = 0.5 * (hit + miss)
            ok = depth_at(source(mid)) >= mid
            hit = torch.where(ok, mid, hit)
            miss = torch.where(ok, miss, mid)
        t = torch.where(hit_now, hit, t)
        found = found | hit_now
        previous = cand
    grid = torch.stack([source(t).clamp(0, 1) * 2 - 1, y * 2 - 1], -1)[None]
    return F.grid_sample(color, grid, mode="bilinear", padding_mode="border", align_corners=True)


def timed(fn, runs=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / runs * 1e3


def main() -> int:
    global DIVERGENCE, CONVERGENCE, DEPTH_SIZE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nunif", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, default=Path("models/depth/dav2s_518x294_fp16.onnx"))
    parser.add_argument("--out", type=Path, default=Path("target/iw3"))
    parser.add_argument("--divergence", type=float, default=DIVERGENCE)
    parser.add_argument("--depth-size", default="518x294", help="WxH of the depth model (and of iw3's input)")
    parser.add_argument("--convergence", type=float, default=CONVERGENCE)
    args = parser.parse_args()
    DIVERGENCE, CONVERGENCE = args.divergence, args.convergence
    DEPTH_SIZE = tuple(int(v) for v in args.depth_size.lower().split("x"))
    sys.path.insert(0, str(args.nunif.resolve()))
    from iw3.backward_warp import apply_divergence_nn_LR, apply_divergence_grid_sample
    from iw3.dilation import dilate_edge
    from iw3.stereo_model_factory import create_stereo_model

    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    models = {
        "row_flow_v3": create_stereo_model("row_flow_v3", DIVERGENCE, device_id=0),
        "mlbw_l2": create_stereo_model("mlbw_l2", DIVERGENCE, device_id=0),
    }

    for path in args.image:
        rgb = np.asarray(Image.open(path).convert("RGB").resize((1920, 1080), Image.BICUBIC))
        raw = torch.from_numpy(depth_of(session, rgb).copy()).to(device)[None, None]
        raw = dilate_edge(raw, 2)  # iw3 default Edge Fix
        depth = (raw - raw.min()) / (raw.max() - raw.min()).clamp(min=1e-6)
        color = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(device)
        Image.fromarray((depth[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(args.out / f"{path.stem}_depth.png")

        views = {}
        with torch.inference_mode():
            for name, model in models.items():
                views[name] = apply_divergence_nn_LR(model, color, depth, DIVERGENCE, CONVERGENCE, steps=1)
                ms = timed(lambda: apply_divergence_nn_LR(model, color, depth, DIVERGENCE, CONVERGENCE, steps=1))
                print(f"{path.name} {name}: {ms:.2f} ms for both eyes (network at {DEPTH_SIZE[0]}x{DEPTH_SIZE[1]} + 1080p warp)")
            views["backward"] = apply_divergence_grid_sample(color, depth, DIVERGENCE, CONVERGENCE, "both")
            k = DIVERGENCE / 2 * 0.01
            views["shader_step_search"] = (step_search(color, depth, +1, k, CONVERGENCE),
                                           step_search(color, depth, -1, k, CONVERGENCE))
        for name, (left, right) in views.items():
            sbs = torch.cat([left, right], dim=3)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((sbs * 255).astype(np.uint8)).save(args.out / f"{path.stem}_{name}.png")
        # The strongest depth edge, cropped from every method's left eye, side by side.
        grad = torch.abs(depth[0, 0, :, 1:] - depth[0, 0, :, :-1])
        yy, xx = np.unravel_index(int(grad.argmax()), grad.shape)
        cx = int(xx / DEPTH_SIZE[0] * 1920)
        cy = int(yy / DEPTH_SIZE[1] * 1080)
        x0 = min(max(cx - 160, 0), 1920 - 320)
        y0 = min(max(cy - 120, 0), 1080 - 240)
        crops = [views[n][0][0, :, y0:y0 + 240, x0:x0 + 320] for n in views]
        strip = torch.cat(crops, dim=2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        Image.fromarray((strip * 255).astype(np.uint8)).save(args.out / f"{path.stem}_edges_left.png")
        print(f"{path.name}: edge crop at ({cx},{cy}); order {list(views)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
