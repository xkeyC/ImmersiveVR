"""Exports iw3's mlbw_l2 stereo network plus a full-resolution warp as ONNX graphs.

iw3 (nagadomi/nunif, MIT; weights MIT per nunif issue #718) mlbw_l2 predicts,
from the normalized depth map alone, two horizontal sampling offsets and two
blend weights per pixel for the left view; the right view is the same network
on the mirrored depth. ImmersiveVR runs it on the PC and renders both eyes
there:

  fields   depth       float32 [1, 1, h, w]   normalized inverse depth, 0 = far .. 1 = near
                                              (iw3's Edge Fix, --edge-dilation 2, applied in the graph)
           divergence  float32 [1]            total left-right shift, % of width (iw3 --divergence)
           convergence float32 [1]            0 = all behind the screen .. 1 = all in front
        -> fields      float32 [1, 8, h, w]   left (offset0, offset1, weight0, weight1), then right;
                                              offsets in iw3 grid units, right eye already mirrored back

  warp     color       uint8   [1, H, W, 4]   the picture, BGRA (any size)
           fields      float32 [1, 8, h, w]
        -> nv12        uint8   [1, 3H, W]     NV12 of the 2H x W frame (left eye on top, right
                                              eye below; BT.709 limited range), what the encoder takes

mlbw_l2 comes in strength levels (iw3: divergence <= 4 level 1, <= 7 level 2,
above level 3); each is exported as iw3_mlbw_l2_d<level>_<w>x<h>_fields.onnx,
in fp32: TensorRT builds fp16 engines from it anyway, and ONNX Runtime can
fold its constant subgraphs on the CPU (an fp16 graph leaves Sqrt / Gemm on
constants unfolded, with a warning per node, as the CPU has no fp16 kernels).
The warp graph is no longer used at run time (a CUDA kernel does the same);
it stays as the reference the kernel is tested against.
Parity is checked against iw3's own apply_divergence_nn_delta_weight.

    <nunif>/.venv/Scripts/python.exe scripts/export_iw3_stereo.py --nunif <nunif> --size 770x434 --image photo.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn

LEVELS = {1: 2.0, 2: 5.0, 3: 8.0}  # level -> a divergence that selects it in iw3


class Fields(nn.Module):
    def __init__(self, net: nn.Module, width: int, edge_dilation: int) -> None:
        super().__init__()
        self.net = net
        self.width = width
        self.edge_dilation = edge_dilation

    def features(self, depth, divergence, convergence):
        # iw3 make_divergence_feature_value with image_width = max(h, w) = w.
        pixels = divergence * 0.5 * 0.01 * self.width
        div = (pixels / 32.0).view(1, 1, 1, 1).expand_as(depth)
        conv = (-pixels * convergence / 32.0).view(1, 1, 1, 1).expand_as(depth)
        return torch.cat([depth, div, conv], dim=1)

    def forward(self, depth, divergence, convergence):
        from iw3.dilation import dilate, edge_weight, gaussian_blur

        # iw3 dilate_edge(depth, n) without its inference_mode decorator (not traceable).
        for _ in range(self.edge_dilation):
            weight = edge_weight(depth)
            grown = dilate(gaussian_blur(depth), kernel_size=(3, 3))
            depth = depth * (1 - weight) + grown * weight
        left = self.features(depth, divergence, convergence)
        right = self.features(torch.flip(depth, dims=[3]), divergence, convergence)
        dtype = next(self.net.parameters()).dtype
        delta, weight, _ = self.net._forward(torch.cat([left, right], dim=0).to(dtype))
        delta, weight = delta.float(), weight.float()
        # The right view was computed mirrored: mirror back, and its offsets change sign.
        right_delta = -torch.flip(delta[1:2], dims=[3])
        right_weight = torch.flip(weight[1:2], dims=[3])
        return torch.cat([delta[0:1], weight[0:1], right_delta, right_weight], dim=1)


class Warp(nn.Module):
    def __init__(self, depth_width: int) -> None:
        super().__init__()
        # iw3 backward_warp: grid (-1..1) shift = delta / (w // 2 - 1), w = depth width.
        self.delta_scale = 1.0 / (depth_width // 2 - 1)

    def forward(self, color, fields):
        height, width = color.shape[1], color.shape[2]
        picture = color[..., :3].permute(0, 3, 1, 2).to(torch.float32) / 255.0
        fields = F.interpolate(fields, size=(height, width), mode="bilinear", align_corners=True)
        ys = torch.arange(height, dtype=torch.float32) * (2.0 / (height - 1)) - 1.0
        xs = torch.arange(width, dtype=torch.float32) * (2.0 / (width - 1)) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        eyes = []
        for eye in range(2):
            z = torch.zeros_like(picture)
            for layer in range(2):
                delta = fields[:, eye * 4 + layer]
                weight = fields[:, eye * 4 + 2 + layer].unsqueeze(1)
                grid = torch.stack([grid_x + delta * self.delta_scale, grid_y.expand_as(delta)], dim=-1)
                sampled = F.grid_sample(picture, grid, mode="bilinear", padding_mode="border", align_corners=True)
                z = z + sampled.clamp(0, 1) * weight
            eyes.append(z.clamp(0, 1))
        return to_nv12(torch.cat(eyes, dim=2))


def luma(bgr):
    """BT.709 limited-range Y (0..1 scale) of a [1,3(BGR),H,W] picture in 0..1."""
    b, g, r = bgr[:, 0], bgr[:, 1], bgr[:, 2]
    return (16.0 + 219.0 * (0.2126 * r + 0.7152 * g + 0.0722 * b)) / 255.0


def to_nv12(bgr):
    """[1,3(BGR),H,W] in 0..1 -> NV12 u8 [1, 3H/2, W]: Y plane, then interleaved CbCr
    at half resolution (BT.709, limited range: what the stream's VUI says)."""
    y = luma(bgr)
    half = F.avg_pool2d(bgr, 2)
    b, g, r = half[:, 0], half[:, 1], half[:, 2]
    cb = (128.0 + 224.0 * (-0.1146 * r - 0.3854 * g + 0.5 * b)) / 255.0
    cr = (128.0 + 224.0 * (0.5 * r - 0.4542 * g - 0.0458 * b)) / 255.0
    uv = torch.stack([cb, cr], dim=-1).flatten(-2)  # [1, H/2, W]
    planes = torch.cat([y, uv], dim=1)
    return (planes * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)


def save(model_path: Path, metadata: dict) -> None:
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(model_path))
    print(f"[export] {model_path} ({model_path.stat().st_size / 1e6:.2f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nunif", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models/stereo"))
    parser.add_argument("--size", default="770x434", help="depth map WxH (as exported by export_depth.py)")
    parser.add_argument("--edge-dilation", type=int, default=2, help="iw3 --edge-dilation (0 = off)")
    parser.add_argument("--image", type=Path, required=True, help="photo for the parity check")
    parser.add_argument("--fp16", action="store_true", help="mlbw in fp16 (as iw3 runs it under autocast); default fp32")
    args = parser.parse_args()
    sys.path.insert(0, str(args.nunif.resolve()))
    from iw3.backward_warp import apply_divergence_nn_delta_weight
    from iw3.dilation import dilate_edge
    from iw3.stereo_model_factory import create_stereo_model

    width, height = (int(v) for v in args.size.lower().split("x"))
    args.out.mkdir(parents=True, exist_ok=True)
    source = "nagadomi/nunif iw3 mlbw_l2 (iw3_mlbw_l2_d*_20250627.pth), MIT"

    # Real depth for parity: DA-V2-S (our export) on the photo, min-max normalized.
    from PIL import Image
    depth_session = ort.InferenceSession(f"models/depth/dav2s_{width}x{height}_fp32.onnx", providers=["CPUExecutionProvider"])
    small = np.asarray(Image.open(args.image).convert("RGB").resize((width, height), Image.BILINEAR))
    bgra_small = np.concatenate([small[..., ::-1], np.full((height, width, 1), 255, np.uint8)], -1)[None]
    raw = depth_session.run(["depth"], {"bgra": np.ascontiguousarray(bgra_small)})[0][0]
    depth = torch.from_numpy((raw - raw.min()) / (raw.max() - raw.min()))[None, None].float()
    big = np.asarray(Image.open(args.image).convert("RGB").resize((1920, 1080), Image.BICUBIC))
    color = np.ascontiguousarray(np.concatenate([big[..., ::-1], np.full((1080, 1920, 1), 255, np.uint8)], -1)[None])

    warp_path = args.out / f"iw3_warp_{width}x{height}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            Warp(width).eval(), (torch.from_numpy(color), torch.zeros(1, 8, height, width)), str(warp_path),
            input_names=["color", "fields"], output_names=["nv12"],
            dynamic_axes={"color": {1: "height", 2: "width"}, "nv12": {1: "nv12_rows", 2: "width"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    save(warp_path, {"source": "ImmersiveVR (iw3 backward_warp, MLBW blend)", "output": "NV12 (BT.709 limited) of a frame with the left eye on top, right eye below"})
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    warp = ort.InferenceSession(str(warp_path), providers=providers)

    for level, divergence in LEVELS.items():
        net = create_stereo_model("mlbw_l2", divergence, device_id=-1).float().eval().cpu()
        # fp16 is traced on CUDA (CPU half kernels are incomplete); the parity
        # reference stays the fp32 network.
        device = "cuda" if args.fp16 else "cpu"
        export_net = create_stereo_model("mlbw_l2", divergence, device_id=0).half().eval() if args.fp16 else net
        path = args.out / f"iw3_mlbw_l2_d{level}_{width}x{height}_fields.onnx"
        with torch.no_grad():
            torch.onnx.export(
                Fields(export_net, width, args.edge_dilation).eval().to(device),
                (depth.to(device), torch.tensor([divergence], device=device), torch.tensor([0.5], device=device)), str(path),
                input_names=["depth", "divergence", "convergence"], output_names=["fields"],
                opset_version=17, do_constant_folding=True, dynamo=False,
            )
        save(path, {"source": source, "level": str(level), "output": "fields [1,8,h,w]: left d0 d1 w0 w1, right d0 d1 w0 w1"})

        fields_session = ort.InferenceSession(str(path), providers=providers)
        feeds = {"depth": depth.numpy(), "divergence": np.array([divergence], np.float32), "convergence": np.array([0.5], np.float32)}
        fields = fields_session.run(["fields"], feeds)[0]
        nv12 = warp.run(["nv12"], {"color": color, "fields": fields})[0][0].astype(np.float32) / 255.0
        luma_plane = nv12[:2160]
        picture = torch.from_numpy(big).permute(2, 0, 1)[None].float() / 255.0
        with torch.no_grad():
            dilated = dilate_edge(depth, args.edge_dilation) if args.edge_dilation else depth
            reference = [
                apply_divergence_nn_delta_weight(net, picture, dilated, divergence, 0.5, steps=1, shift=shift, enable_amp=False)[0]
                for shift in (-1, 1)
            ]
        for eye, (name, rows) in enumerate((("left", slice(0, 1080)), ("right", slice(1080, 2160)))):
            # Compare luma: the reference is RGB, ours NV12 (chroma at half resolution).
            expected = luma(reference[eye][[2, 1, 0]][None]).numpy()[0]
            error = np.abs(luma_plane[rows] - expected)
            print(f"[parity] level {level} {name} eye luma: max {error.max():.4f} mean {error.mean():.5f} (8-bit output)")
        for _ in range(10):
            fields_session.run(["fields"], feeds)
        started = time.perf_counter()
        for _ in range(100):
            fields_session.run(["fields"], feeds)
        fields_ms = (time.perf_counter() - started) * 10
        print(f"[bench] level {level} fields ({fields_session.get_providers()[0]}): {fields_ms:.2f} ms")

    for size in ((1920, 1080), (2560, 1440)):
        c = np.zeros((1, size[1], size[0], 4), np.uint8)
        for _ in range(5):
            warp.run(["nv12"], {"color": c, "fields": fields})
        started = time.perf_counter()
        for _ in range(50):
            warp.run(["nv12"], {"color": c, "fields": fields})
        print(f"[bench] warp {size[0]}x{size[1]} -> {size[0]}x{size[1] * 2} ({warp.get_providers()[0]}): "
              f"{(time.perf_counter() - started) * 20:.2f} ms incl. host transfers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
