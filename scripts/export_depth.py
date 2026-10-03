"""Exports Depth-Anything-V2-Small to fixed-shape ONNX graphs for depth-infer.

Each graph takes the frame the capture side already has and returns raw
relative depth, so the Rust side does no per-pixel work before inference:

    input  `bgra`  uint8   [1, H, W, 4]   (B, G, R, A as DXGI/WGC deliver it)
    output `depth` float32 [1, H, W]      (relative inverse depth: larger = nearer)

Channel swap, /255 and ImageNet normalization are part of the graph; the DINOv2
position embedding is interpolated once for the fixed size and stored as a
constant. Normalizing the output range is left to the caller (it is smoothed
over time there; a per-frame min/max flickers).

Sizes are W x H with both multiples of 14 and close to 16:9: 518x294 (37x21
patches) and 378x210 (27x15 patches). fp16 is for the CUDA EP; fp32 is for the
TensorRT EP, which builds its own fp16 engine from it (`trt_fp16_enable`).

    python scripts/export_depth.py --source models/source/Depth-Anything-V2-Small-hf \
        --out models/depth --image <a photo>

Needs: torch, transformers, onnx, onnxruntime (or onnxruntime-gpu), numpy, pillow.
Source checkpoint: depth-anything/Depth-Anything-V2-Small-hf (Apache-2.0),
revision 5426e4f0f36572d16453bbda7a8389317b1bef99.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from transformers import DepthAnythingForDepthEstimation

SIZES = ((518, 294), (378, 210))
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
OPSET = 17


class BgraDepth(nn.Module):
    """uint8 BGRA [1,H,W,4] -> float32 relative depth [1,H,W]."""

    def __init__(self, model: DepthAnythingForDepthEstimation, dtype: torch.dtype) -> None:
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, bgra: torch.Tensor) -> torch.Tensor:
        # Cast first: TensorRT takes uint8 only as a network input feeding a
        # Cast, so a Gather/Transpose on uint8 would stay outside its engine
        # (and rule out CUDA graph capture on the TensorRT EP).
        rgb = bgra.to(torch.float32)[..., [2, 1, 0]].permute(0, 3, 1, 2) / 255.0
        pixels = ((rgb - self.mean) / self.std).to(self.dtype)
        return self.model(pixel_values=pixels).predicted_depth.to(torch.float32)


def bake_position_embedding(model: DepthAnythingForDepthEstimation, width: int, height: int) -> None:
    """Replaces the per-run bicubic interpolation with its result for this size."""
    embeddings = model.backbone.embeddings
    patches = (height // 14) * (width // 14)
    dummy = torch.zeros(1, patches + 1, embeddings.position_embeddings.shape[-1])
    with torch.no_grad():
        baked = embeddings.interpolate_pos_encoding(dummy, height, width).detach().clone()
    embeddings.register_buffer("baked_position_embeddings", baked)
    embeddings.interpolate_pos_encoding = lambda emb, h, w: embeddings.baked_position_embeddings.to(emb.dtype)


def load(source: Path, width: int, height: int, dtype: torch.dtype, device: str) -> BgraDepth:
    model = DepthAnythingForDepthEstimation.from_pretrained(source, torch_dtype=torch.float32).eval()
    bake_position_embedding(model, width, height)
    return BgraDepth(model.to(dtype), dtype).to(device).eval()


def export(source: Path, out: Path, width: int, height: int, precision: str, device: str) -> Path:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    wrapper = load(source, width, height, dtype, device)
    dummy = torch.zeros(1, height, width, 4, dtype=torch.uint8, device=device)
    path = out / f"dav2s_{width}x{height}_{precision}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(path),
            input_names=["bgra"],
            output_names=["depth"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    model.producer_name = "ImmersiveVR export_depth.py"
    for key, value in {
        "source": "depth-anything/Depth-Anything-V2-Small-hf@5426e4f0f36572d16453bbda7a8389317b1bef99",
        "license": "Apache-2.0",
        "input_layout": "bgra u8 [1,H,W,4]",
        "output": "relative inverse depth f32 [1,H,W]",
        "width": str(width),
        "height": str(height),
        "precision": precision,
    }.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))
    print(f"[export] {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def test_frame(image: Path | None, width: int, height: int) -> np.ndarray:
    """A BGRA frame at the model size: the given photo resized, or a synthetic scene."""
    if image is not None:
        from PIL import Image

        rgb = np.asarray(Image.open(image).convert("RGB").resize((width, height), Image.BICUBIC))
    else:
        y, x = np.mgrid[0:height, 0:width]
        rgb = np.stack([x * 255 // width, y * 255 // height, (x + y) % 256], axis=-1).astype(np.uint8)
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb[..., ::-1], alpha], axis=-1)[None].copy()


def normalized(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float64)
    return (depth - depth.min()) / max(depth.max() - depth.min(), 1e-12)


def parity(source: Path, paths: list[Path], width: int, height: int, image: Path | None) -> dict:
    import onnxruntime as ort

    frame = test_frame(image, width, height)
    reference_model = load(source, width, height, torch.float32, "cpu")
    with torch.no_grad():
        reference = reference_model(torch.from_numpy(frame)).numpy()
    available = ort.get_available_providers()
    results = {}
    for path in paths:
        providers = ["CPUExecutionProvider"]
        if "fp16" in path.name and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(str(path), providers=providers)
            output = session.run(["depth"], {"bgra": frame})[0]
        except Exception as error:  # e.g. a CUDA EP that cannot load its DLLs
            print(f"[parity] {path.name}: {providers[0]} failed ({error}); retrying on CPU")
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            output = session.run(["depth"], {"bgra": frame})[0]
        diff = np.abs(normalized(output) - normalized(reference))
        results[path.name] = {
            "provider": session.get_providers()[0],
            "shape": list(output.shape),
            "max_abs_normalized": float(diff.max()),
            "mean_abs_normalized": float(diff.mean()),
            "correlation": float(np.corrcoef(output.ravel(), reference.ravel())[0, 1]),
        }
        print(f"[parity] {path.name}: {results[path.name]}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="Depth-Anything-V2-Small-hf checkout")
    parser.add_argument("--out", type=Path, default=Path("models/depth"))
    parser.add_argument("--image", type=Path, default=None, help="photo for the parity check")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--size", action="append", default=None, help="WxH, both multiples of 14 (repeatable; default 518x294 and 378x210)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}
    sizes = [tuple(int(v) for v in size.lower().split("x")) for size in args.size] if args.size else SIZES
    for width, height in sizes:
        paths = [export(args.source, args.out, width, height, precision, device) for precision in ("fp16", "fp32")]
        if not args.skip_parity:
            report[f"{width}x{height}"] = parity(args.source, paths, width, height, args.image)
    if report:
        (args.out / "parity.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
