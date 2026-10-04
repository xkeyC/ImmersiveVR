"""Packs the release zips into target/dist. Every zip unpacks into the same
folder:

  ImmersiveVR-v<ver>-windows-x64.zip          the programs: immersive-vr.exe
                                               (web streaming) and ImmersiveVR.exe
                                               (PCVR client), plus the VC++ runtime
  ImmersiveVR-models-v<ver>.zip               models/depth, models/stereo
  ImmersiveVR-runtime-v<ver>-windows-x64.zip  runtime/ort: ONNX Runtime (CUDA 12),
                                               CUDA 12, cuDNN 9
  ImmersiveVR-runtime-tensorrt-v<ver>-windows-x64.zip
                                               runtime/ort: TensorRT 10 for RTX 20-50

Before packing the programs:
  cargo build --release -p immersive-vr -p ivr-native
  Unity: ImmersiveVR > Build Release (target/unity/ImmersiveVR)

The runtime libraries come from this machine's runtime/runtime.txt (`ort=`
and `lib=` lines, as the programs read them): each file from the first
directory that has it, the same rule the programs load them by.

  python scripts/package_release.py [--version 0.0.1] [--only programs,models,runtime,tensorrt]
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIST = REPO / "target" / "dist"

# What the CUDA and TensorRT providers load (see depth-infer's runtime.rs).
ORT_FILES = [
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_tensorrt.dll",
]
CUDA_FILES = [
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    # cuDNN's runtime-compiled engines.
    "nvrtc64_120_0.dll",
    "nvrtc-builtins64_124.dll",
]
# TensorRT builds engines with the resource of the GPU's architecture:
# Turing, Ampere, Ada and Blackwell consumer cards (RTX 20 / 30 / 40 / 50).
# Other GPUs fall back to the CUDA provider.
TENSORRT_FILES = [
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll",
    "nvinfer_builder_resource_sm75_10.dll",
    "nvinfer_builder_resource_sm86_10.dll",
    "nvinfer_builder_resource_sm89_10.dll",
    "nvinfer_builder_resource_sm120_10.dll",
]
MODEL_FILES = [
    "models/depth/dav2s_770x434_fp32.onnx",
    "models/depth/dav2s_770x434_fp16.onnx",
    "models/stereo/iw3_mlbw_l2_d1_770x434_fields.onnx",
    "models/stereo/iw3_mlbw_l2_d2_770x434_fields.onnx",
    "models/stereo/iw3_mlbw_l2_d3_770x434_fields.onnx",
]
# The MSVC runtime both programs and ONNX Runtime link against, app-local.
VC_RUNTIME = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"]

RUNTIME_NOTICE = """ImmersiveVR runtime libraries (unpack next to immersive-vr.exe / ImmersiveVR.exe)

runtime/ort/ holds third-party libraries, redistributed under their licenses:
- ONNX Runtime 1.30 (MIT): see LICENSE-onnxruntime.txt and ThirdPartyNotices-onnxruntime.txt
- NVIDIA CUDA 12 runtime, cuBLAS, NVRTC: NVIDIA CUDA Toolkit EULA,
  https://docs.nvidia.com/cuda/eula/
- NVIDIA cuDNN 9: NVIDIA cuDNN Software License Agreement,
  https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html
"""
TENSORRT_NOTICE = """ImmersiveVR TensorRT libraries (optional; unpack next to immersive-vr.exe / ImmersiveVR.exe)

runtime/ort/ gets NVIDIA TensorRT 10 (CUDA 12) for RTX 20 / 30 / 40 / 50 GPUs,
redistributed under the NVIDIA TensorRT Software License Agreement:
https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html
"""
MODELS_NOTICE = """ImmersiveVR models (unpack next to immersive-vr.exe / ImmersiveVR.exe)

- models/depth: Depth-Anything-V2-Small (Apache-2.0),
  https://github.com/DepthAnything/Depth-Anything-V2
  Exported to ONNX at 770x434 (scripts/export_depth.py).
- models/stereo: iw3 mlbw_l2 (MIT, nagadomi), https://github.com/nagadomi/nunif
  Exported to ONNX at 770x434 (scripts/export_iw3_stereo.py).
"""


def runtime_dirs():
    """The `ort=` library's directory, then the `lib=` directories, in order."""
    file = REPO / "runtime" / "runtime.txt"
    ort, libs = None, []
    if file.is_file():
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ort="):
                ort = Path(line[4:].strip())
            elif line.startswith("lib="):
                libs.append(Path(line[4:].strip()))
    if ort is None:
        ort = REPO / "runtime" / "ort" / "onnxruntime.dll"
    if not ort.is_file():
        sys.exit(f"ONNX Runtime not found: {ort} (set ort= in runtime/runtime.txt)")
    return ort.parent, [ort.parent, *libs]


def find(name, dirs):
    for directory in dirs:
        path = directory / name
        if path.is_file():
            return path
    sys.exit(f"{name} not found in: " + ", ".join(str(d) for d in dirs))


class Zip:
    def __init__(self, name):
        self.path = DIST / name
        self.path.unlink(missing_ok=True)
        self.zip = zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED, compresslevel=6)
        self.raw = 0

    def add(self, source, name):
        self.zip.write(source, name)
        self.raw += Path(source).stat().st_size

    def text(self, name, content):
        self.zip.writestr(name, content)

    def close(self):
        self.zip.close()
        size = self.path.stat().st_size
        print(f"{self.path.name}: {self.raw / 2**20:,.0f} MB -> {size / 2**20:,.0f} MB")
        if size >= 2 * 2**30:
            sys.exit(f"{self.path.name} is over GitHub's 2 GB asset limit")


def programs(version):
    server = REPO / "target" / "release" / "immersive-vr.exe"
    player = REPO / "target" / "unity" / "ImmersiveVR"
    if not server.is_file():
        sys.exit("build it first: cargo build --release -p immersive-vr -p ivr-native")
    if not (player / "ImmersiveVR.exe").is_file():
        sys.exit("build the PCVR client first: Unity > ImmersiveVR > Build Release")
    if not (player / "ImmersiveVR_Data" / "Plugins" / "x86_64" / "ivr_native.dll").is_file():
        sys.exit("the PCVR client has no ivr_native.dll: rebuild it from Unity")
    out = Zip(f"ImmersiveVR-v{version}-windows-x64.zip")
    out.add(server, "immersive-vr.exe")
    for root, dirs, files in os.walk(player):
        # IL2CPP / Burst debug output must not ship.
        dirs[:] = [d for d in dirs if "DontShip" not in d and not d.endswith("_DoNotShip")]
        for file in files:
            path = Path(root) / file
            out.add(path, path.relative_to(player).as_posix())
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for name in VC_RUNTIME:
        out.add(find(name, [system]), name)
    for name in ["LICENSE-MIT", "LICENSE-APACHE"]:
        out.add(REPO / name, name)
    out.close()


def models(version):
    out = Zip(f"ImmersiveVR-models-v{version}.zip")
    for name in MODEL_FILES:
        path = REPO / name
        if not path.is_file():
            sys.exit(f"{name} is missing (see the README's model export)")
        out.add(path, name)
    out.text("models/NOTICE.txt", MODELS_NOTICE)
    out.close()


def runtime(version):
    ort_dir, dirs = runtime_dirs()
    out = Zip(f"ImmersiveVR-runtime-v{version}-windows-x64.zip")
    for name in ORT_FILES + CUDA_FILES:
        out.add(find(name, dirs), f"runtime/ort/{name}")
    package = ort_dir.parent
    for name in ["LICENSE", "ThirdPartyNotices.txt"]:
        if (package / name).is_file():
            stem = Path(name).stem
            out.add(package / name, f"runtime/ort/{stem}-onnxruntime.txt")
    out.text("runtime/ort/NOTICE.txt", RUNTIME_NOTICE)
    out.close()


def tensorrt(version):
    _, dirs = runtime_dirs()
    out = Zip(f"ImmersiveVR-runtime-tensorrt-v{version}-windows-x64.zip")
    for name in TENSORRT_FILES:
        out.add(find(name, dirs), f"runtime/ort/{name}")
    out.text("runtime/ort/NOTICE-tensorrt.txt", TENSORRT_NOTICE)
    out.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="0.0.1")
    parser.add_argument("--only", default="programs,models,runtime,tensorrt",
                        help="comma-separated: programs, models, runtime, tensorrt")
    args = parser.parse_args()
    DIST.mkdir(parents=True, exist_ok=True)
    steps = {"programs": programs, "models": models, "runtime": runtime, "tensorrt": tensorrt}
    for name in args.only.split(","):
        if name not in steps:
            sys.exit(f"unknown package {name!r}; choose from {', '.join(steps)}")
        steps[name](args.version)


if __name__ == "__main__":
    main()
