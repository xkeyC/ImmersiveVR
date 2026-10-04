# ImmersiveVR

[中文](README.md) | English

ImmersiveVR turns your Windows desktop into stereoscopic 3D in real time and streams it to a VR headset such as a Meta Quest. You watch it in the headset's browser, so there is nothing to install on the headset.

The PC captures the screen and estimates depth for every pixel. It then renders both eyes with [iw3](https://github.com/nagadomi/nunif)'s mlbw_l2 network and encodes them with NVENC straight from GPU memory. The stream travels over your LAN to the headset. There the browser decodes it in hardware with WebCodecs and shows it as an adjustable, curved virtual screen through WebXR compositor layers.

## Features

- **Real-time 2D to 3D.** Depth-Anything-V2-Small depth plus iw3 mlbw_l2 stereo, computed for every frame. Object outlines stay clean, with no ghosting.
- **Entirely on the GPU.** Capture (D3D11 to CUDA), scaling, the depth and stereo models (TensorRT), the eye rendering (a CUDA kernel) and encoding (NVENC) all stay in GPU memory. At 1440p every frame is new at 60 fps, using about half the GPU.
- **Low latency.**
  - Encoder settings follow [Sunshine](https://github.com/LizardByte/Sunshine). Every frame fits within a frame time of the link, and keyframes are sent only on demand, so there are no periodic bursts.
  - The server drops frames to catch up when a client falls behind.
  - Capture to display is about 12–15 ms on the local machine.
- **Smooth motion.**
  - However fast the monitor refreshes (4K 240 Hz, for example), the newest capture is sampled on a steady 60 Hz grid.
  - The headset switches to 120 Hz when it offers that rate, so every frame is shown for the same time.
- **Sound.** The PC's audio plays in the headset with almost no added delay. Optionally the PC's speakers are muted while streaming (on by default).
- **Adjust everything in VR.** Stream resolution (1080p / 1440p / 2160p), codec (H.265 / AV1), 3D strength, convergence, and screen distance, size, curvature and height. In passthrough you can also set background opacity. Settings are saved automatically.
- **Passthrough.** Watch the virtual screen in your real room, where the device supports it.
- **Installable.** The page has a web app manifest, so you can add it to the headset's app library.

## Requirements

| | |
|---|---|
| OS | Windows 10 2004 or later / Windows 11 (64-bit) |
| GPU | NVIDIA RTX 20 series or newer (AV1 encoding needs RTX 40 series), driver 550 or newer |
| Headset | Meta Quest 3 / 3S (Meta Quest Browser). Other headset browsers with WebXR and WebCodecs (hardware H.265 or AV1 decoding) should work too |
| Network | PC and headset on the same LAN. A wired PC and a Wi-Fi 6/6E headset are recommended |

## Usage

### 1. Download

Download the two v0.0.1 packages from [Releases](../../releases):

| File | Contents |
|---|---|
| `ImmersiveVR-v0.0.1-windows-x64.zip` | `immersive-vr.exe`, ONNX Runtime (CUDA 12 build) and the CUDA / cuDNN runtime libraries it needs |
| `ImmersiveVR-models-v0.0.1.zip` | Models: `models/depth/` (Depth-Anything-V2-Small, 770×434) and `models/stereo/` (iw3 mlbw_l2, three strength levels) |

Extract both into **the same folder**:

```
ImmersiveVR/
├─ immersive-vr.exe
├─ runtime/ort/…          ONNX Runtime and runtime libraries
└─ models/
   ├─ depth/…
   └─ stereo/…
```

Optional: for faster depth and stereo models (about 2×), get [TensorRT 10](https://developer.nvidia.com/tensorrt) (CUDA 12 build) and add `lib=<TensorRT bin or lib directory>` to `runtime/runtime.txt`. The first start builds the engines, which takes a few minutes; later starts load them from a cache. Without TensorRT it falls back to CUDA.

### 2. Start

Run `immersive-vr.exe` from that folder. On first run, allow it through Windows Firewall on **private networks**.

The log lists the addresses to open, for example:

```
serving on https://192.168.1.20:13256/
```

### 3. Open it in the headset

1. In the Quest browser, open the address from the log (`https://<PC IP>:13256`).
2. The certificate is self-signed (generated at startup), so the browser warns once. Choose *Advanced → Proceed*.
3. To see your room, turn on the **passthrough** toggle next to the enter button.
4. Press **进入 VR** (Enter VR).

You can also install the page as an app from the browser menu and open it from the app library next time.

### 4. In VR

- Point the controller ray at the panel and pull the trigger, or pinch with hand tracking.
- The grip button shows or hides the panel. While it is hidden, a trigger pull or pinch brings it back.
- Panel options:
  - **Resolution** and **codec** go to the PC, which restarts its encoder. A codec the device cannot decode is swapped for the other one automatically.
  - **3D strength**: how strong the depth effect is. The default is 0.5%; too high becomes uncomfortable.
  - **Convergence**: at 0 the image pops out of the screen; at 1 it recedes into it.
  - **Screen distance / size / curvature / height** shape the virtual screen. **Recenter** moves it in front of you.
  - **Background opacity** (passthrough): how much of the room shows. At 100% you see all of it; at 0% the background is fully opaque, so only the virtual screen shows. The slider moves in 5% steps, and clicking just past either end sets 0% or 100%.
  - **Mute PC**: mutes the PC's speakers while streaming, so sound only plays in the headset. It is on by default, and the PC's state is restored when you disconnect or the server stops.
- The panel's header shows resolution, frame rate, bitrate and latency. "收 / 显" are capture-to-received and capture-to-displayed times, corrected for the clock difference between PC and headset.

On the PC, open `https://localhost:13256` and press 预览 (Preview) to see both eyes side by side and use the panel with the mouse.

### 5. Command line

The most useful options are below; run `immersive-vr.exe --help` for the full list.

| Option | Default | Meaning |
|---|---|---|
| `--resolution` | `1440` | Initial stream resolution (1080 / 1440 / 2160, never above the screen's) |
| `--codec` | `hevc` | Initial codec: `hevc` or `av1` |
| `--bitrate` | by resolution | Fixed bitrate in Mbit/s (default 40 / 60 / 100 for 1080p / 1440p / 2160p) |
| `--fps` | `60` | Output frame rate |
| `--preset` | `1` | NVENC preset: 1 fastest and lowest latency, 7 best quality per bit |
| `--monitor` | `0` | Monitor to capture: 0 = primary, otherwise its Windows number (1-based) |
| `--listen` | `0.0.0.0:13256` | Address to serve on |
| `--mute-pc` | `true` | Whether streaming mutes the PC by default (also switchable in the panel) |
| `--divergence` / `--convergence` | `0.5` / `0.5` | Initial 3D strength and convergence |
| `--providers` | `trt,cuda,dml,cpu` | Inference backends for the depth and stereo models, tried in order |
| `--http` | off | Plain HTTP instead of HTTPS. WebXR then only works as `http://localhost`, for example via `adb reverse` |

### 6. Troubleshooting

- **The headset cannot open the page.** Check that both are on the same LAN, that the firewall allows private networks, and that the address starts with `https`.
- **"Cannot decode".** Switch the codec in the panel. Quest 3 decodes both H.265 and AV1 in hardware.
- **Latency or stutter.** Lower the resolution or bitrate. Wire the PC to the router, keep the headset close to it, and use the 5 GHz or 6 GHz band.
- **3D too strong or too weak.** Adjust 3D strength. Lower it if the image is straining.

## How it works

```
WGC capture (D3D11 -> CUDA memory, up to 4x the output rate)
  └─ the newest capture on a steady 60 Hz grid
       ├─ GPU scale to the stream resolution ──────────────────────────┐
       └─ GPU scale to 770x434 -> depth (TensorRT) -> iw3 mlbw_l2 (TensorRT)
                     per eye: two sampling offsets + blend weights (stay on the GPU)
            CUDA kernel: both eyes at full resolution, stacked, as NV12   ▼
          NVENC (H.265 / AV1, from GPU memory) -> WebSocket (video + PCM sound)
                                                                         ▼
          headset browser: WebCodecs hardware decode -> WebXR stereo cylinder layer (one image per eye)
```

## Performance

### Test setup

- **Hardware:** RTX 4090, a 4K 240 Hz primary monitor, driver 616.56.
- **Runtime:** ONNX Runtime 1.30 (CUDA 12) with TensorRT 10.16; NVENC preset P1, H.265.
- **Content:** the `--synthetic` test pattern, 4K and scrolling sideways every frame like a game in constant motion. This is harder than a normal desktop.
- **Latency:** measured on the same PC with headless Chrome as the client. Wi-Fi and headset decoding are not included.

### Stage times (every frame new, 60 fps out)

| | 1440p per eye (frame 2560×2880) | 2160p per eye (frame 3840×4320) |
|---|---|---|
| New frames / s | 60 | 60 |
| GPU load | ~48% | ~48% |
| Capture scaled to eye size (GPU) | 0.14 ms | 0 (the capture is the eye size: shared) |
| Depth model (TensorRT, with normalization) | ~3.4 ms | ~3.6 ms |
| mlbw_l2 stereo fields (TensorRT) | ~2.3 ms | ~3.4 ms |
| Eye rendering (CUDA kernel) | 0.12 ms | 0.20 ms |
| NVENC encode (P1) | 2.9 ms | 5.3 ms |
| **Capture to encoded (median)** | **~8 ms** | **~10 ms** |
| Capture to received / displayed in the browser | 9–11 ms / 12–15 ms | — |

Bitrate is about 48 Mbit/s at 1440p (target 60) and 82–100 Mbit/s at 2160p. The browser decodes both H.265 and AV1 at 60 fps.

### Frame pacing (60 fps out from sources of different refresh rates)

| Source | Output | Arrival gaps p10 / median / p90 | Capture to encoded |
|---|---|---|---|
| 60 Hz | 60.0 fps | 16.2 / 16.7 / 17.1 ms | 4.5 ms |
| 144 Hz | 60.0 fps | 15.9 / 16.7 / 17.4 ms | 7.7 ms |
| 240 Hz (1440p) | 60.0 fps | 16.2 / 16.6 / 17.2 ms | 7.8 ms |
| 240 Hz (2160p) | 60.0 fps | 16.1 / 16.7 / 17.2 ms | 10.2 ms |

### Encoded frame sizes (2160p, 11 s)

Frames averaged 167 KB with a maximum of 183 KB, and the run contained a single keyframe, the one requested on connect. No frame exceeds what the link carries in a frame time, so there is no periodic keyframe stutter.

### Before and after optimization

| | First version | Now |
|---|---|---|
| New frames / s at 1440p | ~52 (visible drops) | 60 |
| New frames / s at 2160p | ~29 | 60 |
| GPU load at 1440p | 60–88% | ~48% (with depth and stereo now on every frame) |
| Encoder input to client received | 32 ms (ffmpeg pipe) | ~4 ms (in-process NVENC) |
| Eye rendering at 2160p | 17.8 ms (ONNX graph) | 0.2 ms (CUDA kernel) |
| Depth / stereo models | 4.1 / 5.0 ms (CUDA) | 1.9 / 2.3 ms (TensorRT, standalone) |
| Encode at 1440p | 5.4 ms (P4) | 2.9 ms (P1 + two-pass) |
| Keyframes | one per second, up to ~15× an average frame | on demand, none bigger than a frame time |

## Building from source

### Prerequisites

- Windows 10/11 x64, an NVIDIA GPU and driver.
- [Rust](https://rustup.rs/). `rust-toolchain.toml` pins 1.95.0, which rustup installs automatically.
- ONNX Runtime 1.30, GPU build. Either:
  - `python scripts/fetch_onnxruntime.py`, which puts the CUDA 13 build in `runtime/ort/` and needs the CUDA 13 and cuDNN 9 runtimes; or
  - `onnxruntime-win-x64-gpu_cuda12-1.30.0.zip` from the [official releases](https://github.com/microsoft/onnxruntime/releases), the CUDA 12 build, which works with TensorRT 10.
- CUDA Toolkit 12.x, only to regenerate `kernels.ptx` after editing `kernels.cu`. The compiled PTX is in the repository.

### Build

```bash
cargo build --release -p immersive-vr
```

The binary is `target/release/immersive-vr.exe`. Run from the repository root, it reads models from `models/` and ONNX Runtime from `runtime/ort/` by default.

For libraries elsewhere, list them in `runtime/runtime.txt` (not tracked). Entries are added to the front of the DLL search path in order. `--ort` and `--lib-dir` on the command line take precedence.

```
# ONNX Runtime library
ort=D:/libs/onnxruntime-win-x64-gpu_cuda12-1.30.0/lib/onnxruntime.dll
# CUDA 12 / cuDNN 9 / TensorRT 10 DLL directories
lib=D:/libs/cuda12/bin
lib=D:/libs/TensorRT-10/lib
```

### Models

Use the model release package, or export them yourself:

```bash
# 1. Depth: put depth-anything/Depth-Anything-V2-Small-hf in models/source/
python scripts/export_depth.py --source models/source/Depth-Anything-V2-Small-hf --out models/depth --size 770x434 --image <photo>
# 2. Stereo: needs nunif's Python environment (git clone https://github.com/nagadomi/nunif and follow its setup)
<nunif>/.venv/Scripts/python.exe scripts/export_iw3_stereo.py --nunif <nunif> --size 770x434 --image <photo>
```

### Tests

```bash
cargo test --release --workspace
# tests that need the models and a GPU (skipped without these variables)
IVR_DEPTH_MODEL_DIR=models/depth IVR_STEREO_MODEL_DIR=models/stereo ORT_DYLIB_PATH=runtime/ort/onnxruntime.dll \
  cargo test --release --workspace
```

Development tools:
- `--synthetic` streams a moving test pattern instead of the screen, for throughput measurements.
- `--synthetic-fps 240` emulates a high-refresh monitor.
- `--never-mute-pc` keeps tests away from the PC's speakers.
- `scripts/ws_probe.mjs` reports a stream's frame rate, bitrate and arrival gaps.
- `scripts/page_check.mjs` drives the web client in headless Chrome.

### Layout

| Path | Contents |
|---|---|
| `crates/immersive-vr` | The server: capture, pipeline, CUDA kernels (`kernels.cu`), NVENC, sound, HTTPS / WebSocket. `web/` is the client, embedded at build time |
| `crates/depth-infer` | ONNX Runtime inference: depth, mlbw stereo fields, depth normalization and smoothing |
| `scripts/` | Model export, ONNX Runtime download, test tools |
| `docs/` | Research notes |

## Acknowledgements

- [nunif / iw3](https://github.com/nagadomi/nunif) (nagadomi, MIT; weights MIT): the mlbw_l2 stereo network and backward warp.
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2): depth estimation (the Small model, Apache-2.0).
- [Sunshine](https://github.com/LizardByte/Sunshine) (LizardByte): ideas for low-latency encoder settings and frame pacing.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [ort](https://github.com/pykeio/ort): inference.
- [moq-nvenc](https://crates.io/crates/moq-nvenc) (derived from [nvidia-video-codec-sdk](https://github.com/ViliamVadocz/nvidia-video-codec-sdk)) and [cudarc](https://github.com/coreylowman/cudarc): NVENC and CUDA bindings.
- [windows-capture](https://github.com/NiiightmareXD/windows-capture), [wasapi-rs](https://github.com/HEnquist/wasapi-rs), [axum](https://github.com/tokio-rs/axum) and the rest of the Rust ecosystem.

## License

The code is dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. The models in the model release keep their own licenses: Depth-Anything-V2-Small is Apache-2.0, and the iw3 weights are MIT.
