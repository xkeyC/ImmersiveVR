# ImmersiveVR

中文 | [English](README.en.md)

ImmersiveVR 把 Windows 桌面实时转成立体 3D，串流到 Meta Quest 等 VR 头显，在浏览器里观看。头显上不用装任何应用。

PC 采集屏幕后估计每个像素的深度，用 [iw3](https://github.com/nagadomi/nunif) 的 mlbw_l2 网络在 PC 上渲染出左右两只眼的画面，再用 NVENC 直接从显存编码，通过局域网发给头显。头显浏览器用 WebCodecs 硬件解码，再通过 WebXR 合成层显示成一块可调的虚拟曲面屏。

## 特性

- **实时 2D 转 3D**：Depth-Anything-V2-Small 深度估计 + iw3 mlbw_l2 立体生成，每帧都算。物体轮廓清晰，没有重影。
- **全程在显卡上**：采集（D3D11 → CUDA）、缩放、深度和立体模型（TensorRT）、左右眼渲染（CUDA 核函数）、编码（NVENC）都在显存里完成。1440p 下每秒 60 帧都是新画面，GPU 占用约一半。
- **低延迟**：编码参数参考 [Sunshine](https://github.com/LizardByte/Sunshine)，每帧大小控制在一帧时间内能传完，关键帧只在需要时发，不会周期性突发。客户端积压时服务端主动丢帧追上。本机实测采集到上屏约 12–15 ms。
- **画面平稳**：显示器刷新率再高（例如 4K 240 Hz），也按固定 60 Hz 节拍取最新一帧。头显支持时自动把刷新率设成 120 Hz，每帧显示的时长一致。
- **声音**：电脑的声音同步传到头显，几乎没有额外延迟。可以选择串流时把电脑扬声器静音（默认开启）。
- **在 VR 里调整**：传输分辨率（1080p / 1440p / 2160p）、编码（H.265 / AV1）、立体强度、会聚、屏幕距离、宽度、曲率、高度，透视模式下还能调背景透明度。设置自动保存。
- **透视**：设备支持时可以在真实房间里看虚拟屏幕。
- **可安装为应用**：网页带 PWA 清单，可以安装到头显的应用列表里。

## 系统要求

| | 要求 |
|---|---|
| 系统 | Windows 10 2004 及以上 / Windows 11（64 位） |
| 显卡 | NVIDIA RTX 20 系列或更新（AV1 编码需要 RTX 40 系列），驱动 550 或更新 |
| 头显 | Meta Quest 3 / 3S（Meta Quest 浏览器）；其他支持 WebXR 和 WebCodecs（H.265 或 AV1 硬解）的头显浏览器理论上也可用 |
| 网络 | PC 与头显在同一局域网。推荐 PC 用有线连接，头显用 Wi-Fi 6/6E |

## 使用说明

### 1. 下载

在 [Releases](../../releases) 下载 v0.0.1 的两个包：

| 文件 | 内容 |
|---|---|
| `ImmersiveVR-v0.0.1-windows-x64.zip` | 主程序 `immersive-vr.exe`、ONNX Runtime（CUDA 12 版）及其需要的 CUDA / cuDNN 运行库 |
| `ImmersiveVR-models-v0.0.1.zip` | 模型：`models/depth/`（Depth-Anything-V2-Small，770×434）和 `models/stereo/`（iw3 mlbw_l2 三个强度档） |

两个包解压到**同一个文件夹**，目录结构如下：

```
ImmersiveVR/
├─ immersive-vr.exe
├─ runtime/ort/…          ONNX Runtime 与运行库
└─ models/
   ├─ depth/…
   └─ stereo/…
```

可选：想让深度和立体模型更快（大约快 2 倍），可以另外下载 [TensorRT 10](https://developer.nvidia.com/tensorrt)（CUDA 12 版），在 `runtime/runtime.txt` 里加一行 `lib=<TensorRT 的 bin 或 lib 目录>`。首次启动会构建引擎，需要几分钟，之后直接读取缓存。不装也能用，会自动改用 CUDA。

### 2. 启动

在该文件夹里双击 `immersive-vr.exe`，或者在终端运行它。首次运行 Windows 防火墙会弹窗，请允许**专用网络**。

启动日志会列出访问地址，例如：

```
serving on https://192.168.1.20:13256/
```

### 3. 在头显里打开

1. 在 Quest 浏览器里打开日志中的地址（`https://<电脑 IP>:13256`）。
2. 程序用的是启动时自动生成的自签证书，浏览器会提示「连接不是私密连接」：点「高级」→「继续前往」。每台设备只需要一次。
3. 想在真实房间里看，先打开「进入 VR」旁边的**透视**开关。
4. 点击**进入 VR**。

也可以在浏览器菜单里把网页安装为应用，以后直接从应用列表打开。

### 4. VR 里的操作

- 用手柄射线指向面板按扳机，或者手势捏合，操作控制面板。
- 握持键开关面板；面板隐藏时，按扳机或捏合会把它叫回来。
- 面板选项：
  - **传输分辨率**、**编码**：发回 PC，PC 会重启编码器。设备解不了的编码会自动换另一种。
  - **立体强度**：3D 感有多强。默认 0.5%，太高会觉得画面不舒服。
  - **会聚**：0 时物体往外凸出，1 时往屏幕里凹进去。
  - **屏幕距离 / 宽度 / 曲率 / 高度**：调整虚拟屏幕。**重新居中**把屏幕挪到你面前。
  - **背景透明度**（透视模式）：控制现实房间显示多少，100% 完整显示房间，调到 0% 完全不透明（只看虚拟屏幕）。滑块按 5% 一档，点到滑轨两端外侧即取 0% / 100%。
  - **电脑静音**：串流时让电脑扬声器静音，声音只在头显里放。默认开启；断开连接或退出程序后恢复原样。
- 面板顶部显示分辨率、帧率、码率和延迟。「收 / 显」分别是采集到收到、采集到显示的时间，已经按 PC 与头显的时钟差校正过。

在 PC 浏览器打开 `https://localhost:13256` 并点「预览」，可以并排预览左右眼，用鼠标操作面板。

### 5. 命令行参数

常用参数如下，完整列表见 `immersive-vr.exe --help`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--resolution` | `1440` | 初始传输分辨率（1080 / 1440 / 2160，不超过屏幕分辨率） |
| `--codec` | `hevc` | 初始编码：`hevc` 或 `av1` |
| `--bitrate` | 按分辨率 | 固定码率（Mbit/s）。默认 1080p 40、1440p 60、2160p 100 |
| `--fps` | `60` | 输出帧率 |
| `--preset` | `1` | NVENC 预设：1 最快、延迟最低，7 同码率画质最好 |
| `--monitor` | `0` | 采集哪个显示器：0 是主显示器，其他按 Windows 编号（从 1 开始） |
| `--listen` | `0.0.0.0:13256` | 监听地址 |
| `--mute-pc` | `true` | 串流时默认是否让电脑静音（面板里也能切换） |
| `--divergence` / `--convergence` | `0.5` / `0.5` | 初始立体强度与会聚 |
| `--providers` | `trt,cuda,dml,cpu` | 深度和立体模型的推理后端，按顺序尝试 |
| `--http` | 关 | 用 HTTP 代替 HTTPS（WebXR 这时只能通过 `http://localhost` 使用，比如配合 `adb reverse`） |

### 6. 常见问题

- **头显打不开网页**：确认在同一局域网，并且防火墙允许了专用网络；地址里是 `https`。
- **提示无法解码**：换另一种编码（面板「编码」）。Quest 3 支持 H.265 和 AV1 硬件解码。
- **延迟或卡顿**：先降低传输分辨率或码率。PC 有线连接路由器，头显尽量靠近路由器，用 5 GHz 或 6 GHz 频段。
- **3D 感太强或太弱**：调「立体强度」；觉得刺眼就调小一点。

## 工作原理

```
WGC 采集（D3D11 → CUDA 显存，最高 4 倍输出帧率）
  └─ 按固定 60 Hz 节拍取最新一帧
       ├─ GPU 缩放到传输分辨率 ───────────────────────────────────┐
       └─ GPU 缩放到 770×434 → 深度 (TensorRT) → iw3 mlbw_l2 (TensorRT) │
                       每只眼两层采样偏移 + 混合权重（留在显存）     ▼
            CUDA 核函数：按全分辨率画出左右眼，上下叠成一帧，输出 NV12
                                                                    ▼
          NVENC（H.265 / AV1，直接读显存）→ WebSocket（视频 + PCM 声音）
                                                                    ▼
          头显浏览器：WebCodecs 硬解 → WebXR stereo 圆柱合成层（每只眼一层）
```

## 性能

### 测试环境

- **硬件**：RTX 4090，主显示器 4K 240 Hz，驱动 616.56。
- **运行库**：ONNX Runtime 1.30（CUDA 12）+ TensorRT 10.16；NVENC 预设 P1，H.265。
- **测试内容**：`--synthetic` 合成测试图（4K、每帧都在横向滚动，相当于一直在动的游戏画面），比普通桌面更吃力。
- **延迟测法**：在同一台 PC 上测，客户端是无头 Chrome。无线网络和头显解码的时间不在其中。

### 各阶段耗时（每帧都是新画面，60 fps 输出）

| | 1440p 每眼（帧 2560×2880） | 2160p 每眼（帧 3840×4320） |
|---|---|---|
| 每秒新画面 | 60 | 60 |
| GPU 占用 | 约 48% | 约 48% |
| 采集缩放到眼尺寸（GPU） | 0.14 ms | 0（采集即眼尺寸，直接共用） |
| 深度模型（TensorRT，含归一化） | 约 3.4 ms | 约 3.6 ms |
| mlbw_l2 立体字段（TensorRT） | 约 2.3 ms | 约 3.4 ms |
| 左右眼渲染（CUDA 核函数） | 0.12 ms | 0.20 ms |
| NVENC 编码（P1） | 2.9 ms | 5.3 ms |
| **采集 → 编码完成（中位数）** | **约 8 ms** | **约 10 ms** |
| 采集 → 浏览器收到 / 显示 | 9–11 ms / 12–15 ms | — |

码率：1440p 约 48 Mbps（目标 60），2160p 约 82–100 Mbps。H.265 和 AV1 都能在浏览器里以 60 fps 解码。

### 帧节奏（输出 60 fps，画面源刷新率不同）

| 画面源 | 实际输出 | 到达间隔 p10 / 中位 / p90 | 采集 → 编码完成 |
|---|---|---|---|
| 60 Hz | 60.0 fps | 16.2 / 16.7 / 17.1 ms | 4.5 ms |
| 144 Hz | 60.0 fps | 15.9 / 16.7 / 17.4 ms | 7.7 ms |
| 240 Hz（1440p） | 60.0 fps | 16.2 / 16.6 / 17.2 ms | 7.8 ms |
| 240 Hz（2160p） | 60.0 fps | 16.1 / 16.7 / 17.2 ms | 10.2 ms |

### 编码帧大小（2160p，11 秒）

每帧平均 167 KB，最大 183 KB，整段只有连接时请求的 1 个关键帧。每帧都不会超过一帧时间能传完的大小，所以不会出现关键帧突发导致的周期性卡顿。

### 优化前后

| 指标 | 初版 | 现在 |
|---|---|---|
| 1440p 每秒新画面 | 约 52（可见掉帧） | 60 |
| 2160p 每秒新画面 | 约 29 | 60 |
| 1440p GPU 占用 | 60–88% | 约 48%（深度和立体也已改为每帧计算） |
| 编码器入口 → 客户端收到 | 32 ms（ffmpeg 管道） | 约 4 ms（进程内 NVENC） |
| 2160p 左右眼渲染 | 17.8 ms（ONNX 图） | 0.2 ms（CUDA 核函数） |
| 深度 / 立体模型 | 4.1 / 5.0 ms（CUDA） | 1.9 / 2.3 ms（TensorRT，单独测） |
| 1440p 编码 | 5.4 ms（P4） | 2.9 ms（P1 + 两遍） |
| 关键帧 | 每秒一个，单帧最大约平均的 15 倍 | 按需发，单帧不超过一帧时间 |

## 从源码构建

### 依赖

- Windows 10/11 x64，NVIDIA 显卡和驱动
- [Rust](https://rustup.rs/)：`rust-toolchain.toml` 固定为 1.95.0，rustup 会自动安装
- ONNX Runtime 1.30 GPU 版：
  - `python scripts/fetch_onnxruntime.py` 会下载 CUDA 13 版到 `runtime/ort/`，需要系统里有 CUDA 13 和 cuDNN 9 运行库。
  - 或者使用 [官方 Release](https://github.com/microsoft/onnxruntime/releases) 的 `onnxruntime-win-x64-gpu_cuda12-1.30.0.zip`（CUDA 12 版，搭配 TensorRT 10 可用 TensorRT）。
- CUDA Toolkit 12.x：只有修改 `kernels.cu` 后重新生成 `kernels.ptx` 时才需要，仓库里已经带了编译好的 PTX。

### 编译

```bash
cargo build --release -p immersive-vr
```

产物在 `target/release/immersive-vr.exe`。在仓库根目录运行时，程序默认从 `models/` 读模型，从 `runtime/ort/` 读 ONNX Runtime。

运行库不在默认位置时，可以在 `runtime/runtime.txt`（不入库）里写本机路径。按顺序加到 DLL 搜索路径前面，命令行的 `--ort` / `--lib-dir` 优先：

```
# ONNX Runtime 库
ort=D:/libs/onnxruntime-win-x64-gpu_cuda12-1.30.0/lib/onnxruntime.dll
# CUDA 12 / cuDNN 9 / TensorRT 10 的 DLL 目录
lib=D:/libs/cuda12/bin
lib=D:/libs/TensorRT-10/lib
```

### 模型

可以直接用模型 Release 包，也可以自己导出：

```bash
# 1. 深度：depth-anything/Depth-Anything-V2-Small-hf 放到 models/source/
python scripts/export_depth.py --source models/source/Depth-Anything-V2-Small-hf --out models/depth --size 770x434 --image <照片>
# 2. 立体：需要 nunif 的 Python 环境（git clone https://github.com/nagadomi/nunif，按其说明安装）
<nunif>/.venv/Scripts/python.exe scripts/export_iw3_stereo.py --nunif <nunif> --size 770x434 --image <照片>
```

### 测试

```bash
cargo test --release --workspace
# 需要模型和 GPU 的测试（没有设置环境变量时会跳过）
IVR_DEPTH_MODEL_DIR=models/depth IVR_STEREO_MODEL_DIR=models/stereo ORT_DYLIB_PATH=runtime/ort/onnxruntime.dll \
  cargo test --release --workspace
```

开发工具：
- `--synthetic` 用一张移动的测试图代替屏幕，用来测吞吐。
- `--synthetic-fps 240` 模拟高刷新率显示器。
- `--never-mute-pc` 保证测试时不动电脑的扬声器。
- `scripts/ws_probe.mjs` 统计流的帧率、码率和到达间隔。
- `scripts/page_check.mjs` 用无头 Chrome 跑网页端检查。

### 目录

| 路径 | 内容 |
|---|---|
| `crates/immersive-vr` | 主程序：采集、流水线、CUDA 核函数（`kernels.cu`）、NVENC、声音、HTTPS / WebSocket 服务；`web/` 是网页端，编译时嵌入程序 |
| `crates/depth-infer` | ONNX Runtime 推理：深度、mlbw 立体字段、深度归一化与平滑 |
| `scripts/` | 模型导出、ONNX Runtime 下载、测试工具 |
| `docs/` | 研究笔记 |

## 致谢

- [nunif / iw3](https://github.com/nagadomi/nunif)（nagadomi，MIT，模型权重同为 MIT）：mlbw_l2 立体生成网络和 backward warp 算法。
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)：深度估计，使用 Small 模型（Apache-2.0）。
- [Sunshine](https://github.com/LizardByte/Sunshine)（LizardByte）：低延迟编码参数和帧节奏的思路。
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) 与 [ort](https://github.com/pykeio/ort)：模型推理。
- [moq-nvenc](https://crates.io/crates/moq-nvenc)（源自 [nvidia-video-codec-sdk](https://github.com/ViliamVadocz/nvidia-video-codec-sdk)）与 [cudarc](https://github.com/coreylowman/cudarc)：NVENC 与 CUDA 绑定。
- [windows-capture](https://github.com/NiiightmareXD/windows-capture)、[wasapi-rs](https://github.com/HEnquist/wasapi-rs)、[axum](https://github.com/tokio-rs/axum) 等 Rust 生态项目。

## 许可证

代码以 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE) 双许可证发布，可任选其一。模型 Release 包里的模型遵循各自的许可证：Depth-Anything-V2-Small 为 Apache-2.0，iw3 权重为 MIT。
