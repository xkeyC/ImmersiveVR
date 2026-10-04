# ImmersiveVR

中文 | [English](README.en.md)

ImmersiveVR 是一个**高性能**的实时 2D 转 3D 方案：把 Windows 桌面实时转成立体 3D，串流到 Meta Quest 等 VR 头显，在浏览器里观看，头显上不用装任何应用。从采集、深度估计、立体生成到编码全程在显卡上完成，在 1440p 和 4K 下都能稳定输出每秒 60 帧，而且每一帧都是重新计算的 3D 画面，从采集到编码完成只需约 8–10 ms（RTX 4090 实测，见[性能](#性能)）。

PC 采集屏幕后估计每个像素的深度，用 [iw3](https://github.com/nagadomi/nunif) 的 mlbw_l2 网络在 PC 上渲染出左右两只眼的画面，再用 NVENC 直接从显存编码，通过局域网发给头显。头显浏览器用 WebCodecs 硬件解码，再通过 WebXR 合成层显示成一块可调的虚拟曲面屏。

另有 [PCVR 客户端](#pcvr-客户端)（Unity + OpenXR）：同一套转换在 PC 上直接画进 VR 场景，不经过视频编码，可配合 SteamVR、Pimax、Virtual Desktop、Quest Link 等 PC 端 VR 运行时使用。

## 特性

- **实时 2D 转 3D**：使用 Depth-Anything-V2-Small 估计深度，再用 iw3 的 mlbw_l2 生成立体画面，每一帧都会重新计算。转换后的物体轮廓清晰，不会出现重影。
- **高性能，全程在显卡上**：采集（D3D11 → CUDA）、缩放、深度和立体模型（TensorRT）、左右眼渲染（CUDA 核函数）、编码（NVENC）都在显存里完成。在 1440p 下，每秒 60 帧都是新生成的画面。
- **低延迟**：编码参数参考了 [Sunshine](https://github.com/LizardByte/Sunshine)，每帧的数据量都控制在一帧时间内能传完；关键帧只在需要时才发送，不会周期性地出现流量突增。客户端来不及处理时，服务端会主动丢弃积压的帧，尽快追上最新画面。本机实测从采集到显示约 12–15 ms。
- **画面平稳**：即使显示器刷新率很高（例如 4K 240 Hz），程序也会按固定的 60 Hz 节奏取最新的一帧。头显支持时会自动把刷新率设为 120 Hz，让每一帧的显示时长保持一致。
- **声音**：电脑的声音会同步传到头显，几乎不增加延迟。串流时还可以让电脑扬声器静音（默认开启）。
- **在 VR 里调整**：可以调整传输分辨率（1080p / 1440p / 2160p）、编码（H.265 / AV1）、立体强度、会聚、屏幕距离、宽度、曲率和高度，透视模式下还能调背景透明度。所有设置都会自动保存。
- **透视**：设备支持时，可以在看得见真实房间的环境里观看虚拟屏幕。
- **可安装为应用**：网页支持 PWA，可以安装到头显的应用列表里。

## 系统要求

| | 要求 |
|---|---|
| 系统 | Windows 10 2004 及以上 / Windows 11（64 位） |
| 显卡 | NVIDIA RTX 20 系列或更新（AV1 编码需要 RTX 40 系列），驱动 550 或更新 |
| 头显 | Meta Quest 3 / 3S（Meta Quest 浏览器）；其他支持 WebXR 和 WebCodecs（能硬件解码 H.265 或 AV1）的头显浏览器，理论上也可以使用 |
| 网络 | PC 与头显在同一局域网。推荐 PC 用有线连接，头显用 Wi-Fi 6/6E |

## 使用说明

### 1. 下载

程序、模型、运行库分成三个 Release 发布，按需下载后**解压到同一个文件夹**：

| Release | 文件 | 内容 | 需要吗 |
|---|---|---|---|
| [程序 v0.0.1](../../releases/tag/v0.0.1) | `ImmersiveVR-v0.0.1-windows-x64.zip` | `immersive-vr.exe`（网页串流）和 `ImmersiveVR.exe`（PCVR 客户端） | 必需 |
| [模型 models-v0.0.1](../../releases/tag/models-v0.0.1) | `ImmersiveVR-models-v0.0.1.zip` | `models/depth/`（Depth-Anything-V2-Small，770×434）、`models/stereo/`（iw3 mlbw_l2 三个强度档） | 必需 |
| [运行库 runtime-v0.0.1](../../releases/tag/runtime-v0.0.1) | `ImmersiveVR-runtime-v0.0.1-windows-x64.zip` | `runtime/ort/`：ONNX Runtime 1.30（CUDA 12 版）、CUDA 12、cuDNN 9 | 必需 |
| | `ImmersiveVR-runtime-tensorrt-v0.0.1-windows-x64.zip` | `runtime/ort/`：TensorRT 10（RTX 20 / 30 / 40 / 50 系列） | 推荐 |

TensorRT 包是可选的。不安装时程序会自动改用 CUDA，同样能稳定在 60 帧；安装后深度和立体模型的速度约提高 2 倍，显卡负载也更低。首次使用 TensorRT 时需要为你的显卡构建引擎，大约要几分钟，之后会直接读取 `models/` 下的缓存。

模型和运行库很少变化，以后升级时通常只需要更换程序包。

解压后的目录：

```
ImmersiveVR/
├─ immersive-vr.exe       网页串流模式
├─ ImmersiveVR.exe        PCVR 模式（连同 ImmersiveVR_Data/ 等文件）
├─ runtime/ort/…          ONNX Runtime、CUDA、cuDNN（和可选的 TensorRT）
└─ models/
   ├─ depth/…
   └─ stereo/…
```

两种使用模式，任选其一：

| 模式 | 启动 | 头显 | 适合 |
|---|---|---|---|
| **网页串流** | `immersive-vr.exe` | 在 Quest 浏览器里打开网页，无需安装应用 | 一体机无线使用，见下文第 2–6 节 |
| **PCVR** | `ImmersiveVR.exe` | 通过 PC 端 VR 运行时（SteamVR、Pimax Play、Virtual Desktop、Quest Link 等） | PCVR 头显，或已连接电脑的 Quest；画面不经过视频编码，见 [PCVR 客户端](#pcvr-客户端) |

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
  - **传输分辨率**、**编码**：修改后会发送给 PC，PC 随即重启编码器。如果设备无法解码所选的编码，会自动换成另一种。
  - **立体强度**：控制 3D 效果的强弱，默认 0.5%。调得太高时，画面看起来会不舒服。
  - **会聚**：设为 0 时物体向屏幕外凸出，设为 1 时向屏幕里凹进去。
  - **屏幕距离 / 宽度 / 曲率 / 高度**：调整虚拟屏幕的位置和形状。点**重新居中**可以把屏幕移回你面前。
  - **背景透明度**（透视模式）：控制现实房间的可见程度。100% 时完整显示房间，0% 时完全不透明，只能看到虚拟屏幕。滑块以 5% 为一档，点击滑轨两端外侧可以直接设为 0% 或 100%。
  - **电脑静音**：串流时让电脑扬声器静音，声音只在头显里播放。默认开启，断开连接或退出程序后会恢复原来的状态。
- 面板顶部会显示分辨率、帧率、码率和延迟。其中「收 / 显」分别表示从采集到头显收到、从采集到画面显示所用的时间，已经按 PC 与头显之间的时钟差做了校正。

在 PC 浏览器里打开 `https://localhost:13256` 并点击「预览」，可以并排查看左右眼画面，并用鼠标操作面板。

### 5. 命令行参数

常用参数如下，完整列表见 `immersive-vr.exe --help`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--resolution` | `1440` | 初始传输分辨率（1080 / 1440 / 2160，不超过屏幕分辨率） |
| `--codec` | `hevc` | 初始编码：`hevc` 或 `av1` |
| `--bitrate` | 按分辨率 | 固定码率（Mbit/s）。默认 1080p 40、1440p 60、2160p 100 |
| `--fps` | `60` | 输出帧率 |
| `--preset` | `1` | NVENC 预设：1 速度最快、延迟最低，7 在相同码率下画质最好 |
| `--monitor` | `0` | 采集哪个显示器：0 是主显示器，其他按 Windows 编号（从 1 开始） |
| `--listen` | `0.0.0.0:13256` | 监听地址 |
| `--mute-pc` | `true` | 串流时默认是否让电脑静音（面板里也能切换） |
| `--divergence` / `--convergence` | `0.5` / `0.5` | 初始立体强度与会聚 |
| `--providers` | `trt,cuda,dml,cpu` | 深度和立体模型的推理后端，按顺序尝试 |
| `--http` | 关 | 用 HTTP 代替 HTTPS（WebXR 这时只能通过 `http://localhost` 使用，比如配合 `adb reverse`） |

### 6. 常见问题

- **头显打不开网页**：确认 PC 和头显在同一局域网，防火墙已允许专用网络，并且地址是以 `https` 开头的。
- **提示无法解码**：在面板的「编码」里换成另一种编码。Quest 3 支持 H.265 和 AV1 硬件解码。
- **延迟高或卡顿**：先降低传输分辨率或码率。PC 最好用网线连接路由器，头显尽量靠近路由器，并使用 5 GHz 或 6 GHz 频段。
- **3D 效果太强或太弱**：调整「立体强度」，觉得刺眼就调小一些。

## PCVR 客户端

`ImmersiveVR.exe` 通过 PC 端的 VR 运行时（OpenXR）显示 3D 桌面。转换方式和网页版相同，但左右眼画面直接从显存交给 VR 场景，既不需要视频编码，也不需要浏览器。适合 Pimax、Index 等 PCVR 头显，或者通过 Virtual Desktop、Steam Link、Quest Link 连接电脑的 Quest。

1. 按[下载](#1-下载)说明准备好程序、模型和运行库（PCVR 客户端 `ImmersiveVR.exe` 就在程序包里）。
2. 启动头显的 PC 端软件，让它成为系统的 OpenXR 运行时（SteamVR、Pimax Play、Virtual Desktop、Meta Quest Link 都可以在各自的设置里设为默认 OpenXR 运行时）。
3. 双击 `ImmersiveVR.exe`。电脑上只会打开一个小窗口，切换到其他窗口后程序也会继续运行。虚拟屏幕会出现在你正前方，每次重新戴上头显时都会自动摆回面前。
4. 面板的用法和网页版一样：按握持键打开或关闭，用扳机点击。可以调整画面分辨率（默认使用显示器的原生分辨率，4K 显示器即 2160p）、立体强度、会聚、屏幕距离 / 宽度 / 曲率 / 高度，以及**锐度**：虚拟屏幕在视野里比原始画面小时，锐度越高越清晰，但调得过高会出现闪烁。所有设置都会自动保存。

和网页版相比，这里没有编码选项、透视和电脑静音：画面不经过编码，声音也由 VR 运行时（例如 Virtual Desktop）自行处理。不过如果通过 Virtual Desktop 等软件无线串流，整个 VR 画面仍会被它们编码，清晰度会受其码率影响。

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
- **测试内容**：`--synthetic` 合成测试图（4K，每一帧都在横向滚动，相当于一直在动的游戏画面），比普通桌面的负载更高。
- **延迟测法**：在同一台 PC 上测量，客户端是无头 Chrome，因此不包含无线网络传输和头显解码的时间。

### 各阶段耗时（每帧都是新画面，60 fps 输出）

| | 1440p 每眼（帧 2560×2880） | 2160p 每眼（帧 3840×4320） |
|---|---|---|
| 每秒新画面 | 60 | 60 |
| GPU 占用 | 约 48% | 约 48% |
| 采集缩放到眼尺寸（GPU） | 0.14 ms | 0（采集尺寸就是单眼尺寸，不需要缩放） |
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

每帧平均 167 KB，最大 183 KB，整段只有连接时请求的 1 个关键帧。每一帧的大小都不会超过一帧时间内能传完的量，所以不会因为关键帧突然变大而周期性卡顿。

### 优化前后

| 指标 | 初版 | 现在 |
|---|---|---|
| 1440p 每秒新画面 | 约 52（可见掉帧） | 60 |
| 2160p 每秒新画面 | 约 29 | 60 |
| 1440p GPU 占用 | 60–88% | 约 48%（同时深度和立体模型也已改为每帧计算） |
| 编码器入口 → 客户端收到 | 32 ms（ffmpeg 管道） | 约 4 ms（进程内 NVENC） |
| 2160p 左右眼渲染 | 17.8 ms（ONNX 图） | 0.2 ms（CUDA 核函数） |
| 深度 / 立体模型 | 4.1 / 5.0 ms（CUDA） | 1.9 / 2.3 ms（TensorRT，单独测） |
| 1440p 编码 | 5.4 ms（P4） | 2.9 ms（P1 + 两遍） |
| 关键帧 | 每秒一个，单帧最大约为平均的 15 倍 | 按需发送，单帧不超过一帧时间内能传完的量 |

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

如果运行库不在默认位置，可以在 `runtime/runtime.txt`（不提交到仓库）里写上本机路径。这些目录会按顺序加到 DLL 搜索路径的最前面；命令行的 `--ort` / `--lib-dir` 优先级更高：

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

### PCVR 客户端

1. 编译原生库：`cargo build --release -p ivr-native`（产物 `target/release/ivr_native.dll`，即整条流水线的 C 接口）。
2. 用 Unity 6.6（6000.6.4f1）打开 `unity/`，场景是 `Assets/ImmersiveVR/Scenes/Desktop3D`。
3. 在编辑器里按 Play 即可在头显里运行。每次 Play 加载的都是 `ivr_native.dll` 的副本，退出 Play 时会卸载，所以重新编译 Rust 代码后不需要重启编辑器。菜单 **ImmersiveVR → Play Mode Runtime** 选择 Play 时用哪个 OpenXR 运行时（重启编辑器后保留）。
4. 菜单 **ImmersiveVR → Build Release** 生成正式版，输出到 `target/unity/ImmersiveVR/`。

`cargo test --release -p ivr-native` 会模拟 Unity 的加载方式（受限的 DLL 搜索路径、反复加载和卸载），完整运行一遍整条流水线。

### 打包发布

```bash
cargo build --release -p immersive-vr -p ivr-native
# Unity：ImmersiveVR → Build Release
python scripts/package_release.py --version 0.0.1
```

在 `target/dist/` 生成[下载](#1-下载)表里的四个 zip。运行库按 `runtime/runtime.txt` 里的目录顺序收集，和程序加载它们的规则一致；用 `--only programs,models,runtime,tensorrt` 可以只打包其中几个。

开发工具：
- `--synthetic` 用一张移动的测试图代替屏幕，用来测试处理能力。
- `--synthetic-fps 240` 模拟高刷新率显示器。
- `--never-mute-pc` 确保测试时不会改变电脑扬声器的静音状态。
- `scripts/ws_probe.mjs` 统计流的帧率、码率和到达间隔。
- `scripts/page_check.mjs` 用无头 Chrome 自动检查网页端。

### 目录

| 路径 | 内容 |
|---|---|
| `crates/immersive-vr` | 主程序：采集、流水线、CUDA 核函数（`kernels.cu`）、NVENC、声音、HTTPS / WebSocket 服务；`web/` 是网页端，编译时嵌入程序 |
| `crates/depth-infer` | ONNX Runtime 推理：深度、mlbw 立体字段、深度归一化与平滑 |
| `crates/ivr-native` | 流水线的原生库（`ivr_native.dll`），供 PCVR 客户端在进程内调用 |
| `unity/` | PCVR 客户端（Unity 6.6 + OpenXR + XR Interaction Toolkit） |
| `scripts/` | 模型导出、ONNX Runtime 下载、测试工具 |
| `docs/` | 研究笔记 |

## 致谢

- [nunif / iw3](https://github.com/nagadomi/nunif)（nagadomi，MIT，模型权重同为 MIT）：mlbw_l2 立体生成网络和 backward warp 算法。
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)：深度估计，使用 Small 模型（Apache-2.0）。
- [Sunshine](https://github.com/LizardByte/Sunshine)（LizardByte）：低延迟编码参数和帧节奏的思路。
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) 与 [ort](https://github.com/pykeio/ort)：模型推理。
- [moq-nvenc](https://crates.io/crates/moq-nvenc)（源自 [nvidia-video-codec-sdk](https://github.com/ViliamVadocz/nvidia-video-codec-sdk)）与 [cudarc](https://github.com/coreylowman/cudarc)：NVENC 与 CUDA 绑定。
- [Unity](https://unity.com/) 的 OpenXR Plugin 与 XR Interaction Toolkit：PCVR 客户端。
- [windows-capture](https://github.com/NiiightmareXD/windows-capture)、[wasapi-rs](https://github.com/HEnquist/wasapi-rs)、[axum](https://github.com/tokio-rs/axum) 等 Rust 生态项目。

## 许可证

代码以 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE) 双许可证发布，可任选其一。模型 Release 包里的模型遵循各自的许可证：Depth-Anything-V2-Small 为 Apache-2.0，iw3 权重为 MIT。
