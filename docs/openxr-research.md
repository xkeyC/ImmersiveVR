# OpenXR / PCVR 客户端调研

日期：2026-10。目标：在 PC 上用 OpenXR（或 SteamVR）客户端直接把 3D 转换后的桌面显示在头显里，本程序不再做视频编码。头显由 PC 运行时驱动：Meta Quest Link / Air Link、SteamVR（包括通过 ALVR 或 Virtual Desktop 串流），或 Virtual Desktop 自带的 VDXR。

现有管线不变：WGC 采集 → CUDA → 深度 + iw3 mlbw_l2（TensorRT）→ CUDA 核函数输出左右眼。变的只有最后一步：「NVENC + WebSocket + WebXR」换成「写进 OpenXR 交换链，作为合成层提交」。

## 本机环境

- 当前激活的 OpenXR 运行时是 **VDXR**（`C:\Program Files\Virtual Desktop Streamer\OpenXR\virtualdesktop-openxr.json`）。另外装了 **SteamVR**（`steamxr_win64.json`）和 Pimax 运行时。没有 Meta Quest Link。
- 装有 Unity 2022.3.22f1（LTS）。

## 1. 各运行时对合成层的支持

来源：[Khronos OpenXR-Inventory](https://github.com/KhronosGroup/OpenXR-Inventory/tree/main/runtimes)，以及 VDXR 源码。

| 运行时 | 平面（quad） | 曲面（cylinder） | 全景（equirect） | 备注 |
|---|---|---|---|---|
| Meta PC（Link / Air Link） | 支持 | **支持** | v1 支持 | 支持 `XR_FB_display_refresh_rate`；超采样 / 锐化的 `XR_FB_composition_layer_settings` 只在一体机上有 |
| SteamVR 2.14 | 支持 | **不支持** | 不支持 | 只有投影层和平面层（Unreal 的文档也这么写）；支持 `XR_FB_display_refresh_rate`。ALVR 和 Virtual Desktop 的 SteamVR 模式都走这个运行时 |
| VDXR（Virtual Desktop） | 支持 | 公开版本没有编译进去 | 不支持 | [源码](https://github.com/mbucchia/VirtualDesktop-OpenXR)里 cylinder 在 `HAS_CYLINDER_LAYERS` 宏后面，公开构建没开。平面层的按眼可见（`eyeVisibility`）在 VD 模式下已实现 |
| WMR | 支持 | 不支持 | 不支持 | Windows 11 24H2 起已被移除，可以不考虑 |

**立体内容的做法**：一个合成层只有一张图。按眼显示时，在同一位置提交两个层，`eyeVisibility` 分别设为 LEFT 和 RIGHT，各自取同一张左右并排交换链图像的一半（核心规范功能）。

**曲面屏**：只有 Meta Link 能用曲面合成层。SteamVR 和 VDXR 只能用平面层；想要曲面，就得自己往投影层里画一个弯曲的网格。

**合成层和自绘投影层的区别**：
- 合成层由合成器直接采样一次，画面最清晰；即使程序掉帧，合成器也会按显示刷新率平滑重投影。
- 投影层要经过两次采样（先由程序渲染，再经过畸变校正），文字会更软，大约要 1.4 倍超采样才能补回来；掉帧时还会出现 ASW 或运动平滑带来的伪影。
- 不过 Link、Air Link、VD、ALVR 最终都要再把整个画面编码串流到头显，所以合成层带来的清晰度提升，上限受串流码率限制。

## 2. 帧节奏

- 释放过的交换链图像一直有效。每次 `xrEndFrame` 都提交层，但只在有新的立体画面时才获取、写入、释放交换链图像；没有新画面就重复提交同一个层。VDXR 和 OpenKneeboard 都是这么做的（[参考](https://community.khronos.org/t/display-overlay-without-submitting-frame-possible/108397)）。
- `xrWaitFrame` 放在单独的线程，与采集解耦；采集仍按 60 Hz 网格取最新一帧。
- 用 `XR_FB_display_refresh_rate` 请求 120 Hz（Meta PC 和 SteamVR 都支持），这样 60 fps 的内容不会在 90 Hz 下抖动。

## 3. 路线对比

### A. 原生 Rust：`openxr` crate + D3D11（长期推荐）

- [openxrs](https://github.com/Ralith/openxrs) 0.22.0（2026-09）仍在活跃维护。支持 D3D11/D3D12/Vulkan，有 quad、cylinder、equirect、composition layer settings 等层的构建器。
- **不要把交换链图像直接注册给 CUDA**：规范规定 D3D 交换链图像是 TYPELESS 格式（例如 `R8G8B8A8_TYPELESS`），而 `cuGraphicsD3D11RegisterResource` 不支持这种格式。而且这些图像归运行时所有，经常与合成器进程共享，运行时也不知道 CUDA 流的存在。
- **正确做法**：CUDA 照旧写我们自己的 `R8G8B8A8_UNORM` 纹理，然后用 `CopySubresourceRegion` 拷进获取到的交换链图像（同一格式族，合法）。两张 2–4K 的图在 4090 上远不到 0.1 ms。
- **sRGB**：桌面像素是 gamma 编码的，交换链要建成 `R8G8B8A8_UNORM_SRGB`，否则颜色会偏。
- **UI**：用 egui 配 [egui-directx11](https://crates.io/crates/egui-directx11) 渲染到独立的平面层；手柄射线与平面求交，转成 egui 的指针事件。
- 好处：完全复用现有的 Rust/CUDA 代码，没有引擎依赖，所有运行时都能用，延迟最低。

### B. StereoKit / stereokit-rust

- 能包装外部纹理，也能提交自定义合成层，自带 UI、手部和手柄交互。
- 但 stereokit-rust 0.4（alpha）跟随的 StereoKit 0.4 已改用 Vulkan，D3D11 后端已去掉，那样要做 CUDA↔Vulkan 互操作。等于为了一套 UI 多引入一个引擎，外加换图形 API。

### C. Unity + OpenXR + 原生插件

- Rust/CUDA 管线编成 DLL，用 `Texture2D.CreateExternalTexture`（D3D11 SRV）加原生渲染插件回调把纹理交给 Unity。
- XR Composition Layers 包在 SteamVR 上同样只有平面层。
- UI、手柄交互、3D 环境几乎都是现成的；代价是运行时 100 MB 以上，两种语言，构建更复杂。本机已有 Unity 2022.3。

### D. SteamVR 叠加层（OpenVR `IVROverlay`）

- 支持左右并排的立体叠加层（`VROverlayFlags_SideBySide_Parallel`）和曲率（`SetOverlayCurvature`），`SetOverlayTexture` 直接传 D3D11 纹理，有新画面时才更新。
- 能叠在任何 SteamVR 游戏上和仪表盘里。只能用于 SteamVR，Link 用户要在 Link 上再开 SteamVR。
- 参考：[Desktop+](https://github.com/elvissteinjr/DesktopPlus)（GPL-3，带 SBS/OU 3D 模式）、[wlx-overlay-s](https://github.com/galister/wlx-overlay-s)（Rust）；Rust 绑定是 `openvr` crate。
- OpenXR 的叠加层扩展 `XR_EXTX_overlay` 只有 Monado 支持，Meta PC 和 SteamVR 都不支持。

### 其他

Godot 4.3+（有 OpenXR 合成层，外部纹理需要 GDExtension）和 Bevy + bevy_mod_openxr（wgpu/Vulkan，还在演进），对一个已经输出成品左右眼图像的管线来说，增加的麻烦比省下的多。

### 可参考的项目

- iw3-desktop：除了网页串流，还有一个本地窗口，可以交给 Virtual Desktop 或 Bigscreen 的 SBS 模式显示。不用写代码就能在 PCVR 上用，但会被二次编码，也没什么可调的。
- Virtual Desktop、Bigscreen、Desktop+ 能显示 SBS 内容，但都不做 2D 转 3D。
- OpenKneeboard 和 [HTCC](https://github.com/fredemmott/HTCC)：在 Meta、SteamVR、VDXR 上都能跑的 D3D11 平面层实现，可以参考。

## 结论与计划

| | 最快出原型 | 长期质量和延迟 |
|---|---|---|
| 1 | D：SteamVR 叠加层（约 200 行） | **A：原生 Rust OpenXR + D3D11** |
| 2 | A：两个按眼平面层的最小版本 | C：Unity（只在需要丰富 3D 环境时） |
| 3 | C：Unity | B：StereoKit |

建议**主线走 A**。本机当前运行时 VDXR 和 SteamVR 都只有平面层，A 能直接在本机验证，同时覆盖 Meta Link（可用曲面层）。D 以后可以作为「叠加在游戏上」的附加模式。

### A 的最小架构

- **设备**：调用 `xrGetD3D11GraphicsRequirementsKHR`，在要求的显卡（LUID）上创建 D3D11 设备，必须和 CUDA 用的是同一块 4090；开启 `ID3D11Multithread::SetMultithreadProtected`。
- **管线线程**（沿用现有代码）：采集 → CUDA → 深度和立体 → 左右眼核函数。输出写进自有的 3 组 `R8G8B8A8_UNORM` 纹理（与 D3D11 共享，通过 CUDA 注册）。流同步后，以原子方式发布 `{序号, 帧号}`。
- **XR 线程**：每帧依次执行 `xrWaitFrame` → `xrBeginFrame` → 处理事件 → `xrSyncActions`。帧号变了才获取交换链图像，把左右眼 `CopySubresourceRegion` 到一张左右并排的 `R8G8B8A8_UNORM_SRGB` 图像，再释放。最后 `xrEndFrame` 提交以下层：
  - 可选的投影层（环境或黑底）；
  - 左右两个平面层，位置相同，`eyeVisibility` 分别为 LEFT 和 RIGHT；运行时支持 cylinder 时改用曲面层；
  - UI 平面层（egui，只在界面变化时更新，开启源纹理 alpha 混合）。
- **刷新率**：通过 `XR_FB_display_refresh_rate` 选 120 Hz。
- **输入**：一个动作集（瞄准姿态、扳机、握持、摇杆），绑定 Oculus Touch、Index、simple controller 三种手柄。射线击中平面层驱动 egui；握持拖动屏幕；摇杆调立体强度和会聚。
- **兜底**：层或按眼可见出问题时，往投影层里画曲面网格，约 1.4 倍超采样。

### 分阶段

1. **最小验证**：openxrs + D3D11，在 VDXR 和 SteamVR 上显示两个按眼平面层；用合成测试图验证交换链拷贝、sRGB 和帧节奏。
2. **接入现有管线**：把现有管线拆成库（采集、推理、核函数），输出改为 D3D11 共享纹理；保留现有的浏览器串流模式。
3. **交互**：手柄、egui 面板、屏幕拖动，设置项与浏览器版对齐。
4. **适配与优化**：运行时支持 cylinder 时用曲面层，加刷新率设置、投影层兜底；按需再做 SteamVR 叠加层模式。
