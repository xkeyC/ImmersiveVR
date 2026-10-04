# PCVR 客户端计划：Unity 负责显示，Rust 原生 DLL 负责转换

背景：原生 Rust 的 OpenXR 客户端以后加 VR 特性不方便（交互、UI、环境、设备适配），所以 PCVR 客户端用 Unity 6.6（OpenXR）。现有 Rust 管线编成原生 DLL，在 Unity 进程内运行。调研见 [openxr-research.md](openxr-research.md)。

## 架构

```
Unity 6.6 PCVR 应用（openxr 分支的 unity/）
├─ C#：IvrNative（加载层，见下）、IvrScreen（显示）、面板（XR Interaction Toolkit）
├─ ivr_native.dll（Rust cdylib，本仓库 crates/ivr-native）
│    采集 → CUDA 缩放 → 深度 + mlbw（TensorRT）→ warp 核函数（RGBA 左右眼）
│    → 渲染线程回调里写进 Unity 的 D3D11 纹理
└─ onnxruntime.dll 等运行库（由 ivr_native 按路径动态加载）
        │ OpenXR
        ▼  VDXR / SteamVR / Meta Link → 头显
```

浏览器版 `immersive-vr.exe`（NVENC + WebSocket）照旧，和 DLL 共用同一份管线库。

## 编辑器里重新加载 DLL

Unity 编辑器加载过的原生插件无法卸载，Unity 6 的文档依然这么写（[Unity 手册](https://docs.unity3d.com/Manual/plug-ins-native-overview.html)、[issue](https://issuetracker.unity3d.com/issues/external-plugins-are-not-unloaded-when-exiting-play-mode-in-editor)）。绕法是不用 `[DllImport]`，自己 `LoadLibrary` / `GetProcAddress` / `FreeLibrary`（[Forrest Smith 的方法](https://www.forrestthewoods.com/blog/how-to-reload-native-plugins-in-unity/)；现成工具 [UnityNativeTool](https://github.com/MCpiroman/UnityNativeTool)）。

我们的做法：

- **自己写加载层**（`IvrNative.cs`）：接口只有十来个函数，用委托包装，不依赖第三方工具。
- **影子副本**：编辑器里每次进入播放模式，先把 `ivr_native.dll` 复制成 `Temp/ivr_native_<时间戳>.dll` 再加载。这样 cargo 随时能覆盖编译产物，改完 Rust 重新进入播放模式就生效，不用重启编辑器。
- **退出播放模式**：调用 `ivr_shutdown()`，停止发渲染回调，等一帧（让渲染线程空闲），再 `FreeLibrary`。
- **发布版本**：同一个加载层直接加载程序目录下的 DLL，不做影子副本。

Rust 侧为了能安全卸载要做到：

- `ivr_shutdown` 停止并回收所有线程（采集、缩放、深度、立体、渲染），释放 CUDA 内存和纹理注册，销毁会话。
- 对外的每个 `extern "C"` 函数都用 `catch_unwind` 包住，panic 不能跨越 FFI。
- ONNX Runtime 的全局环境留在 `onnxruntime.dll` 里不卸载，重新加载 DLL 时沿用（[ONNX Runtime 每进程一个环境](../crates/depth-infer/src/runtime.rs)，重复初始化只打警告）。需要实测确认。

## 纹理交接（进程内）

- **拿到 Unity 的 D3D11 设备**：C# 把任意 Unity 纹理的 `GetNativeTexturePtr()`（`ID3D11Texture2D*`）传给 `ivr_attach`，插件用 `GetDevice()` 取得设备，不需要 `IUnityInterfaces`。Unity 项目固定用 D3D11。CUDA 用的显卡必须与 Unity 设备相同（按 LUID 检查）。
- **插件自己建纹理**（在 Unity 的设备上），每只眼两张：
  - 给 CUDA 写的 `R8G8B8A8_UNORM`，注册到 CUDA；
  - 给 Unity 采样的纹理：TYPELESS 资源，SRV 用 `UNORM_SRGB`，线性色彩空间下颜色正确；
  - 如果 CUDA 能直接注册 `_SRGB` 格式的纹理，就合成一张（实现时验证）。
- **C# 包成外部纹理**：`Texture2D.CreateExternalTexture(w, h, RGBA32, false, false, srv)`（D3D11 下传 SRV 指针）；分辨率变化时 `UpdateExternalTexture`。
- **每帧拷贝**：C# 每帧 `CommandBuffer.IssuePluginEvent(ivr_render_event_func(), 0)`。Unity 在渲染线程回调插件，插件在帧号变化时：
  1. 映射已注册的纹理，把最新的 CUDA 输出 `cuMemcpy2D` 进去，取消映射；
  2. 需要时再 `CopyResource` 到 SRGB 纹理。

  全部在 Unity 的渲染线程上执行，不跨线程碰 Unity 的 D3D11 上下文。
- **管线产出**：warp 核函数新增 RGBA 输出（左右眼各一张，或左右并排一张）。渲染线程只交出「最新一帧」（三缓冲，帧号递增），Unity 端取最新的。

## C 接口（草案）

```c
// 配置：JSON（模型目录、ONNX Runtime 路径、显示器、分辨率、立体参数……），出错时把信息写进 err
int  ivr_init(const char* config_json, char* err, int err_len);
int  ivr_attach(void* unity_texture, char* err, int err_len);   // 取 Unity 的 D3D11 设备，建纹理
void ivr_set(const char* settings_json);                         // 同浏览器版的设置协议
int  ivr_poll(IvrInfo* info);         // 新帧？尺寸、代号（尺寸变了要重建外部纹理）、各阶段耗时
void ivr_textures(void** left_srv, void** right_srv);
void* ivr_render_event_func(void);    // UnityRenderingEvent
void ivr_shutdown(void);
```

## Rust 侧改动

1. **拆出管线库**：`crates/immersive-vr` 拆成库和二进制。库包含采集、gpu、深度和立体、流水线；二进制保留编码器、服务器和 main。
2. **输出抽象**：流水线最后一步改成「输出端」，浏览器端是 NVENC，Unity 端是 RGBA 左右眼。新增 `warp_rgba` 核函数。
3. **可停止**：流水线和采集支持停止，所有线程都能回收（目前只在出错时退出）。
4. **`crates/ivr-native`**（cdylib）：C 接口、Unity 纹理交接、渲染线程回调，所有接口捕获 panic。
5. **测试**：Rust 集成测试在 headless D3D11 设备上模拟 Unity 的调用顺序（初始化、关联设备、轮询、渲染回调、关闭），并连续加载和卸载 DLL 多次，验证能干净卸载。

## Unity 侧

- 位置：`openxr` 分支的 `unity/`（Unity 6.6，URP，Linear 色彩空间，D3D11，OpenXR + XR Interaction Toolkit，之后加 XR Composition Layers）。
- 用 Unity CLI（`C:\Program Files\Unity Hub\resources\unity.exe`）创建项目和批处理构建。
- **显示**：
  - 先做立体网格（曲面屏网格 + 按 `unity_StereoEyeIndex` 选左右纹理的着色器，单通道实例化），所有运行时都能用；
  - 再做合成层（平面层，或 Meta Link 上的曲面层）；
  - 用 `XR_FB_display_refresh_rate` 请求 120 Hz。
- **交互**：手柄射线 + 世界空间面板，选项与浏览器版一致。握持拖动屏幕。立体相关设置走 `ivr_set`。
- PCVR 下声音由运行时（Link / VD / SteamVR）送到头显，不需要我们传。

## 分阶段

1. **Rust：库 + DLL**：拆库、RGBA 输出、可停止、`ivr-native` 的 C 接口；用 headless 测试验证纹理交接和反复加载卸载。
2. **Unity 最小场景**：项目骨架和加载层（影子副本），在 VDXR 和 SteamVR 下看到立体平面屏。
3. **交互**：面板、射线、拖动，设置同步。
4. **画质与适配**：合成层、120 Hz、超采样，在 Link / VD / SteamVR 上实测。
5. **打包**：Unity 构建 + DLL + 运行库 + 模型。

## 风险

- 卸载 DLL 时如果还有线程没退出、还有回调没执行完，编辑器会崩，所以关闭顺序要严格（C# 加载层负责）。
- Unity 设备与 CUDA 必须在同一块显卡上。
- VDXR 回退到 Oculus 运行时时，平面层的按眼可见没有实现，届时只能用网格方案。
