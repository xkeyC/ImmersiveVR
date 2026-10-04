//! ImmersiveVR's 2D-to-3D pipeline as a native plugin, for Unity (or any
//! host with a D3D11 device): the desktop is captured and turned into a left
//! and a right eye image on the GPU, which land in D3D11 textures on the
//! host's own device.
//!
//! Call order (C ABI; all strings UTF-8, NUL-terminated):
//!
//! 1. `ivr_init(config_json)` loads the models and starts capture and the
//!    pipeline (config: see [`Config`]).
//! 2. `ivr_attach(any_host_texture)` gives the host's `ID3D11Device` (taken
//!    from the texture: Unity's `GetNativeTexturePtr()`).
//! 3. Every frame, the host runs `ivr_render_event_func()` on its render
//!    thread (Unity: `CommandBuffer.IssuePluginEvent`): the newest eye pair
//!    is copied into the eye textures, created there (and again on a size
//!    change) as `R8G8B8A8_UNORM_SRGB`.
//! 4. `ivr_poll(&info)` reports the textures' shader resource views (Unity:
//!    `Texture2D.CreateExternalTexture`), their generation (new views after a
//!    size change), the frame number and the stereo settings.
//! 5. `ivr_set(settings_json)` changes settings (as the browser client does:
//!    `{"divergence": 0.5, "convergence": 0.5, "resolution": 1440}`).
//! 6. `ivr_shutdown()` stops and joins every thread and frees GPU resources,
//!    so the host can unload this library (Unity's editor reloads it that
//!    way). Stop issuing render events before calling it.
//!
//! Functions return 0 on success; `ivr_last_error` gives the reason
//! otherwise. Panics never cross the boundary.

use anyhow::{anyhow, bail, Context as _, Result};
use cudarc::driver::{sys, CudaStream, DevicePtr};
use depth_infer::{init_runtime, Provider, RuntimeOptions};
use immersive_vr::{
    capture::{self, Capture, LatestFrame},
    codec::Codec,
    engines,
    gpu::Gpu,
    pipeline::{Controls, EyeOutput, Eyes, Output, Pipeline, Settings, SettingsUpdate},
};
use serde::Deserialize;
use std::{
    ffi::{c_char, c_void, CStr},
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock,
    },
    thread::JoinHandle,
};
use windows::{
    core::Interface,
    Win32::Graphics::{
        Direct3D11::{
            ID3D11Device, ID3D11Resource, ID3D11ShaderResourceView, ID3D11Texture2D,
            D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_RESOURCE_MISC_GENERATE_MIPS, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
        },
        Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_SAMPLE_DESC},
    },
};

/// `ivr_init`'s configuration (JSON); every field is optional.
#[derive(Deserialize)]
#[serde(default)]
pub struct Config {
    /// onnxruntime.dll; default: beside this library, then runtime/ort.
    pub ort: Option<PathBuf>,
    /// Directories with CUDA / cuDNN / TensorRT libraries.
    pub lib_dirs: Vec<PathBuf>,
    pub models: PathBuf,
    pub stereo_models: PathBuf,
    pub depth_size: String,
    pub providers: String,
    /// 0 = the primary monitor.
    pub monitor: usize,
    pub fps: u32,
    /// Eye image height (1080, 1440 or 2160; never above the capture's).
    pub resolution: usize,
    pub divergence: f32,
    pub convergence: f32,
    pub depth_fps: f64,
    /// A moving test pattern instead of the screen.
    pub synthetic: bool,
    pub synthetic_fps: Option<u32>,
    /// Where to log (the first `ivr_init` of a process decides).
    pub log_file: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ort: None,
            lib_dirs: Vec::new(),
            models: "models/depth".into(),
            stereo_models: "models/stereo".into(),
            depth_size: "770x434".into(),
            providers: "trt,cuda,dml,cpu".into(),
            monitor: 0,
            fps: 60,
            resolution: 1440,
            divergence: 0.5,
            convergence: 0.5,
            depth_fps: 0.0,
            synthetic: false,
            synthetic_fps: None,
            log_file: None,
        }
    }
}

/// What `ivr_poll` reports.
#[repr(C)]
pub struct IvrInfo {
    /// The frame number now in the textures (0: none yet).
    pub frame: u64,
    /// Changes whenever the textures (and views) are new.
    pub generation: u64,
    pub width: i32,
    pub height: i32,
    /// `ID3D11ShaderResourceView*` of the left and right eye textures.
    pub left: *mut c_void,
    pub right: *mut c_void,
    /// 1 while the pipeline runs; 0 after it failed (see `ivr_last_error`).
    pub running: i32,
    pub divergence: f32,
    pub convergence: f32,
}

/// The running pipeline.
struct Instance {
    stop: Arc<AtomicBool>,
    pipeline: Option<JoinHandle<Result<()>>>,
    capture: Option<Capture>,
    controls: Arc<Controls>,
    eyes: Arc<EyeOutput>,
    gpu: Arc<Gpu>,
}

/// One eye texture on the host's device, registered with CUDA.
struct EyeTexture {
    _texture: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    resource: sys::CUgraphicsResource,
}

/// The host side: its device and the eye textures on it.
struct Targets {
    device: ID3D11Device,
    eyes: Option<(EyeTexture, EyeTexture)>,
    width: usize,
    height: usize,
    generation: u64,
    frame: u64,
    /// Replaced textures, kept a few frames in case the host still draws with them.
    retired: Vec<((EyeTexture, EyeTexture), u32)>,
    stream: Option<Arc<CudaStream>>,
}

// SAFETY: the D3D11 objects are free-threaded COM interfaces, and the CUDA
// handles are plain values; all access goes through the mutexes below.
unsafe impl Send for Targets {}
unsafe impl Send for EyeTexture {}

static INSTANCE: Mutex<Option<Instance>> = Mutex::new(None);
static TARGETS: Mutex<Option<Targets>> = Mutex::new(None);
static LAST_ERROR: Mutex<String> = Mutex::new(String::new());
static RUNTIME: OnceLock<std::result::Result<(), String>> = OnceLock::new();
static LOGGING: OnceLock<()> = OnceLock::new();

/// Frames a replaced texture is kept before it is released.
const RETIRE_FRAMES: u32 = 8;

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn set_error(error: impl std::fmt::Display) {
    let message = error.to_string();
    tracing::error!("{message}");
    *lock(&LAST_ERROR) = message;
}

/// Runs `body`, turning errors and panics into a status code and the last error.
fn guarded(body: impl FnOnce() -> Result<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_error(format!("{error:#}"));
            1
        }
        Err(panic) => {
            let what = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown".into());
            set_error(format!("panic: {what}"));
            2
        }
    }
}

fn read_str<'a>(text: *const c_char) -> Result<&'a str> {
    if text.is_null() {
        return Ok("");
    }
    // SAFETY: the caller passes a NUL-terminated string valid for the call.
    unsafe { CStr::from_ptr(text) }
        .to_str()
        .context("not UTF-8")
}

fn start_logging(file: Option<&PathBuf>) {
    LOGGING.get_or_init(|| {
        let filter = tracing_subscriber::EnvFilter::new("info,ort=warn");
        let builder = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_ansi(false);
        // Appended: each load of the library (one per Play in the editor)
        // adds its session after the previous ones.
        let opened = file.and_then(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()
        });
        let result = match opened {
            Some(file) => builder.with_writer(Mutex::new(file)).try_init(),
            None => builder.try_init(),
        };
        let _ = result;
    });
}

fn init(config: &Config) -> Result<()> {
    if lock(&INSTANCE).is_some() {
        bail!("already initialized: call ivr_shutdown first");
    }
    start_logging(config.log_file.as_ref());
    RUNTIME
        .get_or_init(|| {
            init_runtime(&RuntimeOptions {
                dylib: config.ort.clone(),
                library_dirs: config.lib_dirs.clone(),
            })
            .map(|_| ())
            .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(|e| anyhow!("ONNX Runtime: {e}"))?;
    let engines = engines::load(&engines::EngineOptions {
        models: config.models.clone(),
        stereo_models: config.stereo_models.clone(),
        depth_size: engines::parse_size(&config.depth_size)?,
        providers: Provider::parse_list(&config.providers)?,
        divergence: config.divergence,
        convergence: config.convergence,
    })?;
    let gpu = engines.gpu.clone();
    let controls = Arc::new(Controls::new(Settings {
        divergence: config.divergence,
        convergence: config.convergence,
        resolution: config.resolution,
        codec: Codec::Hevc,
        mute_pc: false,
    }));
    let latest = Arc::new(LatestFrame::default());
    let capture = if config.synthetic {
        let fps = config.synthetic_fps.unwrap_or(config.fps);
        capture::start_synthetic((3840, 2160), fps, latest.clone(), gpu.clone())?
    } else {
        capture::start(config.monitor, config.fps, latest.clone(), gpu.clone())?
    };
    let eyes = Arc::new(EyeOutput::default());
    let stop = Arc::new(AtomicBool::new(false));
    let pipeline = Pipeline {
        capture_size: capture.size,
        output: Output::Eyes(eyes.clone()),
        fps: config.fps,
        stop: stop.clone(),
        engine: engines.depth,
        stereo: engines.stereo,
        gpu: engines.gpu,
        delta_scale: engines.delta_scale,
        controls: controls.clone(),
        latest,
        depth_fps: config.depth_fps,
    };
    let pipeline = std::thread::Builder::new()
        .name("ivr-pipeline".into())
        .spawn(move || {
            let result = pipeline.run();
            if let Err(error) = &result {
                set_error(format!("pipeline: {error:#}"));
            }
            result
        })?;
    *lock(&INSTANCE) = Some(Instance {
        stop,
        pipeline: Some(pipeline),
        capture: Some(capture),
        controls,
        eyes,
        gpu,
    });
    tracing::info!("ivr_native initialized");
    Ok(())
}

/// Loads the models and starts capture and the pipeline.
///
/// # Safety
/// `config_json`: NUL-terminated or null.
#[no_mangle]
pub unsafe extern "C" fn ivr_init(config_json: *const c_char) -> i32 {
    guarded(|| {
        let text = read_str(config_json)?;
        let config: Config = if text.trim().is_empty() {
            Config::default()
        } else {
            serde_json::from_str(text).context("config")?
        };
        init(&config)
    })
}

/// Takes the host's D3D11 device from one of its textures.
///
/// # Safety
/// `texture`: a live `ID3D11Texture2D*` (or any `ID3D11Resource*`).
#[no_mangle]
pub unsafe extern "C" fn ivr_attach(texture: *mut c_void) -> i32 {
    guarded(|| {
        if texture.is_null() {
            bail!("no texture");
        }
        // SAFETY: a live COM interface pointer, borrowed for the call.
        let resource = unsafe { ID3D11Resource::from_raw_borrowed(&texture) }
            .context("not a D3D11 resource")?;
        let device = unsafe { resource.GetDevice() }.context("its device")?;
        let mut targets = lock(&TARGETS);
        if let Some(old) = targets.take() {
            release(old);
        }
        *targets = Some(Targets {
            device,
            eyes: None,
            width: 0,
            height: 0,
            generation: 0,
            frame: 0,
            retired: Vec::new(),
            stream: None,
        });
        Ok(())
    })
}

/// Changes settings (JSON, as the browser client sends them).
///
/// # Safety
/// `settings_json`: NUL-terminated or null.
#[no_mangle]
pub unsafe extern "C" fn ivr_set(settings_json: *const c_char) -> i32 {
    guarded(|| {
        let update: SettingsUpdate =
            serde_json::from_str(read_str(settings_json)?).context("settings")?;
        let instance = lock(&INSTANCE);
        instance
            .as_ref()
            .context("not initialized")?
            .controls
            .apply(update);
        Ok(())
    })
}

/// Fills `info`.
///
/// # Safety
/// `info`: a valid pointer to an `IvrInfo`.
#[no_mangle]
pub unsafe extern "C" fn ivr_poll(info: *mut IvrInfo) -> i32 {
    guarded(|| {
        if info.is_null() {
            bail!("no info");
        }
        let instance = lock(&INSTANCE);
        let instance = instance.as_ref().context("not initialized")?;
        let (settings, _) = instance.controls.get();
        let running = instance
            .pipeline
            .as_ref()
            .is_some_and(|thread| !thread.is_finished());
        let targets = lock(&TARGETS);
        let (frame, generation, width, height, left, right) = match targets.as_ref() {
            Some(t) => match &t.eyes {
                Some((l, r)) => (
                    t.frame,
                    t.generation,
                    t.width,
                    t.height,
                    l.view.as_raw(),
                    r.view.as_raw(),
                ),
                None => (
                    0,
                    t.generation,
                    0,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
            },
            None => (0, 0, 0, 0, std::ptr::null_mut(), std::ptr::null_mut()),
        };
        // SAFETY: checked non-null; the caller owns it.
        unsafe {
            *info = IvrInfo {
                frame,
                generation,
                width: width as i32,
                height: height as i32,
                left,
                right,
                running: running as i32,
                divergence: settings.divergence,
                convergence: settings.convergence,
            };
        }
        Ok(())
    })
}

/// Copies the last error into `buffer` (truncated, NUL-terminated); returns its full length.
///
/// # Safety
/// `buffer`: `length` writable bytes, or null.
#[no_mangle]
pub unsafe extern "C" fn ivr_last_error(buffer: *mut c_char, length: i32) -> i32 {
    let message = lock(&LAST_ERROR).clone();
    if !buffer.is_null() && length > 0 {
        let bytes = message.as_bytes();
        let n = bytes.len().min(length as usize - 1);
        // SAFETY: the caller gives `length` writable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), n);
            *buffer.add(n) = 0;
        }
    }
    message.len() as i32
}

/// The render-thread callback (Unity's `UnityRenderingEvent`).
#[no_mangle]
pub extern "C" fn ivr_render_event_func() -> extern "system" fn(i32) {
    render_event
}

extern "system" fn render_event(_event: i32) {
    let status = guarded(render);
    let _ = status;
}

/// On the host's render thread: (re)creates the eye textures for the newest
/// pair's size and copies a new pair into them.
fn render() -> Result<()> {
    let instance = lock(&INSTANCE);
    let Some(instance) = instance.as_ref() else {
        return Ok(());
    };
    let mut guard = lock(&TARGETS);
    let Some(targets) = guard.as_mut() else {
        return Ok(());
    };
    targets.retired.retain_mut(|(_, frames)| {
        *frames = frames.saturating_sub(1);
        *frames > 0
    });
    let Some((frame, eyes)) = instance.eyes.latest() else {
        return Ok(());
    };
    let context = instance.gpu.context();
    context
        .bind_to_thread()
        .map_err(|e| anyhow!("CUDA context: {e:?}"))?;
    let (width, height) = (eyes.left.width(), eyes.left.height());
    if targets.eyes.is_none() || (targets.width, targets.height) != (width, height) {
        let pair = (
            eye_texture(&targets.device, width, height)?,
            eye_texture(&targets.device, width, height)?,
        );
        if let Some(old) = targets.eyes.replace(pair) {
            targets.retired.push((old, RETIRE_FRAMES));
        }
        (targets.width, targets.height) = (width, height);
        targets.generation += 1;
        targets.frame = 0;
        tracing::info!(
            width,
            height,
            generation = targets.generation,
            "eye textures"
        );
    }
    if frame == targets.frame {
        return Ok(());
    }
    if targets.stream.is_none() {
        targets.stream = Some(
            context
                .new_stream()
                .map_err(|e| anyhow!("CUDA stream: {e:?}"))?,
        );
    }
    let stream = targets.stream.clone().expect("created above");
    let (left, right) = targets.eyes.as_ref().expect("created above");
    copy_eyes(&stream, &eyes, left, right)?;
    // The smaller levels from the new picture: the screen usually shows it
    // smaller than it is, and mipmaps keep that sharp instead of aliased.
    // SAFETY: the host's own immediate context, on its render thread.
    unsafe {
        let context = targets.device.GetImmediateContext()?;
        context.GenerateMips(&left.view);
        context.GenerateMips(&right.view);
    }
    targets.frame = frame;
    Ok(())
}

/// A `width x height` sRGB eye texture on `device` with a full mip chain
/// (level 0 written by CUDA, the rest generated), its view, registered with CUDA.
fn eye_texture(device: &ID3D11Device, width: usize, height: usize) -> Result<EyeTexture> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width as u32,
        Height: height as u32,
        // 0: every level down to 1x1.
        MipLevels: 0,
        ArraySize: 1,
        // The pixels are gamma-encoded, as the desktop's: sampled as sRGB.
        Format: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        // GenerateMips renders the levels: it needs a render target.
        BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
        CPUAccessFlags: 0,
        MiscFlags: D3D11_RESOURCE_MISC_GENERATE_MIPS.0 as u32,
    };
    let mut texture = None;
    // SAFETY: plain D3D11 calls on a live device.
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }.context("eye texture")?;
    let texture = texture.context("eye texture")?;
    let mut view = None;
    unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut view)) }
        .context("eye texture view")?;
    let view = view.context("eye texture view")?;
    let register = capture::register_d3d11()?;
    let mut resource = std::ptr::null_mut();
    // SAFETY: registers a live texture; unregistered in `release`.
    unsafe {
        check(
            register(&mut resource, texture.as_raw(), 0),
            "registering an eye texture with CUDA",
        )?;
        check(
            sys::cuGraphicsResourceSetMapFlags_v2(
                resource,
                sys::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD
                    as u32,
            ),
            "eye texture map flags",
        )?;
    }
    Ok(EyeTexture {
        _texture: texture,
        view,
        resource,
    })
}

fn check(result: sys::CUresult, what: &str) -> Result<()> {
    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        bail!("{what}: {result:?}")
    }
}

/// Copies both eye images into their textures; done when this returns.
fn copy_eyes(
    stream: &Arc<CudaStream>,
    eyes: &Eyes,
    left: &EyeTexture,
    right: &EyeTexture,
) -> Result<()> {
    let cu_stream = stream.cu_stream();
    let mut resources = [left.resource, right.resource];
    // SAFETY: both resources are registered, mapped only here, and the
    // copies stay within the arrays and the eye images (same size); the
    // stream is synchronized before returning.
    unsafe {
        check(
            sys::cuGraphicsMapResources(2, resources.as_mut_ptr(), cu_stream),
            "mapping the eye textures",
        )?;
        let copied = (|| -> Result<()> {
            for (resource, image) in [(left.resource, &eyes.left), (right.resource, &eyes.right)] {
                let mut array = std::ptr::null_mut();
                check(
                    sys::cuGraphicsSubResourceGetMappedArray(&mut array, resource, 0, 0),
                    "eye texture array",
                )?;
                let (source, guard) = image.buffer().device_ptr(stream);
                let pitch = image.width() * 4;
                let copy = sys::CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcHost: std::ptr::null(),
                    srcDevice: source,
                    srcArray: std::ptr::null_mut(),
                    srcPitch: pitch,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: 0,
                    dstArray: array,
                    dstPitch: 0,
                    WidthInBytes: pitch,
                    Height: image.height(),
                };
                check(sys::cuMemcpy2DAsync_v2(&copy, cu_stream), "copying an eye")?;
                drop(guard);
            }
            Ok(())
        })();
        check(
            sys::cuGraphicsUnmapResources(2, resources.as_mut_ptr(), cu_stream),
            "unmapping the eye textures",
        )?;
        copied?;
        check(sys::cuStreamSynchronize(cu_stream), "copying the eyes")?;
    }
    Ok(())
}

/// Unregisters and drops the host-side textures.
fn release(targets: Targets) {
    let mut pairs: Vec<(EyeTexture, EyeTexture)> =
        targets.retired.into_iter().map(|(pair, _)| pair).collect();
    pairs.extend(targets.eyes);
    for (left, right) in pairs {
        for eye in [left, right] {
            // SAFETY: registered once in eye_texture, unregistered once here.
            unsafe { sys::cuGraphicsUnregisterResource(eye.resource) };
        }
    }
}

/// Stops and joins every thread and frees the GPU resources (the library
/// can be unloaded afterwards). Stop issuing render events first.
#[no_mangle]
pub extern "C" fn ivr_shutdown() -> i32 {
    guarded(|| {
        let instance = lock(&INSTANCE).take();
        if let Some(targets) = lock(&TARGETS).take() {
            if let Some(instance) = &instance {
                let _ = instance.gpu.context().bind_to_thread();
            }
            release(targets);
        }
        if let Some(mut instance) = instance {
            instance.stop.store(true, Ordering::Release);
            if let Some(thread) = instance.pipeline.take() {
                let _ = thread.join();
            }
            // Stops WGC (or the test pattern) and joins its thread.
            drop(instance.capture.take());
            tracing::info!("ivr_native shut down");
        }
        Ok(())
    })
}
