//! Desktop capture with Windows.Graphics.Capture, into GPU memory.
//!
//! WGC delivers a frame only when the screen changes, as a D3D11 texture.
//! Each one is copied (on the GPU) into a texture registered with CUDA once,
//! and from there into a [`GpuImage`] that replaces the previous one in a
//! [`LatestFrame`] slot: the pipeline always takes the newest and never
//! queues. Nothing crosses to host memory. Should the D3D11 / CUDA interop
//! be unavailable, frames are read back and uploaded instead.

use crate::gpu::{Gpu, GpuImage};
use anyhow::{anyhow, bail, Context as _, Result};
use cudarc::driver::{sys, CudaStream, DevicePtrMut};
use std::{
    ffi::c_void,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Condvar, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};
use windows::{
    core::Interface,
    Win32::Graphics::{
        Direct3D11::{ID3D11Device, ID3D11Texture2D, D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT},
        Dxgi::Common::DXGI_SAMPLE_DESC,
    },
};
use windows_capture::{
    capture::{CaptureControl, Context, GraphicsCaptureApiHandler},
    frame::Frame,
    graphics_capture_api::InternalCaptureControl,
    monitor::Monitor,
    settings::{
        ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
        MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
    },
};

/// One captured frame, BGRA, on the GPU.
pub struct CapturedFrame {
    pub image: Arc<GpuImage>,
    /// Increases with every captured frame.
    pub sequence: u64,
    /// When it arrived (for end-to-end latency).
    pub at: Instant,
}

/// The newest captured frame, shared between the capture callback and the
/// pipeline.
#[derive(Default)]
pub struct LatestFrame {
    frame: Mutex<Option<Arc<CapturedFrame>>>,
    sequence: AtomicU64,
    arrived: Condvar,
}

impl LatestFrame {
    fn store(&self, image: Arc<GpuImage>, at: Instant) {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        *self.frame.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(CapturedFrame {
            image,
            sequence,
            at,
        }));
        self.arrived.notify_all();
    }

    /// Waits up to `timeout` for a frame newer than `sequence`.
    pub fn wait_newer_than(&self, sequence: u64, timeout: Duration) -> Option<Arc<CapturedFrame>> {
        let frame = self.frame.lock().unwrap_or_else(|e| e.into_inner());
        let (frame, _) = self
            .arrived
            .wait_timeout_while(frame, timeout, |frame| {
                frame.as_ref().is_none_or(|frame| frame.sequence <= sequence)
            })
            .unwrap_or_else(|e| e.into_inner());
        frame.clone().filter(|frame| frame.sequence > sequence)
    }
}

/// Capture images to write into: one that nothing else holds any more is
/// reused (the newest frame and the one the pipeline is scaling stay held).
struct ImagePool {
    stream: Arc<CudaStream>,
    images: Vec<Arc<GpuImage>>,
}

/// More than the pipeline ever holds at once; past it a frame is dropped.
const POOL_LIMIT: usize = 6;

impl ImagePool {
    fn new(gpu: &Gpu) -> Result<Self> {
        Ok(Self {
            stream: gpu.stream()?,
            images: Vec::new(),
        })
    }

    /// A free image of `width x height`, or `None` if all are in use.
    fn take(&mut self, width: usize, height: usize) -> Result<Option<usize>> {
        if self
            .images
            .first()
            .is_some_and(|image| (image.width(), image.height()) != (width, height))
        {
            self.images.clear();
        }
        if let Some(free) = self.images.iter().position(|image| Arc::strong_count(image) == 1) {
            return Ok(Some(free));
        }
        if self.images.len() == POOL_LIMIT {
            return Ok(None);
        }
        self.images
            .push(Arc::new(GpuImage::new(&self.stream, width, height)?));
        Ok(Some(self.images.len() - 1))
    }
}

/// `cuGraphicsD3D11RegisterResource`, which cudarc does not bind.
type RegisterD3d11 =
    unsafe extern "system" fn(*mut sys::CUgraphicsResource, *mut c_void, u32) -> sys::CUresult;

fn register_d3d11() -> Result<RegisterD3d11> {
    static FUNCTION: OnceLock<std::result::Result<RegisterD3d11, String>> = OnceLock::new();
    FUNCTION
        .get_or_init(|| {
            // SAFETY: nvcuda.dll is the CUDA driver; the symbol has this
            // signature, and the library stays loaded for the process.
            unsafe {
                let library = libloading::Library::new("nvcuda.dll").map_err(|e| e.to_string())?;
                let function = *library
                    .get::<RegisterD3d11>(b"cuGraphicsD3D11RegisterResource\0")
                    .map_err(|e| e.to_string())?;
                std::mem::forget(library);
                Ok(function)
            }
        })
        .clone()
        .map_err(|error| anyhow!("cuGraphicsD3D11RegisterResource: {error}"))
}

fn check(result: sys::CUresult, what: &str) -> Result<()> {
    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        bail!("{what}: {result:?}")
    }
}

/// A texture of ours on the capture's D3D11 device, registered with CUDA:
/// each frame is copied into it on the GPU, then from it into CUDA memory.
struct Interop {
    texture: ID3D11Texture2D,
    resource: sys::CUgraphicsResource,
    width: usize,
    height: usize,
}

// SAFETY: used only on the capture callback's thread after creation.
unsafe impl Send for Interop {}

impl Interop {
    fn new(device: &ID3D11Device, frame: &ID3D11Texture2D, gpu: &Gpu) -> Result<Self> {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: plain D3D11 calls on live interfaces.
        unsafe { frame.GetDesc(&mut desc) };
        let ours = D3D11_TEXTURE2D_DESC {
            Width: desc.Width,
            Height: desc.Height,
            MipLevels: 1,
            ArraySize: 1,
            Format: desc.Format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture = None;
        // SAFETY: as above.
        unsafe { device.CreateTexture2D(&ours, None, Some(&mut texture)) }.context("interop texture")?;
        let texture = texture.context("interop texture")?;
        gpu.context()
            .bind_to_thread()
            .map_err(|error| anyhow!("CUDA context: {error:?}"))?;
        let mut resource = std::ptr::null_mut();
        // SAFETY: registers a live texture of this device, read only from CUDA.
        unsafe {
            check(
                register_d3d11()?(&mut resource, texture.as_raw(), 0),
                "registering the capture texture with CUDA",
            )?;
            check(
                sys::cuGraphicsResourceSetMapFlags_v2(
                    resource,
                    sys::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY as u32,
                ),
                "capture texture map flags",
            )?;
        }
        Ok(Self {
            texture,
            resource,
            width: desc.Width as usize,
            height: desc.Height as usize,
        })
    }

    /// Copies `frame` into `image` (the same size); done when this returns.
    fn copy(&mut self, frame: &Frame, stream: &CudaStream, image: &mut GpuImage) -> Result<()> {
        // SAFETY: both textures belong to the frame's device, same size and format.
        unsafe { frame.device_context().CopyResource(&self.texture, frame.as_raw_texture()) };
        let pitch = self.width * 4;
        let (target, guard) = image.buffer_mut().device_ptr_mut(stream);
        let cu_stream = stream.cu_stream();
        // SAFETY: the resource is registered and mapped only here; the copy
        // stays within the mapped array and the image (width * 4 x height);
        // the stream is synchronized before the image is used or unmapped
        // memory is touched again.
        unsafe {
            check(
                sys::cuGraphicsMapResources(1, &mut self.resource, cu_stream),
                "mapping the capture texture",
            )?;
            let mut array = std::ptr::null_mut();
            let copied = check(
                sys::cuGraphicsSubResourceGetMappedArray(&mut array, self.resource, 0, 0),
                "capture texture array",
            )
            .and_then(|()| {
                // Every field spelled out: the memory types are enums without
                // a zero value, so the struct cannot start zeroed.
                let copy = sys::CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    srcHost: std::ptr::null(),
                    srcDevice: 0,
                    srcArray: array,
                    srcPitch: 0,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: target,
                    dstArray: std::ptr::null_mut(),
                    dstPitch: pitch,
                    WidthInBytes: pitch,
                    Height: self.height,
                };
                check(sys::cuMemcpy2DAsync_v2(&copy, cu_stream), "copying the capture")
            });
            check(
                sys::cuGraphicsUnmapResources(1, &mut self.resource, cu_stream),
                "unmapping the capture texture",
            )?;
            copied?;
            check(sys::cuStreamSynchronize(cu_stream), "copying the capture")?;
        }
        drop(guard);
        Ok(())
    }
}

impl Drop for Interop {
    fn drop(&mut self) {
        // SAFETY: registered once in new, unregistered once here.
        unsafe { sys::cuGraphicsUnregisterResource(self.resource) };
    }
}

pub struct CaptureFlags {
    pub latest: Arc<LatestFrame>,
    pub gpu: Arc<Gpu>,
}

struct Handler {
    latest: Arc<LatestFrame>,
    gpu: Arc<Gpu>,
    pool: ImagePool,
    /// The GPU route, once set up; `Err` once it failed (frames are read back).
    interop: Option<std::result::Result<Interop, ()>>,
    scratch: Vec<u8>,
    dropped: u64,
}

impl GraphicsCaptureApiHandler for Handler {
    type Flags = CaptureFlags;
    type Error = anyhow::Error;

    fn new(context: Context<Self::Flags>) -> Result<Self, Self::Error> {
        let CaptureFlags { latest, gpu } = context.flags;
        Ok(Self {
            pool: ImagePool::new(&gpu)?,
            latest,
            gpu,
            interop: None,
            scratch: Vec::new(),
            dropped: 0,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let at = Instant::now();
        let (width, height) = (frame.width() as usize, frame.height() as usize);
        let Some(slot) = self.pool.take(width, height)? else {
            self.dropped += 1;
            if self.dropped.is_power_of_two() {
                tracing::warn!(dropped = self.dropped, "no free capture buffer: frame dropped");
            }
            return Ok(());
        };
        let image = Arc::get_mut(&mut self.pool.images[slot]).expect("free means unshared");
        if self
            .interop
            .as_ref()
            .is_some_and(|interop| interop.as_ref().is_ok_and(|i| (i.width, i.height) != (width, height)))
        {
            self.interop = None;
        }
        let interop = self.interop.get_or_insert_with(|| {
            Interop::new(frame.device(), frame.as_raw_texture(), &self.gpu)
                .inspect(|_| tracing::info!("capture: D3D11 -> CUDA on the GPU"))
                .map_err(|error| tracing::warn!(%error, "capture: no D3D11 / CUDA interop, reading frames back"))
        });
        match interop {
            Ok(interop) => interop.copy(frame, &self.pool.stream, image)?,
            Err(()) => {
                let buffer = frame.buffer()?;
                let bgra = buffer.as_nopadding_buffer(&mut self.scratch);
                image.upload(&self.pool.stream, bgra)?;
            }
        }
        self.latest.store(self.pool.images[slot].clone(), at);
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        tracing::warn!("capture item closed");
        Ok(())
    }
}

/// Starts capturing monitor `index` (1-based, as Windows numbers them; 0 is
/// the primary) on its own thread, at up to 4x `fps` (a GPU copy each): the
/// pipeline samples the newest on its own frame grid, so the capture should
/// be fresh whenever it looks.
pub fn start(index: usize, fps: u32, latest: Arc<LatestFrame>, gpu: Arc<Gpu>) -> Result<Capture> {
    let monitor = if index == 0 {
        Monitor::primary()
    } else {
        Monitor::from_index(index)
    }
    .map_err(|error| anyhow!("monitor {index}: {error}"))?;
    let size = (
        monitor.width().context("monitor width")? as usize,
        monitor.height().context("monitor height")? as usize,
    );
    tracing::info!(
        monitor = %monitor.name().unwrap_or_default(),
        width = size.0,
        height = size.1,
        "capturing"
    );
    let settings = Settings::new(
        monitor,
        CursorCaptureSettings::WithCursor,
        DrawBorderSettings::WithoutBorder,
        SecondaryWindowSettings::Default,
        MinimumUpdateIntervalSettings::Custom(Duration::from_secs_f64(0.25 / fps as f64)),
        DirtyRegionSettings::Default,
        ColorFormat::Bgra8,
        CaptureFlags { latest, gpu },
    );
    let control = Handler::start_free_threaded(settings)
        .map_err(|error| anyhow!("starting capture: {error}"))?;
    Ok(Capture {
        size,
        _control: Some(control),
    })
}

/// A moving test pattern of `size` at `fps` instead of the screen: every
/// frame differs, like a fast game, to measure what the pipeline sustains.
/// The pattern lives on the GPU (twice as wide), and each frame is a window
/// of it moved a few pixels: cheap enough for a 240 Hz monitor's rate.
pub fn start_synthetic(
    size: (usize, usize),
    fps: u32,
    latest: Arc<LatestFrame>,
    gpu: Arc<Gpu>,
) -> Result<Capture> {
    let (width, height) = size;
    // Bands and blocks at several scales, so the depth model sees structure;
    // each row twice over, so any window of `width` wraps around.
    let row = |y: usize| {
        (0..width).flat_map(move |x| {
            let block = ((x / 160) + (y / 120)).is_multiple_of(2);
            let v = ((x * 255 / width) as u8) ^ if block { 0x40 } else { 0 };
            [v, (y * 255 / height) as u8, ((x + y) % 256) as u8, 255]
        })
    };
    let pattern: Vec<u8> = (0..height).flat_map(|y| row(y).chain(row(y))).collect();
    let mut pool = ImagePool::new(&gpu)?;
    let mut source = pool
        .stream
        .clone_htod(&pattern)
        .map_err(|error| anyhow!("uploading the test pattern: {error:?}"))?;
    std::thread::Builder::new()
        .name("synthetic-capture".into())
        .spawn(move || -> Result<()> {
            let tick = Duration::from_secs_f64(1.0 / fps as f64);
            let mut next = Instant::now();
            let mut shift = 0usize;
            let (mut made, mut report) = (0u64, Instant::now());
            let stream = pool.stream.clone();
            loop {
                let at = Instant::now();
                // The whole picture scrolls sideways a few pixels a frame.
                shift = (shift + 7) % width;
                if let Some(slot) = pool.take(width, height)? {
                    let image = Arc::get_mut(&mut pool.images[slot]).expect("free means unshared");
                    let (from, from_guard) = source.device_ptr_mut(&stream);
                    let (to, to_guard) = image.buffer_mut().device_ptr_mut(&stream);
                    let copy = sys::CUDA_MEMCPY2D {
                        srcXInBytes: shift * 4,
                        srcY: 0,
                        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                        srcHost: std::ptr::null(),
                        srcDevice: from,
                        srcArray: std::ptr::null_mut(),
                        srcPitch: width * 8,
                        dstXInBytes: 0,
                        dstY: 0,
                        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                        dstHost: std::ptr::null_mut(),
                        dstDevice: to,
                        dstArray: std::ptr::null_mut(),
                        dstPitch: width * 4,
                        WidthInBytes: width * 4,
                        Height: height,
                    };
                    // SAFETY: both buffers are live and as large as the copy says.
                    unsafe {
                        check(sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()), "test pattern")?;
                        check(sys::cuStreamSynchronize(stream.cu_stream()), "test pattern")?;
                    }
                    drop((from_guard, to_guard));
                    latest.store(pool.images[slot].clone(), at);
                    made += 1;
                }
                if report.elapsed() >= Duration::from_secs(5) {
                    let rate = made as f64 / report.elapsed().as_secs_f64();
                    tracing::info!(fps = format!("{rate:.1}"), "synthetic capture (last 5 s)");
                    (made, report) = (0, Instant::now());
                }
                // Paced by deadlines, so the rate holds however long a frame took.
                next += tick;
                let now = Instant::now();
                if next > now {
                    std::thread::sleep(next - now);
                } else {
                    next = now;
                }
            }
        })
        .context("spawning the synthetic capture thread")?;
    tracing::info!(width, height, fps, "synthetic capture (test pattern)");
    Ok(Capture {
        size,
        _control: None,
    })
}

/// A running capture; it stops when this is dropped.
pub struct Capture {
    pub size: (usize, usize),
    _control: Option<CaptureControl<Handler, anyhow::Error>>,
}
