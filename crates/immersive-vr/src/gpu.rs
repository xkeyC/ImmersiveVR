//! The GPU side of the pipeline: CUDA kernels (`kernels.cu`, embedded as
//! PTX) and the buffers they work on, all in the CUDA primary context that
//! ONNX Runtime and NVENC use too.
//!
//! * [`GpuImage`]: a BGRA picture (a capture, or one scaled for an eye or for
//!   the depth model), made by [`Gpu::resize`].
//! * [`GpuFrame`]: an NV12 frame of both eyes, made by [`Gpu::warp`] and
//!   encoded in place by NVENC.
//!
//! The warp kernel replaces the ONNX warp graph, which spent most of its
//! time moving full-resolution float intermediates through memory (4K:
//! 12 ms there, well under 1 ms here).

use anyhow::{ensure, Result};
use cudarc::{
    driver::{
        result, CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    },
    nvrtc::Ptx,
};
use depth_infer::StereoFields;
use std::sync::Arc;

const PTX: &str = include_str!("kernels.ptx");

/// Threads per warp block (x, y): each thread covers 2x2 pixels.
const WARP_BLOCK: (u32, u32) = (16, 8);
/// Threads per resize block: one per destination pixel.
const RESIZE_BLOCK: (u32, u32) = (16, 16);

fn cuda(what: &str) -> impl FnOnce(cudarc::driver::DriverError) -> anyhow::Error + '_ {
    move |error| anyhow::anyhow!("{what}: {error:?}")
}

/// The loaded kernels.
pub struct Gpu {
    context: Arc<CudaContext>,
    warp: CudaFunction,
    warp_rgba: CudaFunction,
    resize: CudaFunction,
}

impl Gpu {
    pub fn new(context: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let module = context
            .load_module(Ptx::from_src(PTX))
            .map_err(cuda("loading the GPU kernels"))?;
        Ok(Arc::new(Self {
            context: context.clone(),
            warp: module
                .load_function("warp_nv12")
                .map_err(cuda("warp kernel"))?,
            warp_rgba: module
                .load_function("warp_rgba")
                .map_err(cuda("warp kernel"))?,
            resize: module
                .load_function("resize_bgra")
                .map_err(cuda("resize kernel"))?,
        }))
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// A stream for one thread's work.
    pub fn stream(&self) -> Result<Arc<CudaStream>> {
        self.context.new_stream().map_err(cuda("CUDA stream"))
    }

    /// Like [`Self::warp`], but each eye as an RGBA image (`left`, `right`:
    /// the picture's size); done when this returns.
    pub fn warp_eyes(
        &self,
        stream: &CudaStream,
        picture: &GpuImage,
        fields: &StereoFields,
        delta_scale: f32,
        left: &mut GpuImage,
        right: &mut GpuImage,
    ) -> Result<()> {
        let (width, height) = (picture.width, picture.height);
        ensure!(
            (left.width, left.height, right.width, right.height) == (width, height, width, height),
            "eye images must be the picture's size ({width}x{height})"
        );
        ensure!(
            fields.on_gpu(),
            "the warp kernel needs the fields on the GPU (CUDA or TensorRT)"
        );
        let (channels, field_height, field_width) = fields.shape();
        ensure!(channels == 8, "mlbw fields have 8 channels, got {channels}");
        let fields_address = fields.data_ptr() as u64;
        let (width_i, height_i) = (width as i32, height as i32);
        let (field_width_i, field_height_i) = (field_width as i32, field_height as i32);
        let shift = delta_scale * (width - 1) as f32 * 0.5;
        let config = LaunchConfig {
            grid_dim: (
                (width as u32).div_ceil(RESIZE_BLOCK.0),
                (height as u32).div_ceil(RESIZE_BLOCK.1),
                2,
            ),
            block_dim: (RESIZE_BLOCK.0, RESIZE_BLOCK.1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.warp_rgba);
        launch
            .arg(&picture.buffer)
            .arg(&fields_address)
            .arg(&mut left.buffer)
            .arg(&mut right.buffer)
            .arg(&width_i)
            .arg(&height_i)
            .arg(&field_width_i)
            .arg(&field_height_i)
            .arg(&shift);
        // SAFETY: the arguments match warp_rgba's parameters; every buffer is
        // as large as the sizes say, and all stay alive (borrowed) until the
        // synchronize below.
        unsafe { launch.launch(config) }.map_err(cuda("warp kernel launch"))?;
        stream.synchronize().map_err(cuda("warp kernel"))
    }

    /// Scales `source` into `target` (any sizes); done when this returns.
    pub fn resize(
        &self,
        stream: &CudaStream,
        source: &GpuImage,
        target: &mut GpuImage,
    ) -> Result<()> {
        let (sw, sh) = (source.width as i32, source.height as i32);
        let (dw, dh) = (target.width as i32, target.height as i32);
        let config = LaunchConfig {
            grid_dim: (
                (target.width as u32).div_ceil(RESIZE_BLOCK.0),
                (target.height as u32).div_ceil(RESIZE_BLOCK.1),
                1,
            ),
            block_dim: (RESIZE_BLOCK.0, RESIZE_BLOCK.1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.resize);
        launch
            .arg(&source.buffer)
            .arg(&sw)
            .arg(&sh)
            .arg(&mut target.buffer)
            .arg(&dw)
            .arg(&dh);
        // SAFETY: the arguments match resize_bgra's parameters and each
        // image's buffer holds width * height * 4 bytes.
        unsafe { launch.launch(config) }.map_err(cuda("resize kernel launch"))?;
        stream.synchronize().map_err(cuda("resize kernel"))
    }

    /// Renders both eyes of `picture` with `fields` (on the GPU) into
    /// `frame` (the picture's size); done when this returns. `delta_scale`:
    /// iw3's grid shift per field offset, 1 / (depth width / 2 - 1).
    pub fn warp(
        &self,
        stream: &CudaStream,
        picture: &GpuImage,
        fields: &StereoFields,
        delta_scale: f32,
        frame: &mut GpuFrame,
    ) -> Result<()> {
        let (width, height) = (picture.width, picture.height);
        ensure!(
            (frame.width, frame.height) == (width, height * 2),
            "picture is {width}x{height}, the frame {}x{} (both eyes stacked)",
            frame.width,
            frame.height
        );
        ensure!(
            fields.on_gpu(),
            "the warp kernel needs the fields on the GPU (CUDA or TensorRT)"
        );
        let (channels, field_height, field_width) = fields.shape();
        ensure!(channels == 8, "mlbw fields have 8 channels, got {channels}");
        let fields_address = fields.data_ptr() as u64;
        let frame_address = frame.address;
        let (width_i, height_i) = (width as i32, height as i32);
        let (field_width_i, field_height_i) = (field_width as i32, field_height as i32);
        let shift = delta_scale * (width - 1) as f32 * 0.5;
        let config = LaunchConfig {
            grid_dim: (
                (width as u32 / 2).div_ceil(WARP_BLOCK.0),
                (height as u32 / 2).div_ceil(WARP_BLOCK.1),
                2,
            ),
            block_dim: (WARP_BLOCK.0, WARP_BLOCK.1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.warp);
        launch
            .arg(&picture.buffer)
            .arg(&fields_address)
            .arg(&frame_address)
            .arg(&width_i)
            .arg(&height_i)
            .arg(&field_width_i)
            .arg(&field_height_i)
            .arg(&shift);
        // SAFETY: the arguments match warp_nv12's parameters; every buffer
        // is as large as the sizes passed say (checked above / allocated so),
        // and the fields and frame stay alive (borrowed) until the
        // synchronize below.
        unsafe { launch.launch(config) }.map_err(cuda("warp kernel launch"))?;
        stream.synchronize().map_err(cuda("warp kernel"))
    }
}

/// A BGRA picture in GPU memory, rows `width * 4` bytes apart.
pub struct GpuImage {
    buffer: CudaSlice<u8>,
    width: usize,
    height: usize,
}

impl GpuImage {
    pub fn new(stream: &Arc<CudaStream>, width: usize, height: usize) -> Result<Self> {
        Ok(Self {
            buffer: stream
                .alloc_zeros::<u8>(width * height * 4)
                .map_err(cuda("image buffer"))?,
            width,
            height,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Replaces the picture with `bgra` from host memory; done when this returns.
    pub fn upload(&mut self, stream: &Arc<CudaStream>, bgra: &[u8]) -> Result<()> {
        ensure!(
            bgra.len() == self.buffer.len(),
            "picture has {} bytes, the image {}x{} BGRA",
            bgra.len(),
            self.width,
            self.height
        );
        stream
            .memcpy_htod(bgra, &mut self.buffer)
            .map_err(cuda("uploading a picture"))?;
        stream.synchronize().map_err(cuda("uploading a picture"))
    }

    /// The picture in host memory.
    pub fn download(&self, stream: &Arc<CudaStream>, bgra: &mut [u8]) -> Result<()> {
        ensure!(
            bgra.len() == self.buffer.len(),
            "download buffer has the wrong size"
        );
        stream
            .memcpy_dtoh(&self.buffer, bgra)
            .map_err(cuda("downloading a picture"))?;
        stream.synchronize().map_err(cuda("downloading a picture"))
    }

    /// The buffer, for a copy into it from elsewhere on the GPU.
    pub fn buffer_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.buffer
    }

    /// The buffer, for a copy out of it elsewhere on the GPU.
    pub fn buffer(&self) -> &CudaSlice<u8> {
        &self.buffer
    }
}

/// An NV12 frame in GPU memory, `width x height`, rows `width` bytes apart.
/// Its address never changes, so NVENC registers it once. Allocated with
/// plain `cuMemAlloc`: NVENC cannot register stream-ordered (pool)
/// allocations, cudarc's default.
pub struct GpuFrame {
    context: Arc<CudaContext>,
    address: u64,
    width: usize,
    height: usize,
}

impl GpuFrame {
    pub fn new(gpu: &Gpu, width: usize, height: usize) -> Result<Self> {
        ensure!(
            width.is_multiple_of(2) && height.is_multiple_of(2),
            "NV12 needs an even size, got {width}x{height}"
        );
        let context = gpu.context.clone();
        let bytes = width * height * 3 / 2;
        context.bind_to_thread().map_err(cuda("CUDA context"))?;
        // SAFETY: a plain allocation of `bytes`, zeroed before use and freed
        // by Drop.
        let address = unsafe {
            let address = result::malloc_sync(bytes).map_err(cuda("frame buffer"))?;
            result::memset_d8_sync(address, 0, bytes).map_err(cuda("frame buffer"))?;
            address
        };
        // The memset runs on the legacy default stream, which the pipeline's
        // (non-blocking) streams are not ordered after: wait for it, or it
        // may land on top of the first render.
        context.synchronize().map_err(cuda("frame buffer"))?;
        Ok(Self {
            context,
            address,
            width,
            height,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Device address of the luma plane; the chroma plane follows it.
    pub fn address(&self) -> u64 {
        self.address
    }
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        // SAFETY: allocated by malloc_sync in this context and freed once;
        // the pipeline drops a frame only when nothing uses it any more.
        let _ = self
            .context
            .bind_to_thread()
            .and_then(|()| unsafe { result::free_sync(self.address) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use depth_infer::{init_runtime, Provider, RuntimeOptions, StereoEngine, WarpEngine};

    fn picture(width: usize, height: usize) -> Vec<u8> {
        (0..width * height)
            .flat_map(|i| {
                let (x, y) = (i % width, i / width);
                [
                    (x * 7 % 256) as u8,
                    (y * 3 % 256) as u8,
                    ((x / 8 + y / 8) % 2 * 200) as u8,
                    255,
                ]
            })
            .collect()
    }

    /// The warp kernel against the ONNX warp graph it replaces. Needs
    /// IVR_STEREO_MODEL_DIR (models/stereo) and an ONNX Runtime with CUDA
    /// (ORT_DYLIB_PATH); skipped without them.
    #[test]
    fn warp_matches_the_onnx_warp() {
        let Some(dir) = std::env::var_os("IVR_STEREO_MODEL_DIR").map(std::path::PathBuf::from)
        else {
            eprintln!("IVR_STEREO_MODEL_DIR not set: skipped");
            return;
        };
        init_runtime(&RuntimeOptions::default()).unwrap();
        let providers = [Provider::Cuda];
        let mut stereo = StereoEngine::new(
            &dir.join("iw3_mlbw_l2_d1_770x434_fields.onnx"),
            &providers,
            0,
        )
        .unwrap();
        let mut onnx = WarpEngine::new(&dir.join("iw3_warp_770x434.onnx"), &providers, 0).unwrap();
        let (fw, fh) = (stereo.width(), stereo.height());
        // Depth with a near square in the middle (strong edges) over a ramp.
        let depth: Vec<f32> = (0..fw * fh)
            .map(|i| {
                let (x, y) = (i % fw, i / fw);
                let square = (fw / 3..2 * fw / 3).contains(&x) && (fh / 3..2 * fh / 3).contains(&y);
                if square {
                    0.9
                } else {
                    y as f32 / fh as f32 * 0.5
                }
            })
            .collect();
        let fields = stereo.infer_fields(&depth, 3.0, 0.5).unwrap();
        let (width, height) = (1280, 720);
        let color = picture(width, height);
        let expected = onnx.render(&color, width, height, &fields).unwrap();

        let gpu = Gpu::new(&CudaContext::new(0).unwrap()).unwrap();
        let stream = gpu.stream().unwrap();
        let mut image = GpuImage::new(&stream, width, height).unwrap();
        image.upload(&stream, &color).unwrap();
        let mut frame = GpuFrame::new(&gpu, width, height * 2).unwrap();
        gpu.warp(
            &stream,
            &image,
            &fields,
            1.0 / (fw / 2 - 1) as f32,
            &mut frame,
        )
        .unwrap();
        let mut got = vec![0u8; expected.len()];
        // SAFETY: the frame holds exactly expected.len() bytes.
        unsafe { result::memcpy_dtoh_sync(&mut got, frame.address) }.unwrap();

        let diffs: Vec<u8> = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| a.abs_diff(*b))
            .collect();
        let mean = diffs.iter().map(|&d| d as f64).sum::<f64>() / diffs.len() as f64;
        let max = *diffs.iter().max().unwrap();
        let off = diffs.iter().filter(|&&d| d > 1).count() as f64 / diffs.len() as f64;
        eprintln!(
            "kernel vs ONNX warp: mean |diff| {mean:.4}, max {max}, >1 on {:.4} %",
            off * 100.0
        );
        // Float rounding differs (fused multiply-adds, interpolation order):
        // off by one in places, more only on a handful of edge pixels.
        assert!(mean < 0.05, "mean difference {mean}");
        assert!(
            off < 0.001,
            "{:.4} % of samples differ by more than 1",
            off * 100.0
        );
    }

    /// The resize kernel against fast_image_resize's bilinear convolution
    /// (what the CPU path used). Needs a CUDA GPU; skipped without one.
    #[test]
    fn resize_matches_the_cpu_filter() {
        let Ok(context) = CudaContext::new(0) else {
            eprintln!("no CUDA device: skipped");
            return;
        };
        let gpu = Gpu::new(&context).unwrap();
        let stream = gpu.stream().unwrap();
        let (sw, sh) = (1920, 1080);
        let source = picture(sw, sh);
        let mut image = GpuImage::new(&stream, sw, sh).unwrap();
        image.upload(&stream, &source).unwrap();
        for (dw, dh) in [(1280, 720), (770, 434), (2560, 1440)] {
            let mut target = GpuImage::new(&stream, dw, dh).unwrap();
            gpu.resize(&stream, &image, &mut target).unwrap();
            let mut got = vec![0u8; dw * dh * 4];
            target.download(&stream, &mut got).unwrap();

            use fast_image_resize::{
                images::{Image, ImageRef},
                FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
            };
            let mut expected = vec![0u8; dw * dh * 4];
            let src = ImageRef::new(sw as u32, sh as u32, &source, PixelType::U8x4).unwrap();
            let mut dst =
                Image::from_slice_u8(dw as u32, dh as u32, &mut expected, PixelType::U8x4).unwrap();
            let options = ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
                .use_alpha(false);
            Resizer::new().resize(&src, &mut dst, &options).unwrap();

            let colour = |v: &[u8]| {
                v.chunks_exact(4)
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect::<Vec<_>>()
            };
            let (got, expected) = (colour(&got), colour(&expected));
            let mean = got
                .iter()
                .zip(&expected)
                .map(|(a, b)| a.abs_diff(*b) as f64)
                .sum::<f64>()
                / got.len() as f64;
            eprintln!("resize {sw}x{sh} -> {dw}x{dh}: mean |diff| vs CPU {mean:.3}");
            assert!(mean < 1.0, "mean difference {mean}");
        }
    }
}
