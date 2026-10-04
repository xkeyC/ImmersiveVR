//! Loading the depth and stereo models and the GPU kernels, once at startup.

use crate::gpu::{Gpu, GpuFrame, GpuImage};
use anyhow::{Context as _, Result};
use depth_infer::{DepthEngine, EngineConfig, ModelFiles, Provider, StereoEngine};
use std::{path::PathBuf, sync::Arc};

pub struct EngineOptions {
    /// Directory with dav2s_{W}x{H}_{fp16,fp32}.onnx.
    pub models: PathBuf,
    /// Directory with iw3_mlbw_l2_d{1,2,3}_{W}x{H}_fields.onnx.
    pub stereo_models: PathBuf,
    /// Depth model input size.
    pub depth_size: (usize, usize),
    /// Provider order for the models.
    pub providers: Vec<Provider>,
    /// For the warm-up runs.
    pub divergence: f32,
    pub convergence: f32,
}

/// Everything the pipeline computes with.
pub struct Engines {
    pub depth: DepthEngine,
    /// mlbw_l2 fields models by level (iw3 levels 1, 2, 3).
    pub stereo: Vec<StereoEngine>,
    /// The GPU kernels, in the CUDA primary context (ONNX Runtime's too).
    pub gpu: Arc<Gpu>,
    /// iw3's grid shift per field offset: 1 / (depth width / 2 - 1).
    pub delta_scale: f32,
}

/// "WxH" as a size.
pub fn parse_size(size: &str) -> Result<(usize, usize)> {
    size.split_once(['x', 'X'])
        .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
        .with_context(|| format!("a size is WxH, got {size:?}"))
}

/// Loads and warms up the models (ONNX Runtime must be initialized) and the
/// GPU kernels, checking the stereo fields end up where the warp reads them.
pub fn load(options: &EngineOptions) -> Result<Engines> {
    let (depth_width, depth_height) = options.depth_size;
    let mut config = EngineConfig::new(ModelFiles::in_dir(
        &options.models,
        depth_width,
        depth_height,
    )?);
    config.providers = options.providers.clone();
    let mut depth = DepthEngine::new(&config)?;
    if depth.provider() == Provider::Cuda && config.cuda_graph {
        // ONNX Runtime's CUDA provider records its graph per thread and in
        // global capture mode: the depth thread would record it while the
        // capture and warp threads use CUDA, which breaks the recording
        // (cudaErrorStreamCaptureInvalidated). TensorRT's is unaffected.
        let providers = std::mem::replace(&mut config.providers, vec![Provider::Cuda]);
        config.cuda_graph = false;
        depth = DepthEngine::new(&config)?;
        config.providers = providers;
    }
    depth.warmup(5)?;
    tracing::info!(provider = %depth.provider(), "depth ready");

    let size = format!("{depth_width}x{depth_height}");
    let mut stereo = Vec::new();
    for level in 1..=3 {
        let path = options
            .stereo_models
            .join(format!("iw3_mlbw_l2_d{level}_{size}_fields.onnx"));
        if !path.is_file() {
            anyhow::ensure!(
                level > 1,
                "iw3 model {} not found: run scripts/export_iw3_stereo.py --size {size} (see README)",
                path.display()
            );
            tracing::warn!(model = %path.display(), "missing; stronger settings use the previous level");
            break;
        }
        let mut engine = StereoEngine::new(&path, &config.providers, config.device_id)?;
        anyhow::ensure!(
            (engine.width(), engine.height()) == (depth_width, depth_height),
            "{} is {}x{}, the depth model {size}",
            path.display(),
            engine.width(),
            engine.height()
        );
        // The first runs allocate and autotune (tens of ms): not on a live frame.
        let flat = vec![0.5; engine.width() * engine.height()];
        for _ in 0..3 {
            engine.infer_fields(&flat, options.divergence, options.convergence)?;
        }
        stereo.push(engine);
    }
    // The CUDA primary context (ONNX Runtime's too): the kernels and NVENC
    // work on the fields and frames in place.
    let cuda = cudarc::driver::CudaContext::new(config.device_id as usize)
        .map_err(|error| anyhow::anyhow!("CUDA device {}: {error:?}", config.device_id))?;
    let gpu = Gpu::new(&cuda)?;
    let delta_scale = 1.0 / (depth_width / 2 - 1) as f32;
    {
        // Check the fields are where the warp kernel reads them, and run it once.
        let fields = stereo[0].infer_fields(&vec![0.5; depth_width * depth_height], 3.0, 0.5)?;
        anyhow::ensure!(
            fields.on_gpu(),
            "the stereo model runs on {}; the warp needs it on CUDA or TensorRT",
            stereo[0].provider()
        );
        let stream = gpu.stream()?;
        let picture = GpuImage::new(&stream, 1920, 1080)?;
        let mut frame = GpuFrame::new(&gpu, 1920, 2160)?;
        gpu.warp(&stream, &picture, &fields, delta_scale, &mut frame)?;
    }
    tracing::info!(levels = stereo.len(), provider = %stereo[0].provider(), "iw3 mlbw_l2 ready");
    Ok(Engines {
        depth,
        stereo,
        gpu,
        delta_scale,
    })
}
