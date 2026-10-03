use crate::{Error, Provider, Result};
#[cfg(any(
    feature = "cuda",
    feature = "tensorrt",
    all(feature = "directml", windows)
))]
use ort::ep::ExecutionProvider;
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        IoBinding, Session,
    },
    value::{Tensor, TensorElementType, TensorValueType, ValueType},
};
use std::path::{Path, PathBuf};

const INPUT: &str = "bgra";
const OUTPUT: &str = "depth";

/// The exported graphs of one input size, by precision.
#[derive(Debug, Clone, Default)]
pub struct ModelFiles {
    pub fp16: Option<PathBuf>,
    pub fp32: Option<PathBuf>,
}

impl ModelFiles {
    /// `dav2s_{width}x{height}_{fp16,fp32}.onnx` in `dir`, as
    /// `scripts/export_depth.py` names them; at least one must exist.
    pub fn in_dir(dir: &Path, width: usize, height: usize) -> Result<Self> {
        let pick = |precision: &str| {
            let path = dir.join(format!("dav2s_{width}x{height}_{precision}.onnx"));
            path.is_file().then_some(path)
        };
        let files = Self {
            fp16: pick("fp16"),
            fp32: pick("fp32"),
        };
        if files.fp16.is_none() && files.fp32.is_none() {
            return Err(Error::Model {
                path: dir.to_path_buf(),
                reason: format!(
                    "no dav2s_{width}x{height}_{{fp16,fp32}}.onnx here (run scripts/export_depth.py)"
                ),
            });
        }
        Ok(files)
    }

    /// One graph for every provider.
    pub fn single(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        Self {
            fp16: Some(path.clone()),
            fp32: Some(path),
        }
    }

    /// TensorRT builds its own fp16 engine from fp32 (keeping precision where
    /// it matters) and the CPU runs fp16 slowly, so both prefer fp32; CUDA
    /// and DirectML prefer fp16.
    fn for_provider(&self, provider: Provider) -> Option<&Path> {
        let (first, second) = match provider {
            Provider::TensorRt | Provider::Cpu => (&self.fp32, &self.fp16),
            Provider::Cuda | Provider::DirectMl => (&self.fp16, &self.fp32),
        };
        first.as_deref().or(second.as_deref())
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub models: ModelFiles,
    /// Tried in order; the first that loads the model is used.
    pub providers: Vec<Provider>,
    pub device_id: u32,
    /// Replay each frame as a CUDA graph (CUDA and TensorRT), which removes
    /// the per-kernel launch cost.
    pub cuda_graph: bool,
    /// Let TensorRT build fp16 kernels.
    pub trt_fp16: bool,
    /// TensorRT engine and timing caches; default `trt-cache/` beside the model.
    pub trt_cache_dir: Option<PathBuf>,
    /// CPU provider threads; ORT's default when `None`.
    pub cpu_threads: Option<usize>,
}

impl EngineConfig {
    pub fn new(models: ModelFiles) -> Self {
        Self {
            models,
            providers: Provider::DEFAULT_ORDER.to_vec(),
            device_id: 0,
            cuda_graph: true,
            trt_fp16: true,
            trt_cache_dir: None,
            cpu_threads: None,
        }
    }
}

/// A provider passed over while creating an engine, and why.
#[derive(Debug, Clone)]
pub struct ProviderFailure {
    pub provider: Provider,
    pub reason: String,
}

/// One Depth-Anything graph loaded on one provider, with its buffers bound
/// for repeated frames of the model's fixed size.
pub struct DepthEngine {
    // Dropped before the session: bound values and the allocator they use.
    io: Io,
    session: Session,
    provider: Provider,
    model: PathBuf,
    width: usize,
    height: usize,
    depth: Vec<f32>,
    failures: Vec<ProviderFailure>,
}

impl DepthEngine {
    /// Loads the model on the first provider of `config.providers` that
    /// accepts it. A provider that fails is recorded (see
    /// [`Self::provider_failures`]) and the next one is tried with a fresh
    /// session, so a requested GPU never silently becomes the CPU inside a
    /// half-assigned session.
    pub fn new(config: &EngineConfig) -> Result<Self> {
        if config.providers.is_empty() {
            return Err(Error::Invalid("no providers requested".into()));
        }
        let mut failures = Vec::new();
        for &provider in &config.providers {
            let Some(model) = config.models.for_provider(provider) else {
                failures.push(ProviderFailure {
                    provider,
                    reason: "no model file".into(),
                });
                continue;
            };
            match Self::load(provider, model, config) {
                Ok(mut engine) => {
                    tracing::info!(
                        %provider,
                        model = %model.display(),
                        width = engine.width,
                        height = engine.height,
                        "depth engine ready"
                    );
                    engine.failures = failures;
                    return Ok(engine);
                }
                Err(error) => {
                    tracing::warn!(%provider, %error, "provider unusable, trying the next");
                    failures.push(ProviderFailure {
                        provider,
                        reason: error.to_string(),
                    });
                }
            }
        }
        Err(Error::NoProvider(
            failures
                .iter()
                .map(|failure| format!("{}: {}", failure.provider, failure.reason))
                .collect::<Vec<_>>()
                .join("; "),
        ))
    }

    fn load(provider: Provider, model: &Path, config: &EngineConfig) -> Result<Self> {
        let session = build_session(provider, model, config)?;
        let (width, height) = check_io(&session, model)?;
        let io = if provider.is_cuda() {
            Io::Device(DeviceIo::new(&session, config.device_id, width, height)?)
        } else {
            Io::Host(HostIo::new(&session, width, height)?)
        };
        Ok(Self {
            io,
            session,
            provider,
            model: model.to_path_buf(),
            width,
            height,
            depth: vec![0.0; width * height],
            failures: Vec::new(),
        })
    }

    /// Model input width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Model input height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    pub fn provider(&self) -> Provider {
        self.provider
    }

    pub fn model_path(&self) -> &Path {
        &self.model
    }

    /// Providers tried before [`Self::provider`] and why each was passed over.
    pub fn provider_failures(&self) -> &[ProviderFailure] {
        &self.failures
    }

    /// Bytes of one input frame: `width * height * 4` (BGRA).
    pub fn frame_len(&self) -> usize {
        self.width * self.height * 4
    }

    /// Runs one BGRA frame of the model's size; returns its relative depth,
    /// row-major `height x width`, larger = nearer. The slice is overwritten
    /// by the next call.
    pub fn infer(&mut self, bgra: &[u8]) -> Result<&[f32]> {
        if bgra.len() != self.frame_len() {
            return Err(Error::FrameSize {
                actual: bgra.len(),
                expected: self.frame_len(),
                width: self.width,
                height: self.height,
            });
        }
        match &mut self.io {
            Io::Device(io) => io.run(&mut self.session, bgra, &mut self.depth)?,
            Io::Host(io) => io.run(&mut self.session, bgra, &mut self.depth)?,
        }
        Ok(&self.depth)
    }

    /// Runs `runs` blank frames: autotuning, lazy allocations and CUDA graph
    /// capture happen here instead of on the first real frames.
    pub fn warmup(&mut self, runs: usize) -> Result<()> {
        let frame = vec![0u8; self.frame_len()];
        for _ in 0..runs {
            self.infer(&frame)?;
        }
        Ok(())
    }
}

enum Io {
    Device(DeviceIo),
    Host(HostIo),
}

/// CUDA and TensorRT: the input and output are device buffers bound once at
/// fixed addresses (what a CUDA graph replays). Each frame is copied in, and
/// the depth out, through host staging tensors.
///
/// No view over device memory is built with `TensorRefMut::from_raw`: in ort
/// 2.0.0-rc.13 it tags the memory as CPU (see the root Cargo.toml).
struct DeviceIo {
    binding: IoBinding,
    input_host: Tensor<u8>,
    output_host: Tensor<f32>,
    input_device: Tensor<u8>,
    // Last: the device buffers above were allocated by it.
    _allocator: Allocator,
}

impl DeviceIo {
    fn new(session: &Session, device_id: u32, width: usize, height: usize) -> Result<Self> {
        let memory = MemoryInfo::new(
            AllocationDevice::CUDA,
            device_id as i32,
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let allocator = Allocator::new(session, memory.clone())?;
        let input_shape = [1, height, width, 4];
        let output_shape = [1, height, width];
        let input_device = Tensor::<u8>::new(&allocator, input_shape)?;
        let output_device = Tensor::<f32>::new(&allocator, output_shape)?;
        let mut binding = session.create_binding()?;
        // A device input on the session's device is used in place, so later
        // writes into `input_device` reach every run. The binding keeps the
        // output buffer and every run hands back a value over it.
        binding.bind_input(INPUT, &input_device)?;
        binding.bind_output(OUTPUT, output_device)?;
        let host = Allocator::default();
        Ok(Self {
            binding,
            input_host: Tensor::<u8>::new(&host, input_shape)?,
            output_host: Tensor::<f32>::new(&host, output_shape)?,
            input_device,
            _allocator: allocator,
        })
    }

    fn run(&mut self, session: &mut Session, bgra: &[u8], depth: &mut [f32]) -> Result<()> {
        self.input_host.extract_tensor_mut().1.copy_from_slice(bgra);
        self.input_host.copy_into(&mut self.input_device)?;
        let outputs = session.run_binding(&self.binding)?;
        outputs
            .get(OUTPUT)
            .ok_or_else(|| Error::Ort(format!("run returned no `{OUTPUT}`")))?
            .downcast_ref::<TensorValueType<f32>>()?
            .copy_into(&mut self.output_host)?;
        drop(outputs);
        depth.copy_from_slice(self.output_host.extract_tensor().1);
        Ok(())
    }
}

/// DirectML and CPU: the frame is bound from host memory for each run and the
/// depth comes back to host memory.
struct HostIo {
    binding: IoBinding,
    input: Tensor<u8>,
}

impl HostIo {
    fn new(session: &Session, width: usize, height: usize) -> Result<Self> {
        let mut binding = session.create_binding()?;
        let host = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )?;
        binding.bind_output_to_device(OUTPUT, &host)?;
        let input = Tensor::<u8>::new(&Allocator::default(), [1, height, width, 4])?;
        Ok(Self { binding, input })
    }

    fn run(&mut self, session: &mut Session, bgra: &[u8], depth: &mut [f32]) -> Result<()> {
        self.input.extract_tensor_mut().1.copy_from_slice(bgra);
        // Binding copies the frame to the provider's device now (DirectML) or
        // refers to it in place (CPU).
        self.binding.bind_input(INPUT, &self.input)?;
        let outputs = session.run_binding(&self.binding)?;
        let output = outputs
            .get(OUTPUT)
            .ok_or_else(|| Error::Ort(format!("run returned no `{OUTPUT}`")))?;
        depth.copy_from_slice(output.try_extract_tensor::<f32>()?.1);
        Ok(())
    }
}

pub(crate) fn build_session(
    provider: Provider,
    model: &Path,
    config: &EngineConfig,
) -> Result<Session> {
    let builder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    let mut builder = match provider {
        Provider::TensorRt => tensorrt(builder, model, config)?,
        Provider::Cuda => cuda(builder, config)?,
        Provider::DirectMl => directml(builder, config)?,
        Provider::Cpu => match config.cpu_threads {
            Some(threads) => builder.with_intra_threads(threads)?,
            None => builder,
        },
    };
    Ok(builder.commit_from_file(model)?)
}

#[cfg(feature = "tensorrt")]
fn tensorrt(
    builder: SessionBuilder,
    model: &Path,
    config: &EngineConfig,
) -> Result<SessionBuilder> {
    let trt = ort::ep::TensorRT::default();
    if !trt.is_available()? {
        return Err(Error::Runtime(
            "the loaded ONNX Runtime has no TensorRT provider".into(),
        ));
    }
    let cache = config.trt_cache_dir.clone().unwrap_or_else(|| {
        model
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("trt-cache")
    });
    std::fs::create_dir_all(&cache).map_err(|source| Error::Io {
        path: cache.clone(),
        source,
    })?;
    let cache = cache.display().to_string();
    tracing::info!(
        %cache,
        "TensorRT: the first session for a model and GPU builds an engine (can take minutes), later ones load it from the cache"
    );
    let trt = trt
        .with_device_id(config.device_id as i32)
        .with_fp16(config.trt_fp16)
        .with_engine_cache(true)
        .with_engine_cache_path(&cache)
        .with_timing_cache(true)
        .with_timing_cache_path(&cache)
        .with_cuda_graph(config.cuda_graph);
    // Nodes TensorRT rejects run on CUDA rather than the CPU. CUDA's own
    // graph capture stays off: it requires the whole graph on CUDA.
    let cuda = ort::ep::CUDA::default().with_device_id(config.device_id as i32);
    Ok(builder.with_execution_providers([
        trt.build().error_on_failure(),
        cuda.build().error_on_failure(),
    ])?)
}

#[cfg(not(feature = "tensorrt"))]
fn tensorrt(_: SessionBuilder, _: &Path, _: &EngineConfig) -> Result<SessionBuilder> {
    Err(Error::Runtime(
        "built without the `tensorrt` feature".into(),
    ))
}

#[cfg(feature = "cuda")]
fn cuda(builder: SessionBuilder, config: &EngineConfig) -> Result<SessionBuilder> {
    let cuda = ort::ep::CUDA::default();
    if !cuda.is_available()? {
        return Err(Error::Runtime(
            "the loaded ONNX Runtime has no CUDA provider".into(),
        ));
    }
    let cuda = cuda
        .with_device_id(config.device_id as i32)
        .with_cuda_graph(config.cuda_graph);
    Ok(builder.with_execution_providers([cuda.build().error_on_failure()])?)
}

#[cfg(not(feature = "cuda"))]
fn cuda(_: SessionBuilder, _: &EngineConfig) -> Result<SessionBuilder> {
    Err(Error::Runtime("built without the `cuda` feature".into()))
}

#[cfg(all(feature = "directml", windows))]
fn directml(builder: SessionBuilder, config: &EngineConfig) -> Result<SessionBuilder> {
    let dml = ort::ep::DirectML::default();
    if !dml.is_available()? {
        return Err(Error::Runtime(
            "the loaded ONNX Runtime has no DirectML provider (it needs the DirectML build)".into(),
        ));
    }
    let dml = dml.with_device_id(config.device_id as i32);
    // DirectML supports neither memory patterns nor parallel execution.
    Ok(builder
        .with_memory_pattern(false)?
        .with_parallel_execution(false)?
        .with_execution_providers([dml.build().error_on_failure()])?)
}

#[cfg(not(all(feature = "directml", windows)))]
fn directml(_: SessionBuilder, _: &EngineConfig) -> Result<SessionBuilder> {
    Err(Error::Runtime(
        "built without the `directml` feature (or not on Windows)".into(),
    ))
}

/// Checks the graph's contract (`bgra` u8 `[1,H,W,4]` -> `depth` f32
/// `[1,H,W]`, fixed H and W) and returns `(W, H)`.
fn check_io(session: &Session, model: &Path) -> Result<(usize, usize)> {
    let bad = |reason: String| Error::Model {
        path: model.to_path_buf(),
        reason,
    };
    let ([input], [output]) = (session.inputs(), session.outputs()) else {
        return Err(bad(format!(
            "expected one input `{INPUT}` and one output `{OUTPUT}`, found {} and {}",
            session.inputs().len(),
            session.outputs().len()
        )));
    };
    if input.name() != INPUT || output.name() != OUTPUT {
        return Err(bad(format!(
            "expected `{INPUT}` -> `{OUTPUT}`, found `{}` -> `{}`",
            input.name(),
            output.name()
        )));
    }
    let (
        ValueType::Tensor {
            ty: input_ty,
            shape: input_shape,
            ..
        },
        ValueType::Tensor {
            ty: output_ty,
            shape: output_shape,
            ..
        },
    ) = (input.dtype(), output.dtype())
    else {
        return Err(bad("input and output must be tensors".into()));
    };
    let fixed = input_shape.len() == 4
        && input_shape[0] == 1
        && input_shape[1] > 0
        && input_shape[2] > 0
        && input_shape[3] == 4;
    if *input_ty != TensorElementType::Uint8 || !fixed {
        return Err(bad(format!(
            "input must be u8 [1,H,W,4] with fixed H and W, found {input_ty:?} {:?}",
            &input_shape[..]
        )));
    }
    let (height, width) = (input_shape[1], input_shape[2]);
    if *output_ty != TensorElementType::Float32 || output_shape[..] != [1, height, width] {
        return Err(bad(format!(
            "output must be f32 [1,{height},{width}], found {output_ty:?} {:?}",
            &output_shape[..]
        )));
    }
    Ok((width as usize, height as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn engine_can_move_to_an_inference_thread() {
        assert_send::<DepthEngine>();
    }

    #[test]
    fn model_choice_follows_provider() {
        let files = ModelFiles {
            fp16: Some("a_fp16.onnx".into()),
            fp32: Some("a_fp32.onnx".into()),
        };
        assert_eq!(
            files.for_provider(Provider::TensorRt),
            Some(Path::new("a_fp32.onnx"))
        );
        assert_eq!(
            files.for_provider(Provider::Cpu),
            Some(Path::new("a_fp32.onnx"))
        );
        assert_eq!(
            files.for_provider(Provider::Cuda),
            Some(Path::new("a_fp16.onnx"))
        );
        let only16 = ModelFiles {
            fp16: Some("b.onnx".into()),
            fp32: None,
        };
        assert_eq!(
            only16.for_provider(Provider::TensorRt),
            Some(Path::new("b.onnx"))
        );
    }

    #[test]
    fn missing_models_are_reported() {
        let error = ModelFiles::in_dir(Path::new("does/not/exist"), 518, 294).unwrap_err();
        assert!(error.to_string().contains("dav2s_518x294"));
    }
}
