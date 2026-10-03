//! Learned stereo: iw3's mlbw_l2 (nagadomi/nunif, MIT) and the warp that
//! renders both eyes from its output. The graphs come from
//! `scripts/export_iw3_stereo.py`:
//!
//! ```text
//! fields: depth [1,1,h,w] f32 (0 = far .. 1 = near), divergence [1] f32 (% of width),
//!         convergence [1] f32  ->  fields [1,C,h,w] f32 (per eye: sampling offsets and blend weights)
//! warp:   color [1,H,W,4] u8 (BGRA, any size), fields [1,C,h,w] f32
//!         ->  nv12 [1,3H,W] u8 (NV12 of the 2H x W frame: left eye on top,
//!             right eye below; BT.709 limited range)
//! ```
//!
//! Learned offsets handle depth edges better than shifting by depth (no
//! ghosting; mlbw blends two layers so foreground outlines stay sharp).
//!
//! On CUDA the fields never leave the GPU: the fields session writes each set
//! into a new device tensor ([`StereoFields`]) that the warp binds in place;
//! the warp's NV12 can stay there too ([`WarpEngine::render_into`]).
//! (Pinned host staging was tried and dropped: no faster here, and an output
//! bound to pinned memory crashes ort 2.0.0-rc.13 with ORT 1.30.)

use crate::{
    engine::build_session, EngineConfig, Error, ModelFiles, Provider, ProviderFailure, Result,
};
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, Session},
    value::{Tensor, TensorElementType, TensorRef, ValueType},
};
use std::path::{Path, PathBuf};

/// Where a session's device outputs live: the GPU with CUDA / TensorRT,
/// host memory otherwise.
fn device_allocator(session: &Session, provider: Provider, device_id: u32) -> Result<Allocator> {
    if !matches!(provider, Provider::Cuda | Provider::TensorRt) {
        return Ok(Allocator::default());
    }
    Ok(Allocator::new(
        session,
        MemoryInfo::new(
            AllocationDevice::CUDA,
            device_id as i32,
            AllocatorType::Device,
            MemoryType::Default,
        )?,
    )?)
}

/// One set of mlbw fields from [`StereoEngine::infer_fields`], left where the
/// model made it (GPU memory on CUDA) for [`WarpEngine::render`].
pub struct StereoFields {
    value: Tensor<f32>,
}

impl StereoFields {
    /// `(channels, height, width)` of the fields.
    pub fn shape(&self) -> (usize, usize, usize) {
        let shape = self.value.shape();
        (shape[1] as usize, shape[2] as usize, shape[3] as usize)
    }

    /// Whether the fields are in GPU memory (CUDA / TensorRT sessions).
    pub fn on_gpu(&self) -> bool {
        self.value.memory_info().allocation_device() == AllocationDevice::CUDA
    }

    /// Address of the fields, `[channels, height, width]` f32, contiguous; a
    /// device address when [`Self::on_gpu`]. Valid while this value lives.
    pub fn data_ptr(&self) -> *const std::ffi::c_void {
        self.value.data_ptr()
    }
}

/// Loads `model` on the first of `providers` that accepts it and passes `check`.
fn load<T>(
    model: &Path,
    providers: &[Provider],
    device_id: u32,
    check: impl Fn(&Session, &Path) -> Result<T>,
) -> Result<(Session, Provider, T, Vec<ProviderFailure>)> {
    let mut config = EngineConfig::new(ModelFiles::single(model));
    config.providers = providers.to_vec();
    config.device_id = device_id;
    // Inputs come from host memory every run, which CUDA graphs cannot replay.
    config.cuda_graph = false;
    let mut failures = Vec::new();
    for &provider in providers {
        let loaded = build_session(provider, model, &config)
            .and_then(|session| check(&session, model).map(|checked| (session, checked)));
        match loaded {
            Ok((session, checked)) => {
                tracing::info!(%provider, model = %model.display(), "stereo model ready");
                return Ok((session, provider, checked, failures));
            }
            Err(error) => {
                tracing::warn!(%provider, %error, model = %model.display(), "provider unusable, trying the next");
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

/// `(element type, shape)` of a graph input or output.
fn tensor(
    outlets: &[ort::value::Outlet],
    name: &str,
    model: &Path,
) -> Result<(TensorElementType, Vec<i64>)> {
    let outlet = outlets
        .iter()
        .find(|outlet| outlet.name() == name)
        .ok_or_else(|| Error::Model {
            path: model.to_path_buf(),
            reason: format!("missing `{name}`"),
        })?;
    match outlet.dtype() {
        ValueType::Tensor { ty, shape, .. } => Ok((*ty, shape.to_vec())),
        other => Err(Error::Model {
            path: model.to_path_buf(),
            reason: format!("`{name}` must be a tensor, found {other:?}"),
        }),
    }
}

/// iw3 mlbw_l2: per-eye sampling fields from a normalized depth map.
pub struct StereoEngine {
    session: Session,
    provider: Provider,
    model: PathBuf,
    width: usize,
    height: usize,
    channels: usize,
    fields: Vec<f32>,
    failures: Vec<ProviderFailure>,
    device: Allocator,
    binding: IoBinding,
    /// Host inputs, copied to the device when bound.
    depth_input: Tensor<f32>,
    divergence_input: Tensor<f32>,
    convergence_input: Tensor<f32>,
}

impl StereoEngine {
    pub fn new(model: &Path, providers: &[Provider], device_id: u32) -> Result<Self> {
        let (session, provider, (width, height, channels), failures) =
            load(model, providers, device_id, check_fields)?;
        let device = device_allocator(&session, provider, device_id)?;
        let binding = session.create_binding()?;
        let host = Allocator::default();
        Ok(Self {
            depth_input: Tensor::new(&host, [1usize, 1, height, width])?,
            divergence_input: Tensor::new(&host, [1usize])?,
            convergence_input: Tensor::new(&host, [1usize])?,
            session,
            provider,
            model: model.to_path_buf(),
            width,
            height,
            channels,
            fields: vec![0.0; channels * width * height],
            failures,
            device,
            binding,
        })
    }

    /// The fields for a normalized depth map (`height x width`, 0..1), left
    /// on the model's device for [`WarpEngine::render`]. Each call makes a new
    /// set; earlier ones stay valid while held.
    pub fn infer_fields(
        &mut self,
        depth: &[f32],
        divergence: f32,
        convergence: f32,
    ) -> Result<StereoFields> {
        let (width, height) = (self.width, self.height);
        if depth.len() != width * height {
            return Err(Error::Invalid(format!(
                "depth has {} values, the stereo model takes {width}x{height}",
                depth.len()
            )));
        }
        self.depth_input.extract_tensor_mut().1.copy_from_slice(depth);
        self.divergence_input.extract_tensor_mut().1[0] = divergence;
        self.convergence_input.extract_tensor_mut().1[0] = convergence;
        self.binding.bind_input("depth", &self.depth_input)?;
        self.binding.bind_input("divergence", &self.divergence_input)?;
        self.binding.bind_input("convergence", &self.convergence_input)?;
        // A new output buffer each run: the warp may still read the last one.
        let output = Tensor::<f32>::new(&self.device, [1usize, self.channels, height, width])?;
        self.binding.bind_output("fields", output)?;
        let mut outputs = self.session.run_binding(&self.binding)?;
        let value = outputs
            .remove("fields")
            .ok_or_else(|| Error::Ort("run returned no `fields`".into()))?
            .downcast::<ort::value::TensorValueType<f32>>()?;
        Ok(StereoFields { value })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Values per pixel of the output (8 for mlbw_l2: two offsets and two
    /// weights per eye).
    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn provider(&self) -> Provider {
        self.provider
    }

    pub fn model_path(&self) -> &Path {
        &self.model
    }

    pub fn provider_failures(&self) -> &[ProviderFailure] {
        &self.failures
    }

    /// The fields for a normalized depth map (`height x width`, 0..1), as
    /// `[channels, height, width]`. Overwritten by the next call.
    pub fn infer(&mut self, depth: &[f32], divergence: f32, convergence: f32) -> Result<&[f32]> {
        let (width, height) = (self.width, self.height);
        if depth.len() != width * height {
            return Err(Error::Invalid(format!(
                "depth has {} values, the stereo model takes {width}x{height}",
                depth.len()
            )));
        }
        let divergence = [divergence];
        let convergence = [convergence];
        let outputs = self.session.run(ort::inputs![
            "depth" => TensorRef::from_array_view(([1usize, 1, height, width], depth))?,
            "divergence" => TensorRef::from_array_view(([1usize], &divergence[..]))?,
            "convergence" => TensorRef::from_array_view(([1usize], &convergence[..]))?,
        ])?;
        let (_, fields) = outputs["fields"].try_extract_tensor::<f32>()?;
        self.fields.copy_from_slice(fields);
        Ok(&self.fields)
    }
}

/// `depth [1,1,h,w]`, `divergence [1]`, `convergence [1]` -> `fields [1,C,h,w]`,
/// all f32 with fixed sizes; returns `(w, h, C)`.
fn check_fields(session: &Session, model: &Path) -> Result<(usize, usize, usize)> {
    let bad = |reason: String| Error::Model {
        path: model.to_path_buf(),
        reason,
    };
    let (ty, depth) = tensor(session.inputs(), "depth", model)?;
    if ty != TensorElementType::Float32
        || depth.len() != 4
        || depth[..2] != [1, 1]
        || depth[2] <= 0
        || depth[3] <= 0
    {
        return Err(bad(format!(
            "`depth` must be f32 [1,1,h,w] with fixed h and w, found {ty:?} {depth:?}"
        )));
    }
    for scalar in ["divergence", "convergence"] {
        let (ty, shape) = tensor(session.inputs(), scalar, model)?;
        if ty != TensorElementType::Float32 || shape != [1] {
            return Err(bad(format!(
                "`{scalar}` must be f32 [1], found {ty:?} {shape:?}"
            )));
        }
    }
    let (ty, fields) = tensor(session.outputs(), "fields", model)?;
    if ty != TensorElementType::Float32
        || fields.len() != 4
        || fields[0] != 1
        || fields[1] <= 0
        || fields[2..] != depth[2..]
    {
        return Err(bad(format!(
            "`fields` must be f32 [1,C,{},{}], found {ty:?} {fields:?}",
            depth[2], depth[3]
        )));
    }
    Ok((depth[3] as usize, depth[2] as usize, fields[1] as usize))
}

/// Renders both eyes from a picture and [`StereoEngine`] fields.
pub struct WarpEngine {
    session: Session,
    provider: Provider,
    /// Fields `(channels, height, width)` the graph takes.
    fields_shape: (usize, usize, usize),
    device: Allocator,
    binding: IoBinding,
    /// The picture's host buffer, for the current size.
    color: Option<Tensor<u8>>,
}

/// A frame buffer in the warp's device memory (GPU with CUDA) for
/// [`WarpEngine::render_into`]: NV12 of a `width x 2*height` frame (left eye
/// on top), rows `width` bytes apart. It keeps its address for life, so an
/// encoder can register it once.
pub struct DeviceFrame {
    value: Tensor<u8>,
    width: usize,
    height: usize,
}

impl DeviceFrame {
    /// Width of the frame (and of one eye).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height of one eye; the frame is twice as tall.
    pub fn eye_height(&self) -> usize {
        self.height
    }

    /// Device address of the luma plane; the chroma plane follows it.
    pub fn device_ptr(&self) -> *const std::ffi::c_void {
        self.value.data_ptr()
    }
}

impl WarpEngine {
    pub fn new(model: &Path, providers: &[Provider], device_id: u32) -> Result<Self> {
        let (session, provider, fields_shape, _) = load(model, providers, device_id, check_warp)?;
        let device = device_allocator(&session, provider, device_id)?;
        let binding = session.create_binding()?;
        Ok(Self {
            session,
            provider,
            fields_shape,
            device,
            binding,
            color: None,
        })
    }

    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// Both eyes for a `width x height` BGRA picture: the NV12 planes of a
    /// `width x 2*height` frame, left eye on top.
    pub fn render(
        &mut self,
        color: &[u8],
        width: usize,
        height: usize,
        fields: &StereoFields,
    ) -> Result<Vec<u8>> {
        let output = Tensor::<u8>::new(&Allocator::default(), [1usize, height * 3, width])?;
        let nv12 = self.run(color, width, height, fields, output)?;
        Ok(nv12.extract_tensor().1.to_vec())
    }

    /// A frame buffer for `width x height` pictures in the warp's device memory.
    pub fn new_frame(&self, width: usize, height: usize) -> Result<DeviceFrame> {
        Ok(DeviceFrame {
            value: Tensor::new(&self.device, [1usize, height * 3, width])?,
            width,
            height,
        })
    }

    /// Like [`Self::render`], but into `frame` (made by [`Self::new_frame`]
    /// for the picture's size), where the result stays.
    pub fn render_into(
        &mut self,
        color: &[u8],
        fields: &StereoFields,
        frame: &mut DeviceFrame,
    ) -> Result<()> {
        // The output is bound over the frame's own memory: a second handle to
        // the same tensor, not a copy.
        let output = frame
            .value
            .view_mut()
            .try_upgrade()
            .map_err(|_| Error::Ort("frame tensor cannot be bound as an output".into()))?;
        self.run(color, frame.width, frame.height, fields, output)?;
        Ok(())
    }

    fn run(
        &mut self,
        color: &[u8],
        width: usize,
        height: usize,
        fields: &StereoFields,
        output: Tensor<u8>,
    ) -> Result<Tensor<u8>> {
        let (channels, field_height, field_width) = self.fields_shape;
        if color.len() != width * height * 4 {
            return Err(Error::Invalid(format!(
                "warp takes a {width}x{height} BGRA picture ({} bytes), got {}",
                width * height * 4,
                color.len()
            )));
        }
        if fields.value.shape()[..] != [1, channels as i64, field_height as i64, field_width as i64] {
            return Err(Error::Invalid(format!(
                "warp takes {channels}x{field_height}x{field_width} fields, got {:?}",
                fields.value.shape()
            )));
        }
        let shape = [1, height as i64, width as i64, 4];
        if self.color.as_ref().is_none_or(|color| color.shape()[..] != shape) {
            self.color = Some(Tensor::new(&Allocator::default(), [1usize, height, width, 4])?);
        }
        let staged = self.color.as_mut().expect("allocated above");
        staged.extract_tensor_mut().1.copy_from_slice(color);
        // A host input is copied to the device when bound; the fields are
        // already there and are read in place.
        self.binding.bind_input("color", &*staged)?;
        self.binding.bind_input("fields", &fields.value)?;
        self.binding.bind_output("nv12", output)?;
        let mut outputs = self.session.run_binding(&self.binding)?;
        Ok(outputs
            .remove("nv12")
            .ok_or_else(|| Error::Ort("run returned no `nv12`".into()))?
            .downcast::<ort::value::TensorValueType<u8>>()?)
    }
}

/// `color [1,H,W,4]` u8 (H, W free) and `fields [1,C,h,w]` f32 -> `nv12 [1,3H,W]` u8;
/// returns `(C, h, w)`.
fn check_warp(session: &Session, model: &Path) -> Result<(usize, usize, usize)> {
    let bad = |reason: String| Error::Model {
        path: model.to_path_buf(),
        reason,
    };
    let (ty, color) = tensor(session.inputs(), "color", model)?;
    if ty != TensorElementType::Uint8
        || color.len() != 4
        || !matches!(color[0], 1 | -1)
        || color[3] != 4
    {
        return Err(bad(format!(
            "`color` must be u8 [1,H,W,4], found {ty:?} {color:?}"
        )));
    }
    let (ty, fields) = tensor(session.inputs(), "fields", model)?;
    if ty != TensorElementType::Float32
        || fields.len() != 4
        || fields[0] != 1
        || fields[1..].iter().any(|d| *d <= 0)
    {
        return Err(bad(format!(
            "`fields` must be f32 [1,C,h,w] with fixed sizes, found {ty:?} {fields:?}"
        )));
    }
    let (ty, nv12) = tensor(session.outputs(), "nv12", model)?;
    if ty != TensorElementType::Uint8 || nv12.len() != 3 || !matches!(nv12[0], 1 | -1) {
        return Err(bad(format!(
            "`nv12` must be u8 [1,3H,W], found {ty:?} {nv12:?}"
        )));
    }
    Ok((fields[1] as usize, fields[2] as usize, fields[3] as usize))
}
