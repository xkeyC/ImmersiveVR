//! Monocular depth for ImmersiveVR: Depth-Anything-V2-Small on ONNX Runtime.
//!
//! The graphs come from `scripts/export_depth.py`: a BGRA `u8` frame at a fixed
//! size in, raw relative depth (`f32`, larger = nearer) out. [`DepthEngine`]
//! loads one on the first usable provider of a list (TensorRT, CUDA, DirectML,
//! CPU), keeps its buffers bound across frames and, on CUDA, can replay each
//! frame as a CUDA graph. [`DepthNormalizer`] turns raw depth into a stable
//! 8-bit map.
//!
//! Call [`init_runtime`] once, before the first engine, to load ONNX Runtime.

mod engine;
mod error;
mod normalize;
mod provider;
mod runtime;
mod stereo;

pub use engine::{DepthEngine, EngineConfig, ModelFiles, ProviderFailure};
pub use error::{Error, Result};
pub use normalize::{DepthNormalizer, NormalizerConfig};
pub use provider::Provider;
pub use runtime::{init_runtime, RuntimeOptions};
pub use stereo::{DeviceFrame, StereoEngine, StereoFields, WarpEngine};
