//! ImmersiveVR: the Windows desktop as real-time stereo 3D.
//!
//! capture (WGC, into CUDA) -> depth (Depth-Anything-V2-Small) -> iw3
//! mlbw_l2 fields -> both eyes rendered on the GPU. From there the frames go
//! either to NVENC and a WebSocket for a WebXR browser (the `immersive-vr`
//! binary), or, as RGBA eye images, to an in-process consumer such as the
//! Unity plugin (`ivr-native`).

pub mod audio;
pub mod capture;
pub mod codec;
pub mod encoder;
pub mod engines;
pub mod gpu;
pub mod pipeline;
pub mod server;
pub mod tls;
pub mod web;
