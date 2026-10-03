//! Video encoding with NVENC, in process.
//!
//! The warp renders each frame (NV12, BT.709 limited range) into a GPU buffer
//! ([`GpuFrame`]); NVENC encodes it where it is, in the same CUDA context, so
//! a frame never visits host memory before it is compressed. Each call returns exactly one frame's bitstream, sent at
//! once: no muxer, no splitting, no frame of delay.
//!
//! Latency settings follow Sunshine's (LizardByte/Sunshine, nvenc_base.cpp):
//! a VBV of one frame (no frame bigger than the link carries in a frame
//! time, so none queues behind another), keyframes only on request (an
//! infinite GOP: no periodic burst), two-pass at quarter resolution to keep
//! frames within that budget, no reordering or lookahead. Each encoder carries a
//! generation number so chunks of a replaced encoder (after a resolution or
//! codec change) can be told apart.

use crate::{
    codec::{Codec, Level},
    gpu::GpuFrame,
};
use anyhow::{Context as _, Result};
use cudarc::driver::CudaContext;
use moq_nvenc::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_AV1_GUID, NV_ENC_CODEC_HEVC_GUID, NV_ENC_INPUT_RESOURCE_TYPE,
        GUID, NVENC_INFINITE_GOPLENGTH, NV_ENC_MULTI_PASS, NV_ENC_PARAMS_RC_MODE, NV_ENC_PRESET_P1_GUID, NV_ENC_PRESET_P2_GUID, NV_ENC_PRESET_P3_GUID,
        NV_ENC_PRESET_P4_GUID, NV_ENC_PRESET_P5_GUID, NV_ENC_PRESET_P6_GUID, NV_ENC_PRESET_P7_GUID,
        NV_ENC_TUNING_INFO, NV_ENC_VUI_COLOR_PRIMARIES,
        NV_ENC_VUI_MATRIX_COEFFS, NV_ENC_VUI_TRANSFER_CHARACTERISTIC, NV_ENC_VUI_VIDEO_FORMAT,
        NV_ENC_AV1_PROFILE_MAIN_GUID, NV_ENC_HEVC_PROFILE_MAIN_GUID,
    },
    Bitstream, EncodePictureParams, Encoder as Nvenc, EncoderInitParams, RegisteredResource, Session,
};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::broadcast;

/// What a frame was made with, carried to the client beside it.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameMeta {
    /// Stereo divergence (% of width) the frame's offsets were computed for.
    pub divergence: f32,
    pub convergence: f32,
}

/// One encoded frame.
#[derive(Debug)]
pub struct Chunk {
    /// The encoder (stream configuration) that produced it.
    pub generation: u64,
    pub key: bool,
    /// When what the frame shows was captured, microseconds since the Unix epoch.
    pub timestamp_us: u64,
    pub meta: FrameMeta,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub codec: Codec,
    pub width: usize,
    pub height: usize,
    pub fps: u32,
    pub bitrate_mbps: u32,
    /// NVENC preset 1 (fastest) ..= 7 (best quality per bit).
    pub preset: u8,
}

impl EncoderConfig {
    pub fn level(&self) -> Level {
        self.codec
            .level(self.width, self.height, self.fps, self.bitrate_mbps)
    }
}

pub struct Encoder {
    // Dropped first: registrations and the bitstream belong to the session.
    /// Frame buffers registered with this session, by device address.
    registered: HashMap<usize, RegisteredResource<()>>,
    bitstream: Option<Bitstream>,
    session: Session,
    codec: Codec,
    width: usize,
    height: usize,
    generation: u64,
    chunks: broadcast::Sender<Arc<Chunk>>,
    frames: u64,
}

impl Encoder {
    /// Opens an NVENC session; encoded frames are published on `chunks`,
    /// tagged with `generation`.
    pub fn start(
        config: &EncoderConfig,
        generation: u64,
        chunks: broadcast::Sender<Arc<Chunk>>,
        cuda: Arc<CudaContext>,
    ) -> Result<Self> {
        let level = config.level();
        let nvenc = Nvenc::initialize_with_cuda(cuda).context("opening NVENC")?;
        let (codec_guid, profile_guid) = match config.codec {
            Codec::Hevc => (NV_ENC_CODEC_HEVC_GUID, NV_ENC_HEVC_PROFILE_MAIN_GUID),
            Codec::Av1 => (NV_ENC_CODEC_AV1_GUID, NV_ENC_AV1_PROFILE_MAIN_GUID),
        };
        let tuning = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
        let preset_guid = preset_guid(config.preset)?;
        let mut preset = nvenc
            .get_preset_config(codec_guid, preset_guid, tuning)
            .context("NVENC preset")?;
        let encode = &mut preset.presetCfg;
        let bitrate = config.bitrate_mbps * 1_000_000;
        encode.profileGUID = profile_guid;
        // Keyframes only when a client asks (Encoder::send's `keyframe`).
        encode.gopLength = NVENC_INFINITE_GOPLENGTH;
        // No B-frames: every frame comes out as soon as it goes in.
        encode.frameIntervalP = 1;
        let rc = &mut encode.rcParams;
        rc.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR;
        rc.averageBitRate = bitrate;
        rc.maxBitRate = bitrate;
        // One frame's worth: every frame, keyframes too, fits a frame time.
        let frame_bits = bitrate / config.fps;
        rc.vbvBufferSize = frame_bits;
        rc.vbvInitialDelay = frame_bits;
        rc.multiPass = NV_ENC_MULTI_PASS::NV_ENC_TWO_PASS_QUARTER_RESOLUTION;
        rc.lowDelayKeyFrameScale = 1;
        rc.set_zeroReorderDelay(1);
        rc.set_enableLookahead(0);
        match config.codec {
            Codec::Hevc => {
                // SAFETY: the union member for the codec chosen above.
                let hevc = unsafe { &mut encode.encodeCodecConfig.hevcConfig };
                hevc.level = level.nvenc;
                hevc.tier = 0;
                hevc.idrPeriod = NVENC_INFINITE_GOPLENGTH;
                // Parameter sets with every keyframe: a client can join at any.
                hevc.set_repeatSPSPPS(1);
                let vui = &mut hevc.hevcVUIParameters;
                vui.videoSignalTypePresentFlag = 1;
                vui.videoFormat = NV_ENC_VUI_VIDEO_FORMAT::NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
                vui.videoFullRangeFlag = 0;
                vui.colourDescriptionPresentFlag = 1;
                vui.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
                vui.transferCharacteristics =
                    NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
                vui.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_BT709;
            }
            Codec::Av1 => {
                // SAFETY: as above.
                let av1 = unsafe { &mut encode.encodeCodecConfig.av1Config };
                av1.level = level.nvenc;
                av1.tier = 0;
                av1.idrPeriod = NVENC_INFINITE_GOPLENGTH;
                // The sequence header with every keyframe, as for H.265.
                av1.set_repeatSeqHdr(1);
                av1.colorPrimaries = NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
                av1.transferCharacteristics =
                    NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
                av1.matrixCoefficients = NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_BT709;
                av1.colorRange = 0;
            }
        }
        let mut init = EncoderInitParams::new(codec_guid, config.width as u32, config.height as u32);
        init.preset_guid(preset_guid)
            .tuning_info(tuning)
            .framerate(config.fps, 1)
            .enable_picture_type_decision();
        // SAFETY: a configuration from the driver's own preset, changed only
        // in fields that the chosen codec defines.
        unsafe { init.encode_config(preset.presetCfg) };
        let session = nvenc
            .start_session(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12, init)
            .context("starting the NVENC session")?;
        let bitstream = session
            .create_output_bitstream()
            .context("NVENC output buffer")?;
        tracing::info!(
            codec = config.codec.label(),
            level = level.name,
            preset = config.preset,
            width = config.width,
            height = config.height,
            bitrate_mbps = config.bitrate_mbps,
            "NVENC ready"
        );
        Ok(Self {
            registered: HashMap::new(),
            bitstream: Some(bitstream),
            session,
            codec: config.codec,
            width: config.width,
            height: config.height,
            generation,
            chunks,
            frames: 0,
        })
    }

    /// Encodes `frame` (which must stay alive and unchanged until this
    /// returns), captured at `captured_us` (Unix epoch), as a keyframe if
    /// `keyframe`, and publishes it.
    pub fn send(&mut self, frame: &GpuFrame, meta: FrameMeta, captured_us: u64, keyframe: bool) -> Result<()> {
        anyhow::ensure!(
            (frame.width(), frame.height()) == (self.width, self.height),
            "frame is {}x{}, the encoder {}x{}",
            frame.width(),
            frame.height(),
            self.width,
            self.height
        );
        let address = frame.address() as usize;
        let resource = match self.registered.remove(&address) {
            Some(resource) => resource,
            // SAFETY: the buffer is a live NV12 frame of the session's size
            // with rows `width` bytes apart; it is registered once and the
            // pipeline keeps its frame buffers for the life of the stream.
            None => unsafe {
                self.session.register_generic_resource(
                    (),
                    NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                    address as *mut std::ffi::c_void,
                    self.width as u32,
                )
            }
            .context("registering a frame buffer with NVENC")?,
        };
        let bitstream = self.bitstream.take().context("NVENC output buffer lost")?;
        let submission = self
            .session
            .encode_picture(
                resource,
                bitstream,
                EncodePictureParams {
                    input_timestamp: self.frames,
                    force_idr: keyframe,
                },
            )
            .context("NVENC encode")?;
        let (data, resource, bitstream) = submission.finish().context("NVENC output")?;
        self.registered.insert(address, resource);
        self.bitstream = Some(bitstream);
        self.frames += 1;
        // No receivers is fine: nobody is watching yet.
        let _ = self.chunks.send(Arc::new(Chunk {
            generation: self.generation,
            key: self.codec.is_keyframe(&data),
            timestamp_us: captured_us,
            meta,
            data,
        }));
        Ok(())
    }
}

fn preset_guid(preset: u8) -> Result<GUID> {
    Ok(match preset {
        1 => NV_ENC_PRESET_P1_GUID,
        2 => NV_ENC_PRESET_P2_GUID,
        3 => NV_ENC_PRESET_P3_GUID,
        4 => NV_ENC_PRESET_P4_GUID,
        5 => NV_ENC_PRESET_P5_GUID,
        6 => NV_ENC_PRESET_P6_GUID,
        7 => NV_ENC_PRESET_P7_GUID,
        other => anyhow::bail!("NVENC presets are 1..=7, got {other}"),
    })
}
