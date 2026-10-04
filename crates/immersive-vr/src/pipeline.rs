//! Captured frame -> depth -> iw3 mlbw_l2 fields -> both eyes rendered on
//! the PC -> one video frame, left eye on top and right eye below:
//!
//! ```text
//! +-----------------------------+  y = 0
//! | left eye  (the picture, warped) |
//! +-----------------------------+  eye height
//! | right eye                   |
//! +-----------------------------+  2 x eye height
//! ```
//!
//! Five threads, the pictures on the GPU throughout: scale takes each new
//! capture to the streamed size (and, at most `depth_fps` times a second, to
//! the depth model's, the one image read back to host memory); depth and fields
//! turn those into depth and mlbw fields; render warps every new picture with
//! the newest fields into a GPU frame buffer (a CUDA kernel, [`GpuWarp`]);
//! the main thread hands each rendered frame to NVENC as soon as it exists. Depth and fields run less often than
//! pictures arrive: a picture is warped with fields of an earlier one (at most
//! a frame or two older), which keeps every streamed frame new while the GPU
//! time per frame drops to little more than the warp.
//!
//! The client picks the streamed resolution and codec at run time
//! ([`Controls`]); a change restarts the encoder with a new layout and
//! publishes a new [`StreamInfo`].

use crate::{
    capture::{CapturedFrame, LatestFrame},
    codec::Codec,
    encoder::{Chunk, Encoder, EncoderConfig, FrameMeta},
    gpu::{Gpu, GpuFrame, GpuImage},
};
use anyhow::{Context as _, Result};
use depth_infer::{DepthEngine, DepthNormalizer, NormalizerConfig, StereoEngine, StereoFields};
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};
use tokio::sync::{broadcast, watch};

/// Streamed picture heights a client can choose (capped at the capture's).
pub const RESOLUTIONS: [usize; 3] = [1080, 1440, 2160];

/// Largest divergence offered: mlbw_l2's strongest level is trained to ~10.
pub const MAX_DIVERGENCE: f32 = 10.0;

/// Where the two eyes sit in the video frame, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Layout {
    pub width: usize,
    pub height: usize,
    pub left: Rect,
    pub right: Rect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Layout {
    /// Two eyes of a `capture` scaled to `height` rows (never up), stacked.
    pub fn new(capture: (usize, usize), height: usize) -> Self {
        let scale = (height as f64 / capture.1 as f64).min(1.0);
        let even = |v: f64| ((v / 2.0).round() as usize * 2).max(2);
        let (width, eye_height) = (
            even(capture.0 as f64 * scale),
            even(capture.1 as f64 * scale),
        );
        let eye = |y| Rect {
            x: 0,
            y,
            width,
            height: eye_height,
        };
        Self {
            width,
            height: eye_height * 2,
            left: eye(0),
            right: eye(eye_height),
        }
    }
}

/// What a client can change while streaming.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Settings {
    /// Total left-right shift in % of the width (iw3 `--divergence`).
    pub divergence: f32,
    /// 0 = everything behind the screen .. 1 = everything in front.
    pub convergence: f32,
    /// Requested picture height (a [`RESOLUTIONS`] entry).
    pub resolution: usize,
    pub codec: Codec,
    /// Mute the PC's speakers while a client listens (the sound still goes to it).
    pub mute_pc: bool,
}

/// A client's change request; absent fields stay as they are.
#[derive(Debug, Default, Deserialize)]
pub struct SettingsUpdate {
    pub divergence: Option<f32>,
    pub convergence: Option<f32>,
    pub resolution: Option<usize>,
    pub codec: Option<Codec>,
    pub mute_pc: Option<bool>,
}

/// The current [`Settings`], shared by the server (writes) and the pipeline
/// (reads every frame).
pub struct Controls {
    settings: Mutex<Settings>,
    version: AtomicU64,
    /// A client needs a keyframe now (it joined, fell behind, or lost one).
    keyframe: std::sync::atomic::AtomicBool,
}

impl Controls {
    pub fn new(settings: Settings) -> Self {
        let controls = Self {
            settings: Mutex::new(settings),
            version: AtomicU64::new(0),
            keyframe: std::sync::atomic::AtomicBool::new(false),
        };
        controls.apply(SettingsUpdate {
            resolution: Some(settings.resolution),
            ..Default::default()
        });
        controls
    }

    /// Asks the encoder for a keyframe with its next frame.
    pub fn request_keyframe(&self) {
        self.keyframe.store(true, Ordering::Release);
    }

    /// Whether a keyframe was asked for since the last call.
    pub fn take_keyframe_request(&self) -> bool {
        self.keyframe.swap(false, Ordering::AcqRel)
    }

    /// The settings and a version that changes on every update.
    pub fn get(&self) -> (Settings, u64) {
        let settings = self.settings.lock().unwrap_or_else(|e| e.into_inner());
        (*settings, self.version.load(Ordering::Acquire))
    }

    pub fn apply(&self, update: SettingsUpdate) {
        let mut settings = self.settings.lock().unwrap_or_else(|e| e.into_inner());
        let clean = |v: f32, max: f32| {
            if v.is_finite() {
                v.clamp(0.0, max)
            } else {
                0.0
            }
        };
        if let Some(divergence) = update.divergence {
            settings.divergence = divergence;
        }
        if let Some(convergence) = update.convergence {
            settings.convergence = convergence;
        }
        settings.divergence = clean(settings.divergence, MAX_DIVERGENCE);
        settings.convergence = clean(settings.convergence, 1.0);
        if let Some(resolution) = update.resolution {
            // The nearest offered height.
            settings.resolution = *RESOLUTIONS
                .iter()
                .min_by_key(|height| height.abs_diff(resolution))
                .expect("RESOLUTIONS is not empty");
        }
        if let Some(codec) = update.codec {
            settings.codec = codec;
        }
        if let Some(mute_pc) = update.mute_pc {
            settings.mute_pc = mute_pc;
        }
        self.version.fetch_add(1, Ordering::Release);
    }
}

/// What a client needs to decode and show the current stream.
#[derive(Debug, Clone, Serialize)]
pub struct StreamInfo {
    /// Chunks carry it; it changes whenever the encoder is replaced.
    pub generation: u64,
    pub codec: Codec,
    /// For `VideoDecoder.configure`.
    pub codec_string: String,
    pub fps: u32,
    pub bitrate_mbps: u32,
    pub layout: Layout,
    /// Picture heights the capture allows.
    pub resolutions: Vec<usize>,
    pub settings: Settings,
}

/// Encoder settings that do not change at run time.
pub struct EncoderDefaults {
    pub fps: u32,
    /// Fixed bitrate; otherwise chosen from resolution and codec.
    pub bitrate_mbps: Option<u32>,
    /// NVENC preset, 1 (fastest) ..= 7.
    pub preset: u8,
}

/// The default bitrate for an eye height (the frame carries two eyes), in Mbit/s.
pub fn default_bitrate(eye_height: usize, _codec: Codec) -> u32 {
    match eye_height {
        0..=1080 => 40,
        1081..=1440 => 60,
        _ => 100,
    }
}

/// Which mlbw_l2 level iw3 uses for a divergence (index into the loaded levels).
pub fn stereo_level(divergence: f32, levels: usize) -> usize {
    let level = if divergence <= 4.0 {
        0
    } else if divergence <= 7.0 {
        1
    } else {
        2
    };
    level.min(levels.saturating_sub(1))
}

/// Where rendered frames go.
pub enum Output {
    /// NVENC and the WebSocket clients (the browser): both eyes stacked as NV12.
    Stream {
        encoder: EncoderDefaults,
        chunks: broadcast::Sender<Arc<Chunk>>,
        stream: watch::Sender<Option<Arc<StreamInfo>>>,
    },
    /// Both eyes as RGBA images for an in-process consumer (the Unity plugin).
    Eyes(Arc<EyeOutput>),
}

/// One rendered pair of eye images (RGBA, gamma-encoded) on the GPU.
pub struct Eyes {
    pub left: GpuImage,
    pub right: GpuImage,
    pub meta: FrameMeta,
    /// When the picture's capture arrived.
    pub captured: Instant,
}

/// The newest eye pair, for a consumer that takes frames at its own pace.
/// A pair the consumer holds (an `Arc`) is not rendered into again until
/// it lets go.
#[derive(Default)]
pub struct EyeOutput {
    latest: Mutex<Option<Arc<Eyes>>>,
    frames: AtomicU64,
}

impl EyeOutput {
    /// The newest pair and its frame number (counts up from 1), if any.
    pub fn latest(&self) -> Option<(u64, Arc<Eyes>)> {
        let latest = self
            .latest
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()?;
        Some((self.frames.load(Ordering::Acquire), latest))
    }

    fn publish(&self, eyes: Arc<Eyes>) {
        *self.latest.lock().unwrap_or_else(|e| e.into_inner()) = Some(eyes);
        self.frames.fetch_add(1, Ordering::AcqRel);
    }
}

pub struct Pipeline {
    pub capture_size: (usize, usize),
    pub output: Output,
    /// Output frames per second (the capture sampling grid's rate).
    pub fps: u32,
    /// Set to stop every thread; `run` returns once they have.
    pub stop: Arc<std::sync::atomic::AtomicBool>,
    pub engine: DepthEngine,
    /// mlbw_l2 fields models by level (iw3 levels 1, 2, 3).
    pub stereo: Vec<StereoEngine>,
    /// The GPU kernels (in the CUDA primary context: ONNX Runtime's and NVENC's too).
    pub gpu: Arc<Gpu>,
    /// iw3's grid shift per field offset: 1 / (depth width / 2 - 1).
    pub delta_scale: f32,
    pub controls: Arc<Controls>,
    pub latest: Arc<LatestFrame>,
    /// How often depth and fields may run; 0 = for every captured frame.
    pub depth_fps: f64,
}

/// Whether the pipeline was asked to stop.
fn stopping(stop: &std::sync::atomic::AtomicBool) -> bool {
    stop.load(Ordering::Acquire)
}

/// A picture scaled to the eye size, on the GPU.
struct Picture {
    image: Arc<GpuImage>,
    /// When its capture arrived.
    captured: Instant,
}

/// mlbw fields for one depth map and what they were made with.
struct Fields {
    data: StereoFields,
    meta: FrameMeta,
    /// Made for a settings change (the picture did not change): render it now
    /// rather than waiting for the next picture.
    settings_changed: bool,
}

/// A rendered frame: both eyes, stacked, as NV12 on the GPU.
struct Rendered {
    width: usize,
    height: usize,
    frame: Arc<GpuFrame>,
    meta: FrameMeta,
    /// When the picture's capture arrived.
    captured: Instant,
}

/// Frame buffers the render thread keeps per picture size: enough for the
/// one being rendered, the newest one and the one being encoded.
const FRAME_BUFFERS: usize = 4;

/// Newest picture and fields for the render thread; `serial` changes with either.
#[derive(Default)]
struct RenderInput {
    picture: Option<Arc<Picture>>,
    fields: Option<Arc<Fields>>,
    serial: u64,
}

/// The current layout (pictures are scaled to its eye size, frames rendered
/// to it); `version` changes with it.
#[derive(Default)]
struct ScaleTarget {
    layout: Option<Layout>,
    version: u64,
}

/// State shared by the threads.
#[derive(Default)]
struct Shared {
    target: Mutex<ScaleTarget>,
    input: Mutex<RenderInput>,
    changed: Condvar,
    output: Mutex<Option<Arc<Rendered>>>,
    /// Signalled with each new `output`.
    rendered: Condvar,
}

impl Shared {
    fn update(&self, change: impl FnOnce(&mut RenderInput)) {
        let mut input = self.input.lock().unwrap_or_else(|e| e.into_inner());
        change(&mut input);
        input.serial += 1;
        self.changed.notify_one();
    }
}

/// Mean absolute difference of a BGRA frame's green channel (a cheap stand-in
/// for luminance) above which the next frame starts a new scene: depth
/// smoothing restarts instead of blending two unrelated pictures.
const SCENE_CUT: f32 = 40.0;

/// The newest normalized depth map (0..1) for the fields thread; `serial`
/// changes with each one.
#[derive(Default)]
struct DepthSlot {
    depth: Option<Arc<Vec<f32>>>,
    serial: u64,
}

/// Depth and normalization for each new picture, on its own thread so the
/// next picture's depth overlaps this one's mlbw fields.
struct DepthStage {
    stop: Arc<std::sync::atomic::AtomicBool>,
    engine: DepthEngine,
    /// Model-sized BGRA frames, newest only (the sender drops when full).
    inputs: mpsc::Receiver<Vec<u8>>,
    slot: Arc<(Mutex<DepthSlot>, Condvar)>,
}

impl DepthStage {
    fn run(mut self) -> Result<()> {
        let mut previous_input: Vec<u8> = Vec::new();
        let mut depth_map = vec![0u8; self.engine.width() * self.engine.height()];
        let mut normalizer = DepthNormalizer::new(NormalizerConfig::default())?;
        let (mut runs, mut cuts, mut time) = (0u64, 0u64, Duration::ZERO);
        let mut report = Instant::now();
        loop {
            if stopping(&self.stop) {
                return Ok(());
            }
            match self.inputs.recv_timeout(Duration::from_millis(100)) {
                Ok(input) => {
                    let started = Instant::now();
                    if is_scene_cut(&previous_input, &input) {
                        normalizer.reset();
                        cuts += 1;
                    }
                    let depth = self.engine.infer(&input)?;
                    normalizer.normalize(depth, &mut depth_map)?;
                    let depth01: Vec<f32> =
                        normalizer.smoothed().iter().map(|v| v / 255.0).collect();
                    previous_input = input;
                    let (slot, changed) = &*self.slot;
                    let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
                    slot.depth = Some(Arc::new(depth01));
                    slot.serial += 1;
                    changed.notify_one();
                    drop(slot);
                    time += started.elapsed();
                    runs += 1;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
            }
            if report.elapsed() >= Duration::from_secs(5) {
                tracing::info!(
                    depth_runs = runs,
                    depth_ms = format!("{:.2}", per(time, runs)),
                    scene_cuts = cuts,
                    "depth (last 5 s)"
                );
                (runs, cuts, time) = (0, 0, Duration::ZERO);
                report = Instant::now();
            }
        }
    }
}

/// mlbw fields for each new depth map, and again with the last one when the
/// stereo settings change.
struct FieldsStage {
    stop: Arc<std::sync::atomic::AtomicBool>,
    stereo: Vec<StereoEngine>,
    controls: Arc<Controls>,
    slot: Arc<(Mutex<DepthSlot>, Condvar)>,
    shared: Arc<Shared>,
}

impl FieldsStage {
    fn run(mut self) -> Result<()> {
        let (mut seen, mut fields_version) = (0, None);
        let mut depth: Option<Arc<Vec<f32>>> = None;
        let (mut runs, mut time) = (0u64, Duration::ZERO);
        let mut report = Instant::now();
        loop {
            if stopping(&self.stop) {
                return Ok(());
            }
            let new_depth = {
                let (slot, changed) = &*self.slot;
                let slot = slot.lock().unwrap_or_else(|e| e.into_inner());
                // Wake for a new depth map, or now and then to see settings changes.
                let (slot, _) = changed
                    .wait_timeout_while(slot, Duration::from_millis(20), |slot| slot.serial == seen)
                    .unwrap_or_else(|e| e.into_inner());
                let new = slot.serial != seen;
                seen = slot.serial;
                if new {
                    depth = slot.depth.clone();
                }
                new
            };
            let (settings, version) = self.controls.get();
            if let Some(depth) = depth
                .as_ref()
                .filter(|_| new_depth || fields_version != Some(version))
            {
                let started = Instant::now();
                let level = stereo_level(settings.divergence, self.stereo.len());
                let data = self.stereo[level].infer_fields(
                    depth,
                    settings.divergence,
                    settings.convergence,
                )?;
                let fields = Arc::new(Fields {
                    data,
                    meta: FrameMeta {
                        divergence: settings.divergence,
                        convergence: settings.convergence,
                    },
                    settings_changed: !new_depth,
                });
                self.shared.update(|input| input.fields = Some(fields));
                fields_version = Some(version);
                time += started.elapsed();
                runs += 1;
            }
            if report.elapsed() >= Duration::from_secs(5) {
                tracing::info!(
                    field_runs = runs,
                    fields_ms = format!("{:.2}", per(time, runs)),
                    "fields (last 5 s)"
                );
                (runs, time) = (0, Duration::ZERO);
                report = Instant::now();
            }
        }
    }
}

/// Renders both eyes once per new picture (with the newest fields), and
/// again when fields arrive for a settings change.
struct Render {
    stop: Arc<std::sync::atomic::AtomicBool>,
    gpu: Arc<Gpu>,
    delta_scale: f32,
    shared: Arc<Shared>,
    /// Eye images for an in-process consumer instead of NV12 frames.
    eyes: Option<Arc<EyeOutput>>,
}

/// Eye pairs the render thread keeps per picture size (see [`FRAME_BUFFERS`]).
const EYE_BUFFERS: usize = 4;

impl Render {
    fn run(self) -> Result<()> {
        let stream = self.gpu.stream()?;
        let mut seen = 0;
        // What the last render used.
        let mut rendered_picture: Option<Arc<Picture>> = None;
        let mut rendered_fields: Option<Arc<Fields>> = None;
        // Buffers for the current picture size; one is reused once nothing
        // else holds it.
        let mut buffers: Vec<Arc<GpuFrame>> = Vec::new();
        let mut eye_buffers: Vec<Arc<Eyes>> = Vec::new();
        let (mut renders, mut render_time) = (0u64, Duration::ZERO);
        let mut report = Instant::now();
        loop {
            if stopping(&self.stop) {
                return Ok(());
            }
            let (picture, fields) = {
                let input = self.shared.input.lock().unwrap_or_else(|e| e.into_inner());
                let (input, _) = self
                    .shared
                    .changed
                    .wait_timeout_while(input, Duration::from_millis(100), |input| {
                        input.serial == seen
                    })
                    .unwrap_or_else(|e| e.into_inner());
                seen = input.serial;
                (input.picture.clone(), input.fields.clone())
            };
            if let (Some(picture), Some(fields)) = (picture, fields) {
                fn same<T>(last: &Option<Arc<T>>, now: &Arc<T>) -> bool {
                    last.as_ref().is_some_and(|last| Arc::ptr_eq(last, now))
                }
                let new_picture = !same(&rendered_picture, &picture);
                let new_settings = fields.settings_changed && !same(&rendered_fields, &fields);
                if new_picture || new_settings || rendered_fields.is_none() {
                    let started = Instant::now();
                    let layout = self
                        .shared
                        .target
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .layout;
                    // A picture for an earlier layout waits for its own.
                    let Some(layout) = layout.filter(|l| {
                        (l.left.width, l.left.height)
                            == (picture.image.width(), picture.image.height())
                    }) else {
                        continue;
                    };
                    if let Some(output) = &self.eyes {
                        let (width, height) = (picture.image.width(), picture.image.height());
                        if eye_buffers.first().is_some_and(|eyes| {
                            (eyes.left.width(), eyes.left.height()) != (width, height)
                        }) {
                            eye_buffers.clear();
                        }
                        let free = match eye_buffers
                            .iter()
                            .position(|eyes| Arc::strong_count(eyes) == 1)
                        {
                            Some(free) => free,
                            None => {
                                anyhow::ensure!(
                                    eye_buffers.len() < EYE_BUFFERS,
                                    "all {EYE_BUFFERS} eye buffers are in use"
                                );
                                eye_buffers.push(Arc::new(Eyes {
                                    left: GpuImage::new(&stream, width, height)?,
                                    right: GpuImage::new(&stream, width, height)?,
                                    meta: fields.meta,
                                    captured: picture.captured,
                                }));
                                eye_buffers.len() - 1
                            }
                        };
                        let eyes = Arc::get_mut(&mut eye_buffers[free]).expect("held only here");
                        self.gpu.warp_eyes(
                            &stream,
                            &picture.image,
                            &fields.data,
                            self.delta_scale,
                            &mut eyes.left,
                            &mut eyes.right,
                        )?;
                        eyes.meta = fields.meta;
                        eyes.captured = picture.captured;
                        output.publish(eye_buffers[free].clone());
                        render_time += started.elapsed();
                        renders += 1;
                        rendered_picture = Some(picture);
                        rendered_fields = Some(fields);
                        continue;
                    }
                    let (width, height) = (layout.width, layout.height);
                    if buffers
                        .first()
                        .is_some_and(|frame| (frame.width(), frame.height()) != (width, height))
                    {
                        buffers.clear();
                    }
                    let free = match buffers
                        .iter()
                        .position(|frame| Arc::strong_count(frame) == 1)
                    {
                        Some(free) => free,
                        None => {
                            anyhow::ensure!(
                                buffers.len() < FRAME_BUFFERS,
                                "all {FRAME_BUFFERS} frame buffers are in use"
                            );
                            buffers.push(Arc::new(GpuFrame::new(&self.gpu, width, height)?));
                            buffers.len() - 1
                        }
                    };
                    let frame = Arc::get_mut(&mut buffers[free]).expect("held only here");
                    self.gpu.warp(
                        &stream,
                        &picture.image,
                        &fields.data,
                        self.delta_scale,
                        frame,
                    )?;
                    let rendered = Arc::new(Rendered {
                        width,
                        height,
                        frame: buffers[free].clone(),
                        meta: fields.meta,
                        captured: picture.captured,
                    });
                    *self.shared.output.lock().unwrap_or_else(|e| e.into_inner()) = Some(rendered);
                    self.shared.rendered.notify_all();
                    render_time += started.elapsed();
                    renders += 1;
                    rendered_picture = Some(picture);
                    rendered_fields = Some(fields);
                }
            }
            if report.elapsed() >= Duration::from_secs(5) {
                tracing::info!(
                    renders,
                    render_ms = format!("{:.2}", per(render_time, renders)),
                    "render (last 5 s)"
                );
                (renders, render_time) = (0, Duration::ZERO);
                report = Instant::now();
            }
        }
    }
}

/// How long after a capture's arrival the sampling grid aims its ticks.
const TICK_AFTER_CAPTURE: f64 = 0.001;

/// The scale stage's sampling grid: ticks every frame interval from a start,
/// shifted earlier by a phase that slowly follows the captures (ticks just
/// after they arrive: least waiting) when they come at about the stream's
/// rate. Faster sources leave it alone: a capture is always fresh then, and
/// chasing one would skip to the previous. The phase stays within an
/// interval, so the tick rate is exactly the stream's whatever it does.
struct Grid {
    start: Instant,
    ticks: u32,
    phase: f64,
    /// The last frame sampled (sequence, arrival): the source's own rate.
    last: Option<(u64, Instant)>,
}

impl Grid {
    fn new(start: Instant) -> Self {
        Self {
            start,
            ticks: 1,
            phase: 0.0,
            last: None,
        }
    }

    fn tick(&self, interval: Duration) -> Instant {
        let at = self.start + interval * self.ticks;
        at.checked_sub(Duration::from_secs_f64(self.phase))
            .unwrap_or(at)
    }

    /// Moves to the next tick past sampling `frame` at `tick`; with a source
    /// at about the stream's rate, the phase a tenth of the way to having had
    /// the capture wait `TICK_AFTER_CAPTURE`.
    fn advance(&mut self, frame: &CapturedFrame, tick: Instant, interval: Duration) {
        let source_interval = self.last.and_then(|(sequence, at)| {
            let frames = frame.sequence.checked_sub(sequence).filter(|&n| n > 0)?;
            Some(frame.at.saturating_duration_since(at).as_secs_f64() / frames as f64)
        });
        self.last = Some((frame.sequence, frame.at));
        let near_stream_rate = source_interval.is_some_and(|s| s > interval.as_secs_f64() * 0.75);
        if near_stream_rate {
            let waited = tick.saturating_duration_since(frame.at).as_secs_f64();
            self.phase = (self.phase + 0.1 * (waited - TICK_AFTER_CAPTURE))
                .clamp(0.0, interval.as_secs_f64() * 0.9);
        }
        self.ticks += 1;
    }
}

/// Captured frames whose depth input comes this close to `depth_fps`'s
/// interval count as due: captures jitter around their own rate.
const DEPTH_SLACK: Duration = Duration::from_millis(4);

/// Scales each new capture (on the GPU) to the streamed eye size for render
/// and, when depth is due, to the depth model's size, which it reads back.
struct ScaleStage {
    stop: Arc<std::sync::atomic::AtomicBool>,
    gpu: Arc<Gpu>,
    latest: Arc<LatestFrame>,
    shared: Arc<Shared>,
    inputs: mpsc::SyncSender<Vec<u8>>,
    model_size: (usize, usize),
    /// Least time between depth inputs.
    depth_interval: Duration,
    /// The stream's frame interval: captures are sampled on this grid.
    frame_interval: Duration,
}

#[derive(Default)]
struct ScaleStats {
    captured: u64,
    depth_inputs: u64,
    /// Depth was due but still busy with the previous input.
    depth_busy: u64,
    color: Duration,
    input: Duration,
}

/// Pictures for the current eye size: one that nothing else holds (the
/// render input, the last rendered picture) is reused.
const PICTURE_BUFFERS: usize = 4;

impl ScaleStage {
    fn run(self) -> Result<()> {
        let stream = self.gpu.stream()?;
        let mut pictures: Vec<Arc<GpuImage>> = Vec::new();
        let mut depth_image = GpuImage::new(&stream, self.model_size.0, self.model_size.1)?;
        let mut sequence = 0;
        let mut scaled_version = None;
        let mut last_capture: Option<Arc<CapturedFrame>> = None;
        let mut last_depth: Option<Instant> = None;
        let mut grid: Option<Grid> = None;
        let mut grid_start: Option<Instant> = None;
        let mut grid_reset = false;
        let mut stats = ScaleStats::default();
        let mut report = Instant::now();
        loop {
            if stopping(&self.stop) {
                return Ok(());
            }
            if report.elapsed() >= Duration::from_secs(5) {
                tracing::info!(
                    captured = stats.captured,
                    depth_inputs = stats.depth_inputs,
                    depth_busy = stats.depth_busy,
                    color_ms = format!("{:.2}", per(stats.color, stats.captured)),
                    input_ms = format!("{:.2}", per(stats.input, stats.depth_inputs)),
                    "scale (last 5 s)"
                );
                stats = ScaleStats::default();
                report = Instant::now();
            }
            // Frame pacing as Sunshine's: while the screen changes faster
            // than the stream, take the newest capture on a fixed grid of
            // frame intervals (a steady cadence however fast the monitor
            // refreshes); once there is nothing new at a tick, or a tick was
            // missed, wait for the next capture and start a new grid at it.
            let arrived = match grid.as_mut() {
                Some(grid) => {
                    let tick = grid.tick(self.frame_interval);
                    if let Some(wait) = tick.checked_duration_since(Instant::now()) {
                        std::thread::sleep(wait);
                    }
                    let frame = self.latest.wait_newer_than(sequence, Duration::ZERO);
                    if let Some(frame) = &frame {
                        grid.advance(frame, tick, self.frame_interval);
                    }
                    let missed = grid.tick(self.frame_interval) <= Instant::now();
                    if frame.is_none() || missed {
                        grid_reset = true;
                    }
                    frame
                }
                None => {
                    // Wake now and then to see a new target size.
                    let frame = self
                        .latest
                        .wait_newer_than(sequence, Duration::from_millis(20));
                    if let Some(frame) = &frame {
                        grid_start = Some(frame.at);
                    }
                    frame
                }
            };
            if grid_reset {
                grid = None;
                grid_reset = false;
            } else if let Some(start) = grid_start.take() {
                grid = Some(Grid::new(start));
            }
            let (eye, version) = {
                let target = self.shared.target.lock().unwrap_or_else(|e| e.into_inner());
                (target.layout.map(|layout| layout.left), target.version)
            };
            let (frame, is_new) = match arrived {
                Some(frame) => {
                    sequence = frame.sequence;
                    last_capture = Some(frame.clone());
                    (frame, true)
                }
                // A new size needs the last capture scaled again.
                None if scaled_version != Some(version) => match &last_capture {
                    Some(frame) => (frame.clone(), false),
                    None => continue,
                },
                None => continue,
            };
            let Some(eye) = eye else { continue };
            scaled_version = Some(version);
            let started = Instant::now();
            let capture = &frame.image;
            let image = if (capture.width(), capture.height()) == (eye.width, eye.height) {
                // Already the eye size: shared, not copied.
                capture.clone()
            } else {
                if pictures
                    .first()
                    .is_some_and(|p| (p.width(), p.height()) != (eye.width, eye.height))
                {
                    pictures.clear();
                }
                let free = match pictures.iter().position(|p| Arc::strong_count(p) == 1) {
                    Some(free) => free,
                    None => {
                        anyhow::ensure!(
                            pictures.len() < PICTURE_BUFFERS,
                            "all {PICTURE_BUFFERS} picture buffers are in use"
                        );
                        pictures.push(Arc::new(GpuImage::new(&stream, eye.width, eye.height)?));
                        pictures.len() - 1
                    }
                };
                let target = Arc::get_mut(&mut pictures[free]).expect("held only here");
                self.gpu.resize(&stream, capture, target)?;
                pictures[free].clone()
            };
            stats.color += started.elapsed();
            stats.captured += u64::from(is_new);
            let depth_due =
                last_depth.is_none_or(|last| last.elapsed() + DEPTH_SLACK >= self.depth_interval);
            if is_new && depth_due {
                let started = Instant::now();
                self.gpu.resize(&stream, capture, &mut depth_image)?;
                let mut input = vec![0u8; self.model_size.0 * self.model_size.1 * 4];
                depth_image.download(&stream, &mut input)?;
                stats.input += started.elapsed();
                match self.inputs.try_send(input) {
                    Ok(()) => {
                        last_depth = Some(started);
                        stats.depth_inputs += 1;
                    }
                    // Tried again with the next capture.
                    Err(mpsc::TrySendError::Full(_)) => stats.depth_busy += 1,
                    Err(mpsc::TrySendError::Disconnected(_)) => return Ok(()),
                }
            }
            let picture = Arc::new(Picture {
                image,
                captured: frame.at,
            });
            self.shared.update(|input| input.picture = Some(picture));
        }
    }
}

fn per(total: Duration, runs: u64) -> f64 {
    total.as_secs_f64() * 1e3 / runs.max(1) as f64
}

/// The encoder for one stream configuration.
struct Active {
    resolution: usize,
    codec: Codec,
    layout: Layout,
    encoder: Encoder,
}

#[derive(Default)]
struct Stats {
    /// Frames sent to the encoder, and how many of them were newly rendered
    /// (the rest repeat the last one to keep the stream going).
    frames: u64,
    fresh: u64,
    send: Duration,
    /// Capture to encoded, per fresh frame, in ms.
    latency: Vec<f64>,
}

/// What starting a stream needs: the capture size, encoder defaults and
/// where chunks and stream info go.
struct StreamContext {
    capture_size: (usize, usize),
    defaults: EncoderDefaults,
    cuda: Arc<cudarc::driver::CudaContext>,
    chunks: broadcast::Sender<Arc<Chunk>>,
    stream: watch::Sender<Option<Arc<StreamInfo>>>,
}

impl Pipeline {
    /// Runs until a thread fails or `stop` is set (then it waits for every
    /// thread). With [`Output::Stream`] it sends each newly rendered frame to
    /// the encoder as soon as it exists, and the last one again if nothing
    /// new came for 1.5 frame intervals (so the stream and its keyframes keep
    /// flowing; a render that is merely late is not preceded by a repeat).
    pub fn run(self) -> Result<()> {
        let Pipeline {
            capture_size,
            output,
            fps,
            stop,
            engine,
            stereo,
            gpu,
            delta_scale,
            controls,
            latest,
            depth_fps,
        } = self;
        let model_size = (engine.width(), engine.height());
        let shared = Arc::new(Shared::default());
        let (inputs, inputs_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let depth_slot = Arc::new((Mutex::new(DepthSlot::default()), Condvar::new()));
        let scale_stage = ScaleStage {
            stop: stop.clone(),
            gpu: gpu.clone(),
            latest,
            shared: shared.clone(),
            inputs,
            model_size,
            depth_interval: if depth_fps > 0.0 {
                Duration::from_secs_f64(1.0 / depth_fps)
            } else {
                Duration::ZERO
            },
            frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
        };
        let depth_stage = DepthStage {
            stop: stop.clone(),
            engine,
            inputs: inputs_rx,
            slot: depth_slot.clone(),
        };
        let fields_stage = FieldsStage {
            stop: stop.clone(),
            stereo,
            controls: controls.clone(),
            slot: depth_slot,
            shared: shared.clone(),
        };
        let render = Render {
            stop: stop.clone(),
            gpu: gpu.clone(),
            delta_scale,
            shared: shared.clone(),
            eyes: match &output {
                Output::Eyes(eyes) => Some(eyes.clone()),
                Output::Stream { .. } => None,
            },
        };
        let workers: Vec<JoinHandle<Result<()>>> = vec![
            std::thread::Builder::new()
                .name("scale".into())
                .spawn(move || scale_stage.run())?,
            std::thread::Builder::new()
                .name("depth".into())
                .spawn(move || depth_stage.run())?,
            std::thread::Builder::new()
                .name("fields".into())
                .spawn(move || fields_stage.run())?,
            std::thread::Builder::new()
                .name("render".into())
                .spawn(move || render.run())?,
        ];
        let result = match output {
            Output::Stream {
                encoder,
                chunks,
                stream,
            } => {
                let context = StreamContext {
                    capture_size,
                    defaults: encoder,
                    cuda: gpu.context().clone(),
                    chunks,
                    stream,
                };
                stream_frames(&context, &controls, &shared, &stop, &workers, fps)
            }
            Output::Eyes(_) => follow_layout(capture_size, &controls, &shared, &stop, &workers),
        };
        // Every thread sees the flag within a frame or two; a failed one has
        // stopped already.
        stop.store(true, Ordering::Release);
        let mut failure = result.err();
        for worker in workers {
            match worker.join() {
                Ok(Err(error)) if failure.is_none() => {
                    failure = Some(error.context("pipeline thread stopped"))
                }
                Err(_) if failure.is_none() => {
                    failure = Some(anyhow::anyhow!("pipeline thread panicked"))
                }
                _ => {}
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

/// [`Output::Eyes`]: no encoder; keeps the scale target at the settings'
/// resolution until stopped or a thread fails.
fn follow_layout(
    capture_size: (usize, usize),
    controls: &Controls,
    shared: &Shared,
    stop: &std::sync::atomic::AtomicBool,
    workers: &[JoinHandle<Result<()>>],
) -> Result<()> {
    let mut resolution = None;
    while !stopping(stop) {
        if workers.iter().any(JoinHandle::is_finished) {
            anyhow::bail!("a pipeline thread stopped");
        }
        let (settings, _) = controls.get();
        if resolution != Some(settings.resolution) {
            resolution = Some(settings.resolution);
            let mut target = shared.target.lock().unwrap_or_else(|e| e.into_inner());
            target.layout = Some(Layout::new(capture_size, settings.resolution));
            target.version += 1;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}

/// [`Output::Stream`]: encodes and sends rendered frames until stopped or a
/// thread fails.
fn stream_frames(
    context: &StreamContext,
    controls: &Controls,
    shared: &Shared,
    stop: &std::sync::atomic::AtomicBool,
    workers: &[JoinHandle<Result<()>>],
    fps: u32,
) -> Result<()> {
    let mut active: Option<Active> = None;
    let mut generation = 0;
    // The last rendered frame looked at, and the one being streamed.
    let mut seen: Option<Arc<Rendered>> = None;
    let mut current: Option<Arc<Rendered>> = None;
    let repeat_after = Duration::from_secs_f64(1.5 / fps as f64);
    let mut last_send = Instant::now();
    let mut stats = Stats::default();
    let mut report = Instant::now();
    loop {
        if stopping(stop) {
            return Ok(());
        }
        if workers.iter().any(JoinHandle::is_finished) {
            anyhow::bail!("a pipeline thread stopped");
        }
        let (settings, _) = controls.get();
        let stale = active
            .as_ref()
            .is_none_or(|a| (a.resolution, a.codec) != (settings.resolution, settings.codec));
        if stale {
            // The old NVENC session closes before the new one opens.
            drop(active.take());
            generation += 1;
            let started = context.start(generation, settings)?;
            let mut target = shared.target.lock().unwrap_or_else(|e| e.into_inner());
            target.layout = Some(started.layout);
            target.version += 1;
            drop(target);
            active = Some(started);
            current = None;
        }
        let active_stream = active.as_mut().expect("started above");
        let layout = active_stream.layout;

        // Wait for a new rendered frame, at most until a repeat is due.
        let deadline = last_send + repeat_after;
        let newest = {
            let output = shared.output.lock().unwrap_or_else(|e| e.into_inner());
            let timeout = deadline.saturating_duration_since(Instant::now());
            let (output, _) = shared
                .rendered
                .wait_timeout_while(output, timeout, |output| same_frame(output, &seen))
                .unwrap_or_else(|e| e.into_inner());
            output.clone()
        };
        let mut fresh = false;
        if !same_frame(&newest, &seen) {
            seen = newest.clone();
            // Frames rendered for an earlier layout are dropped.
            if let Some(rendered) =
                newest.filter(|r| (r.width, r.height) == (layout.width, layout.height))
            {
                current = Some(rendered);
                fresh = true;
            }
        }
        if fresh || Instant::now() >= deadline {
            last_send = Instant::now();
            if let Some(frame) = &current {
                // What the frame shows was captured then (repeats included).
                let captured_us =
                    unix_us().saturating_sub(frame.captured.elapsed().as_micros() as u64);
                active_stream
                    .encoder
                    .send(
                        &frame.frame,
                        frame.meta,
                        captured_us,
                        controls.take_keyframe_request(),
                    )
                    .context("encoding")?;
                stats.send += last_send.elapsed();
                stats.frames += 1;
                stats.fresh += u64::from(fresh);
                if fresh {
                    stats
                        .latency
                        .push(frame.captured.elapsed().as_secs_f64() * 1e3);
                }
            }
        }

        if report.elapsed() >= Duration::from_secs(5) {
            stats.latency.sort_by(f64::total_cmp);
            let quantile = |q: f64| {
                stats
                    .latency
                    .get((q * stats.latency.len() as f64) as usize)
                    .map_or("-".to_string(), |ms| format!("{ms:.1}"))
            };
            tracing::info!(
                sent = stats.frames,
                fresh = stats.fresh,
                send_ms = format!("{:.2}", per(stats.send, stats.frames)),
                capture_to_encoded_p50_ms = quantile(0.5),
                capture_to_encoded_p90_ms = quantile(0.9),
                "encoder input (last 5 s)"
            );
            stats = Stats::default();
            report = Instant::now();
        }
    }
}

fn unix_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or_default()
}

/// Whether two optional frames are the same one.
fn same_frame<T>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

impl StreamContext {
    /// An encoder for `settings`; publishes the new [`StreamInfo`].
    fn start(&self, generation: u64, settings: Settings) -> Result<Active> {
        let layout = Layout::new(self.capture_size, settings.resolution);
        let config = EncoderConfig {
            codec: settings.codec,
            width: layout.width,
            height: layout.height,
            fps: self.defaults.fps,
            bitrate_mbps: self
                .defaults
                .bitrate_mbps
                .unwrap_or_else(|| default_bitrate(layout.left.height, settings.codec)),
            preset: self.defaults.preset,
        };
        let encoder = Encoder::start(&config, generation, self.chunks.clone(), self.cuda.clone())?;
        let mut resolutions: Vec<usize> = RESOLUTIONS
            .iter()
            .copied()
            .filter(|&height| height <= self.capture_size.1)
            .collect();
        if resolutions.is_empty() {
            resolutions.push(self.capture_size.1);
        }
        let info = StreamInfo {
            generation,
            codec: settings.codec,
            codec_string: config.level().codec_string,
            fps: config.fps,
            bitrate_mbps: config.bitrate_mbps,
            layout,
            resolutions,
            settings,
        };
        tracing::info!(
            generation,
            codec = settings.codec.label(),
            codec_string = %info.codec_string,
            eye_width = layout.left.width,
            eye_height = layout.left.height,
            bitrate_mbps = config.bitrate_mbps,
            "stream configured (left eye top, right eye bottom)"
        );
        self.stream.send_replace(Some(Arc::new(info)));
        Ok(Active {
            resolution: settings.resolution,
            codec: settings.codec,
            layout,
            encoder,
        })
    }
}

/// Whether `current` (BGRA) differs from `previous` enough to be a new scene;
/// false when there is no previous frame.
fn is_scene_cut(previous: &[u8], current: &[u8]) -> bool {
    if previous.len() != current.len() || current.is_empty() {
        return false;
    }
    let total: u64 = previous
        .chunks_exact(4)
        .zip(current.chunks_exact(4))
        .map(|(a, b)| a[1].abs_diff(b[1]) as u64)
        .sum();
    total as f32 / (current.len() / 4) as f32 > SCENE_CUT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_stacks_two_eyes() {
        let layout = Layout::new((3840, 2160), 1440);
        assert_eq!((layout.width, layout.height), (2560, 2880));
        assert_eq!(
            layout.left,
            Rect {
                x: 0,
                y: 0,
                width: 2560,
                height: 1440
            }
        );
        assert_eq!(layout.right.y, 1440);
        // Never upscaled.
        let small = Layout::new((1280, 720), 1080);
        assert_eq!((small.width, small.height), (1280, 1440));
    }

    #[test]
    fn stereo_level_follows_iw3() {
        assert_eq!(stereo_level(3.0, 3), 0);
        assert_eq!(stereo_level(4.0, 3), 0);
        assert_eq!(stereo_level(5.0, 3), 1);
        assert_eq!(stereo_level(9.0, 3), 2);
        assert_eq!(stereo_level(9.0, 1), 0);
    }

    #[test]
    fn controls_clamp_snap_and_version() {
        let controls = Controls::new(Settings {
            divergence: 3.0,
            convergence: 0.5,
            resolution: 1440,
            codec: Codec::Hevc,
            mute_pc: true,
        });
        let (_, first) = controls.get();
        controls.apply(SettingsUpdate {
            divergence: Some(50.0),
            convergence: Some(f32::NAN),
            resolution: Some(2000),
            codec: Some(Codec::Av1),
            mute_pc: Some(false),
        });
        let (settings, second) = controls.get();
        assert_eq!(settings.divergence, MAX_DIVERGENCE);
        assert_eq!(settings.convergence, 0.0);
        assert_eq!(settings.resolution, 2160);
        assert_eq!(settings.codec, Codec::Av1);
        assert!(!settings.mute_pc);
        assert_ne!(first, second);
        let update: SettingsUpdate = serde_json::from_str(r#"{"codec":"hevc"}"#).unwrap();
        controls.apply(update);
        assert_eq!(controls.get().0.codec, Codec::Hevc);
        // H.264 is gone: an old client's request is rejected.
        assert!(serde_json::from_str::<SettingsUpdate>(r#"{"codec":"h264"}"#).is_err());
    }

    #[test]
    fn scene_cuts_need_a_large_picture_change() {
        let dark = [10u8, 10, 10, 255].repeat(100);
        let mut slightly = dark.clone();
        slightly[1] = 250;
        let bright = [200u8, 200, 200, 255].repeat(100);
        assert!(!is_scene_cut(&[], &dark));
        assert!(!is_scene_cut(&dark, &slightly));
        assert!(is_scene_cut(&dark, &bright));
    }
}
