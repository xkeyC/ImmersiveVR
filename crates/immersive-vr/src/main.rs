//! ImmersiveVR: streams the Windows desktop as stereo 3D to a WebXR headset.
//!
//! capture (WGC) -> depth (Depth-Anything-V2-Small) -> iw3 mlbw_l2 fields ->
//! both eyes rendered on the PC (left on top, right below) -> H.265 / AV1
//! (NVENC, in process, straight from GPU memory) -> WebSocket -> browser,
//! where WebCodecs decodes and WebXR shows each eye in a stereo layer.

use anyhow::Result;
use clap::Parser;
use depth_infer::{init_runtime, Provider, RuntimeOptions};
use immersive_vr::{audio, capture, codec, engines, pipeline, server, tls};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};

#[derive(Parser)]
#[command(about = "Stream the desktop as stereo 3D to a WebXR headset")]
struct Args {
    /// Monitor to capture: 0 = primary, otherwise its 1-based Windows index.
    #[arg(long, default_value_t = 0)]
    monitor: usize,
    /// Stream a moving test pattern of the monitor's size instead of the
    /// screen (every frame new, like a fast game): measures throughput.
    #[arg(long)]
    synthetic: bool,
    /// The test pattern's own frame rate (default: --fps); higher shows how
    /// a fast monitor is sampled down to the stream's rate.
    #[arg(long)]
    synthetic_fps: Option<u32>,
    /// Output frames per second.
    #[arg(long, default_value_t = 60)]
    fps: u32,
    /// How often depth and stereo fields are recomputed at most (0 = for
    /// every captured frame). Every frame is warped with the newest fields;
    /// a lower rate saves GPU time but moving objects' depth lags behind.
    #[arg(long, default_value_t = 0.0)]
    depth_fps: f64,
    /// Initial streamed picture height (1080, 1440 or 2160; never above the
    /// capture's). Clients can change it.
    #[arg(long, default_value_t = 1440)]
    resolution: usize,
    /// Initial codec; clients can change it.
    #[arg(long, value_enum, default_value_t = CodecArg::Hevc)]
    codec: CodecArg,
    /// Fixed video bitrate in Mbit/s; by default it follows resolution and codec.
    #[arg(long)]
    bitrate: Option<u32>,
    /// NVENC preset: 1 (fastest encode, least latency; Sunshine's default)
    /// ..= 7 (best quality per bit). At 4K, presets up to 4 also split each
    /// frame across the GPU's encoders where it has more than one.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=7))]
    preset: u8,
    /// Address to serve on; 0.0.0.0 opens it to the LAN.
    #[arg(long, default_value = "0.0.0.0:13256")]
    listen: SocketAddr,
    /// Serve plain HTTP instead of HTTPS; WebXR then works only through
    /// http://localhost (e.g. `adb reverse`).
    #[arg(long)]
    http: bool,
    /// Where the self-signed certificate is kept between runs.
    #[arg(long, default_value = "runtime/tls")]
    tls_dir: PathBuf,
    /// Directory with dav2s_{W}x{H}_{fp16,fp32}.onnx.
    #[arg(long, default_value = "models/depth")]
    models: PathBuf,
    /// Depth model input size, WxH. Larger keeps object outlines tighter
    /// (less stretching beside them) at more GPU time; 518x294 is the light option.
    #[arg(long, default_value = "770x434")]
    depth_size: String,
    /// Directory with the iw3 mlbw_l2 graphs (scripts/export_iw3_stereo.py):
    /// iw3_mlbw_l2_d{1,2,3}_{depth size}_fields.onnx.
    #[arg(long, default_value = "models/stereo")]
    stereo_models: PathBuf,
    /// Initial divergence: total left-right shift in % of the width.
    #[arg(long, default_value_t = 0.5)]
    divergence: f32,
    /// Mute the PC's speakers while a headset is connected (its sound goes
    /// there); clients can change it.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    mute_pc: bool,
    /// Never touch the PC's speakers, whatever clients ask (tests on a PC in use).
    #[arg(long)]
    never_mute_pc: bool,
    /// Initial convergence: 0 = everything in front of the screen .. 1 = behind it.
    #[arg(long, default_value_t = 0.5)]
    convergence: f32,
    /// Provider order for the depth and stereo models (trt, cuda, dml, cpu).
    /// TensorRT is used when the loaded ONNX Runtime has it (the first start
    /// builds its engines, minutes); the stereo fields must end up on the GPU.
    #[arg(long, default_value = "trt,cuda,dml,cpu")]
    providers: String,
    /// onnxruntime library (default: `ort=` in runtime/runtime.txt,
    /// ORT_DYLIB_PATH, beside the exe, runtime/ort).
    #[arg(long)]
    ort: Option<PathBuf>,
    /// Extra directory for CUDA/cuDNN/TensorRT libraries (repeatable), after
    /// the `lib=` lines of runtime/runtime.txt.
    #[arg(long = "lib-dir")]
    lib_dirs: Vec<PathBuf>,
}

/// This machine's runtime paths, kept out of git in runtime/runtime.txt:
/// `ort=<onnxruntime library>` and `lib=<library directory>` lines (`#`
/// starts a comment). For example, ONNX Runtime's CUDA 12 build with
/// TensorRT 10 and the CUDA 12 / cuDNN 9 libraries it needs.
const RUNTIME_FILE: &str = "runtime/runtime.txt";

fn runtime_options(args: &Args) -> Result<RuntimeOptions> {
    let mut options = RuntimeOptions::default();
    if let Ok(text) = std::fs::read_to_string(RUNTIME_FILE) {
        for line in text.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            match line.split_once('=').map(|(key, value)| (key.trim(), value.trim())) {
                Some(("ort", path)) => options.dylib = Some(path.into()),
                Some(("lib", dir)) => options.library_dirs.push(dir.into()),
                _ => anyhow::bail!("{RUNTIME_FILE}: expected ort=<file> or lib=<dir>, found {line:?}"),
            }
        }
        tracing::info!(file = RUNTIME_FILE, "runtime paths");
    }
    if let Some(ort) = &args.ort {
        options.dylib = Some(ort.clone());
    }
    options.library_dirs.extend(args.lib_dirs.iter().cloned());
    Ok(options)
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum CodecArg {
    Hevc,
    Av1,
}

impl From<CodecArg> for codec::Codec {
    fn from(codec: CodecArg) -> Self {
        match codec {
            CodecArg::Hevc => codec::Codec::Hevc,
            CodecArg::Av1 => codec::Codec::Av1,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn".into()),
        )
        .init();
    let args = Args::parse();

    init_runtime(&runtime_options(&args)?)?;
    let engines = engines::load(&engines::EngineOptions {
        models: args.models.clone(),
        stereo_models: args.stereo_models.clone(),
        depth_size: engines::parse_size(&args.depth_size)?,
        providers: Provider::parse_list(&args.providers)?,
        divergence: args.divergence,
        convergence: args.convergence,
    })?;
    let gpu = engines.gpu.clone();

    let controls = Arc::new(pipeline::Controls::new(pipeline::Settings {
        divergence: args.divergence,
        convergence: args.convergence,
        resolution: args.resolution,
        codec: args.codec.into(),
        mute_pc: args.mute_pc,
    }));
    let latest = Arc::new(capture::LatestFrame::default());
    let capture = if args.synthetic {
        capture::start_synthetic((3840, 2160), args.synthetic_fps.unwrap_or(args.fps), latest.clone(), gpu.clone())?
    } else {
        capture::start(args.monitor, args.fps, latest.clone(), gpu.clone())?
    };
    // A client more than a few frames behind drops to a keyframe (server.rs);
    // the buffer only has to cover that.
    let (chunks, _) = tokio::sync::broadcast::channel(16);
    let (stream, stream_rx) = tokio::sync::watch::channel(None);
    // A quarter second of sound; a client that far behind skips ahead.
    let (sound, _) = tokio::sync::broadcast::channel(25);
    audio::start(sound.clone())?;
    let clients = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    // Restores the PC's speakers when dropped (at the end of main).
    let _pc_mute = (!args.never_mute_pc)
        .then(|| audio::PcMute::start(controls.clone(), clients.clone()))
        .transpose()?;
    let pipeline = pipeline::Pipeline {
        capture_size: capture.size,
        output: pipeline::Output::Stream {
            encoder: pipeline::EncoderDefaults {
                fps: args.fps,
                bitrate_mbps: args.bitrate,
                preset: args.preset,
            },
            chunks: chunks.clone(),
            stream,
        },
        fps: args.fps,
        stop: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        engine: engines.depth,
        stereo: engines.stereo,
        gpu: engines.gpu,
        delta_scale: engines.delta_scale,
        controls: controls.clone(),
        latest,
        depth_fps: args.depth_fps,
    };
    let worker = std::thread::Builder::new()
        .name("pipeline".into())
        .spawn(move || pipeline.run())?;

    let transport = if args.http {
        server::Transport::Http
    } else {
        rustls::crypto::ring::default_provider()
            .install_default()
            .map_err(|_| anyhow::anyhow!("a rustls crypto provider was already installed"))?;
        server::Transport::Https(tls::load_or_create(&args.tls_dir, &tls::lan_addresses())?)
    };
    let runtime = tokio::runtime::Runtime::new()?;
    let result = runtime.block_on(async move {
        tokio::select! {
            served = server::serve(args.listen, transport, stream_rx, chunks, sound, controls, clients) => served,
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("stopping");
                Ok(())
            }
            finished = tokio::task::spawn_blocking(move || worker.join()) => {
                match finished {
                    Ok(Ok(Err(error))) => Err(error.context("pipeline stopped")),
                    _ => Err(anyhow::anyhow!("pipeline stopped")),
                }
            }
        }
    });
    // The pipeline threads run until the process ends, and the blocking task
    // above waits on them: shut down without waiting for it (dropping the
    // runtime would, forever).
    runtime.shutdown_background();
    drop(capture);
    result
}
