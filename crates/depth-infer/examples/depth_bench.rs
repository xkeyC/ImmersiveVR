//! Loads a depth graph on the first usable provider, times repeated frames and
//! writes the normalized depth of a photo.
//!
//! cargo run --release -p depth-infer --example depth_bench -- \
//!     --size 518x294 --providers trt,cuda,cpu --image photo.jpg --out depth.png

use clap::Parser;
use depth_infer::{
    init_runtime, DepthEngine, DepthNormalizer, EngineConfig, ModelFiles, NormalizerConfig,
    Provider, RuntimeOptions,
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Args {
    /// Directory with dav2s_{W}x{H}_{fp16,fp32}.onnx.
    #[arg(long, default_value = "models/depth")]
    models: PathBuf,
    /// Model input size, WxH.
    #[arg(long, default_value = "518x294")]
    size: String,
    /// One graph for every provider instead of --models/--size.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Provider order, comma-separated (trt, cuda, dml, cpu).
    #[arg(long, default_value = "trt,cuda,dml,cpu")]
    providers: String,
    /// Photo to run (resized to the model size); a synthetic frame otherwise.
    #[arg(long)]
    image: Option<PathBuf>,
    /// Where to write the normalized depth of the frame as PNG.
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = 20)]
    warmup: usize,
    #[arg(long, default_value_t = 200)]
    iters: usize,
    #[arg(long)]
    no_cuda_graph: bool,
    #[arg(long, default_value_t = 0)]
    device: u32,
    /// onnxruntime library (default: ORT_DYLIB_PATH, beside the exe, runtime/ort).
    #[arg(long)]
    ort: Option<PathBuf>,
    /// Extra directory for CUDA/cuDNN/TensorRT libraries (repeatable).
    #[arg(long = "lib-dir")]
    lib_dirs: Vec<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn".into()),
        )
        .init();
    let args = Args::parse();
    let (width, height) = parse_size(&args.size)?;

    init_runtime(&RuntimeOptions {
        dylib: args.ort.clone(),
        library_dirs: args.lib_dirs.clone(),
    })?;
    let models = match &args.model {
        Some(path) => ModelFiles::single(path),
        None => ModelFiles::in_dir(&args.models, width, height)?,
    };
    let mut config = EngineConfig::new(models);
    config.providers = Provider::parse_list(&args.providers)?;
    config.cuda_graph = !args.no_cuda_graph;
    config.device_id = args.device;

    let started = Instant::now();
    let mut engine = DepthEngine::new(&config)?;
    let load = started.elapsed();
    for failure in engine.provider_failures() {
        println!("skipped {}: {}", failure.provider, failure.reason);
    }
    println!(
        "provider {} | model {} | {}x{} | load {:.2}s",
        engine.provider(),
        engine.model_path().display(),
        engine.width(),
        engine.height(),
        load.as_secs_f64()
    );

    let frame = match &args.image {
        Some(path) => photo_frame(path, engine.width(), engine.height())?,
        None => synthetic_frame(engine.width(), engine.height()),
    };

    let started = Instant::now();
    engine.warmup(args.warmup)?;
    println!(
        "warmup {} runs {:.2}s",
        args.warmup,
        started.elapsed().as_secs_f64()
    );

    let mut times = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let started = Instant::now();
        engine.infer(&frame)?;
        times.push(started.elapsed());
    }
    report(&mut times);

    let depth = engine.infer(&frame)?.to_vec();
    let mut normalizer = DepthNormalizer::new(NormalizerConfig::default())?;
    let mut map = vec![0u8; depth.len()];
    let started = Instant::now();
    normalizer.normalize(&depth, &mut map)?;
    println!(
        "normalize {:.3} ms, range {:?}",
        started.elapsed().as_secs_f64() * 1e3,
        normalizer.range()
    );
    if let Some(out) = &args.out {
        image::GrayImage::from_raw(engine.width() as u32, engine.height() as u32, map)
            .ok_or("depth map size")?
            .save(out)?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

fn parse_size(size: &str) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let (w, h) = size.split_once(['x', 'X']).ok_or("size must be WxH")?;
    Ok((w.trim().parse()?, h.trim().parse()?))
}

fn photo_frame(
    path: &PathBuf,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let rgba = image::open(path)?.to_rgba8();
    let resized = image::imageops::resize(
        &rgba,
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    // RGBA -> BGRA, as screen capture delivers it.
    Ok(resized
        .pixels()
        .flat_map(|p| [p[2], p[1], p[0], p[3]])
        .collect())
}

fn synthetic_frame(width: usize, height: usize) -> Vec<u8> {
    (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                [
                    ((x + y) % 256) as u8,
                    (y * 255 / height) as u8,
                    (x * 255 / width) as u8,
                    255,
                ]
            })
        })
        .collect()
}

fn report(times: &mut [Duration]) {
    if times.is_empty() {
        return;
    }
    times.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let at = |q: f64| ms(times[((times.len() - 1) as f64 * q).round() as usize]);
    let mean = times.iter().map(|d| ms(*d)).sum::<f64>() / times.len() as f64;
    println!(
        "infer x{}: mean {:.2} ms | min {:.2} | p50 {:.2} | p90 {:.2} | p99 {:.2} | max {:.2} | {:.0} fps",
        times.len(),
        mean,
        ms(times[0]),
        at(0.5),
        at(0.9),
        at(0.99),
        ms(times[times.len() - 1]),
        1e3 / mean
    );
}
