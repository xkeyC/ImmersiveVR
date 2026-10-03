//! Real-model checks, skipped unless `IVR_DEPTH_MODEL_DIR` names a directory
//! with `dav2s_378x210_{fp16,fp32}.onnx`. ONNX Runtime comes from
//! `ORT_DYLIB_PATH` or `runtime/ort` under the crate (cargo runs tests there,
//! so pass an absolute ORT_DYLIB_PATH).
//!
//! IVR_DEPTH_MODEL_DIR=<repo>/models/depth ORT_DYLIB_PATH=<repo>/runtime/ort/onnxruntime.dll \
//!     cargo test --release -p depth-infer --test real_model -- --nocapture

use depth_infer::{init_runtime, DepthEngine, EngineConfig, ModelFiles, Provider, RuntimeOptions};
use std::path::{Path, PathBuf};

const WIDTH: usize = 378;
const HEIGHT: usize = 210;

fn model_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("IVR_DEPTH_MODEL_DIR")?);
    init_runtime(&RuntimeOptions::default()).expect("ONNX Runtime");
    Some(dir)
}

/// Two different scenes as BGRA frames.
fn frames() -> [Vec<u8>; 2] {
    let frame = |f: fn(usize, usize) -> u8| -> Vec<u8> {
        (0..HEIGHT)
            .flat_map(|y| (0..WIDTH).flat_map(move |x| [f(x, y), f(y, x), f(x + y, y), 255]))
            .collect()
    };
    [
        frame(|x, y| ((x * 255 / WIDTH + y) % 256) as u8),
        frame(|x, y| if (x / 40 + y / 30) % 2 == 0 { 230 } else { 20 }),
    ]
}

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|v| *v as f64).sum::<f64>() / n,
        b.iter().map(|v| *v as f64).sum::<f64>() / n,
    );
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (*x as f64 - ma, *y as f64 - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va * vb).sqrt()
}

fn engine(dir: &Path, provider: Provider, cuda_graph: bool) -> DepthEngine {
    let mut config = EngineConfig::new(ModelFiles::in_dir(dir, WIDTH, HEIGHT).unwrap());
    config.providers = vec![provider];
    config.cuda_graph = cuda_graph;
    DepthEngine::new(&config).unwrap()
}

/// Frames alternate through a CUDA graph (captured on blank warmup frames):
/// each output must follow its own frame, matching the CPU reference.
#[test]
fn cuda_graph_tracks_new_frames() {
    let Some(dir) = model_dir() else {
        eprintln!("IVR_DEPTH_MODEL_DIR not set; skipped");
        return;
    };
    let [a, b] = frames();
    let mut cpu = engine(&dir, Provider::Cpu, false);
    let reference = [
        cpu.infer(&a).unwrap().to_vec(),
        cpu.infer(&b).unwrap().to_vec(),
    ];
    assert!(
        correlation(&reference[0], &reference[1]) < 0.9,
        "test frames too alike"
    );

    for cuda_graph in [true, false] {
        let mut gpu = engine(&dir, Provider::Cuda, cuda_graph);
        gpu.warmup(4).unwrap();
        for round in 0..3 {
            for (frame, expected) in [&a, &b].into_iter().zip(&reference) {
                let r = correlation(gpu.infer(frame).unwrap(), expected);
                assert!(
                    r > 0.999,
                    "cuda_graph={cuda_graph} round {round}: correlation {r}"
                );
            }
        }
    }
}

#[test]
fn wrong_frame_size_is_rejected() {
    let Some(dir) = model_dir() else {
        return;
    };
    let mut cpu = engine(&dir, Provider::Cpu, false);
    assert!(cpu.infer(&[0; 16]).is_err());
}
