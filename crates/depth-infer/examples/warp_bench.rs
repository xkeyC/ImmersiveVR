//! Times mlbw_l2 fields and the warp on their own, and checks the warp's
//! output is a plausible stereo frame.
//!
//! ```text
//! cargo run --release -p depth-infer --example warp_bench -- \
//!     [--models models/stereo] [--size 770x434] [--picture 2560x1440] [--runs 100]
//!     [--stereo-providers trt,cuda]
//! ```
//! With other stereo providers than CUDA, their fields are compared with CUDA's.

use depth_infer::{init_runtime, Provider, RuntimeOptions, StereoEngine, WarpEngine};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info,ort=warn"))
        .init();
    let mut models = PathBuf::from("models/stereo");
    let mut size = "770x434".to_string();
    let mut picture = "2560x1440".to_string();
    let mut runs = 100usize;
    let mut stereo_providers = vec![Provider::Cuda];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--models" => models = value()?.into(),
            "--size" => size = value()?,
            "--picture" => picture = value()?,
            "--runs" => runs = value()?.parse()?,
            "--stereo-providers" => stereo_providers = Provider::parse_list(&value()?)?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let parse = |s: &str| -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let (w, h) = s.split_once('x').ok_or("size is WxH")?;
        Ok((w.parse()?, h.parse()?))
    };
    let (width, height) = parse(&picture)?;
    init_runtime(&RuntimeOptions::default())?;
    let providers = [Provider::Cuda];
    let fields_model = models.join(format!("iw3_mlbw_l2_d1_{size}_fields.onnx"));
    let mut stereo = StereoEngine::new(&fields_model, &stereo_providers, 0)?;
    println!("stereo provider: {}", stereo.provider());
    let mut warp = WarpEngine::new(&models.join(format!("iw3_warp_{size}.onnx")), &providers, 0)?;

    // A ramp in depth (near at the bottom) and a striped picture.
    let (fw, fh) = (stereo.width(), stereo.height());
    let depth: Vec<f32> = (0..fw * fh).map(|i| (i / fw) as f32 / fh as f32).collect();
    let color: Vec<u8> = (0..width * height)
        .flat_map(|i| {
            let x = i % width;
            let v = if (x / 16) % 2 == 0 { 230 } else { 20 };
            [v, v, v, 255]
        })
        .collect();

    if stereo.provider() != Provider::Cuda {
        let mut reference = StereoEngine::new(&fields_model, &providers, 0)?;
        let expected = reference.infer(&depth, 3.0, 0.5)?.to_vec();
        let got = stereo.infer(&depth, 3.0, 0.5)?;
        let max = expected.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let mean = expected.iter().zip(got).map(|(a, b)| (a - b).abs()).sum::<f32>() / got.len() as f32;
        println!("fields vs CUDA: max |diff| {max:.5}, mean {mean:.6}");
    }
    let fields = stereo.infer_fields(&depth, 3.0, 0.5)?;
    println!("fields ok");
    let nv12 = warp.render(&color, width, height, &fields)?;
    println!("warp ok: {} bytes", nv12.len());
    assert_eq!(nv12.len(), width * height * 3);
    // Bottom rows (near) of the two eyes differ; luma only.
    let row = height - 8;
    let left = &nv12[row * width..(row + 1) * width];
    let right = &nv12[(height + row) * width..(height + row + 1) * width];
    let diff: u64 = left.iter().zip(right).map(|(a, b)| a.abs_diff(*b) as u64).sum();
    println!("eyes differ near the bottom by {:.1} per pixel", diff as f64 / width as f64);

    let started = Instant::now();
    for _ in 0..runs {
        stereo.infer_fields(&depth, 3.0, 0.5)?;
    }
    println!("fields: {:.2} ms", started.elapsed().as_secs_f64() * 1e3 / runs as f64);
    let started = Instant::now();
    for _ in 0..runs {
        warp.render(&color, width, height, &fields)?;
    }
    println!("warp {width}x{height} to host: {:.2} ms", started.elapsed().as_secs_f64() * 1e3 / runs as f64);
    let mut frame = warp.new_frame(width, height)?;
    let started = Instant::now();
    for _ in 0..runs {
        warp.render_into(&color, &fields, &mut frame)?;
    }
    println!("warp {width}x{height} on device: {:.2} ms", started.elapsed().as_secs_f64() * 1e3 / runs as f64);
    Ok(())
}
