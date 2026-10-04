//! Drives ivr_native.dll the way the Unity plugin does: a shadow copy loaded
//! with LoadLibrary, init (test pattern), attach to a D3D11 device, render
//! events and polls for a few seconds, shutdown, FreeLibrary; three times in
//! one process (the editor's reload cycle).
//!
//! Needs models/ and an ONNX Runtime with CUDA (runtime/runtime.txt, else
//! runtime/ort), a CUDA GPU; skipped when models/ is missing.

use std::{
    ffi::{c_char, c_void, CString},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use windows::{
    core::Interface,
    Win32::Graphics::{
        Direct3D::D3D_DRIVER_TYPE_HARDWARE,
        Direct3D11::*,
        Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_SAMPLE_DESC,
        },
    },
};

#[repr(C)]
#[derive(Default, Debug)]
struct IvrInfo {
    frame: u64,
    generation: u64,
    width: i32,
    height: i32,
    left: usize,
    right: usize,
    running: i32,
    divergence: f32,
    convergence: f32,
}

type Init = unsafe extern "C" fn(*const c_char) -> i32;
type Attach = unsafe extern "C" fn(*mut c_void) -> i32;
type Set = unsafe extern "C" fn(*const c_char) -> i32;
type Poll = unsafe extern "C" fn(*mut IvrInfo) -> i32;
type LastError = unsafe extern "C" fn(*mut c_char, i32) -> i32;
type RenderEventFunc = extern "C" fn() -> extern "system" fn(i32);
type Shutdown = extern "C" fn() -> i32;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// `ort` and `lib_dirs` from runtime/runtime.txt, as the server reads it.
fn runtime_paths(repo: &Path) -> (Option<String>, Vec<String>) {
    let Ok(text) = std::fs::read_to_string(repo.join("runtime/runtime.txt")) else {
        return (
            Some(
                repo.join("runtime/ort/onnxruntime.dll")
                    .display()
                    .to_string(),
            ),
            Vec::new(),
        );
    };
    let (mut ort, mut libs) = (None, Vec::new());
    for line in text.lines().map(str::trim).filter(|l| !l.starts_with('#')) {
        match line.split_once('=') {
            Some(("ort", v)) => ort = Some(v.trim().to_string()),
            Some(("lib", v)) => libs.push(v.trim().to_string()),
            _ => {}
        }
    }
    (ort, libs)
}

fn device() -> (ID3D11Device, ID3D11DeviceContext) {
    let (mut device, mut context) = (None, None);
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            Default::default(),
            D3D11_CREATE_DEVICE_FLAG(0),
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
        .unwrap();
    }
    (device.unwrap(), context.unwrap())
}

/// The RGBA bytes of a row through the middle of mip `level` of the texture
/// behind `view`.
fn middle_row(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    view: usize,
    level: u32,
) -> Vec<u8> {
    unsafe {
        let view_ptr = view as *mut c_void;
        let view = ID3D11ShaderResourceView::from_raw_borrowed(&view_ptr).unwrap();
        let texture: ID3D11Texture2D = view.GetResource().unwrap().cast().unwrap();
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        texture.GetDesc(&mut desc);
        assert_eq!(desc.Format, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
        assert!(desc.MipLevels > 1, "the eye textures have mipmaps");
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            ..desc
        };
        let mut staging = None;
        device
            .CreateTexture2D(&staging_desc, None, Some(&mut staging))
            .unwrap();
        let staging = staging.unwrap();
        context.CopyResource(&staging, &texture);
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        context
            .Map(&staging, level, D3D11_MAP_READ, 0, Some(&mut mapped))
            .unwrap();
        let (width, height) = ((desc.Width >> level).max(1), (desc.Height >> level).max(1));
        let row = (height / 2) as usize;
        let data = std::slice::from_raw_parts(
            (mapped.pData as *const u8).add(row * mapped.RowPitch as usize),
            width as usize * 4,
        )
        .to_vec();
        context.Unmap(&staging, level);
        data
    }
}

#[test]
fn unity_style_load_run_unload_three_times() {
    let repo = repo();
    if !repo.join("models/stereo").is_dir() {
        eprintln!("models/ missing: skipped");
        return;
    }
    // The DLL next to this test's deps directory (target/<profile>/).
    let dll = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("ivr_native.dll");
    assert!(
        dll.is_file(),
        "build it first: cargo build --release -p ivr-native ({})",
        dll.display()
    );
    let (ort, lib_dirs) = runtime_paths(&repo);
    let config = serde_json::json!({
        "ort": ort,
        "lib_dirs": lib_dirs,
        "models": repo.join("models/depth"),
        "stereo_models": repo.join("models/stereo"),
        "synthetic": true,
        "resolution": 1440,
        "log_file": repo.join("target/run/ivr_native_test.log"),
    })
    .to_string();
    // As the Unity editor does: from here on PATH is not searched for DLLs.
    #[link(name = "kernel32")]
    extern "system" {
        fn SetDefaultDllDirectories(flags: u32) -> i32;
    }
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x1000;
    assert_ne!(unsafe { SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }, 0);
    let (device, context) = device();
    let host_texture = unsafe {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: 4,
            Height: 4,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture = None;
        device
            .CreateTexture2D(&desc, None, Some(&mut texture))
            .unwrap();
        texture.unwrap()
    };

    for cycle in 0..3 {
        // A shadow copy, as the Unity loader does: the build output stays writable.
        let copy = std::env::temp_dir().join(format!(
            "ivr_native_test_{}_{cycle}.dll",
            std::process::id()
        ));
        std::fs::copy(&dll, &copy).unwrap();
        let library = unsafe { libloading::Library::new(&copy) }.unwrap();
        unsafe {
            let init: libloading::Symbol<Init> = library.get(b"ivr_init").unwrap();
            let attach: libloading::Symbol<Attach> = library.get(b"ivr_attach").unwrap();
            let set: libloading::Symbol<Set> = library.get(b"ivr_set").unwrap();
            let poll: libloading::Symbol<Poll> = library.get(b"ivr_poll").unwrap();
            let last_error: libloading::Symbol<LastError> = library.get(b"ivr_last_error").unwrap();
            let render_event_func: libloading::Symbol<RenderEventFunc> =
                library.get(b"ivr_render_event_func").unwrap();
            let shutdown: libloading::Symbol<Shutdown> = library.get(b"ivr_shutdown").unwrap();
            let error = || {
                let mut buffer = vec![0u8; 1024];
                last_error(buffer.as_mut_ptr().cast(), buffer.len() as i32);
                String::from_utf8_lossy(&buffer)
                    .trim_end_matches('\0')
                    .to_string()
            };

            let started = Instant::now();
            let config = CString::new(config.clone()).unwrap();
            assert_eq!(init(config.as_ptr()), 0, "init: {}", error());
            assert_eq!(attach(host_texture.as_raw()), 0, "attach: {}", error());
            eprintln!(
                "cycle {cycle}: init {:.1} s",
                started.elapsed().as_secs_f64()
            );
            let render_event = render_event_func();
            let mut info = IvrInfo::default();
            let mut first_frame = None;
            let started = Instant::now();
            while started.elapsed() < Duration::from_secs(3) {
                render_event(0);
                assert_eq!(poll(&mut info), 0, "poll: {}", error());
                assert_eq!(info.running, 1, "pipeline stopped: {}", error());
                if info.frame > 0 && first_frame.is_none() {
                    first_frame = Some((info.frame, Instant::now()));
                }
                if started.elapsed() > Duration::from_millis(1500) && info.divergence < 1.0 {
                    let settings = CString::new(r#"{"divergence": 2.0}"#).unwrap();
                    assert_eq!(set(settings.as_ptr()), 0, "set: {}", error());
                }
                std::thread::sleep(Duration::from_millis(16));
            }
            let (first, at) = first_frame.expect("no frame reached the textures");
            let rate = (info.frame - first) as f64 / at.elapsed().as_secs_f64();
            eprintln!("cycle {cycle}: {info:?}, {rate:.0} new frames/s");
            assert_eq!((info.width, info.height), (2560, 1440));
            assert_eq!(info.generation, 1);
            assert!(rate > 40.0, "frames came at {rate:.0}/s");
            assert!((info.divergence - 2.0).abs() < 1e-6);
            let left = middle_row(&device, &context, info.left, 0);
            let right = middle_row(&device, &context, info.right, 0);
            let mean = |row: &[u8]| row.iter().map(|&v| v as f64).sum::<f64>() / row.len() as f64;
            // The generated levels hold the same picture, smaller.
            let half = middle_row(&device, &context, info.left, 1);
            assert_eq!(half.len(), left.len() / 2);
            assert!(
                (mean(&half) - mean(&left)).abs() < 20.0,
                "mip 1 mean {:.1} vs level 0 {:.1}",
                mean(&half),
                mean(&left)
            );
            let differ = left
                .iter()
                .zip(&right)
                .filter(|(a, b)| a.abs_diff(**b) > 8)
                .count();
            eprintln!(
                "cycle {cycle}: left mean {:.1}, right mean {:.1}, {differ} bytes differ",
                mean(&left),
                mean(&right)
            );
            assert!(mean(&left) > 10.0, "left eye is empty");
            assert!(differ > 100, "the eyes are identical");

            let started = Instant::now();
            assert_eq!(shutdown(), 0, "shutdown: {}", error());
            eprintln!(
                "cycle {cycle}: shutdown {:.2} s",
                started.elapsed().as_secs_f64()
            );
        }
        let image = loaded_image(&copy);
        library.close().unwrap();
        let _ = std::fs::remove_file(&copy);
        // Keeps the unloaded copy's addresses unusable: the next copy loads
        // elsewhere, and a callback something kept from this one (ONNX
        // Runtime's logger, say) faults instead of landing in the new copy.
        reserve(image);
    }
}

#[link(name = "kernel32")]
extern "system" {
    fn GetModuleHandleW(name: *const u16) -> *mut u8;
    fn VirtualAlloc(address: *mut u8, size: usize, kind: u32, protect: u32) -> *mut u8;
}

/// Base and size of the loaded library at `path`.
fn loaded_image(path: &Path) -> (usize, usize) {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let base = unsafe { GetModuleHandleW(wide.as_ptr()) };
    assert!(!base.is_null(), "{} is not loaded", path.display());
    // SAFETY: a mapped PE image: e_lfanew at 0x3c, SizeOfImage 0x50 past it.
    let size = unsafe {
        let pe = base.add(std::ptr::read_unaligned(base.add(0x3c) as *const u32) as usize);
        std::ptr::read_unaligned(pe.add(0x50) as *const u32) as usize
    };
    (base as usize, size)
}

fn reserve((base, size): (usize, usize)) {
    const MEM_RESERVE: u32 = 0x2000;
    const PAGE_NOACCESS: u32 = 0x01;
    let reserved = unsafe { VirtualAlloc(base as *mut u8, size, MEM_RESERVE, PAGE_NOACCESS) };
    assert!(!reserved.is_null(), "the unloaded library's range is still in use");
}
