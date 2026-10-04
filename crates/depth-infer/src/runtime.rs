//! Loading ONNX Runtime before the first session.
//!
//! ort is built with `load-dynamic`: nothing links against ONNX Runtime and the
//! library is opened on first use. Left to itself, Windows resolves a bare
//! `onnxruntime.dll` through the system search path and may pick the old copy
//! in System32, so [`init_runtime`] always loads an explicit path.

use crate::{Error, Result};
use std::path::{Path, PathBuf};

#[cfg(windows)]
const DYLIB_NAME: &str = "onnxruntime.dll";
#[cfg(target_os = "linux")]
const DYLIB_NAME: &str = "libonnxruntime.so";
#[cfg(target_os = "macos")]
const DYLIB_NAME: &str = "libonnxruntime.dylib";

/// Where ONNX Runtime and the GPU libraries it loads come from.
#[derive(Debug, Clone, Default)]
pub struct RuntimeOptions {
    /// The ONNX Runtime library. When `None`: `ORT_DYLIB_PATH`, else the
    /// library beside the executable, else `runtime/ort/` under the working
    /// directory (where `scripts/fetch_onnxruntime.py` puts it).
    pub dylib: Option<PathBuf>,
    /// Directories holding CUDA, cuDNN or TensorRT libraries, searched before
    /// `PATH`. Windows only: on Linux set `LD_LIBRARY_PATH` before starting.
    pub library_dirs: Vec<PathBuf>,
}

/// Loads ONNX Runtime and creates its environment. Call once, before any
/// [`crate::DepthEngine`]; returns the library path that was loaded.
pub fn init_runtime(options: &RuntimeOptions) -> Result<PathBuf> {
    let dylib = resolve_dylib(options.dylib.as_deref())?;
    let mut dirs = options.library_dirs.clone();
    if let Some(parent) = dylib.parent() {
        // The provider libraries sit beside onnxruntime itself.
        dirs.insert(0, parent.to_path_buf());
    }
    prepend_library_dirs(&dirs)?;
    tracing::info!(path = %dylib.display(), "loading ONNX Runtime");
    let committed = ort::init_from(&dylib)
        .map_err(|error| Error::Runtime(format!("cannot load {}: {error}", dylib.display())))?
        .with_name("immersive-vr")
        .commit();
    if !committed {
        tracing::warn!("ONNX Runtime environment already existed; its settings are kept");
    }
    Ok(dylib)
}

fn resolve_dylib(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return existing(path.to_path_buf());
    }
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH") {
        return existing(PathBuf::from(path));
    }
    let candidates = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(DYLIB_NAME)))
        .into_iter()
        .chain(std::iter::once(
            PathBuf::from("runtime").join("ort").join(DYLIB_NAME),
        ));
    for candidate in candidates {
        if candidate.is_file() {
            return absolute(candidate);
        }
    }
    Err(Error::Runtime(format!(
        "{DYLIB_NAME} not found beside the executable or in runtime/ort; run \
         `python scripts/fetch_onnxruntime.py`, or name it with ORT_DYLIB_PATH"
    )))
}

fn existing(path: PathBuf) -> Result<PathBuf> {
    if path.is_file() {
        absolute(path)
    } else {
        Err(Error::Runtime(format!(
            "ONNX Runtime library {} does not exist",
            path.display()
        )))
    }
}

fn absolute(path: PathBuf) -> Result<PathBuf> {
    std::path::absolute(&path).map_err(|source| Error::Io { path, source })
}

#[cfg(windows)]
fn prepend_library_dirs(dirs: &[PathBuf]) -> Result<()> {
    if dirs.is_empty() {
        return Ok(());
    }
    // LoadLibrary searches the process PATH as it is at call time, which is
    // how the CUDA/TensorRT providers find cudart, cuDNN and nvinfer.
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = Vec::with_capacity(dirs.len());
    for dir in dirs {
        if !dir.is_dir() {
            return Err(Error::Runtime(format!(
                "library directory {} does not exist",
                dir.display()
            )));
        }
        paths.push(absolute(dir.clone())?);
    }
    // A host that called SetDefaultDllDirectories (the Unity editor does)
    // no longer searches PATH: register the directories there as well.
    for dir in &paths {
        add_dll_directory(dir);
    }
    preload_first_copies(&paths);
    let joined = std::env::join_paths(paths.into_iter().chain(std::env::split_paths(&current)))
        .map_err(|error| Error::Runtime(format!("cannot extend PATH: {error}")))?;
    std::env::set_var("PATH", joined);
    Ok(())
}

#[cfg(windows)]
fn add_dll_directory(dir: &Path) {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn AddDllDirectory(new_directory: *const u16) -> *mut std::ffi::c_void;
    }
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    // SAFETY: a NUL-terminated wide path; the cookie is never removed (the
    // directories stay for the process, as the PATH entries do).
    if unsafe { AddDllDirectory(wide.as_ptr()) }.is_null() {
        tracing::warn!(dir = %dir.display(), "AddDllDirectory failed");
    }
}

/// The libraries the CUDA and TensorRT providers link against, dependencies
/// first. cuDNN's own parts are loaded by cuDNN, by name.
#[cfg(windows)]
const PROVIDER_DEPENDENCIES: &[&str] = &[
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cufft64_11.dll",
    "cudnn64_9.dll",
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll",
];

/// Loads each provider dependency from the first directory that has it.
/// PATH gives the directories an order, AddDllDirectory does not: a host
/// that only searches the latter could pick an older cuBLAS from a later
/// directory (error 127, a missing entry point). Once loaded, a library is
/// what every later load of that name gets.
#[cfg(windows)]
fn preload_first_copies(dirs: &[PathBuf]) {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut std::ffi::c_void, flags: u32) -> *mut std::ffi::c_void;
    }
    // The library's own directory first, then the default search.
    const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x100;
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x1000;
    for name in PROVIDER_DEPENDENCIES {
        let Some(path) = dirs.iter().map(|dir| dir.join(name)).find(|path| path.is_file()) else {
            continue;
        };
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        // SAFETY: a NUL-terminated wide path; the library stays loaded for
        // the process, as the providers that use it do.
        let module = unsafe {
            LoadLibraryExW(
                wide.as_ptr(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        };
        if module.is_null() {
            tracing::warn!(path = %path.display(), error = %std::io::Error::last_os_error(), "cannot preload");
        } else {
            tracing::info!(path = %path.display(), "preloaded");
        }
    }
}

#[cfg(not(windows))]
fn prepend_library_dirs(dirs: &[PathBuf]) -> Result<()> {
    if dirs.len() > 1 {
        tracing::warn!("library directories are ignored on this platform; set LD_LIBRARY_PATH");
    }
    Ok(())
}
