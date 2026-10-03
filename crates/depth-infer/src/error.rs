use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ONNX Runtime: {0}")]
    Ort(String),
    #[error("runtime setup: {0}")]
    Runtime(String),
    #[error("model {path:?}: {reason}")]
    Model { path: PathBuf, reason: String },
    #[error("no provider could run the model: {0}")]
    NoProvider(String),
    #[error("frame has {actual} bytes, the model takes {expected} (BGRA {width}x{height})")]
    FrameSize {
        actual: usize,
        expected: usize,
        width: usize,
        height: usize,
    },
    #[error("{0}")]
    Invalid(String),
    #[error("I/O error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl<R> From<ort::Error<R>> for Error {
    fn from(error: ort::Error<R>) -> Self {
        Self::Ort(error.to_string())
    }
}
