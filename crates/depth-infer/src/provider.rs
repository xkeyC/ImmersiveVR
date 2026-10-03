use crate::{Error, Result};
use std::{fmt, str::FromStr};

/// An ONNX Runtime execution provider a [`crate::DepthEngine`] can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    /// TensorRT, with CUDA for any nodes TensorRT rejects. Builds an engine on
    /// first use (minutes) and caches it.
    TensorRt,
    Cuda,
    /// DirectML (any DX12 GPU on Windows); needs a DirectML build of ORT.
    DirectMl,
    Cpu,
}

impl Provider {
    /// The default order: fastest first, CPU last.
    pub const DEFAULT_ORDER: [Provider; 4] = [
        Provider::TensorRt,
        Provider::Cuda,
        Provider::DirectMl,
        Provider::Cpu,
    ];

    /// Whether frames run against fixed CUDA buffers (and may replay as CUDA
    /// graphs) rather than host buffers.
    pub fn is_cuda(self) -> bool {
        matches!(self, Provider::TensorRt | Provider::Cuda)
    }

    /// Parses a comma-separated order such as `trt,cuda,cpu`.
    pub fn parse_list(list: &str) -> Result<Vec<Provider>> {
        let providers = list
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::parse)
            .collect::<Result<Vec<_>>>()?;
        if providers.is_empty() {
            return Err(Error::Invalid("empty provider list".into()));
        }
        Ok(providers)
    }
}

impl FromStr for Provider {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "trt" | "tensorrt" => Ok(Provider::TensorRt),
            "cuda" => Ok(Provider::Cuda),
            "dml" | "directml" => Ok(Provider::DirectMl),
            "cpu" => Ok(Provider::Cpu),
            other => Err(Error::Invalid(format!(
                "unknown provider `{other}` (expected trt, cuda, dml or cpu)"
            ))),
        }
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Provider::TensorRt => "tensorrt",
            Provider::Cuda => "cuda",
            Provider::DirectMl => "directml",
            Provider::Cpu => "cpu",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lists_and_aliases() {
        assert_eq!(
            Provider::parse_list("trt, CUDA,dml,cpu").unwrap(),
            Provider::DEFAULT_ORDER
        );
        assert_eq!(
            Provider::parse_list("tensorrt,directml").unwrap(),
            [Provider::TensorRt, Provider::DirectMl]
        );
        assert!(Provider::parse_list("cuda,rocm").is_err());
        assert!(Provider::parse_list(" , ").is_err());
    }
}
