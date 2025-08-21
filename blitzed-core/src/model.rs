// Copyright 2025 Gibran Rodriguez <brangi000@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Model representation and loading functionality

use crate::{BlitzedError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "onnx")]
use ort::session::Session;

/// Supported model formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    Onnx,
    PyTorch,
    TensorFlow,
    TfLite,
    CoreML,
}

impl ModelFormat {
    /// Detect format from file extension
    pub fn from_path<P: AsRef<Path>>(path: P) -> Option<Self> {
        let path = path.as_ref();
        match path.extension()?.to_str()? {
            "onnx" => Some(Self::Onnx),
            "pt" | "pth" => Some(Self::PyTorch),
            "pb" => Some(Self::TensorFlow),
            "tflite" => Some(Self::TfLite),
            "mlmodel" => Some(Self::CoreML),
            _ => None,
        }
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::PyTorch => "pt",
            Self::TensorFlow => "pb",
            Self::TfLite => "tflite",
            Self::CoreML => "mlmodel",
        }
    }
}

/// Model metadata and information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub format: ModelFormat,
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
    pub parameter_count: usize,
    pub model_size_bytes: usize,
    pub operations_count: usize,
}

/// High-level model representation
#[derive(Debug)]
pub struct Model {
    pub info: ModelInfo,
    pub data: ModelData,
}

/// Internal model data representation
#[derive(Debug)]
pub enum ModelData {
    #[cfg(feature = "onnx")]
    Onnx(Session),
    Raw(Vec<u8>),
}

impl Model {
    /// Load a model from file path
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let format = ModelFormat::from_path(path)
            .ok_or_else(|| BlitzedError::UnsupportedFormat {
                format: path.extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
            })?;

        match format {
            ModelFormat::Onnx => Self::load_onnx(path),
            _ => Err(BlitzedError::UnsupportedFormat {
                format: format!("{:?}", format),
            }),
        }
    }

    /// Load ONNX model
    #[cfg(feature = "onnx")]
    fn load_onnx<P: AsRef<Path>>(path: P) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path.as_ref())?;

        // Extract model metadata - simplified for now
        let inputs = session.inputs.iter()
            .map(|_input| vec![1, 3, 224, 224]) // Placeholder dimensions
            .collect();

        let outputs = session.outputs.iter()
            .map(|_output| vec![1, 1000]) // Placeholder dimensions
            .collect();

        let model_size = std::fs::metadata(path.as_ref())?.len() as usize;

        let info = ModelInfo {
            format: ModelFormat::Onnx,
            input_shapes: inputs,
            output_shapes: outputs,
            parameter_count: 0, // TODO: Calculate from ONNX graph
            model_size_bytes: model_size,
            operations_count: 0, // TODO: Calculate from ONNX graph
        };

        Ok(Model {
            info,
            data: ModelData::Onnx(session),
        })
    }

    /// Load ONNX model fallback when feature is disabled
    #[cfg(not(feature = "onnx"))]
    fn load_onnx<P: AsRef<Path>>(_path: P) -> Result<Self> {
        Err(BlitzedError::UnsupportedFormat {
            format: "ONNX support not compiled in".to_string(),
        })
    }

    /// Get model information
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Estimate memory usage for inference
    pub fn estimate_memory_usage(&self) -> usize {
        let mut total = self.info.model_size_bytes;
        
        // Add input tensor memory
        for shape in &self.info.input_shapes {
            total += shape.iter().product::<i64>() as usize * 4; // Assume f32
        }
        
        // Add output tensor memory
        for shape in &self.info.output_shapes {
            total += shape.iter().product::<i64>() as usize * 4; // Assume f32
        }
        
        // Add overhead (rough estimate)
        total + (total / 4)
    }

    /// Check if model fits within memory constraints
    pub fn check_memory_constraints(&self, memory_limit: usize) -> Result<()> {
        let usage = self.estimate_memory_usage();
        if usage > memory_limit {
            return Err(BlitzedError::MemoryLimit {
                used: usage,
                limit: memory_limit,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(ModelFormat::from_path("model.onnx"), Some(ModelFormat::Onnx));
        assert_eq!(ModelFormat::from_path("model.pt"), Some(ModelFormat::PyTorch));
        assert_eq!(ModelFormat::from_path("model.tflite"), Some(ModelFormat::TfLite));
        assert_eq!(ModelFormat::from_path("model.unknown"), None);
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(ModelFormat::Onnx.extension(), "onnx");
        assert_eq!(ModelFormat::PyTorch.extension(), "pt");
        assert_eq!(ModelFormat::TfLite.extension(), "tflite");
    }
}