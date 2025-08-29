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

//! PyTorch model converter

use super::ModelConverter;
use crate::{BlitzedError, Model, Result};

#[cfg(feature = "pytorch")]
use crate::model::{ModelData, ModelFormat, ModelInfo};
use std::path::Path;

#[cfg(feature = "pytorch")]
use tch::{CModule, Device, Tensor};

/// PyTorch model converter
pub struct PyTorchConverter;

impl PyTorchConverter {
    pub fn new() -> Self {
        Self
    }

    /// Load PyTorch TorchScript model (.pt/.pth file)
    #[cfg(feature = "pytorch")]
    fn load_pytorch_model<P: AsRef<Path>>(&self, path: P) -> Result<Model> {
        let path = path.as_ref();

        log::info!("Loading PyTorch model from: {}", path.display());

        // Load the TorchScript model
        let module = CModule::load(path)
            .map_err(|e| BlitzedError::Internal(format!("Failed to load PyTorch model: {}", e)))?;

        // Extract model information by analyzing the module
        let model_info = self.analyze_pytorch_model(&module, path)?;

        Ok(Model {
            info: model_info,
            data: ModelData::PyTorch(module),
        })
    }

    /// Fallback for when PyTorch feature is not enabled
    #[cfg(not(feature = "pytorch"))]
    fn load_pytorch_model<P: AsRef<Path>>(&self, _path: P) -> Result<Model> {
        Err(BlitzedError::UnsupportedFormat {
            format: "PyTorch (feature not enabled)".to_string(),
        })
    }

    /// Analyze PyTorch model to extract metadata
    #[cfg(feature = "pytorch")]
    fn analyze_pytorch_model<P: AsRef<Path>>(
        &self,
        module: &CModule,
        path: P,
    ) -> Result<ModelInfo> {
        let path = path.as_ref();

        // Get file size for model size estimation
        let model_size_bytes = std::fs::metadata(path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);

        // Try to run a forward pass with dummy data to get input/output shapes
        let (input_shapes, output_shapes) = self.infer_shapes(module)?;

        // Rough parameter count estimation (actual count would require deeper analysis)
        let parameter_count = self.estimate_parameters(&input_shapes, &output_shapes);

        // Rough operation count estimation
        let operations_count = self.estimate_operations(&input_shapes, &output_shapes);

        Ok(ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes,
            output_shapes,
            parameter_count,
            model_size_bytes,
            operations_count,
            layers: vec![], // INCOMPLETE: Layer extraction from PyTorch module not implemented
        })
    }

    /// Attempt to infer input/output shapes by running dummy inference
    #[cfg(feature = "pytorch")]
    fn infer_shapes(&self, module: &CModule) -> Result<(Vec<Vec<i64>>, Vec<Vec<i64>>)> {
        // Common input shapes to try for shape inference
        let test_shapes = vec![
            vec![1, 3, 224, 224], // ImageNet standard
            vec![1, 3, 32, 32],   // CIFAR-10
            vec![1, 1, 28, 28],   // MNIST
            vec![1, 100],         // Simple dense input
            vec![1, 10],          // Small dense input
        ];

        for shape in test_shapes {
            if let Ok((input_shapes, output_shapes)) = self.try_shape(&module, &shape) {
                return Ok((vec![input_shapes], vec![output_shapes]));
            }
        }

        // If all attempts fail, return default shapes
        log::warn!("Could not infer model shapes, using defaults");
        Ok((vec![vec![1, 3, 224, 224]], vec![vec![1, 1000]]))
    }

    /// Try a specific input shape to infer model I/O
    #[cfg(feature = "pytorch")]
    fn try_shape(&self, module: &CModule, shape: &[i64]) -> Result<(Vec<i64>, Vec<i64>)> {
        // Create dummy input tensor
        let dummy_input = Tensor::randn(shape, tch::kind::FLOAT_CPU);

        // Try to run forward pass
        let output = module
            .forward_ts(&[dummy_input])
            .map_err(|e| BlitzedError::Internal(format!("Forward pass failed: {}", e)))?;

        // Get output shape
        let output_shape: Vec<i64> = output.size();

        Ok((shape.to_vec(), output_shape))
    }

    /// Estimate parameter count based on input/output shapes
    #[cfg(feature = "pytorch")]
    fn estimate_parameters(&self, input_shapes: &[Vec<i64>], output_shapes: &[Vec<i64>]) -> usize {
        // Very rough estimation - in practice, we'd need to inspect the actual model
        let input_size: i64 = input_shapes
            .first()
            .map(|s| s.iter().product())
            .unwrap_or(1);
        let output_size: i64 = output_shapes
            .first()
            .map(|s| s.iter().product())
            .unwrap_or(1);

        // Assume a simple dense connection as baseline
        (input_size * output_size) as usize
    }

    /// Estimate operation count based on model complexity
    #[cfg(feature = "pytorch")]
    fn estimate_operations(&self, input_shapes: &[Vec<i64>], output_shapes: &[Vec<i64>]) -> usize {
        // Very rough estimation - multiply-accumulate operations
        let input_size: i64 = input_shapes
            .first()
            .map(|s| s.iter().product())
            .unwrap_or(1);
        let output_size: i64 = output_shapes
            .first()
            .map(|s| s.iter().product())
            .unwrap_or(1);

        (input_size * output_size * 2) as usize // 2 ops per MAC
    }
}

impl ModelConverter for PyTorchConverter {
    fn load_model<P: AsRef<Path>>(&self, path: P) -> Result<Model> {
        self.load_pytorch_model(path)
    }

    fn save_model<P: AsRef<Path>>(&self, _model: &Model, _path: P) -> Result<()> {
        // TODO: Implement PyTorch model saving
        Err(BlitzedError::Internal(
            "PyTorch saving not implemented".to_string(),
        ))
    }

    fn supported_extensions(&self) -> &'static [&'static str] {
        &["pt", "pth"]
    }
}

impl Default for PyTorchConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_converter_creation() {
        let _converter = PyTorchConverter::new();
        assert_eq!(_converter.supported_extensions(), &["pt", "pth"]);
    }

    #[test]
    fn test_pytorch_converter_default() {
        let converter = PyTorchConverter::new();
        assert_eq!(converter.supported_extensions(), &["pt", "pth"]);
    }

    #[test]
    #[allow(unused_variables)]
    fn test_unsupported_save_operation() {
        let converter = PyTorchConverter::new();

        // Create a dummy model for testing save failure
        #[cfg(feature = "pytorch")]
        {
            let model_info = ModelInfo {
                format: ModelFormat::PyTorch,
                input_shapes: vec![vec![1, 3, 224, 224]],
                output_shapes: vec![vec![1, 1000]],
                parameter_count: 1000,
                model_size_bytes: 4096,
                operations_count: 2000,
                layers: vec![],
            };
            let model = Model {
                info: model_info,
                data: ModelData::Raw(vec![1, 2, 3, 4]),
            };

            let result = converter.save_model(&model, "/tmp/test.pt");
            assert!(result.is_err());
            match result.unwrap_err() {
                BlitzedError::Internal(msg) => {
                    assert_eq!(msg, "PyTorch saving not implemented");
                }
                _ => panic!("Expected Internal error"),
            }
        }
    }

    #[test]
    #[cfg(not(feature = "pytorch"))]
    fn test_load_model_without_pytorch_feature() {
        let converter = PyTorchConverter::new();
        let result = converter.load_model("/tmp/nonexistent.pt");

        assert!(result.is_err());
        match result.unwrap_err() {
            BlitzedError::UnsupportedFormat { format } => {
                assert_eq!(format, "PyTorch (feature not enabled)");
            }
            _ => panic!("Expected UnsupportedFormat error"),
        }
    }

    #[test]
    #[cfg(feature = "pytorch")]
    fn test_load_nonexistent_model_with_pytorch_feature() {
        let converter = PyTorchConverter::new();
        let result = converter.load_model("/tmp/nonexistent_model_file_12345.pt");

        // Should fail due to file not existing, not feature disabled
        assert!(result.is_err());
        match result.unwrap_err() {
            BlitzedError::Internal(msg) => {
                assert!(msg.contains("Failed to load PyTorch model"));
            }
            _ => panic!("Expected Internal error for missing file"),
        }
    }
}
