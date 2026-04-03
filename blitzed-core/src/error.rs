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

//! Error handling for Blitzed Core

use thiserror::Error;

/// Result type alias for Blitzed operations
pub type Result<T> = std::result::Result<T, BlitzedError>;

/// Comprehensive error types for Blitzed operations
#[derive(Error, Debug)]
pub enum BlitzedError {
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Model format not supported: {format}")]
    UnsupportedFormat { format: String },

    #[error("Optimization failed: {reason}")]
    OptimizationFailed { reason: String },

    #[error("Tensor operation error: {message}")]
    TensorError { message: String },

    #[error("Target hardware not supported: {target}")]
    UnsupportedTarget { target: String },

    #[error("Code generation failed: {0}")]
    CodeGeneration(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Hardware constraint violation: {constraint}")]
    HardwareConstraint { constraint: String },

    #[error("Accuracy threshold exceeded: expected {threshold}%, got {actual}%")]
    AccuracyThreshold { threshold: f32, actual: f32 },

    #[error("Memory limit exceeded: {used} bytes > {limit} bytes")]
    MemoryLimit { used: usize, limit: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(String),

    #[error("Internal error: {0}")]
    Internal(String),

    // Benchmarking-specific errors
    #[error("Model not found: {model}")]
    ModelNotFound { model: String },

    #[error("Framework not supported: {framework}")]
    UnsupportedFramework { framework: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Conversion failed from {from} to {to}: {error}")]
    ConversionFailed {
        from: String,
        to: String,
        error: String,
    },

    #[error("Validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("System error: {message}")]
    SystemError { message: String },

    #[error("Report generation error: {message}")]
    ReportGenerationError { message: String },

    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },
}

#[cfg(feature = "onnx")]
impl From<ort::Error> for BlitzedError {
    fn from(err: ort::Error) -> Self {
        BlitzedError::OnnxRuntime(err.to_string())
    }
}

impl From<candle_core::Error> for BlitzedError {
    fn from(err: candle_core::Error) -> Self {
        BlitzedError::Internal(format!("Candle error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_model_load() {
        let error = BlitzedError::ModelLoad("test".into());
        let display = error.to_string();
        assert!(display.contains("Model loading error: test"));
    }

    #[test]
    fn test_error_display_unsupported_format() {
        let error = BlitzedError::UnsupportedFormat { format: "X".into() };
        let display = error.to_string();
        assert!(display.contains("X"));
    }

    #[test]
    fn test_error_display_optimization_failed() {
        let error = BlitzedError::OptimizationFailed { reason: "R".into() };
        let display = error.to_string();
        assert!(display.contains("R"));
    }

    #[test]
    fn test_error_display_tensor_error() {
        let error = BlitzedError::TensorError {
            message: "M".into(),
        };
        let display = error.to_string();
        assert!(display.contains("M"));
    }

    #[test]
    fn test_error_display_unsupported_target() {
        let error = BlitzedError::UnsupportedTarget { target: "T".into() };
        let display = error.to_string();
        assert!(display.contains("T"));
    }

    #[test]
    fn test_error_display_hardware_constraint() {
        let error = BlitzedError::HardwareConstraint {
            constraint: "C".into(),
        };
        let display = error.to_string();
        assert!(display.contains("C"));
    }

    #[test]
    fn test_error_display_accuracy_threshold() {
        let error = BlitzedError::AccuracyThreshold {
            threshold: 5.0,
            actual: 8.0,
        };
        let display = error.to_string();
        assert!(display.contains("5"));
        assert!(display.contains("8"));
    }

    #[test]
    fn test_error_display_memory_limit() {
        let error = BlitzedError::MemoryLimit {
            used: 200,
            limit: 100,
        };
        let display = error.to_string();
        assert!(display.contains("200"));
        assert!(display.contains("100"));
    }

    #[test]
    fn test_error_from_io() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let blitzed_error = BlitzedError::from(io_error);
        let display = blitzed_error.to_string();
        assert!(display.contains("test"));
    }

    #[test]
    fn test_error_from_serde() {
        let result: std::result::Result<String, _> = serde_json::from_str("invalid");
        let serde_error = result.unwrap_err();
        let blitzed_error = BlitzedError::from(serde_error);
        let display = blitzed_error.to_string();
        assert!(display.contains("Serialization"));
    }
}
